"""
Explainable AI Engine for OceanEmbed
Grad-CAM spatial saliency + per-input-variable feature attribution.

Answers the operational question: "WHICH surface anomaly (eddy, upwelling
filament, wind curl) drove the predicted thermocline dip at depth X?"
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input channel order (config.yaml -> input_channels)
VARIABLE_NAMES = ["SST", "SSS", "SSH", "U_curr", "V_curr", "U_wind", "V_wind", "mask"]
# Dynamical variables relevant to eddy / upwelling drivers (mask excluded)
DYNAMIC_VARIABLES = ["SST", "SSS", "SSH", "U_curr", "V_curr", "U_wind", "V_wind"]


def _auto_target_layer(model: nn.Module) -> nn.Module:
    """Auto-select the last convolutional layer of the decoder for Grad-CAM."""
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "final_upsample"):
        convs = [m for m in decoder.final_upsample.modules() if isinstance(m, nn.Conv2d)]
        if convs:
            return convs[-1]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layer found for Grad-CAM target")
    return convs[-1]


class OceanGradCAM:
    """
    Grad-CAM explainer for OceanEmbed spatiotemporal models.

    Targets the last convolutional layer of the depth-reconstruction decoder
    (or a user-supplied layer, e.g. a ConvLSTM hidden conv) and produces:

      * Spatial saliency (Grad-CAM) maps for temperature at a given depth level
      * Channel attribution: relative contribution of each input variable
        (SST, SSS, SSH, currents, winds) to the prediction at that depth
      * Side-by-side attribution overlays highlighting mesoscale eddy /
        upwelling drivers
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()  # inference-mode explanations

        self.target_layer = target_layer if target_layer is not None else _auto_target_layer(model)

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------ #
    # Hook plumbing
    # ------------------------------------------------------------------ #
    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ------------------------------------------------------------------ #
    # Core forward/backward helper
    # ------------------------------------------------------------------ #
    def _forward_with_grad(self, surface: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient tracking. Input: (B,T,C,H,W) or (B,C,H,W)."""
        self.model.zero_grad(set_to_none=True)
        pred = self.model(surface)
        assert pred.requires_grad, (
            "Prediction has no grad_fn - ensure model parameters require grad "
            "and torch.is_grad_enabled() is True"
        )
        return pred

    # ------------------------------------------------------------------ #
    # Spatial saliency (Grad-CAM)
    # ------------------------------------------------------------------ #
    def generate_cam(
        self,
        surface: torch.Tensor,
        depth_idx: int,
        target_lat_idx: Optional[int] = None,
        target_lon_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM saliency for the temperature prediction at one depth.

        Args:
            surface: input tensor (B, T, C, H, W) or (B, C, H, W)
            depth_idx: index into config depth_levels (e.g. 50m/100m/200m)
            target_lat_idx: optional lat pixel to explain (default: global mean)
            target_lon_idx: optional lon pixel to explain

        Returns:
            cam: (H, W) float array in [0, 1], upsampled to input resolution
        """
        surface = surface.detach().clone()
        if surface.dim() == 4:
            surface = surface.unsqueeze(0)

        with torch.enable_grad():
            pred = self._forward_with_grad(surface)
            score = pred[:, depth_idx]
            if target_lat_idx is not None and target_lon_idx is not None:
                score = score[:, :, target_lat_idx, target_lon_idx]
            score.mean().backward()

        grads = self._gradients      # (B, C_feat, H', W')
        acts = self._activations     # (B, C_feat, H', W')

        weights = grads.mean(dim=(2, 3), keepdim=True)           # GAP weights
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (B, 1, H', W')
        cam = F.interpolate(cam, size=surface.shape[-2:], mode="bilinear", align_corners=False)

        cam_np = cam[0, 0].cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-12:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np

    # ------------------------------------------------------------------ #
    # Input-variable attribution (channel importance)
    # ------------------------------------------------------------------ #
    def attribute_input_variables(
        self,
        surface: torch.Tensor,
        depth_idx: int,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Gradient-based feature attribution: contribution of each surface input
        variable to the temperature prediction at `depth_idx`.

        Returns:
            importances: {var_name: relative importance (sums to 1.0)}
            spatial_maps: {var_name: (H, W) attribution map in [0, 1]}
        """
        surface = surface.detach().clone().requires_grad_(True)
        if surface.dim() == 4:
            surface = surface.unsqueeze(0)

        with torch.enable_grad():
            pred = self._forward_with_grad(surface)
            pred[:, depth_idx].mean().backward()

        grads = surface.grad.abs().sum(dim=0)  # (T, C, H, W)

        importances: Dict[str, float] = {}
        spatial_maps: Dict[str, np.ndarray] = {}

        for c, name in enumerate(VARIABLE_NAMES):
            g = grads[:, c]                        # (T, H, W)
            spatial = g.mean(dim=0).cpu().numpy()  # (H, W) mean over time
            importances[name] = float(g.mean().item())
            smin, smax = spatial.min(), spatial.max()
            if smax - smin > 1e-12:
                spatial = (spatial - smin) / (smax - smin)
            else:
                spatial = np.zeros_like(spatial)
            spatial_maps[name] = spatial

        total = sum(importances.values())
        if total > 1e-12:
            importances = {k: v / total for k, v in importances.items()}

        return importances, spatial_maps

    # ------------------------------------------------------------------ #
    # Eddy / upwelling driver analysis
    # ------------------------------------------------------------------ #
    def identify_oceanographic_drivers(
        self,
        importances: Dict[str, float],
        top_k: int = 3,
    ) -> Dict[str, object]:
        """
        Interpret variable importances in oceanographic terms:
          * High SSH + currents  -> mesoscale eddy driving
          * High winds           -> wind-driven upwelling / Ekman pumping
          * High SST + SSS only  -> thermohaline / barrier-layer control
        """
        ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        top_names = {n for n, _ in ranked[:top_k]}

        if "SSH" in top_names and ({"U_curr", "V_curr"} & top_names):
            regime = "MESOSCALE EDDY"
            description = ("Sea-surface height anomaly and geostrophic current "
                           "signature dominate - consistent with an eddy-driven "
                           "thermocline displacement.")
        elif {"U_wind", "V_wind"} & top_names:
            regime = "WIND-DRIVEN UPWELLING"
            description = ("Wind stress components dominate - consistent with "
                           "Ekman pumping / coastal upwelling forcing.")
        elif {"SST", "SSS"} & top_names:
            regime = "THERMOHALINE"
            description = "Surface temperature/salinity structure dominates."
        else:
            regime = "MIXED FORCING"
            description = "No single driver dominates; mixed dynamical forcing."

        return {
            "regime": regime,
            "description": description,
            "top_variables": [{"variable": n, "importance": float(v)} for n, v in ranked[:top_k]],
        }

    # ------------------------------------------------------------------ #
    # Overlay visualization
    # ------------------------------------------------------------------ #
    def generate_attribution_overlay(
        self,
        surface: torch.Tensor,
        depth_idx: int,
        depth_value_m: Optional[float] = None,
        save_path: Optional[str] = None,
        title: str = "OceanEmbed XAI Attribution",
    ):
        """
        Side-by-side attribution overlay:
          [ SST field ] [ SSH eddy signature ] [ Grad-CAM @ depth ] [ variable bars ]

        Returns a matplotlib Figure; optionally saves PNG to `save_path`.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surface_in = surface[:1] if surface.dim() == 5 else surface.unsqueeze(0)

        cam = self.generate_cam(surface_in, depth_idx)
        importances, _ = self.attribute_input_variables(surface_in, depth_idx)
        drivers = self.identify_oceanographic_drivers(importances)

        sst = surface_in[0, -1, 0].cpu().numpy()
        ssh = surface_in[0, -1, 2].cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.patch.set_facecolor("#0b1220")

        ax = axes[0]; ax.set_facecolor("#0b1220")
        im0 = ax.imshow(sst, cmap="RdYlBu_r", origin="lower")
        ax.set_title("SST (Day 7)", color="#10b981", fontsize=11)
        plt.colorbar(im0, ax=ax, fraction=0.046)

        ax = axes[1]; ax.set_facecolor("#0b1220")
        im1 = ax.imshow(ssh, cmap="viridis", origin="lower")
        ax.set_title("SSH / Sea Level Anomaly (eddy signature)", color="#10b981", fontsize=11)
        plt.colorbar(im1, ax=ax, fraction=0.046)

        ax = axes[2]; ax.set_facecolor("#0b1220")
        ax.imshow(sst, cmap="gray", origin="lower")
        im2 = ax.imshow(cam, cmap="inferno", alpha=0.55, origin="lower", vmin=0, vmax=1)
        depth_lbl = f"{depth_value_m:.0f}m" if depth_value_m is not None else f"idx {depth_idx}"
        ax.set_title(f"Grad-CAM @ {depth_lbl} | {drivers['regime']}", color="#fbbf24", fontsize=11)
        plt.colorbar(im2, ax=ax, fraction=0.046)

        ax = axes[3]; ax.set_facecolor("#0b1220")
        names = DYNAMIC_VARIABLES
        vals = [importances[n] for n in names]
        colors = ["#10b981" if v >= max(vals) else "#3b82f6" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.set_title("Input Variable Attribution", color="#fbbf24", fontsize=11)
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")

        fig.suptitle(title, color="#e2e8f0", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
            print(f"  Saved attribution overlay -> {save_path}")

        return fig

    # ------------------------------------------------------------------ #
    # Convenience: full explanation package (used by API / dashboard)
    # ------------------------------------------------------------------ #
    def explain(
        self,
        surface: torch.Tensor,
        depth_idx: int,
        depth_value_m: Optional[float] = None,
    ) -> Dict[str, object]:
        """Full explanation bundle (JSON-serializable)."""
        surface_in = surface[:1] if surface.dim() == 5 else surface.unsqueeze(0)

        cam = self.generate_cam(surface_in, depth_idx)
        importances, spatial_maps = self.attribute_input_variables(surface_in, depth_idx)
        drivers = self.identify_oceanographic_drivers(importances)

        return {
            "depth_idx": depth_idx,
            "depth_m": depth_value_m,
            "cam_heatmap": cam.tolist(),
            "cam_shape": list(cam.shape),
            "variable_importances": importances,
            "spatial_maps": {k: v.tolist() for k, v in spatial_maps.items()},
            "drivers": drivers,
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Testing OceanGradCAM Explainability Engine")
    print("=" * 70)

    from src.models.ocean_embed_net import OceanSpatiotemporalNet

    model = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    explainer = OceanGradCAM(model)
    print(f"  Target layer: {explainer.target_layer}")

    B, T, C, H, W = 1, 7, 8, 64, 96
    x = torch.randn(B, T, C, H, W)

    depth_idx = 7  # 100m - thermocline core
    cam = explainer.generate_cam(x, depth_idx)
    assert cam.shape == (H, W), f"CAM shape mismatch: {cam.shape}"
    print(f"  Grad-CAM shape: {cam.shape}, range [{cam.min():.3f}, {cam.max():.3f}]")

    importances, spatial_maps = explainer.attribute_input_variables(x, depth_idx)
    print("  Variable importances:")
    for name, val in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"    {name:8s}: {val:.3f}")

    drivers = explainer.identify_oceanographic_drivers(importances)
    print(f"  Regime: {drivers['regime']}")

    fig = explainer.generate_attribution_overlay(
        x, depth_idx, depth_value_m=100.0, save_path="assets/xai_attribution_test.png"
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    print("\n✓ XAI engine tests passed!")




