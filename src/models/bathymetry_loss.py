"""
GEBCO Bathymetry Masking for OceanEmbed
=======================================

Wraps any existing OceanEmbed criterion (e.g. PhysicsInformedLoss,
DepthWeightedMSELoss) so that predicted/target voxels lying *below the
sea floor* (solid rock in the GEBCO grid) contribute zero to the loss
and receive zero gradient during backpropagation.

Mask conventions
----------------
gebco_mask : torch.Tensor of shape (D, H, W), float or bool
    1.0 / True  -> valid ocean voxel (level above sea floor)
    0.0 / False -> solid rock (voxel below sea floor) -> masked out

Invalid voxels are zeroed BEFORE the inner loss runs, which guarantees:
  * no numerical contribution to the scalar loss
  * zero gradient flow through masked positions (d(0*x)/dx == 0)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Default OceanEmbed depth levels (meters), matching config.yaml
DEFAULT_DEPTH_LEVELS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
)


def build_gebco_3d_mask(
    gebco_elevation: np.ndarray,
    lat_grid: np.ndarray,
    depth_levels: np.ndarray = DEFAULT_DEPTH_LEVELS,
) -> np.ndarray:
    """
    Build a 3D ocean voxel mask from a 2D GEBCO elevation grid.

    Args:
        gebco_elevation: (H, W) GEBCO 'elevation' band in meters.
                         Negative = below sea level (sea floor depth),
                         positive = land.
        lat_grid:        (H,) latitude coordinates (kept for future
                         geoid/pressure corrections; unused here).
        depth_levels:    (D,) model output depth levels (positive down).

    Returns:
        (D, H, W) float32 mask: 1.0 where the level lies ABOVE the sea
        floor (valid water), else 0.0 (rock).
    """
    elev = np.asarray(gebco_elevation, dtype=np.float32)
    depths = np.asarray(depth_levels, dtype=np.float32).reshape(-1, 1, 1)

    # Sea floor depth per column (positive down). Land columns get a
    # 0 m floor => only the surface level survives as "water".
    seafloor_depth = np.clip(-elev, a_min=0.0, a_max=None)[None, :, :]

    # Water exists where the level is shallower than the floor.
    # Small +1 m tolerance keeps the exact-seafloor level valid.
    return (depths <= seafloor_depth + 1.0).astype(np.float32)


class BathymetryMaskedLoss(nn.Module):
    """
    Production loss wrapper applying a 3D GEBCO bathymetry mask.

    Example
    -------
    >>> base = PhysicsInformedLoss()                    # any module loss
    >>> crit = BathymetryMaskedLoss(base, gebco_mask)   # (D, H, W)
    >>> out = crit(pred, target, surface_mask)
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        gebco_mask: torch.Tensor,
        nan_rock: bool = False,
    ):
        super().__init__()
        if isinstance(gebco_mask, np.ndarray):
            gebco_mask = torch.from_numpy(gebco_mask)
        if gebco_mask.dim() != 3:
            raise ValueError(
                f"gebco_mask must be (D,H,W), got {tuple(gebco_mask.shape)}"
            )
        # Buffer: moves with .to(device) but is NOT a learned parameter --
        # it never receives gradients itself.
        self.register_buffer("gebco_mask", gebco_mask.float())
        self.base_criterion = base_criterion
        self.nan_rock = nan_rock

    def apply_bathymetry(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Zero-out rock voxels. Returns (pred, target, combined_mask),
        where combined_mask = surface_mask AND gebco_mask at (B,D,H,W).
        """
        B, D, H, W = pred.shape
        if tuple(self.gebco_mask.shape) != (D, H, W):
            raise ValueError(
                f"gebco_mask {tuple(self.gebco_mask.shape)} does not match "
                f"prediction grid {(D, H, W)}"
            )

        geo = self.gebco_mask.unsqueeze(0)                  # (1, D, H, W)
        if mask is None:
            combined = geo.expand(B, -1, -1, -1)
        else:
            if mask.dim() == 4 and mask.shape[1] == 1:      # (B,1,H,W) surface
                surf = mask.expand(-1, D, -1, -1)
            elif mask.dim() == 3:                           # (H, W)
                surf = mask.unsqueeze(0).unsqueeze(0).expand(B, D, -1, -1)
            else:
                surf = mask                                 # already (B,D,H,W)
            combined = surf * geo

        if self.nan_rock:
            keep = combined > 0
            pred = torch.where(keep, pred, torch.full_like(pred, float("nan")))
            target = torch.where(keep, target,
                                 torch.full_like(target, float("nan")))
        else:
            # Hard-zeroing rock voxels guarantees d(loss)/d(pred) == 0 there.
            pred = pred * combined
            target = target * combined

        return pred, target, combined

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Run the base criterion on bathymetry-masked fields."""
        pred_m, target_m, combined = self.apply_bathymetry(pred, target, mask)
        out = self.base_criterion(pred_m, target_m, combined)

        # Normalise dict-style criteria so the trainer works unchanged;
        # attach bathymetry diagnostics for logging.
        if isinstance(out, dict):
            out = dict(out)
            n_valid = combined.sum()
            out["n_valid_voxels"] = n_valid.detach()
            out["rock_fraction"] = (
                1.0 - n_valid / combined.numel()).detach()
        return out


class BathymetryMaskedForward(nn.Module):
    """
    Inference-side companion: wraps an OceanEmbed network so its 3D output
    has rock voxels forced to NaN (or a fill value) at forward time --
    downstream products (MLD, Hovmoller) never see fabricated rock temps.
    """

    def __init__(self, net: nn.Module, gebco_mask: torch.Tensor,
                 fill_value: float = float("nan")):
        super().__init__()
        if isinstance(gebco_mask, np.ndarray):
            gebco_mask = torch.from_numpy(gebco_mask)
        self.register_buffer("gebco_mask", gebco_mask.float())
        self.net = net
        self.fill_value = fill_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)                                     # (B, D, H, W)
        geo = self.gebco_mask.unsqueeze(0)
        return torch.where(geo > 0, y, torch.full_like(y, self.fill_value))


if __name__ == "__main__":
    from .loss import PhysicsInformedLoss   # existing criterion

    torch.manual_seed(0)
    B, D, H, W = 2, 15, 101, 241
    # Synthetic GEBCO-like elevation: seafloor between 200-4000 m
    elev = -np.random.uniform(200, 4000, size=(H, W)).astype(np.float32)
    mask3d = build_gebco_3d_mask(elev, np.linspace(5, 30, H))
    print(f"Rock fraction of domain: {1 - mask3d.mean():.2%}")

    crit = BathymetryMaskedLoss(PhysicsInformedLoss(),
                                torch.from_numpy(mask3d))
    pred = torch.randn(B, D, H, W, requires_grad=True)
    target = torch.randn(B, D, H, W)

    out = crit(pred, target)
    print(f"Masked loss: {out['loss'].item():.4f} | "
          f"rock_fraction={out['rock_fraction'].item():.2f}")
    out["loss"].backward()

    # Verify zero gradient at rock voxels below the deepest valid level
    deep_rock = torch.from_numpy(mask3d)[-1] == 0           # (H, W) @1000 m
    grad_rock = pred.grad[:, -1][0].reshape(H, W)[deep_rock]
    print(f"Max |grad| at rock voxels (expect ~0): "
          f"{grad_rock.abs().max():.2e}")
