"""End-to-end PoC validation for a trained OceanEmbed spatiotemporal model.

Loads the latest checkpoint in checkpoints/, runs inference on mock surface
data (grid_stride-aware), computes depth-wise skill metrics against the mock
target, and prints a physical-temperature summary.

Usage:
    python scripts/validate_poc.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import xarray as xr
import torch

from src.models.ocean_embed_net import OceanSpatiotemporalNet
from src.data.preprocessor import OceanDataPreprocessor


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    grid_stride = int(config.get("data", {}).get("grid_stride", 1))
    depth_levels = config["depth_levels"]

    # Preprocessor
    pre = OceanDataPreprocessor("config.yaml")
    pre.load_statistics("preprocessor_stats.pkl")

    # Model (must match how it was trained)
    model = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    if grid_stride > 1:
        model.decoder.target_shape = tuple(
            int(np.ceil(g / grid_stride)) for g in config["domain"]["grid_shape"]
        )
    ckpt = torch.load("checkpoints/best_model.pth", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load the first mock surface file and its target
    surf_files = sorted(Path("mock_data/surface").glob("*.nc"))
    subsrf_files = sorted(Path("mock_data/subsurface").glob("*.nc"))
    surf_ds = xr.open_dataset(surf_files[0])
    subsrf_ds = xr.open_dataset(subsrf_files[0])

    surf = pre.preprocess_surface(surf_ds)
    target = pre.preprocess_subsurface(subsrf_ds)
    surf, target = pre.handle_nan_mask(surf, target)

    # Replicate one day into a 7-day sequence + optional stride
    seq = np.repeat(surf[None, ...], 7, axis=0)
    tgt = target
    if grid_stride > 1:
        seq = seq[..., ::grid_stride, ::grid_stride]
        tgt = np.ascontiguousarray(tgt[..., ::grid_stride, ::grid_stride])

    x = torch.from_numpy(seq[None, ...]).float()
    with torch.no_grad():
        pred = model(x)

    # Denormalize predictions and target to °C
    pred_np = pred.numpy()[0]
    pred_deg = np.zeros_like(pred_np)
    tgt_deg = np.zeros_like(tgt)
    for d in range(len(depth_levels)):
        pred_deg[d] = pre.denormalize(pred_np[d], "temperature")
        tgt_deg[d] = pre.denormalize(tgt[d], "temperature")

    print("=" * 72)
    print("  OCEANEMBED POC VALIDATION (trained model on mock data)")
    print("=" * 72)
    print(f"  Input grid (stride {grid_stride}): {seq.shape[2:]} -> output {pred_np.shape}")
    print(f"  Predicted temperature range: "
          f"[{np.nanmin(pred_deg):.2f}, {np.nanmax(pred_deg):.2f}] °C")
    print(f"  Mean predicted: {np.nanmean(pred_deg):.2f} °C  "
          f"| Mean target: {np.nanmean(tgt_deg):.2f} °C")
    print(f"  Surface (0m) mean pred/target: "
          f"{np.nanmean(pred_deg[0]):.2f} / {np.nanmean(tgt_deg[0]):.2f} °C")
    print(f"  1000m   mean pred/target: "
          f"{np.nanmean(pred_deg[-1]):.2f} / {np.nanmean(tgt_deg[-1]):.2f} °C")

    # Depth-wise metrics on the (denormalized, physical) field
    print("\n  Depth-wise skill (denormalized °C):")
    print(f"  {'Depth(m)':>9} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'Corr':>8}")
    rmses, cors = [], []
    for i, dep in enumerate(depth_levels):
        p = pred_deg[i].ravel()
        t = tgt_deg[i].ravel()
        ok = ~(np.isnan(p) | np.isnan(t))
        p, t = p[ok], t[ok]
        if len(p) < 2:
            continue
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        bias = float(np.mean(p - t))
        if np.std(p) > 0 and np.std(t) > 0:
            corr = float(np.corrcoef(p, t)[0, 1])
        else:
            corr = np.nan
        rmses.append(rmse)
        cors.append(corr)
        print(f"  {dep:9.0f} {rmse:8.3f} {mae:8.3f} {bias:8.3f} {corr:8.3f}")
    if rmses:
        print("\n  Mean RMSE: %.3f °C | Mean Corr: %.3f" %
              (np.mean(rmses), np.nanmean(cors)))
    print("=" * 72)


if __name__ == "__main__":
    main()
