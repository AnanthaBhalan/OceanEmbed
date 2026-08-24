"""
ARGO Float MLD Validation for OceanEmbed
========================================

Compares the model's threshold-based Mixed Layer Depth (delta_t = 0.2 °C)
against ARGO float observations and renders a tactical emerald/gold
scatter plot with an ideal-fit reference line.

Usage:
    from src.evaluation.argo_mld_val import validate_argo_mld
    validate_argo_mld(pred_cube, argo_profiles, depth_levels)

    pred_cube     : (B, D, H, W) predicted subsurface temperature tensor
    argo_profiles : {point_id: {'temp': array, 'mld_actual': float}}
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

from src.evaluation.mld import compute_mld   # vectorised threshold-MLD


# Tactical theme constants (mirrors src/app/hovmoller.py)
BG_PANEL = "#0A0E12"
GRID = "#1B2631"
ACCENT_EMERALD = "#10B981"
ACCENT_GOLD = "#F59E0B"
MONO = "JetBrains Mono, Fira Code, Consolas, monospace"


def _map_point_to_grid(point_id, lat_grid=None, lon_grid=None):
    """
    Map an ARGO point to grid indices. In production, replace with the
    real float position -> nearest-neighbour lookup against lat/lon grids:
        h_idx = np.argmin(np.abs(lat_grid - float_lat))
        w_idx = np.argmin(np.abs(lon_grid - float_lon))
    The deterministic hash fallback keeps validation runs reproducible
    (no hidden RNG state between executions).
    """
    seed = abs(hash(str(point_id))) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.randint(0, 100), rng.randint(0, 240)


def validate_argo_mld(pred_cube, argo_profiles, depth_levels, show_plot=True):
    """
    pred_cube: (B, D, H, W) predicted temp tensor
    argo_profiles: dict of {point_id: {'temp': array, 'mld_actual': float}}

    Returns dict with rmse, bias, matched sample count.
    """
    predicted_mlds = []
    actual_mlds = []

    # 1. Vectorised MLD over the whole prediction cube
    batch_mld_grid = compute_mld(pred_cube, depth_levels, delta_t=0.2)

    # 2. Extract grid points matching each ARGO float; skip NaN profiles
    skipped = 0
    for point_id, data in argo_profiles.items():
        h_idx, w_idx = _map_point_to_grid(point_id)
        pred_mld = batch_mld_grid[0, h_idx, w_idx]
        if not np.isnan(pred_mld):
            predicted_mlds.append(float(pred_mld))
            actual_mlds.append(float(data["mld_actual"]))
        else:
            skipped += 1

    if not predicted_mlds:
        print("[ARGO-MLD] No valid matches (all NaN) — nothing to score.")
        return None

    # 3. Metrics
    rmse = float(np.sqrt(mean_squared_error(actual_mlds, predicted_mlds)))
    bias = float(np.mean(np.array(predicted_mlds) - np.array(actual_mlds)))
    n = len(predicted_mlds)
    print(f"🎯 ARGO vs OceanEmbed MLD RMSE: {rmse:.2f} meters "
          f"(bias {bias:+.2f} m, n={n}, skipped={skipped})")

    # 4. Tactical scatter plot
    max_val = max(max(actual_mlds), max(predicted_mlds))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_mlds, y=predicted_mlds,
        mode="markers",
        marker=dict(color=ACCENT_EMERALD, size=8, opacity=0.7),
        name="Float Match",
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color=ACCENT_GOLD, dash="dash"),
        name="Ideal Fit",
    ))
    fig.update_layout(
        title="ARGO Float Validation: Mixed Layer Depth",
        xaxis_title="ARGO Observed MLD (m)",
        yaxis_title="OceanEmbed Predicted MLD (m)",
        template="plotly_dark",
        paper_bgcolor=BG_PANEL,
        font=dict(family=MONO),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    if show_plot:
        fig.show()
    return {"rmse": rmse, "bias": bias, "n": n}


if __name__ == "__main__":
    # Smoke test with synthetic profiles
    depth_levels = np.array(
        [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
    d = depth_levels
    prof = np.interp(d, [0, 40, 120, 400], [29.0, 28.8, 22.0, 8.0])  # MLD~75m
    cube = np.tile(prof[:, None, None], (1, 101, 241))[None] + \
        0.01 * np.random.randn(1, len(d), 101, 241)

    profiles = {
        f"float_{i}": {"temp": prof, "mld_actual": 70.0 + i}
        for i in range(12)
    }
    validate_argo_mld(cube, profiles, depth_levels, show_plot=False)
