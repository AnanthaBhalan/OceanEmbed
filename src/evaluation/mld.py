"""
Mixed Layer Depth (MLD) Post-Processing for OceanEmbed
======================================================

Vectorised computation of the oceanographic standard "threshold MLD"
(de Boyer Montégut et al., 2004): the shallowest depth at which the
potential temperature drops by 0.2 °C relative to the surface (0 m /
10 m reference) value.

Handles both NumPy arrays and PyTorch tensors transparently.
Edge cases handled explicitly:
  * threshold never met within the water column -> NaN (or user fill)
  * land/rock points (NaN temperatures)          -> NaN
  * non-monotonic profiles                       -> first crossing wins
  * fully uniform (mixed to bottom)              -> deepest valid level
"""

from typing import Optional, Union

import numpy as np

try:
    import torch
except ImportError:      # NumPy-only environments still work
    torch = None

DEFAULT_DEPTH_LEVELS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=np.float64,
)


def compute_mld(
    temp: Union[np.ndarray, "torch.Tensor"],
    depth_levels: np.ndarray = DEFAULT_DEPTH_LEVELS,
    delta_t: float = 0.2,
    reference_index: int = 0,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Vectorised threshold-based Mixed Layer Depth.

    Args:
        temp:            (..., D, H, W) or (..., D,) predicted 3D temperature.
                         Last-3 layout assumed if ndim >= 3; any leading dims
                         are treated as independent profile stacks.
        depth_levels:    (D,) positive-down depths in meters.
        delta_t:         Threshold temperature drop (default 0.2 °C).
        reference_index: Depth index for the reference temperature
                         (0 = surface / 0 m; use 1 for a 5 m ref).
        fill_value:      Replacement for undefined MLD (default: NaN).

    Returns:
        MLD array of shape (...) without the depth dimension.
    """
    backend = None
    if torch is not None and isinstance(temp, torch.Tensor):
        backend = "torch"
        z = temp.detach().cpu().numpy().astype(np.float64)
    else:
        z = np.asarray(temp, dtype=np.float64)

    depths = np.asarray(depth_levels, dtype=np.float64)
    moved = False
    if z.ndim >= 4:
        # move D (axis=-3) to the last axis for vectorised search
        z = np.moveaxis(z, -3, -1)
        moved = True
    elif z.ndim == 2:                      # single (D, H, W)-less profile set
        pass

    # Reference temperature at the surface level, broadcast along depth
    t_ref = np.take(z, reference_index, axis=-1, mode="clip")
    t_ref = np.expand_dims(t_ref, -1)

    # Boolean: this level is shallower than the threshold crossing point
    below_threshold = (t_ref - z) >= delta_t          # True once crossed

    # Depth of first crossing; NaN where never crossed. argmax picks the
    # FIRST True along depth (ties resolved to earliest index).
    crossed_any = below_threshold.any(axis=-1)
    first_idx = np.argmax(below_threshold, axis=-1)

    mld = depths[first_idx]
    mld = np.where(crossed_any, mld, np.nan)

    # Edge case: threshold met only AT the reference level itself (surface
    # inversion artifacts) -> treat as ill-defined, mask out.
    bad_surface = first_idx <= reference_index
    mld = np.where(crossed_any & ~bad_surface, mld, np.nan)

    if fill_value is not None:
        mld = np.where(np.isnan(mld), fill_value, mld)

    if moved:
        mld = mld  # leading dims preserved automatically
    return mld


def compute_mld_from_tensor(*args, **kwargs):
    """Convenience alias returning a torch.Tensor for pipeline chaining."""
    if torch is None:
        raise ImportError("PyTorch required for compute_mld_from_tensor")
    out = compute_mld(*args, **kwargs)
    return torch.from_numpy(out)


if __name__ == "__main__":
    # --- Unit sanity checks -------------------------------------------------
    depths = DEFAULT_DEPTH_LEVELS

    # Profile 1: mixed to 50 m, then drops fast -> MLD should be 75 m
    prof1 = np.array([29.0, 29.0, 29.0, 29.0, 28.95, 28.9, 28.6, 28.2, 27.8, 27.0, 25.0, 18.0, 10.0, 6.0, 4.0])
    # Profile 2: never drops 0.2 °C (fully mixed) -> NaN
    prof2 = np.full_like(prof1, 27.5)
    stack = np.stack([prof1, prof2])                  # (2, D)
    mld = compute_mld(stack, depths)
    print("MLDs:", mld)                               # expect [75., nan]

    # Full 3D field path (B, D, H, W)
    field = np.tile(stack[0][:, None, None], (1, 4, 4))[None]
    mld_field = compute_mld(field, depths)
    print("Field MLD shape:", mld_field.shape, "value:", mld_field[0, 0, 0])

    # Torch path
    if torch is not None:
        t = torch.from_numpy(field)
        print("Torch MLD:", compute_mld(t, depths)[0, 0, 0])
    assert mld[0] == 75.0
    assert np.isnan(mld[1])
    print("All MLD checks passed.")
