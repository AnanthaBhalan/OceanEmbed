"""
Tactical Hovmöller Diagram Component for OceanEmbed (Streamlit + Plotly)
========================================================================

Interactive time-depth Hovmöller diagram for a user-selected lat/lon.
Strict command-center styling:
  * forced dark background (#0A0E12 panel / #060A0F page)
  * monospace typography everywhere (ticks, labels, hover)
  * custom thermal colormap with emerald (#00FFA3) and gold (#FFC857) accents

Usage inside dashboard.py:

    from src.app.hovmoller import render_hovmoller_panel
    render_hovmoller_panel(temp_cube, times, lats, lons, depth_levels)

    temp_cube : (T, D, H, W) ndarray of predicted subsurface temperature (°C)
"""

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- Tactical theme constants (mirrors .streamlit/config.toml) -------------
BG_PAGE = "#060A0F"
BG_PANEL = "#0A0E12"
GRID = "#1B2631"
TEXT = "#C8D6E5"
ACCENT_EMERALD = "#00FFA3"
ACCENT_GOLD = "#FFC857"
MONO = "JetBrains Mono, Fira Code, Consolas, monospace"

# Custom thermal colormap: deep navy -> teal -> EMERALD -> gold -> white-hot
TACTICAL_THERMAL = [
    [0.00, "#04121F"],
    [0.20, "#0A3A52"],
    [0.40, "#0E7A6F"],
    [0.58, "#00FFA3"],   # emerald accent (thermocline)
    [0.78, "#FFC857"],   # gold accent (warm pool)
    [1.00, "#FFF3D6"],
]


def make_hovmoller_figure(
    temp_cube: np.ndarray,
    times: Sequence,
    lats: np.ndarray,
    lons: np.ndarray,
    depth_levels: Sequence[float],
    lat_idx: int,
    lon_idx: int,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Build the interactive time-depth Hovmöller plot for one (lat, lon) pixel.

    Args:
        temp_cube: (T, D, H, W) temperature field (°C); NaN allowed.
        times:     length-T iterable of datetimes or ISO strings.
        lats/lons: 1-D coordinate vectors (degrees).
        depth_levels: length-D positive-down depths (m).
        lat_idx/lon_idx: selected grid indices.
    """
    # Extract the vertical time series at the chosen coordinate -> (T, D)
    column = np.asarray(temp_cube)[:, :, lat_idx, lon_idx]

    fig = go.Figure(
        go.Heatmap(
            z=column.T,                          # (D, T) -> depth vs time
            x=list(times),
            y=list(depth_levels),
            colorscale=TACTICAL_THERMAL,
            zmin=zmin,
            zmax=zmax,
            reversescale=False,
            colorbar=dict(
                title=dict(text="TEMP °C", font=dict(family=MONO, size=11, color=TEXT)),
                tickfont=dict(family=MONO, color=TEXT, size=10),
                outlinecolor=GRID,
                outlinewidth=1,
                thickness=14,
                ticklen=4,
            ),
            hovertemplate=(
                "<span style='font-family:%s'>"
                "TIME %%{x}<br>DEPTH %%{y:.0f} m<br>TEMP %%{z:.2f} °C"
                "</span><extra></extra>" % MONO
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PAGE,
        font=dict(family=MONO, color=TEXT),
        title=dict(
            text=(
                "<b>HOVMÖLLER // TIME-DEPTH SECTION</b><br>"
                f"<sup>LAT {float(lats[lat_idx]):+.2f}°  "
                f"LON {float(lons[lon_idx]):+.2f}°</sup>"
            ),
            font=dict(family=MONO, size=15, color=ACCENT_EMERALD),
            x=0.02,
        ),
        margin=dict(l=60, r=30, t=70, b=50),
        height=520,
    )

    # Depth increases downward — invert Y axis; gold labels, mono ticks
    fig.update_yaxes(
        autorange="reversed",
        title_text="DEPTH (m)",
        title_font=dict(family=MONO, size=12, color=ACCENT_GOLD),
        tickfont=dict(family=MONO, size=10, color=TEXT),
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
    )
    fig.update_xaxes(
        title_text="TIME (UTC)",
        title_font=dict(family=MONO, size=12, color=ACCENT_GOLD),
        tickfont=dict(family=MONO, size=10, color=TEXT),
        gridcolor=GRID,
        linecolor=GRID,
    )

    # Emerald dotted contour marking the warm-pool envelope
    try:
        peak = float(np.nanmax(column))
        fig.add_trace(
            go.Contour(
                z=column.T,
                x=list(times),
                y=list(depth_levels),
                showscale=False,
                contours=dict(start=peak - 1.0, end=peak - 1.0, coloring="none"),
                line=dict(color=ACCENT_EMERALD, width=1, dash="dot"),
                hoverinfo="skip",
                opacity=0.7,
            )
        )
    except (ValueError, IndexError):
        pass  # degenerate column (all-NaN e.g. land) — skip overlay

    return fig


def render_hovmoller_panel(
    temp_cube: np.ndarray,
    times: Sequence,
    lats: np.ndarray,
    lons: np.ndarray,
    depth_levels: Sequence[float],
    key: str = "hov",
):
    """
    Self-contained Streamlit block: coordinate pickers + styled figure +
    tactical stat readouts. Call directly from dashboard.py.
    """
    st.markdown(
        f"""
        <style>
            .tactical-header {{ font-family:{MONO}; color:{ACCENT_EMERALD};
                                letter-spacing:2px; border-bottom:1px solid {GRID}; }}
        </style>
        <h3 class="tactical-header">▚ SUBSURFACE TIME-DEPTH SURVEILLANCE</h3>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        lat_val = st.slider(
            "LATITUDE ▸", float(np.min(lats)), float(np.max(lats)),
            float(np.mean(lats)), 0.25, format="%.2f° N", key=f"{key}_lat",
        )
    with c2:
        lon_val = st.slider(
            "LONGITUDE ▸", float(np.min(lons)), float(np.max(lons)),
            float(np.mean(lons)), 0.25, format="%.2f° E", key=f"{key}_lon",
        )

    lat_idx = int(np.argmin(np.abs(np.asarray(lats) - lat_val)))
    lon_idx = int(np.argmin(np.abs(np.asarray(lons) - lon_val)))

    finite = np.asarray(temp_cube)[np.isfinite(temp_cube)]
    zmin, zmax = (float(finite.min()), float(finite.max())) if finite.size else (None, None)

    fig = make_hovmoller_figure(
        temp_cube, times, lats, lons, depth_levels, lat_idx, lon_idx, zmin, zmax
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Stat strip: SST / deep temp / quick MLD estimate at the selected pixel
    col = np.asarray(temp_cube)[:, :, lat_idx, lon_idx]
    sst = float(np.nanmean(col[:, 0]))
    crossed = col < (sst - 0.2)
    k1, k2, k3 = st.columns(3)
    k1.metric("SST (°C)", f"{sst:.2f}")
    k2.metric("DEEP TEMP @1000m", f"{np.nanmean(col[:, -1]):.2f}")
    if crossed.any() and depth_levels[int(np.argmax(crossed[0] | ~np.isfinite(col[0])))] > 0:
        mld_idx = int(np.argmax(crossed.mean(axis=0) > 0))
        k3.metric("MLD EST (m)", f"{list(depth_levels)[mld_idx]:.0f}")
    else:
        k3.metric("MLD EST (m)", "—")
