"""
OceanEmbed Tactical Dashboard
Defense-grade oceanographic command center for MoES / INCOIS evaluation.

Features:
  * High-contrast dark theme (emerald / gold accents, monospace telemetry)
  * 7-day temporal playback with dynamic subsurface response
  * Interactive 3D thermal vertical slices (Arabian Sea / Bay of Bengal)
  * XAI inspection panel (Grad-CAM surface attribution at selected depth)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import yaml

from src.models.ocean_embed_net import OceanSpatiotemporalNet
from src.evaluation.xai_explainer import OceanGradCAM

# --------------------------------------------------------------------- #
# Theme & page config
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="OCEANEMBED // TACTICAL",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMERALD = "#10b981"
GOLD = "#fbbf24"
BG = "#0b1220"
PANEL = "#111a2e"
TEXT = "#e2e8f0"

st.markdown(f"""
    <style>
        .stApp {{ background-color: {BG}; color: {TEXT}; }}
        section.main > div {{ padding-top: 1rem; }}
        h1, h2, h3 {{ color: {EMERALD} !important;
                      font-family: 'Consolas', monospace; letter-spacing: 2px; }}
        .stMarkdown, .stText {{ color: {TEXT}; }}
        div[data-testid="stMetricValue"] {{
            font-family: 'Consolas', monospace; color: {GOLD}; font-size: 1.4rem; }}
        div[data-testid="stMetricLabel"] p {{ color: {TEXT}; }}
        .stSlider > div > div > div {{ background: linear-gradient(90deg, {EMERALD}, {GOLD}); }}
        code, pre, .mono {{ font-family: 'Consolas', monospace; color: {GOLD}; }}
        .stButton > button {{
            border: 1px solid {EMERALD}; background: {PANEL}; color: {EMERALD};
            font-family: 'Consolas', monospace; font-weight: bold; }}
        .stButton > button:hover {{ background: {EMERALD}; color: {BG}; }}
        /* Dark plotly-friendly expander */
        details, summary {{ color: {TEXT}; }}
    </style>
""", unsafe_allow_html=True)


def panel_header(title: str, subtitle: str):
    st.markdown(f"""
        <div style="background:{PANEL};border:1px solid #1e293b;border-left:4px solid {EMERALD};
                    padding:14px 20px;border-radius:6px;margin-bottom:14px;">
          <div style="font-family:Consolas,monospace;font-size:22px;color:{EMERALD};
                      letter-spacing:3px;font-weight:bold;">{title}</div>
          <div style="font-family:Consolas,monospace;font-size:12px;color:#94a3b8;">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------- #
# Cached model + synthetic tactical feed
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading spatiotemporal backbone...")
def load_engine():
    """Load OceanSpatiotemporalNet + XAI engine (cached)."""
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_type = "lightweight"  # edge-grade inference for live ops
    model = OceanSpatiotemporalNet(config_path, encoder_type=encoder_type)
    checkpoint = Path("checkpoints/best_model.pth")
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    explainer = OceanGradCAM(model)
    return model, explainer, config, device


def generate_tactical_feed(T: int = 7, C: int = 8, H: int = 64, W: int = 96,
                           seed: int = 42) -> np.ndarray:
    """
    Synthetic 7-day surface observation sequence with embedded
    oceanographic features: a cold-core eddy, a warm-core eddy and a
    coastal upwelling filament off the Arabian Sea coast.
    Shape: (T=7, C=8, H, W).
    """
    rng = np.random.default_rng(seed)
    lon_g, lat_g = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    seq = np.zeros((T, C, H, W), dtype=np.float32)

    for t in range(T):
        phase = t / max(T - 1, 1)
        sst = 28.0 - 4.0 * lat_g + 0.5 * np.sin(2 * np.pi * lon_g) + rng.normal(0, 0.05, (H, W))
        # Cold-core eddy drifting westward
        cx, cy = 0.62 - 0.03 * t, 0.55
        sst -= 1.8 * np.exp(-(((lon_g - cx) ** 2 + (lat_g - cy) ** 2) / 0.012))
        # Warm-core eddy
        wx, wy = 0.30 + 0.02 * t, 0.35
        sst += 1.5 * np.exp(-(((lon_g - wx) ** 2 + (lat_g - wy) ** 2) / 0.010))
        # Upwelling filament intensifying through the week
        upw = (0.6 + 2.4 * phase) * np.exp(-((lat_g - 0.72) ** 2) / 0.004) * \
              np.exp(-((lon_g - 0.18) ** 2) / 0.006)
        sst -= upw

        ssh = 0.10 * np.sin(2 * np.pi * lon_g * 2) - 0.08 * np.exp(
            -(((lon_g - cx) ** 2 + (lat_g - cy) ** 2) / 0.015)) + rng.normal(0, 0.005, (H, W))
        u_wind = -1.5 * np.exp(-((lat_g - 0.75) ** 2) / 0.01) * (0.5 + phase)
        v_wind = 0.8 * np.cos(2 * np.pi * lat_g)
        u_curr = 0.4 * np.gradient(ssh, axis=1)
        v_curr = -0.4 * np.gradient(ssh, axis=0)
        sss = 35.0 - 1.5 * lon_g + rng.normal(0, 0.03, (H, W))
        mask = np.ones((H, W), dtype=np.float32)

        seq[t] = np.stack([sst, sss, ssh * 100, u_curr, v_curr, u_wind, v_wind, mask]).astype(np.float32)

    return seq

# --------------------------------------------------------------------- #
# Plot helpers (dark tactical styling)
# --------------------------------------------------------------------- #
def dark_layout(fig: go.Figure, title: str, height: int = 480) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<span style='color:{EMERALD};font-family:Consolas'>{title}</span>",
                   font=dict(size=15)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Consolas"),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", title_font_color=TEXT),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", title_font_color=TEXT),
        margin=dict(l=50, r=30, t=60, b=40), height=height,
    )
    return fig


def plot_surface_field(field, y_axis, x_axis, title, colorscale="thermal"):
    fig = go.Figure(go.Heatmap(
        z=field, x=x_axis, y=y_axis, colorscale=colorscale,
        colorbar=dict(title=dict(text=title, font=dict(color=TEXT)),
                      tickfont=dict(color=TEXT)),
        hovertemplate="lon: %{x:.2f}°E<br>lat: %{y:.2f}°N<br>val: %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title="Longitude (°E)")
    fig.update_yaxes(title="Latitude (°N)")
    return dark_layout(fig, title)

def plot_3d_thermal_slice(temp_3d, depths, lon_axis, region_name):
    """
    Interactive 3D thermal vertical contour: X=longitude, Y=depth, Z=temperature.
    temp_3d: (D, H, W) predicted temperature slice for the region.
    Latitude band averaged to a (depth x lon) vertical section.
    """
    section = temp_3d.mean(axis=1)  # (D, W)
    fig = go.Figure(data=[go.Surface(
        z=section, x=lon_axis, y=depths,
        colorscale="RdYlBu_r",
        colorbar=dict(title=dict(text="°C", font=dict(color=TEXT)),
                      tickfont=dict(color=TEXT)),
        lighting=dict(ambient=0.6, diffuse=0.8),
        hovertemplate="lon: %{x:.1f}°E<br>depth: %{y:.0f} m<br>T: %{z:.2f} °C<extra></extra>",
    )])
    fig.update_layout(scene=dict(
        xaxis_title="Longitude (°E)", yaxis_title="Depth (m)", zaxis_title="Temperature (°C)",
        bgcolor=PANEL,
        camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
    ))
    return dark_layout(fig, f"3D THERMAL SECTION // {region_name}", height=520)

# --------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------- #
panel_header("OCEANEMBED // TACTICAL OCEANOGRAPHIC COMMAND CENTER",
             "MoES / INCOIS EVALUATION BUILD · 7-DAY CONVLSTM SPATIOTEMPORAL BACKBONE "
             "· SOVEREIGN EDGE-READY")

model, explainer, config, device = load_engine()

depth_levels = config['depth_levels']
lat = np.linspace(config['domain']['lat_min'], config['domain']['lat_max'],
                  config['domain']['grid_shape'][0])
lon = np.linspace(config['domain']['lon_min'], config['domain']['lon_max'],
                  config['domain']['grid_shape'][1])

with st.sidebar:
    st.markdown(f"<h3 style='color:{GOLD}'>MISSION CONTROL</h3>", unsafe_allow_html=True)
    st.caption("Synthetic tactical feed · Arabian Sea + Bay of Bengal")

    day_idx = st.select_slider("7-DAY TEMPORAL PLAYBACK",
                               options=list(range(7)),
                               format_func=lambda d: f"D-{6 - d:02d} ({d + 1}/7)",
                               value=6)
    play = st.toggle("▶ AUTO-PLAY TIMELINE")
    depth_idx = st.select_slider("TARGET DEPTH LAYER",
                                 options=list(range(len(depth_levels))),
                                 format_func=lambda i: f"{depth_levels[i]} m",
                                 value=7)
    run_xai = st.button("⚡ RUN GRAD-CAM ATTRIBUTION")

if play:
    import time
    placeholder = st.empty()
    for d in range(7):
        placeholder.info(f"TIMELINE PLAYBACK · DAY {d + 1}/7 · T-{6 - d}")
        time.sleep(1.2)
    st.rerun()

st.markdown(f"""
    <div style="font-family:Consolas,monospace;color:{GOLD};letter-spacing:2px;">
      TELEMETRY ▸ DAY {day_idx + 1}/7 &nbsp;|&nbsp; TARGET DEPTH {depth_levels[depth_idx]} m
      &nbsp;|&nbsp; DEVICE {str(device).upper()}
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------- #
# 7-Day temporal playback + subsurface response
# --------------------------------------------------------------------- #
feed = generate_tactical_feed()
surface_day = feed[day_idx]
H_f, W_f = surface_day.shape[1:]
feed_lat = np.linspace(lat.min(), lat.max(), H_f)
feed_lon = np.linspace(lon.min(), lon.max(), W_f)


@st.cache_data(show_spinner="Reconstructing subsurface thermal field...")
def predict_at_day(day: int, _feed: np.ndarray) -> np.ndarray:
    """Predict using a 7-frame window ending on the selected day."""
    pad = 6 - day
    window = np.concatenate([np.repeat(_feed[[0]], pad, axis=0), _feed[:day + 1]], axis=0)[-7:]
    with torch.no_grad():
        pred = model(torch.from_numpy(window).unsqueeze(0).to(device))
    return pred[0].cpu().numpy()  # (D, H, W)


pred = predict_at_day(day_idx, feed)
H_p, W_p = pred.shape[1:]
pred_lat = np.linspace(lat.min(), lat.max(), H_p)
pred_lon = np.linspace(lon.min(), lon.max(), W_p)

left, right = st.columns(2)
with left:
    st.plotly_chart(plot_surface_field(surface_day[0], feed_lat, feed_lon,
                                       f"SST FIELD // DAY {day_idx + 1}/7", "thermal"),
                    use_container_width=True)
with right:
    st.plotly_chart(plot_surface_field(pred[depth_idx], pred_lat, pred_lon,
                                       f"PREDICTED TEMPERATURE @ {depth_levels[depth_idx]} m "
                                       f"// DAY {day_idx + 1}", "RdYlBu_r"),
                    use_container_width=True)

# --------------------------------------------------------------------- #
# 3D Thermal Slicing - Arabian Sea & Bay of Bengal
# --------------------------------------------------------------------- #
panel_header("3D THERMAL VERTICAL SLICING",
             "REGIONAL SUBSURFACE STRUCTURE · ARABIAN SEA / BAY OF BENGAL · DRAG TO ROTATE")


def region_slice(region):
    (la0, la1), (lo0, lo1) = region
    la_mask = (pred_lat >= la0) & (pred_lat <= la1)
    lo_mask = (pred_lon >= lo0) & (pred_lon <= lo1)
    lon_axis = pred_lon[lo_mask]
    return pred[:, la_mask][:, :, lo_mask], lon_axis


arabian_sea = ((8, 25), (50, 78))
bay_of_bengal = ((8, 23), (80, 100))

col_as, col_bob = st.columns(2)
with col_as:
    sl_as, as_lon = region_slice(arabian_sea)
    st.plotly_chart(plot_3d_thermal_slice(sl_as, depth_levels, as_lon,
                                          "ARABIAN SEA"), use_container_width=True)
with col_bob:
    sl_bob, bob_lon = region_slice(bay_of_bengal)
    st.plotly_chart(plot_3d_thermal_slice(sl_bob, depth_levels, bob_lon,
                                          "BAY OF BENGAL"), use_container_width=True)

# --------------------------------------------------------------------- #
# XAI Inspection Panel
# --------------------------------------------------------------------- #
panel_header("XAI INSPECTION // GRAD-CAM ATTRIBUTION",
             "WHICH SURFACE ANOMALY DROVE THE THERMAL FIELD AT THE SELECTED DEPTH?")

if run_xai:
    with st.spinner("Computing Grad-CAM saliency + variable attribution..."):
        window = np.concatenate([np.repeat(feed[[0]], 6 - day_idx, axis=0),
                                 feed[:day_idx + 1]], axis=0)[-7:]
        x_t = torch.from_numpy(window).unsqueeze(0).to(device)
        result = explainer.explain(x_t, depth_idx,
                                   depth_value_m=float(depth_levels[depth_idx]))

    drv = result["drivers"]
    c1, c2, c3, _ = st.columns([1, 1, 2, 0.5])
    c1.metric("DRIVER REGIME", drv["regime"])
    c2.metric("TARGET DEPTH", f"{depth_levels[depth_idx]} m")
    c3.info(drv["description"])

    xc1, xc2 = st.columns(2)

    with xc1:
        # CAM heatmap over the latest SST field
        cam = np.array(result["cam_heatmap"])
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=surface_day[0], x=feed_lon, y=feed_lat,
                                 colorscale="Gray", opacity=0.75, showscale=False))
        fig.add_trace(go.Heatmap(z=cam, x=feed_lon, y=feed_lat, colorscale="Inferno",
                                 opacity=0.55,
                                 colorbar=dict(title=dict(text="saliency",
                                                          font=dict(color=TEXT)),
                                               tickfont=dict(color=TEXT))))
        for name, (cx, cy) in {"COLD-CORE EDDY": (0.62 * feed_lon[-1], feed_lat[int(0.55 * H_f)]),
                               "WARM-CORE EDDY": (0.30 * feed_lon[-1], feed_lat[int(0.35 * H_f)])}.items():
            fig.add_annotation(x=cx, y=cy, text=name, showarrow=True, arrowhead=2,
                               arrowcolor=GOLD, font=dict(color=GOLD, size=10))
        fig.update_xaxes(title="Longitude (°E)")
        fig.update_yaxes(title="Latitude (°N)")
        st.plotly_chart(dark_layout(fig, f"GRAD-CAM @ {depth_levels[depth_idx]} m "
                                         "// SURFACE ANOMALY OVERLAY"),
                        use_container_width=True)

    with xc2:
        imps = result["variable_importances"]
        names = [n for n in ["SST", "SSS", "SSH", "U_curr", "V_curr", "U_wind", "V_wind"]]
        vals = [imps[n] for n in names]
        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=[GOLD if v >= max(vals) else EMERALD for v in vals],
            text=[f"{v:.1%}" for v in vals], textposition="auto",
        ))
        fig.update_xaxes(title="Relative contribution")
        st.plotly_chart(dark_layout(fig, "INPUT VARIABLE ATTRIBUTION "
                                         "(SST/SSS/SSH/CURRENTS/WINDS)"),
                        use_container_width=True)

    # Top driver table
    st.markdown(f"<div class='mono'>TOP DRIVERS ▸ " +
                " &nbsp;|&nbsp; ".join(
                    f"{tv['variable']} <span style='color:{EMERALD}'>"
                    f"{tv['importance']:.1%}</span>"
                    for tv in drv["top_variables"]) + "</div>",
                unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div style='font-family:Consolas,monospace;color:#94a3b8;border:1px dashed #334155;
                    padding:16px;border-radius:6px;background:{PANEL};'>
          ▸ Awaiting operator command. Select a depth layer and press
          <span style='color:{EMERALD}'>⚡ RUN GRAD-CAM ATTRIBUTION</span> to inspect the
          surface drivers behind the predicted thermal anomaly.
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("OceanEmbed · MoES / INCOIS sovereign evaluation build · "
           "XAI engine + ConvLSTM backbone + ONNX edge export pipeline")






