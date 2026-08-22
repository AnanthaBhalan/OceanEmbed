# 🌊 OceanEmbed
**Spatiotemporal Deep Learning Framework for 3D Subsurface Ocean Temperature Reconstruction**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Edge_Ready-005C84)
![License](https://img.shields.io/badge/License-MIT-green)

OceanEmbed is an end-to-end, physics-informed deep learning framework developed for **INCOIS / Ministry of Earth Sciences (MoES) Problem Statement ID26066**. It reconstructs highly accurate, three-dimensional subsurface ocean temperature fields (0–1000m) using only surface satellite observations.

## 🎯 The Mission
Direct measurements of subsurface ocean temperatures via ARGO floats are highly accurate but spatially sparse and expensive. OceanEmbed solves this data-sparsity problem by transforming continuous, high-resolution surface satellite data into a 3D volumetric map of the ocean's thermal structure across the North Indian Ocean (Arabian Sea and Bay of Bengal).

## 🚀 Key Innovations

*   **🧠 Spatiotemporal Memory (ConvLSTM):** Unlike standard 2D CNNs, OceanEmbed processes a 7-day rolling window of satellite data. By capturing the kinetic momentum of mesoscale eddies and ocean currents, the model predicts deep-water shifts driven by fluid dynamics.
*   **⚖️ Physics-Informed Learning:** Integrated a **Thermal Stratification Penalty** into the loss function. This strictly penalizes the network for predicting physically impossible density/temperature inversions in the water column, ensuring predictions obey thermodynamic laws.
*   **🔍 Explainable AI (OceanGradCAM):** Eliminates the "black-box" problem. The XAI engine generates spatial saliency maps and variable attribution charts, allowing oceanographers to see exactly *which* surface anomaly (e.g., wind-driven upwelling vs. thermohaline fronts) triggered a deep-water prediction.
*   **⚡ Edge-Sovereign Deployment:** Optimized via ONNX export with dynamic batching. Engineered for offline, low-latency inference on edge hardware (such as an NVIDIA Jetson module) aboard research vessels without requiring cloud connectivity.
*   **📟 Tactical Command Dashboard:** A highly responsive Streamlit PoC featuring a dark command-center aesthetic, emerald telemetry, and interactive Plotly 3D thermal volumetric sections.

## 🏗️ System Architecture
*   **Input (5D Tensor):** 8 Channels (SST, SSS, SSH, U/V Currents, U/V Winds, Bathymetry) × 7 Days × 101 (Lat) × 241 (Lon). Resolution: 0.25°.
*   **Encoder:** Hierarchical ConvLSTM that compresses spatial sequences into latent dynamic representations.
*   **Decoder:** Multi-scale spatial-channel attention network projecting latent memory into 15 depth layers.
*   **Output:** 3D Temperature field at standard depths (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 meters).

## 🛠️ Installation & Setup

Designed to run seamlessly on standard PC environments, including Linux desktop distributions.

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/OceanEmbed.git
cd OceanEmbed

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate mock data & run integration tests
python tests/test_pipeline.py
```

# 💻 Usage: Launching the Command Center
OceanEmbed ships with a REST API backend and an interactive Streamlit frontend.

### Terminal 1: Start the FastAPI Backend

```bash
python src/app/api.py
# API runs on http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Terminal 2: Start the Tactical UI

```bash
streamlit run src/app/dashboard.py
# Dashboard launches on http://localhost:8501
```

## 📁 Repository Structure

```text
ocean_embed/
├── assets/                 # High-res 300 DPI visualization plots
├── src/
│   ├── data/               # NetCDF ingestion, regridding, 7-day windowing
│   ├── models/             # ConvLSTM Encoder, Attention Decoder, Physics Loss
│   ├── deployment/         # ONNX exporter and TensorRT benchmarking
│   ├── evaluation/         # Depth-wise metrics, ARGO validator, OceanGradCAM
│   └── app/                # FastAPI backend & Streamlit tactical dashboard
├── tests/                  # End-to-end integration & parity tests
├── config.yaml             # Centralized hyperparameter configuration
└── requirements.txt        # Python dependencies
```

## 👨‍💻 Author
Developed by Anantha Bhalan R for the Smart India Hackathon.
