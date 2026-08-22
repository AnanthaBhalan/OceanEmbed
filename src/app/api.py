"""
FastAPI REST API for OceanEmbed
Inference + XAI (Grad-CAM) + Edge benchmark telemetry endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Dict, Optional
import os
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ocean_embed_net import OceanSpatiotemporalNet
from src.data.preprocessor import OceanDataPreprocessor
from src.evaluation.xai_explainer import OceanGradCAM
from src.deployment.export_onnx import export_onnx, verify_onnx_export, benchmark_inference


app = FastAPI(
    title="OceanEmbed Tactical API",
    description=("Subsurface ocean temperature reconstruction with Explainable AI "
                 "and sovereign edge deployment telemetry"),
    version="2.0.0",
)

model = None
explainer = None
preprocessor = None
config = None
device = None


class PredictionResponse(BaseModel):
    temperature: List[List[List[float]]]
    depth_levels: List[float]
    latitude: List[float]
    longitude: List[float]
    metadata: Dict[str, object]


@app.on_event("startup")
async def load_model():
    """Load model, XAI engine and preprocessor on startup."""
    global model, explainer, preprocessor, config, device

    print("Loading OceanEmbed spatiotemporal model...")

    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    preprocessor = OceanDataPreprocessor(config_path)
    stats_file = "preprocessor_stats.pkl"
    if Path(stats_file).exists():
        preprocessor.load_statistics(stats_file)
    else:
        print("Warning: Preprocessor statistics not found. Using default normalization.")

    # Encoder type can be overridden via env var for edge / test deployments
    encoder_type = os.environ.get("OCEANEMBED_ENCODER", "lightweight")
    model = OceanSpatiotemporalNet(config_path, encoder_type=encoder_type)
    model.to(device)
    model.eval()

    checkpoint_path = Path("checkpoints/best_model.pth")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Using untrained model.")

    # XAI engine targeting the last decoder convolution
    explainer = OceanGradCAM(model)
    print(f"✓ XAI engine ready (target: {type(explainer.target_layer).__name__})")


def _stack_surface_tensor(data: Dict[str, List]) -> torch.Tensor:
    """
    Build a normalized input tensor from channel arrays.
    Accepts (H, W) per channel [replicated to 7 days] or (T, H, W).
    Returns (1, T, C, H, W).
    """
    required_channels = ['SST', 'SSS', 'SSH', 'U_curr', 'V_curr', 'U_wind', 'V_wind', 'mask']

    arrays = []
    for channel in required_channels:
        if channel not in data:
            raise HTTPException(status_code=400, detail=f"Missing channel: {channel}")
        arr = np.array(data[channel], dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        arrays.append(arr)

    surface_tensor = np.stack(arrays, axis=0).transpose(1, 0, 2, 3)  # (T, C, H, W)
    T = surface_tensor.shape[0]

    for i, channel in enumerate(required_channels):
        if channel != 'mask' and getattr(preprocessor, "fitted", False):
            surface_tensor[:, i] = preprocessor.normalize(surface_tensor[:, i], channel)

    surface_tensor = np.nan_to_num(surface_tensor, nan=0.0)

    if T == 1:
        surface_tensor = np.repeat(surface_tensor, 7, axis=0)

    return torch.from_numpy(surface_tensor).unsqueeze(0).float().to(device)

@app.get("/")
async def root():
    return {
        "message": "OceanEmbed Tactical API",
        "version": "2.0.0",
        "endpoints": {
            "/predict_array": "POST - Predict subsurface temperature",
            "/explain": "POST - Grad-CAM heatmap + variable attribution",
            "/benchmark": "GET - Edge runtime telemetry (latency/FPS/memory)",
            "/health": "GET - Health check",
            "/info": "GET - Model information"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "xai_loaded": explainer is not None,
        "preprocessor_loaded": preprocessor is not None,
        "device": str(device)
    }


@app.get("/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    params = model.count_parameters()
    return {
        "architecture": "OceanSpatiotemporalNet (7-day ConvLSTM)",
        "parameters": params,
        "depth_levels": config['depth_levels'],
        "input_channels": config['input_channels'],
        "domain": config['domain'],
        "device": str(device),
    }


@app.post("/predict_array")
async def predict_from_array(surface_data: Dict[str, List]):
    """
    Predict subsurface temperature from a 7-day surface sequence.
    Each channel may be (H, W) [replicated over 7 days] or (T, H, W).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        surface_torch = _stack_surface_tensor(surface_data)

        with torch.no_grad():
            pred = model(surface_torch)

        pred_np = pred.cpu().numpy()[0]  # (D, H, W)
        fitted = getattr(preprocessor, "fitted", False)
        if fitted:
            pred_denorm = np.zeros_like(pred_np)
            for d in range(pred_np.shape[0]):
                pred_denorm[d] = preprocessor.denormalize(pred_np[d], 'temperature')
        else:
            pred_denorm = pred_np  # identity when stats unavailable

        return {
            "temperature": pred_denorm.tolist(),
            "depth_levels": config['depth_levels'],
            "input_shape": list(surface_torch.shape),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --------------------------------------------------------------------- #
# XAI endpoint
# --------------------------------------------------------------------- #
class ExplainRequest(BaseModel):
    surface_data: Dict[str, List] = Field(..., description="Channel arrays (H,W) or (T,H,W)")
    depth_idx: int = Field(default=7, ge=0, le=14, description="Target depth index (7 = 100m)")


@app.post("/explain")
async def explain_prediction(req: ExplainRequest):
    """
    Grad-CAM saliency heatmap + input-variable attribution
    (SST, SSS, SSH, currents, winds) for the predicted temperature at a
    given depth, with oceanographic driver interpretation.
    """
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model/XAI engine not ready")

    try:
        surface_torch = _stack_surface_tensor(req.surface_data)
        depth_value = config['depth_levels'][req.depth_idx]

        result = explainer.explain(surface_torch, req.depth_idx,
                                   depth_value_m=float(depth_value))
        result["region_note"] = ("Attribution computed on full Indian-Ocean domain "
                                 "(Arabian Sea + Bay of Bengal)")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# --------------------------------------------------------------------- #
# Benchmark endpoint
# --------------------------------------------------------------------- #
@app.get("/benchmark")
async def run_benchmark(num_iters: int = 5, height: int = 64, width: int = 96):
    """
    Automated edge-runtime latency benchmark: FPS, mean inference time and
    RAM/VRAM footprint for PyTorch and ONNX Runtime backends. Exports a
    fresh ONNX artifact if none exists.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        sample = torch.randn(1, 7, 8, height, width, device=device)

        onnx_path = Path("exports/ocean_embed_st.onnx")
        if not onnx_path.exists():
            onnx_path = export_onnx(model, sample.cpu(), str(onnx_path))

        parity = verify_onnx_export(model, sample.cpu(), str(onnx_path), atol=1e-3)

        bench_torch = benchmark_inference(model, (1, 7, 8, height, width),
                                          num_iters=num_iters, device=str(device))
        bench_ort = benchmark_inference(str(onnx_path), (1, 7, 8, height, width),
                                        num_iters=num_iters, device=str(device))

        return JSONResponse(content={
            "telemetry": {
                "pytorch": bench_torch,
                "onnxruntime": bench_ort,
                "speedup_onnx_vs_pytorch": (
                    bench_torch["mean_inference_ms"] / bench_ort["mean_inference_ms"]
                    if bench_ort["mean_inference_ms"] > 0 else None
                ),
            },
            "parity": parity,
            "onnx_artifact": str(onnx_path),
            "deployment_targets": ["NVIDIA Jetson", "Research vessel compute node",
                                   "MoES sovereign Linux cluster"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    api_config = cfg['api']
    uvicorn.run(app, host=api_config['host'], port=api_config['port'],
                reload=api_config['reload'])


