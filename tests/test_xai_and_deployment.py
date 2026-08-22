"""
Integration Test: XAI Engine + Edge Deployment + Tactical API
Verifies:
  1. Grad-CAM generation on 5D input tensors (no gradient-graph errors)
  2. ONNX export + ONNX Runtime inference parity
  3. API endpoints /predict_array, /explain, /benchmark
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.models.ocean_embed_net import OceanSpatiotemporalNet
from src.evaluation.xai_explainer import OceanGradCAM, VARIABLE_NAMES
from src.deployment.export_onnx import (
    export_onnx, verify_onnx_export, benchmark_inference, full_deployment_report,
)

B, T, C, H, W = 1, 7, 8, 64, 96  # reduced grid (divisible by 32) for CPU speed
DEPTH_IDX = 7  # 100m - thermocline core


def test_gradcam():
    print("\n" + "=" * 70)
    print("TEST 1: Grad-CAM / XAI Engine")
    print("=" * 70)

    model = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    explainer = OceanGradCAM(model)
    x = torch.randn(B, T, C, H, W)

    # Spatial saliency
    cam = explainer.generate_cam(x, DEPTH_IDX)
    assert cam.shape == (H, W), f"CAM shape mismatch: {cam.shape}"
    assert cam.min() >= 0.0 and cam.max() <= 1.0, "CAM not normalized to [0,1]"
    print("    ✓ Grad-CAM heatmap generated on 5D input (no grad-graph errors)")

    # Variable attribution
    importances, spatial_maps = explainer.attribute_input_variables(x, DEPTH_IDX)
    assert set(importances.keys()) == set(VARIABLE_NAMES)
    assert abs(sum(importances.values()) - 1.0) < 1e-6, "Importances do not sum to 1"
    assert all(spatial_maps[n].shape == (H, W) for n in VARIABLE_NAMES)
    print("    ✓ Per-variable channel attribution computed (SST/SSS/SSH/currents/winds)")

    drivers = explainer.identify_oceanographic_drivers(importances)
    assert drivers["regime"] and drivers["top_variables"]
    print(f"    ✓ Driver interpretation: {drivers['regime']}")

    # Overlay utility
    fig = explainer.generate_attribution_overlay(x, DEPTH_IDX, depth_value_m=100.0,
                                                 save_path="assets/xai_test_overlay.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("    ✓ Side-by-side attribution overlay generated")

    # Full explain bundle
    bundle = explainer.explain(x, DEPTH_IDX, depth_value_m=100.0)
    assert "cam_heatmap" in bundle and "variable_importances" in bundle
    print("    ✓ Full explanation bundle (API-ready) OK")

    explainer.remove_hooks()
    print("✓ TEST 1 PASSED")


def test_onnx_export_and_parity():
    print("\n" + "=" * 70)
    print("TEST 2: ONNX Export + ONNX Runtime Parity + Benchmarks")
    print("=" * 70)

    model = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    x = torch.randn(B, T, C, H, W)

    onnx_path = export_onnx(model, x, "exports/test_ocean_embed_st.onnx")
    assert Path(onnx_path).exists(), "ONNX file not created"

    # Dynamic batch check: run with a different batch size through ORT
    parity = verify_onnx_export(model, x, onnx_path, atol=1e-4)
    assert parity["allclose"], f"Parity failed: {parity}"

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out2 = sess.run(None, {sess.get_inputs()[0].name:
                           np.random.randn(2, T, C, H, W).astype(np.float32)})[0]
    assert out2.shape[0] == 2, f"Dynamic batching failed: {out2.shape}"
    print("    ✓ Dynamic batching verified (batch=2 through ONNX Runtime)")

    report = full_deployment_report(model, x, onnx_path, num_iters=5)
    assert report["parity"]["allclose"]
    assert report["benchmark_onnx_cpu"]["fps"] > 0
    print("    ✓ Latency/FPS/memory telemetry collected")

    print("✓ TEST 2 PASSED")


def test_api_endpoints():
    print("\n" + "=" * 70)
    print("TEST 3: Tactical API Endpoints (/predict_array, /explain, /benchmark)")
    print("=" * 70)

    from fastapi.testclient import TestClient
    from src.app.api import app

    with TestClient(app) as client:  # triggers startup model load
        # /health
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"]
        print("    ✓ /health OK")

        # /predict_array with (H, W) channels -> auto 7-day replication
        payload = {name: np.random.randn(H, W).astype(float).tolist()
                   for name in VARIABLE_NAMES}
        r = client.post("/predict_array", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["temperature"]) == 15, "Expected 15 depth levels"
        print("    ✓ /predict_array OK (15 depth levels returned)")

        # /explain
        r = client.post("/explain", json={"surface_data": payload, "depth_idx": DEPTH_IDX})
        assert r.status_code == 200, r.text
        ex = r.json()
        assert "cam_heatmap" in ex and "drivers" in ex
        assert ex["depth_m"] == 100.0
        print(f"    ✓ /explain OK (regime: {ex['drivers']['regime']})")

        # /benchmark
        r = client.get("/benchmark?num_iters=3")
        assert r.status_code == 200, r.text
        bm = r.json()
        assert bm["parity"]["allclose"], "Benchmark parity failed"
        assert bm["telemetry"]["onnxruntime"]["fps"] > 0
        print("    ✓ /benchmark OK (telemetry + parity)")

    print("✓ TEST 3 PASSED")


if __name__ == "__main__":
    test_gradcam()
    test_onnx_export_and_parity()
    test_api_endpoints()

    print("\n" + "=" * 70)
    print("✓✓✓ ALL XAI + DEPLOYMENT TESTS PASSED ✓✓✓")
    print("=" * 70)
