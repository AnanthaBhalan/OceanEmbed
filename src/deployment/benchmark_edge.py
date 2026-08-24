"""
Edge Hardware Benchmark for OceanEmbed — TensorRT / CUDA latency & FPS
======================================================================

Run directly on the target NVIDIA Jetson module or Linux desktop to log
exact end-to-end inference latency and throughput of the exported ONNX
graph under the TensorRT execution provider (INT8), with CUDA fallback.

    python -m src.deployment.benchmark_edge \
        --onnx exports/ocean_embed_st.onnx --iterations 200
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def benchmark_tensorrt_engine(
    onnx_path: str,
    batch_size: int = 1,
    window_size: int = 7,
    channels: int = 8,
    h: int = 101,
    w: int = 241,
    iterations: int = 100,
    warmup: int = 10,
):
    print("🚀 Initializing TensorRT Execution Provider...")
    # Prioritize TensorRT (INT8 + engine cache); graceful CUDA fallback.
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_int8_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./exports/trt_cache",
                "trt_timing_cache_enable": True,
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(onnx_path, providers=providers)
    active = session.get_providers()[0]

    # OceanEmbed spatiotemporal input tensor: (B, T=7 days, C=8 channels, H, W)
    dummy_input = np.random.randn(
        batch_size, window_size, channels, h, w
    ).astype(np.float32)
    input_name = session.get_inputs()[0].name

    print("🔥 Warming up engine...")
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    print(f"⏱️ Benchmarking {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    total_time = time.time() - start_time

    latency_ms = (total_time / iterations) * 1000.0
    fps = 1000.0 / latency_ms

    print(f"✅ Active provider: {active}")
    print(f"📊 Latency: {latency_ms:.2f} ms per batch")
    print(f"📈 Throughput: {fps:.2f} FPS")
    return {"provider": active, "latency_ms": latency_ms, "fps": fps}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OceanEmbed edge benchmark")
    ap.add_argument("--onnx", default="exports/ocean_embed_st.onnx")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    benchmark_tensorrt_engine(
        args.onnx,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
    )
