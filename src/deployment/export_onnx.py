"""
Edge Optimization & Sovereign Export Engine for OceanEmbed
ONNX export with dynamic batching, PyTorch-vs-ONNX Runtime parity
verification, and inference latency / memory benchmarking.

Deployment targets: NVIDIA Jetson, research vessels, sovereign on-prem
Linux compute nodes. TensorRT-ready via `trtexec --onnx=model.onnx`.
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Dict, Optional

import numpy as np
import torch

DEFAULT_ONNX_NAME = "ocean_embed_st.onnx"


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #
def export_onnx(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    onnx_path: Optional[str] = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """
    Export OceanSpatiotemporalNet to an optimized .onnx artifact with
    dynamic batch axis.

    Args:
        model: PyTorch model
        sample_input: example input, e.g. (1, 7, 8, H, W)
        onnx_path: output path (default: ./exports/ocean_embed_st.onnx)
        opset_version: ONNX opset (17 recommended for ConvLSTM-style graphs)
        dynamic_batch: make the batch dimension dynamic

    Returns:
        Absolute path of the exported model.
    """
    if onnx_path is None:
        out_dir = Path("exports")
        onnx_path = str(out_dir / DEFAULT_ONNX_NAME)

    # Ensure output directory exists (also for .data external-weight sidecars)
    onnx_path = str(Path(onnx_path).resolve())  # new exporter requires absolute paths
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "temperature": {0: "batch_size"},
        }

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,   # constant folding optimization
                input_names=["input"],
                output_names=["temperature"],
                dynamic_axes=dynamic_axes,
                dynamo=False,  # stable TorchScript exporter (Windows-safe)
            )
        except TypeError:
            # Older torch without the dynamo kwarg
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["temperature"],
                dynamic_axes=dynamic_axes,
            )

    if was_training:
        model.train()

    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"  ONNX exported: {onnx_path} ({size_mb:.2f} MB)")
    return onnx_path


# --------------------------------------------------------------------- #
# Verification (PyTorch vs ONNX Runtime parity)
# --------------------------------------------------------------------- #
def verify_onnx_export(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    onnx_path: Optional[str] = None,
    atol: float = 1e-4,
) -> Dict:
    """
    Verify ONNX Runtime inference matches PyTorch outputs.

    Args:
        model: PyTorch model
        sample_input: test input tensor
        onnx_path: exported model path
        atol: absolute tolerance for np.allclose

    Returns:
        dict with max_abs_diff, mean_abs_diff, allclose flag, paths
    """
    import onnxruntime as ort

    if onnx_path is None:
        onnx_path = str(Path("exports") / DEFAULT_ONNX_NAME)

    model.eval()
    with torch.no_grad():
        pt_out = model(sample_input).cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: sample_input.cpu().numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    mean_diff = float(np.abs(pt_out - ort_out).mean())
    ok = bool(np.allclose(pt_out, ort_out, atol=atol))

    result = {
        "onnx_path": onnx_path,
        "allclose": ok,
        "atol": atol,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "pytorch_shape": list(pt_out.shape),
        "onnx_shape": list(ort_out.shape),
    }

    status = "PARITY OK" if ok else "PARITY FAILED"
    print(f"  {status} | max|delta|={max_diff:.2e} mean|delta|={mean_diff:.2e} (atol={atol})")
    return result


# --------------------------------------------------------------------- #
# Latency / throughput / memory benchmarking
# --------------------------------------------------------------------- #
def _process_memory_mb() -> float:
    """Resident RAM footprint of this process in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return -1.0


def benchmark_inference(
    model_or_path,
    input_shape: tuple = (1, 7, 8, 64, 96),
    num_warmup: int = 3,
    num_iters: int = 10,
    device: str = "cpu",
) -> Dict:
    """
    Automated inference latency benchmark.

    Accepts either a torch.nn.Module (PyTorch backend) or a path to an
    exported .onnx model (ONNX Runtime backend).

    Args:
        model_or_path: nn.Module or str path to .onnx
        input_shape: benchmark tensor shape (B, T, C, H, W)
        num_warmup: warm-up runs before timing
        num_iters: timed iterations
        device: 'cpu' or 'cuda' (PyTorch backend only)

    Returns:
        Telemetry dict: mean_ms, fps, p95_ms, memory_mb, backend, device
    """
    is_onnx = isinstance(model_or_path, (str, Path))

    if not is_onnx:
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model_or_path = model_or_path.to(dev).eval()
        dummy = torch.randn(*input_shape, device=dev)
    else:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
            else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(model_or_path), providers=providers)
        dummy = np.random.randn(*input_shape).astype(np.float32)

    def run_once():
        if not is_onnx:
            with torch.no_grad():
                model_or_path(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        else:
            sess.run(None, {sess.get_inputs()[0].name: dummy})

    for _ in range(num_warmup):
        run_once()

    latencies = []
    mem_before = _process_memory_mb()
    for _ in range(num_iters):
        t0 = time.perf_counter()
        run_once()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _process_memory_mb()

    lat = np.array(latencies)
    vram_mb = -1.0
    if not is_onnx and dev.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    telemetry = {
        "backend": "onnxruntime" if is_onnx else "pytorch",
        "device": "cuda" if (not is_onnx and dev.type == "cuda") else ("cuda" if is_onnx and device == "cuda" else "cpu"),
        "input_shape": list(input_shape),
        "num_iters": num_iters,
        "mean_inference_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "min_latency_ms": float(lat.min()),
        "max_latency_ms": float(lat.max()),
        "fps": float(1000.0 / lat.mean()),
        "ram_footprint_mb": float(mem_after),
        "ram_delta_mb": float(mem_after - mem_before),
        "vram_footprint_mb": float(vram_mb),
    }

    print(f"  [{telemetry['backend']}:{telemetry['device']}] "
          f"mean={telemetry['mean_inference_ms']:.1f} ms | "
          f"FPS={telemetry['fps']:.2f} | p95={telemetry['p95_latency_ms']:.1f} ms")
    return telemetry


def full_deployment_report(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    onnx_path: Optional[str] = None,
    num_iters: int = 10,
) -> Dict:
    """
    End-to-end sovereign deployment report: export -> parity -> benchmarks.
    """
    print("\n[1/3] Exporting ONNX...")
    onnx_path = export_onnx(model, sample_input, onnx_path)

    print("[2/3] Verifying PyTorch <-> ONNX Runtime parity...")
    parity = verify_onnx_export(model, sample_input, onnx_path)

    print("[3/3] Benchmarking inference latency...")
    bench_pytorch = benchmark_inference(model, tuple(sample_input.shape), num_iters=num_iters, device="cpu")
    bench_onnx = benchmark_inference(onnx_path, tuple(sample_input.shape), num_iters=num_iters, device="cpu")

    report = {
        "onnx_path": onnx_path,
        "parity": parity,
        "benchmark_pytorch_cpu": bench_pytorch,
        "benchmark_onnx_cpu": bench_onnx,
        "speedup_onnx_vs_torch": (
            bench_pytorch["mean_inference_ms"] / bench_onnx["mean_inference_ms"]
            if bench_onnx["mean_inference_ms"] > 0 else None
        ),
    }
    if report["speedup_onnx_vs_torch"] is not None:
        print(f"  ONNX speedup vs PyTorch CPU: {report['speedup_onnx_vs_torch']:.2f}x")
    return report


if __name__ == "__main__":
    from src.models.ocean_embed_net import OceanSpatiotemporalNet

    print("=" * 70)
    print("OceanEmbed Edge Export & Benchmark Engine")
    print("=" * 70)

    model = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    x = torch.randn(1, 7, 8, 64, 96)

    report = full_deployment_report(model, x, num_iters=5)
    assert report["parity"]["allclose"], "ONNX parity check failed!"
    print("\nDeployment engine tests passed!")


