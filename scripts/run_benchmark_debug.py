import traceback
import json
import sys
from pathlib import Path
# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.models.ocean_embed_net import OceanSpatiotemporalNet
    from src.deployment.export_onnx import full_deployment_report
    import torch

    model = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    sample = torch.randn(1, 7, 8, 64, 96)

    report = full_deployment_report(model, sample, num_iters=2)
    print(json.dumps(report, indent=2, default=str))
except Exception as e:
    with open('logs/benchmark_traceback.txt', 'w', encoding='utf-8') as f:
        f.write('Exception: ' + repr(e) + '\n\n')
        traceback.print_exc(file=f)
    raise
