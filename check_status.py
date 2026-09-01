import json, os, sys
from pathlib import Path

print("=== TRAINING HISTORY ===")
with open('checkpoints/training_history.json') as f:
    raw = f.read()
try:
    h = json.loads(raw)
    print(f'Train epochs: {len(h["train"])}')
    print(f'Val epochs: {len(h["val"])}')
    for i in range(len(h['train'])):
        t = h['train'][i]
        v = h['val'][i]
        rmse = v.get('avg_rmse', 'N/A')
        corr = v.get('avg_correlation', 'N/A')
        rmse_str = f'{rmse:.4f}' if isinstance(rmse, float) else str(rmse)
        corr_str = f'{corr:.4f}' if isinstance(corr, float) else str(corr)
        print(f'Epoch {i+1}: train_loss={t["loss"]:.4f}, val_loss={v["loss"]:.4f}, val_rmse={rmse_str}, val_corr={corr_str}')
except json.JSONDecodeError:
    print(f'JSON is truncated/incomplete. Total chars: {len(raw)}')
    train_losses = raw.count('"loss"')
    rmse_count = raw.count('avg_rmse')
    print(f'Approximate loss entries: {train_losses}')
    print(f'avg_rmse entries: {rmse_count}')
    print(f'Last 200 chars: ...{raw[-200:]}')

print()
print("=== MOCK DATA ===")
surface_files = sorted(Path('mock_data/surface').glob('*.nc'))
print(f'Surface files: {len(surface_files)}')
target_dirs = list(Path('mock_data').iterdir())
for d in target_dirs:
    if d.is_dir():
        nc_files = list(d.glob('*.nc'))
        print(f'Dir {d.name}: {len(nc_files)} files')

print()
print("=== CHECKPOINTS ===")
for f in sorted(Path('checkpoints').glob('*')):
    size_mb = f.stat().st_size / 1e6
    print(f'  {f.name}: {size_mb:.1f} MB')

print()
print("=== EXPORTS ===")
for f in sorted(Path('exports').glob('*')):
    size_mb = f.stat().st_size / 1e6
    print(f'  {f.name}: {size_mb:.1f} MB')

print()
print("=== ASSETS ===")
for f in sorted(Path('assets').glob('*')):
    if f.is_file():
        print(f'  {f.name}: {f.stat().st_size} bytes')
