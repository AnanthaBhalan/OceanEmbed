# OceanEmbed Quick Start Guide

## 🚀 Installation & Setup (5 minutes)

### Step 1: Create Virtual Environment

```powershell
# Navigate to project directory
cd ocean_embed

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate

# Verify activation (you should see (venv) in prompt)
```

### Step 2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# This will install:
# - PyTorch (deep learning)
# - xarray & netCDF4 (ocean data)
# - FastAPI & Streamlit (applications)
# - scikit-learn, matplotlib (evaluation)
# - And more...
```

**Note**: Installation may take 5-10 minutes depending on your internet speed.

---

## 🧪 Verify Installation (2 minutes)

```powershell
# Run structure validation test
python test_structure.py

# You should see "✓ ALL CHECKS PASSED"
```

---

## 📊 Generate Synthetic Ocean Data (1 minute)

```powershell
# Generate 50 samples of realistic ocean data
python -c "from src.data.mock_generator import MockOceanDataGenerator; MockOceanDataGenerator('config.yaml').generate_dataset(50, 'mock_data')"

# This creates:
# - mock_data/surface/*.nc    (8-channel surface observations)
# - mock_data/subsurface/*.nc (15-depth temperature profiles)
```

**What's being generated:**
- Realistic SST, SSS, SSH patterns
- Monsoon wind dynamics
- Bay of Bengal warm pool
- Arabian Sea upwelling
- Thermocline structure with depth

---

## 🎓 Train the Model (10-30 minutes)

### Quick Training (1 epoch for testing)

```powershell
# Edit config.yaml and set num_epochs: 1
python src/training/train.py
```

### Full Training (100 epochs)

```powershell
# Use default config.yaml settings
python src/training/train.py

# Training will:
# ✓ Compute normalization statistics
# ✓ Create train/val/test splits
# ✓ Train with early stopping
# ✓ Save best model checkpoint
# ✓ Log to TensorBoard
```

**Monitor Training:**
```powershell
# In a separate terminal
tensorboard --logdir=logs
# Open http://localhost:6006 in browser
```

---

## 🔮 Run Inference

### Python API

```python
import torch
import xarray as xr
from src.models.ocean_embed_net import OceanEmbedNet
from src.data.preprocessor import OceanDataPreprocessor

# Load model
model = OceanEmbedNet("config.yaml")
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load preprocessor
preprocessor = OceanDataPreprocessor("config.yaml")
preprocessor.load_statistics("preprocessor_stats.pkl")

# Load surface data
surface_ds = xr.open_dataset("mock_data/surface/surface_20200101.nc")

# Preprocess
surface_tensor = preprocessor.preprocess_surface(surface_ds)
surface_tensor, _ = preprocessor.handle_nan_mask(
    surface_tensor, 
    np.zeros((15, 101, 241))
)

# Predict
surface_torch = torch.from_numpy(surface_tensor).unsqueeze(0).float()
with torch.no_grad():
    temperature = model(surface_torch)

# Denormalize
temp_np = temperature.numpy()[0]
for d in range(15):
    temp_np[d] = preprocessor.denormalize(temp_np[d], 'temperature')

print(f"Predicted temperature shape: {temp_np.shape}")  # (15, 101, 241)
print(f"Mean temperature: {np.nanmean(temp_np):.2f} °C")
```

---

## 🌐 Launch Web Applications

### 1. FastAPI REST API

```powershell
# Start API server
python src/app/api.py

# API will be available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

**Test the API:**
```powershell
# In another terminal or browser
curl http://localhost:8000/health

# Or use Python
import requests
response = requests.get("http://localhost:8000/info")
print(response.json())
```

### 2. Streamlit Dashboard

```powershell
# Launch interactive dashboard
streamlit run src/app/dashboard.py

# Dashboard will open at: http://localhost:8501
```

**Dashboard Features:**
- 📊 View surface input parameters (SST, SSS, SSH, currents)
- 🗺️ Explore temperature at different depths
- 📈 Examine vertical temperature profiles
- 🎯 Interactive 3D visualization
- 🌊 Real-time prediction from synthetic data

---

## 🧪 Run Complete Test Suite

```powershell
# Run full integration test
python tests/test_pipeline.py

# This tests:
# 1. Data generation
# 2. Preprocessing
# 3. Dataset loading
# 4. Model architecture
# 5. Training (1 epoch)
# 6. Inference
# 7. Evaluation metrics
# 8. API imports
```

**Expected output:**
```
================================================================================
                        ✓ ALL TESTS PASSED
================================================================================
```

---

## 📈 Evaluation & Metrics

### Generate Evaluation Report

```python
from src.evaluation.metrics import OceanMetrics
from src.evaluation.argo_validator import ArgoValidator

# Initialize metrics calculator
metrics_calc = OceanMetrics("config.yaml")

# Compute depth-wise metrics
metrics = metrics_calc.compute_depth_wise_metrics(pred, target, mask)

# Print summary
metrics_calc.print_metrics_summary(metrics, "Model Performance")

# Regional analysis
arabian_metrics = metrics_calc.compute_regional_metrics(
    pred, target, lat, lon, 'arabian_sea'
)
bay_metrics = metrics_calc.compute_regional_metrics(
    pred, target, lat, lon, 'bay_of_bengal'
)
```

### ARGO Float Validation

```python
# Validate against ARGO floats
validator = ArgoValidator("config.yaml")
argo_metrics = validator.validate_against_argo(
    predictions, argo_profiles, argo_depths
)
validator.generate_validation_report(argo_metrics)
```

---

## 🎯 Common Tasks

### Change Model Architecture

Edit `config.yaml`:
```yaml
model:
  encoder:
    type: "convnext"  # or "resnet"
    embed_dim: 512
    depths: [3, 3, 9, 3]
    dims: [96, 192, 384, 768]
```

### Adjust Training Parameters

```yaml
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.0001
  early_stopping_patience: 15
```

### Modify Loss Function Weights

```yaml
loss:
  mse_weight: 1.0
  stratification_weight: 0.3
  gradient_smoothness_weight: 0.1
```

---

## 📊 Expected Performance

| Metric | Target | Description |
|--------|--------|-------------|
| RMSE (0-100m) | < 1.5°C | Surface mixed layer |
| RMSE (100-300m) | < 2.0°C | Thermocline |
| RMSE (>300m) | < 1.0°C | Deep ocean |
| Correlation | > 0.90 | Spatial correlation |
| Training Time | ~2-4 hours | On GPU (100 epochs) |
| Inference Time | < 0.5s | Per prediction (GPU) |

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in `config.yaml`
```yaml
training:
  batch_size: 4  # Reduce from 8
```

### Issue: Import Errors

**Solution**: Ensure virtual environment is activated
```powershell
.\venv\Scripts\Activate
```

### Issue: Missing Dependencies

**Solution**: Reinstall requirements
```powershell
pip install --upgrade -r requirements.txt
```

### Issue: Slow Training

**Solution**: Check if GPU is being used
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 📚 Next Steps

1. **Experiment with Real Data**: Replace mock data with actual GLORYS12V1 and ARGO data
2. **Hyperparameter Tuning**: Optimize learning rate, batch size, architecture
3. **Transfer Learning**: Pre-train on global ocean data, fine-tune on North Indian Ocean
4. **Ensemble Methods**: Combine multiple models for robust predictions
5. **Production Deployment**: Deploy API to cloud (AWS, Azure, GCP)

---

## 💡 Tips for Best Results

1. **Data Quality**: Ensure satellite observations have minimal gaps
2. **Normalization**: Always use statistics from training data
3. **Validation**: Regularly check against independent ARGO data
4. **Monitoring**: Use TensorBoard to track training progress
5. **Checkpointing**: Save models frequently to prevent data loss

---

## 📞 Support & Resources

- **Documentation**: See README.md for detailed information
- **Configuration**: All settings in config.yaml
- **Examples**: Check tests/test_pipeline.py for usage examples

---

**You're ready to reconstruct subsurface ocean temperatures! 🌊**
