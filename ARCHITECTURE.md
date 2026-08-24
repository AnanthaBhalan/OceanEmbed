# OceanEmbed: System Architecture & Design

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Application Layer](#application-layer)
7. [Design Decisions](#design-decisions)

---

## System Overview

OceanEmbed is a production-ready deep learning system for reconstructing 3D subsurface ocean temperature from satellite surface observations.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  Satellite Surface Observations (8 channels, 101×241 grid)      │
│  • SST, SSS, SSH, Ocean Currents (U,V), Winds (U,V), Mask      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA PIPELINE                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ NetCDF Files │→ │ Preprocessor │→ │ PyTorch      │         │
│  │ (.nc format) │  │ (normalize)  │  │ DataLoader   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL ARCHITECTURE                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ENCODER (ConvNeXt)                                      │  │
│  │  Surface (8×101×241) → Latent Embedding (512×H'×W')     │  │
│  │  • Multi-scale hierarchical features                     │  │
│  │  • Skip connections for decoder                          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │  DECODER (UNet + Attention)                              │  │
│  │  Embedding → 15 Depth Temperature Maps (15×101×241)      │  │
│  │  • Depth-aware modulation                                │  │
│  │  • Spatial & channel attention                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                OUTPUT & VALIDATION                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Temperature  │→ │ Denormalize  │→ │ Quality      │         │
│  │ Predictions  │  │ (physical)   │  │ Control      │         │
│  │ (15 depths)  │  └──────────────┘  └──────────────┘         │
│  └──────────────┘                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Depth-wise   │  │ Regional     │  │ ARGO Float   │         │
│  │ Metrics      │  │ Analysis     │  │ Validation   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                              │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │  FastAPI Server  │              │  Streamlit UI    │        │
│  │  RESTful API     │              │  Interactive     │        │
│  │  /predict        │              │  Visualization   │        │
│  │  /health         │              │  Dashboard       │        │
│  └──────────────────┘              └──────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### 1. Mock Data Generator (`src/data/mock_generator.py`)

**Purpose**: Generate realistic synthetic ocean data for testing and development

**Features**:
- Physically consistent oceanography
- Monsoon dynamics (SW/NE patterns)
- Bay of Bengal warm pool
- Arabian Sea upwelling
- Thermocline structure
- Mesoscale eddies

**Output**:
```
mock_data/
├── surface/
│   ├── surface_20200101.nc  (8 channels × 101 × 241)
│   ├── surface_20200102.nc
│   └── ...
└── subsurface/
    ├── subsurface_20200101.nc  (15 depths × 101 × 241)
    ├── subsurface_20200102.nc
    └── ...
```

### 2. Preprocessor (`src/data/preprocessor.py`)

**Purpose**: Normalize and prepare data for neural network

**Pipeline**:
```python
# Step 1: Compute statistics from training data
mean, std, min, max = compute_statistics(train_data)

# Step 2: Z-score normalization
normalized = (data - mean) / std

# Step 3: Handle NaN values (land pixels)
normalized[is_land] = 0.0

# Step 4: Apply ocean mask
normalized = normalized * ocean_mask
```

**Statistics Storage**:
```python
{
  'SST': {'mean': 27.5, 'std': 2.3, 'min': 20.0, 'max': 32.0},
  'SSS': {'mean': 34.5, 'std': 1.2, 'min': 30.0, 'max': 37.0},
  ...
}
```

### 3. PyTorch Dataset (`src/data/dataset.py`)

**Purpose**: Efficient batch loading with data augmentation

**Data Augmentation**:
- Horizontal flip (50% probability)
- Gaussian noise (σ = 0.01)
- U-component sign flip (when horizontally flipped)

**Batch Structure**:
```python
batch = {
    'surface': Tensor(B, 8, 101, 241),    # Surface inputs
    'target': Tensor(B, 15, 101, 241),    # Temperature targets
    'metadata': {
        'filename': List[str],
        'time': List[datetime],
        'lat_range': List[tuple],
        'lon_range': List[tuple]
    }
}
```

---

## Model Architecture

### Encoder: Satellite Embedding Network

**Architecture Details**:

```
Input: (B, 8, 101, 241)
│
├─ Stem (Patchify)
│  └─ Conv2d(8→96, kernel=4, stride=4)
│  └─ LayerNorm
│  Output: (B, 96, 25, 60)
│
├─ Stage 1 (3 ConvNeXt blocks)
│  └─ dims=96, no downsampling
│  Output: (B, 96, 25, 60) ─────────┐ Skip 4
│                                     │
├─ Stage 2 (3 ConvNeXt blocks)       │
│  └─ Downsample 2x                  │
│  └─ dims=192                        │
│  Output: (B, 192, 13, 30) ─────┐   │ Skip 3
│                                  │   │
├─ Stage 3 (9 ConvNeXt blocks)    │   │
│  └─ Downsample 2x                │   │
│  └─ dims=384                     │   │
│  Output: (B, 384, 7, 15) ────┐  │   │ Skip 2
│                                │  │   │
├─ Stage 4 (3 ConvNeXt blocks)  │  │   │
│  └─ Downsample 2x              │  │   │
│  └─ dims=768                   │  │   │
│  Output: (B, 768, 4, 8) ───┐  │  │   │ Skip 1
│                              │  │  │   │
└─ Projection                  │  │  │   │
   └─ Conv2d(768→512, k=1)     │  │  │   │
   Output: (B, 512, 4, 8) ─────┼──┼──┼───┼─ To Decoder
                                │  │  │   │
                         Skip Connections
```

**ConvNeXt Block**:
```
Input x
│
├─ Depthwise Conv 7×7
├─ LayerNorm
├─ Pointwise Conv (expand 4x)
├─ GELU activation
├─ Pointwise Conv (compress)
├─ Layer Scale
└─ DropPath + Residual
   │
   └─ Output: x + drop_path(block(x))
```

### Decoder: Depth Reconstruction Network

**Architecture Details**:

```
Embedding (B, 512, 4, 8) + Skip Features
│
├─ Init Projection
│  └─ Conv2d(512→256) + BN + ReLU
│  Output: (B, 256, 4, 8)
│
├─ Upsample Block 1
│  ├─ Upsample 2x → (B, 128, 8, 16)
│  ├─ Concat Skip 1 (768) → (B, 896, 8, 16)
│  ├─ Conv + BN + ReLU
│  ├─ Spatial Attention
│  └─ Channel Attention
│  Output: (B, 256, 8, 16)
│
├─ Upsample Block 2
│  ├─ Upsample 2x → (B, 128, 16, 32)
│  ├─ Concat Skip 2 (384) → (B, 512, 16, 32)
│  └─ Attention
│  Output: (B, 128, 16, 32)
│
├─ Upsample Block 3
│  ├─ Upsample 2x → (B, 64, 32, 64)
│  ├─ Concat Skip 3 (192) → (B, 256, 32, 64)
│  └─ Attention
│  Output: (B, 64, 32, 64)
│
├─ Upsample Block 4
│  ├─ Upsample 2x → (B, 32, 64, 128)
│  ├─ Concat Skip 4 (96) → (B, 128, 64, 128)
│  └─ Attention
│  Output: (B, 64, 64, 128)
│
├─ Final Upsample
│  └─ ConvTranspose2d 4x → (B, 32, 101, 241)
│
└─ Depth-Specific Heads (15 heads)
   ├─ Depth 0m:   Conv(32→1) → (B, 1, 101, 241)
   ├─ Depth 5m:   Conv(32→1) → (B, 1, 101, 241)
   ├─ ...
   └─ Depth 1000m: Conv(32→1) → (B, 1, 101, 241)
   
   Stack → (B, 15, 101, 241)
```

**Attention Mechanisms**:

**Spatial Attention**:
```python
Conv(C→C/8) → ReLU → Conv(C/8→1) → Sigmoid
attention_map = sigmoid(conv2(relu(conv1(x))))
output = x * attention_map
```

**Channel Attention**:
```python
AvgPool(spatial) + MaxPool(spatial)
→ FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid
channel_weights = sigmoid(fc2(relu(fc1(pool(x)))))
output = x * channel_weights
```

### Loss Function

**Total Loss**:
```
L_total = λ₁·L_MSE + λ₂·L_strat + λ₃·L_smooth
```

**Component 1: MSE Loss**
```python
L_MSE = mean((pred - target)² · mask)
```
Purpose: Reconstruction accuracy

**Component 2: Stratification Loss**
```python
for depth i in 0..13:
    dT = T[i+1] - T[i]  # Temperature difference
    dz = z[i+1] - z[i]  # Depth difference
    gradient = dT / dz
    
    # Penalize positive gradients (temperature increasing with depth)
    L_strat += mean(ReLU(gradient) · mask)
```
Purpose: Prevent temperature inversions

**Component 3: Smoothness Loss**
```python
# Second-order spatial derivatives
d²T/dx² = T[i,j+1] - 2·T[i,j] + T[i,j-1]
d²T/dy² = T[i+1,j] - 2·T[i,j] + T[i-1,j]

L_smooth = mean((d²T/dx²)² + (d²T/dy²)²)
```
Purpose: Spatial continuity

### Physics-Informed Pipeline Enhancements
1. **BathymetryMaskedLoss:** Wraps the primary loss function (MSE + Stratification Penalty). Masking is applied via PyTorch registered buffers (`.to(device)`) ensuring zero overhead during the forward pass while explicitly zeroing gradients for landmass/seabed voxels.
2. **INT8 Edge Deployment:** Employs `IInt8EntropyCalibrator2` on a representative 7-day sliding window dataset. A strict RMSE parity guardrail enforces that quantization degradation remains below the operational threshold prior to vessel deployment.

---

## Training Pipeline

### Training Loop (`src/training/train.py`)

**Workflow**:
```
for epoch in 1..num_epochs:
    # Training phase
    for batch in train_loader:
        pred = model(batch.surface)
        loss = compute_loss(pred, batch.target, batch.surface)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    for batch in val_loader:
        pred = model(batch.surface)
        val_loss = compute_loss(pred, batch.target, batch.surface)
        metrics = compute_metrics(pred, batch.target)
    
    # Learning rate scheduling
    scheduler.step()
    
    # Checkpointing
    if val_loss < best_val_loss:
        save_checkpoint("best_model.pth")
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience_threshold:
        break
```

**Optimization**:
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (smooth decay)
- Gradient Clipping: max_norm=1.0
- Mixed Precision: Optional (FP16)

**Monitoring**:
- TensorBoard logging (losses, metrics, learning rate)
- Checkpoint saving (best model, periodic backups)
- Training history (JSON file)

---

## Inference Pipeline

### Prediction Workflow

```python
# 1. Load model
model = OceanEmbedNet("config.yaml")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 2. Load preprocessor
preprocessor = OceanDataPreprocessor("config.yaml")
preprocessor.load_statistics("preprocessor_stats.pkl")

# 3. Load surface data
surface_ds = xr.open_dataset("surface_20200101.nc")

# 4. Preprocess
surface_tensor = preprocessor.preprocess_surface(surface_ds)
surface_tensor, _ = preprocessor.handle_nan_mask(surface_tensor, dummy)

# 5. Predict
surface_torch = torch.from_numpy(surface_tensor).unsqueeze(0)
with torch.no_grad():
    pred = model(surface_torch)

# 6. Denormalize
for depth in range(15):
    pred[0, depth] = preprocessor.denormalize(pred[0, depth], 'temperature')

# 7. Save output
output_ds = xr.Dataset({
    'temperature': (['depth', 'lat', 'lon'], pred[0].numpy()),
    'depth': depth_levels,
    'lat': lat,
    'lon': lon
})
output_ds.to_netcdf("prediction_20200101.nc")
```

---

## Application Layer

### FastAPI Server (`src/app/api.py`)

**Endpoints**:

1. **GET /** - Root endpoint with API information
2. **GET /health** - Health check
3. **GET /info** - Model information
4. **POST /predict** - Main prediction endpoint
5. **POST /predict_array** - Prediction from numpy arrays

**Prediction Flow**:
```
Client → Upload NetCDF → API
                        ↓
                   Preprocess
                        ↓
                   Model Inference
                        ↓
                   Denormalize
                        ↓
                   JSON Response → Client
```

### Streamlit Dashboard (`src/app/dashboard.py`)

**Features**:
1. **Data Input**:
   - Generate synthetic data
   - Upload NetCDF files

2. **Visualization Tabs**:
   - Surface Inputs (SST, SSS, SSH, currents)
   - Depth Maps (temperature at selected depth)
   - Depth Profiles (vertical temperature curve)
   - 3D Visualization (surface plot)

3. **Interactive Controls**:
   - Depth level selection
   - Location selection (lat/lon)
   - Time index for synthetic data
   - Colormap selection

---

## Design Decisions

### Why ConvNeXt over ResNet?
- **Better feature extraction**: Hierarchical multi-scale features
- **Efficiency**: Depthwise convolutions reduce parameters
- **Stability**: Layer scaling prevents gradient issues
- **Performance**: State-of-the-art on vision tasks

### Why Separate Depth Heads?
- **Depth-specific learning**: Each depth has unique characteristics
- **Flexibility**: Independent optimization per depth
- **Interpretability**: Clear attribution to depth level
- **Robustness**: Failure at one depth doesn't affect others

### Why Physics-Informed Loss?
- **Physical consistency**: Prevents unphysical predictions
- **Improved generalization**: Constraints reduce overfitting
- **Domain knowledge**: Incorporates oceanographic principles
- **Interpretability**: Loss components have physical meaning

### Why Synthetic Data Generator?
- **Development speed**: Immediate testing without real data
- **Experimentation**: Easy to test edge cases
- **Reproducibility**: Consistent test conditions
- **Education**: Understanding data characteristics

### Why Both API and Dashboard?
- **API**: Production deployment, automation, integration
- **Dashboard**: Analysis, visualization, quality control
- **Complementary**: Different use cases, same model

---

**This architecture provides a robust, scalable, and maintainable framework for operational ocean temperature reconstruction.**
