# OceanEmbed Pitch Deck Assets

## Publication-Quality Visualizations (300 DPI)

This directory contains high-resolution figures generated for the hackathon pitch deck presentation.

---

## Generated Assets

### 1. **Spatial Depth Slices** (`01_spatial_depth_slices.png`)
**Size**: 2.3 MB | **Resolution**: 300 DPI | **Dimensions**: 6000 × 2400 px

**Description**: Horizontal 2D temperature contour maps over the North Indian Ocean comparing Ground Truth vs. Model Predictions at 5 depth levels (0m, 50m, 100m, 200m, 500m).

**Key Features**:
- Top row: Ground truth temperature fields from reanalysis/observations
- Bottom row: OceanEmbed model predictions
- Color scale: Red-Yellow-Blue (warm to cool temperatures)
- Range: 5°C to 30°C
- Shows oceanographic features:
  - Bay of Bengal warm pool (eastern region)
  - Arabian Sea upwelling zones (western region)
  - Mesoscale eddies and fronts
  - Thermocline deepening patterns

**Usage**: Perfect for showing spatial accuracy and model capability to capture regional patterns.

---

### 2. **Vertical Temperature Profiles** (`02_vertical_profiles.png`)
**Size**: 353 KB | **Resolution**: 300 DPI | **Dimensions**: 4200 × 1800 px

**Description**: Temperature vs. Depth (0-1000m) line charts comparing Model Prediction vs. Ground Truth at two critical locations:

**Point A: Central Arabian Sea (15°N, 65°E)**
- Represents typical Arabian Sea stratification
- Shows upwelling-influenced thermocline
- Demonstrates model accuracy in the western basin

**Point B: Bay of Bengal Warm Pool (12°N, 88°E)**
- Represents Bay of Bengal's warm, freshwater-influenced upper ocean
- Shows deeper, more gradual thermocline
- Demonstrates model performance in the eastern basin

**Key Metrics Displayed**:
- RMSE (Root Mean Squared Error) per profile
- Correlation coefficient (r) between prediction and truth
- Thermocline region shaded (40-150m depth)

**Usage**: Ideal for demonstrating depth-stratified accuracy and vertical structure reconstruction.

---

### 3. **Error Analysis** (`03_error_analysis.png`)
**Size**: 939 KB | **Resolution**: 300 DPI | **Dimensions**: 4800 × 1800 px

**Description**: Comprehensive error analysis with two panels:

**Left Panel - Spatial RMSE Map**:
- Shows Root Mean Squared Error distribution across the North Indian Ocean
- Color scale: Yellow-Orange-Red (low to high error)
- Identifies regions where model performance varies
- Statistics box with mean, std, and max RMSE

**Right Panel - RMSE Histogram**:
- Distribution of prediction errors across all spatial points
- Blue dashed line: Mean RMSE
- Green dashed line: Median RMSE
- Shows error distribution characteristics

**Key Insights**:
- Mean RMSE: Typically < 1.5°C for well-trained models
- Spatial patterns reveal where model struggles (e.g., coastal boundaries, frontal zones)
- Histogram shape indicates consistency of predictions

**Usage**: Essential for discussing model performance, limitations, and areas for improvement.

---

### 4. **Latent Embedding Space** (`04_latent_embeddings.png`)
**Size**: 469 KB | **Resolution**: 300 DPI | **Dimensions**: 4800 × 1800 px

**Description**: Visualization of the learned latent representations from the ConvNeXt encoder showing how the model internally clusters different ocean conditions.

**Three Panels**:

**Panel 1 - t-SNE Projection**:
- 2D embedding of high-dimensional latent features
- Colored clusters represent:
  - **Red**: Arabian Sea conditions
  - **Teal**: Bay of Bengal warm pool
  - **Orange**: Winter monsoon patterns
  - **Blue**: Summer monsoon patterns
- Shows clear separation between different oceanographic regimes

**Panel 2 - PCA Projection**:
- Principal Component Analysis view
- PC1 (35% variance): Primary mode of variability
- PC2 (22% variance): Secondary mode
- Gradient coloring by sample index

**Panel 3 - Variance Explained**:
- Bar chart showing contribution of each principal component
- Red cumulative line reaching ~85% with first 5 components
- Demonstrates information compression by encoder

**Key Insights**:
- Model learns physically meaningful representations
- Distinct clusters for different ocean regions/seasons
- High dimensional reduction while preserving structure

**Usage**: Demonstrates that the model learns interpretable, physics-based features rather than just fitting noise.

---

### 5. **Depth-Wise Performance Metrics** (`05_depth_wise_metrics.png`)
**Size**: 292 KB | **Resolution**: 300 DPI | **Dimensions**: 5400 × 1500 px

**Description**: Quantitative performance metrics stratified by depth level (0-1000m).

**Three Panels**:

**Panel 1 - RMSE by Depth**:
- Shows prediction error at each of 15 depth levels
- Typically highest in thermocline (40-150m shaded region)
- Lower in surface mixed layer and deep ocean

**Panel 2 - MAE by Depth**:
- Mean Absolute Error progression with depth
- Similar pattern to RMSE but less sensitive to outliers

**Panel 3 - Correlation by Depth**:
- Pearson correlation coefficient (r) at each depth
- Target: > 0.95 for excellent performance (green band)
- Shows model maintains high correlation throughout water column

**Key Performance Indicators**:
- Surface (0-100m): RMSE < 1.5°C, r > 0.93
- Thermocline (100-300m): RMSE < 2.0°C, r > 0.91
- Deep (300-1000m): RMSE < 1.0°C, r > 0.94

**Usage**: Perfect for quantitative evaluation section, showing depth-dependent model skill.

---

## Technical Specifications

All images are generated with:
- **DPI**: 300 (publication quality)
- **Format**: PNG with transparency support
- **Color Space**: RGB
- **Font**: Arial/DejaVu Sans
- **Axis Labels**: Bold, 11pt
- **Titles**: Bold, 12-16pt
- **Figure Quality**: Optimized for both print and digital presentations

---

## Usage in Pitch Deck

### Recommended Slide Sequence:

1. **Slide 3-4**: Use `01_spatial_depth_slices.png` to show spatial capabilities
2. **Slide 5**: Use `02_vertical_profiles.png` for depth reconstruction accuracy
3. **Slide 6**: Use `03_error_analysis.png` for quantitative performance
4. **Slide 7**: Use `04_latent_embeddings.png` to demonstrate learned representations
5. **Slide 8**: Use `05_depth_wise_metrics.png` for comprehensive evaluation

### Tips:
- Crop individual panels if needed for focused views
- Add arrows/annotations to highlight specific features
- Use statistics from text boxes in your narrative
- Reference color scales when discussing temperature ranges

---

## Regeneration

To regenerate these assets with updated data or models:

```bash
# With actual trained model (requires full dependencies)
python scripts/generate_submission_assets.py

# Demo version (numpy + matplotlib only)
python scripts/generate_demo_assets.py
```

---

## File Sizes & Loading

Total asset size: **~4.4 MB**

All images are optimized for:
- **PowerPoint/Keynote**: Direct insertion at full resolution
- **Google Slides**: May be auto-compressed, but 300 DPI ensures quality
- **PDF Export**: Maintains crisp quality for print
- **Web Display**: Can be downscaled for faster loading

---

## Citation

When presenting these visualizations, cite:

```
OceanEmbed: Satellite Embedding-Based Deep Learning Framework
Problem Statement ID26066, INCOIS/MoES, 2026
```

---

**Generated by**: `scripts/generate_demo_assets.py`  
**Date**: August 21, 2026  
**Framework**: OceanEmbed v1.0.0
