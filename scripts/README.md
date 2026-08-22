# OceanEmbed Scripts

This directory contains utility scripts for generating visualizations and assets.

---

## Available Scripts

### 1. `generate_demo_assets.py` ✓ **READY TO USE**

**Purpose**: Generate publication-quality demo visualizations using synthetic data only.

**Requirements**:
- Python 3.10+
- numpy
- matplotlib

**Installation**:
```bash
pip install numpy matplotlib
```

**Usage**:
```bash
python scripts/generate_demo_assets.py
```

**Output**: Creates 5 high-resolution PNG files (300 DPI) in `assets/` directory:
1. `01_spatial_depth_slices.png` - Horizontal temperature maps at 5 depths
2. `02_vertical_profiles.png` - Temperature profiles at 2 key locations
3. `03_error_analysis.png` - Spatial RMSE distribution and histogram
4. `04_latent_embeddings.png` - t-SNE/PCA latent space visualization
5. `05_depth_wise_metrics.png` - RMSE/MAE/Correlation by depth

**Features**:
- ✅ No trained model required
- ✅ No full dependency installation needed
- ✅ Generates realistic synthetic oceanographic data
- ✅ 300 DPI publication quality
- ✅ ~2-3 minutes execution time

**When to Use**: Perfect for quick demos, testing visualization code, or when trained model is not available.

---

### 2. `generate_submission_assets.py` (Full Version)

**Purpose**: Generate visualizations using actual trained OceanEmbed model predictions.

**Requirements**:
- All packages from `requirements.txt`
- Trained model checkpoint in `checkpoints/`
- Preprocessor statistics in `preprocessor_stats.pkl`
- Test data in `mock_data/` directory

**Installation**:
```bash
pip install -r requirements.txt
```

**Usage**:
```bash
# First ensure you have trained the model
python src/training/train.py

# Then generate assets
python scripts/generate_submission_assets.py
```

**Additional Features** (vs demo version):
- Uses actual model predictions (not simulated)
- Real latent embeddings from ConvNeXt encoder
- Batch processing of multiple test samples
- More sophisticated error analysis
- t-SNE on real learned features

**When to Use**: For final submission, publications, or when you want to showcase actual model performance.

---

## Output Directory Structure

```
assets/
├── README.md                       # Asset documentation
├── 01_spatial_depth_slices.png    # 2.3 MB, 6000×2400 px
├── 02_vertical_profiles.png       # 353 KB, 4200×1800 px
├── 03_error_analysis.png          # 939 KB, 4800×1800 px
├── 04_latent_embeddings.png       # 469 KB, 4800×1800 px
└── 05_depth_wise_metrics.png      # 292 KB, 5400×1500 px
```

---

## Customization

### Modify Plot Parameters

Both scripts use matplotlib's `rcParams` at the top:

```python
plt.rcParams['figure.dpi'] = 300        # Change DPI
plt.rcParams['font.size'] = 10          # Change font size
plt.rcParams['savefig.dpi'] = 300       # Output DPI
```

### Change Colormap

Find lines like:
```python
cmap='RdYlBu_r'  # Red-Yellow-Blue reversed
```

Popular alternatives:
- `'viridis'` - Perceptually uniform
- `'plasma'` - High contrast
- `'thermal'` - Ocean-focused
- `'YlOrRd'` - Yellow-Orange-Red for errors

### Adjust Depth Levels

In `generate_demo_assets.py`, change:
```python
depth_indices = [0, 5, 7, 10, 12]  # Indices for 0, 50, 100, 200, 500m
```

### Modify Point Locations

In profile plots:
```python
points = {
    'Point A: (Lat, Lon)': (15.0, 65.0),
    'Point B: (Lat, Lon)': (12.0, 88.0)
}
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'numpy'`
**Solution**: Install required packages
```bash
pip install numpy matplotlib
```

### Issue: `No such file or directory: 'assets'`
**Solution**: Script creates it automatically, but you can create manually:
```bash
mkdir assets
```

### Issue: "No trained model found" warning
**Solution**: This is expected for demo script. For full script, train model first:
```bash
python src/training/train.py
```

### Issue: Plots look pixelated in presentation
**Solution**: Ensure you're using PNG files at 300 DPI (default). Don't resize in PowerPoint - crop instead.

### Issue: Script runs slowly
**Solution**: Demo script should complete in 2-3 minutes. If slower:
- Check disk space in `assets/` directory
- Close other applications
- For full script, ensure GPU is available

---

## Performance Benchmarks

### Demo Script (`generate_demo_assets.py`)
- **Execution Time**: 2-3 minutes
- **Memory Usage**: ~500 MB
- **Output Size**: ~4.4 MB (5 files)
- **Dependencies**: 2 packages (numpy, matplotlib)

### Full Script (`generate_submission_assets.py`)
- **Execution Time**: 5-10 minutes (with trained model)
- **Memory Usage**: ~2-3 GB
- **Output Size**: ~4.4 MB (5 files)
- **Dependencies**: All packages from requirements.txt

---

## Adding New Visualizations

To add a new visualization function:

1. **Define function** in the script:
```python
def plot_my_new_viz(data):
    """Create my custom visualization."""
    print("\n[X/Y] Creating my visualization...")
    
    # Your plotting code here
    fig, ax = plt.subplots(figsize=(12, 8))
    # ... plot code ...
    
    output_file = ASSETS_DIR / "06_my_new_viz.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")
```

2. **Call in main()**:
```python
def main():
    data = generate_synthetic_data()
    # ... existing plots ...
    plot_my_new_viz(data)  # Add here
```

3. **Update README.md** in assets/ directory with description

---

## Best Practices

1. **Always use 300 DPI** for publication quality
2. **Save with `bbox_inches='tight'`** to remove whitespace
3. **Use `facecolor='white'`** for clean backgrounds
4. **Close figures** with `plt.close()` to free memory
5. **Add error handling** for robustness
6. **Print progress** for user feedback
7. **Use consistent color schemes** across related plots
8. **Add text annotations** for key features
9. **Include metrics** in plots where relevant
10. **Test on multiple display sizes**

---

## Integration with Pitch Deck

### PowerPoint
1. Insert → Pictures → Select PNG files
2. Don't resize - use PowerPoint crop tool
3. Ensure "High quality" export settings

### Google Slides
1. Insert → Image → Upload from computer
2. May auto-compress - check preview
3. For best quality, insert at native size

### LaTeX/Beamer
```latex
\includegraphics[width=\textwidth]{assets/01_spatial_depth_slices.png}
```

---

## Future Enhancements

Potential additions:
- [ ] Animated depth slice movie (GIF/MP4)
- [ ] Interactive HTML visualizations (Plotly)
- [ ] Comparison with baseline methods
- [ ] Seasonal variation analysis
- [ ] Regional zoom-in plots
- [ ] 3D volumetric rendering
- [ ] Uncertainty quantification plots

---

**Last Updated**: August 21, 2026  
**Version**: 1.0.0  
**Status**: ✓ Tested and working
