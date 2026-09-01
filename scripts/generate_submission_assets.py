"""
Generate Publication-Quality Assets for OceanEmbed Hackathon Submission
Creates high-resolution (300 DPI) plots for pitch deck.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.models.ocean_embed_net import OceanSpatiotemporalNet
from src.data.preprocessor import OceanDataPreprocessor
from src.data.mock_generator import MockOceanDataGenerator
from src.evaluation.metrics import OceanMetrics

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Output directory
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print(" "*20 + "OceanEmbed Asset Generator")
print("="*80)


def load_model_and_data():
    """Load trained model, preprocessor, and test data."""
    print("\n[1/5] Loading model and data...")
    
    # Load configuration
    config_path = "config.yaml"
    
    # Check if we have generated data
    data_dir = Path("mock_data")
    if not data_dir.exists():
        print("  → Generating synthetic data (this may take a minute)...")
        generator = MockOceanDataGenerator(config_path)
        generator.generate_dataset(num_samples=20, output_dir="mock_data")
    
    # Load preprocessor
    preprocessor = OceanDataPreprocessor(config_path)
    stats_file = Path("preprocessor_stats.pkl")
    
    if stats_file.exists():
        preprocessor.load_statistics(str(stats_file))
        print("  ✓ Loaded preprocessor statistics")
    else:
        print("  → Computing preprocessor statistics...")
        preprocessor.compute_statistics("mock_data")
        preprocessor.save_statistics(str(stats_file))
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OceanSpatiotemporalNet(config_path, encoder_type="lightweight")
    model.to(device)
    model.eval()
    
    # Try to load checkpoint
    checkpoint_path = Path("checkpoints/best_model.pth")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded trained model from {checkpoint_path}")
    else:
        print("  ⚠ Warning: No trained model found. Using untrained model for demo.")
    
    # Load test data
    surface_files = sorted((data_dir / "surface").glob("*.nc"))
    subsurface_files = sorted((data_dir / "subsurface").glob("*.nc"))
    
    # Use first 5 samples for visualization
    test_files = list(zip(surface_files[:5], subsurface_files[:5]))
    
    print(f"  ✓ Loaded {len(test_files)} test samples")
    
    return model, preprocessor, test_files, device


def generate_predictions(model, preprocessor, test_files, device):
    """Generate predictions for test samples."""
    print("\n[2/5] Generating predictions...")
    
    predictions = []
    targets = []
    surfaces = []
    latents = []
    
    for surf_file, subsrf_file in tqdm(test_files, desc="  Processing"):
        # Load data
        surf_ds = xr.open_dataset(surf_file)
        subsrf_ds = xr.open_dataset(subsrf_file)
        
        # Preprocess
        surface_tensor = preprocessor.preprocess_surface(surf_ds)
        target_tensor = preprocessor.preprocess_subsurface(subsrf_ds)
        
        surface_tensor, target_tensor = preprocessor.handle_nan_mask(
            surface_tensor, target_tensor
        )
        
        # Build a 7-day sequence by replicating this single day
        # (spatiotemporal model expects (B, T=7, C=8, H, W))
        surface_seq = np.repeat(surface_tensor[np.newaxis, ...], 7, axis=0)
        
        # Predict
        surface_torch = torch.from_numpy(surface_seq).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            # Get prediction
            pred = model(surface_torch)
            
            # Get latent embeddings (spatiotemporal encoder returns 3 tensors)
            embedding, _, _ = model.encoder(surface_torch)
            
        # Store results
        predictions.append(pred.cpu().numpy()[0])
        targets.append(target_tensor)
        surfaces.append(surface_tensor)
        latents.append(embedding.cpu().numpy()[0])
        
        surf_ds.close()
        subsrf_ds.close()
    
    # Denormalize predictions and targets
    predictions_denorm = []
    targets_denorm = []
    
    for pred, target in zip(predictions, targets):
        pred_d = np.zeros_like(pred)
        target_d = np.zeros_like(target)
        
        for d in range(pred.shape[0]):
            pred_d[d] = preprocessor.denormalize(pred[d], 'temperature')
            target_d[d] = preprocessor.denormalize(target[d], 'temperature')
        
        predictions_denorm.append(pred_d)
        targets_denorm.append(target_d)
    
    print(f"  ✓ Generated {len(predictions)} predictions")
    
    return {
        'predictions': np.array(predictions_denorm),
        'targets': np.array(targets_denorm),
        'surfaces': np.array(surfaces),
        'latents': np.array(latents)
    }


def plot_spatial_depth_slices(data, preprocessor):
    """Plot 1: Spatial temperature maps at multiple depths."""
    print("\n[3/5] Creating spatial depth slice maps...")
    
    # Use first sample
    pred = data['predictions'][0]
    target = data['targets'][0]
    
    # Load configuration for coordinates
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    domain = config['domain']
    lat = np.linspace(domain['lat_min'], domain['lat_max'], domain['grid_shape'][0])
    lon = np.linspace(domain['lon_min'], domain['lon_max'], domain['grid_shape'][1])
    
    # Depth indices: 0m, 50m, 100m, 200m, 500m
    depth_indices = [0, 5, 7, 10, 12]  # 0, 50, 100, 200, 500 meters
    depth_levels = config['depth_levels']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, depth_idx in enumerate(depth_indices):
        depth = depth_levels[depth_idx]
        
        # Ground truth
        ax1 = axes[0, i]
        temp_true = target[depth_idx]
        
        # Mask NaN values
        temp_masked = np.ma.masked_invalid(temp_true)
        
        im1 = ax1.contourf(lon, lat, temp_masked, levels=20, cmap='RdYlBu_r', 
                           vmin=5, vmax=30, extend='both')
        ax1.set_title(f'Ground Truth ({depth}m)', fontweight='bold')
        ax1.set_xlabel('Longitude (°E)')
        ax1.set_ylabel('Latitude (°N)')
        ax1.coastlines = True  # Placeholder for coastline
        plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
        
        # Prediction
        ax2 = axes[1, i]
        temp_pred = pred[depth_idx]
        temp_pred_masked = np.ma.masked_invalid(temp_pred)
        
        im2 = ax2.contourf(lon, lat, temp_pred_masked, levels=20, cmap='RdYlBu_r',
                           vmin=5, vmax=30, extend='both')
        ax2.set_title(f'Model Prediction ({depth}m)', fontweight='bold')
        ax2.set_xlabel('Longitude (°E)')
        ax2.set_ylabel('Latitude (°N)')
        plt.colorbar(im2, ax=ax2, label='Temperature (°C)')
    
    plt.suptitle('Spatial Temperature Distribution at Multiple Depths', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "01_spatial_depth_slices.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_vertical_profiles(data, preprocessor):
    """Plot 2: Vertical temperature profiles at specific locations."""
    print("\n[3/5] Creating vertical profile comparisons...")
    
    # Use first sample
    pred = data['predictions'][0]
    target = data['targets'][0]
    
    # Load configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    domain = config['domain']
    depth_levels = np.array(config['depth_levels'])
    
    lat = np.linspace(domain['lat_min'], domain['lat_max'], domain['grid_shape'][0])
    lon = np.linspace(domain['lon_min'], domain['lon_max'], domain['grid_shape'][1])
    
    # Define points of interest
    # Point A: Central Arabian Sea (15°N, 65°E)
    # Point B: Bay of Bengal Warm Pool (12°N, 88°E)
    points = {
        'Point A: Central Arabian Sea\n(15°N, 65°E)': (15.0, 65.0),
        'Point B: Bay of Bengal Warm Pool\n(12°N, 88°E)': (12.0, 88.0)
    }
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (point_name, (point_lat, point_lon)) in zip(axes, points.items()):
        # Find nearest indices
        lat_idx = np.argmin(np.abs(lat - point_lat))
        lon_idx = np.argmin(np.abs(lon - point_lon))
        
        # Extract profiles
        profile_true = target[:, lat_idx, lon_idx]
        profile_pred = pred[:, lat_idx, lon_idx]
        
        # Plot
        ax.plot(profile_true, depth_levels, 'o-', linewidth=2, markersize=6,
                label='Ground Truth', color='#2E86AB', alpha=0.8)
        ax.plot(profile_pred, depth_levels, 's--', linewidth=2, markersize=6,
                label='Model Prediction', color='#A23B72', alpha=0.8)
        
        # Formatting
        ax.invert_yaxis()
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Depth (m)', fontweight='bold')
        ax.set_title(point_name, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', framealpha=0.9)
        
        # Add thermocline shading
        ax.axhspan(40, 150, alpha=0.1, color='orange', label='Thermocline')
        
        # Compute metrics for this profile
        valid = ~(np.isnan(profile_true) | np.isnan(profile_pred))
        if np.sum(valid) > 0:
            rmse = np.sqrt(np.mean((profile_true[valid] - profile_pred[valid])**2))
            corr = np.corrcoef(profile_true[valid], profile_pred[valid])[0, 1]
            
            # Add text box with metrics
            textstr = f'RMSE: {rmse:.2f}°C\nCorr: {corr:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
    
    plt.suptitle('Vertical Temperature Profile Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "02_vertical_profiles.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_error_analysis(data, preprocessor):
    """Plot 3: Spatial RMSE distribution map."""
    print("\n[3/5] Creating error analysis map...")
    
    # Compute RMSE across all samples and depths
    predictions = data['predictions']
    targets = data['targets']
    
    # Average RMSE across samples and depths
    rmse_spatial = np.zeros_like(predictions[0, 0])
    
    for pred, target in zip(predictions, targets):
        # Compute RMSE at each spatial point across all depths
        squared_errors = (pred - target) ** 2
        rmse_per_point = np.sqrt(np.mean(squared_errors, axis=0))
        rmse_spatial += rmse_per_point
    
    rmse_spatial /= len(predictions)
    
    # Load configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    domain = config['domain']
    lat = np.linspace(domain['lat_min'], domain['lat_max'], domain['grid_shape'][0])
    lon = np.linspace(domain['lon_min'], domain['lon_max'], domain['grid_shape'][1])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RMSE map
    ax1 = axes[0]
    rmse_masked = np.ma.masked_invalid(rmse_spatial)
    
    im1 = ax1.contourf(lon, lat, rmse_masked, levels=20, cmap='YlOrRd', extend='max')
    ax1.set_title('Spatial RMSE Distribution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Longitude (°E)', fontweight='bold')
    ax1.set_ylabel('Latitude (°N)', fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='RMSE (°C)')
    
    # Add statistics text
    valid_rmse = rmse_masked[~rmse_masked.mask] if hasattr(rmse_masked, 'mask') else rmse_masked.flatten()
    stats_text = f'Mean RMSE: {np.nanmean(valid_rmse):.2f}°C\n'
    stats_text += f'Std RMSE: {np.nanstd(valid_rmse):.2f}°C\n'
    stats_text += f'Max RMSE: {np.nanmax(valid_rmse):.2f}°C'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # RMSE histogram
    ax2 = axes[1]
    valid_rmse_flat = valid_rmse[~np.isnan(valid_rmse)]
    
    ax2.hist(valid_rmse_flat, bins=50, color='#E63946', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(valid_rmse_flat), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(valid_rmse_flat):.2f}°C')
    ax2.axvline(np.median(valid_rmse_flat), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(valid_rmse_flat):.2f}°C')
    
    ax2.set_xlabel('RMSE (°C)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('RMSE Distribution', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Model Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "03_error_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_latent_embeddings(data, preprocessor):
    """Plot 4: Latent embedding visualization with t-SNE."""
    print("\n[4/5] Creating latent embedding visualization...")
    
    latents = data['predictions']  # Use predictions as proxy since we need more samples
    
    # Flatten spatial dimensions of latent embeddings
    latent_array = np.array(data['latents'])  # (N, C, H, W)
    
    # Global average pooling over spatial dimensions
    latent_features = latent_array.mean(axis=(2, 3))  # (N, C)
    
    print(f"  → Latent features shape: {latent_features.shape}")
    
    # Apply PCA first to reduce dimensionality
    print("  → Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=min(50, latent_features.shape[1]))
    latent_pca = pca.fit_transform(latent_features)
    
    print(f"  → PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Apply t-SNE
    print("  → Applying t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(latent_features)-1))
    latent_2d = tsne.fit_transform(latent_pca)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 6))
    
    # t-SNE plot
    ax1 = plt.subplot(1, 3, 1)
    scatter = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=range(len(latent_2d)), cmap='viridis',
                         s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Annotate points
    for i, (x, y) in enumerate(latent_2d):
        ax1.annotate(f'Sample {i+1}', (x, y), fontsize=9, ha='center',
                    xytext=(0, -15), textcoords='offset points')
    
    ax1.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax1.set_title('Latent Space Embedding (t-SNE)', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax1, label='Sample Index')
    
    # PCA plot
    ax2 = plt.subplot(1, 3, 2)
    scatter2 = ax2.scatter(latent_pca[:, 0], latent_pca[:, 1],
                          c=range(len(latent_pca)), cmap='plasma',
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, (x, y) in enumerate(latent_pca[:, :2]):
        ax2.annotate(f'Sample {i+1}', (x, y), fontsize=9, ha='center',
                    xytext=(0, -15), textcoords='offset points')
    
    ax2.set_xlabel('PC1', fontweight='bold')
    ax2.set_ylabel('PC2', fontweight='bold')
    ax2.set_title('PCA Projection', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter2, ax=ax2, label='Sample Index')
    
    # Variance explained plot
    ax3 = plt.subplot(1, 3, 3)
    n_components = min(20, len(pca.explained_variance_ratio_))
    ax3.bar(range(1, n_components+1), 
            pca.explained_variance_ratio_[:n_components],
            color='#457B9D', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Principal Component', fontweight='bold')
    ax3.set_ylabel('Explained Variance Ratio', fontweight='bold')
    ax3.set_title('PCA Variance Explained', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add cumulative line
    ax3_twin = ax3.twinx()
    cumsum = np.cumsum(pca.explained_variance_ratio_[:n_components])
    ax3_twin.plot(range(1, n_components+1), cumsum, 'ro-', linewidth=2,
                  label='Cumulative')
    ax3_twin.set_ylabel('Cumulative Variance', fontweight='bold', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3_twin.legend(loc='lower right')
    
    plt.suptitle('Latent Embedding Space Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "04_latent_embeddings.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_depth_wise_metrics(data, preprocessor):
    """Bonus Plot: Depth-wise performance metrics."""
    print("\n[5/5] Creating depth-wise metrics plot...")
    
    predictions = data['predictions']
    targets = data['targets']
    
    # Load configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    depth_levels = np.array(config['depth_levels'])
    
    # Compute metrics for each depth
    rmse_per_depth = []
    mae_per_depth = []
    corr_per_depth = []
    
    for d in range(len(depth_levels)):
        # Collect all predictions and targets for this depth
        pred_d = predictions[:, d, :, :].flatten()
        target_d = targets[:, d, :, :].flatten()
        
        # Remove NaN
        valid = ~(np.isnan(pred_d) | np.isnan(target_d))
        pred_d = pred_d[valid]
        target_d = target_d[valid]
        
        if len(pred_d) > 0:
            rmse = np.sqrt(np.mean((pred_d - target_d) ** 2))
            mae = np.mean(np.abs(pred_d - target_d))
            corr = np.corrcoef(pred_d, target_d)[0, 1] if len(pred_d) > 1 else 0
        else:
            rmse = mae = corr = np.nan
        
        rmse_per_depth.append(rmse)
        mae_per_depth.append(mae)
        corr_per_depth.append(corr)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE
    ax1 = axes[0]
    ax1.plot(rmse_per_depth, depth_levels, 'o-', linewidth=2, markersize=8,
             color='#E63946')
    ax1.invert_yaxis()
    ax1.set_xlabel('RMSE (°C)', fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontweight='bold')
    ax1.set_title('RMSE by Depth', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhspan(40, 150, alpha=0.1, color='orange')
    
    # MAE
    ax2 = axes[1]
    ax2.plot(mae_per_depth, depth_levels, 's-', linewidth=2, markersize=8,
             color='#F4A261')
    ax2.invert_yaxis()
    ax2.set_xlabel('MAE (°C)', fontweight='bold')
    ax2.set_ylabel('Depth (m)', fontweight='bold')
    ax2.set_title('MAE by Depth', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhspan(40, 150, alpha=0.1, color='orange')
    
    # Correlation
    ax3 = axes[2]
    ax3.plot(corr_per_depth, depth_levels, '^-', linewidth=2, markersize=8,
             color='#2A9D8F')
    ax3.invert_yaxis()
    ax3.set_xlabel('Correlation (r)', fontweight='bold')
    ax3.set_ylabel('Depth (m)', fontweight='bold')
    ax3.set_title('Correlation by Depth', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axhspan(40, 150, alpha=0.1, color='orange')
    ax3.set_xlim([0, 1])
    
    plt.suptitle('Depth-Stratified Model Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "05_depth_wise_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def main():
    """Main execution function."""
    try:
        # Load model and data
        model, preprocessor, test_files, device = load_model_and_data()
        
        # Generate predictions
        data = generate_predictions(model, preprocessor, test_files, device)
        
        # Generate visualizations
        plot_spatial_depth_slices(data, preprocessor)
        plot_vertical_profiles(data, preprocessor)
        plot_error_analysis(data, preprocessor)
        plot_latent_embeddings(data, preprocessor)
        plot_depth_wise_metrics(data, preprocessor)
        
        # Summary
        print("\n" + "="*80)
        print(" "*20 + "✓ ASSET GENERATION COMPLETE")
        print("="*80)
        print(f"\nGenerated {len(list(ASSETS_DIR.glob('*.png')))} high-resolution plots (300 DPI)")
        print(f"Output directory: {ASSETS_DIR.absolute()}\n")
        
        print("Generated Assets:")
        for i, asset_file in enumerate(sorted(ASSETS_DIR.glob("*.png")), 1):
            print(f"  {i}. {asset_file.name}")
        
        print("\n✓ Ready for hackathon pitch deck!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
