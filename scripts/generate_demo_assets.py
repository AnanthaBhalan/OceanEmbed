"""
Generate Demo Publication-Quality Assets for OceanEmbed
Creates high-resolution (300 DPI) demo plots using synthetic data only.
No trained model required - uses realistic mock data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

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
print(" "*20 + "OceanEmbed Demo Asset Generator")
print("="*80)


def generate_synthetic_data():
    """Generate realistic synthetic ocean temperature data."""
    print("\n[1/5] Generating synthetic ocean data...")
    
    # Domain configuration
    lat = np.linspace(5, 30, 101)
    lon = np.linspace(45, 105, 241)
    depth_levels = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Generate realistic temperature field with oceanographic features
    def generate_temperature_field(depth, time_phase=0):
        """Generate temperature with depth-dependent patterns."""
        # Base temperature (decreases with depth)
        base_temp = 28 - 20 * (1 - np.exp(-depth / 300))
        
        # Latitude gradient (warmer near equator)
        lat_factor = 2 * np.sin((lat_grid - 5) / 25 * np.pi / 2)
        
        # Bay of Bengal warm pool
        bay_mask = (lon_grid > 80) & (lon_grid < 100) & (lat_grid < 20)
        bay_warm = 2.0 * np.exp(-depth / 50) * bay_mask
        
        # Arabian Sea upwelling (cooler)
        arabian_mask = (lon_grid > 60) & (lon_grid < 70) & (lat_grid < 15)
        arabian_cool = -1.5 * np.exp(-depth / 100) * arabian_mask
        
        # Mesoscale eddies
        eddy1 = 1.5 * np.exp(-((lon_grid - 75)**2 + (lat_grid - 15)**2) / 50) * np.exp(-depth / 100)
        eddy2 = -1.0 * np.exp(-((lon_grid - 90)**2 + (lat_grid - 20)**2) / 40) * np.exp(-depth / 100)
        
        # Seasonal variation
        seasonal = 1.0 * np.cos(time_phase) * np.exp(-depth / 50)
        
        # Combine all
        temp = base_temp + lat_factor + bay_warm + arabian_cool + eddy1 + eddy2 + seasonal
        
        # Add realistic noise
        np.random.seed(42 + int(depth))
        noise = np.random.randn(*temp.shape) * 0.5 * np.exp(-depth / 200)
        temp += noise
        
        return temp
    
    # Generate ground truth and prediction
    ground_truth = np.zeros((len(depth_levels), len(lat), len(lon)))
    prediction = np.zeros((len(depth_levels), len(lat), len(lon)))
    
    for i, depth in enumerate(depth_levels):
        ground_truth[i] = generate_temperature_field(depth, time_phase=0)
        # Prediction with small errors
        prediction[i] = ground_truth[i] + np.random.randn(*ground_truth[i].shape) * 0.8
    
    print(f"  ✓ Generated data: {len(depth_levels)} depths × {len(lat)} × {len(lon)} grid")
    
    return {
        'lat': lat,
        'lon': lon,
        'depth_levels': depth_levels,
        'ground_truth': ground_truth,
        'prediction': prediction
    }


def plot_spatial_depth_slices(data):
    """Plot 1: Spatial temperature maps at multiple depths."""
    print("\n[2/5] Creating spatial depth slice maps...")
    
    lat = data['lat']
    lon = data['lon']
    depth_levels = data['depth_levels']
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    
    # Depth indices: 0m, 50m, 100m, 200m, 500m
    depth_indices = [0, 5, 7, 10, 12]
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, depth_idx in enumerate(depth_indices):
        depth = depth_levels[depth_idx]
        
        # Ground truth
        ax1 = axes[0, i]
        im1 = ax1.contourf(lon, lat, ground_truth[depth_idx], levels=20, 
                          cmap='RdYlBu_r', vmin=5, vmax=30, extend='both')
        ax1.set_title(f'Ground Truth ({depth}m)', fontweight='bold')
        ax1.set_xlabel('Longitude (°E)')
        ax1.set_ylabel('Latitude (°N)')
        plt.colorbar(im1, ax=ax1, label='Temperature (°C)', fraction=0.046)
        
        # Prediction
        ax2 = axes[1, i]
        im2 = ax2.contourf(lon, lat, prediction[depth_idx], levels=20,
                          cmap='RdYlBu_r', vmin=5, vmax=30, extend='both')
        ax2.set_title(f'Model Prediction ({depth}m)', fontweight='bold')
        ax2.set_xlabel('Longitude (°E)')
        ax2.set_ylabel('Latitude (°N)')
        plt.colorbar(im2, ax=ax2, label='Temperature (°C)', fraction=0.046)
    
    plt.suptitle('Spatial Temperature Distribution at Multiple Depths',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "01_spatial_depth_slices.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_vertical_profiles(data):
    """Plot 2: Vertical temperature profiles at specific locations."""
    print("\n[3/5] Creating vertical profile comparisons...")
    
    lat = data['lat']
    lon = data['lon']
    depth_levels = data['depth_levels']
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    
    # Define points
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
        profile_true = ground_truth[:, lat_idx, lon_idx]
        profile_pred = prediction[:, lat_idx, lon_idx]
        
        # Plot
        ax.plot(profile_true, depth_levels, 'o-', linewidth=2.5, markersize=8,
                label='Ground Truth', color='#2E86AB', alpha=0.8)
        ax.plot(profile_pred, depth_levels, 's--', linewidth=2.5, markersize=8,
                label='Model Prediction', color='#A23B72', alpha=0.8)
        
        # Formatting
        ax.invert_yaxis()
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Depth (m)', fontweight='bold')
        ax.set_title(point_name, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='black')
        
        # Thermocline shading
        ax.axhspan(40, 150, alpha=0.15, color='orange', zorder=0)
        ax.text(0.02, 0.3, 'Thermocline', transform=ax.transAxes,
                fontsize=9, style='italic', alpha=0.7)
        
        # Compute metrics
        rmse = np.sqrt(np.mean((profile_true - profile_pred)**2))
        corr = np.corrcoef(profile_true, profile_pred)[0, 1]
        
        # Add text box
        textstr = f'RMSE: {rmse:.2f}°C\nCorr: {corr:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.suptitle('Vertical Temperature Profile Comparison',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "02_vertical_profiles.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_error_analysis(data):
    """Plot 3: Spatial RMSE distribution map."""
    print("\n[4/5] Creating error analysis map...")
    
    lat = data['lat']
    lon = data['lon']
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    
    # Compute RMSE across all depths
    squared_errors = (ground_truth - prediction) ** 2
    rmse_spatial = np.sqrt(np.mean(squared_errors, axis=0))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RMSE map
    ax1 = axes[0]
    im1 = ax1.contourf(lon, lat, rmse_spatial, levels=20, cmap='YlOrRd', extend='max')
    ax1.set_title('Spatial RMSE Distribution', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Longitude (°E)', fontweight='bold')
    ax1.set_ylabel('Latitude (°N)', fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='RMSE (°C)')
    
    # Statistics
    stats_text = f'Mean RMSE: {np.mean(rmse_spatial):.2f}°C\n'
    stats_text += f'Std RMSE: {np.std(rmse_spatial):.2f}°C\n'
    stats_text += f'Max RMSE: {np.max(rmse_spatial):.2f}°C\n'
    stats_text += f'Min RMSE: {np.min(rmse_spatial):.2f}°C'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontweight='bold')
    
    # RMSE histogram
    ax2 = axes[1]
    rmse_flat = rmse_spatial.flatten()
    
    ax2.hist(rmse_flat, bins=50, color='#E63946', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(rmse_flat), color='blue', linestyle='--',
                linewidth=2.5, label=f'Mean: {np.mean(rmse_flat):.2f}°C')
    ax2.axvline(np.median(rmse_flat), color='green', linestyle='--',
                linewidth=2.5, label=f'Median: {np.median(rmse_flat):.2f}°C')
    
    ax2.set_xlabel('RMSE (°C)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('RMSE Distribution', fontweight='bold', fontsize=14)
    ax2.legend(framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Model Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "03_error_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_latent_embeddings(data):
    """Plot 4: Simulated latent embedding visualization."""
    print("\n[5/5] Creating latent embedding visualization...")
    
    # Simulate latent embeddings for demonstration
    np.random.seed(42)
    n_samples = 20
    
    # Generate clusters representing different ocean regions/conditions
    cluster1 = np.random.randn(n_samples//4, 2) * 0.5 + np.array([2, 2])  # Arabian Sea
    cluster2 = np.random.randn(n_samples//4, 2) * 0.5 + np.array([-2, 2])  # Bay of Bengal
    cluster3 = np.random.randn(n_samples//4, 2) * 0.5 + np.array([2, -2])  # Winter monsoon
    cluster4 = np.random.randn(n_samples//4, 2) * 0.5 + np.array([-2, -2])  # Summer monsoon
    
    latent_2d = np.vstack([cluster1, cluster2, cluster3, cluster4])
    labels = ['Arabian Sea']*5 + ['Bay of Bengal']*5 + ['Winter Monsoon']*5 + ['Summer Monsoon']*5
    colors = ['#E63946']*5 + ['#2A9D8F']*5 + ['#F4A261']*5 + ['#457B9D']*5
    
    # Simulate PCA variance
    pca_variance = np.array([0.35, 0.22, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02])
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    
    # t-SNE plot
    ax1 = plt.subplot(1, 3, 1)
    for i, (x, y) in enumerate(latent_2d):
        ax1.scatter(x, y, c=colors[i], s=200, alpha=0.7, 
                   edgecolors='black', linewidth=1.5)
        ax1.annotate(f'{i+1}', (x, y), fontsize=9, ha='center', va='center',
                    fontweight='bold', color='white')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E63946', edgecolor='black', label='Arabian Sea'),
        Patch(facecolor='#2A9D8F', edgecolor='black', label='Bay of Bengal'),
        Patch(facecolor='#F4A261', edgecolor='black', label='Winter Monsoon'),
        Patch(facecolor='#457B9D', edgecolor='black', label='Summer Monsoon')
    ]
    ax1.legend(handles=legend_elements, loc='best', framealpha=0.95, edgecolor='black')
    
    ax1.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax1.set_title('Latent Space Embedding (t-SNE)', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # PCA plot
    ax2 = plt.subplot(1, 3, 2)
    scatter2 = ax2.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=range(len(latent_2d)), cmap='viridis',
                          s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, (x, y) in enumerate(latent_2d):
        ax2.annotate(f'{i+1}', (x, y), fontsize=8, ha='center', va='center',
                    fontweight='bold', color='white')
    
    ax2.set_xlabel('PC1 (35% variance)', fontweight='bold')
    ax2.set_ylabel('PC2 (22% variance)', fontweight='bold')
    ax2.set_title('PCA Projection', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter2, ax=ax2, label='Sample Index')
    
    # Variance explained
    ax3 = plt.subplot(1, 3, 3)
    n_components = len(pca_variance)
    ax3.bar(range(1, n_components+1), pca_variance,
            color='#457B9D', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Principal Component', fontweight='bold')
    ax3.set_ylabel('Explained Variance Ratio', fontweight='bold')
    ax3.set_title('PCA Variance Explained', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Cumulative line
    ax3_twin = ax3.twinx()
    cumsum = np.cumsum(pca_variance)
    ax3_twin.plot(range(1, n_components+1), cumsum, 'ro-', linewidth=2.5,
                  markersize=8, label='Cumulative')
    ax3_twin.set_ylabel('Cumulative Variance', fontweight='bold', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3_twin.legend(loc='lower right', framealpha=0.95, edgecolor='black')
    ax3_twin.set_ylim([0, 1])
    
    plt.suptitle('Latent Embedding Space Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "04_latent_embeddings.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def plot_depth_wise_metrics(data):
    """Bonus: Depth-wise performance metrics."""
    print("\n[Bonus] Creating depth-wise metrics plot...")
    
    depth_levels = data['depth_levels']
    ground_truth = data['ground_truth']
    prediction = data['prediction']
    
    # Compute metrics for each depth
    rmse_per_depth = []
    mae_per_depth = []
    corr_per_depth = []
    
    for d in range(len(depth_levels)):
        pred_d = prediction[d].flatten()
        target_d = ground_truth[d].flatten()
        
        rmse = np.sqrt(np.mean((pred_d - target_d) ** 2))
        mae = np.mean(np.abs(pred_d - target_d))
        corr = np.corrcoef(pred_d, target_d)[0, 1]
        
        rmse_per_depth.append(rmse)
        mae_per_depth.append(mae)
        corr_per_depth.append(corr)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE
    ax1 = axes[0]
    ax1.plot(rmse_per_depth, depth_levels, 'o-', linewidth=2.5, markersize=10,
             color='#E63946', markeredgecolor='black', markeredgewidth=1.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('RMSE (°C)', fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontweight='bold')
    ax1.set_title('RMSE by Depth', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhspan(40, 150, alpha=0.15, color='orange', zorder=0)
    ax1.text(0.95, 0.5, 'Thermocline', transform=ax1.transAxes,
             rotation=90, fontsize=9, va='center', ha='right', alpha=0.7)
    
    # MAE
    ax2 = axes[1]
    ax2.plot(mae_per_depth, depth_levels, 's-', linewidth=2.5, markersize=10,
             color='#F4A261', markeredgecolor='black', markeredgewidth=1.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('MAE (°C)', fontweight='bold')
    ax2.set_ylabel('Depth (m)', fontweight='bold')
    ax2.set_title('MAE by Depth', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhspan(40, 150, alpha=0.15, color='orange', zorder=0)
    
    # Correlation
    ax3 = axes[2]
    ax3.plot(corr_per_depth, depth_levels, '^-', linewidth=2.5, markersize=10,
             color='#2A9D8F', markeredgecolor='black', markeredgewidth=1.5)
    ax3.invert_yaxis()
    ax3.set_xlabel('Correlation (r)', fontweight='bold')
    ax3.set_ylabel('Depth (m)', fontweight='bold')
    ax3.set_title('Correlation by Depth', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axhspan(40, 150, alpha=0.15, color='orange', zorder=0)
    ax3.set_xlim([0.85, 1.0])
    
    # Add performance bands
    ax3.axvspan(0.95, 1.0, alpha=0.1, color='green', zorder=0)
    ax3.text(0.975, 50, 'Excellent', rotation=90, fontsize=8, va='bottom', ha='center')
    
    plt.suptitle('Depth-Stratified Model Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = ASSETS_DIR / "05_depth_wise_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")


def main():
    """Main execution."""
    try:
        # Generate synthetic data
        data = generate_synthetic_data()
        
        # Generate all visualizations
        plot_spatial_depth_slices(data)
        plot_vertical_profiles(data)
        plot_error_analysis(data)
        plot_latent_embeddings(data)
        plot_depth_wise_metrics(data)
        
        # Summary
        print("\n" + "="*80)
        print(" "*20 + "✓ ASSET GENERATION COMPLETE")
        print("="*80)
        
        assets = sorted(ASSETS_DIR.glob("*.png"))
        print(f"\nGenerated {len(assets)} high-resolution plots (300 DPI)")
        print(f"Output directory: {ASSETS_DIR.absolute()}\n")
        
        print("Generated Assets:")
        for i, asset_file in enumerate(assets, 1):
            file_size = asset_file.stat().st_size / 1024  # KB
            print(f"  {i}. {asset_file.name:<40} ({file_size:.1f} KB)")
        
        print("\n✓ Ready for hackathon pitch deck!")
        print("\nNote: These are demo visualizations using synthetic data.")
        print("For actual model predictions, install dependencies and run:")
        print("  python scripts/generate_submission_assets.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
