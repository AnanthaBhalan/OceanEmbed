"""
Mock Ocean Data Generator
Generates realistic synthetic NetCDF files for North Indian Ocean with physical oceanography patterns.
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List
import yaml


class MockOceanDataGenerator:
    """Generate realistic synthetic ocean data for testing and development."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the mock data generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.domain = self.config['domain']
        self.depth_levels = np.array(self.config['depth_levels'])
        
        # Create coordinate grids
        self.lat = np.linspace(
            self.domain['lat_min'],
            self.domain['lat_max'],
            self.domain['grid_shape'][0]
        )
        self.lon = np.linspace(
            self.domain['lon_min'],
            self.domain['lon_max'],
            self.domain['grid_shape'][1]
        )
        
        # Create 2D meshgrid
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)
        
        # Define realistic bathymetry (simplified)
        self.bathymetry = self._generate_bathymetry()
        self.land_mask = self.bathymetry < 10  # Land pixels
        
    def _generate_bathymetry(self) -> np.ndarray:
        """Generate realistic bathymetry for North Indian Ocean."""
        # Simplified bathymetry: deeper in center, shallower near coasts
        lat_factor = (self.lat_grid - self.domain['lat_min']) / (self.domain['lat_max'] - self.domain['lat_min'])
        lon_factor = (self.lon_grid - self.domain['lon_min']) / (self.domain['lon_max'] - self.domain['lon_min'])
        
        # Bay of Bengal: relatively shallow in north
        bay_of_bengal_mask = (self.lon_grid > 80) & (self.lon_grid < 100) & (self.lat_grid < 23)
        
        # Arabian Sea: deeper with upwelling zones
        arabian_sea_mask = (self.lon_grid > 50) & (self.lon_grid < 78)
        
        # Base depth pattern
        depth = 3000 + 1000 * np.sin(lat_factor * np.pi) * np.sin(lon_factor * np.pi)
        
        # Shallow regions
        depth[bay_of_bengal_mask] *= 0.6
        
        # Coastal shelves
        coastal_north = self.lat_grid > 25
        depth[coastal_north] *= 0.4
        
        # Land regions (India, Sri Lanka, etc.)
        # Simplified land mask
        india_mask = (
            ((self.lon_grid > 68) & (self.lon_grid < 85) & (self.lat_grid > 8) & (self.lat_grid < 25)) |
            ((self.lon_grid > 75) & (self.lon_grid < 80) & (self.lat_grid > 6) & (self.lat_grid < 10))
        )
        depth[india_mask] = 0
        
        return np.maximum(depth, 0)
    
    def _generate_sst(self, time_idx: int) -> np.ndarray:
        """Generate realistic Sea Surface Temperature with seasonal cycle."""
        # Base temperature with latitude gradient
        sst = 28 + 2 * np.sin((self.lat_grid - 10) / 25 * np.pi)
        
        # Seasonal cycle (monsoon effects)
        seasonal_phase = (time_idx % 365) / 365.0 * 2 * np.pi
        seasonal_amplitude = 3.0
        sst += seasonal_amplitude * np.cos(seasonal_phase - np.pi/2)
        
        # Bay of Bengal warm pool
        bay_mask = (self.lon_grid > 85) & (self.lon_grid < 95) & (self.lat_grid < 20)
        sst[bay_mask] += 1.5
        
        # Arabian Sea upwelling (cooler in summer)
        upwelling_mask = (self.lon_grid > 60) & (self.lon_grid < 70) & (self.lat_grid < 15)
        sst[upwelling_mask] -= 2.0 * np.cos(seasonal_phase)
        
        # Add spatial variability (eddies, fronts)
        noise = np.random.randn(*sst.shape) * 0.5
        from scipy.ndimage import gaussian_filter
        sst += gaussian_filter(noise, sigma=2)
        
        # Mask land
        sst[self.land_mask] = np.nan
        
        return sst.astype(np.float32)
    
    def _generate_sss(self, time_idx: int) -> np.ndarray:
        """Generate realistic Sea Surface Salinity."""
        # Base salinity
        sss = 34.5 + 0.5 * (self.lat_grid - 15) / 10
        
        # Bay of Bengal: fresher due to river discharge
        bay_mask = (self.lon_grid > 80) & (self.lon_grid < 100) & (self.lat_grid < 23)
        sss[bay_mask] -= 2.5
        
        # Arabian Sea: saltier
        arabian_mask = (self.lon_grid > 55) & (self.lon_grid < 75)
        sss[arabian_mask] += 0.8
        
        # Seasonal monsoon freshening
        seasonal_phase = (time_idx % 365) / 365.0 * 2 * np.pi
        sss += 0.5 * np.sin(seasonal_phase) * (self.lat_grid < 18)
        
        # Add noise
        noise = np.random.randn(*sss.shape) * 0.1
        from scipy.ndimage import gaussian_filter
        sss += gaussian_filter(noise, sigma=3)
        
        sss[self.land_mask] = np.nan
        return sss.astype(np.float32)
    
    def _generate_ssh(self, time_idx: int) -> np.ndarray:
        """Generate Sea Surface Height anomaly."""
        # Base SSH with geostrophic patterns
        ssh = 0.1 * np.sin(self.lat_grid / 10) * np.cos(self.lon_grid / 20)
        
        # Seasonal thermosteric expansion
        seasonal_phase = (time_idx % 365) / 365.0 * 2 * np.pi
        ssh += 0.15 * np.cos(seasonal_phase)
        
        # Mesoscale eddies
        num_eddies = 5
        for _ in range(num_eddies):
            eddy_lat = np.random.uniform(self.domain['lat_min'] + 2, self.domain['lat_max'] - 2)
            eddy_lon = np.random.uniform(self.domain['lon_min'] + 5, self.domain['lon_max'] - 5)
            eddy_strength = np.random.uniform(-0.3, 0.3)
            eddy_radius = np.random.uniform(2, 5)
            
            dist = np.sqrt((self.lat_grid - eddy_lat)**2 + (self.lon_grid - eddy_lon)**2)
            ssh += eddy_strength * np.exp(-dist**2 / (2 * eddy_radius**2))
        
        ssh[self.land_mask] = np.nan
        return ssh.astype(np.float32)
    
    def _generate_currents(self, ssh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate surface ocean currents from SSH (geostrophic approximation)."""
        # Compute geostrophic currents from SSH gradients
        f = 2 * 7.2921e-5 * np.sin(np.deg2rad(self.lat_grid))  # Coriolis parameter
        f[np.abs(f) < 1e-10] = 1e-10  # Avoid division by zero near equator
        
        g = 9.81  # gravity
        
        # Compute gradients (finite differences)
        dy = np.gradient(self.lat)[1] * 111000  # meters (scalar, uniform lat spacing)
        dx = np.gradient(self.lon)[1] * 111000 * np.cos(np.deg2rad(self.lat))  # (lat,) meters
        
        dssh_dy, dssh_dx = np.gradient(ssh)
        dssh_dy /= dy
        dssh_dx /= dx[:, np.newaxis]
        
        # Geostrophic currents (v = g/f * dssh/dx, u = -g/f * dssh/dy)
        u_curr = -g / f * dssh_dy
        v_curr = g / f * dssh_dx
        
        # Clip unrealistic values
        u_curr = np.clip(u_curr, -2, 2)
        v_curr = np.clip(v_curr, -2, 2)
        
        # Add noise
        u_curr += np.random.randn(*u_curr.shape) * 0.05
        v_curr += np.random.randn(*v_curr.shape) * 0.05
        
        u_curr[self.land_mask] = np.nan
        v_curr[self.land_mask] = np.nan
        
        return u_curr.astype(np.float32), v_curr.astype(np.float32)
    
    def _generate_winds(self, time_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate surface wind components with monsoon patterns."""
        seasonal_phase = (time_idx % 365) / 365.0 * 2 * np.pi
        
        # Summer monsoon (SW winds)
        summer_strength = np.maximum(0, np.cos(seasonal_phase))
        u_wind_summer = -3.0 * summer_strength
        v_wind_summer = 5.0 * summer_strength
        
        # Winter monsoon (NE winds)
        winter_strength = np.maximum(0, -np.cos(seasonal_phase))
        u_wind_winter = 2.0 * winter_strength
        v_wind_winter = -4.0 * winter_strength
        
        u_wind = u_wind_summer + u_wind_winter
        v_wind = v_wind_summer + v_wind_winter
        
        # Spatial variability
        u_wind = u_wind * (1 + 0.3 * np.sin(self.lat_grid / 5))
        v_wind = v_wind * (1 + 0.2 * np.cos(self.lon_grid / 10))
        
        # Add turbulent noise
        u_wind += np.random.randn(*u_wind.shape) * 1.0
        v_wind += np.random.randn(*v_wind.shape) * 1.0
        
        u_wind[self.land_mask] = np.nan
        v_wind[self.land_mask] = np.nan
        
        return u_wind.astype(np.float32), v_wind.astype(np.float32)
    
    def _generate_temperature_profile(self, sst: np.ndarray, time_idx: int) -> np.ndarray:
        """Generate realistic 3D temperature field with thermocline."""
        n_lat, n_lon = sst.shape
        n_depths = len(self.depth_levels)
        
        temp_3d = np.zeros((n_depths, n_lat, n_lon), dtype=np.float32)
        
        # Surface temperature
        temp_3d[0] = sst
        
        # Mixed layer depth (seasonal variation)
        seasonal_phase = (time_idx % 365) / 365.0 * 2 * np.pi
        mld = 40 + 30 * np.cos(seasonal_phase)  # 10-70m seasonal cycle
        
        # Thermocline parameters
        thermocline_depth = 100  # meters
        thermocline_strength = 15  # °C temperature drop
        
        # Deep ocean temperature (latitude dependent)
        deep_temp = 5 + 2 * (self.lat_grid - 15) / 15
        
        for i, depth in enumerate(self.depth_levels):
            if depth == 0:
                continue
            
            # Mixed layer: isothermal
            if depth <= mld:
                temp_3d[i] = sst - 0.1 * depth / mld
            else:
                # Thermocline: exponential decay
                z_norm = (depth - mld) / thermocline_depth
                temp_3d[i] = sst - thermocline_strength * (1 - np.exp(-z_norm))
                
                # Approach deep temperature at great depths
                if depth > 300:
                    deep_factor = 1 - np.exp(-(depth - 300) / 500)
                    temp_3d[i] = temp_3d[i] * (1 - deep_factor) + deep_temp * deep_factor
        
            # Ensure no temperature inversion (physically consistent)
            if i > 0:
                temp_3d[i] = np.minimum(temp_3d[i], temp_3d[i-1])
            
            # Add small-scale variability (internal waves, eddies)
            noise = np.random.randn(n_lat, n_lon) * (0.5 * np.exp(-depth / 200))
            from scipy.ndimage import gaussian_filter
            temp_3d[i] += gaussian_filter(noise, sigma=2)
            
            # Mask land and shallow regions
            shallow_mask = self.bathymetry < depth
            temp_3d[i][shallow_mask] = np.nan
        
        return temp_3d
    
    def generate_sample(self, time_idx: int = 0) -> Tuple[xr.Dataset, xr.Dataset]:
        """Generate a single sample of surface inputs and subsurface targets.
        
        Returns:
            surface_data: xarray Dataset with 8 surface channels
            subsurface_data: xarray Dataset with 15-depth temperature profile
        """
        # Generate surface variables
        sst = self._generate_sst(time_idx)
        sss = self._generate_sss(time_idx)
        ssh = self._generate_ssh(time_idx)
        u_curr, v_curr = self._generate_currents(ssh)
        u_wind, v_wind = self._generate_winds(time_idx)
        
        # Generate subsurface temperature
        temp_profile = self._generate_temperature_profile(sst, time_idx)
        
        # Create time coordinate
        base_date = datetime(2020, 1, 1)
        time = base_date + timedelta(days=time_idx)
        
        # Create surface dataset
        surface_data = xr.Dataset(
            {
                'SST': (['lat', 'lon'], sst),
                'SSS': (['lat', 'lon'], sss),
                'SSH': (['lat', 'lon'], ssh),
                'U_curr': (['lat', 'lon'], u_curr),
                'V_curr': (['lat', 'lon'], v_curr),
                'U_wind': (['lat', 'lon'], u_wind),
                'V_wind': (['lat', 'lon'], v_wind),
                'mask': (['lat', 'lon'], (~self.land_mask).astype(np.float32)),
                'bathymetry': (['lat', 'lon'], self.bathymetry.astype(np.float32))
            },
            coords={
                'lat': self.lat,
                'lon': self.lon,
                'time': time
            },
            attrs={
                'title': 'Mock North Indian Ocean Surface Data',
                'institution': 'OceanEmbed',
                'source': 'Synthetic data generator',
                'Conventions': 'CF-1.7'
            }
        )
        
        # Create subsurface dataset
        subsurface_data = xr.Dataset(
            {
                'temperature': (['depth', 'lat', 'lon'], temp_profile)
            },
            coords={
                'depth': self.depth_levels,
                'lat': self.lat,
                'lon': self.lon,
                'time': time
            },
            attrs={
                'title': 'Mock North Indian Ocean Subsurface Temperature',
                'institution': 'OceanEmbed',
                'source': 'Synthetic data generator',
                'Conventions': 'CF-1.7'
            }
        )
        
        return surface_data, subsurface_data
    
    def generate_dataset(self, num_samples: int, output_dir: str = "mock_data"):
        """Generate multiple samples and save to NetCDF files.
        
        Args:
            num_samples: Number of daily samples to generate
            output_dir: Directory to save NetCDF files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        surface_dir = output_path / "surface"
        subsurface_dir = output_path / "subsurface"
        surface_dir.mkdir(exist_ok=True)
        subsurface_dir.mkdir(exist_ok=True)
        
        print(f"Generating {num_samples} samples of mock ocean data...")
        
        for i in range(num_samples):
            surface_data, subsurface_data = self.generate_sample(time_idx=i)
            
            # Save to NetCDF
            date_str = surface_data.time.dt.strftime('%Y%m%d').values
            surface_file = surface_dir / f"surface_{date_str}.nc"
            subsurface_file = subsurface_dir / f"subsurface_{date_str}.nc"
            
            surface_data.to_netcdf(surface_file)
            subsurface_data.to_netcdf(subsurface_file)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        print(f"Dataset generation complete. Files saved to {output_path}")
        
        # Create a summary file
        summary = {
            'num_samples': num_samples,
            'domain': self.domain,
            'depth_levels': self.depth_levels.tolist(),
            'surface_channels': 8,
            'grid_shape': self.domain['grid_shape']
        }
        
        import json
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Generate sample dataset
    generator = MockOceanDataGenerator("config.yaml")
    generator.generate_dataset(num_samples=50, output_dir="mock_data")
