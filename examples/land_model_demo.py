"""Example demonstrating LandModel component usage.

This example shows:
1. Standalone land model simulation
2. Land model configuration options
3. Integration with other components (conceptual)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr

from jax_esm.components.LandModel import LandModel
from jax_esm.components.base import ComponentConfig


class MockCoordinates:
    """Mock coordinate system for demonstration."""
    
    def __init__(self, nlat=32, nlon=64, nlevels=3):
        self.nodal_shape = (nlevels, nlat, nlon)
        self.nlat = nlat
        self.nlon = nlon
        
        class HorizontalCoords:
            def __init__(self, nlat, nlon):
                self.latitudes = jnp.linspace(-jnp.pi/2, jnp.pi/2, nlat)
                self.longitudes = jnp.linspace(0, 2*jnp.pi, nlon)
        
        self.horizontal = HorizontalCoords(nlat, nlon)


class MockBoundaries:
    """Mock boundary data with land/ocean mask."""
    
    def __init__(self, nlat=32, nlon=64):
        # Create realistic land mask (continents)
        lon_grid, lat_grid = jnp.meshgrid(
            jnp.linspace(0, 360, nlon),
            jnp.linspace(-90, 90, nlat),
        )
        
        # Simple continental shapes
        land = jnp.zeros((nlat, nlon), dtype=jnp.float32)
        
        # "North America" - roughly
        land = jnp.where(
            (lon_grid > 200) & (lon_grid < 300) & (lat_grid > 15) & (lat_grid < 70),
            0.9,
            land,
        )
        
        # "Europe/Asia"
        land = jnp.where(
            (lon_grid > 0) & (lon_grid < 180) & (lat_grid > 35) & (lat_grid < 70),
            0.8,
            land,
        )
        
        # "Africa"
        land = jnp.where(
            (lon_grid > 0) & (lon_grid < 50) & (lat_grid > -35) & (lat_grid < 35),
            0.85,
            land,
        )
        
        self.fmask = land
        
        # Albedo (high at poles = ice, low elsewhere = soil/vegetation)
        alb = jnp.ones((nlat, nlon), dtype=jnp.float32) * 0.2
        alb = jnp.where(jnp.abs(lat_grid) > 70, 0.6, alb)  # Ice sheets at poles
        self.alb0 = alb


def create_example_climatology(nlat=32, nlon=64, save_path="land_climatology.nc"):
    """Create example land climatology file.
    
    Args:
        nlat: Number of latitude points
        nlon: Number of longitude points
        save_path: Path to save NetCDF file
        
    Returns:
        Path to created file
    """
    
    months = 12
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    
    # Land surface temperature (K)
    stl = np.zeros((months, nlat, nlon), dtype=np.float32)
    for m in range(months):
        # Base temperature gradient
        base_T = 273.15 + 30.0 * np.cos(np.deg2rad(lat))
        
        # Seasonal cycle (stronger in mid-latitudes)
        seasonal_amp = 20.0 * np.sin(np.abs(np.deg2rad(lat)))**2
        seasonal_phase = 2 * np.pi * (m - 2) / 12.0
        seasonal = seasonal_amp * np.cos(seasonal_phase)
        
        stl[m, :, :] = (base_T + seasonal)[:, None]
    
    # Snow depth (mm water equivalent)
    snowd = np.zeros((months, nlat, nlon), dtype=np.float32)
    for m in range(months):
        # Winter snow at high latitudes
        winter_nh = np.maximum(0, np.cos(2 * np.pi * (m - 2) / 12.0))  # NH winter
        winter_sh = np.maximum(0, np.cos(2 * np.pi * (m - 8) / 12.0))  # SH winter
        
        # Northern hemisphere
        snow_nh = np.where(lat > 50, 150.0 * winter_nh, 0.0)
        # Southern hemisphere
        snow_sh = np.where(lat < -50, 150.0 * winter_sh, 0.0)
        
        snowd[m, :, :] = (snow_nh + snow_sh)[:, None]
    
    # Soil water availability (0-1)
    soilw = np.ones((months, nlat, nlon), dtype=np.float32) * 0.5
    for m in range(months):
        # Wetter in tropics, drier in subtropics
        moisture = 0.3 + 0.4 * np.cos(2 * np.deg2rad(lat))**2
        # Seasonal variation
        seasonal_moisture = 0.2 * np.cos(2 * np.pi * (m - 2) / 12.0)
        soilw[m, :, :] = np.clip(moisture + seasonal_moisture, 0.1, 0.9)[:, None]
    
    # Create dataset
    ds = xr.Dataset(
        {
            "stl": (["month", "lat", "lon"], stl, {
                "long_name": "Land surface temperature",
                "units": "K",
            }),
            "snowd": (["month", "lat", "lon"], snowd, {
                "long_name": "Snow depth (water equivalent)",
                "units": "mm",
            }),
            "soilw": (["month", "lat", "lon"], soilw, {
                "long_name": "Soil water availability",
                "units": "1",
            }),
        },
        coords={
            "month": np.arange(1, 13),
            "lat": lat,
            "lon": lon,
        },
    )
    
    ds.to_netcdf(save_path)
    print(f"Created climatology file: {save_path}")
    
    return save_path


def run_standalone_land_model():
    """Run standalone land model simulation."""
    
    print("\n" + "="*70)
    print("STANDALONE LAND MODEL SIMULATION")
    print("="*70 + "\n")
    
    # Configuration
    nlat, nlon = 32, 64
    
    # Create components
    coords = MockCoordinates(nlat=nlat, nlon=nlon)
    boundaries = MockBoundaries(nlat=nlat, nlon=nlon)
    
    # Create climatology file
    clim_file = create_example_climatology(nlat=nlat, nlon=nlon)
    
    # Configure land model
    config = ComponentConfig(
        name="land",
        start_dt=pd.Timestamp("2001-01-01"),
        timestep=86400.0,  # 1 day
        substeps=1,
        save_interval=86400.0,
        grid={"nlat": nlat, "nlon": nlon},
        params={
            "coords": coords,
            "boundaries": boundaries,
            "boundary_file": clim_file,
            "depth_soil": 1.0,  # 1m soil depth
            "depth_lice": 5.0,  # 5m ice depth
            "tdland": 40.0,  # 40-day relaxation timescale
        },
    )
    
    print(f"Configuration:")
    print(f"  Grid: {nlat} x {nlon}")
    print(f"  Timestep: {config.timestep/86400:.1f} days")
    print(f"  Soil depth: {config.params['depth_soil']} m")
    print(f"  Dissipation timescale: {config.params['tdland']} days")
    print()
    
    # Initialize land model
    land = LandModel(config)
    state = land.initialize()
    
    print("Initial state:")
    print(f"  Mean land temperature: {jnp.mean(state.prog.T):.2f} K ({jnp.mean(state.prog.T)-273.15:.2f}°C)")
    print(f"  Temperature range: [{jnp.min(state.prog.T)-273.15:.1f}, {jnp.max(state.prog.T)-273.15:.1f}]°C")
    print(f"  Mean snow depth: {jnp.mean(state.phydata.snowd):.1f} mm")
    print(f"  Mean soil moisture: {jnp.mean(state.phydata.soilw):.3f}")
    print()
    
    # Run simulation
    days_to_run = 365
    print(f"Running {days_to_run}-day simulation...")
    
    # Create mock coupled state
    class MockCoupledState:
        def __init__(self, land_state):
            self.lnd = land_state
    
    cpl = MockCoupledState(state)
    step_fn = land.gen_step_fn()
    
    # Store history
    T_history = []
    snowd_history = []
    soilw_history = []
    time_history = []
    
    # Time stepping
    for day in range(days_to_run):
        if day % 30 == 0:
            print(f"  Day {day}/{days_to_run}...")
        
        new_state, predictions = step_fn(cpl, day)
        cpl = MockCoupledState(new_state)
        
        # Save every 7 days
        if day % 7 == 0:
            T_history.append(np.array(new_state.prog.T))
            snowd_history.append(np.array(new_state.phydata.snowd))
            soilw_history.append(np.array(new_state.phydata.soilw))
            time_history.append(new_state.prog.sim_time / 86400.0)  # days
    
    print(f"\nSimulation complete!")
    
    # Convert to arrays
    T_history = np.array(T_history)
    snowd_history = np.array(snowd_history)
    soilw_history = np.array(soilw_history)
    time_history = np.array(time_history)
    
    print(f"\nFinal state:")
    print(f"  Mean land temperature: {jnp.mean(new_state.prog.T):.2f} K ({jnp.mean(new_state.prog.T)-273.15:.2f}°C)")
    print(f"  Temperature range: [{jnp.min(new_state.prog.T)-273.15:.1f}, {jnp.max(new_state.prog.T)-273.15:.1f}]°C")
    
    # Plotting
    print("\nGenerating plots...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Final land temperature
    ax1 = plt.subplot(2, 3, 1)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon)
    im1 = ax1.contourf(lon, lat, T_history[-1, :, :] - 273.15, levels=20, cmap='RdBu_r')
    ax1.set_title('Final Land Surface Temperature (°C)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Final snow depth
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.contourf(lon, lat, snowd_history[-1, :, :], levels=20, cmap='Blues')
    ax2.set_title('Final Snow Depth (mm)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Final soil moisture
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.contourf(lon, lat, soilw_history[-1, :, :], levels=20, cmap='YlGnBu')
    ax3.set_title('Final Soil Water Availability')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    plt.colorbar(im3, ax=ax3)
    
    # Plot 4: Time series of global mean temperature
    ax4 = plt.subplot(2, 3, 4)
    global_T = np.mean(T_history, axis=(1, 2)) - 273.15
    ax4.plot(time_history, global_T, 'b-', linewidth=2)
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Global Mean Land Temperature (°C)')
    ax4.set_title('Temperature Evolution')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Zonal mean temperature
    ax5 = plt.subplot(2, 3, 5)
    zonal_T = np.mean(T_history, axis=2) - 273.15
    im5 = ax5.contourf(time_history, lat, zonal_T.T, levels=20, cmap='RdBu_r')
    ax5.set_xlabel('Time (days)')
    ax5.set_ylabel('Latitude')
    ax5.set_title('Zonal Mean Temperature Evolution (°C)')
    plt.colorbar(im5, ax=ax5)
    
    # Plot 6: Seasonal cycle at mid-latitude
    ax6 = plt.subplot(2, 3, 6)
    mid_lat_idx = nlat * 2 // 3  # Northern mid-latitude
    mid_lon_idx = nlon // 2
    point_T = T_history[:, mid_lat_idx, mid_lon_idx] - 273.15
    ax6.plot(time_history, point_T, 'r-', linewidth=2)
    ax6.set_xlabel('Time (days)')
    ax6.set_ylabel('Temperature (°C)')
    ax6.set_title(f'Temperature at ({lat[mid_lat_idx]:.0f}°N, {lon[mid_lon_idx]:.0f}°E)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('land_model_demo_results.png', dpi=150, bbox_inches='tight')
    print("Saved results to: land_model_demo_results.png")
    
    # Create xarray output
    ds = land.predictions_to_xarray({
        "prog": new_state.prog,
        "phydata": new_state.phydata,
    })
    
    print("\nOutput dataset structure:")
    print(ds)
    
    return land, new_state, T_history


def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    
    print("\n" + "="*70)
    print("CONFIGURATION OPTIONS DEMONSTRATION")
    print("="*70 + "\n")
    
    coords = MockCoordinates(nlat=16, nlon=32)
    
    configs = [
        {
            "name": "Fast relaxation",
            "tdland": 10.0,
            "description": "10-day relaxation to climatology",
        },
        {
            "name": "Slow relaxation",
            "tdland": 100.0,
            "description": "100-day relaxation to climatology",
        },
        {
            "name": "Deep soil",
            "depth_soil": 2.0,
            "tdland": 40.0,
            "description": "2m soil depth (higher heat capacity)",
        },
    ]
    
    for cfg in configs:
        print(f"\n{cfg['name']}: {cfg['description']}")
        
        config = ComponentConfig(
            name="land",
            start_dt=pd.Timestamp("2001-01-01"),
            timestep=86400.0,
            substeps=1,
            save_interval=86400.0,
            grid={"nlat": 16, "nlon": 32},
            params={
                "coords": coords,
                "depth_soil": cfg.get("depth_soil", 1.0),
                "tdland": cfg.get("tdland", 40.0),
            },
        )
        
        land = LandModel(config)
        print(f"  Heat capacity (soil): {land.hcapl:.2e} J/(m^2 K)")
        print(f"  Relaxation timescale: {land.tdland} days")


if __name__ == "__main__":
    # Run standalone simulation
    land, final_state, history = run_standalone_land_model()
    
    # Demonstrate configuration options
    demonstrate_configuration_options()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe LandModel component can be integrated into coupled models by:")
    print("  1. Adding it to a Coupler alongside atmosphere and ocean components")
    print("  2. Exchanging heat fluxes through the flux component")
    print("  3. Using land surface temperature for atmospheric boundary conditions")
    print("\nSee jax_esm/coupling/coupler.py for integration patterns.")
