"""Simple test of the slab ocean model component."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax_esm import (
    BoundaryFluxes,
    ComponentConfig,
    SlabOceanModel,
)


def test_slab_ocean_response():
    """Test slab ocean response to constant heat flux."""
    
    # Configuration
    nlat, nlon = 32, 64
    
    # Create ocean component
    ocean_config = ComponentConfig(
        name="ocean",
        timestep=3600.0,  # 1 hour
        grid={
            "type": "latlon",
            "nlat": nlat,
            "nlon": nlon,
        },
        params={
            "mixed_layer_depth": 50.0,  # 50m mixed layer
            "relaxation_time": 30.0,     # 30-day relaxation
        },
    )
    ocean = SlabOceanModel(ocean_config)
    
    # Initialize
    rng_key = jax.random.PRNGKey(42)
    state = ocean.initialize(rng_key)
    
    print("Testing slab ocean model...")
    print(f"Mixed layer depth: {ocean.mixed_layer_depth} m")
    print(f"Relaxation time: {ocean.relaxation_time} days")
    print(f"Initial SST range: [{jnp.min(state.prognostic['sst'])-273.15:.1f}, "
          f"{jnp.max(state.prognostic['sst'])-273.15:.1f}]°C")
    
    # Apply constant heat flux
    heat_flux = jnp.ones((nlat, nlon)) * 50.0  # 50 W/m² warming
    
    # Create forcing
    forcing = BoundaryFluxes(
        heat=heat_flux,
        moisture=jnp.zeros((nlat, nlon)),
        momentum_u=jnp.zeros((nlat, nlon)),
        momentum_v=jnp.zeros((nlat, nlon)),
        tracers={},
    )
    
    # Run for 60 days
    n_steps = 60 * 24  # Hourly steps
    sst_history = []
    sst_anom_history = []
    
    for step in range(n_steps):
        state, _ = ocean.step(state, forcing, ocean.timestep)
        
        # Save daily averages
        if step % 24 == 0:
            sst_history.append(jnp.mean(state.prognostic["sst"]))
            sst_anom_history.append(jnp.mean(state.prognostic["sst_anomaly"]))
    
    # Convert to arrays
    sst_history = jnp.array(sst_history)
    sst_anom_history = jnp.array(sst_anom_history)
    days = jnp.arange(len(sst_history))
    
    # Calculate theoretical response
    # dT/dt = Q/(ρ*cp*h) - T_anom/τ
    # For constant Q and τ >> dt: T_anom(t) ≈ Q*τ/(ρ*cp*h) * (1 - exp(-t/τ))
    q_equilibrium = heat_flux[0, 0] * ocean.relaxation_time * 86400.0 / ocean.heat_capacity
    theoretical_anom = q_equilibrium * (1 - jnp.exp(-days / ocean.relaxation_time))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # SST evolution
    ax1.plot(days, sst_history - 273.15, "b-", linewidth=2, label="Modeled SST")
    ax1.axhline(y=sst_history[0] - 273.15, color="k", linestyle="--", 
                alpha=0.5, label="Initial SST")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("SST (°C)")
    ax1.set_title("Slab Ocean Response to 50 W/m² Heating")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SST anomaly
    ax2.plot(days, sst_anom_history, "b-", linewidth=2, label="Modeled anomaly")
    ax2.plot(days, theoretical_anom, "r--", linewidth=2, label="Theoretical")
    ax2.axhline(y=q_equilibrium, color="k", linestyle=":", 
                alpha=0.5, label=f"Equilibrium ({q_equilibrium:.3f}K)")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("SST Anomaly (K)")
    ax2.set_title("SST Anomaly Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("slab_ocean_test.png", dpi=150, bbox_inches="tight")
    
    # Print diagnostics
    print(f"\nAfter {len(sst_history)} days:")
    print(f"  Mean SST: {sst_history[-1] - 273.15:.2f}°C")
    print(f"  SST warming: {sst_history[-1] - sst_history[0]:.3f}K")
    print(f"  SST anomaly: {sst_anom_history[-1]:.3f}K")
    print(f"  Theoretical equilibrium anomaly: {q_equilibrium:.3f}K")
    print(f"  Fraction of equilibrium: {sst_anom_history[-1]/q_equilibrium:.1%}")
    
    # Test with varying heat flux
    print("\n\nTesting with spatially varying heat flux...")
    
    # Reset state
    state = ocean.initialize(rng_key)
    
    # Create latitude-dependent heat flux
    lat = jnp.linspace(-90, 90, nlat)
    heat_flux_varied = 100.0 * jnp.cos(jnp.pi * lat / 180.0)[:, None]
    heat_flux_varied = jnp.broadcast_to(heat_flux_varied, (nlat, nlon))
    
    forcing_varied = BoundaryFluxes(
        heat=heat_flux_varied,
        moisture=jnp.zeros((nlat, nlon)),
        momentum_u=jnp.zeros((nlat, nlon)),
        momentum_v=jnp.zeros((nlat, nlon)),
        tracers={},
    )
    
    # Run for 30 days
    for _ in range(30 * 24):
        state, _ = ocean.step(state, forcing_varied, ocean.timestep)
    
    # Plot spatial pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heat flux pattern
    im1 = ax1.imshow(
        heat_flux_varied,
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        extent=[0, 360, -90, 90],
        vmin=-100,
        vmax=100,
    )
    ax1.set_title("Applied Heat Flux (W/m²)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    plt.colorbar(im1, ax=ax1)
    
    # SST anomaly pattern
    im2 = ax2.imshow(
        state.prognostic["sst_anomaly"],
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        extent=[0, 360, -90, 90],
    )
    ax2.set_title("SST Anomaly after 30 days (K)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig("slab_ocean_spatial_test.png", dpi=150, bbox_inches="tight")
    
    print("\nSpatial test complete!")
    print(f"Results saved to slab_ocean_test.png and slab_ocean_spatial_test.png")


if __name__ == "__main__":
    test_slab_ocean_response()