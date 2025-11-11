"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":

    from pathlib import Path
    from jax_esm.components.SlabOceanModel import SlabOceanModel
    from jax_esm.components.SlabAtmosphereModel import SlabAtmosphereModel
    from jax_esm.components.base import CoupledComponentConfig
    from jax_esm.components.domain import Domain
    import jax_datetime as jdt

    from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn as couple

    import jax.numpy as jnp
    import jcm  
    
    output_dir = Path("output").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True) 

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime("2025-01-01")

    domain_data_file = (Path(jcm.__file__).parent / "data/bc/boundaries_daily_t31.nc").resolve()

    atmosphere_topography_file = domain_data_file
    atmosphere_mask_file = domain_data_file
    ocean_topography_file = domain_data_file
    ocean_mask_file = domain_data_file

    # Creating component
    components = dict(
        atm = SlabAtmosphereModel(
            grid_specification = "JCM::T31",
            timestep = 3600.0,
            start_datetime = start_datetime,
            save_interval = coupling_timestep,
            topography_file = atmosphere_topography_file,
            mask_file = atmosphere_mask_file,
        ),
        ocn = SlabOceanModel(
            grid_specification = "JCM::T31",
            timestep = 3600 * 6,
            start_datetime = start_datetime,
            save_interval = coupling_timestep,
            relaxation_time = 60 * 86400.0,
            topography_file = ocean_topography_file,
            mask_file = ocean_mask_file,
            SST_clim_file = ocean_topography_file,
        ),
    )

    # Creating model
    model = couple(**components) 
    # Obtain initial condition
    initial_state = model.initialize()

    # Run coupled model 
    print("Running model...")
    final_state, predictions = model.run(
        init_coupled_state = initial_state,
        start_time = 0.0,
        end_time = 86400.0 * 420,
        jax_scan = True,
    )

    # Convert output into xarray
    output_dict = model.predictions_to_xarray(predictions)

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"JESM-SAM_SOM-{component_name:s}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")
