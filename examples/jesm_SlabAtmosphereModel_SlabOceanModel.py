"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":

    from pathlib import Path       
    from jax_esm.components.SlabOceanModel import SlabOceanModel
    from jax_esm.components.SlabAtmosphereModel import SlabAtmosphereModel
    from jax_esm.components.base import ComponentConfig
    from jax_esm.components.domain import Domain
    from datetime import datetime

    from jax_esm.coupling.coupled_model_generator.coupled_SlabAtmosphereModel_SlabOceanModel import couple_SlabAtmosphereModel_and_SlabOceanModel
    
    output_dir = Path("output").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True) 

    coupling_timestep = 86400.0
    start_datetime = datetime(2025, 1, 1)

    atmosphere_topography_file = None
    atmosphere_mask_file = None
    ocean_topography_file = None
    ocean_mask_file = None

    # Creating model
    components = dict(
        atm = SlabAtmosphereModel(ComponentConfig(
            name="atmosphere",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = 24,
            save_interval = coupling_timestep,
            domain = Domain.from_grid_specification(
                "JCM::T31",
                topography_file = atmosphere_topography_file,
                mask_file = atmosphere_mask_file,
            ),
            params = dict(),
        )),
        ocn = SlabOceanModel(ComponentConfig(
            name="ocean",
            timestep = coupling_timestep,
            start_dt = start_datetime,
            substeps = 1,
            save_interval = coupling_timestep,
            domain =  Domain.from_grid_specification(
                "Veros::4deg",
                topography_file = ocean_topography_file,
                mask_file = ocean_mask_file,
            ),
            params=dict(
                relaxation_time = 60 * 86400.0,
                SST_clim_file = ocean_topography_file,
            ),
        )),
    )

    model = couple_SlabAtmosphereModel_and_SlabOceanModel(**components) 
    # Obtain initial condition
    initial_state = model.initialize()
    # Run coupled model 
    final_state, predictions = model.run(
        init_cplstate = initial_state,
        start_time = 0.0,
        end_time = 86400.0 * 5,
        jax_scan = False,
    )

    # Convert output into xarray
    output_dict = model.predictions_to_xarray(predictions)

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"JESM-SAM_SOM-{component_name:s}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")
