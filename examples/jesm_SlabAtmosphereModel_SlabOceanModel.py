"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":

    from pathlib import Path       
    from jax_esm.coupling.couplers.CoupledSlabAtmosphereModelSlabOceanModel import CoupledSlabAtmosphereModelSlabOceanModel
    from datetime import datetime
    
    output_dir = Path("output").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True) 

    # Creating model
    model = CoupledSlabAtmosphereModelSlabOceanModel(
        start_datetime = datetime(year=2001, month=1, day=1),
        atmosphere_grid_specification = "JCM::T31",
        ocean_grid_specification = "JCM::T31",
        coupling_timestep = 86400.0,
        atmosphere_substeps = 24,
        ocean_substeps = 1,
    )
    
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
