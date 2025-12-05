"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":
   

    from jax_esm.utils.datetime_tools import days_of_month_in_date
    from jax_esm.components import JCM, SlabOceanModel
    from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn as couple
    import jcm 
    import jax_datetime as jdt
    from pathlib import Path
 
    resolution = 31
    grid_specification = f"JCM::T{resolution:d}"

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime('2000-01-01')
    simulation_interval = jdt.to_timedelta(30, "day")
    output_dir = Path("output/JCM_SOM_aqua").resolve()
    
    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Creating components
    components = dict(
        atm=JCM(
            model=jcm.model.Model(start_date=start_datetime),
            coupling_timestep=coupling_timestep,
        ),
        ocn=SlabOceanModel(
            grid_specification=grid_specification,
            timestep=coupling_timestep,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
            relaxation_time=60 * 86400.0,
        ),
    )

    # Creating model
    model = couple(**components)

    # Obtain initial condition
    initial_state = model.initialize()

    # Run coupled model
    print("Running model...")
    state_holder, predictions = model.run(
        init_coupled_state=initial_state,
        start_time=0,
        end_time=simulation_interval / jdt.to_timedelta(1, "second"),
        jax_scan=True,
    )
    # Convert output into xarray
    output_dict = model.predictions_to_xarray(predictions)

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")

