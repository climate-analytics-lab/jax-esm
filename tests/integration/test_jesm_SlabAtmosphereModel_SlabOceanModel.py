"""Example of coupling of simple slab atmosphere and ocean models."""

def test_integration():

    from jax_esm.components import SlabOceanModel, SlabAtmosphereModel
    import jax_datetime as jdt
    from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn as couple
    from pathlib import Path

    resolution = 31
    grid_specification = f"JCM::T{resolution:d}"

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime("2000-01-01")
    simulation_interval = jdt.to_timedelta(30, "day")
    output_dir = Path("output/SAM_SOM").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Creating components
    components = dict(
        atm=SlabAtmosphereModel(
            grid_specification=grid_specification,
            timestep=3600.0,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
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

    output_dir = Path("output").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)
if __name__ == "__main__":
    test_integration()


