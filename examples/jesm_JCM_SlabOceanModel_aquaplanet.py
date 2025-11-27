"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":
    import jcm
    import jax_datetime as jdt
    from jax_esm.components.JCM import JCM
    from jax_esm.components.SlabOceanModel import SlabOceanModel
    from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn as couple
    from pathlib import Path

    output_dir = Path("output").resolve()
    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Time configuration
    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime("2000-01-01")

    # Creating components
    components = dict(
        atm=JCM(
            model=jcm.model.Model(start_date=start_datetime),
            coupling_timestep=coupling_timestep,
        ),
        ocn = SlabOceanModel(
            grid_specification = "JCM::T31",
            timestep = coupling_timestep,
            start_datetime = start_datetime,
            save_interval = coupling_timestep,
        ),
    )

    # Couple models together
    model = couple(**components)

    # Obtain initial condition
    initial_state = model.initialize()

    print("Add Gaussian noise to break symmetry...")
    import jax

    initial_state["ocn"].prog.T += 0.1 * jax.numpy.array(
        jax.random.normal(jax.random.key(1000), shape=initial_state["ocn"].prog.T.shape)
    )

    # Run coupled model
    print("Running model...")
    final_state, predictions = model.run(
        init_coupled_state = initial_state,
        start_time = 0.0,
        end_time = 86400.0 * 3,
        jax_scan = False,
    )

    # Convert output into xarray
    output_dict = model.predictions_to_xarray(predictions)

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"JESM-JCM_SOM-{component_name:s}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")
