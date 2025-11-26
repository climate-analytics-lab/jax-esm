"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":
    import jax_esm
    import jcm
    import jax_datetime as jdt
    from jax_esm.utils.datetime_tools import days_of_month_in_date
     
    from pathlib import Path

    JCM_grid_specification = "JCM::T31"
    JCM_domain_data_file = (Path(jcm.__file__).parent / "data/bc/terrain_t31.nc").resolve()
    JCM_forcing_data_file = (Path(jcm.__file__).parent / "data/bc/forcing_t31.nc").resolve()
    
    JCM_topography_file = JCM_domain_data_file
    JCM_mask_file = JCM_domain_data_file

    SlabOceanModel_grid_specification = "JCM::T31"
    SlabOceanModel_topography_file = JCM_domain_data_file
    SlabOceanModel_mask_file = JCM_mask_file
    SlabOceanModel_SST_clim_file = JCM_forcing_data_file

    SlabLandModel_grid_specification = "JCM::T31"
    SlabLandModel_topography_file = JCM_domain_data_file
    SlabLandModel_mask_file = JCM_mask_file
    SlabLandModel_land_clim_file = JCM_forcing_data_file

    from jax_esm.components.JCM import JCM
    from jax_esm.components.SlabOceanModel import SlabOceanModel
    from jax_esm.components.SlabLandModel import SlabLandModel
    from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn_lnd as couple
    from jax_esm.components.domain import Domain
    from datetime import datetime
    
    output_dir = Path("output").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime('2000-01-01')
    simulate_months = 1
 
    # Creating components
    components = dict(
        atm=JCM(
            model=jcm.model.Model(start_date=start_datetime),
            coupling_timestep=coupling_timestep,
        ),
        ocn=SlabOceanModel(
            grid_specification=SlabOceanModel_grid_specification,
            timestep=coupling_timestep,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
            relaxation_time=60 * 86400.0,
            SST_clim_file=SlabOceanModel_SST_clim_file,
        ),
        lnd = SlabLandModel(
            grid_specification = "JCM::T31",
            timestep = 3600 * 6,
            start_datetime = start_datetime,
            save_interval = coupling_timestep,
            relaxation_time = 60 * 86400.0,
            topography_file = SlabLandModel_topography_file,
            mask_file = SlabOceanModel_mask_file,
            land_clim_file = SlabLandModel_land_clim_file,
        ),
    )

    # Creating model
    model = couple(**components)

    # Obtain initial condition
    initial_state = model.initialize()

    # Run coupled model
    print("Running model...")
    state_holder = initial_state
    _start_datetime = start_datetime
    for rel_month in range(simulate_months):
       
        _end_datetime = _start_datetime + jdt.to_timedelta(days_of_month_in_date(_start_datetime)*0+5, "day")
        current_year_month = _start_datetime.to_pydatetime().strftime("%Y-%m")
        
        print(f"Running year {current_year_month:s} (relative month {rel_month:d})")
        state_holder, predictions = model.run(
            init_coupled_state=state_holder,
            start_time=(_start_datetime - start_datetime)
            / jdt.to_timedelta(1, "second"),
            end_time=(_end_datetime - start_datetime) / jdt.to_timedelta(1, "second"),
            jax_scan=True,
        )
        # Convert output into xarray
        output_dict = model.predictions_to_xarray(predictions)

        for component_name, ds in output_dict.items():
            output_file = output_dir / f"JESM-JCM_SOM_SLM-{component_name:s}-{current_year_month:s}.nc"
            print("Output file: ", str(output_file))
            ds.to_netcdf(output_file, engine="netcdf4")

        # Set the start time of the next iteration
        _start_datetime = _end_datetime
