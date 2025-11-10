"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":

    import jax_esm, jcm
    import jax_datetime as jdt 
    from pathlib import Path
 
    JCM_grid_specification = "JCM::T31"
    JCM_topography_file = (Path(jcm.__file__).parent / "data/bc/boundaries_daily_t31.nc").resolve()
    JCM_mask_file = JCM_topography_file

    SlabOceanModel_grid_specification = "Veros::4deg"
    SlabOceanModel_topography_file = None

    if SlabOceanModel_grid_specification[:5] == "JCM::": 
        SlabOceanModel_mask_file = JCM_mask_file
        SlabOceanModel_SST_clim_file = JCM_topography_file

    elif SlabOceanModel_grid_specification[:7] == "Veros::":
        SlabOceanModel_mask_file = (Path(jax_esm.__file__).parent / "data/veros/veros_4deg.nc").resolve()
        SlabOceanModel_SST_clim_file = None

    from jax_esm.components.JCM import JCM
    from jax_esm.components.SlabOceanModel import SlabOceanModel
    from jax_esm.coupling.factory.simple_coupling import couple_atm_ocn as couple
    from jax_esm.components.domain import Domain
    from datetime import datetime
    
    output_dir = Path("output").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True) 

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime('2000-01-01')
    
    # Creating model
    components = dict(
        atm = JCM(
            model = jcm.model.Model(time_step=3600.0 / 60), # in minutes
            save_interval = 3600.0,
        ),
        ocn = SlabOceanModel(
            timestep = coupling_timestep,
            start_dt = start_datetime,
            save_interval = coupling_timestep,
            domain =  Domain.from_grid_specification(
                "Veros::4deg",
                topography_file = SlabOceanModel_topography_file,
                mask_file = SlabOceanModel_mask_file,
            ),
            relaxation_time = 60 * 86400.0,
            SST_clim_file = SlabOceanModel_SST_clim_file,
        ),
    )

    # Creating model
    model = couple(**components) 
    
    # Obtain initial condition
    initial_state = model.initialize()

    # Run coupled model 
    final_state, predictions = model.run(
        init_coupled_state = initial_state,
        start_time = 0.0,
        end_time = 86400.0 * 5,
        jax_scan = True,
    )

    # Convert output into xarray
    output_dict = model.predictions_to_xarray(predictions)
   
    

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"JESM-JCM_SOM-{component_name:s}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")
