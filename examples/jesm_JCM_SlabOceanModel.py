"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":

    import jax_esm, jcm 
    from pathlib import Path
 
    
    JCM_grid_specification = "JCM::T31"
    JCM_topography_file = (Path(jcm.__file__).parent / "data/bc/boundaries_daily_t31.nc").resolve()
    JCM_mask_file = JCM_topography_file

    SlabOceanModel_grid_specification = "JCM::T31"
    SlabOceanModel_topography_file = None

    if SlabOceanModel_grid_specification[:5] == "JCM::": 
        SlabOceanModel_mask_file = JCM_mask_file
        SlabOceanModel_SST_clim_file = JCM_topography_file

    elif SlabOceanModel_grid_specification[:7] == "Veros::":
        SlabOceanModel_mask_file = (Path(jax_esm.__file__).parent / "data/veros/veros_4deg.nc").resolve()
        SlabOceanModel_SST_clim_file = None


    from jax_esm.coupling.couplers.CoupledJCMSlabOceanModel import CoupledJCMSlabOceanModel
    from datetime import datetime
    
    output_dir = Path("output").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True) 

    # Creating model
    model = CoupledJCMSlabOceanModel(
        JCM_topography_file = JCM_topography_file,
        JCM_mask_file = JCM_mask_file,
        SlabOceanModel_topography_file = SlabOceanModel_topography_file,
        SlabOceanModel_mask_file = SlabOceanModel_mask_file,
        SlabOceanModel_SST_clim_file = SlabOceanModel_SST_clim_file,
        JCM_grid_specification = JCM_grid_specification,
        SlabOceanModel_grid_specification = SlabOceanModel_grid_specification,

        JCM_layers = 8,
        coupling_timestep = 86400.0,
        JCM_substeps = 48,
        SlabOceanModel_substeps = 1,
        start_datetime = datetime(year=2001, month=1, day=1),
    )
    
    # Obtain initial condition
    initial_state = model.initialize()

    # Run coupled model 
    final_state, predictions = model.run(
        init_cplstate = initial_state,
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
