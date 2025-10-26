"""Example of coupling jax-gcm with a simple slab ocean model."""

if __name__ == "__main__":
       
    import jcm
    topo_file = (Path(jcm.__file__).parent / "data/bc/t30/clim/boundaries_daily.nc").resolve()
    
    from jax_esm.coupling.couplers.CoupledJCMSlabOceanModel import CoupledJCMSlabOceanModel
    from datetime import datetime

    # Creating model
    model = CoupledJCMSlabOceanModel(
        topo_file = topo_file,
        horizontal_resolution = 31,
        JCM_layers = 8,
        coupling_timestep = 86400.0,
        JCM_substeps = 24,
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
        jax_scan = False,
    )

    # Convert output into xarray
    ds_output = model.predictions_to_xarray(predictions)
    
     
