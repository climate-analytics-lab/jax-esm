"""Example of coupling jax-gcm with simple slab ocean and slab land models."""

def test_integration():
    
    import jcm
    import jax_datetime as jdt
    from pathlib import Path

    from jem.components import JCM, SlabOceanModel
    from jem.coupling.transformer import IdentityTransformer
    from jem.coupling.forcing_mapper import ForcingMapper
    from jem.coupling.coupler import Coupler
    import jem.utils.tree_tools as tree_tools


    resolution = 31
    grid_specification = f"JCM::T{resolution:d}"

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime("2000-01-01")
    simulation_interval = jdt.to_timedelta(10, "day")
    output_dir = Path("output/JCM_SOM").resolve()

    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)
        
    # Creating components
    components = dict(
        atm=JCM.make_jem_compatible(
            jcm.model.Model(start_date=start_datetime),
            land_model_active=False,
            save_interval=86400.0,
        ),
        ocn=SlabOceanModel(
            grid_specification=grid_specification,
            timestep=coupling_timestep,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
        ),
    )

    # Creating transformations
    transformers = dict(
        a2o = dict(
            identity_transformer = IdentityTransformer(
                source_grid = components["atm"].domain.horizontal_grids["T"],
                target_grid = components["ocn"].domain.horizontal_grids["T"],
            ),
        ),
        o2a = dict(
            identity_transformer = IdentityTransformer(
                source_grid = components["ocn"].domain.horizontal_grids["T"],
                target_grid = components["atm"].domain.horizontal_grids["T"],
            ),
        ),
    )

    forcing_mapper = ForcingMapper(components=components)
    forcing_mapper.add_forcing_mapping(
        source = ("atm", "extra.total_heat_flux"),
        target = ("ocn", "flux.total_heat_flux"),
        transformer = transformers["a2o"]["identity_transformer"],
    )
    forcing_mapper.add_forcing_mapping(
        source = ("ocn", "prog.sea_surface_temperature"),
        target = ("atm", "sea_surface_temperature"),
        transformer = transformers["o2a"]["identity_transformer"],
    )

    # Creating model
    model = Coupler(
        components=components,
        forcing_mapper=forcing_mapper,
        coupling_timestep=coupling_timestep,
    )

    print("Model info: ") 
    tree_tools.print_tree(model.get_info(), root="Model")

    # Obtain initial condition
    initial_coupled_state_forcing = model.initialize()

    print("Model state:")
    tree_tools.print_tree(initial_coupled_state_forcing, root="ModelState")

    # Run coupled model
    print("Running model...")
    trajectory_function = model.generate_trajectory_function(
        start_time=0,
        end_time=simulation_interval / jdt.to_timedelta(1, "second"),
        jitted=True,
        show_progress=True,
        tqdm_kwargs=dict(desc="Simulation"),
    )
 
    # Run coupled model
    print("Running model...")
    state_holder, predictions = trajectory_function(initial_coupled_state_forcing)
 
    # Convert output into xarray
    output_dict = model.predictions_to_xarray(predictions)

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")

if __name__ == "__main__":
    test_integration()

