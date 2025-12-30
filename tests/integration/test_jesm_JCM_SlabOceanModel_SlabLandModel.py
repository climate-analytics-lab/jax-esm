"""Example of coupling jax-gcm with simple slab ocean and slab land models."""

def test_integration():
    
    import jcm
    from jcm.geometry import Geometry
    import jax_datetime as jdt
    from pathlib import Path

    from jax_esm.tool_scripts.generate_jcm_forcing_and_topography_files import (
        generate_jcm_forcing_and_topography_files,
    )
    from jax_esm.components import JCM, SlabLandModel, SlabOceanModel
    from jax_esm.coupling.transformer import IdentityTransformer
    from jax_esm.coupling.forcing_mapper import ForcingMapper
    from jax_esm.coupling.coupler import Coupler
    import jax_esm.utils.tree_tools as tree_tools


    resolution = 31
    grid_specification = f"JCM::T{resolution:d}"

    coupling_timestep = 86400.0
    start_datetime = jdt.to_datetime("2000-01-01")
    simulation_interval = jdt.to_timedelta(10, "day")
    output_dir = Path("output/JCM_SOM_SLM").resolve()

    external_files = generate_jcm_forcing_and_topography_files(resolution=resolution)
    print("Output dir: ", str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)

    geometry = Geometry.from_file(external_files["terrain"])

    # Creating components
    components = dict(
        atm=JCM(
            model=jcm.model.Model(start_date=start_datetime, geometry=geometry),
            coupling_timestep=coupling_timestep,
            save_interval=coupling_timestep,
        ),
        ocn=SlabOceanModel(
            grid_specification=grid_specification,
            timestep=coupling_timestep,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
            relaxation_time=60 * 86400.0,
            mask_file=external_files["terrain"],
            SST_clim_file=external_files["forcing"],
        ),
        lnd=SlabLandModel(
            grid_specification=grid_specification,
            timestep=coupling_timestep,
            start_datetime=start_datetime,
            save_interval=coupling_timestep,
            relaxation_time=60 * 86400.0,
            topography_file=external_files["terrain"],
            mask_file=external_files["terrain"],
            land_clim_file=external_files["forcing"],
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
        a2l = dict(
            identity_transformer = IdentityTransformer(
                source_grid = components["atm"].domain.horizontal_grids["T"],
                target_grid = components["lnd"].domain.horizontal_grids["T"],
            ),
        ),
        l2a = dict(
            identity_transformer = IdentityTransformer(
                source_grid = components["lnd"].domain.horizontal_grids["T"],
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
    forcing_mapper.add_forcing_mapping(
        source = ("atm", "extra.total_heat_flux"),
        target = ("lnd", "flux.total_heat_flux"),
        transformer = transformers["a2l"]["identity_transformer"],
    )
    forcing_mapper.add_forcing_mapping(
        source = ("lnd", "prog.land_surface_temperature"),
        target = ("atm", "stl_am"),
        transformer = transformers["l2a"]["identity_transformer"],
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

