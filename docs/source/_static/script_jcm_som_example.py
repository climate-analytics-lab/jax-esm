from pathlib import Path

import jcm
import jcm.model
import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jem import Coupler
from jem.components import SlabOceanModel

def _make_jcm_to_be_jem_compatible(
    model: jcm.model.Model,
    coupling_timestep: jdt.Timedelta,
) -> jcm.model.Model:

    D2_nodal_shape = model.coords.nodal_shape[1:]
    def initialize():
        return {
            "state" : model._prepare_initial_modal_state(),
            "derived" : {
                "total_heat_flux" : jnp.zeros(D2_nodal_shape),
            },
            "forcing" : jcm.forcing.default_forcing(model.coords.horizontal).copy(
                lfluxland = jnp.bool_(False),
            )
        }

    # Notice: since save_interval and total_time are claimed
    #         static parameters, we cannot pass in traceable
    #         object. So use item() to convert from scalar
    #         jax.Array to float.
    save_interval_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item()
    total_time_day=(coupling_timestep / jdt.to_timedelta(1, "day")).item()
    def generate_step_function():
        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]
            new_atm_modal_state, predictions = model.run_from_state(
                initial_state=state,
                save_interval=save_interval_day,
                total_time=total_time_day,
                forcing=forcing,
                output_averages=True,
            )
            physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)
            total_heat_flux = jnp.sum(physics_no_time_dimension.surface_flux.hfluxn, axis=2)

            # This is a bug in jcm: Time dimension vanishes when save_interval == total_time
            if len(predictions.dynamics.normalized_surface_pressure.shape) == 2:
                nsp = predictions.dynamics.normalized_surface_pressure
                predictions.dynamics.normalized_surface_pressure = jnp.reshape(nsp, (1,) + nsp.shape)

            return (
                # Important: the structure of the returned carry value must
                #            match the carry value structure received in the
                #            first place, which is the same structure returned
                #            by the function `initialize`
                {
                    "state" : new_atm_modal_state,
                    "derived" : {
                        "total_heat_flux" : total_heat_flux,
                    },
                    "forcing" : forcing,
                },
                predictions
            )
        return step_function

    def predictions_to_xarray(predictions):
        return predictions.to_xarray()

    def get_info():
        return {
            "diffusion" : str(model.diffusion),
        }

    setattr(model, "initialize", initialize)
    setattr(model, "generate_step_function", generate_step_function)

    # Optional
    setattr(model, "predictions_to_xarray", predictions_to_xarray)
    setattr(model, "get_info", get_info)

    return model


if __name__ == "__main__":

    coupling_timestep = jdt.to_timedelta(1, "day")
    start_datetime = jdt.to_datetime("2000-01-01")

    def interaction_between_atm_and_ocn(coupled_carry):
        atm = coupled_carry["atm"]
        ocn = coupled_carry["ocn"]

        atm["forcing"].sea_surface_temperature = ocn["state"].sea_surface_temperature
        ocn["forcing"].total_heat_flux = atm["derived"]["total_heat_flux"]

        return coupled_carry

    atm_model = jcm.model.Model(
        start_date=start_datetime,
    )

    atm_model = _make_jcm_to_be_jem_compatible(
        atm_model,
        coupling_timestep=coupling_timestep,
    )

    model = Coupler(
        components=dict(
            atm=atm_model,
            ocn=SlabOceanModel(
                start_datetime=start_datetime,
                timestep=coupling_timestep / jdt.to_timedelta(1, "second"),
            ),
        ),
        mappers=dict(interaction_between_atm_and_ocn=interaction_between_atm_and_ocn),
    )

    simulation_interval = jdt.to_timedelta(30, "day")
    initial_state, final_state, predictions = model.run(
        workflow=["interaction_between_atm_and_ocn", "atm", "ocn"],
        iterations = int(simulation_interval / coupling_timestep),
    )

    output_dict = model.predictions_to_xarray(predictions)
    output_dir = Path("test_output/JCM_SOM").resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")

    # Additional code for visualizing quickly using matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    import numpy as np

    data = output_dict["atm"]["specific_humidity"]

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.gridlines(draw_labels=True)
    cb = None
    cf = None

    def update(frame):
        global cf, cb
        _data = data.isel(time=frame, level=0)

        if cf is not None:
            for coll in cf.collections:
                coll.remove()

        # Plot the humidity field for the current time step
        lat = _data.coords["lat"]
        lon = _data.coords["lon"]
        cyclic_data, cyclic_lon = add_cyclic_point(_data.to_numpy().transpose(), coord=lon)
        mappable = ax.contourf(
            cyclic_lon, lat,
            cyclic_data,
            levels=1 + np.linspace(0, 1, 21) * 10,
            transform=ccrs.PlateCarree(),
            cmap='GnBu',
            extend="both",
        )

        ax.set_title(f"[{_data['time'].dt.strftime('%Y-%m-%d').to_numpy().item()}] Surface specific humidity")
        if cb is None:
            cb = plt.colorbar(ax=ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.07)
            cb.set_label("[g/kg]", fontsize=12)

        return [cf,]

    # Generate and save
    output_animation = Path("humidity_map.gif").resolve()
    print(f"Generate animation: {str(output_animation)}")
    ani = FuncAnimation(fig, update, frames=len(data.coords["time"]), interval=120, blit=False)
    ani.save(output_animation, writer='pillow', dpi=200)
