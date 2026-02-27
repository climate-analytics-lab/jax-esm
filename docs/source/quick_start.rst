Quick Start
=============


Install JEM
-----------


.. code-block::
 
    # Install published jem
    pip install jem
    
    # Using locally cloned jem
    git clone -b [version_tag] https://github.com/climate-analytics-lab/jax-esm
    cd jax-esm
    pip install -e "."


Run the First Coupled Run
-------------------------

Here is an example to run an aquaplanet simulation.

.. code-block::

    import jcm
    import jax_datetime as jdt

    from jem import Coupler
    from jem.components import JCM, SlabOceanModel
    from jem.mapping import BasicMapper

    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    one_second = jdt.to_timedelta(1, "second")
    
    mapper = BasicMapper()
    mapper.add_mapping(
        source = ("atm", "derived.total_heat_flux"),
        target = ("ocn", "forcing.total_heat_flux"),
    )
    mapper.add_mapping(
        source = ("ocn", "state.sea_surface_temperature"),
        target = ("atm", "forcing.sea_surface_temperature"),
    )

    atm_model = jcm.model.Model(
        start_date=start_datetime,
    )

    atm_model = JCM.make_jem_compatible(
        atm_model,
        coupling_timestep=coupling_timestep,
        land_model_active=False,
    )

    model = Coupler(
        components=dict(
            atm=atm_model,
            ocn=SlabOceanModel(
                start_datetime=start_datetime,
                timestep=coupling_timestep / one_second,
            ),
        ),
        mappers=dict(mapper=mapper),
    )

    simulation_interval = jdt.to_timedelta(30, "day")
    initial_state, final_state, predictions = model.run(
        workflow=["mapper", "atm", "ocn"],
        iterations = int(simulation_interval / coupling_timestep),
    )

    output_dict = model.predictions_to_xarray(predictions)
    print(output_dict["atm"]) # xarray.Dataset
    print(output_dict["ocn"]) 

Here we provide a template code to make an animation of surface specific
humidity.

.. code-block:: python

    from matplotlib.animation import FuncAnimation
    from IPython.display import Image
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
    ani = FuncAnimation(fig, update, frames=len(data.coords["time"]), interval=120, blit=False)
    ani.save('humidity_map.gif', writer='pillow', dpi=200)
    display(Image('humidity_map.gif'))



