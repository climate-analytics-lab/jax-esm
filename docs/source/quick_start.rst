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



