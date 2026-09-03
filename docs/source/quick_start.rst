Copy-and-paste Quick Start
==========================


Install JEM
-----------


.. code-block:: bash

    git clone https://github.com/climate-analytics-lab/jax-esm
    cd jax-esm
    pip install -e "."

    # Optional: the jittable Veros fork, only needed for the JCM-Veros examples
    cd ..
    git clone https://github.com/meteorologytoday/veros-jittable.git
    cd veros-jittable
    pip install -e "."


Run the First Coupled Run
-------------------------

A complete, runnable aquaplanet simulation coupling the JCM atmosphere to JEM's
slab ocean. It takes a couple of minutes on a laptop CPU, and writes ``atm.nc``
and ``ocn.nc`` into ``output/``.

.. code-block:: python

    from pathlib import Path

    import jax_datetime as jdt
    import jcm
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    from jem import Coupler
    from jem.components import JCM, SlabOceanModel
    from jem.components.slab.grid import generate_slab_grid

    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    one_second = jdt.to_timedelta(1, "second")


    # A mapper is any function CoupledCarry -> CoupledCarry: it is the only place
    # where components exchange information.
    def atm_ocn_mapper(coupled_carry):
        atm = coupled_carry["atm"]
        ocn = coupled_carry["ocn"]
        ocn["forcing"].total_heat_flux = atm["derived"].total_heat_flux
        atm["forcing"].sea_surface_temperature = ocn["state"].sea_surface_temperature
        return coupled_carry


    # The JCM atmosphere: a plain jcm.model.Model, adapted in place.
    atm_model = JCM.make_jem_compatible(
        jcm.model.Model(coords=get_speedy_coords(), start_date=start_datetime),
        coupling_timestep=coupling_timestep,
    )

    # Aquaplanet: no mask file, so the fractional mask is all zero (no land).
    grid = generate_slab_grid("JCM::T31")

    model = Coupler(
        components=dict(
            atm=atm_model,
            ocn=SlabOceanModel(
                grid=grid,
                start_datetime=start_datetime,
                timestep=coupling_timestep / one_second,
            ),
        ),
        mappers=dict(atm_ocn_mapper=atm_ocn_mapper),
    )

    # The workflow is the coupling scheme: exchange, then step each component.
    simulation_interval = jdt.to_timedelta(10, "day")
    initial_carry, final_carry, predictions = model.run(
        workflow=["atm_ocn_mapper", "atm", "ocn"],
        iterations=int(simulation_interval / coupling_timestep),
    )

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    for component_name, ds in model.predictions_to_xarray(predictions).items():
        ds.to_netcdf(output_dir / f"{component_name:s}.nc", engine="netcdf4")

The pieces, in the order they appear:

- **The mapper** is a plain function ``CoupledCarry -> CoupledCarry``. It is the
  only place where components exchange anything: here, the atmosphere's surface
  heat flux goes to the ocean, and the ocean's SST comes back as the
  atmosphere's boundary condition.
- **The adapter** ``JCM.make_jem_compatible`` attaches JEM's interface to a
  stock ``jcm.model.Model`` in place. It checks that the coupling timestep is a
  whole multiple of JCM's own timestep.
- **The workflow** ``["atm_ocn_mapper", "atm", "ocn"]`` is the coupling scheme:
  exchange first, then step each component, once per coupling timestep.

For the same run with a sea-ice component and plotting, see
:doc:`examples/01_basic/01_aquaplanet`.
