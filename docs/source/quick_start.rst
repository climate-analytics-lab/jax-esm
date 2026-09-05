Copy-and-paste Quick Start
==========================


Install JEM
-----------


.. code-block:: bash

    # JAX-GCM (jcm) >= 2.1 is not on PyPI yet: install its dev branch from source FIRST
    git clone https://github.com/climate-analytics-lab/jax-gcm
    cd jax-gcm
    git switch dev
    pip install -e "."
    cd ..

    # Install JEM
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
    from jem.components import JCMComponent, SlabOceanModel
    from jem.components.slab import SlabGrid

    start_date = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")


    # An exchanger is the only place where components exchange information. It is
    # traced with everything else, so it must not write into the carries it is
    # handed: it builds new ones and returns the mapping to continue with.
    def atm_ocn_exchange(components, time):
        del time  # this exchange does not depend on the date
        atm, ocn = components["atm"], components["ocn"]
        ocn = dict(
            ocn,
            forcing=ocn["forcing"].replace(
                total_heat_flux=atm["derived"].total_heat_flux,
            ),
        )
        atm = dict(
            atm,
            forcing=atm["forcing"].replace(
                sea_surface_temperature=ocn["state"].sea_surface_temperature,
            ),
        )
        return dict(components, atm=atm, ocn=ocn)


    # The JCM atmosphere: a plain jcm.model.Model, wrapped as a component.
    atm_model = jcm.model.Model(coords=get_speedy_coords(), start_date=start_date)
    atm = JCMComponent(atm_model)

    # Aquaplanet: the slab grid is built from the atmosphere's own horizontal grid,
    # and with no fractional mask every cell is ocean.
    grid = SlabGrid.from_coords(atm_model.coords.horizontal)

    coupler = Coupler(
        {"atm": atm, "ocn": SlabOceanModel(grid)},
        {"atm_ocn_exchange": atm_ocn_exchange},
        coupling_timestep=coupling_timestep,
        start_date=start_date,
    )
    print(repr(coupler))

    # The default workflow is every exchanger followed by every component, so the
    # fields are exchanged first and both components then step on the same state.
    simulation_interval = jdt.to_timedelta(10, "day")
    run = coupler.generate_trajectory_function(
        int(simulation_interval / coupling_timestep)
    )
    final_carry, diagnostics = run(coupler.initialize())

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    for component_name, ds in coupler.to_xarray(diagnostics).items():
        ds.to_netcdf(output_dir / f"{component_name:s}.nc", engine="netcdf4")

The pieces, in the order they appear:

- **The exchanger** is a plain function
  ``(dict[str, carry], CouplingTime) -> dict[str, carry]``. It is the only place
  where components exchange anything: here, the atmosphere's surface heat flux
  goes to the ocean, and the ocean's SST comes back as the atmosphere's boundary
  condition. It is traced with the rest of the step, so it must build new structs
  rather than assign into the ones it was handed, and must not change their
  pytree structure.
- **The wrapper** ``JCMComponent`` adapts a stock ``jcm.model.Model`` without
  touching it — no methods are attached to the model. The coupler calls its
  ``bind()`` when it is registered, which is where the model's start date,
  calendar and timestep are checked against the coupler's.
- **The grid** comes from the atmosphere's own ``coords.horizontal``, so the
  ocean cannot end up on a grid that merely resembles the atmosphere's. Pass
  ``fractional_mask=`` (e.g. ``jcm.terrain.TerrainData.from_file(...).fmask``)
  for a land-sea mask; without one every cell is ocean.
- **The coupler** owns the clock: the coupling timestep, the start date and the
  calendar live here and nowhere else, and every component's ``step`` is handed
  the same ``CouplingTime``.
- **The workflow** — printed by ``repr(coupler)`` — is the coupling scheme.
  It defaults to every exchanger followed by every component; pass
  ``workflow=["atm", "atm_ocn_exchange", "ocn"]`` to reorder it.
- **The trajectory function** is a pure ``carry -> (carry, diagnostics)``
  function built on ``jax.lax.scan``. Call it again on the carry it returned and
  the run continues, because the step counter lives in the carry.
- **The output**: ``to_xarray`` labels every component's dataset on the same
  time axis and coordinates, so ``xr.merge`` of the two aligns. For a chunked
  run, pass ``first_step=`` (the ``step`` of the carry the chunk started from) or
  every chunk is labelled with the first chunk's dates.

For the same run with a sea-ice component and plotting, see
:doc:`examples/01_basic/01_aquaplanet`.
