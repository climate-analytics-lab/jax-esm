Copy-and-paste Quick Start
==========================


Install JEM
-----------


.. code-block:: bash

    # JAX-GCM (jcm) >= 2.1 is not on PyPI yet: install its dev branch from source FIRST
    git clone https://github.com/climate-analytics-lab/jax-gcm
    cd jax-gcm
    git switch dev          # then `git checkout <JCM_SUPPORTED_REV>` to pin it
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

JAX-ESM is developed and tested against **one** JAX-GCM revision, recorded as
``JCM_SUPPORTED_REV`` in :mod:`jem.components.jcm.contract` together with every
JAX-GCM name JAX-ESM calls. Check that revision out if a coupled run fails with
an ``AttributeError`` inside ``jcm``: ``pytest tests/unit/test_jcm_contract.py``
reports exactly which name moved.


Run the First Coupled Run
-------------------------

A complete, runnable aquaplanet simulation coupling the JCM atmosphere to JEM's
slab ocean. It takes a couple of minutes on a laptop CPU, and writes
``atm-00000000.nc``, ``atm-00000005.nc``, ``ocn-00000000.nc`` and
``ocn-00000005.nc`` into ``output/`` -- one file per component per chunk, named
after the coupled step its chunk starts at (here 0 and 5, the two five-day
chunks of a ten-day run).

.. code-block:: python

    import jax_datetime as jdt
    import jcm
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    from jem import Coupler, default_exchangers, run_chunked
    from jem.components import JCMComponent, SlabOceanModel
    from jem.components.slab import SlabGrid

    start_date = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")

    # The JCM atmosphere: a plain jcm.model.Model, wrapped as a component.
    atm_model = jcm.model.Model(coords=get_speedy_coords(), start_date=start_date)
    atm = JCMComponent(atm_model)

    # Aquaplanet: the slab grid is built from the atmosphere's own horizontal
    # grid, and with no fractional mask every cell is ocean.
    grid = SlabGrid.from_coords(atm_model.coords.horizontal)

    # `default_exchangers` is the standard coupling written down once: the
    # atmosphere's surface heat flux drives the ocean, and the ocean's SST comes
    # back as the atmosphere's boundary condition.
    components = {"atm": atm, "ocn": SlabOceanModel(grid)}
    coupler = Coupler(
        components,
        default_exchangers(components),
        coupling_timestep=coupling_timestep,
        start_date=start_date,
    )
    print(repr(coupler))

    result = run_chunked(
        coupler, total_time="10 days", chunk="5 days", output_dir="output"
    )
    print(result.steps_completed, "coupled steps;", len(result.paths), "files")

The pieces, in the order they appear:

- **The wrapper** :class:`~jem.components.jcm.component.JCMComponent` adapts a
  stock ``jcm.model.Model`` without touching it -- no methods are attached to
  the model. The coupler calls its ``bind()`` when it is registered, which is
  where the model's start date, calendar and timestep are checked against the
  coupler's.
- **The grid** comes from the atmosphere's own ``coords.horizontal``, so the
  ocean cannot end up on a grid that merely resembles the atmosphere's. Pass
  ``fractional_mask=`` (e.g. ``jcm.terrain.TerrainData.from_file(...).fmask``)
  for a land-sea mask; without one every cell is ocean.
- **The exchanger** is the only place where components exchange anything.
  :func:`~jem.exchangers.default_exchangers` builds the standard wiring for
  whichever of the standard components (``atm``, ``ocn``, ``lnd``, ``seaice``)
  are present. An exchange it cannot express -- one that regrids, computes a
  flux, converts units or blends two fields -- is a plain function
  ``(dict[str, carry], CouplingTime) -> dict[str, carry]``; :doc:`tutorial`
  writes one out. Coupling is **lagged**: with the default workflow the
  exchanger at step *n* moves what each component produced during step *n-1*.
- **The coupler** owns the clock: the coupling timestep, the start date and the
  calendar live here and nowhere else, and every component's ``step`` is handed
  the same ``CouplingTime``.
- **The workflow** -- printed by ``repr(coupler)`` -- is the coupling scheme.
  It defaults to every exchanger followed by every component; pass
  ``workflow=["atm", "exchange", "ocn"]`` to reorder it. It may be nested, and a
  name may appear more than once: an element listed *n* times runs *n* times per
  coupled step, on a clock *n* times faster. So
  ``workflow=[["atm_lnd_exchange", "atm", "lnd"] * 24, "atm_ocn_exchange",
  "ocn"]`` couples the atmosphere and the land hourly inside a daily ocean
  coupling, and the hourly components write 24 output records per coupled step.
  The same model can be written as an hourly ``Coupler`` registered as a
  component of the daily one -- a ``Coupler`` satisfies the component contract.
  See :doc:`design/architecture` for both forms.
- **The run loop** :func:`~jem.driver.run_chunked` integrates in chunks: per
  chunk it writes one file per component, checkpoints if it was given a path,
  and runs a health check on the result, stopping the run if the atmosphere has
  gone unstable. Every run default lives on its signature. ``total_time`` and
  ``chunk`` must both be whole multiples of the coupling timestep, and
  ``total_time`` a whole multiple of ``chunk``.

For the same run with a sea-ice component and plotting, see
:doc:`examples/01_basic/01_aquaplanet`.


The same run from the command line
----------------------------------

The configuration layer is a thin wiring layer over exactly those objects, so
the run above is also one command:

.. code-block:: bash

    python -m jem.main +configuration=aquaplanet-slab coupled_run=smoke

JAX-ESM's own config groups (``ocean``, ``land``, ``seaice``, ``coupling``,
``regrid``, ``coupled_run``, ``configuration``) sit at the top level, and
**JAX-GCM's own groups are composed under** ``atmosphere``, with their names
unchanged -- so the group's package is spelled out in an override:

.. code-block:: bash

    # a named coupled configuration, and the run settings
    python -m jem.main +configuration=earth-slab coupled_run.total_time="90 days"

    # a whole JAX-GCM configuration bundle as the atmosphere, tweaked on top
    python -m jem.main +configuration@atmosphere=speedy-t31 atmosphere.run.time_step=7

    # single groups and keys
    python -m jem.main physics@atmosphere.physics=held_suarez \
        grid@atmosphere.grid=held_suarez_t31_l8 \
        ocean=slab_relax ocean.sst_clim_file='${jcm_data:bc/t30/clim/forcing.nc}' \
        land=none +ocean.params.relaxation_time=1e6

    # every group, option and override spelling
    python -m jem.main --help

Two things to know. Spell the ``@atmosphere``: ``+configuration=speedy-t31``
without it composes that JAX-GCM bundle at the *root*, where nothing reads its
keys. And the coupled run's own settings are ``coupled_run``, not ``run`` --
``atmosphere.run`` is the atmosphere's own run config, and a ``run`` group here
would shadow JAX-GCM's.

A long run checkpoints and writes chunk means:

.. code-block:: bash

    python -m jem.main +configuration=earth-slab coupled_run=longrun

Run it again with the same ``coupled_run.checkpoint_path`` and it continues from
the coupled step the checkpoint holds. In Python that is the same call:

.. code-block:: python

    result = run_chunked(
        coupler,
        total_time="10 years",
        chunk="30 days",          # a file, a restart and a health check a month
        output_dir="output",
        output_averages=True,     # one record per chunk: the monthly mean
        checkpoint_path="checkpoint",
    )
