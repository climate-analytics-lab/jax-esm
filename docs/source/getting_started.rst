Getting started
===============


Install JEM
-----------


.. code-block:: bash

    # JAX-GCM (jcm) >= 3.0 is not on PyPI yet: install its dev branch from source FIRST
    git clone https://github.com/climate-analytics-lab/jax-gcm
    cd jax-gcm
    git switch dev                # then `git checkout <JCM_SUPPORTED_REV>` to pin it
    pip install -e "."
    cd ..

    # Install JEM
    git clone https://github.com/climate-analytics-lab/jax-esm
    cd jax-esm
    pip install -e "."
    cd ..

    # Optional: the jittable Veros fork, only needed for the Veros
    # configurations (+configuration=veros-double-drake / veros-earth)
    git clone https://github.com/meteorologytoday/veros-jittable.git
    cd veros-jittable
    pip install -e "."
    cd ..

Plotting the examples needs the plotting extras, which are JEM's own, so
install them from the JEM checkout: ``cd jax-esm && pip install -e ".[plot]"``.
Every block above returns to the parent directory it started in, so that
relative path is correct whether or not the optional Veros step was run.

JAX-ESM is developed and tested against **one** JAX-GCM revision, recorded as
``JCM_SUPPORTED_REV`` in :mod:`jem.components.jcm.contract` together with every
JAX-GCM name JAX-ESM calls. Check that revision out if a coupled run fails with
an ``AttributeError`` inside ``jcm``: ``pytest tests/unit/test_jcm_contract.py``
reports exactly which name moved.


Your first coupled run
-----------------------

.. code-block:: bash

    python -m jem.main +configuration=aquaplanet-slab coupled_run=short_run

This composes a coupled model from the shipped ``aquaplanet-slab``
configuration (the JCM atmosphere and JEM's slab ocean, coupled with
``default_exchangers`` -- :doc:`python_api`'s worked example plus the
thermodynamic sea ice that page leaves out) and integrates two coupled days --
``short_run`` is the shortest run that exercises the whole path, meant to
check a machine can run anything at all before committing to a real
integration. It writes into a fresh Hydra run directory,
``outputs/<date>/<time>/``: one file per component per chunk,
``atm-00000000.nc``, ``ocn-00000000.nc`` and ``seaice-00000000.nc``, plus a
``checkpoint/`` directory. About 50 seconds on a laptop CPU, mostly XLA
compilation.


The command line
-----------------

``python -m jem.main`` (or the ``jem`` console script) composes the model from
Hydra config groups: JAX-ESM's own groups (``ocean``, ``land``, ``seaice``,
``coupling``, ``regrid``, ``coupled_run``, ``configuration``) sit at the top
level, and **JAX-GCM's own groups re-rooted under** ``atmosphere``, so
anything that works in ``python -m jcm.main`` works here with the group's
package spelled out.

.. code-block:: bash

    # every group, option and override spelling -- JAX-ESM's own help, not
    # the atmosphere's
    python -m jem.main --help

    # compose the fully-resolved config and print it; run nothing
    python -m jem.main +configuration=earth-slab --cfg job

.. list-table::
   :header-rows: 1

   * - To do this
     - Write this
   * - Run a named coupled configuration
     - ``+configuration=earth-slab``
   * - Compose a whole jax-gcm bundle as the atmosphere
     - ``+configuration@atmosphere=speedy-t31``
   * - Change one atmosphere group
     - ``physics@atmosphere.physics=held_suarez grid@atmosphere.grid=held_suarez_t31_l8``
   * - Set one atmosphere key
     - ``atmosphere.run.time_step=7``
   * - Choose a surface component
     - ``ocean=slab_relax ocean.sst_clim_file='${jcm_data:bc/t30/clim/forcing.nc}'``
   * - Drop one
     - ``land=none``
   * - Set a component parameter
     - ``+ocean.params.relaxation_time=1e6``
   * - Override a physical constant, for every component
     - ``+atmosphere.constants.grav=9.7``
   * - Choose the run settings
     - ``coupled_run=short_run``, or ``coupled_run.total_time="90 days"``

Several things worth knowing:

- **Spell the** ``@atmosphere``. ``+configuration=speedy-t31`` without it
  composes that jax-gcm bundle at the *root*, where its ``physics``,
  ``terrain`` and ``run`` keys are nobody's and nothing reads them. The
  atmosphere's groups always carry their package:
  ``<group>@atmosphere.<group>=<option>``.
- **The coupled run's own settings are** ``coupled_run``, **not** ``run``.
  ``atmosphere.run`` is the atmosphere's own run config, and a ``run`` group
  here would shadow jax-gcm's -- Hydra resolves a group option from the first
  search-path entry that has it, and jax-gcm's ``run/default.yaml`` and
  ``run/longrun.yaml`` would otherwise be picked up in its place.
  ``coupled_run/default.yaml`` is the complete schema, so every key is
  overridable without a ``+``.
- **Single-quote a** ``${...}`` **resolver.** ``'${jcm_data:bc/t30/clim/forcing.nc}'``
  is a resolver Hydra expands when the config is composed; unquoted, the shell
  expands ``${...}`` to nothing first, so the override arrives empty.
- **A ``+`` prefixes a key that has no YAML entry to override**, such as
  ``+ocean.params.relaxation_time=1e6``: the component's physics defaults live
  on its Python class, not in a YAML file, so there is no key there for a
  plain override to find.
- **The exit status means what a scheduler thinks it means.** ``0`` when the
  run reached the time it was asked for, ``1`` when the health gate stopped it
  early (the output and checkpoint written so far are kept, and the reason is
  logged). So ``python -m jem.main ... && <post-processing>`` runs the
  post-processing only on a run that finished.

The YAML is wiring only -- ``_target_``, required input files, and the
non-default choices that define a named configuration. Every physics default
lives on the Python class that owns it, and every *run* default on
``jem.driver.run_chunked``.


The same run in Python
-----------------------

The configuration layer is a thin wiring layer over exactly the objects it
builds -- :doc:`python_api` gives the complete construction, in the order the
pieces above come from: the wrapper, the grid, the exchanger, the coupler and
the run loop.

``+configuration=<name>``'s Python equivalent is :func:`jem.configurations.load`
(issue #131) -- the *recipe door* onto the same
``jem/config/configuration/*.yaml`` this section's ``+configuration=`` composes,
built through the same ``jem.runners`` the CLI uses, with no Hydra visible to
the caller:

.. code-block:: python

    from jem import configurations, run_chunked

    exp = configurations.load("earth-slab")
    exp.coupler                                       # the built Coupler
    result = run_chunked(exp.coupler, **exp.run_kwargs)

``run_chunked(exp.coupler, **exp.run_kwargs)`` reproduces the CLI's build and
its ``coupled_run`` settings exactly, but NOT everything ``python -m
jem.main`` does around that build -- see :func:`jem.configurations.load`'s
own docstring for the precise, short list (a fresh ``output_dir`` of the
door's own rather than the CLI's Hydra-managed one, no working-directory
change, no logger-level change) and for why a ``+atmosphere.constants.*``
override outlives the call that applied it.

See :doc:`python_api`'s *Validated configurations from Python* section for
the escape hatch onto an override (a dotted key or a config-group selection
such as ``seaice="none"``) and why a notebook that runs a shipped
configuration should load it through this door rather than rebuilding it by
hand.


Long runs: checkpoints and resume
-----------------------------------

.. code-block:: bash

    python -m jem.main +configuration=earth-slab coupled_run=long_run

Checkpointing is on by default, into ``<output_dir>/checkpoint`` -- a relative
``checkpoint_path`` resolves against the run's own output directory, which
Hydra makes fresh each run. Point a second run at the first's output directory
(``coupled_run.output_dir=outputs/2026-09-16/11-04-02``) and it continues from
the coupled step the checkpoint holds, saying so in its log; an absolute
``coupled_run.checkpoint_path`` is used as given, and
``coupled_run.checkpoint_path=null`` turns checkpointing off. See the
README's *Long runs* section and :doc:`design/architecture` for
``checkpoint_interval``, ``subsample``, ``output_averages`` and the in-scan
reductions in ``jem.accumulate``.


Where next
-----------

- :doc:`examples` -- every example, and the command or notebook that runs it,
  is listed in ``examples/README.md``.
- :doc:`adding_a_component` -- wrapping an external model to join a coupled
  run.
- :doc:`experimental` -- features still under development.
- :doc:`design/architecture` -- the carry layout, the exchanger contract and
  the checkpoint format, for anyone debugging or extending the coupler.
