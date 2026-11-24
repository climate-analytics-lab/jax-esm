Examples
========

Every runnable configuration is either one ``python -m jem.main
+configuration=...`` command (also spelled ``jem +configuration=...``) or a
notebook built entirely in Python -- no Hydra in any notebook (issue #131) --
that either constructs its coupled model directly or loads a named
configuration through :mod:`jem.configurations`; see :doc:`python_api`'s
*Validated configurations from Python* section for that door.


Named configurations
--------------------

Eight configurations ship in ``jem/config/configuration/``;
``jem.configurations.available()`` lists them from Python. Each is a command
with no other setup, each file's comments say why every setting is what it
is, and appending ``coupled_run=short_run`` turns any of them into a two-day
check that it composes, builds, steps and writes output on your machine
(``python -m jem.main +configuration=<name> --cfg job`` prints the fully
composed config instead of running). Unless told otherwise, a run integrates
the ``coupled_run`` default of 30 days, coupling once per day, and writes
``<component>-<first coupled step>.nc`` files plus a ``checkpoint/``
directory into ``outputs/<date>/<time>/``.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Configuration
     - What it couples
     - Command
   * - ``aquaplanet-slab``
     - SPEEDY T31L8 with jax-gcm's coupled-calibrated parameters (its
       ``speedy-t31`` bundle, 30-minute step) over a free slab ocean with
       SST-diagnosed slab sea ice; no land. The first thing to run.
     - ``python -m jem.main +configuration=aquaplanet-slab``
   * - ``aquaplanet-slab-echam``
     - jax-gcm's own ``t63-echam-1m`` atmosphere (ECHAM 1-moment microphysics
       and RRTMGP at T63 with 47 hybrid levels, 12-minute step, dry
       Jablonowski-Williamson start, upper sponge), composed whole, over a
       free slab ocean with SST-diagnosed slab sea ice on a planet with no
       land. Not a laptop run: a simulated day took about half an hour on 16
       CPU threads (five days, no GPU tried) and writes about 290 MB of
       atmosphere output, and the first run fetches jax-gcm's data mirror.
       Another jax-gcm atmosphere is one override,
       ``configuration@atmosphere=t63-echam-2m``. The Winton sea ice and the
       bucket land need SPEEDY's flux tiles and are refused with ECHAM.
     - ``python -m jem.main +configuration=aquaplanet-slab-echam``
   * - ``aquaplanet-slab-mixed-grid``
     - The same atmosphere (calibrated SPEEDY, 30-minute step), with the ocean
       and sea ice (every cell ocean) on a
       displaced-pole curvilinear grid and ESMF regridding weights mapping
       fields to and from the atmosphere's T31 Gaussian grid. Proves the
       coupler does not need one shared grid. The grid's displaced pole is
       open (a gap over Greenland covering 0.04% of the sphere), so 11 of the
       atmosphere's 4608 cells are only partly covered by the maps; the
       configuration's comments give the numbers.
     - ``python -m jem.main +configuration=aquaplanet-slab-mixed-grid``
   * - ``earth-slab``
     - Calibrated SPEEDY T31L8 (30-minute step) on jax-gcm's packaged T30
       terrain and climatology, over a slab ocean relaxed to the observed SST (30-day relaxation) and a slab
       sea ice started from the observed cover. The land is SPEEDY's own slab
       land; there is no land component. ``land=bucket`` adds a soil-water
       bucket under it, and ``land=slab`` replaces it with JEM's own slab land
       (``python -m jem.main +configuration=earth-slab land=slab``).
     - ``python -m jem.main +configuration=earth-slab``
   * - ``earth-slab-echam``
     - jax-gcm's ``t63-echam-1m`` configuration unchanged -- the atmosphere of
       ``aquaplanet-slab-echam`` on its native T63 orography and present-day
       boundary conditions, with ECHAM's own land (SLIM) -- over a slab ocean
       relaxed to the SST of the atmosphere's forcing file, with
       SST-diagnosed slab sea ice started from the same file's ice cover.
       ECHAM publishes grid-cell-mean surface fluxes, so the ocean under a
       partly-land cell also receives the land part's flux
       (:doc:`issues`). Same cost and data needs as the aquaplanet.
     - ``python -m jem.main +configuration=earth-slab-echam``
   * - ``earth-slab-winton``
     - ``earth-slab`` with the Winton thermodynamic sea ice (snow, two ice
       layers, an energy budget) in place of the SST-diagnosed ice; the ocean
       clamps at freezing and hands the ice its freezing deficit.
     - ``python -m jem.main +configuration=earth-slab-winton``
   * - ``veros-double-drake``
     - SPEEDY T31L8 (jax-gcm's coupled-calibrated parameters, 30-minute step)
       over the Veros ocean GCM on an idealised two-continent planet, the
       ocean built on the atmosphere's own cell centres, with an idealised
       land. Flat-bottomed ocean, uniform initial state, no land or sea-ice
       component. Needs Veros.
     - ``python -m jem.main +configuration=veros-double-drake``
   * - ``veros-earth``
     - SPEEDY T31L8 (jax-gcm's coupled-calibrated parameters, 30-minute step)
       over the Veros ocean GCM on a rotated Gaussian global grid, exchanged
       through ESMF weights: ETOPO 2022 bathymetry, a World Ocean Atlas initial
       state, the observed land climatology, and an atmosphere coastline
       derived from the ocean's. The Winton thermodynamic sea ice is on the
       ocean's own grid: the Veros ocean clamps its top cell at the freezing
       point and hands the ice its freezing deficit, and the ice hands the
       ocean the heat and freshwater it receives. Starts ice-free over the
       observed ocean state; in a 90-day run from 1 January the ice grows to
       7.6% of the northern ocean, 1.2 m thick on average poleward of 65N.
       Needs Veros.
     - ``python -m jem.main +configuration=veros-earth``

The six SPEEDY configurations (all but the two ``-echam``) run the SPEEDY
parameters that jax-gcm calibrated for coupled use, taken whole from its
``speedy-t31`` configuration bundle at the 30-minute step the calibration was
made at (jax-gcm's ``docs/source/design/speedy_retune.md``). The calibration
was a fit at T31 against observations with prescribed SSTs, so a coupled run is
not guaranteed drift-free, and it belongs to T31: the T106 command below
applies the same values on a grid they were not fitted on. A bare
``python -m jem.main`` and a Python-built ``jcm.model.Model()`` run SPEEDY's
reference values.

The Veros configurations are experimental and have known limitations; see
:doc:`experimental` and :doc:`issues`. A longer, higher-resolution run is the
same command with more overrides, for example a T106 aquaplanet:

.. code-block:: bash

    python -m jem.main +configuration=aquaplanet-slab coupled_run=long_run \
        atmosphere.grid.spectral_truncation=106 atmosphere.run.time_step=10


Notebooks
---------

The notebooks are committed without outputs, and the documentation build does
not execute them, so the pages below show each notebook's code and
explanation, not its plots. To see the plots, run the notebook yourself:

.. code-block:: bash

    pip install -e ".[plot]" jupyter   # the plotting extra, and a notebook runtime
    jupyter lab examples/01_basic/01_aquaplanet.ipynb
    ./examples/run_notebooks.sh examples/01_basic/   # or execute them in place

The notebooks reach their model in one of two ways: ``01_aquaplanet``,
``02_aquaplanet_customized_initial_condition`` and
``04_jcm_slabs_mixed_grid_aqua_planet`` *teach* how a coupled model is put
together and construct their components directly (they have the coupled
structure of ``aquaplanet-slab`` and ``aquaplanet-slab-mixed-grid``). ``01``
and ``04`` wire them with ``default_exchangers``; ``02`` writes the same
exchange out as a plain function, the form to read first for what an exchanger
is (:doc:`design/exchange`). ``03``
and the Earth-like notebook *run* a named configuration to show something else
and load it with ``configurations.load``; ``SpringSystem`` is the coupler with
no climate in it at all. The geoscience notebooks share their plotting through
:mod:`jem.plot`.

JCM with slab models
~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples/01_basic/01_aquaplanet
   examples/01_basic/02_aquaplanet_customized_initial_condition
   examples/01_basic/03_aquaplanet_response_to_SST_perturbation_using_gradient
   examples/01_basic/04_jcm_slabs_mixed_grid_aqua_planet

Miscellaneous
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples/03_non_geoscience/01_SpringSystem

The Earth-like notebook, ``examples/02_experimental/01_earth.ipynb``, is
listed under :doc:`experimental`.
