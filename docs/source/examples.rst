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

Six configurations ship in ``jem/config/configuration/``;
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
     - SPEEDY T31L8 (12-minute step) over a free slab ocean with
       SST-diagnosed slab sea ice; no land. The first thing to run.
     - ``python -m jem.main +configuration=aquaplanet-slab``
   * - ``aquaplanet-slab-mixed-grid``
     - The same, with the ocean and sea ice on a displaced-pole curvilinear
       grid and ESMF regridding weights mapping fields to and from the
       atmosphere's T31 Gaussian grid. Proves the coupler does not need one
       shared grid.
     - ``python -m jem.main +configuration=aquaplanet-slab-mixed-grid``
   * - ``earth-slab``
     - SPEEDY T31L8 on jax-gcm's packaged T30 terrain and climatology, over a
       slab ocean relaxed to the observed SST (30-day relaxation) and a slab
       sea ice started from the observed cover. The land is SPEEDY's own slab
       land; there is no land component.
     - ``python -m jem.main +configuration=earth-slab``
   * - ``earth-slab-winton``
     - ``earth-slab`` with the Winton thermodynamic sea ice (snow, two ice
       layers, an energy budget) in place of the SST-diagnosed ice; the ocean
       clamps at freezing and hands the ice its freezing deficit.
     - ``python -m jem.main +configuration=earth-slab-winton``
   * - ``veros-double-drake``
     - SPEEDY T31L8 (30-minute step) over the Veros ocean GCM on an idealised
       two-continent planet, on a uniform lat-lon grid that shares the
       atmosphere's longitudes. No land or sea-ice component. Needs Veros.
     - ``python -m jem.main +configuration=veros-double-drake``
   * - ``veros-earth``
     - SPEEDY T31L8 (30-minute step) over the Veros ocean GCM on a rotated
       Gaussian global grid read from a SCRIP file, exchanged through ESMF
       weights. No land or sea-ice component. Needs Veros.
     - ``python -m jem.main +configuration=veros-earth``

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

The notebooks reach their model in one of two ways: ``01_aquaplanet`` and
``04_jcm_slabs_mixed_grid_aqua_planet`` *teach* how a coupled model is put
together and construct their components directly (they are equivalent to
``aquaplanet-slab`` and ``aquaplanet-slab-mixed-grid``); ``02`` and ``03``
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
