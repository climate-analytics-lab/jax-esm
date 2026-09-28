Examples
========

Every example is either one ``python -m jem.main +configuration=...``
command, or a notebook built entirely in Python -- no Hydra in any notebook
(issue #131) -- either constructing its coupled model directly or loading a
validated configuration through :mod:`jem.configurations`; see
:doc:`python_api`'s *Validated configurations from Python* section for the
door, and ``examples/README.md`` for which notebook uses which and why. The
pages below show the plotting each one adds, through the shared helpers in
:mod:`jem.plot`.

JCM with slab models
--------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples/01_basic/01_aquaplanet
   examples/01_basic/02_aquaplanet_customized_initial_condition
   examples/01_basic/03_aquaplanet_response_to_SST_perturbation_using_gradient
   examples/01_basic/04_jcm_slabs_mixed_grid_aqua_planet

Miscellaneous
-------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   examples/03_non_geoscience/01_SpringSystem


