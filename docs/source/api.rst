API Reference
=============

The coupling core
-----------------

.. autosummary::
   :toctree: generated
   :recursive:

   jem.base.component
   jem.base.coupler
   jem.exchangers

Running a model
---------------

``jem.driver`` is the run loop a Python user calls; ``jem.runners`` and
``jem.main`` are the same loop reached from a composed Hydra config.
``jem.configurations`` is the recipe door onto a validated
``jem/config/configuration/*.yaml`` -- ``jem.configurations.load(name)`` builds
the SAME objects ``+configuration=<name>`` composes, through ``jem.runners``,
with no Hydra visible to the caller (issue #131; see :doc:`python_api`).
``jem.checkpoint`` and ``jem.accumulate`` are module-level APIs -- they are not
re-exported from ``jem``, because a run reaches them through
``Coupler.save_carry`` and ``generate_trajectory_function(accumulate=...)``.
``jem.adjoint`` is the gradient of a run too long to differentiate in one
piece (see :doc:`design/differentiating_long_runs`).

.. autosummary::
   :toctree: generated
   :recursive:

   jem.driver
   jem.output
   jem.checkpoint
   jem.accumulate
   jem.adjoint
   jem.regrid
   jem.config
   jem.runners
   jem.configurations
   jem.main

Components
----------

.. autosummary::
   :toctree: generated
   :recursive:

   jem.components.clock
   jem.components.coordinates
   jem.components.jcm.component
   jem.components.jcm.contract
   jem.components.jcm.exchange_fields
   jem.components.jcm.forcing_window
   jem.components.veros_component
   jem.components.veros.setups.double_drake
   jem.components.veros.setups.earth
   jem.components.veros.setups.global_latlon

   jem.components.slab.slab_ocean_model
   jem.components.slab.slab_seaice_model
   jem.components.slab.slab_atmosphere_model
   jem.components.slab.bucket_land_model
   jem.components.slab.winton_seaice_model
   jem.components.slab.winton_seaice_model.winton_seaice_model
   jem.components.slab.winton_seaice_model.ice_transport
   jem.components.slab.grid
   jem.components.slab.base

Packages
--------

The package docstrings, which say where each family of modules above sits.
A name a package re-exports is documented on the page of the module that
defines it.

.. autosummary::
   :toctree: generated
   :template: autosummary/package.rst

   jem.base
   jem.components
   jem.components.jcm
   jem.components.slab
   jem.components.veros
   jem.components.veros.setups
   jem.data
   jem.tools
   jem.utils

Constants and utilities
-----------------------

.. autosummary::
   :toctree: generated
   :recursive:

   jem.constants
   jem.fluxes
   jem.plot
   jem.tools.idealised_terrain
   jem.tools.ocean_bathymetry
   jem.tools.ocean_initial_state
   jem.tools.veros_earth_terrain
   jem.tools.veros_earth_ice_mask
   jem.utils.esmf_regrid
   jem.utils.latlon_weights
   jem.utils.cycles
   jem.utils.idealized_distribution
