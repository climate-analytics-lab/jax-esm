API Reference
=============

The coupling core
-----------------

.. autosummary::
   :toctree: generated
   :recursive:

   jem.base.component
   jem.base.coupler.Coupler
   jem.exchangers

Running a model
---------------

``jem.driver`` is the run loop a Python user calls; ``jem.runners`` and
``jem.main`` are the same loop reached from a composed Hydra config.
``jem.checkpoint`` and ``jem.accumulate`` are module-level APIs -- they are not
re-exported from ``jem``, because a run reaches them through
``Coupler.save_state`` and ``generate_trajectory_function(accumulate=...)``.

.. autosummary::
   :toctree: generated
   :recursive:

   jem.driver
   jem.output
   jem.checkpoint
   jem.accumulate
   jem.regrid
   jem.config
   jem.runners
   jem.main

Components
----------

.. autosummary::
   :toctree: generated
   :recursive:

   jem.components.clock
   jem.components.jcm.component
   jem.components.jcm.contract
   jem.components.jcm.exchange_fields
   jem.components.veros_component

   jem.components.slab.slab_ocean_model.SlabOceanModel
   jem.components.slab.slab_ocean_model.SlabOceanParameters
   jem.components.slab.slab_land_model.SlabLandModel
   jem.components.slab.slab_land_model.SlabLandParameters
   jem.components.slab.slab_seaice_model.SlabSeaiceModel
   jem.components.slab.slab_seaice_model.SlabSeaiceParameters
   jem.components.slab.slab_atmosphere_model.SlabAtmosphereModel
   jem.components.slab.slab_atmosphere_model.SlabAtmosphereParameters

   jem.components.slab.grid.SlabGrid
   jem.components.slab.base.SlabModelBase

Constants and utilities
-----------------------

.. autosummary::
   :toctree: generated
   :recursive:

   jem.constants
   jem.utils.esmf_regrid.ESMFRegridder
   jem.utils.esmf_regrid.ESMFWeights
