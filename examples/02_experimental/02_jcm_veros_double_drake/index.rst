JCM-Veros Coupled Run: Double-Drake Configuration
==================================================

.. note::

   This example is experimental: it is not tested as extensively as the
   examples above and serves as a prototype for future development.

This example couples JCM and Veros in a double-drake configuration: two
meridional continents on an otherwise aquaplanet-like Gaussian lat-lon grid.
Because JCM and Veros share exactly the same grid here, the coupling needs no
regridding or vector rotation. "Land" is a :code:`SlabOceanModel` registered
as :code:`fakelnd` with :code:`ocean_mask_value=1.0`, so it integrates the
cells the grid marks as land. Its grid carries no fractional mask, so it in
fact integrates none of them and reports a constant surface temperature: a
fixed land boundary condition for the atmosphere, and a stand-in until a
proper slab land model is wired into this example in a future release.

Running the example
--------------------

Run :code:`run.sh` to produce 60 days of daily-averaged output:

.. literalinclude:: run.sh
   :language: bash

:code:`run.sh` is a thin wrapper around :code:`main.py`; it can be submitted
as-is to an HPC scheduler, or copied and edited to change the run length,
timesteps, or output naming.

Command-line options
---------------------

:code:`main.py` accepts the following options (see :code:`main.py --help` for
the authoritative list):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - :code:`--total-simulation-days`
     - 10
     - Total time of simulation in days.
   * - :code:`--simulation-interval-days`
     - 5
     - Simulation interval in days; the model runs and checkpoints in chunks
       of this length.
   * - :code:`--simulation-name`
     - :code:`default`
     - Name used for the output subdirectory.
   * - :code:`--truncation-number`
     - 31
     - Spectral truncation number (T-number) for JCM.
   * - :code:`--jcm-timestep-min`
     - 30
     - JCM timestep, in minutes.
   * - :code:`--veros-timestep-min`
     - 30
     - Veros timestep, in minutes.
   * - :code:`--terrain-planet-type`
     - *(required)*
     - Terrain preset to generate; :code:`run.sh` uses :code:`double_drake`.
   * - :code:`--do-not-average-time`
     - off
     - Write raw per-step output instead of averaging over each interval.
   * - :code:`--max-rerun-attempts`
     - 0
     - If the model explodes, rerun up to this many times (stochastic
       forcing may avoid the instability on a retry).
   * - :code:`--explode-log`
     - :code:`explode.log`
     - Path (relative to the output directory) to log model-explosion
       events.
   * - :code:`--debug-mode`
     - off
     - Detect NaNs and drop into a breakpoint when they occur.

Output is written under :code:`output_T{truncation_number}/{simulation_name}/`,
relative to the working directory :code:`run.sh` was launched from.

Files in this example
-----------------------

- :code:`run.sh`: bash-side entry point, suitable for HPC job submission.
- :code:`main.py`: python-side entry point invoked by :code:`run.sh`; parses
  the options above, builds the coupled model, runs it interval by interval,
  and writes output/checkpoints.
- :code:`model_setup.py`: builds the coupled JCM + Veros + :code:`SlabOceanModel`
  system via :code:`jem.base.coupler.Coupler`, so that :code:`main.py` and any
  other script (e.g. a :code:`jax.grad` sensitivity experiment) construct
  exactly the same model.
- :code:`modify_jcm_terrain.py`: generates the double-drake terrain used by
  JCM from a reference terrain file.
- :code:`veros_case_setup.py`: Veros-side configuration for this case.
- :code:`veros_helper.py`: small Veros-side utilities shared by
  :code:`model_setup.py` and :code:`veros_case_setup.py`.
