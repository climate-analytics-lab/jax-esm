Known limitations
=================

What JAX-ESM 1.0 does not do, and what that means for a run. Each item names
the code that enforces or documents it, so it can be checked and, when it is
lifted, removed.

Veros ocean
-----------

**The sea ice under a Veros ocean is a demonstration, with limits.**
    A Winton sea ice is coupled to Veros in ``+configuration=veros-earth``
    (``jem.fluxes.VerosExchange``, ``VerosComponent(clamp_at_freezing=True)``;
    jax-esm#160). Consequences, each enforced or documented where named:

    - the clamp holds the **top cell only** at the freezing point; sub-freezing
      water below it is not frazil ice and is clamped only when mixed to the
      surface (:doc:`design/winton_seaice`);
    - the freezing point is the constant -1.8 degC the slab ocean and the ice use,
      not Veros' salinity-dependent one;
    - the ocean receives the atmosphere's wind stress over the sea tile whether or
      not ice covers it (no ice-ocean stress), and the ice does not drift unless
      built with transport;
    - ice growth is a virtual salt flux on the top cell (the ice carries no salt in
      the water budget);
    - the coupling is lagged by one step, so ocean plus ice conserve energy
      exactly only with the energy in transit counted
      (``tests/unit/test_veros_seaice.py``);
    - the ice's grid must be the ocean's and its sea exactly Veros' wet mask
      (``VerosExchange.check_ice_mask``); ``veros-earth`` uses the binary
      mask of ``jem.tools.veros_earth_ice_mask``, which duplicates the
      thresholding the ocean setup does;
    - the ice starts from nothing over the observed annual-mean ocean and has
      been run for 90 days (it grows to 7.6% of the northern ocean, with a
      closed energy budget); the **equilibrated seasonal cycle of the ice has
      not been validated**, which needs years of integration;
    - ``veros-double-drake`` sets ``seaice=none`` (no ice, the swamp mask), and
      a Winton ice composed without the ocean's clamp, or an SST-diagnosed ice
      over a clamping ocean, is refused when the coupling is built.

**``veros-earth`` is a prototype ocean.**
    Its setup reads a binary land-sea mask and a packaged ETOPO 2022 bathymetry
    averaged onto the grid's cells (see :doc:`experimental`), so the bottom is
    real but at the grid's resolution of about 3.75 degrees, with a 120 m
    minimum depth. It starts from the World Ocean Atlas 2018 annual-mean
    temperature and salinity (see :doc:`experimental`) with the currents at
    rest, with the Winton sea ice growing from nothing. Consequence: the starting state is a
    climatological mean that this model has not been spun up to, so a short run
    is the atmosphere's fluxes and the model's own adjustment acting on it, not
    an equilibrated climate. ``jem.components.veros.setups.global_latlon``
    (one-degree Levitus bathymetry and initial state) is the setup to use when
    the ocean's realism matters.

**``veros-earth``'s land data describe the original coastline.**
    The atmosphere's land fraction is derived from the ocean's wet mask
    (``terrain_veros_earth_T31.nc``; :doc:`design/exchange`), so the two
    components agree on where the sea is, but the albedo, snow, soil water and
    land temperature it reads from jax-gcm's ``bc/t30/clim/forcing.nc`` are
    for the old coast. About 0.2 sr of the atmosphere's cells gain land
    fraction where that file holds mixed or ocean-like values, and the
    orography of cells that turn to sea is kept. Consequence: a small, coastal
    error in the land surface (about 0.5 W m-2 of absorbed shortwave in the
    global mean if all of it were wrong, 1.6 % of the sphere's area); the
    mismatch between the heat the atmosphere gives up and the ocean receives
    is about 0.3 W m-2 of ocean, not zero. ``veros-double-drake`` uses one mask
    on one grid and is exact.

**A Veros run is double precision for the whole process.**
    Veros runs in float64 and switches JAX's process-wide ``jax_enable_x64`` on,
    so ``ocean=veros`` selects ``jax_enable_x64=true`` and the runner applies it
    before any component is built: the atmosphere and every other component run
    in float64 too (a handful of leaves jax-gcm builds as float32 whatever the
    flag excepted). Consequence: a Veros run costs more than the same
    atmosphere alone in float32 (a few percent to a quarter more wall time, and
    about 9 % more memory for a slab run, in a measurement on a loaded host;
    :doc:`experimental`), and it cannot share a process with a float32 run
    -- the flag outlives the run, including a
    :func:`jem.configurations.load` of a Veros configuration. Tests that build
    Veros run in their own process, and restore the flag, for the same reason.
    Asking for ``jax_enable_x64=false`` with Veros is refused when the Veros
    operators are first imported in the process; in a process where they
    already were, it is not detected and Veros' float64 arrays are held in
    float32. A Python user who builds the atmosphere in float32 and then calls
    ``VerosComponent.from_setup``, without the runner, gets an atmosphere
    allocated in float32 and stepped in float64; they must set the flag before
    building the atmosphere.

**The Veros adjoint is spiky without an unmerged fix.**
    See :doc:`design/differentiating_long_runs`: the smoothing of the
    Prandtl-number switch is on an open Veros pull request, not in the
    fork a user installs.

Surface components and the atmosphere
-------------------------------------

**The Winton sea ice and the bucket land need SPEEDY physics.**
    ``default_exchangers`` and ``VerosExchange`` raise for the Winton ice
    under an atmosphere that publishes no sea flux tile and for the bucket
    land under one that publishes no land tile; ECHAM delivers its fluxes
    untiled and integrates its own sea ice
    (``jem.exchangers.check_winton_coupling`` and ``check_land_coupling``).
    Consequence: of the slab
    components, only the ocean and the SST-diagnosed slab ice can be coupled
    to ECHAM, and the ocean's heat flux there is the grid-cell mean, so a
    coastal ocean cell also receives the land part's flux
    (:doc:`design/exchange`).

**The calibrated SPEEDY of the named configurations was fitted with prescribed SSTs.**
    The six SPEEDY configurations (all but the two ``-echam``) run jax-gcm's
    coupled-calibrated SPEEDY parameters
    (its ``speedy-t31`` bundle, at the 30-minute step). That fit was made at T31
    against observations with the SST prescribed, not coupled, so a run in which
    the SST responds is not guaranteed to stay at the observed climate: the
    top-of-atmosphere imbalance and the surface fluxes it hands the ocean are
    credible, not drift-free (jax-gcm's ``docs/source/design/speedy_retune.md``
    records the biases that remain). The fit belongs to SPEEDY at T31 and does
    not carry to another grid, such as the T106 command in :doc:`examples`.

**A coupling timestep is a whole number of seconds.**
    The coupler's clock is a ``jax_datetime`` one, which counts whole
    seconds, so every coupled timestep and every sub-timestep of a nested or
    repeated element must be a whole number of seconds
    (:doc:`design/nesting`); one second is the shortest step there is.

Runs, output and checkpoints
----------------------------

**The in-scan monthly-mean accumulator is not checkpointed.**
    ``run_chunked(..., accumulate=monthly_mean(...))`` resumes the model from
    its checkpoint but starts a fresh accumulator, so a mean across a resume
    covers only what the resumed call integrated (``jem.driver.run_chunked``'s
    Notes, :doc:`design/running`). Consequence: a requeued long run must
    finalize each call's accumulator and combine them, or run the whole span in
    one call. The command line's ``output_averages`` is a different thing, a
    30-day *window* mean of the written files, and is unaffected.

Installation
------------

**jax-gcm has to be installed from source at a pinned revision.**
    JAX-ESM is tested against one jax-gcm revision
    (``JCM_SUPPORTED_REV`` in :mod:`jem.components.jcm.contract`), which is
    ahead of the newest release on PyPI (``jcm`` 3.0.0, which JAX-ESM cannot
    run against). JAX-ESM requires ``jcm>=3.1.0``, so until jax-gcm 3.1.0 is
    released ``pip install jax-esm`` stops with "no matching distribution for
    jcm>=3.1.0": clone jax-gcm, check out the revision, install it, then
    install JAX-ESM (:doc:`getting_started`). Only the pinned revision is
    supported, since several unreleased revisions report 3.1.0.

**``jax`` is capped below 0.11.2.**
    jax 0.11.2 removed ``jax.experimental.hijax.HiPrimitive``, which every
    released ``flax`` subclasses at import time, so ``import jcm`` fails. The
    ceiling is flax's, not JAX-ESM's, and stays until a flax release carries
    the compatibility alias (jax-esm#117). Consequence: JAX-ESM cannot be
    installed beside a newer JAX.
