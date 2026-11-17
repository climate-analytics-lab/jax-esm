"""The jax-gcm revision JAX-ESM is supported against, and the names it calls.

JAX-ESM (``jem``) is built on jax-gcm (``jcm``) but lives in its own
repository and is installed against a source checkout of jax-gcm rather than a
PyPI release. Without a recorded pin, "which jax-gcm does this work with?" has
no answer, and a rename on the jax-gcm side shows up as an ``AttributeError``
or a ``KeyError`` deep inside a coupled run instead of at import time.

This module is the single place that records both halves of that contract:

``JCM_SUPPORTED_REV`` / ``JCM_SUPPORTED_VERSION``
    the jax-gcm revision every JAX-ESM gate runs against;
``JCM_INTEGRATION_POINTS``
    every jax-gcm name JAX-ESM reaches for, one entry per name, with what it
    is used for.

``tests/unit/test_jcm_contract.py`` walks the second list against the
installed ``jcm`` and fails with a message naming the missing name *and* the
supported revision, so a jax-gcm rename is reported as "jax-gcm renamed or
removed X" rather than as a mid-run crash. It also reads every ``import`` of
``jcm`` under ``jem/`` from the source and fails naming any that no entry
covers, so the list cannot fall behind the code, and it asserts that the
revision the CI workflow checks out is this one, so the pin and the workflow
cannot drift apart.

Why a ``dev`` revision rather than a release
--------------------------------------------
``JCM_SUPPORTED_REV`` is ``9764c56b36cd71b55c644aeb93cb787eb8c4db97``. A sha is
the most precise thing there is to name; ``pyproject.toml``'s ``jcm`` floor,
:data:`JCM_SUPPORTED_VERSION`, is the loosest true statement of the same
fact, since jax-gcm's version string is bumped only at release. It is still
load-bearing: it is what stops ``pip`` resolving to an older jcm release on
PyPI that this package cannot import against.

The policy, stated once here: the pin is a commit on jax-gcm's ``dev`` branch
(a merge commit there is fine: it is a point on ``dev`` that jax-gcm
merged as a whole) that this repository's own gates -- the
required CI jobs, which check out exactly ``JCM_SUPPORTED_REV`` -- have been
run against and pass on. A commit is chosen because those gates pass at it,
not because a pull request that JAX-ESM wanted merged there. The two things
the policy rules out are a sha that is not on ``dev`` (a pull-request branch
head, which may never be merged or may be rewritten) and ``dev``'s tip taken
at bump time without running the gates at it, which can carry commits this
repository has never been run against. When a tagged jax-gcm release
finally contains this revision, replace the sha with the tag and raise the
``pyproject.toml`` floor to match.

The current pin is a ``dev`` merge commit -- the merge of jax-gcm#1213 -- and the
required CI jobs run at exactly it, so it satisfies the policy. It publishes
SPEEDY's land and sea flux tiles and a coupled sea ice's forcing on
``SurfaceExchange``. It carries jax-gcm#1199 (SPEEDY's own slab land,
``land_model.f90``, as ``SpeedySurfaceFlux``'s default) and jax-gcm#1205
(``ForcingData.sea_ice_temperature``) with it: those are what this
repository's ocean coupling reads (the ``sea`` tile, jax-esm#147) and what
its land is (SPEEDY's own slab; JAX-ESM integrates no land temperature of its
own for the atmosphere). It also relies on: ECHAM physics tracing
under ``jax_enable_x64`` (jax-gcm#946), so an ECHAM atmosphere couples to
Veros, and ``SurfaceExchange`` publishes a near-surface wind vector,
``wind_u``/``wind_v``, identically from every physics package
(jax-gcm#911/#914); see :mod:`jem.components.jcm.exchange_fields` for the
translation and ``docs/source/design/jcm_adapter.md`` for the full adapter
design.

Current behaviour a JEM user must know:

* ``jcm.forcing.resolve_align`` is the one rule for the time alignment of an
  atmosphere boundary condition (climatology vs. transient). An explicit
  ``wrap_year``, ``by_date`` or ``by_date_interp`` is taken as given.
  ``auto`` (the default) resolves only a data-mirror or packaged product,
  whose alignment the mirror manifest records; for any other file -- a
  user's own, or an in-memory dataset -- ``auto`` raises ``ValueError``
  rather than guess, so such a forcing needs ``atmosphere.forcing.align``
  set explicitly. The configurations in ``jem/config/configuration/`` that
  read a user-supplied forcing file say which value they use and why.
* jax-gcm configures no logging of its own. ``jem.main`` sets the ``jem``
  logger's level from ``coupled_run.log_level`` and nothing sets the ``jcm``
  logger's: jax-gcm applies its ``run.log_level`` knob only in its own CLI
  entry point (``jcm.runners.run``), which JAX-ESM does not call, so
  jax-gcm's messages follow the root logger Hydra configures for the job.
* ``JCMComponent`` threads jax-gcm's own :class:`~jcm.model.RunState`
  (``dynamics``, ``physics``, ``time``, ``step``) through
  ``carry["state"]``, ``carry["physics"]``, ``carry["time"]`` and
  ``carry["step"]`` and ``Model.run_from_state_with_carry(initial_state=,
  initial_physics_state=, initial_time=, initial_step=)`` every coupled
  step; the ``time`` and ``step`` are jax-gcm's own clock, kept separately
  from the coupler's.

How to bump the pin
-------------------
1. Change ``JCM_SUPPORTED_REV`` (and ``JCM_SUPPORTED_VERSION``, if jax-gcm's
   version string moved) here, and update the "why" paragraph above to say what
   the new revision brings.
2. Change ``JCM_REV`` in ``.github/workflows/tests.yml`` and in the
   ``Dockerfile`` to the same sha, the sha in the install instructions
   (``README.md``, ``docs/source/getting_started.rst``,
   ``docs/source/developers.rst``), and the ``jcm`` floor in
   ``pyproject.toml`` if the version moved -- ``test_jcm_contract.py`` fails
   on each one that disagrees.
3. Run ``pytest tests/unit/test_jcm_contract.py``; any integration point the
   new revision renamed or removed fails there, with the name, before it can
   fail inside a run. Fix the adapter and update the entry in the same change.
"""

from __future__ import annotations

from typing import NamedTuple

#: The jax-gcm revision JAX-ESM is developed, tested and supported against.
#: The full 40-character sha, not an abbreviation, because that is what
#: ``actions/checkout`` needs and what ``git rev-parse`` in a jax-gcm checkout
#: can be compared against directly.
#:
#: This ``dev`` commit is validated as a whole against this repository's own
#: gates -- see "Why a ``dev`` revision rather than a release" above.
JCM_SUPPORTED_REV = "9764c56b36cd71b55c644aeb93cb787eb8c4db97"

#: The version string ``jcm`` reports at :data:`JCM_SUPPORTED_REV`. jax-gcm's
#: version is only bumped at release, so it is a weaker statement than the sha
#: -- many revisions share it -- but it is what a user's environment can be
#: checked against without a git checkout, which is why
#: ``test_installed_jcm_matches_contract`` reads ``jcm.__version__`` rather
#: than the distribution metadata: an editable install records its version
#: when it is installed, so the metadata of a checkout that has since moved
#: to another revision is stale, which is exactly the situation a pin bump
#: creates.
JCM_SUPPORTED_VERSION = "3.1.0"


class IntegrationPoint(NamedTuple):
    """One jax-gcm name JAX-ESM depends on.

    Attributes
    ----------
    target : str
        Dotted path of the thing that owns the name: a module
        (``"jcm.runners"``), a class (``"jcm.model.Model"``), or -- for
        ``access="diagnostics"`` -- the physics package whose diagnostics dict
        carries the key (``"speedy"``).
    attribute : str
        The name itself. For ``access="diagnostics"`` it is a dotted
        ``key.field`` path into the diagnostics dict; for
        ``access="package data"`` it is a path relative to the package root.
    access : str
        ``"public"``, ``"private"`` (followed, where a jax-gcm issue tracks
        the missing public API, by the issue -- ``"private,
        TODO(jax-gcm#123)"``), ``"diagnostics"`` or ``"package data"``. The
        contract test dispatches on the first word, so the trailing note is
        free-form.
    used_for : str
        Why JAX-ESM needs it. Read this before deleting an entry: if nothing
        described here is still true, the entry goes.

    """

    target: str
    attribute: str
    access: str
    used_for: str


#: Every jax-gcm name JAX-ESM reaches for, checked by
#: ``tests/unit/test_jcm_contract.py`` against the installed ``jcm``.
#:
#: The list covers three kinds of dependency, because all three break a user's
#: run when jax-gcm renames something: names JAX-ESM *calls*; diagnostics-dict
#: keys and struct fields it *reads*; and the constructors JAX-ESM's public
#: workflow, quick start and tests tell a user to call to build the objects
#: they hand to JAX-ESM. Names JAX-ESM merely mentions in prose are not here.
#:
#: ``jcm.__version__`` is listed as well. It is not part of anybody's run, but
#: it is what ``test_installed_jcm_matches_contract`` compares against
#: :data:`JCM_SUPPORTED_VERSION`, so a jax-gcm that stopped defining it should
#: fail as a named contract point rather than as an ``AttributeError`` inside
#: the test.
JCM_INTEGRATION_POINTS: tuple[IntegrationPoint, ...] = (
    # ------------------------------------------------------------------
    # The package itself: how the contract test identifies what is installed.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm", "__version__", "public",
        "The version string the contract test compares with"
        " JCM_SUPPORTED_VERSION; read from the module rather than"
        " importlib.metadata because an editable install's metadata is frozen"
        " at install time while the attribute tracks the checkout.",
    ),
    # ------------------------------------------------------------------
    # Driver (jem.runners, jem.driver, jem.accumulate): building a model and
    # a forcing from a composed Hydra config, and running it in health-gated
    # chunks.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm.runners", "build_model", "public",
        "Build the atmosphere Model from the composed `atmosphere` config"
        " subtree (jem.runners.build_atmosphere), so JAX-ESM never"
        " reimplements jax-gcm's own wiring."
        " Signature: build_model(cfg: DictConfig) -> Model.",
    ),
    IntegrationPoint(
        "jcm.runners", "build_forcing", "public",
        "Build the atmosphere's ForcingData from the same config subtree and"
        " the built coordinates (jem.runners.build_atmosphere)."
        " Signature: build_forcing(cfg: DictConfig, coords, dycore=None).",
    ),
    IntegrationPoint(
        "jcm.runners", "apply_constants_overrides", "public",
        "Apply `+atmosphere.constants.<name>=<value>` to the process-global"
        " jcm.constants singleton before the atmosphere is built, exactly as"
        " jax-gcm's own CLI does -- the dynamical core reads the live"
        " singleton at construction, and JAX-ESM's surface components read the"
        " same one. Signature: apply_constants_overrides(cfg: DictConfig)"
        " -> None.",
    ),
    IntegrationPoint(
        "jcm.runners", "warn_on_config_traps", "public",
        "Warn about known jax-gcm configuration traps before a coupled run"
        " starts, so a JAX-ESM user gets the same diagnostics a jax-gcm user"
        " does. Signature: warn_on_config_traps(cfg: DictConfig, physics,"
        " forcing, coords=None, dycore=None) -> None.",
    ),
    IntegrationPoint(
        "jcm.diagnostics", "check_health", "public",
        "Per-chunk health gate on the atmosphere's output dataset"
        " (jem.driver.default_health_check), so a run that has gone unstable"
        " stops instead of burning the queue."
        " Signature: check_health(ds, chunk_idx: int, elapsed_days: float)"
        " -> tuple[bool, dict].",
    ),
    IntegrationPoint(
        "jcm.date", "parse_duration_seconds", "public",
        "Turn a run length, chunk, coupling interval or averaging window"
        " ('12 hours', '10 days', or a number of days) into whole seconds"
        " (jem.driver.run_length_seconds and its whole-step helper,"
        " jem.accumulate); refuses 'months'/'years' and fractional seconds."
        " Signature: parse_duration_seconds(value) -> int.",
    ),
    IntegrationPoint(
        "jcm.date", "to_datetime", "public",
        "Parse a run's `end_time` (and the coupler's start date) to a"
        " datetime, so jem.driver.run_length_seconds subtracts the two as"
        " datetimes, exact to the second, rather than as float days."
        " Signature: to_datetime(value, *, name='time') -> jax_datetime.Datetime.",
    ),
    IntegrationPoint(
        "jcm.date", "fraction_of_year_elapsed", "public",
        "Position in the annual cycle in [0, 1), on the proleptic Gregorian"
        " calendar. CouplingTime.year_fraction (jem.base.component) calls"
        " this directly, and so does the slab models' base class"
        " (jem.components.slab.base), so a component's seasonal cycle and the"
        " atmosphere's agree by construction. Signature: fraction_of_year_elapsed(dt:"
        " jax_datetime.Datetime) -> jax.Array.",
    ),
    IntegrationPoint(
        "jcm.date", "gregorian_ymd_from_days", "public",
        "Exact Gregorian (year, month, day) from days since the 1970 epoch;"
        " jem.accumulate.monthly_mean bins each record's own interval"
        " midpoint by the month this returns."
        " Signature: gregorian_ymd_from_days(days_since_epoch) -> (year,"
        " month, day).",
    ),
    # ------------------------------------------------------------------
    # The wrapped model (jem.components.jcm.component).
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm.model", "Model", "public",
        "The object JCMComponent wraps (jem.components.jcm.component, and"
        " the deprecated make_jem_compatible shim in"
        " jem.components.jcm_component).",
    ),
    IntegrationPoint(
        "jcm.model", "RunState", "public",
        "JCM's own resumable struct (dynamics, physics, time, step), the first"
        " element of what run_from_state_with_carry returns. JCMComponent.step"
        " stores all four fields in the atmosphere's carry and threads"
        " 'time'/'step' back into run_from_state_with_carry every coupled"
        " step, exactly as jcm's own chunked runner does.",
    ),
    IntegrationPoint(
        "jcm.model.RunState", "dynamics", "public",
        "The dycore state a step returns: carry['state'].",
    ),
    IntegrationPoint(
        "jcm.model.RunState", "physics", "public",
        "The cross-step physics carry a step returns: carry['physics'].",
    ),
    IntegrationPoint(
        "jcm.model.RunState", "time", "public",
        "jax-gcm's own clock after a step: carry['time'].",
    ),
    IntegrationPoint(
        "jcm.model.RunState", "step", "public",
        "jax-gcm's own step counter after a step: carry['step'].",
    ),
    IntegrationPoint(
        "jcm.model.Model", "bootstrap_state", "public",
        "Build the initial dycore state and physics carry without integrating"
        " a step, which is what lets JCMComponent.initialize() produce a carry"
        " of the exact structure step 1 will return. Returns the"
        " `(dycore_state, physics_carry)` pair directly (jax-gcm#824, which"
        " closed jax-gcm#755), so JAX-ESM unpacks the return value; the same"
        " two objects are also installed on the model as `Model.dycore_state`"
        " / `Model.physics_carry`, which JAX-ESM deliberately does not read,"
        " so neither name is watched here.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "run_from_state_with_carry", "public",
        "Advance the atmosphere by exactly one coupling interval, threading"
        " the cross-step physics carry and jcm's own RunState clock"
        " (initial_time=, initial_step=); the whole coupled step is this"
        " call. Returns (RunState, ModelPredictions).",
    ),
    IntegrationPoint(
        "jcm.model.Model", "dt_si", "public",
        "The model's own timestep, checked against the coupler's coupling"
        " timestep in JCMComponent.bind().",
    ),
    IntegrationPoint(
        "jcm.model.Model", "start_time", "public",
        "Checked against the coupler's start date in JCMComponent.bind() so a"
        " mismatch is refused up front rather than drifting silently.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "coords", "public",
        "The nodal shape every exchanged field is on (JCMComponent), the grid"
        " the default forcing is built on, and the horizontal grid"
        " jem.runners gives a surface component's SlabGrid when it shares the"
        " atmosphere's grid.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "terrain", "public",
        "The atmosphere's land-sea mask (`terrain.fmask`): JCMComponent's"
        " sea_fraction, and the mask jem.runners builds a surface component's"
        " SlabGrid from when that component shares the atmosphere's grid.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "physics", "public",
        "Source of the diagnostics template (see get_empty_data), and the"
        " owner of require_surface_exchange (jem.runners.build_atmosphere).",
    ),
    IntegrationPoint(
        "jcm.model.Model", "dycore", "public",
        "The dynamical core: the owner of the state's simulation clock (see"
        " DynamicalCore.sim_time) and of to_physics_state."
        " ModelPredictions.with_context(model) reads it too.",
    ),
    IntegrationPoint(
        "jcm.dycore.base.DynamicalCore", "to_physics_state", "public",
        "Project the dycore state to a gridpoint PhysicsState:"
        " JCMComponent._initial_surface_state reads the lowest level's air off"
        " it to seed the derived fields before the first step."
        " Signature: to_physics_state(state) -> PhysicsState.",
    ),
    IntegrationPoint(
        "jcm.dycore.base.DynamicalCore", "sim_time", "public",
        "Read the dycore state's native elapsed-seconds counter, which"
        " JCMComponent.restart_clock zeroes when a state is reused under a"
        " new start date. Returns None for a dycore with no such counter.",
    ),
    IntegrationPoint(
        "jcm.dycore.base.DynamicalCore", "with_sim_time", "public",
        "Replace the dycore state's native elapsed-seconds counter;"
        " JCMComponent.restart_clock uses it to put the counter back at zero,"
        " the way jax-gcm's own as_initial_condition import and"
        " Model._at_fresh_clock do.",
    ),
    IntegrationPoint(
        "jcm.physics_interface.Physics", "get_empty_data", "public",
        "Zero template of one step's physics diagnostics dict, used to seed"
        " JCMDerived with the exact structure, shapes and dtypes step 1"
        " produces -- without integrating a step to find out.",
    ),
    IntegrationPoint(
        "jcm.forcing", "ForcingData", "public",
        "Type of the `forcing` entry of JCMComponent's carry, which"
        " exchangers overwrite (the SST an ocean component computes) every"
        " coupling step.",
    ),
    IntegrationPoint(
        "jcm.forcing", "TimeSeries", "public",
        "The time-varying forcing leaf a from-file boundary condition is"
        " built as. Whether a ForcingData field IS one decides the pytree"
        " structure of the atmosphere's carry, which is what"
        " JCMComponent.initialize() has to settle before a coupled run can"
        " scan; restrict_forcing_to_window reads its values/times/align_mode/"
        "persist fields to cut it down to a run's window.",
    ),
    IntegrationPoint(
        "jcm.forcing", "make_time_series", "public",
        "Build the restricted, by-date TimeSeries in"
        " jem.components.jcm.forcing_window. Signature: make_time_series("
        "values, times, align_mode=BY_DATE, persist='strict') -> TimeSeries.",
    ),
    IntegrationPoint(
        "jcm.forcing", "WRAP_YEAR", "public",
        "The align_mode code of a climatology replayed every year, which"
        " restrict_forcing_to_window re-expresses by date.",
    ),
    IntegrationPoint(
        "jcm.forcing", "BY_DATE", "public",
        "The align_mode code of a step-held dated series.",
    ),
    IntegrationPoint(
        "jcm.forcing", "BY_DATE_INTERP", "public",
        "The align_mode code of a linearly interpolated dated series.",
    ),
    IntegrationPoint(
        "jcm.initial_states", "era5_state", "public",
        "The ERA5 initial condition behind init=era5"
        " (jem.runners.build_initial_state). Lazily re-exported from"
        " jcm.data.era5.initial_state. Signature: (coords, date: str) ->"
        " PhysicsState.",
    ),
    IntegrationPoint(
        "jcm.initial_states", "jw_state", "public",
        "The Jablonowski-Williamson initial condition behind init=jw."
        " Signature: jw_state(model, rh=0.6).",
    ),
    IntegrationPoint(
        "jcm.initial_states", "balanced_isothermal_state", "public",
        "The orography-balanced isothermal initial condition behind"
        " init=balanced_isothermal. Signature: (model) -> PhysicsState.",
    ),
    IntegrationPoint(
        "jcm.initial_states", "checkpoint_state", "public",
        "The warm start behind init=from_state, returning the donor's"
        " physics carry with its state. Signature: checkpoint_state(model,"
        " path, *, unstamped_scale=None) -> (state, physics_carry, days).",
    ),
    IntegrationPoint(
        "jcm.checkpoint", "parse_unstamped_scale", "public",
        "Parse init.unstamped_scale for checkpoint_state, as jax-gcm's CLI"
        " does.",
    ),
    IntegrationPoint(
        "jcm.data.remote", "fetch", "public",
        "Resolve an hf:// data-mirror path to a local file, as jax-gcm's own"
        " CLI does: atmosphere.init.file, and any file a surface component"
        " is handed by that name (a slab ocean's sst_clim_file set to"
        " ${atmosphere.forcing.file}).",
    ),
    IntegrationPoint(
        "jcm.forcing.ForcingData", "sea_surface_temperature", "public",
        "The prescribed SST an ocean overwrites every coupling step: the"
        " target of the standard exchange tables (jem.exchangers), and the"
        " field JCMComponent.set_exchanged_forcing validates against.",
    ),
    IntegrationPoint(
        "jcm.forcing.ForcingData", "sice_am", "public",
        "The ice fraction of the sea part of a cell, which a sea ice writes"
        " (jem.exchangers' standard tables).",
    ),
    IntegrationPoint(
        "jcm.forcing.ForcingData", "soilw_am", "public",
        "SPEEDY's soil-water availability, which a land component writes"
        " (jem.exchangers' standard tables).",
    ),
    IntegrationPoint(
        "jcm.forcing.ForcingData", "select", "public",
        "Collapse every TimeSeries leaf to one date's slice."
        " JCMComponent.initialize() takes the fields the coupling supplies"
        " from the start-date slice, so they are the plain arrays an"
        " exchanger writes rather than time series."
        " Signature: select(date: DateData) -> ForcingData.",
    ),
    IntegrationPoint(
        "jcm.date", "DateData", "public",
        "The per-step date object ForcingData.select takes; JCMComponent"
        " builds one for the run's start date with DateData.set_date.",
    ),
    IntegrationPoint(
        "jcm.date.DateData", "set_date", "public",
        "Build a DateData at a given jax_datetime.Datetime."
        " Signature: set_date(model_time, model_step=None,"
        " dt_seconds=None) -> DateData.",
    ),
    IntegrationPoint(
        "jcm.forcing", "default_forcing", "public",
        "Default prescribed-SST boundary conditions when a JCMComponent is"
        " built without an explicit forcing.",
    ),
    IntegrationPoint(
        "jcm.predictions", "ModelPredictions", "public",
        "The per-step diagnostics object JCMComponent returns whole, so the"
        " coupler can stack it and hand it back to jax-gcm's serialization.",
    ),
    IntegrationPoint(
        "jcm.predictions.ModelPredictions", "physics", "public",
        "The physics diagnostics dict the surface exchange is read out of.",
    ),
    IntegrationPoint(
        "jcm.predictions.ModelPredictions", "times", "public",
        "Record count check in JCMComponent.to_xarray: the coupler's time axis"
        " must have as many entries as jax-gcm produced records.",
    ),
    IntegrationPoint(
        "jcm.predictions.ModelPredictions", "to_xarray", "public",
        "Serialization of the atmosphere's output; JAX-ESM deliberately does"
        " not reimplement jax-gcm's CF metadata.",
    ),
    IntegrationPoint(
        "jcm.predictions.ModelPredictions", "with_context", "public",
        "Re-attaches the coords/physics/dycore a JAX pytree round trip drops,"
        " so a ModelPredictions stacked by JAX-ESM's lax.scan can serialize"
        " itself again. Public since jax-gcm#824 (closing jax-gcm#756), so"
        " JAX-ESM reads no private payload of ModelPredictions.",
    ),
    # ------------------------------------------------------------------
    # Private names. One is left: JAX-ESM copies its *values* rather than
    # importing it, so there is nothing for jax-gcm to make public and no
    # issue to track -- but a jax-gcm that stopped defining it would leave
    # the copies describing nothing, so it is watched here. An entry that
    # does have a jax-gcm issue behind it names that issue in `access`
    # ("private, TODO(jax-gcm#123)"), and so does the code that reads it.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm.cf_metadata", "_COORD_ATTRS", "private",
        "The CF attributes jax-gcm gives its horizontal coordinates."
        " jem.components.coordinates' _LONGITUDE_ATTRS / _LATITUDE_ATTRS (used"
        " by every component's dataset) are copies of these"
        " values, so a slab dataset and an atmosphere dataset describe their"
        " shared axes identically. JAX-ESM copies rather than imports because"
        " a private mapping is not an API -- but it is still pinned to this"
        " revision, so its disappearance must be noticed here.",
    ),
    # ------------------------------------------------------------------
    # Physics diagnostics. Every physics package that resolves a surface
    # (SPEEDY, ECHAM; Held-Suarez opts out) publishes the SAME
    # package-independent SurfaceExchange struct under
    # diagnostics["surface_exchange"] (jax-gcm#754/#911/#914), so `target`
    # is always "surface_exchange" and `attribute` is one of its field
    # names -- checked directly against SurfaceExchange's own fields
    # (dataclasses.fields), which needs no model build at all.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "surface_exchange", "net_heat_flux", "diagnostics",
        "Net downward heat flux into the surface, W m-2; negated at the"
        " component boundary to JAX-ESM's upward-positive convention.",
    ),
    IntegrationPoint(
        "surface_exchange", "evaporation", "diagnostics",
        "Evaporation, kg m-2 s-1 upward -- already JAX-ESM's units.",
    ),
    IntegrationPoint(
        "surface_exchange", "precipitation", "diagnostics",
        "Total precipitation (convective + large-scale/stratiform),"
        " kg m-2 s-1 downward -- already the total, computed once by the"
        " publisher from whichever precipitation-producing terms are"
        " actually composed.",
    ),
    IntegrationPoint(
        "surface_exchange", "stress_u", "diagnostics",
        "Eastward surface stress, N m-2, positive down (the stress ON the"
        " surface; westerlies give stress_u > 0) -- already JAX-ESM's"
        " convention. The stress the atmosphere column was given, so the"
        " ocean receives the momentum the atmosphere lost: feeds"
        " JCMDerived.eastward_wind_stress, and its sea tile"
        " JCMDerived.sea_eastward_wind_stress, which jem.fluxes.VerosExchange"
        " hands to Veros as surface_taux (jax-esm#132).",
    ),
    IntegrationPoint(
        "surface_exchange", "stress_v", "diagnostics",
        "Northward surface stress, N m-2; same note as stress_u. Feeds"
        " JCMDerived.northward_wind_stress.",
    ),
    IntegrationPoint(
        "surface_exchange", "wind_u", "diagnostics",
        "Near-surface eastward wind, m s-1, at the package's own reference"
        " (jax-gcm#911/#914; the reference is named in the struct's static"
        " wind_reference field -- '10m' for ECHAM, 'lowest_level' for"
        " SPEEDY). Published as JCMDerived.u0 for a consumer that needs the"
        " wind itself; JAX-ESM derives no stress from it (see stress_u).",
    ),
    IntegrationPoint(
        "surface_exchange", "wind_v", "diagnostics",
        "Near-surface northward wind, m s-1; same note as wind_u. Feeds"
        " JCMDerived.v0.",
    ),
    IntegrationPoint(
        "surface_exchange", "tile_names", "diagnostics",
        "Static names of the tile axis (jcm's TILE_NAMES): JAX-ESM selects"
        " the 'sea' tile by name, or combines 'water' and 'sea_ice', never"
        " by an assumed index.",
    ),
    IntegrationPoint(
        "surface_exchange", "tile_fraction", "diagnostics",
        "Tile fractions; weight the 'water' and 'sea_ice' tiles when a"
        " package publishes those rather than one 'sea' tile.",
    ),
    IntegrationPoint(
        "surface_exchange", "net_heat_flux_tile", "diagnostics",
        "Per-tile net heat flux; the 'sea' tile feeds"
        " JCMDerived.sea_total_heat_flux, the ocean's heat flux"
        " (jax-esm#147).",
    ),
    IntegrationPoint(
        "surface_exchange", "evaporation_tile", "diagnostics",
        "Per-tile evaporation; the 'sea' tile feeds"
        " JCMDerived.sea_evaporation and the ocean's freshwater flux.",
    ),
    IntegrationPoint(
        "surface_exchange", "stress_u_tile", "diagnostics",
        "Per-tile eastward stress; the 'sea' tile is the stress"
        " jem.fluxes.VerosExchange hands the ocean.",
    ),
    IntegrationPoint(
        "surface_exchange", "stress_v_tile", "diagnostics",
        "Per-tile northward stress; same note as stress_u_tile.",
    ),
    IntegrationPoint(
        "jcm.physics.surface.surface_exchange.SurfaceExchange", "tile",
        "public",
        "Selects one tile of a *_tile field by its surface type, so"
        " jem.components.jcm.exchange_fields.sea_tile never indexes the"
        " tile axis itself.",
    ),
    IntegrationPoint(
        "surface_exchange", "downward_shortwave_flux", "diagnostics",
        "Downward shortwave at the surface, W m-2: the Winton sea ice's"
        " rsds (JCMDerived.downward_shortwave_flux).",
    ),
    IntegrationPoint(
        "surface_exchange", "downward_longwave_flux", "diagnostics",
        "Incident downward longwave at the surface, W m-2: the Winton sea"
        " ice's rlds, of which it absorbs its emissivity.",
    ),
    IntegrationPoint(
        "surface_exchange", "wind_speed", "diagnostics",
        "Near-surface wind speed at the package's reference: the Winton"
        " ice's bulk formula.",
    ),
    IntegrationPoint(
        "surface_exchange", "surface_pressure", "diagnostics",
        "Surface pressure, Pa; over jcm.constants.p0 it is the Winton"
        " ice's normalized_surface_pressure.",
    ),
    IntegrationPoint(
        "surface_exchange", "air_temperature", "diagnostics",
        "The air the package's fluxes were computed against, at its"
        " air_reference: the Winton ice's air where no sea tile is published.",
    ),
    IntegrationPoint(
        "surface_exchange", "air_specific_humidity", "diagnostics",
        "Same reference as air_temperature.",
    ),
    IntegrationPoint(
        "surface_exchange", "air_temperature_tile", "diagnostics",
        "Per-tile air; the 'sea' tile is the air SPEEDY's sea fluxes"
        " were computed against, JCMDerived.sea_air_temperature.",
    ),
    IntegrationPoint(
        "surface_exchange", "air_specific_humidity_tile", "diagnostics",
        "Same note as air_temperature_tile.",
    ),
    IntegrationPoint(
        "surface_exchange", "air_potential_temperature", "diagnostics",
        "Lowest-level potential temperature: brought to the surface"
        " pressure, it decides rain or snow by MITgcm AIM's rule"
        " (exchange_fields.split_precipitation).",
    ),
    IntegrationPoint(
        "surface_exchange", "precip_rain", "diagnostics",
        "A package's own rain/snow split, taken where published.",
    ),
    IntegrationPoint(
        "surface_exchange", "precip_snow", "diagnostics",
        "Same note as precip_rain.",
    ),
    IntegrationPoint(
        "jcm.forcing.ForcingData", "sea_ice_temperature", "public",
        "The skin temperature a coupled sea ice hands SPEEDY, which blends it"
        " with the SST by sice_am; the Winton ice writes it, and JCMComponent"
        " seeds it with SPEEDY's own stand-in while jax-gcm leaves it None.",
    ),
    IntegrationPoint(
        "jcm.physics.surface.speedy_surface_flux",
        "SEA_ICE_FREEZING_TEMPERATURE", "public",
        "SPEEDY's stand-in ice temperature cap (sstfr): the seed JCMComponent"
        " gives sea_ice_temperature, so the first coupled step is the"
        " uncoupled one.",
    ),
    IntegrationPoint(
        "jcm.physics_interface", "POST_PHYSICS_STATE_KEY", "public",
        "The cloud schemes' post-physics anchor: it stays in the physics"
        " carry but jax-gcm never saves it, so JCMComponent's diagnostics"
        " template leaves it out, or JCMDerived.physics would change"
        " structure after the first step (ECHAM).",
    ),
    IntegrationPoint(
        "jcm.physics.surface.surface_exchange", "SURFACE_EXCHANGE_KEY",
        "public",
        "The diagnostics key the struct is published under: JCMComponent"
        " reads the struct off the structural diagnostics template at"
        " initialize, to warn when the package publishes no sea tiles.",
    ),
    IntegrationPoint(
        "jcm.physics.surface.surface_exchange", "surface_exchange_from",
        "public",
        "The single, package-independent surface-exchange reader:"
        " jem.components.jcm.exchange_fields.from_diagnostics is a thin"
        " sign/unit translation on top of this.",
    ),
    IntegrationPoint(
        "jcm.physics.composable_physics.ComposablePhysics",
        "require_surface_exchange", "public",
        "Composition-time check that the composed physics package publishes"
        " the surface-exchange struct, called from"
        " jem.runners.build_atmosphere right after the model is built --"
        " so a package that cannot publish (Held-Suarez, which resolves no"
        " surface fluxes) fails at composition with a named error, not at"
        " the first coupled step.",
    ),
    # ------------------------------------------------------------------
    # Package data. Shipped inside the `jcm` wheel, so it is reachable with
    # importlib.resources and needs no path from the user.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm.config", "config.yaml", "package data",
        "jax-gcm ships its Hydra config groups as package data, which is what"
        " lets jem/config/config.yaml add `pkg://jcm.config` to Hydra's search"
        " path and compose jax-gcm's own groups under `atmosphere` instead of"
        " restating them.",
    ),
    IntegrationPoint(
        "jcm.data", "bc/t30/clim/forcing.nc", "package data",
        "The packaged T30 surface climatology that the `${jcm_data:...}`"
        " config resolver (jem.config) points the configurations at, e.g."
        " `ice_clim_file` of earth-slab-winton.",
    ),
    IntegrationPoint(
        "jcm", "data/bc", "package data",
        "The T30 boundary-condition climatology JAX-ESM's forcing and"
        " topography generation tool reads, located relative to the installed"
        " `jcm` package rather than from a user-supplied path.",
    ),
    # ------------------------------------------------------------------
    # Constructors JAX-ESM's public workflow asks the user to call. JAX-ESM
    # does not call these itself -- the user hands the result in -- but its
    # quick start, its slab-grid docstrings and its own tests do, so a rename
    # breaks a documented workflow just as surely as a rename of a call.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm.utils", "get_coords", "public",
        "Builds the CoordinateSystem a user hands to Model and to"
        " jem.components.slab.SlabGrid.from_coords.",
    ),
    IntegrationPoint(
        "jcm.utils", "data_to_xarray", "public",
        "jax-gcm's array-to-Dataset conversion, whose coordinate values a slab"
        " dataset must reproduce bit-for-bit so the two concatenate.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.speedy_coords", "get_speedy_coords", "public",
        "The SPEEDY-native CoordinateSystem constructor used throughout"
        " JAX-ESM's examples and tests.",
    ),
    IntegrationPoint(
        "jcm.terrain", "TerrainData", "public",
        "Orography and land-sea mask; its `fmask` is the land fraction the"
        " slab components are built on.",
    ),
    IntegrationPoint(
        "jcm.terrain.TerrainData", "from_coords", "public",
        "Earth terrain on a given grid.",
    ),
    IntegrationPoint(
        "jcm.terrain.TerrainData", "from_file", "public",
        "Terrain from a user's boundary-condition file.",
    ),
    IntegrationPoint(
        "jcm.terrain.TerrainData", "aquaplanet", "public",
        "The aquaplanet terrain every fast coupled test builds on.",
    ),
    IntegrationPoint(
        "jcm.constants", "cpd", "public",
        "Dry-air specific heat, read by SlabAtmosphereModel's column heat"
        " capacity. jem.constants deliberately does not redefine any constant"
        " jax-gcm already owns.",
    ),
    IntegrationPoint(
        "jcm.constants", "rd", "public",
        "Gas constant of dry air: the Winton ice's air density"
        " (jem.components.slab.winton_seaice_model, read off"
        " physical_constants).",
    ),
    IntegrationPoint(
        "jcm.constants", "alhc", "public",
        "Latent heat of condensation: the latent-heat term of the Winton"
        " ice's bulk surface flux (read off physical_constants).",
    ),
    IntegrationPoint(
        "jcm.constants", "rhow", "public",
        "Density of liquid water: the bucket land's water-mass accounting"
        " (jem.components.slab.bucket_land_model).",
    ),
    IntegrationPoint(
        "jcm.constants", "rearth", "public",
        "Earth's radius: the bucket land's runoff mass rate and the Winton"
        " ice's transport metrics.",
    ),
    IntegrationPoint(
        "jcm.constants", "p0", "public",
        "Reference pressure: the normalised surface pressure"
        " (jem.components.jcm.exchange_fields), the rain/snow split's"
        " dry-adiabatic extrapolation, and the Winton ice's air density and"
        " seed humidity.",
    ),
    IntegrationPoint(
        "jcm.constants", "akap", "public",
        "R/cp of dry air, the same extrapolation's exponent.",
    ),
    IntegrationPoint(
        "jcm.constants", "physical_constants", "public",
        "The live constants singleton. The Winton ice reads cpd, rd, p0, sbc"
        " and alhc off it when traced, so a set_constants override reaches it;"
        " jem.configurations.load saves it before building a configuration so"
        " a failed build can restore it.",
    ),
    IntegrationPoint(
        "jcm.constants", "set_constants", "public",
        "Restore the singleton jem.configurations.load saved, when building a"
        " configuration fails after the atmosphere has applied its"
        " `+atmosphere.constants.*` overrides, so a failed build does not"
        " leave a later one running on the overridden constants.",
    ),
    IntegrationPoint(
        "jcm.constants", "sbc", "public",
        "Stefan-Boltzmann constant: the Winton ice's longwave emission and the"
        " seed longwave of JCMComponent's initial derived fields.",
    ),
    # ------------------------------------------------------------------
    # SPEEDY's parameter structs, imported at module scope by the Winton
    # ice, so a rename breaks `import jem.components`. The ice takes its
    # defaults from jax-gcm's so the two surfaces cannot disagree.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm.physics.speedy.params", "ModRadConParameters", "public",
        "SPEEDY's radiation parameters; the source of the Winton ice's"
        " emissivity default (jem.components.slab.winton_seaice_model.params"
        " and winton_seaice_model).",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.ModRadConParameters", "default", "public",
        "JCM's default radiation parameters, from which the ice's emissivity"
        " default is taken. Signature: default() -> ModRadConParameters.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.ModRadConParameters", "emisfc", "public",
        "The surface longwave emissivity the ice defaults to.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params", "SurfaceFluxParameters", "public",
        "SPEEDY's surface-flux parameters; IceSurfaceFluxParameters"
        " (jem.components.slab.winton_seaice_model.params) copies the ones the"
        " ice reads.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.SurfaceFluxParameters", "default", "public",
        "JCM's default surface-flux parameters: the ice's defaults."
        " Signature: default() -> SurfaceFluxParameters.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.SurfaceFluxParameters", "chs", "public",
        "Heat-exchange coefficient over sea: the ice's bulk formula.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.SurfaceFluxParameters", "vgust", "public",
        "Wind speed of sub-grid gusts, added in quadrature to the wind: the"
        " ice's bulk formula.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.SurfaceFluxParameters", "dtheta", "public",
        "Potential-temperature gradient of the stability correction: the"
        " ice's bulk formula.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.SurfaceFluxParameters", "fstab", "public",
        "Amplitude of the stability correction: the ice's bulk formula.",
    ),
    IntegrationPoint(
        "jcm.physics.speedy.params.SurfaceFluxParameters", "lscasym", "public",
        "Whether the stability correction is asymmetric: the ice reads it as a"
        " static flag.",
    ),
)
