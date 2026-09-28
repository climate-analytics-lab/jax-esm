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
removed X" rather than as a mid-run crash. It also asserts that the revision
the CI workflow checks out is this one, so the pin and the workflow cannot
drift apart.

Why a ``dev`` revision rather than a release
-------------------------------------------
``JCM_SUPPORTED_REV`` is ``0eef9b3a88982886241622fde6530513d025192c``, a
``dev`` commit. jax-gcm has no 3.x tag yet -- ``3.0.0rc1`` is reported from
the source tree, not cut as a release -- so a sha is the most precise thing
there is to name; ``pyproject.toml``'s ``jcm>=3.0.0rc1`` floor is the loosest
true statement of the same fact, since jax-gcm's version string is bumped
only at release and a release-candidate floor is satisfied by every later
3.x release too. The policy, stated once here: pin to a ``dev`` commit that
has actually been validated as a whole against this repository's own gates
-- not to the merge commit of one motivating pull request, and not to
whatever ``dev`` happens to be at bump time, either of which can carry
commits this branch has never been run against. When a tagged jax-gcm
release finally contains this revision, replace the sha with the tag and
raise the ``pyproject.toml`` floor to match.

This revision is needed because jax-gcm's package-independent
``SurfaceExchange`` contract (``diagnostics["surface_exchange"]``) publishes
a near-surface wind vector, ``wind_u``/``wind_v``, identically from every
physics package (jax-gcm#911/#914); see
:mod:`jem.components.jcm.exchange_fields` for the translation and
``docs/source/design/jcm_adapter.md`` for the full adapter design.

Current behaviour a JEM user must know:

* ``jcm.forcing.resolve_align``'s ``auto`` resolves a boundary condition's
  alignment (climatology vs. transient) only for a known mirror/packaged
  product; any other from-file atmosphere forcing needs
  ``atmosphere.forcing.align`` set explicitly -- see the ``forcing.align``
  comments in
  ``jem/config/configuration/{earth-slab,veros-double-drake,veros-earth}
  .yaml``.
* jax-gcm configures no logging of its own; ``jem.main`` sets the ``jem``
  logger's level, and ``jcm.runners`` sets the ``jcm`` logger's level from
  ``run.log_level``.
* ``JCMComponent`` threads jax-gcm's own :class:`~jcm.model.RunState`
  (``time``, ``step``) through ``carry["time"]``/``carry["step"]`` and
  ``Model.run_from_state_with_carry(initial_time=, initial_step=)`` every
  coupled step, separately from the coupler's own clock.

How to bump the pin
-------------------
1. Change ``JCM_SUPPORTED_REV`` (and ``JCM_SUPPORTED_VERSION``, if jax-gcm's
   version string moved) here, and update the "why" paragraph above to say what
   the new revision brings.
2. Change ``JCM_REV`` in ``.github/workflows/tests.yml`` to the same sha --
   ``test_workflow_pins_the_supported_revision`` fails if the two disagree.
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
JCM_SUPPORTED_REV = "0eef9b3a88982886241622fde6530513d025192c"

#: The version string ``jcm`` reports at :data:`JCM_SUPPORTED_REV`. jax-gcm's
#: version is only bumped at release, so it is a weaker statement than the sha
#: -- many revisions share it -- but it is what a user's environment can be
#: checked against without a git checkout, which is why
#: ``test_installed_jcm_matches_contract`` reads ``jcm.__version__`` rather
#: than the distribution metadata: an editable install records its version
#: when it is installed, so the metadata of a checkout that has since moved
#: to another revision is stale, which is exactly the situation a pin bump
#: creates.
JCM_SUPPORTED_VERSION = "3.0.0rc1"


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
    # Driver (jem.driver, T2.2): building a model and a forcing from a
    # composed Hydra config, and running it in health-gated chunks.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "jcm.runners", "build_model", "public",
        "Build the atmosphere Model from the composed `atmosphere` config"
        " subtree, so JAX-ESM never reimplements jax-gcm's own wiring."
        " Signature: build_model(cfg: DictConfig) -> Model.",
    ),
    IntegrationPoint(
        "jcm.runners", "build_forcing", "public",
        "Build the atmosphere's ForcingData from the same config subtree and"
        " the built coordinates."
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
        "Per-chunk health gate on the atmosphere's output dataset, so a run"
        " that has gone unstable stops instead of burning the queue."
        " Signature: check_health(ds, chunk_idx: int, elapsed_days: float)"
        " -> tuple[bool, dict].",
    ),
    IntegrationPoint(
        "jcm.date", "parse_duration_seconds", "public",
        "Turn a run length, chunk, coupling interval or averaging window"
        " ('12 hours', '10 days', or a number of days) into whole seconds;"
        " refuses 'months'/'years' and fractional seconds."
        " Signature: parse_duration_seconds(value) -> int.",
    ),
    IntegrationPoint(
        "jcm.date", "fraction_of_year_elapsed", "public",
        "Position in the annual cycle in [0, 1), on the proleptic Gregorian"
        " calendar. CouplingTime.year_fraction calls this directly, so a"
        " component's seasonal cycle and the atmosphere's agree by"
        " construction. Signature: fraction_of_year_elapsed(dt:"
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
        "The object JCMComponent wraps; also re-exported in JCMComponent's"
        " type hints.",
    ),
    IntegrationPoint(
        "jcm.model", "RunState", "public",
        "JCM's own resumable clock struct (dynamics, physics, time, step)."
        " JCMComponent stores its 'time'/'step' fields in the atmosphere's"
        " carry and threads them back into run_from_state_with_carry every"
        " coupled step, exactly as jcm's own chunked runner does.",
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
        "jcm.model.Model", "date_from_sim_time", "public",
        "jax-gcm's own elapsed-seconds -> DateData conversion, and the method"
        " jcm.model.Model calls internally for date-aware forcing on every"
        " coupled step: JCMComponent.step (jem/components/jcm/component.py)"
        " calls run_from_state_with_carry, which resolves this name"
        " internally. An instance-level override of it -- the shape a"
        " perpetual-season (frozen seasonal cycle) hook would take, tracked"
        " as jax-esm#120 since JEM ships none today -- has to target this"
        " public name.",
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
        "The nodal shape every exchanged field is on, the grid the default"
        " forcing is built on, and part of the context re-attached to a"
        " ModelPredictions after a pytree round trip.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "terrain", "public",
        "The atmosphere's land-sea mask (`terrain.fmask`), which jem.runners"
        " builds a surface component's SlabGrid from when that component"
        " shares the atmosphere's grid.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "physics", "public",
        "Source of the diagnostics template (see get_empty_data) and part of"
        " the ModelPredictions context.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "dycore", "public",
        "Part of the ModelPredictions context, and the owner of the state's"
        " simulation clock (see DynamicalCore.sim_time).",
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
        " built as. JAX-ESM never constructs one; it is watched because"
        " whether a ForcingData field IS one decides the pytree structure of"
        " the atmosphere's carry, which is what JCMComponent.initialize()"
        " has to settle before a coupled run can scan.",
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
        " itself again. Public since jax-gcm#824 (closing jax-gcm#756); it"
        " replaced JAX-ESM's read of the private `_predictions` payload.",
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
        "The CF attributes jax-gcm gives its horizontal coordinates. The slab"
        " components' _LONGITUDE_ATTRS / _LATITUDE_ATTRS are copies of these"
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
        "surface_exchange", "wind_u", "diagnostics",
        "Near-surface eastward wind, m s-1, at the package's own reference"
        " (jax-gcm#911/#914; the reference is named in the struct's static"
        " wind_reference field -- '10m' for ECHAM, 'lowest_level' for"
        " SPEEDY). Feeds jem.fluxes.bulk_wind_stress via JCMDerived.u0.",
    ),
    IntegrationPoint(
        "surface_exchange", "wind_v", "diagnostics",
        "Near-surface northward wind, m s-1; same note as wind_u. Feeds"
        " JCMDerived.v0.",
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
        "jcm.constants", "tmelt", "public",
        "Fresh-water melting point, the reference temperature of"
        " SlabSeaiceModel.",
    ),
    IntegrationPoint(
        "jcm.constants", "rhoi", "public",
        "Sea-ice density, SlabSeaiceModel.",
    ),
    IntegrationPoint(
        "jcm.constants", "alhf", "public",
        "Latent heat of fusion, SlabSeaiceModel.",
    ),
)
