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
--------------------------------------------
``JCM_SUPPORTED_REV`` is jax-gcm ``dev`` as it stood on 2026-09-21, and carries
four things JAX-ESM is written against:

* **#750** -- one ``run`` schema plus the ``configuration`` config group, which
  is what lets ``jem/config/config.yaml`` compose jax-gcm's own Hydra groups
  under an ``atmosphere`` key instead of restating them;
* **#763** -- the input-resolution engine, which is how boundary-condition and
  initial-condition inputs are located at build time;
* **#819** -- jax-gcm configures no logging of its own. Importing ``jcm`` no
  longer calls ``logging.basicConfig``, and ``Model.__init__`` no longer takes
  a ``log_level`` keyword (``jcm.runners`` sets the ``jcm`` logger from
  ``run.log_level`` instead). JAX-ESM never wanted jax-gcm to configure the
  root logger -- ``jem.main`` sets the level of the ``jem`` logger alone -- so
  this is the removal of a conflict, and the ``log_level=50`` the test
  fixtures used to pass purely to silence that ``basicConfig`` is gone with
  it;
* **#824** -- the resumable model state and the date conversion are public.
  ``Model.bootstrap_state()`` returns its ``(dycore_state, physics_carry)``
  pair, the same pair is readable as ``Model.dycore_state`` /
  ``Model.physics_carry``, ``ModelPredictions.with_context(model)`` re-attaches
  the context a pytree round trip drops, and ``Model._date_from_sim_time`` is
  now ``Model.date_from_sim_time`` (the old name kept only as a delegating
  alias). ``JCMComponent`` is built on all four, so it reaches for no private
  jax-gcm attribute at all; and because the old private date name only
  delegates, an instance-level override of it -- the season freeze in
  ``examples/02_experimental/03_jcm_veros_earth`` -- has to move to the public
  name or it silently stops taking effect.

jax-gcm reports ``3.0.0rc1`` here -- its first 3.0 release candidate -- but the
tag is not cut, so a commit sha is still what is pinned. The ``jcm>=3.0.0rc1``
floor in ``pyproject.toml`` is the loosest statement of the same thing: by
PEP 440 a release candidate satisfies a floor naming it, so ``3.0.0rc1`` and
every later 3.x pass, whereas this module says *which one* was verified.

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

When a tagged jax-gcm release finally contains all of the above, replace the
sha with the tag and raise the ``pyproject.toml`` floor to match.
"""

from __future__ import annotations

from typing import NamedTuple

#: The jax-gcm revision JAX-ESM is developed, tested and supported against:
#: ``dev`` as of 2026-09-21. The full 40-character sha, not an abbreviation,
#: because that is what ``actions/checkout`` needs and what ``git rev-parse``
#: in a jax-gcm checkout can be compared against directly.
JCM_SUPPORTED_REV = "9e399ab2840bb02eff3a5ea60a86072ab37da495"

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
JCM_INTEGRATION_POINTS: tuple[IntegrationPoint, ...] = (
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
        "jcm.date", "parse_duration_days", "public",
        "Turn a human run length or coupling interval ('1 year', '10 days')"
        " into days on the model's calendar."
        " Signature: parse_duration_days(value, calendar='gregorian')"
        " -> float.",
    ),
    IntegrationPoint(
        "jcm.date", "days_per_year", "public",
        "Calendar length used by jem.base.component.TimeAxis and"
        " jem.base.coupler for the annual cycle, so every component's seasonal"
        " forcing agrees with the atmosphere's calendar."
        " Signature: days_per_year(calendar='gregorian') -> float.",
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
        "jcm.model.Model", "bootstrap_state", "public",
        "Build the initial dycore state and physics carry without integrating"
        " a step, which is what lets JCMComponent.initialize() produce a carry"
        " of the exact structure step 1 will return. Returns the"
        " `(dycore_state, physics_carry)` pair directly (jax-gcm#824), so"
        " JAX-ESM unpacks the return value rather than reading it back off the"
        " model.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "dycore_state", "public",
        "Read-only view of the dycore state bootstrap_state/run installed on"
        " the model. JCMComponent.initialize() takes the state from"
        " bootstrap_state's return value; this is the name the contract test"
        " watches so that the pair stays readable without a private read"
        " (jax-gcm#824 closed jax-gcm#755 with it).",
    ),
    IntegrationPoint(
        "jcm.model.Model", "physics_carry", "public",
        "Read-only view of the cross-step physics carry paired with"
        " dycore_state; same provenance and the same reason to watch it.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "date_from_sim_time", "public",
        "jax-gcm's own elapsed-seconds -> DateData conversion, and the method"
        " jcm.model.Model calls internally for date-aware forcing. The"
        " season-freeze helper in"
        " examples/02_experimental/03_jcm_veros_earth/model_setup.py overrides"
        " it on the model instance, so a rename turns that override into a"
        " silent no-op rather than an error (which is exactly what the rename"
        " from _date_from_sim_time in jax-gcm#824 would have done).",
    ),
    IntegrationPoint(
        "jcm.model.Model", "run_from_state_with_carry", "public",
        "Advance the atmosphere by exactly one coupling interval, threading"
        " the cross-step physics carry; the whole coupled step is this call.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "dt_si", "public",
        "The model's own timestep, checked against the coupler's coupling"
        " timestep in JCMComponent.bind().",
    ),
    IntegrationPoint(
        "jcm.model.Model", "start_date", "public",
        "Checked against the coupler's start date in JCMComponent.bind() so a"
        " mismatch is refused up front rather than drifting silently.",
    ),
    IntegrationPoint(
        "jcm.model.Model", "calendar", "public",
        "Checked against the coupler's calendar in JCMComponent.bind().",
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
        "jcm.dycore.base.DynamicalCore", "sim_time", "public",
        "The dycore state's own clock, compared with the coupler's in"
        " JCMComponent._report_clock_drift to catch a carry from another run.",
    ),
    IntegrationPoint(
        "jcm.forcing", "ForcingData", "public",
        "Type of the `forcing` entry of JCMComponent's carry, which"
        " exchangers overwrite (the SST an ocean component computes) every"
        " coupling step.",
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
    # Physics diagnostics. jax-gcm has no package-independent surface
    # exchange contract yet (jax-gcm#754), so jem/components/jcm/
    # exchange_fields.py reads each package's own struct. `target` is the
    # physics package; `attribute` is the dotted path into one step's
    # diagnostics dict.
    # ------------------------------------------------------------------
    IntegrationPoint(
        "speedy", "_surface_flux.hfluxn", "diagnostics",
        "Net surface heat flux, W m-2, downward-positive in jax-gcm; negated"
        " at the component boundary to JAX-ESM's upward-positive convention.",
    ),
    IntegrationPoint(
        "speedy", "_surface_flux.evap", "diagnostics",
        "Evaporation, g m-2 s-1 upward; converted to kg m-2 s-1.",
    ),
    IntegrationPoint(
        "speedy", "_surface_flux.u0", "diagnostics",
        "Near-surface zonal wind, m s-1, exchanged with surface components.",
    ),
    IntegrationPoint(
        "speedy", "_surface_flux.v0", "diagnostics",
        "Near-surface meridional wind, m s-1.",
    ),
    IntegrationPoint(
        "speedy", "_convection.precnv", "diagnostics",
        "Convective precipitation, g m-2 s-1 downward; half of the total"
        " precipitation JAX-ESM exchanges.",
    ),
    IntegrationPoint(
        "speedy", "_condensation.precls", "diagnostics",
        "Large-scale precipitation, g m-2 s-1 downward; the other half.",
    ),
    IntegrationPoint(
        "jcm.physics.surface.echam.surface_physics.EchamSurface", "provides",
        "public",
        "jax-gcm's own declaration that the ECHAM surface term writes the"
        " 'surface' diagnostics key -- the key exchange_fields.detect uses to"
        " tell an ECHAM run from a SPEEDY one.",
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
