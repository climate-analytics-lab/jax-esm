"""Tests for the JCM adapter (:mod:`jem.components.jcm`).

These build a real ``jcm`` model, so they use the smallest configuration
SPEEDY physics supports -- T21 with 5 vertical levels on an aquaplanet --
and share it across the module: construction plus the first compiled step
dominates the runtime.
"""

from types import SimpleNamespace
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
from jcm.date import DateData
from jcm.forcing import ForcingData, TimeSeries
from jcm.model import Model
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import (
    SpeedyHumidity,
    SpeedySurfaceFlux,
    speedy_physics,
)
from jcm.physics.surface.echam.surface_exchange_publisher import (
    EchamSurfaceExchange,
)
from jcm.physics.surface.surface_exchange import (
    SurfaceExchange as JcmSurfaceExchange,
)
from jcm.physics.surface.surface_exchange import surface_exchange_from
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

from jem.base.component import (
    Component,
    CouplingTime,
    SupportsBind,
    SupportsXarray,
    TimeAxis,
)
from jem import constants
from jem.components.jcm import JCMComponent, JCMDerived, exchange_fields
from tests.unit import _pre754_exchange_reader

START_DATE = jdt.to_datetime("2000-01-01")
CALENDAR = "365_day"
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")

# T21 on jcm's matching (64, 32) nodal grid; 5 levels is the fewest SPEEDY
# physics accepts (its convective cloud-top search needs kx >= 5).
LAYERS = 5
TRUNCATION = 21
GRID_SHAPE = (64, 32)


# ---------------------------------------------------------------------------
# Real (but minimal) ComposablePhysics objects for exchange_fields.
# has_wind_vector, which decides presence from the composed physics's own
# TERMS rather than from a diagnostics-dict key
# (see that function's docstring) -- so a fixture built from real jcm terms,
# not a stand-in, is what proves the check actually inspects them.
# ---------------------------------------------------------------------------

def _speedy_physics_with_wind() -> ComposablePhysics:
    """Build a minimal composed physics that DOES publish a near-surface wind
    vector: it composes a real ``SpeedySurfaceFlux`` term, the one term that
    fills SPEEDY's private ``_surface_flux.u0``/``.v0`` with a real
    bulk-formula wind. No coords are cached and nothing is ever run -- only
    ``.terms`` is read by ``has_wind_vector`` -- so this is cheap to build.
    """
    return ComposablePhysics([SpeedySurfaceFlux()], checkpoint_terms=False)


def _speedy_legacy_physics_without_wind() -> ComposablePhysics:
    """Build a real, if unusual, hybrid composition: a SPEEDY-legacy term
    (``SpeedyHumidity``) with NO ``SpeedySurfaceFlux`` composed.

    This is the jax-esm#129-review defect's own regression fixture: every
    ``SpeedyTermBase`` term (``SpeedyHumidity`` included) round-trips
    SPEEDY's whole ``PhysicsData`` struct through the diagnostics dict
    (``_data_from_diagnostics``/``_diagnostics_from_data``), so a real step
    of THIS composition would still carry a ``_surface_flux`` diagnostics
    key -- zeroed, because nothing here ever computed a real wind. A
    predicate that read the diagnostics dict's keys (the pre-review
    ``has_wind_vector``) would misreport a wind vector for it; the composed-
    terms predicate must not. No shipped jem/jcm configuration composes
    SPEEDY terms this way (23fba9e's commit message records the gap this
    closes), but nothing stops a hand-built ``ComposablePhysics`` from it,
    which is exactly why the predicate has to be faithful rather than
    incidentally correct for the shipped cases alone.
    """
    return ComposablePhysics([SpeedyHumidity()], checkpoint_terms=False)


class _SurfaceProbe(PhysicsTerm):
    """Publish the diagnostics keys ``EchamSurfaceExchange`` requires
    (``surface``, ``vertical_diffusion``, ``pressure_full``), position-coded
    off the column-vectorized state, so a real column-vectorized ECHAM-style
    composition can be built and run without a full ECHAM physics package.
    Shared by every test that needs a real, minimal, wind-vector-free,
    column-vectorized ``ComposablePhysics`` (see :func:`_echam_style_physics`).
    """

    name: ClassVar[str] = "surface_probe"
    category: ClassVar[str] = "surface"
    provides: ClassVar[tuple[str, ...]] = (
        "surface", "vertical_diffusion", "pressure_full")

    def __call__(self, state, diagnostics, forcing, terrain):
        t_bot = state.temperature[-1]  # (ncols,), jax-gcm's own flatten order
        sst = forcing.sea_surface_temperature.reshape(t_bot.shape)
        zero = jnp.zeros_like(t_bot)
        diagnostics = {
            **diagnostics,
            "surface": SimpleNamespace(
                sensible_heat_flux=t_bot, latent_heat_flux=zero,
                evaporation=sst, momentum_flux_u=zero, momentum_flux_v=zero,
            ),
            "vertical_diffusion": SimpleNamespace(wind_10m=zero),
            "pressure_full": jnp.full(state.temperature.shape, 95000.0),
        }
        return PhysicsTendency.zeros(state.temperature.shape), diagnostics


def _echam_style_physics() -> ComposablePhysics:
    """Build a real, minimal, column-vectorized composition with no wind vector.

    Composes :class:`_SurfaceProbe` (satisfying ``EchamSurfaceExchange``'s
    ``requires``) and the real ``EchamSurfaceExchange`` publisher, with
    ``vectorize_columns=True`` -- exactly ECHAM's own composition shape --
    so ``has_wind_vector`` is exercised against a real column-vectorized
    ``ComposablePhysics``, not only a whole-grid (SPEEDY-shaped) one.
    """
    return ComposablePhysics(
        [_SurfaceProbe(), EchamSurfaceExchange()],
        checkpoint_terms=False, vectorize_columns=True,
    )


def _build_model() -> Model:
    coords = get_speedy_coords(layers=LAYERS, spectral_truncation=TRUNCATION)
    return Model(
        coords=coords,
        terrain=TerrainData.aquaplanet(coords),
        start_date=START_DATE,
        calendar=CALENDAR,
    )


def _bound_component(model: Model) -> JCMComponent:
    component = JCMComponent(model)
    component.bind(
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )
    return component


def _coupling_time(step: int) -> CouplingTime:
    """Build the clock the coupler hands a component on step ``step``."""
    return CouplingTime(
        step=jnp.int32(step),
        sim_time=jnp.float32(step * 86400.0),
        dt=86400.0,
        year_offset_seconds=0.0,
        days_per_year=365.0,
    )


@pytest.fixture(scope="module")
def model() -> Model:
    return _build_model()


@pytest.fixture(scope="module")
def component(model) -> JCMComponent:
    return _bound_component(model)


@pytest.fixture(scope="module")
def stepped(component):
    """Two consecutive coupled steps, computed once for several tests."""
    carry0 = component.initialize()
    carry1, diagnostics1 = component.step(carry0, _coupling_time(0))
    carry2, diagnostics2 = component.step(carry1, _coupling_time(1))
    return carry0, carry1, carry2, diagnostics1, diagnostics2


# --------------------------------------------------------------------------
# Fast tests: no integration.
# --------------------------------------------------------------------------

def test_component_satisfies_protocols(component):
    """The wrapper is what the coupler tests for with ``isinstance``."""
    assert isinstance(component, Component)
    assert isinstance(component, SupportsBind)
    assert isinstance(component, SupportsXarray)
    assert component.name == "atm"


def test_step_before_bind_raises(model):
    """Stepping an unregistered component names the fix."""
    component = JCMComponent(model)
    with pytest.raises(RuntimeError, match="bind"):
        component.step({}, _coupling_time(0))


def test_bind_rejects_mismatched_start_date(model):
    """A start-date mismatch names both dates rather than silently drifting."""
    component = JCMComponent(model)
    other = jdt.to_datetime("1990-06-01")
    with pytest.raises(ValueError, match="Start-date mismatch"):
        component.bind(
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=other,
            calendar=CALENDAR,
        )


def test_bind_rejects_mismatched_calendar(model):
    component = JCMComponent(model)
    with pytest.raises(ValueError, match="Calendar mismatch"):
        component.bind(
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=START_DATE,
            calendar="gregorian",
        )


def test_bind_rejects_non_multiple_timestep(model):
    """The coupling interval must be a whole number of model timesteps."""
    component = JCMComponent(model)
    model_seconds = int(model.dt_si.to_timedelta().total_seconds())
    with pytest.raises(ValueError, match="whole multiple"):
        component.bind(
            coupling_timestep=jdt.to_timedelta(model_seconds + 1, "second"),
            start_date=START_DATE,
            calendar=CALENDAR,
        )


def test_initialize_does_not_integrate(model, monkeypatch):
    """``initialize()`` must build pytrees, not run the model.

    The previous adapter ran a whole coupling interval just to learn the
    shape of the diagnostics dict, which cost a step per run and started
    the atmosphere one interval ahead of the coupler's clock.
    """
    component = _bound_component(model)
    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("initialize() integrated the model")

    monkeypatch.setattr(model, "run_from_state_with_carry", _spy)
    carry = component.initialize()

    assert calls == []
    assert set(carry) == {"state", "physics", "derived", "forcing"}
    assert carry["derived"].total_heat_flux.shape == GRID_SHAPE


def _jcm_surface_exchange(net_heat_flux, evaporation, precipitation,
                          wind_speed=3.0):
    """Build a real jax-gcm ``SurfaceExchange`` (#754) with hand-chosen values.

    The other guaranteed fields (``sensible_heat_flux``, ``latent_heat_flux``,
    ``stress_u``/``stress_v``, ``air_density``, ``air_potential_temperature``)
    are filled with placeholders: JEM's translation does not read them (see
    ``jem/components/jcm/exchange_fields.py``'s module docstring), so their
    values are irrelevant to what is being tested here.
    """
    field = lambda value: jnp.full(GRID_SHAPE, value)  # noqa: E731
    return JcmSurfaceExchange(
        net_heat_flux=field(net_heat_flux),
        sensible_heat_flux=field(0.0),
        latent_heat_flux=field(0.0),
        evaporation=field(evaporation),
        precipitation=field(precipitation),
        stress_u=field(0.0),
        stress_v=field(0.0),
        wind_speed=field(wind_speed),
        air_density=field(1.2),
        air_potential_temperature=field(290.0),
    )


def _fake_speedy_diagnostics(net_heat_flux=10.0, evaporation=0.002,
                             precipitation=0.008, u0=1.5, v0=-2.5):
    """Build a diagnostics dict shaped like SPEEDY's post-#754 output.

    Carries both the published ``surface_exchange`` contract struct and
    SPEEDY's private wind-vector key (``_surface_flux.u0``/``.v0`` --
    ``exchange_fields``'s one remaining package-specific read; see its module
    docstring). The values are already in the contract's units (kg m-2 s-1,
    positive up/down) -- unlike the pre-#754 fixture this replaces, which
    used SPEEDY's private g m-2 s-1 diagnostics and needed a /1000 conversion.
    """
    field = lambda value: jnp.full(GRID_SHAPE, value)  # noqa: E731
    return {
        "_surface_flux": SimpleNamespace(u0=field(u0), v0=field(v0)),
        "surface_exchange": _jcm_surface_exchange(
            net_heat_flux, evaporation, precipitation,
            wind_speed=float(np.hypot(u0, v0)),
        ),
    }


def _fake_echam_diagnostics(net_heat_flux=7.0, evaporation=0.001,
                            precipitation=0.004):
    """Build a diagnostics dict shaped like ECHAM's post-#754 output.

    No ``_surface_flux`` key: ECHAM never carries a near-surface wind
    *vector* anywhere in its diagnostics, contract or no contract (see
    ``exchange_fields``'s module docstring), so this is what an ECHAM run's
    diagnostics genuinely look like from ``from_diagnostics``'s point of
    view -- not a stripped-down fixture.
    """
    return {
        "surface_exchange": _jcm_surface_exchange(
            net_heat_flux, evaporation, precipitation,
        ),
    }


def _fake_column_vectorized_echam_diagnostics():
    """Build column-vectorized ECHAM diagnostics whose ``surface_exchange``
    fields encode their own ``(lon, lat)`` index, not a uniform value.

    ``ComposablePhysics(vectorize_columns=True)`` (ECHAM) flattens the
    horizontal ``(ix, il)`` grid to a single ``ncols`` axis before iterating
    its terms, and every diagnostic it writes -- including the published
    ``surface_exchange`` struct -- stays on that flattened axis; jax-gcm only
    reshapes it back inside its own xarray serialization, never before (see
    ``jem.components.jcm.component._unflatten_to_nodal_shape``'s docstring).
    ``_fake_echam_diagnostics`` above builds every field already on
    ``GRID_SHAPE`` (``(ix, il)``) and so never exercises this -- discovered
    only by running the jax-esm#129 regression test against a real ECHAM
    model (``tests/unit/test_coupled.py``), which is why this fixture exists
    as a second, deliberately different one: an all-``GRID_SHAPE`` fixture
    cannot catch a reshape bug that a real column-vectorized step hits.

    The fixture is position-encoding because a *uniform* flattened fixture
    (``jnp.full``) cannot tell a correct unflatten from a
    transposed or reversed one -- swapping ``_unflatten_to_nodal_shape`` for
    ``value.reshape((il, ix)).T`` or ``value[::-1].reshape(nodal_shape)``
    still passed every test built on it. Each cell's value here is
    ``1000 * i_lon + i_lat`` (``GRID_SHAPE`` is deliberately non-square, so
    even a same-shape transpose changes which value lands where), flattened
    with a plain ``.reshape(-1)`` of the ``(ix, il)`` code array -- C order,
    the same convention jax-gcm's own column flatten uses (see
    ``_unflatten_to_nodal_shape``'s docstring; also verified directly against
    jax-gcm's REAL flatten below, in
    ``test_unflatten_agrees_with_a_real_jax_gcm_column_flatten``).

    Returns
    -------
    tuple[dict, numpy.ndarray]
        The diagnostics dict, and the ``(ix, il)`` code array itself, so a
        test can compare cell by cell without re-deriving it.

    """
    ix, il = GRID_SHAPE
    lon_index, lat_index = np.meshgrid(np.arange(ix), np.arange(il), indexing="ij")
    code = (1000 * lon_index + lat_index).astype(np.float64)
    flat = jnp.asarray(code.reshape(-1))
    zero = jnp.zeros_like(flat)
    diagnostics = {
        "surface_exchange": JcmSurfaceExchange(
            net_heat_flux=flat,
            sensible_heat_flux=zero,
            latent_heat_flux=zero,
            evaporation=flat,
            precipitation=zero,
            stress_u=zero,
            stress_v=zero,
            wind_speed=jnp.full_like(flat, 3.0),
            air_density=jnp.full_like(flat, 1.2),
            air_potential_temperature=jnp.full_like(flat, 290.0),
        ),
    }
    return diagnostics, code


def _fake_column_vectorized_echam_diagnostics_with_precipitation():
    """Like :func:`_fake_column_vectorized_echam_diagnostics`, but with
    ``precipitation`` and ``evaporation`` independently position-encoded and
    nonzero, rather than sharing one code (``evaporation``) with a
    permanently-zero ``precipitation``.

    jax-esm#129-review nit: ECHAM's precipitation is exactly ``0.0`` in the
    two-day slow coupled regression run (``tests/unit/test_coupled.py``), so
    nothing there would ever catch a placement or sign bug specific to
    precipitation -- unlike ``total_heat_flux``, whose placement and sign
    flip are already exercised, with a genuinely nonzero, position-encoded
    value, by :func:`_fake_column_vectorized_echam_diagnostics` /
    ``test_unflatten_places_each_cell_correctly``. This fixture gives
    ``evaporation`` and ``precipitation`` each their own nonzero code (a
    constant offset apart), so a placement bug (a transposed or reversed
    reshape), a field mix-up (the two swapped) or a sign bug (either one
    negated, which neither should be -- see ``exchange_fields``'s own
    docstring table: both are already in JEM's sign convention) each produce
    a distinctly wrong, checkable value rather than an indistinguishable
    uniform or zero one. ``net_heat_flux`` is held at a uniform, uninteresting
    value here since its own placement/sign is not what this fixture is for.

    Returns
    -------
    tuple[dict, numpy.ndarray, numpy.ndarray]
        The diagnostics dict, the ``(ix, il)`` evaporation code array, and
        the ``(ix, il)`` precipitation code array.

    """
    ix, il = GRID_SHAPE
    lon_index, lat_index = np.meshgrid(np.arange(ix), np.arange(il), indexing="ij")
    evaporation_code = (1000 * lon_index + lat_index + 1.0).astype(np.float64)
    precipitation_code = (1000 * lon_index + lat_index + 3000.0).astype(np.float64)
    evaporation_flat = jnp.asarray(evaporation_code.reshape(-1))
    precipitation_flat = jnp.asarray(precipitation_code.reshape(-1))
    zero = jnp.zeros_like(evaporation_flat)
    diagnostics = {
        "surface_exchange": JcmSurfaceExchange(
            net_heat_flux=zero,
            sensible_heat_flux=zero,
            latent_heat_flux=zero,
            evaporation=evaporation_flat,
            precipitation=precipitation_flat,
            stress_u=zero,
            stress_v=zero,
            wind_speed=jnp.full_like(zero, 3.0),
            air_density=jnp.full_like(zero, 1.2),
            air_potential_temperature=jnp.full_like(zero, 290.0),
        ),
    }
    return diagnostics, evaporation_code, precipitation_code


def test_speedy_exchange_shapes_and_signs():
    """Sign flip only: evaporation/precipitation need no unit conversion any
    more, because the #754 contract already publishes them in JEM's units
    (kg m-2 s-1) -- see the module docstring's derivation table.
    """
    diagnostics = _fake_speedy_diagnostics()
    exchange = exchange_fields.from_diagnostics(diagnostics, _speedy_physics_with_wind())

    assert exchange.total_heat_flux.shape == GRID_SHAPE
    # jax-gcm's net_heat_flux is positive DOWN into the surface; JEM is up.
    np.testing.assert_allclose(exchange.total_heat_flux, -10.0)
    # Already kg m-2 s-1 and already the convective+large-scale total in the
    # published contract -- no conversion, no manual summing.
    np.testing.assert_allclose(exchange.evaporation, 0.002)
    np.testing.assert_allclose(exchange.precipitation, 0.008)
    np.testing.assert_allclose(exchange.u0, 1.5)
    np.testing.assert_allclose(exchange.v0, -2.5)
    for field in exchange:
        assert field.shape == GRID_SHAPE


def test_echam_heat_and_water_fluxes_use_the_same_translation_as_speedy():
    """#754 closes: ECHAM's heat/water fluxes now translate identically to
    SPEEDY's, with no per-package code -- where the pre-#754 ``echam()``
    reader always raised ``NotImplementedError`` (git history, commit
    756cc2c), because there was no package-independent struct to read.

    This checks the translation directly against jax-gcm's own public
    reader (:func:`jcm.physics.surface.surface_exchange.surface_exchange_from`)
    rather than against ``from_diagnostics`` end to end, because
    ``from_diagnostics`` raises for ECHAM at the *separate* wind-vector step
    (checked below) before it would return -- the heat/water translation
    itself does not depend on the wind vector being available.
    """
    diagnostics = _fake_echam_diagnostics()
    contract = surface_exchange_from(diagnostics)
    # jax-gcm's net_heat_flux is positive DOWN; JEM's total_heat_flux is the
    # negative of it (positive UP) -- same sign flip as the SPEEDY case above.
    np.testing.assert_allclose(-contract.net_heat_flux, -7.0)
    np.testing.assert_allclose(contract.evaporation, 0.001)
    np.testing.assert_allclose(contract.precipitation, 0.004)


def test_echam_wind_vector_is_none_not_an_error():
    """jax-esm#129: ECHAM has no near-surface wind *vector* anywhere (only a
    speed, which predates and is independent of #754 -- see the module
    docstring's wind-vector note), but that absence is now a static ``None``
    on the returned struct rather than a raise from ``from_diagnostics``
    itself. The heat/water fluxes translate exactly as they do for SPEEDY
    (``test_echam_heat_and_water_fluxes_use_the_same_translation_as_speedy``
    above); only a caller that also needs ``u0``/``v0`` (today:
    ``jem.fluxes.VerosExchange``) has anything to notice, and it is refused at
    composition time instead (see ``tests/unit/test_fluxes.py``).
    """
    diagnostics = _fake_echam_diagnostics()
    exchange = exchange_fields.from_diagnostics(diagnostics, _echam_style_physics())

    assert exchange.u0 is None
    assert exchange.v0 is None
    np.testing.assert_allclose(exchange.total_heat_flux, -7.0)
    np.testing.assert_allclose(exchange.evaporation, 0.001)
    np.testing.assert_allclose(exchange.precipitation, 0.004)


def test_has_wind_vector_is_true_only_when_speedy_surface_flux_is_composed():
    """The structural question ``JCMDerived.zeros`` decides composition-time
    absence from. It is decided from the composed physics's own TERMS (real ``ComposablePhysics`` objects here),
    not from a diagnostics-dict key -- see ``exchange_fields.has_wind_vector``.
    """
    assert exchange_fields.has_wind_vector(_speedy_physics_with_wind())
    assert not exchange_fields.has_wind_vector(_echam_style_physics())


def test_has_wind_vector_is_true_for_standard_speedy_physics():
    """The real, full SPEEDY factory composes ``SpeedySurfaceFlux``."""
    assert exchange_fields.has_wind_vector(speedy_physics())


def test_has_wind_vector_is_true_for_prescribed_flux_speedy():
    """``speedy-forced-flux`` (``SpeedySurfaceFlux(prescribed_fluxes=True)``,
    ``jcm/config/physics/speedy-forced-flux.yaml``) still composes
    ``SpeedySurfaceFlux`` -- forced mode replaces the turbulent fluxes with
    prescribed ones, but the term (and its near-surface wind) is unaffected.
    """
    physics = ComposablePhysics(
        [SpeedySurfaceFlux(prescribed_fluxes=True)], checkpoint_terms=False)
    assert exchange_fields.has_wind_vector(physics)


def test_has_wind_vector_is_false_for_a_hybrid_composition_without_speedy_surface_flux():
    """The jax-esm#129-review defect, fixed here: a hybrid composition with a
    SPEEDY-legacy term (``SpeedyHumidity``) but no ``SpeedySurfaceFlux`` must
    report no wind vector -- even though its diagnostics dict, exactly like a
    real SPEEDY step's (``_fake_speedy_diagnostics``, which carries the
    ``_surface_flux`` key), looks structurally identical to one. Every
    ``SpeedyTermBase`` term writes that key (zeroed here, since nothing in
    this composition ever computed a real wind), which is precisely why a
    diagnostics-dict-key check used to get this wrong (recorded, not fixed,
    in 23fba9e's commit message; fixed here by asking the composed TERMS
    instead -- see ``exchange_fields.has_wind_vector``'s docstring).
    """
    physics = _speedy_legacy_physics_without_wind()
    diagnostics = _fake_speedy_diagnostics()  # has the misleading `_surface_flux` key

    assert not exchange_fields.has_wind_vector(physics)
    exchange = exchange_fields.from_diagnostics(diagnostics, physics)
    assert exchange.u0 is None
    assert exchange.v0 is None
    # The heat/water fluxes are unaffected -- only the wind-vector decision
    # changes for this composition.
    np.testing.assert_allclose(exchange.total_heat_flux, -10.0)


def test_jcm_derived_zeros_is_windless_for_a_template_with_no_wind_key():
    """jax-esm#129: a physics template shaped like ECHAM's (no
    ``_surface_flux`` entry) builds a ``JCMDerived`` whose ``u0``/``v0`` are
    ``None``, not zero arrays -- the same static decision a real ECHAM step
    will make (``exchange_fields.from_diagnostics`` returns ``None`` too), so
    the carry ``JCMComponent.initialize()`` builds already has the structure
    step 1 will produce.
    """
    derived = JCMDerived.zeros(
        _fake_echam_diagnostics(), GRID_SHAPE, _echam_style_physics())

    assert derived.u0 is None
    assert derived.v0 is None
    assert derived.total_heat_flux.shape == GRID_SHAPE
    assert derived.total_freshwater_flux.shape == GRID_SHAPE


def test_jcm_derived_zeros_has_wind_for_a_template_with_the_speedy_key():
    """The unchanged SPEEDY path: a template that carries the wind-vector key
    still builds zero-filled ``u0``/``v0`` arrays of the right shape, exactly
    as before #129.

    Deliberately the SAME non-zero fixture
    ``test_has_wind_vector_is_true_only_when_speedy_surface_flux_is_composed``
    reads for translation, not a special all-zero variant: ``zeros()`` derives
    every field's shape/dtype from a real translated exchange but always
    canonicalises the *value* to positive zero (see the ``signbit``
    assertion below), so a non-zero template reading back as
    all-zero here is the stronger proof that ``zeros()`` truly ignores the
    template's values rather than merely happening to be handed zeros.
    """
    derived = JCMDerived.zeros(
        _fake_speedy_diagnostics(), GRID_SHAPE, _speedy_physics_with_wind())

    assert derived.u0.shape == GRID_SHAPE
    assert derived.v0.shape == GRID_SHAPE
    np.testing.assert_allclose(derived.u0, 0.0)
    np.testing.assert_allclose(derived.v0, 0.0)
    np.testing.assert_allclose(derived.total_heat_flux, 0.0)
    np.testing.assert_allclose(derived.total_freshwater_flux, 0.0)
    # `-0.0 == 0.0` and `np.allclose(-0.0, 0.0)` are both True, so the sign
    # bit needs its own check: `total_heat_flux = -net_heat_flux`'s negation
    # would otherwise leave a template's `+0.0` as `-0.0` here -- a real, if
    # invisible, regression against "SPEEDY's zeros() is bit-for-bit
    # unchanged".
    for name in ("u0", "v0", "total_heat_flux", "total_freshwater_flux",
                 "evaporation", "precipitation"):
        field = np.asarray(getattr(derived, name))
        assert not bool(np.signbit(field).any()), name


@pytest.mark.parametrize("shape_like", [
    GRID_SHAPE,                              # tuple
    list(GRID_SHAPE),                        # list
    np.asarray(GRID_SHAPE),                  # numpy.ndarray
    jnp.asarray(GRID_SHAPE),                 # jax.Array
], ids=["tuple", "list", "numpy_ndarray", "jax_array"])
def test_jcm_derived_zeros_names_the_new_signature_for_a_legacy_positional_call(
    shape_like,
):
    """A pre-#129 call, ``zeros(shape, physics)``, fails with a message
    naming the new signature, not an opaque ``TypeError`` several calls deep
    -- whichever of the ordinary ways a caller might spell a shape: a
    ``tuple``, ``list``, ``numpy.ndarray`` or ``jax.Array``.

    ``zeros()``'s argument order was ``(shape, physics, **overrides)`` before
    jax-esm#129; it is now ``(diagnostics_template, nodal_shape, physics,
    **overrides)`` -- #129 swapped the first two, and this review added the
    required third. Passing the pre-#129 order with the new argument simply
    appended -- a shape where ``diagnostics_template`` now goes -- used to
    fail as an opaque error several calls deep (``TypeError: tuple indices
    must be integers or slices, not str`` out of
    ``exchange_fields.from_diagnostics``'s ``dict.get``, for the ``tuple``
    case) rather than naming this method or the argument that changed.
    """
    with pytest.raises(TypeError, match="zeros\\(diagnostics_template, nodal_shape"):
        JCMDerived.zeros(
            shape_like, _fake_speedy_diagnostics(), _speedy_physics_with_wind())


def test_jcm_derived_zeros_missing_the_physics_argument_names_it():
    """A 2-positional-argument call (the exact pre-#129 spelling) is refused
    by Python's own signature check, naming the missing argument, since
    jax-esm#129's review made ``physics`` a required third argument rather
    than something ``zeros()`` could default or infer.
    """
    with pytest.raises(TypeError, match="physics"):
        JCMDerived.zeros(_fake_speedy_diagnostics(), GRID_SHAPE)


def test_jcm_derived_zeros_unflattens_a_column_vectorized_template():
    """jax-esm#129 regression: a real ECHAM step's surface exchange is
    flattened ``(ncols,)``, not ``(ix, il)`` -- ``JCMDerived.zeros`` (and
    ``JCMComponent.step``, which goes through the same helper) must put it on
    the atmosphere's own nodal grid, or a coupled step that copies
    ``derived.total_heat_flux`` into a plain ``(ix, il)`` component (a slab
    ocean, in the default exchange table) fails with an opaque
    ``ValueError: Incompatible shapes for broadcasting`` the first time it
    actually runs -- which no all-``GRID_SHAPE`` fixture (``_fake_echam_
    diagnostics``) can catch, only ``_fake_column_vectorized_echam_
    diagnostics`` (built flattened, like a real step) or a real model
    (``tests/unit/test_coupled.py``'s slow ECHAM regression test).

    Shape and the windless decision only: ``zeros()`` canonicalises every
    value to zero regardless of what the template carries (see
    ``test_jcm_derived_zeros_has_wind_for_a_template_with_the_speedy_key``'s
    docstring on ``-0.0``/``+0.0``), so it cannot be
    used to check *placement* -- ``test_unflatten_places_each_cell_correctly``
    and ``test_unflatten_agrees_with_a_real_jax_gcm_column_flatten`` below,
    which read the reshape directly off ``_surface_exchange_on_nodal_grid``,
    do that.
    """
    template, _ = _fake_column_vectorized_echam_diagnostics()
    derived = JCMDerived.zeros(template, GRID_SHAPE, _echam_style_physics())

    assert derived.total_heat_flux.shape == GRID_SHAPE
    assert derived.evaporation.shape == GRID_SHAPE
    assert derived.precipitation.shape == GRID_SHAPE
    assert derived.u0 is None
    assert derived.v0 is None
    np.testing.assert_array_equal(np.asarray(derived.total_heat_flux), 0.0)


def test_unflatten_places_each_cell_correctly():
    """A uniform fixture cannot catch a transposed or reversed unflatten --
    this position-encoding one can, and is checked to
    actually do so (teeth verified by hand: swapping
    ``_unflatten_to_nodal_shape`` for ``value.reshape((il, ix)).T`` or
    ``value[::-1].reshape(nodal_shape)`` makes this test fail; see the
    commit message for the exact before/after).
    """
    from jem.components.jcm.component import _surface_exchange_on_nodal_grid

    diagnostics, code = _fake_column_vectorized_echam_diagnostics()
    exchange = _surface_exchange_on_nodal_grid(
        diagnostics, GRID_SHAPE, _echam_style_physics())

    actual = np.asarray(exchange.total_heat_flux)  # = -net_heat_flux = -code
    assert actual.shape == GRID_SHAPE
    np.testing.assert_array_equal(actual, -code)
    # A handful of individual cells, spelled out, so a placement bug is
    # visible in the assertion itself rather than only in an array diff.
    for i, j in ((0, 0), (1, 0), (0, 1), (7, 13), (63, 31)):
        assert actual[i, j] == -code[i, j], (i, j)

    evaporation = np.asarray(exchange.evaporation)  # = evaporation = code
    np.testing.assert_array_equal(evaporation, code)


def test_unflatten_places_precipitation_and_evaporation_correctly():
    """jax-esm#129-review nit: ECHAM's precipitation is exactly ``0.0`` in
    the two-day slow coupled regression run, so nothing exercises a
    placement or sign bug specific to it. Feeds independently
    position-encoded, nonzero precipitation and evaporation through the same
    column-vectorized-ECHAM unflatten path
    (:func:`_fake_column_vectorized_echam_diagnostics_with_precipitation`)
    and asserts each lands at its own cell with no sign flip -- both are
    already in JEM's convention (positive up for evaporation, positive down
    for precipitation), unlike ``total_heat_flux = -net_heat_flux``.
    """
    from jem.components.jcm.component import _surface_exchange_on_nodal_grid

    diagnostics, evaporation_code, precipitation_code = (
        _fake_column_vectorized_echam_diagnostics_with_precipitation()
    )
    exchange = _surface_exchange_on_nodal_grid(
        diagnostics, GRID_SHAPE, _echam_style_physics())

    evaporation = np.asarray(exchange.evaporation)
    precipitation = np.asarray(exchange.precipitation)
    assert evaporation.shape == GRID_SHAPE
    assert precipitation.shape == GRID_SHAPE
    np.testing.assert_array_equal(evaporation, evaporation_code)
    np.testing.assert_array_equal(precipitation, precipitation_code)
    # A handful of individual cells, spelled out, so a placement bug -- or a
    # mix-up between the two fields, which their distinct offsets would also
    # reveal -- is visible in the assertion itself.
    for i, j in ((0, 0), (1, 0), (0, 1), (7, 13), (63, 31)):
        assert evaporation[i, j] == evaporation_code[i, j], ("evaporation", i, j)
        assert precipitation[i, j] == precipitation_code[i, j], ("precipitation", i, j)
    # Neither field is sign-flipped (unlike total_heat_flux = -net_heat_flux).
    assert np.all(precipitation > 0)
    assert np.all(evaporation > 0)


def test_unflatten_agrees_with_a_real_jax_gcm_column_flatten():
    """The unflatten is checked against jax-gcm's REAL column-vectorized
    flatten, not a hand-rolled stand-in for it: a genuine
    ``ComposablePhysics(vectorize_columns=True)`` runs a probe term (which
    seeds the column-vectorized state with the same position code) and the
    real ``EchamSurfaceExchange`` publisher, and the branch's own
    ``_surface_exchange_on_nodal_grid`` must recover the code at the right
    ``(lon, lat)`` cell from the resulting diagnostics dict -- and agree with
    jax-gcm's own ``data_struct_to_dict`` reshape (the one it uses for xarray
    output) on the same diagnostics, so the check runs on jax-gcm's own
    machinery rather than on a stand-in.
    """
    from jem.components.jcm.component import _surface_exchange_on_nodal_grid

    nlon, nlat, nlev = 6, 4, 2  # deliberately non-square, and small
    ncols = nlon * nlat
    lon_index, lat_index = np.meshgrid(
        np.arange(nlon), np.arange(nlat), indexing="ij"
    )
    code = (1000.0 * lon_index + lat_index).astype(np.float64)

    class _Probe(PhysicsTerm):
        """Publish the level-bottom temperature and the forced SST as
        diagnostics ``EchamSurfaceExchange`` reads, so the position code
        travels through a REAL column-vectorized state, not a diagnostics
        dict built by hand.
        """

        name: ClassVar[str] = "probe"
        category: ClassVar[str] = "surface"
        provides: ClassVar[tuple[str, ...]] = (
            "surface", "vertical_diffusion", "pressure_full")

        def __call__(self, state, diagnostics, forcing, terrain):
            assert state.temperature.shape == (nlev, ncols), state.temperature.shape
            t_bot = state.temperature[-1]  # (ncols,), jax-gcm's own flatten order
            sst = forcing.sea_surface_temperature.reshape(ncols)
            zero = jnp.zeros(ncols)
            diagnostics = {
                **diagnostics,
                "surface": SimpleNamespace(
                    sensible_heat_flux=t_bot, latent_heat_flux=zero,
                    evaporation=sst, momentum_flux_u=zero, momentum_flux_v=zero,
                ),
                "vertical_diffusion": SimpleNamespace(wind_10m=zero),
                "pressure_full": jnp.full((nlev, ncols), 95000.0),
            }
            return PhysicsTendency.zeros(state.temperature.shape), diagnostics

    physics = ComposablePhysics(
        [_Probe(), EchamSurfaceExchange()],
        checkpoint_terms=False, vectorize_columns=True,
    )
    temperature = np.zeros((nlev, nlon, nlat))
    temperature[-1] = code  # the level EchamSurfaceExchange reads from
    state = PhysicsState(
        temperature=jnp.asarray(temperature),
        specific_humidity=jnp.zeros((nlev, nlon, nlat)),
        u_wind=jnp.zeros((nlev, nlon, nlat)),
        v_wind=jnp.zeros((nlev, nlon, nlat)),
        geopotential=jnp.zeros((nlev, nlon, nlat)),
        normalized_surface_pressure=jnp.ones((nlon, nlat)),
    )
    forcing = SimpleNamespace(sea_surface_temperature=jnp.asarray(code + 0.5))
    _tendency, diagnostics = physics._compute_tendencies_columns(state, forcing, None)
    published = diagnostics["surface_exchange"]
    assert published.net_heat_flux.shape == (ncols,)

    exchange = _surface_exchange_on_nodal_grid(diagnostics, (nlon, nlat), physics)
    # JEM's total_heat_flux = -net_heat_flux; the probe set
    # sensible_heat_flux = t_bot = code with lhf/radiation/precip all zero,
    # so net_heat_flux = -code and total_heat_flux = +code.
    np.testing.assert_array_equal(np.asarray(exchange.total_heat_flux), code)
    np.testing.assert_array_equal(np.asarray(exchange.evaporation), code + 0.5)

    # Agrees with jax-gcm's OWN xarray-serialization reshape on the same
    # diagnostics dict, not just with this branch's own code.
    as_grid = physics.data_struct_to_dict(
        {"surface_exchange": published}, nodal_shape=(nlon, nlat)
    )
    np.testing.assert_array_equal(
        np.asarray(as_grid["surface_exchange.net_heat_flux"]), -code
    )


def test_unflatten_to_nodal_shape_rejects_an_unexpected_shape_with_trailing_axis():
    """The ``ValueError`` branch of ``_unflatten_to_nodal_shape``, tested
    directly because no other test reaches it. A shape
    that is neither ``nodal_shape`` nor a flattened ``(ncols,)`` -- here,
    ``(ncols, 1)``, e.g. a future package publishing a per-column field with
    a spurious trailing axis -- must raise, naming the field and both shapes
    it was checked against, rather than pass through silently to fail later
    as an opaque broadcast error somewhere downstream.
    """
    from jem.components.jcm.component import _unflatten_to_nodal_shape

    nodal_shape = GRID_SHAPE
    ncols = nodal_shape[0] * nodal_shape[1]
    value = jnp.zeros((ncols, 1))

    with pytest.raises(ValueError, match="'total_heat_flux' has shape"):
        _unflatten_to_nodal_shape(value, nodal_shape, "total_heat_flux")


def test_unflatten_to_nodal_shape_rejects_a_wrong_length_1d_array():
    """Same finding as above, for the other unexpected shape a caller might
    pass: a 1-D array whose length is neither ``prod(nodal_shape)`` (the
    flattened, column-vectorized case) nor a match for ``nodal_shape``
    itself (impossible for a 1-D array against a 2-D ``nodal_shape``, but
    checked here via a length that is simply wrong either way).
    """
    from jem.components.jcm.component import _unflatten_to_nodal_shape

    nodal_shape = GRID_SHAPE
    ncols = nodal_shape[0] * nodal_shape[1]
    value = jnp.zeros((ncols - 1,))

    with pytest.raises(ValueError, match="'evaporation' has shape"):
        _unflatten_to_nodal_shape(value, nodal_shape, "evaporation")


def test_windless_jcm_derived_survives_a_jit_round_trip():
    """jax-esm#129: the static-``None`` design must survive ``jax.jit``, the
    same structural-equality check ``lax.scan`` applies to a coupled step's
    carry every iteration.
    """
    derived = JCMDerived.zeros(
        _fake_echam_diagnostics(), GRID_SHAPE, _echam_style_physics(),
        total_heat_flux=jnp.full(GRID_SHAPE, 3.0),
    )

    roundtripped = jax.jit(lambda d: d)(derived)

    assert roundtripped.u0 is None
    assert roundtripped.v0 is None
    np.testing.assert_allclose(roundtripped.total_heat_flux, 3.0)
    assert jax.eval_shape(lambda: derived) == jax.eval_shape(lambda: roundtripped)


def test_windless_jcm_derived_survives_a_checkpoint_round_trip(tmp_path):
    """jax-esm#129: ``jem.checkpoint.save``/``load`` must treat the absent
    wind the way they already treat any other leaf-free subtree of a carry
    (``jem/checkpoint.py``'s own docstring: "a component whose carry is `{}`
    or `None` ... contributes no leaf") -- reconstructed from the template's
    structure alone, never from anything in the file.
    """
    from jem import checkpoint

    derived = JCMDerived.zeros(
        _fake_echam_diagnostics(), GRID_SHAPE, _echam_style_physics(),
        total_heat_flux=jnp.full(GRID_SHAPE, 3.0),
        evaporation=jnp.full(GRID_SHAPE, 0.001),
    )

    path = checkpoint.save(derived, tmp_path / "carry.msgpack")
    restored = checkpoint.load(derived, path)

    assert restored.u0 is None
    assert restored.v0 is None
    np.testing.assert_allclose(restored.total_heat_flux, 3.0)
    np.testing.assert_allclose(restored.evaporation, 0.001)


def test_missing_surface_exchange_raises_jcms_own_key_error():
    """A package that publishes no ``surface_exchange`` at all (Held-Suarez)
    fails with jax-gcm's own pointed error, not a bare ``KeyError``.
    """
    with pytest.raises(KeyError, match="surface_exchange"):
        exchange_fields.from_diagnostics(
            {"radiation": None, "clouds": None}, _speedy_physics_with_wind())


def test_collapse_save_axis_handles_a_zero_sized_diagnostic():
    """jax-esm#129 regression: a zero-sized diagnostic must not blow up
    ``to_xarray``'s save-axis collapse.

    Discovered running the ECHAM regression test
    (``tests/unit/test_coupled.py::
    test_echam_coupled_to_a_slab_ocean_completes_several_steps``) end to end:
    ECHAM's aerosol diagnostics carry a per-species axis of length 0 with no
    aerosol species configured (jax-gcm's own uncoupled ``to_xarray`` drops
    these entirely -- see ``ComposablePhysics.data_struct_to_dict``'s "Zero-
    size entries ... are skipped" comment), but JEM's own
    ``_collapse_save_axis`` used to reshape with a ``-1`` placeholder, which
    JAX resolves by dividing the leaf's size by the product of its other
    axes -- and a zero-sized leaf makes that product zero too, raising
    ``ZeroDivisionError`` before jax-gcm's own skip logic ever runs. No
    per-package fixture reproduces this without building a real ECHAM
    model, so this test reaches for the private helper directly with a
    fabricated zero-sized leaf, shaped like the real one
    (``(iterations, 1, 0, nlev, ncols)``) -- SPEEDY never has an
    aerosol diagnostic with a zero axis, so this never surfaced there.
    """
    from jem.components.jcm.component import _collapse_save_axis

    zero_sized = jnp.zeros((3, 1, 0, 47, 4608))
    collapsed = _collapse_save_axis(zero_sized)
    assert collapsed.shape == (3, 0, 47, 4608)

    # A normal, non-zero leaf collapses exactly as it did before this fix
    # (the explicit `shape[0] * shape[1]` product agrees with what `-1`
    # resolved to, since the save axis is always exactly 1).
    normal = jnp.arange(3 * 1 * 4 * 5, dtype=jnp.float32).reshape((3, 1, 4, 5))
    collapsed_normal = _collapse_save_axis(normal)
    assert collapsed_normal.shape == (3, 4, 5)
    np.testing.assert_array_equal(collapsed_normal, normal.reshape((-1, 4, 5)))


@pytest.mark.slow
def test_speedy_new_reader_agrees_with_the_pre_754_reader(component, stepped):
    """The #754 collapse must not change what a SPEEDY run exchanges.

    Runs one real coupled step (the ``stepped`` fixture) and reads the SAME
    diagnostics dict two ways: through the pre-#754 adapter and through the
    new single reader. Agreement to floating-point tolerance is the decisive
    check the migration asked for -- not just that the two *formulas* look
    equivalent on paper, but that they give the same numbers on a real model
    step.

    The pre-#754 adapter is ``tests/unit/_pre754_exchange_reader.py``, a
    frozen vendored copy of ``jem/components/jcm/exchange_fields.py`` as it
    stood at commit 756cc2c (the last commit before the #754 migration) --
    see that module's docstring. It is vendored rather than loaded from git
    history (as this test used to do, with ``git show 756cc2c:...``) because
    CI's ``actions/checkout`` is a shallow clone: commit 756cc2c is not in
    the runner's object store, so ``git show`` failed there with exit status
    128 even though the test passed locally, where a full-history
    development checkout hid the problem. Vendoring the old reader once
    makes this test hermetic -- no dependency on git history, checkout
    depth, or the repository at all.
    """
    _, carry1, _, _, _ = stepped
    diagnostics = carry1["derived"].physics

    old_exchange = _pre754_exchange_reader.speedy(diagnostics)
    new_exchange = exchange_fields.from_diagnostics(diagnostics, component.model.physics)

    for name in ("total_heat_flux", "evaporation", "precipitation", "u0", "v0"):
        np.testing.assert_allclose(
            np.asarray(getattr(new_exchange, name)),
            np.asarray(getattr(old_exchange, name)),
            rtol=1e-6, atol=1e-9, err_msg=name,
        )


def test_echam_new_reader_matches_a_real_echam_surface_exchange_step():
    """The new reader against a REAL ``EchamSurfaceExchange`` step's output.

    There is no historical baseline for ECHAM (the pre-#754 ``echam()``
    reader always raised), so this is not an old-vs-new diff -- it is
    evidence that the translation is correct: the diagnostics dict is built
    by actually calling jax-gcm's own ``EchamSurfaceExchange`` term (not a
    reimplementation of it) on hand-chosen inputs, and the expected
    heat/water values are derived by hand from those SAME inputs, following
    the ECHAM energy balance ``EchamSurfaceExchange`` itself documents
    (net radiation minus the turbulent fluxes; stratiform plus convective
    precipitation).
    """
    ncols = 4
    shape_3d = (2, ncols)
    state = PhysicsState(
        temperature=jnp.full(shape_3d, 290.0),
        specific_humidity=jnp.full(shape_3d, 0.008),
        u_wind=jnp.zeros(shape_3d),
        v_wind=jnp.zeros(shape_3d),
        geopotential=jnp.zeros(shape_3d),
        normalized_surface_pressure=jnp.ones((ncols,)),
    )
    sensible_heat_flux, latent_heat_flux = 15.0, 85.0
    sw_down, sw_up, lw_down, lw_up = 200.0, 40.0, 300.0, 350.0
    precip_rain, precip_snow, precip_conv = 2e-5, 0.0, 1e-5
    evaporation = 3e-5
    diagnostics = {
        "surface": SimpleNamespace(
            sensible_heat_flux=jnp.full((ncols,), sensible_heat_flux),
            latent_heat_flux=jnp.full((ncols,), latent_heat_flux),
            evaporation=jnp.full((ncols,), evaporation),
            momentum_flux_u=jnp.full((ncols,), 0.02),
            momentum_flux_v=jnp.full((ncols,), -0.01),
        ),
        "vertical_diffusion": SimpleNamespace(
            wind_10m=jnp.full((ncols,), 5.0)),
        "radiation": SimpleNamespace(
            surface_sw_down=jnp.full((ncols,), sw_down),
            surface_sw_up=jnp.full((ncols,), sw_up),
            surface_lw_down=jnp.full((ncols,), lw_down),
            surface_lw_up=jnp.full((ncols,), lw_up),
        ),
        "clouds": SimpleNamespace(
            precip_rain=jnp.full((ncols,), precip_rain),
            precip_snow=jnp.full((ncols,), precip_snow),
        ),
        "convection": SimpleNamespace(
            precip_conv=jnp.full((ncols,), precip_conv)),
        "pressure_full": jnp.full(shape_3d, 95000.0),
    }
    _tendency, diagnostics = EchamSurfaceExchange()(
        state, diagnostics, None, None)

    expected_net_heat_flux = (
        (sw_down - sw_up) + (lw_down - lw_up)
        - sensible_heat_flux - latent_heat_flux
    )
    expected_precipitation = precip_rain + precip_snow + precip_conv

    # jax-esm#129: `from_diagnostics` now succeeds for ECHAM -- it no longer
    # raises for lack of a wind vector -- and returns u0/v0 as None.
    exchange = exchange_fields.from_diagnostics(diagnostics, _echam_style_physics())
    assert exchange.u0 is None
    assert exchange.v0 is None
    np.testing.assert_allclose(exchange.total_heat_flux, -expected_net_heat_flux)
    np.testing.assert_allclose(exchange.evaporation, evaporation)
    np.testing.assert_allclose(exchange.precipitation, expected_precipitation)

    contract = surface_exchange_from(diagnostics)
    # jax-gcm's net_heat_flux is positive DOWN; JEM's total_heat_flux is its
    # negative (positive UP).
    np.testing.assert_allclose(-contract.net_heat_flux, -expected_net_heat_flux)
    np.testing.assert_allclose(contract.evaporation, evaporation)
    np.testing.assert_allclose(contract.precipitation, expected_precipitation)


def test_make_jem_compatible_is_deprecated(model):
    """The old entry point still works, warns, and leaves the model alone."""
    from jem.components import jcm_component

    with pytest.warns(DeprecationWarning, match="JCMComponent"):
        component = jcm_component.make_jem_compatible(model, COUPLING_TIMESTEP)

    assert isinstance(component, JCMComponent)
    assert component.model is model
    # The wrapper no longer injects methods onto the jcm Model.
    assert not hasattr(model, "generate_step_function")


# --------------------------------------------------------------------------
# Slow tests: these integrate the model.
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_carry_structure_is_scannable(stepped):
    """A step must return exactly the carry structure, shapes and dtypes it got.

    This is what ``lax.scan`` enforces on the coupled step; checking it here
    localises a failure to this component.
    """
    carry0, carry1, _, _, _ = stepped
    assert jax.eval_shape(lambda: carry0) == jax.eval_shape(lambda: carry1)


@pytest.mark.slow
def test_physics_carry_is_threaded(component, stepped):
    """The cross-step physics carry evolves, and threading it is what stepping means.

    Two things at once: the carry is not a constant (so it genuinely holds
    state), and stepping twice from the initial carry gives the same answer
    as one two-step sequence -- i.e. nothing outside the carry is
    remembered between steps.
    """
    carry0, carry1, carry2, _, _ = stepped

    initial_leaves = jax.tree.leaves(carry0["physics"])
    stepped_leaves = jax.tree.leaves(carry1["physics"])
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(initial_leaves, stepped_leaves)
    ), "the physics carry came back unchanged, so it is not being threaded"

    # Re-running the same two steps by hand must reproduce them exactly:
    # the component holds no hidden state of its own.
    replay1, _ = component.step(carry0, _coupling_time(0))
    replay2, _ = component.step(replay1, _coupling_time(1))
    np.testing.assert_allclose(
        replay2["derived"].total_heat_flux,
        carry2["derived"].total_heat_flux,
        rtol=1e-6, atol=1e-6,
    )
    for expected, actual in zip(jax.tree.leaves(carry2["physics"]),
                                jax.tree.leaves(replay2["physics"])):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                                   rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_derived_fields_are_finite_and_consistent(stepped):
    """The published exchange is finite and its freshwater flux is E - P."""
    _, carry1, _, _, _ = stepped
    derived = carry1["derived"]

    for name in ("total_heat_flux", "evaporation", "precipitation", "u0", "v0"):
        field = getattr(derived, name)
        assert field.shape == GRID_SHAPE
        assert bool(jnp.all(jnp.isfinite(field))), name

    np.testing.assert_allclose(
        derived.total_freshwater_flux,
        derived.evaporation - derived.precipitation,
        rtol=1e-6, atol=1e-12,
    )


@pytest.mark.slow
def test_to_xarray_has_time_axis_of_length_n(component, stepped):
    """Stacked diagnostics serialize through jcm with one record per step.

    Also pins how jcm labels that axis: absolute ``datetime64[ns]`` at the
    END of each averaging interval. Any component whose output is merged
    with the atmosphere's has to write the same representation.
    """
    _, _, _, diagnostics1, diagnostics2 = stepped
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), diagnostics1, diagnostics2)
    time_axis = TimeAxis(START_DATE, np.arange(2), COUPLING_TIMESTEP, CALENDAR)

    dataset = component.to_xarray(stacked, time_axis)

    assert dataset.sizes["time"] == 2
    assert dataset.time.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(
        dataset.time.values,
        np.array(["2000-01-02", "2000-01-03"], dtype="datetime64[ns]"),
    )
    assert dataset.sizes["lon"], dataset.sizes["lat"] == GRID_SHAPE


@pytest.mark.slow
def test_to_xarray_rejects_a_mismatched_time_axis(component, stepped):
    """A time axis that does not match the records is a coupler-side bug."""
    _, _, _, diagnostics1, diagnostics2 = stepped
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), diagnostics1, diagnostics2)
    time_axis = TimeAxis(START_DATE, np.arange(3), COUPLING_TIMESTEP, CALENDAR)

    with pytest.raises(ValueError, match="output records"):
        component.to_xarray(stacked, time_axis)


def test_rebinding_to_a_different_timestep_is_rejected(model):
    """One instance belongs to one coupled model; a conflicting second bind raises."""
    component = _bound_component(model)
    # The same clock again is a no-op.
    component.bind(
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE, calendar=CALENDAR
    )
    with pytest.raises(ValueError, match="already bound"):
        component.bind(
            coupling_timestep=COUPLING_TIMESTEP * 2,
            start_date=START_DATE,
            calendar=CALENDAR,
        )


# ---------------------------------------------------------------------------
# Forcing read from a file, and the fields a coupled run overwrites
# ---------------------------------------------------------------------------
#
# The atmosphere's `forcing` section is the one section a coupled model both
# reads from a file and overwrites every step. jax-gcm builds a time-varying
# boundary condition as a `TimeSeries` (values, time axis, alignment mode --
# three pytree leaves) and slices it by date internally; an exchanger writes a
# single `(ix, il)` array into the same field. These pin which fields end up
# which way, and that a coupled step with a file-forced atmosphere really does
# keep its carry structure.


@pytest.fixture(scope="module")
def file_forcing(model) -> ForcingData:
    """jax-gcm's packaged T30 surface climatology on the test model's grid.

    The same file `+configuration=earth-slab` names as
    `${jcm_data:bc/t30/clim/forcing.nc}`, reached through the resolver's own
    helper so the test and the configuration cannot drift onto different data.
    """
    from jem.config import package_data_path

    return ForcingData.from_file(
        package_data_path("jcm.data", "bc/t30/clim/forcing.nc"),
        coords=model.coords,
    )


def _is_time_series(value) -> bool:
    """Return True if ``value`` is a jax-gcm time-varying forcing leaf."""
    return isinstance(value, TimeSeries)


def test_file_forcing_starts_out_as_time_series(file_forcing):
    """The premise: a from-file boundary condition is a `TimeSeries`, not an array.

    Every other test in this section is about what JAX-ESM does with that, so
    if jax-gcm ever stopped building one there would be nothing left to fix
    and these would pass vacuously.
    """
    for name in ("sea_surface_temperature", "sice_am", "stl_am",
                 "snowc_am", "soilw_am"):
        assert _is_time_series(getattr(file_forcing, name)), name


def test_the_component_reports_which_forcing_fields_vary_in_time(
    model, file_forcing
):
    """`time_varying_forcing` is what a hand-written coupling has to declare."""
    assert set(JCMComponent(model, forcing=file_forcing).time_varying_forcing) == {
        "sea_surface_temperature", "sice_am", "stl_am", "snowc_am", "soilw_am",
    }
    # The default forcing is plain arrays throughout, so there is nothing to
    # declare and an exchange into it never changes the carry's structure.
    assert JCMComponent(model).time_varying_forcing == ()


def test_initialize_collapses_only_the_exchanged_forcing(model, file_forcing):
    """Declared fields become per-step arrays; the rest stay climatologies."""
    component = JCMComponent(
        model, forcing=file_forcing,
        exchanged_forcing=("sea_surface_temperature", "sice_am"),
    )
    forcing = component.initialize()["forcing"]

    for name in ("sea_surface_temperature", "sice_am"):
        value = getattr(forcing, name)
        assert not _is_time_series(value), name
        assert value.shape == GRID_SHAPE, name
    # Nothing supplies the land surface here, so it must still vary through
    # the year -- freezing it at the start date would be a silent change to
    # what the atmosphere stands on.
    for name in ("stl_am", "snowc_am", "soilw_am"):
        value = getattr(forcing, name)
        assert _is_time_series(value), name
        assert value.values.shape[0] > 1, name


def test_collapsed_forcing_is_the_climatology_at_the_start_date(
    model, file_forcing
):
    """The value a collapsed field takes is the file's, read at the start date."""
    component = JCMComponent(
        model, forcing=file_forcing, exchanged_forcing=("sea_surface_temperature",),
    )
    expected = file_forcing.select(
        DateData.set_date(START_DATE), calendar=CALENDAR
    ).sea_surface_temperature

    collapsed = np.asarray(
        component.initialize()["forcing"].sea_surface_temperature
    )
    np.testing.assert_array_equal(collapsed, np.asarray(expected))
    # And it is that date's slice rather than any date's: a mid-year one
    # differs, so the start date is doing real work here.
    midyear = np.asarray(file_forcing.select(
        DateData.set_date(jdt.to_datetime("2000-07-01")), calendar=CALENDAR
    ).sea_surface_temperature)
    assert not np.allclose(collapsed, midyear)


def test_initialize_leaves_the_forcing_alone_when_nothing_is_exchanged(
    model, file_forcing
):
    """An atmosphere no exchanger writes to keeps jax-gcm's forcing untouched."""
    component = JCMComponent(model, forcing=file_forcing)

    assert component.exchanged_forcing == ()
    assert component.initialize()["forcing"] is file_forcing


def test_set_exchanged_forcing_rejects_an_unknown_field(model):
    """A field `ForcingData` does not have is refused while the model is built."""
    component = JCMComponent(model)
    with pytest.raises(ValueError, match="sea_ice_fraction"):
        component.set_exchanged_forcing(["sice_am", "sea_ice_fraction"])
    # The declaration is all-or-nothing: the valid name in the same call is
    # not half-applied.
    assert component.exchanged_forcing == ()


def test_set_exchanged_forcing_deduplicates_and_keeps_order(model):
    component = JCMComponent(model)
    component.set_exchanged_forcing(["stl_am", "sice_am", "stl_am"])
    assert component.exchanged_forcing == ("stl_am", "sice_am")


def test_coupled_step_keeps_its_structure_with_file_forcing(model, file_forcing):
    """One coupled step with a file-forced atmosphere scans.

    The regression this section exists for: the standard exchange writes plain
    arrays into `atm.forcing`, so before the fields it writes were collapsed
    the atmosphere's carry had one pytree structure going into the first
    exchange and another coming out -- which `lax.scan` cannot carry, and
    which `Coupler` refuses by name at trace time.

    Traced with `jax.eval_shape` rather than run: the structure check is a
    trace-time check, so tracing is what exercises it, and it costs no
    compilation. No land model, so the file's land climatology is not
    exchanged and has to come through the step still time-varying.
    """
    from jem.base.coupler import Coupler
    from jem.components import SlabOceanModel, SlabSeaiceModel
    from jem.components.slab import SlabGrid
    from jem.exchangers import default_exchangers, exchanged_fields

    atm = JCMComponent(model, forcing=file_forcing)
    grid = SlabGrid.from_coords(model.coords.horizontal)
    components = {
        "atm": atm,
        "ocn": SlabOceanModel(grid),
        "seaice": SlabSeaiceModel(grid, name="seaice"),
    }
    exchangers = default_exchangers(components)
    # What a coupler's runner does: the coupling table is what knows which of
    # the atmosphere's boundary conditions somebody else supplies.
    atm.set_exchanged_forcing(exchanged_fields(exchangers, atm.name))
    assert atm.exchanged_forcing == ("sea_surface_temperature", "sice_am")

    coupler = Coupler(
        components, exchangers,
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE,
        calendar=CALENDAR,
    )
    carry = coupler.initialize()
    final, _ = jax.eval_shape(coupler.generate_trajectory_function(1), carry)

    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)
    # The land surface nothing supplies came through with its time axis, so
    # the atmosphere goes on being given a seasonal cycle for it.
    assert _is_time_series(final.components["atm"]["forcing"].stl_am)
    assert final.components["atm"]["forcing"].stl_am.values.shape[0] > 1


def test_undeclared_file_forcing_is_refused_by_the_structure_check(
    model, file_forcing
):
    """An undeclared time-varying field is a named error, not a silent one.

    The declaration exists because of this: an exchanger writing an array
    into a field that is still a `TimeSeries` changes the carry's pytree
    structure, and the coupler's per-element check is what says so. Pinned
    here so that check is not weakened into accepting it -- the only right
    answer is to declare the field, which is what
    `jem.runners.build_coupler` does from the coupling table.
    """
    from jem.base.coupler import Coupler
    from jem.components import SlabOceanModel
    from jem.components.slab import SlabGrid
    from jem.exchangers import default_exchangers

    grid = SlabGrid.from_coords(model.coords.horizontal)
    components = {
        "atm": JCMComponent(model, forcing=file_forcing),
        "ocn": SlabOceanModel(grid),
    }
    coupler = Coupler(
        components, default_exchangers(components),
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE,
        calendar=CALENDAR,
    )
    with pytest.raises(RuntimeError, match="changed the structure"):
        jax.eval_shape(
            coupler.generate_trajectory_function(1), coupler.initialize()
        )


def test_validate_names_the_spec_for_an_undeclared_file_forcing(
    model, file_forcing
):
    """The pre-flight catches it too, and names the row rather than the element.

    `Exchange.validate` runs on the initial carries before anything is
    compiled, so a `TimeSeries` destination that an exchanger would overwrite
    with one array is a build-time `ValueError` naming the spec -- where the
    coupler's own check, which still fires, can only name the workflow
    element `'exchange'` at trace time.
    """
    from jem.base.coupler import Coupler
    from jem.components import SlabOceanModel
    from jem.components.slab import SlabGrid
    from jem.exchangers import default_exchangers

    grid = SlabGrid.from_coords(model.coords.horizontal)
    components = {
        "atm": JCMComponent(model, forcing=file_forcing),
        "ocn": SlabOceanModel(grid),
    }
    exchangers = default_exchangers(components)
    coupler = Coupler(
        components, exchangers,
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE,
        calendar=CALENDAR,
    )

    with pytest.raises(ValueError, match="atm.forcing.sea_surface_temperature"):
        exchangers["exchange"].validate(coupler.initialize().components)


def test_validate_passes_once_the_forcing_is_declared(model, file_forcing):
    """Declaring the field makes both ends the same pytree, and validate agrees."""
    from jem.base.coupler import Coupler
    from jem.components import SlabOceanModel
    from jem.components.slab import SlabGrid
    from jem.exchangers import default_exchangers, exchanged_fields

    grid = SlabGrid.from_coords(model.coords.horizontal)
    atm = JCMComponent(model, forcing=file_forcing)
    components = {"atm": atm, "ocn": SlabOceanModel(grid)}
    exchangers = default_exchangers(components)
    atm.set_exchanged_forcing(exchanged_fields(exchangers, atm.name))
    coupler = Coupler(
        components, exchangers,
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE,
        calendar=CALENDAR,
    )

    exchangers["exchange"].validate(coupler.initialize().components)


@pytest.mark.slow
def test_earth_slab_runs_from_the_command_line(tmp_path):
    """`+configuration=earth-slab` runs end to end with its file forcing.

    The shipped configuration that couples a from-file-forced atmosphere to a
    slab ocean, land and sea ice -- the one the structure mismatch stopped
    before it had integrated a single step. A subprocess, like the aquaplanet
    smoke test in `test_driver.py`, because Hydra's composition from the
    installed package and the `${jcm_data:}` resolver are part of what is
    being checked.
    """
    import os
    import pathlib
    import subprocess
    import sys

    repository = pathlib.Path(__file__).resolve().parents[2]
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    finished = subprocess.run(
        [sys.executable, "-m", "jem.main",
         "+configuration=earth-slab", "coupled_run=short_run"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=1800,
    )
    assert finished.returncode == 0, finished.stderr[-4000:]

    run_directories = sorted((tmp_path / "outputs").glob("*/*"))
    assert len(run_directories) == 1, run_directories
    written = sorted(path.name for path in run_directories[0].glob("*.nc"))
    assert written == [
        "atm-00000000.nc", "lnd-00000000.nc",
        "ocn-00000000.nc", "seaice-00000000.nc",
    ]

    # The polar surface the run starts from, end to end. The first record is
    # what `seaice.initialize()` published, which under the standard workflow
    # is also what the atmosphere was handed for its first two steps: it has
    # to be the observed cover, not an ice-free ocean. And the ice must still
    # be a plausible thickness two days later -- a run that begins out of
    # balance with its own freezing point answers with tens of metres of ice
    # in a single coupling step.
    import xarray as xr

    with xr.open_dataset(run_directories[0] / "seaice-00000000.nc") as sea_ice:
        first = sea_ice["ice_fraction"].isel(time=0).values
        assert float(first.max()) > 0.9
        assert float(first.mean()) > 0.01
        thickness = sea_ice["ice_thickness"].values
        assert np.isfinite(thickness).all()
        assert float(thickness.max()) < 5.0, float(thickness.max())

    with xr.open_dataset(run_directories[0] / "ocn-00000000.nc") as ocean:
        # The whole field: land carries the 288.15 K fill value, which is
        # above the floor and so cannot hide an ocean cell below it.
        sst = ocean["sea_surface_temperature"].isel(time=0).values
        assert float(sst.min()) >= constants.seawater_freezing_point_K
