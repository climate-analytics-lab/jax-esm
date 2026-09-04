"""Tests for the JCM adapter (:mod:`jem.components.jcm`).

These build a real ``jcm`` model, so they use the smallest configuration
SPEEDY physics supports -- T21 with 5 vertical levels on an aquaplanet --
and share it across the module: construction plus the first compiled step
dominates the runtime.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

from jem.base.component import (
    Component,
    CouplingTime,
    SupportsBind,
    SupportsXarray,
    TimeAxis,
)
from jem.components.jcm import JCMComponent, exchange_fields
from jem.components.jcm.component import (
    CLOCK_TOLERANCE_SECONDS,
    clock_tolerance_seconds,
)

START_DATE = jdt.to_datetime("2000-01-01")
CALENDAR = "365_day"
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")

# T21 on jcm's matching (64, 32) nodal grid; 5 levels is the fewest SPEEDY
# physics accepts (its convective cloud-top search needs kx >= 5).
LAYERS = 5
TRUNCATION = 21
GRID_SHAPE = (64, 32)


def _build_model() -> Model:
    coords = get_speedy_coords(layers=LAYERS, spectral_truncation=TRUNCATION)
    return Model(
        coords=coords,
        terrain=TerrainData.aquaplanet(coords),
        start_date=START_DATE,
        calendar=CALENDAR,
        log_level=50,
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


def _fake_speedy_diagnostics(hfluxn=10.0, evap=2.0, precnv=3.0, precls=5.0,
                             u0=1.5, v0=-2.5):
    """Build a diagnostics dict shaped like SPEEDY's, with hand-chosen values."""
    field = lambda value: jnp.full(GRID_SHAPE, value)  # noqa: E731
    return {
        "_surface_flux": SimpleNamespace(
            hfluxn=field(hfluxn), evap=field(evap),
            u0=field(u0), v0=field(v0),
        ),
        "_convection": SimpleNamespace(precnv=field(precnv)),
        "_condensation": SimpleNamespace(precls=field(precls)),
    }


def test_speedy_exchange_shapes_and_signs():
    """Sign flip and g -> kg conversion, against hand-computed values."""
    diagnostics = _fake_speedy_diagnostics()
    exchange = exchange_fields.detect(diagnostics)(diagnostics)

    assert exchange.total_heat_flux.shape == GRID_SHAPE
    # jcm's hfluxn is positive DOWNWARD into the surface; JEM is upward.
    np.testing.assert_allclose(exchange.total_heat_flux, -10.0)
    # 2 g m-2 s-1 evaporation is 0.002 kg m-2 s-1.
    np.testing.assert_allclose(exchange.evaporation, 0.002)
    # Precipitation is convective plus large-scale: (3 + 5) g m-2 s-1.
    np.testing.assert_allclose(exchange.precipitation, 0.008)
    np.testing.assert_allclose(exchange.u0, 1.5)
    np.testing.assert_allclose(exchange.v0, -2.5)
    for field in exchange:
        assert field.shape == GRID_SHAPE


def test_clock_tolerance_floor_applies_to_a_young_run():
    """Early in a run the tolerance is the fixed floor, well under one step."""
    # 8 float32 ulps of one day is ~0.08 s, so the floor wins.
    assert clock_tolerance_seconds(86400.0) == CLOCK_TOLERANCE_SECONDS
    assert clock_tolerance_seconds(0.0) == CLOCK_TOLERANCE_SECONDS


def test_clock_tolerance_grows_with_float32_resolution():
    """After decades the tolerance follows float32's spacing, by a hand value.

    At 3e9 s the float32 eps is 1.1920928955078125e-07, so eight ulps of the
    simulation time is 8 * 1.1920928955078125e-07 * 3e9 = 2861.02294921875 s
    -- above the ~256 s spacing of a float32 there, and so above the drift
    that rounding alone can produce, while still far below the one-day
    coupling step a real mismatch would be off by.
    """
    assert clock_tolerance_seconds(3.0e9) == pytest.approx(2861.02294921875)
    # Sign of the simulation time cannot shrink the window.
    assert clock_tolerance_seconds(-3.0e9) == clock_tolerance_seconds(3.0e9)


def test_echam_not_implemented_names_issue():
    """The ECHAM reader fails loudly and points at the jax-gcm issue."""
    with pytest.raises(NotImplementedError, match=r"jax-gcm#754"):
        exchange_fields.echam({"surface": object()})


def test_detect_picks_echam_on_surface_key():
    """ECHAM is detected by its own diagnostics key, not by absence of SPEEDY's."""
    assert exchange_fields.detect({"surface": object()}) is exchange_fields.echam


def test_detect_unknown_lists_keys():
    """An unrecognised physics package reports what it did publish."""
    with pytest.raises(KeyError) as excinfo:
        exchange_fields.detect({"radiation": None, "clouds": None})
    message = str(excinfo.value)
    assert "radiation" in message
    assert "clouds" in message


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
