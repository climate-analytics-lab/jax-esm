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
from jcm.date import DateData
from jcm.forcing import ForcingData, TimeSeries
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.surface.echam.surface_exchange_publisher import (
    EchamSurfaceExchange,
)
from jcm.physics.surface.surface_exchange import (
    SurfaceExchange as JcmSurfaceExchange,
)
from jcm.physics.surface.surface_exchange import surface_exchange_from
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData

from jem.base.component import (
    Component,
    CouplingTime,
    SupportsBind,
    SupportsXarray,
    TimeAxis,
)
from jem import constants
from jem.components.jcm import JCMComponent, exchange_fields
from tests.unit import _pre754_exchange_reader

START_DATE = jdt.to_datetime("2000-01-01")
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
        start_time=START_DATE,
    )


def _bound_component(model: Model) -> JCMComponent:
    component = JCMComponent(model)
    component.bind(
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    return component


def _coupling_time(step: int) -> CouplingTime:
    """Build the clock the coupler hands a component on step ``step``."""
    return CouplingTime(
        step=jnp.int32(step),
        time=START_DATE + jdt.to_timedelta(int(step), "day"),
        sim_time=jnp.float32(step * 86400.0),
        dt=86400.0,
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
        )


def test_bind_rejects_non_multiple_timestep(model):
    """The coupling interval must be a whole number of model timesteps."""
    component = JCMComponent(model)
    model_seconds = int(model.dt_si.to_timedelta().total_seconds())
    with pytest.raises(ValueError, match="whole multiple"):
        component.bind(
            coupling_timestep=jdt.to_timedelta(model_seconds + 1, "second"),
            start_date=START_DATE,
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
    assert set(carry) == {"state", "physics", "time", "step", "derived", "forcing"}
    assert carry["time"] == model.start_time
    assert int(carry["step"]) == 0
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


def test_speedy_exchange_shapes_and_signs():
    """Sign flip only: evaporation/precipitation need no unit conversion any
    more, because the #754 contract already publishes them in JEM's units
    (kg m-2 s-1) -- see the module docstring's derivation table.
    """
    diagnostics = _fake_speedy_diagnostics()
    exchange = exchange_fields.from_diagnostics(diagnostics)

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


def test_echam_wind_vector_not_implemented_names_the_reason():
    """ECHAM has no near-surface wind *vector* anywhere (only a speed), which
    predates and is independent of #754 -- see the module docstring's
    wind-vector note. The heat/water fluxes above are unaffected; only a
    caller that also needs ``u0``/``v0`` (today: ``jem.fluxes.VerosExchange``)
    is.
    """
    diagnostics = _fake_echam_diagnostics()
    with pytest.raises(NotImplementedError, match="wind VECTOR"):
        exchange_fields.from_diagnostics(diagnostics)


def test_missing_surface_exchange_raises_jcms_own_key_error():
    """A package that publishes no ``surface_exchange`` at all (Held-Suarez)
    fails with jax-gcm's own pointed error, not a bare ``KeyError``.
    """
    with pytest.raises(KeyError, match="surface_exchange"):
        exchange_fields.from_diagnostics({"radiation": None, "clouds": None})


@pytest.mark.slow
def test_speedy_new_reader_agrees_with_the_pre_754_reader(stepped):
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
    new_exchange = exchange_fields.from_diagnostics(diagnostics)

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

    with pytest.raises(NotImplementedError, match="wind VECTOR"):
        exchange_fields.from_diagnostics(diagnostics)
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

    Also pins how jcm labels that axis: absolute ``datetime64[ms]`` at the
    MIDPOINT of each averaging interval. Any component whose output is
    merged with the atmosphere's has to write the same representation.
    """
    _, _, _, diagnostics1, diagnostics2 = stepped
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), diagnostics1, diagnostics2)
    time_axis = TimeAxis(START_DATE, np.arange(2), COUPLING_TIMESTEP)

    dataset = component.to_xarray(stacked, time_axis)

    assert dataset.sizes["time"] == 2
    assert dataset.time.dtype == np.dtype("datetime64[ms]")
    np.testing.assert_array_equal(
        dataset.time.values,
        np.array(
            ["2000-01-01T12:00", "2000-01-02T12:00"], dtype="datetime64[ms]"
        ),
    )
    assert dataset.sizes["lon"], dataset.sizes["lat"] == GRID_SHAPE


@pytest.mark.slow
def test_to_xarray_rejects_a_mismatched_time_axis(component, stepped):
    """A time axis that does not match the records is a coupler-side bug."""
    _, _, _, diagnostics1, diagnostics2 = stepped
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), diagnostics1, diagnostics2)
    time_axis = TimeAxis(START_DATE, np.arange(3), COUPLING_TIMESTEP)

    with pytest.raises(ValueError, match="output records"):
        component.to_xarray(stacked, time_axis)


def test_rebinding_to_a_different_timestep_is_rejected(model):
    """One instance belongs to one coupled model; a conflicting second bind raises."""
    component = _bound_component(model)
    # The same clock again is a no-op.
    component.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    with pytest.raises(ValueError, match="already bound"):
        component.bind(
            coupling_timestep=COUPLING_TIMESTEP * 2,
            start_date=START_DATE,
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
        DateData.set_date(START_DATE)
    ).sea_surface_temperature

    collapsed = np.asarray(
        component.initialize()["forcing"].sea_surface_temperature
    )
    np.testing.assert_array_equal(collapsed, np.asarray(expected))
    # And it is that date's slice rather than any date's: a mid-year one
    # differs, so the start date is doing real work here.
    midyear = np.asarray(file_forcing.select(
        DateData.set_date(jdt.to_datetime("2000-07-01"))
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
