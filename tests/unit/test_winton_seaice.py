"""Tests for the Winton three-layer sea-ice component.

Three layers of checks:

* the thermodynamic kernels (`winton_temperature_step`, `winton_mass_step`)
  against a line-by-line numpy transcription of MITgcm ``thsice_solve4temp.F``,
  the Stefan growth law and enthalpy conservation -- in float64, scoped to each
  test with the ``x64`` fixture, because they are checks to round-off and the
  process-global flag would promote every other module's float32 arrays;
* the component's contract (the carry, the parameters, the clock, the output
  conventions) in float32, the dtype of a JEM run;
* the component under a real ``Coupler``, because only ``lax.scan`` sees a
  carry-structure mismatch.
"""

import logging

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

import jcm.constants as jcm_constants
from jem.base.component import (
    ROLE_ATTRIBUTE,
    ROLES,
    Component,
    SupportsBind,
    SupportsXarray,
)
from jem.base.coupler import Coupler
from jem.components.slab import SlabGrid, SlabOceanModel, SlabOceanParameters
from jem.components.slab.base import MASKED_SURFACE_TEMPERATURE
from jem.components.slab.winton_seaice_model import (
    IceTransportGrid,
    WintonForcing,
    WintonSeaiceModel,
    WintonSeaiceParameters,
    WintonState,
)
from jem.components.slab.winton_seaice_model.ice_transport import transport_fields
from jem.components.slab.winton_seaice_model.winton_seaice_model import (
    C_ICE,
    K_ICE,
    K_SNOW,
    KELVIN,
    L_ICE,
    RHO_ICE,
    RHO_SNOW,
    T1_from_q,
    T2_from_q,
    T_FREEZE,
    T_MELT,
    column_enthalpy_to_melt,
    q_from_T1,
    q_from_T2,
    winton_mass_step,
    winton_temperature_step,
)
from jem import Exchange, default_exchanges
from tests.unit.slab_test_utils import (
    START_DATE,
    coupling_time,
    make_grid,
    run_steps,
    time_axis,
    tree_signature,
)

DAY = 86400.0
FREEZING_K = KELVIN + T_FREEZE


@pytest.fixture
def x64():
    """Run one test in float64, restoring the previous setting afterwards."""
    with jax.enable_x64(True):
        yield


@pytest.fixture(autouse=True)
def default_precision(request):
    """Run every test that does not ask for ``x64`` in float32, the dtype of a JEM run.

    Another test module in the same worker may have left the process-global
    flag on (importing Veros sets it), and the float32 checks here -- the
    carry's dtypes, the guards that only underflow in float32 -- mean nothing
    if the flag is not pinned. Tests that choose their own precision
    (``x64``, or the parametrized transport gradient) are left alone.
    """
    if "x64" in request.fixturenames or "dtype_x64" in request.fixturenames:
        yield
        return
    with jax.enable_x64(False):
        yield


# ---------------------------------------------------------------------------
# The thermodynamic kernels
# ---------------------------------------------------------------------------


def thsice_solve4temp_numpy(h, hs, T1, T2, Ts, flux_at, dflux, sw_abs, dt, i0=0.3, ksolar=1.5):
    """Transcribe thsice_solve4temp.F for one column, external fluxes, single pass.

    ``flux_at(Ts)`` gives the non-solar downward flux at Ts; ``dflux`` its
    derivative. thsice with external fluxes does not iterate (iterMax = 1).
    Returns (T1, T2, Ts, sHeat, flxCnB, sw_to_ocean).
    """
    cpIce, Lfresh, rhoi, kIce, kSnow = C_ICE, L_ICE, RHO_ICE, K_ICE, K_SNOW
    Tmlt1, tFrz = T_MELT, T_FREEZE
    fswpen = 0.0 if hs > 0.0 else sw_abs * i0
    fswocn = fswpen * np.exp(-ksolar * h)
    fswint = fswpen - fswocn
    sHeat = sw_abs - fswpen
    k12 = 4.0 * kIce * kSnow / (kSnow * h + 4.0 * kIce * hs)
    k32 = 2.0 * kIce / h
    heat_capacity = rhoi * cpIce * h
    a10 = heat_capacity / (2 * dt) + k32 * (4 * dt * k32 + heat_capacity) / (
        6 * dt * k32 + heat_capacity
    )
    b10 = (
        -h * (rhoi * cpIce * T1 + rhoi * Lfresh * Tmlt1 / T1) / (2 * dt)
        - k32 * (4 * dt * k32 * tFrz + heat_capacity * T2) / (6 * dt * k32 + heat_capacity)
        - fswint
    )
    c10 = rhoi * Lfresh * h * Tmlt1 / (2 * dt)

    flxTexSW, dFlxdT = flux_at(Ts), dflux(Ts)
    flxNet = sHeat + flxTexSW
    a1 = a10 - k12 * dFlxdT / (k12 - dFlxdT)
    b1 = b10 - k12 * (flxNet - dFlxdT * Ts) / (k12 - dFlxdT)
    tIc1_free = -(b1 + np.sqrt(b1 * b1 - 4 * a1 * c10)) / (2 * a1)
    dTsrf = (flxNet + k12 * (tIc1_free - Ts)) / (k12 - dFlxdT)
    tSrf_free = Ts + dTsrf

    melting = tSrf_free > 0.0
    if melting:
        a1m = a10 + k12
        b1m = b10 - k12 * 0.0
        tIc1 = (-b1m - np.sqrt(b1m * b1m - 4 * a1m * c10)) / (2 * a1m)
        tSrf = 0.0
    else:
        tIc1 = tIc1_free
        tSrf = tSrf_free

    tIc2 = (2 * dt * k32 * (tIc1 + 2 * tFrz) + heat_capacity * T2) / (6 * dt * k32 + heat_capacity)
    flxFinal = flux_at(tSrf)
    fct = k12 * (tSrf - tIc1)
    sHeat_out = (sHeat + flxFinal - fct) if melting else 0.0
    flxCnB = 4 * kIce * (tIc2 - tFrz) / h
    return tIc1, tIc2, tSrf, sHeat_out, flxCnB, fswocn


def pinned_flux(T_target, gain=1e4):
    """Surface flux that pins the surface temperature: F = gain (T_target - Ts)."""
    return lambda Ts: (gain * (T_target - Ts), -gain * jnp.ones_like(Ts))


def column_step(h, hs, T1, T2, Ts, flux, sw, F_b, snowfall, dt, n_iter=1):
    """One thermodynamic step of a single column: temperature solve, then mass step."""
    T1, T2, Ts, M_s, F_cb, sw_ocn = winton_temperature_step(
        h, hs, T1, T2, Ts, flux, sw, dt, n_iter=n_iter
    )
    h, hs, q1, q2, e_ocn, vanished = winton_mass_step(
        h, hs, q_from_T1(T1), q_from_T2(T2), M_s, F_b, F_cb, snowfall, dt
    )
    return h, hs, T1_from_q(q1), T2_from_q(q2), Ts, sw_ocn, e_ocn, vanished


def test_temperature_step_matches_thsice_transcription(x64):
    h, T1, T2, Ts, dt = 1.5, -8.0, -4.0, -15.0, 6 * 3600.0
    for hs in (0.0, 0.2):
        for name, F0, dF, sw in [
            ("cold", -60.0, -20.0, 0.0),
            ("warm", 80.0, -20.0, 200.0),
            ("mild", 10.0, -25.0, 50.0),
        ]:
            got = winton_temperature_step(
                jnp.array(h), jnp.array(hs), jnp.array(T1), jnp.array(T2), jnp.array(Ts),
                lambda T, F0=F0, dF=dF: (F0 + dF * (T - Ts), dF * jnp.ones_like(T)),
                jnp.array(sw), dt, n_iter=1,
            )
            ref = thsice_solve4temp_numpy(
                h, hs, T1, T2, Ts, lambda T, F0=F0, dF=dF: F0 + dF * (T - Ts),
                lambda T, dF=dF: dF, sw, dt,
            )
            for g, r, what in zip(got[:5], ref[:5], ["T1", "T2", "Ts", "M_s", "F_cb"]):
                assert abs(float(g) - r) < 1e-9 * max(1.0, abs(r)), (hs, name, what, float(g), r)


def integrate_column(n_steps, dt, h, hs, T1, T2, Ts, flux_of_step, sw_of_step, F_b_of_step, n_iter=1):
    """Integrate one column for ``n_steps`` with ``lax.scan``; return the final state and the per-step outputs.

    ``flux_of_step(k)`` and friends give the step's forcing from the traced
    step index, so one compilation covers the whole integration. The outputs
    are ``(sw_ocn, e_ocn, vanished, F_atm)`` per step.
    """
    zero = jnp.zeros(())

    def body(carry, k):
        h, hs, T1, T2, Ts = carry
        flux, sw, F_b = flux_of_step(k), sw_of_step(k), F_b_of_step(k)
        h, hs, T1, T2, Ts, sw_ocn, e_ocn, vanished = column_step(
            h, hs, T1, T2, Ts, flux, sw, F_b, zero, dt, n_iter=n_iter
        )
        F_atm, _ = flux(Ts)
        return (h, hs, T1, T2, Ts), (sw_ocn, e_ocn, vanished, F_atm)

    return jax.lax.scan(body, (h, hs, T1, T2, Ts), jnp.arange(n_steps))


def test_stefan_growth_law(x64):
    """Cold pinned surface, no ocean flux: h^2 - h0^2 ~ 2 K dT t / (rho L)."""
    dt = 3 * 3600.0
    T_s = -30.0
    days = 200
    (h, _, _, _, Ts), _ = integrate_column(
        int(days * DAY / dt), dt, jnp.array(1.0), jnp.array(0.0),
        jnp.array(-15.0), jnp.array(-8.0), jnp.array(T_s),
        lambda k: pinned_flux(T_s), lambda k: 0.0, lambda k: 0.0,
    )
    # Stefan with effective latent heat of the lower ice (enthalpy at Tf)
    L_eff = float(q_from_T2(T_FREEZE))
    h_stefan = np.sqrt(1.0 + 2 * K_ICE * (T_FREEZE - T_s) * days * DAY / (RHO_ICE * L_eff))
    assert abs(float(h) - h_stefan) / h_stefan < 0.03, (float(h), h_stefan)
    assert float(Ts) < T_s + 0.05


def test_snow_insulates_growth(x64):
    """Same pinned cold surface: an insulating snow layer slows Stefan-type growth."""
    dt = 3 * 3600.0
    T_s = -30.0

    def grow(hs0):
        (h, *_), _ = integrate_column(
            int(60 * DAY / dt), dt, jnp.array(1.0), jnp.array(hs0),
            jnp.array(-15.0), jnp.array(-8.0), jnp.array(T_s),
            lambda k: pinned_flux(T_s), lambda k: 0.0, lambda k: 0.0,
        )
        return float(h)

    assert grow(0.3) < grow(0.0)


def test_energy_conservation_over_melt_and_growth(x64):
    """Change in column enthalpy equals time-integrated (atm flux + basal flux - energy to ocean)."""
    dt = 3 * 3600.0
    n_steps = 600
    h0, hs0 = jnp.array(3.0), jnp.array(0.0)
    T1, T2, Ts = jnp.array(-10.0), jnp.array(-5.0), jnp.array(-20.0)

    def column_energy(h, hs, T1, T2):
        return -float(column_enthalpy_to_melt(h, hs, q_from_T1(T1), q_from_T2(T2)))

    def warm(k):
        return (k // 100) % 2 == 1

    def flux_of_step(k):
        # A traced choice between two pinned fluxes: the target and gain are
        # selected, the flux itself is one function.
        target = jnp.where(warm(k), 1.0, -25.0)
        return lambda T: (50.0 * (target - T), -50.0 * jnp.ones_like(T))

    def sw_of_step(k):
        return jnp.where(warm(k), 50.0, 0.0)

    def F_b_of_step(k):
        return jnp.where(warm(k), 10.0, 0.0)

    (h, hs, T1n, T2n, _), (sw_ocn, e_ocn, _, F_atm) = integrate_column(
        n_steps, dt, h0, hs0, T1, T2, Ts, flux_of_step, sw_of_step, F_b_of_step
    )
    steps = jnp.arange(n_steps)
    net_in = float(jnp.sum(
        (F_atm + sw_of_step(steps) - sw_ocn + F_b_of_step(steps)) * dt - e_ocn
    ))
    assert float(h) > 0.05, "ice vanished; test forcing too warm"
    E0 = column_energy(h0, hs0, T1, T2)
    E1 = column_energy(h, hs, T1n, T2n)
    assert abs((E1 - E0) - net_in) < 2e-3 * abs(net_in) + 1e3, ((E1 - E0), net_in)


def test_complete_melt_returns_excess(x64):
    dt = DAY / 4
    h, hs = jnp.array(0.3), jnp.array(0.0)
    T1, T2, Ts = jnp.array(-2.0), jnp.array(-1.9), jnp.array(-1.0)
    total_e_ocn = 0.0
    # An eager loop that stops at melt-out: a vanished column has no thickness
    # left to integrate, so it cannot be scanned past that point.
    for _ in range(40):
        h, hs, T1, T2, Ts, _, e_ocn, vanished = column_step(
            h, hs, T1, T2, Ts, pinned_flux(10.0, gain=100.0), jnp.array(300.0),
            jnp.array(50.0), jnp.array(0.0), dt,
        )
        total_e_ocn += float(e_ocn)
        if bool(vanished) or float(h) < 1e-9:
            break
    assert np.isfinite(total_e_ocn)
    assert float(h) == 0.0
    assert total_e_ocn > 0.0


def test_enthalpy_inversion_roundtrip(x64):
    for T in [-30.0, -10.0, -2.0, -0.2]:
        assert abs(float(T1_from_q(q_from_T1(jnp.array(T)))) - T) < 1e-9
        assert abs(float(T2_from_q(q_from_T2(jnp.array(T)))) - T) < 1e-9


def test_gradient_through_kernels_is_finite(x64):
    def final_thickness(T_s):
        h, hs = jnp.array(1.0), jnp.array(0.0)
        T1, T2, Ts = jnp.array(-10.0), jnp.array(-5.0), T_s
        for _ in range(20):
            h, hs, T1, T2, Ts, *_ = column_step(
                h, hs, T1, T2, Ts, pinned_flux(T_s), jnp.array(20.0), jnp.array(5.0),
                jnp.array(0.0), DAY / 4,
            )
        return h

    g = jax.grad(final_thickness)(jnp.array(-20.0))
    assert jnp.isfinite(g) and float(g) < 0.0   # colder surface -> thicker ice


def test_a_melting_out_column_has_a_finite_gradient_in_float32():
    """The safe-denominator guards must survive float32, the dtype of a JEM run.

    A floor of 1e-300 underflows to exactly zero in float32, so the discarded
    branch of a ``where`` divided by zero and reverse-mode AD turned its
    ``0 * inf`` into NaN -- but only when a column vanished, and only outside
    float64, which is where every other test in this module runs.
    """
    assert jnp.zeros(()).dtype == jnp.float32

    def surplus(h, M_s):
        out = winton_mass_step(
            h, jnp.float32(0.0), q_from_T1(jnp.float32(-1.0)), q_from_T2(jnp.float32(-1.0)),
            M_s, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), 3600.0,
        )
        assert bool(out[5]), "the column should have melted out"
        return out[4]

    grads = jax.grad(surplus, argnums=(0, 1))(jnp.float32(0.02), jnp.float32(3.0e4))
    assert all(bool(jnp.isfinite(g)) for g in grads), grads


# ---------------------------------------------------------------------------
# Building blocks for the component tests
# ---------------------------------------------------------------------------


def make_forcing(grid, **overrides) -> WintonForcing:
    """Return a cold-air forcing on ``grid``, with ``overrides`` (scalars or arrays) on top."""
    base = WintonForcing.initial(grid.shape)
    updates = {
        "rlds": 180.0,
        "air_temperature": 245.0,
    }
    updates.update(overrides)
    zeros = jnp.zeros(grid.shape)
    return base.replace(**{
        name: value if jnp.ndim(value) else zeros + value
        for name, value in updates.items()
    })


def ice_state(grid, thickness, snow=0.0, fraction=1.0, T=-5.0) -> WintonState:
    """Return a uniform ice cover over the ocean cells of ``grid``."""
    zeros = jnp.zeros(grid.shape)
    ocean = grid.binary_mask == 0.0
    return WintonState(
        jnp.where(ocean, thickness, 0.0),
        jnp.where(ocean, snow, 0.0),
        jnp.where(ocean, fraction, 0.0),
        zeros + T, zeros + T / 2, zeros + T,
    )


def with_state_and_forcing(model, state=None, forcing=None):
    """Return the model's initial carry with its state and/or forcing replaced."""
    carry = model.initialize()
    if state is not None:
        carry["state"] = state
    if forcing is not None:
        carry["forcing"] = forcing
    return carry


@pytest.fixture
def grid():
    return make_grid()


@pytest.fixture
def half_land_grid():
    fraction = np.zeros((4, 3))
    fraction[2:, :] = 1.0
    return make_grid(fractional_mask=fraction)


def global_grid(nx=24, ny=12, land_blocks=True):
    """Return a coarse global lon/lat grid with polar land rows and an island."""
    land = np.zeros((nx, ny))
    land[:, 0] = land[:, -1] = 1.0
    if land_blocks:
        land[5:8, 4:7] = 1.0
    latitude = np.deg2rad(np.linspace(-80, 80, ny))
    longitude = np.deg2rad(np.linspace(0, 360, nx, endpoint=False))
    return SlabGrid(
        fractional_mask=jnp.asarray(land),
        latitude_radian=jnp.asarray(np.broadcast_to(latitude[None, :], (nx, ny))),
        longitude_radian=jnp.asarray(np.broadcast_to(longitude[:, None], (nx, ny))),
        threshold=0.5,
        longitude_axis_radian=longitude,
        latitude_axis_radian=latitude,
    )


# ---------------------------------------------------------------------------
# The component contract
# ---------------------------------------------------------------------------


def test_the_model_satisfies_the_component_protocols(grid):
    model = WintonSeaiceModel(grid)
    assert isinstance(model, Component)
    assert isinstance(model, SupportsBind)
    assert isinstance(model, SupportsXarray)
    # The name the standard coupling wires the sea ice under.
    assert model.name == "seaice"
    assert WintonSeaiceModel(grid, name="ice").name == "ice"


def test_the_component_holds_no_clock_or_timestep(grid):
    """The coupler owns the clock: neither the model nor its state carries one."""
    model = WintonSeaiceModel(grid)
    carry = model.initialize()
    assert not hasattr(model, "timestep")
    assert not hasattr(model, "start_datetime")
    assert set(carry) == {"params", "state", "forcing", "derived"}
    assert not any("time" in field for field in WintonState.__dataclass_fields__)


def test_step_advances_by_the_coupling_step_it_is_handed(grid):
    """Cold-air growth over a long coupling step exceeds that over a short one."""
    model = WintonSeaiceModel(grid)
    carry = with_state_and_forcing(model, ice_state(grid, 1.0), make_forcing(grid))
    growth = {}
    for dt in (6 * 3600.0, 4 * DAY):
        stepped, _ = model.step(carry, coupling_time(0, dt=dt))
        growth[dt] = float(stepped["state"].ice_thickness[0, 1] - 1.0)
    assert 0.0 < growth[6 * 3600.0] < growth[4 * DAY]


def test_parameters_travel_in_the_carry_and_a_step_returns_them(grid):
    params = WintonSeaiceParameters(ice_albedo=0.7)
    model = WintonSeaiceModel(grid, params)
    carry = model.initialize()
    assert carry["params"] is params
    stepped, diagnostics = model.step(carry, coupling_time(0))
    assert stepped["params"] is params
    assert set(diagnostics) == {"state", "forcing", "derived"}
    # An explicitly passed parameter set is what the carry carries.
    other = WintonSeaiceParameters(ice_albedo=0.5)
    assert model.initialize(other)["params"] is other


@pytest.mark.parametrize("transport", [False, True])
def test_step_shapes_and_dtypes_are_stable(grid, transport):
    """What lax.scan requires: the carry that comes out is the one that went in."""
    model = WintonSeaiceModel(grid, WintonSeaiceParameters(initial_ice_thickness=1.0), transport=transport)
    carry = model.initialize()
    stepped, _ = model.step(carry, coupling_time(0))
    assert tree_signature(stepped) == tree_signature(carry)


def test_the_initial_forcing_leaves_an_ice_cover_at_the_freezing_point_alone(grid):
    """Coupling is lagged, so the first step runs on the initial forcing.

    That forcing is the one that exerts no melt or growth on ice that is
    already at the freezing point; a supplier that has not written a field yet
    must not kick the ice.
    """
    model = WintonSeaiceModel(grid, WintonSeaiceParameters(initial_ice_thickness=1.0))
    carry = model.initialize()
    stepped, _ = model.step(carry, coupling_time(0))
    change = jnp.abs(stepped["state"].ice_thickness - carry["state"].ice_thickness)
    assert float(change.max()) < 0.02, float(change.max())
    forcing = carry["forcing"]
    np.testing.assert_allclose(np.asarray(forcing.air_temperature), FREEZING_K)
    np.testing.assert_allclose(np.asarray(forcing.sea_surface_temperature), FREEZING_K)


def test_land_cells_carry_no_ice_and_report_the_masked_temperature(half_land_grid):
    model = WintonSeaiceModel(half_land_grid, WintonSeaiceParameters(initial_ice_thickness=1.0))
    carry = model.initialize()
    land = np.asarray(half_land_grid.binary_mask == 1.0)
    stepped, _ = model.step(
        with_state_and_forcing(model, forcing=make_forcing(half_land_grid)), coupling_time(0)
    )
    for got in (carry, stepped):
        assert not np.any(np.asarray(got["state"].ice_thickness)[land])
        assert not np.any(np.asarray(got["state"].ice_fraction)[land])
        np.testing.assert_allclose(
            np.asarray(got["derived"].ice_surface_temperature_K)[land], MASKED_SURFACE_TEMPERATURE, rtol=1e-6
        )
    ocean_surface = np.asarray(stepped["derived"].ice_surface_temperature_K)[~land]
    assert np.all(ocean_surface < MASKED_SURFACE_TEMPERATURE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_closing_thickness", 0.0),
        ("lead_closing_thickness", float("nan")),
        ("min_ice_thickness", -1.0),
        ("min_ice_fraction", -0.1),
        ("initial_ice_thickness", -1.0),
        ("initial_ice_thickness", float("inf")),
        ("ice_albedo", 1.5),
        ("snow_albedo", -0.1),
        ("i0_fraction", 2.0),
        ("ksolar", -1.0),
        ("emissivity", float("nan")),
        ("transport_diffusivity", -1.0),
        ("n_substeps", 0),
        ("n_flux_iterations", 0),
        ("transport_n_substeps", 0),
    ],
)
def test_parameters_outside_the_schemes_domain_are_rejected(grid, field, value):
    with pytest.raises(ValueError, match=field):
        WintonSeaiceModel(grid, WintonSeaiceParameters(**{field: value}))


def test_transport_metrics_must_match_the_grid(grid):
    wrong = IceTransportGrid(dx=jnp.ones((2, 2)), dy=jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="dx"):
        WintonSeaiceModel(grid, transport=wrong)


def test_transport_metrics_of_a_lon_lat_grid():
    grid = global_grid(nx=24, ny=12)
    metrics = IceTransportGrid.from_grid(grid)
    radius = jcm_constants.rearth
    dx, dy = np.asarray(metrics.dx), np.asarray(metrics.dy)
    assert dx.shape == dy.shape == grid.shape
    latitude = np.asarray(grid.latitude_axis_radian)
    np.testing.assert_allclose(dx[3], radius * np.cos(latitude) * 2 * np.pi / 24, rtol=1e-5)
    # Interior rows have the spacing of the latitude axis; a cell is never wider
    # in y than the whole pole-to-pole extent.
    np.testing.assert_allclose(dy[0, 1:-1], radius * np.deg2rad(160.0 / 11), rtol=1e-5)
    assert metrics.cyclic_x


def test_transport_metrics_cannot_be_derived_for_a_curvilinear_grid():
    longitude = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0])[:, None] + np.arange(3)[None, :])
    latitude = np.deg2rad(np.array([-60.0, 0.0, 60.0])[None, :] + np.arange(4)[:, None])
    curvilinear = SlabGrid(
        fractional_mask=jnp.zeros((4, 3)),
        latitude_radian=jnp.asarray(latitude),
        longitude_radian=jnp.asarray(longitude),
        threshold=0.5,
    )
    with pytest.raises(ValueError, match="separable"):
        IceTransportGrid.from_grid(curvilinear)
    with pytest.raises(ValueError, match="separable"):
        WintonSeaiceModel(curvilinear, transport=True)


def test_bind_is_accepted_and_changes_nothing(grid):
    """The model has no climatology to sample and no timestep to reconcile."""
    model = WintonSeaiceModel(grid)
    model.bind(coupling_timestep=jdt.to_timedelta(6, "hour"), start_date=START_DATE)
    assert model.start_year_fraction == 0.0


# ---------------------------------------------------------------------------
# Parameters: differentiable process and initial-condition parameters
# ---------------------------------------------------------------------------


def _thickness_after_one_step(model, carry):
    stepped, _ = model.step(carry, coupling_time(0))
    return jnp.sum(stepped["state"].ice_thickness * stepped["state"].ice_fraction)


def test_a_process_parameter_leaf_in_the_carry_has_a_gradient(grid):
    """Sunlit ice melts faster the darker it is: d(volume)/d(albedo) < 0."""
    model = WintonSeaiceModel(grid)
    forcing = make_forcing(grid, rsds=300.0, rlds=300.0, air_temperature=272.0)
    carry = with_state_and_forcing(model, ice_state(grid, 1.0, T=-0.5), forcing)

    def loss(albedo):
        varied = dict(carry, params=carry["params"].replace(ice_albedo=albedo))
        return _thickness_after_one_step(model, varied)

    gradient = jax.grad(loss)(jnp.float32(0.6))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) > 0.0   # brighter ice absorbs less: more volume survives


def test_the_surface_flux_parameters_are_differentiable_leaves(grid):
    model = WintonSeaiceModel(grid)
    forcing = make_forcing(grid, rlds=250.0, air_temperature=260.0)
    carry = with_state_and_forcing(model, ice_state(grid, 1.0), forcing)

    def loss(chs):
        flux = carry["params"].surface_flux.replace(chs=chs)
        varied = dict(carry, params=carry["params"].replace(surface_flux=flux))
        return _thickness_after_one_step(model, varied)

    gradient = jax.grad(loss)(carry["params"].surface_flux.chs)
    assert bool(jnp.isfinite(gradient)) and float(jnp.abs(gradient)) > 0.0


def test_the_snow_albedo_follows_the_ice_albedo_unless_it_is_given(grid):
    model = WintonSeaiceModel(grid)
    forcing = make_forcing(grid, rsds=250.0, rlds=280.0, air_temperature=265.0)
    state = ice_state(grid, 1.0, snow=0.1)
    results = {}
    for label, params in {
        "follows": WintonSeaiceParameters(ice_albedo=0.5),
        "explicit": WintonSeaiceParameters(ice_albedo=0.5, snow_albedo=0.9),
    }.items():
        carry = model.initialize(params)
        carry["state"], carry["forcing"] = state, forcing
        _, diagnostics = model.step(carry, coupling_time(0))
        results[label] = float(diagnostics["derived"].ice_albedo[0, 1])
    assert results["follows"] == pytest.approx(0.5, abs=1e-6)
    assert results["explicit"] == pytest.approx(0.9, abs=1e-6)


def test_initial_ice_thickness_is_varied_through_initialize(grid):
    """An initial-condition parameter reaches the state only through initialize."""
    model = WintonSeaiceModel(grid)

    def loss(thickness):
        carry = model.initialize(model.params.replace(initial_ice_thickness=thickness))
        return _thickness_after_one_step(model, carry)

    assert float(jax.grad(loss)(jnp.float32(1.0))) > 0.0

    # Replacing the leaf in a carry that exists changes nothing: the value has
    # already been copied into the state.
    carry = model.initialize()
    replaced = dict(carry, params=carry["params"].replace(initial_ice_thickness=2.0))
    np.testing.assert_array_equal(
        np.asarray(replaced["state"].ice_thickness), np.asarray(carry["state"].ice_thickness)
    )


def test_a_gradient_through_two_steps_of_a_melting_ice_cover_is_finite(grid):
    """The unit-level gradient check, through a scan and across a melt-out."""
    model = WintonSeaiceModel(grid)
    forcing = make_forcing(
        grid, rsds=300.0, rlds=320.0, air_temperature=280.0,
        sea_surface_temperature=KELVIN + 2.0, ice_frazil_melt_energy=-5e6,
    )
    carry = with_state_and_forcing(model, ice_state(grid, 0.03, fraction=0.8, T=-0.5), forcing)

    def loss(thickness):
        state = carry["state"].replace(ice_thickness=jnp.where(carry["state"].ice_fraction > 0, thickness, 0.0))
        current = dict(carry, state=state)
        for step in range(2):
            current, _ = model.step(current, coupling_time(step))
        return jnp.sum(current["state"].ice_thickness) + jnp.sum(current["derived"].ocean_heat_flux_up)

    gradient = jax.grad(loss)(jnp.float32(0.03))
    assert bool(jnp.isfinite(gradient))


# ---------------------------------------------------------------------------
# The energy budget (float64, exact to the accuracy of the surface solve)
# ---------------------------------------------------------------------------


def test_energy_budget_closes_in_every_regime(x64):
    """One coupling step: dE_ice/dt - ocean_heat_flux_up + ocean_frazil_heating - atm_sea_heat_flux = 0 per cell,
    for cold growth, melt at several fractions, frazil-only, snowfall, melt-out of thin ice and sliver disposal.
    """
    grid = make_grid(fractional_mask=np.array([[1, 0, 0]] * 4, dtype=float))
    model = WintonSeaiceModel(grid)
    ocean = np.asarray(grid.binary_mask == 0.0)
    warm = dict(rsds=250.0, rlds=310.0, air_temperature=278.0)
    regimes = {
        "cold growth": (ice_state(grid, 1.5, T=-8.0), dict(rlds=180.0, air_temperature=245.0)),
        "melt, full cover": (
            ice_state(grid, 1.0, T=-1.0),
            dict(warm, sea_surface_temperature=KELVIN + 1.0, ice_frazil_melt_energy=-2e6),
        ),
        "melt, half cover": (
            ice_state(grid, 1.0, fraction=0.5, T=-1.0),
            dict(warm, sea_surface_temperature=KELVIN + 1.0, ice_frazil_melt_energy=-2e6),
        ),
        "frazil only": (ice_state(grid, 0.0, fraction=0.0), dict(ice_frazil_melt_energy=3e6)),
        "weak frazil": (ice_state(grid, 0.0, fraction=0.0), dict(ice_frazil_melt_energy=5e5)),
        "snowfall": (ice_state(grid, 1.0, T=-8.0), dict(rlds=180.0, air_temperature=245.0, snowfall=1e-5)),
        "thin ice melts out": (
            ice_state(grid, 0.03, fraction=0.8, T=-0.5),
            dict(rsds=300.0, rlds=320.0, air_temperature=280.0,
                 sea_surface_temperature=KELVIN + 2.0, ice_frazil_melt_energy=-5e6),
        ),
        "sliver, no freezing": (ice_state(grid, 0.005, snow=0.02, fraction=0.5, T=-3.0), dict(rlds=250.0)),
        "sliver, freezing": (
            ice_state(grid, 0.005, fraction=0.005, T=-3.0), dict(ice_frazil_melt_energy=2e5)
        ),
    }
    for name, (state, overrides) in regimes.items():
        forcing = make_forcing(grid, **overrides)
        stepped, _ = model.step(with_state_and_forcing(model, state, forcing), coupling_time(0))
        derived = stepped["derived"]
        residual = np.asarray(
            derived.ice_energy_tendency - derived.ocean_heat_flux_up
            + derived.ocean_frazil_heating - forcing.atm_sea_heat_flux
        )[ocean]
        # Roundoff on ~1e8 J/m2 is ~1e-6 W/m2.
        assert np.abs(residual).max() < 1e-3, f"{name}: energy residual {np.abs(residual).max():.3e} W/m2"
        new = stepped["state"]
        assert float(jnp.max(new.ice_fraction)) <= 1.0 and float(jnp.min(new.ice_thickness)) >= 0.0
    # Weak frazil must make ice rather than vanish.
    stepped, _ = model.step(
        with_state_and_forcing(model, regimes["weak frazil"][0], make_forcing(grid, ice_frazil_melt_energy=5e5)),
        coupling_time(0),
    )
    assert float(jnp.max(stepped["state"].ice_fraction * stepped["state"].ice_thickness)) > 0.0


def test_ice_growth_removes_water_from_the_ocean_and_melt_returns_it(grid):
    model = WintonSeaiceModel(grid)
    cold = with_state_and_forcing(model, ice_state(grid, 1.0), make_forcing(grid))
    _, growing = model.step(cold, coupling_time(0))
    assert float(growing["derived"].ocean_freshwater_flux_up[0, 1]) > 0.0
    warm = make_forcing(
        grid, rsds=300.0, rlds=320.0, air_temperature=280.0,
        sea_surface_temperature=KELVIN + 2.0, ice_frazil_melt_energy=-5e6,
    )
    _, melting = model.step(
        with_state_and_forcing(model, ice_state(grid, 0.5, T=-0.5), warm), coupling_time(0)
    )
    assert float(melting["derived"].ocean_freshwater_flux_up[0, 1]) < 0.0


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def random_ice_state(grid, seed=0):
    rng = np.random.default_rng(seed)
    shape = grid.shape
    ocean = np.asarray(grid.binary_mask == 0.0)
    h = np.where(ocean, rng.uniform(0.5, 3.0, shape) * (rng.uniform(size=shape) > 0.4), 0.0)
    f = np.where(h > 0, rng.uniform(0.3, 1.0, shape), 0.0)
    hs = np.where(h > 0, rng.uniform(0.0, 0.3, shape), 0.0)
    T1 = np.where(h > 0, rng.uniform(-20, -3, shape), T_FREEZE)
    T2 = np.where(h > 0, rng.uniform(-10, -2, shape), T_FREEZE)
    Ts = np.where(h > 0, rng.uniform(-30, -1, shape), T_FREEZE)
    u = rng.uniform(-0.3, 0.3, shape)
    v = rng.uniform(-0.3, 0.3, shape)
    return WintonState(*(jnp.asarray(a) for a in (h, hs, f, T1, T2, Ts))), jnp.asarray(u), jnp.asarray(v)


def test_ice_transport_conserves_mass_and_energy(x64):
    """Transport moves ice between cells but changes neither the global ice/snow mass nor the enthalpy."""
    grid = global_grid()
    diffusive = WintonSeaiceParameters(transport_diffusivity=5e4)
    with_transport = WintonSeaiceModel(grid, diffusive, transport=True)
    without = WintonSeaiceModel(grid, diffusive)
    state, u, v = random_ice_state(grid)
    forcing = WintonForcing.initial(grid.shape).replace(
        air_temperature=jnp.zeros(grid.shape) + 250.0, ice_velocity_u=u, ice_velocity_v=v
    )
    moved, _ = with_transport.step(with_state_and_forcing(with_transport, state, forcing), coupling_time(0))
    stayed, _ = without.step(with_state_and_forcing(without, state, forcing), coupling_time(0))

    area = np.asarray(with_transport.transport.dx * with_transport.transport.dy) * np.asarray(grid.binary_mask == 0.0)

    def mass(s):
        return float((np.asarray(s.ice_fraction) * (
            RHO_ICE * np.asarray(s.ice_thickness) + RHO_SNOW * np.asarray(s.snow_thickness)
        ) * area).sum())

    def energy(s):
        return float((np.asarray(s.ice_fraction) * np.asarray(column_enthalpy_to_melt(
            s.ice_thickness, s.snow_thickness,
            q_from_T1(s.upper_ice_temperature), q_from_T2(s.lower_ice_temperature),
        )) * area).sum())

    out_t, out_0 = moved["state"], stayed["state"]
    displaced = np.abs(np.asarray(out_t.ice_fraction * out_t.ice_thickness - out_0.ice_fraction * out_0.ice_thickness)).sum()
    assert displaced > 0.0, "transport did nothing"
    assert abs(mass(out_t) - mass(out_0)) < 1e-6 * mass(out_0)
    assert abs(energy(out_t) - energy(out_0)) < 1e-6 * abs(energy(out_0))
    assert float(out_t.ice_fraction.max()) <= 1.0 and float(out_t.ice_thickness.min()) >= 0.0
    ocean = np.asarray(grid.binary_mask == 0.0)
    assert not np.any(np.asarray(out_t.ice_thickness)[~ocean] > 0)
    icy = np.asarray(out_t.ice_thickness) > 0.05
    assert np.all(np.asarray(out_t.ice_surface_temperature)[icy] < -0.5), "surface temperature lost its sign"
    assert np.all(np.asarray(out_t.upper_ice_temperature)[icy] < -0.5)
    # The transport step exchanges nothing with the ocean, and says so.
    assert float(jnp.abs(moved["derived"].ice_energy_transport).max()) > 0.0
    assert float(jnp.abs(stayed["derived"].ice_energy_transport).max()) == 0.0
    np.testing.assert_allclose(
        np.asarray(moved["derived"].ocean_heat_flux_up), np.asarray(stayed["derived"].ocean_heat_flux_up)
    )


def test_ice_transport_limiter_positivity_and_conservation(x64):
    """With Courant numbers above one and a huge diffusivity the outflow limiter must keep every field
    non-negative and conserve it exactly (no clipping): the property the scheme advertises.
    """
    rng = np.random.default_rng(3)
    nx, ny = 20, 10
    ocean = np.ones((nx, ny), bool)
    ocean[:, 0] = ocean[:, -1] = False
    ocean[6:9, 3:6] = False
    dx = np.full((nx, ny), 2.0e5)
    dy = np.full((nx, ny), 2.0e5)
    area = dx * dy
    X1 = np.where(ocean, rng.uniform(0, 3, (nx, ny)) * (rng.uniform(size=(nx, ny)) > 0.5), 0.0)
    X2 = np.where(ocean, rng.uniform(0, 1, (nx, ny)), 0.0)
    u = rng.uniform(-3, 3, (nx, ny))
    v = rng.uniform(-3, 3, (nx, ny))           # Courant ~1.3 per day
    out = transport_fields(
        (jnp.asarray(X1), jnp.asarray(X2)), jnp.asarray(u), jnp.asarray(v), jnp.asarray(dx),
        jnp.asarray(dy), jnp.asarray(ocean), DAY, diffusivity=1e6, n_substeps=1,
    )
    for X, Y in zip((X1, X2), out):
        Y = np.asarray(Y)
        assert Y.min() >= 0.0
        assert abs((Y * area).sum() - (X * area).sum()) <= 1e-12 * (X * area).sum()
        assert not np.any(Y[~ocean] != 0.0)
        assert np.abs(Y - X).sum() > 0.0     # transport did happen


def test_derived_fields_match_the_final_state_with_transport(grid):
    model = WintonSeaiceModel(grid, transport=True)
    carry = model.initialize()
    ramp = jnp.linspace(0.2, 1.0, grid.shape[0])[:, None] * jnp.ones(grid.shape)
    state = ice_state(grid, 1.0, T=-5.0).replace(ice_fraction=ramp)
    forcing = make_forcing(
        grid, ice_velocity_u=0.3, ice_velocity_v=-0.2, air_temperature=250.0
    )
    stepped, _ = model.step(dict(carry, state=state, forcing=forcing), coupling_time(0))
    new, derived = stepped["state"], stepped["derived"]
    assert bool(jnp.allclose(derived.ice_fraction, new.ice_fraction))
    assert bool(jnp.allclose(derived.ice_volume, new.ice_fraction * new.ice_thickness))
    assert bool(jnp.allclose(derived.ice_surface_temperature_K, new.ice_surface_temperature + KELVIN))
    assert float(jnp.abs(derived.ice_energy_transport).max()) > 0.0


@pytest.mark.parametrize("dtype_x64", [False, True])
def test_gradient_through_transport_is_finite_and_nonzero(dtype_x64):
    """A non-conserved loss of a non-uniform state has a real, finite sensitivity to the drift.

    (A uniform state, or a loss made only of conserved totals, correctly has a
    zero gradient: transport then changes nothing the loss can see.)
    """
    with jax.enable_x64(dtype_x64):
        grid = global_grid(nx=12, ny=8, land_blocks=False)
        model = WintonSeaiceModel(grid, transport=True)
        state, u, v = random_ice_state(grid, seed=1)
        forcing = make_forcing(grid, ice_velocity_u=u, ice_velocity_v=v, air_temperature=250.0)
        carry = with_state_and_forcing(model, state, forcing)

        def loss(velocity_u):
            varied = dict(carry, forcing=carry["forcing"].replace(ice_velocity_u=velocity_u))
            stepped, _ = model.step(varied, coupling_time(0))
            return jnp.sum((stepped["state"].ice_thickness * stepped["state"].ice_fraction) ** 2)

        gradient = jax.grad(loss)(carry["forcing"].ice_velocity_u)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.abs(gradient).max()) > 0.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def output_dataset():
    grid = make_grid(fractional_mask=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=float))
    model = WintonSeaiceModel(grid, WintonSeaiceParameters(initial_ice_thickness=1.0), transport=True)
    carry = with_state_and_forcing(model, forcing=make_forcing(grid, ice_velocity_u=0.1))
    _, diagnostics = run_steps(model, carry, 3)
    return model.to_xarray(diagnostics, time_axis(3))


def test_every_output_variable_carries_a_role(output_dataset):
    for name, variable in output_dataset.data_vars.items():
        assert variable.attrs[ROLE_ATTRIBUTE] in ROLES, name
        assert "units" in variable.attrs, name


def test_the_prefixed_variables_are_exactly_the_forcing_ones(output_dataset):
    prefixed = {str(name) for name in output_dataset.data_vars if str(name).startswith("forcing_")}
    tagged = set(map(str, output_dataset.filter_by_attrs(**{ROLE_ATTRIBUTE: "forcing"})))
    assert prefixed == tagged == {
        "forcing_ice_frazil_melt_energy", "forcing_ice_velocity_u", "forcing_ice_velocity_v",
    }


def test_roles_of_the_state_and_of_what_the_component_diagnosed(output_dataset):
    state = {
        "ice_thickness", "snow_thickness", "ice_fraction",
        "upper_ice_temperature", "lower_ice_temperature",
    }
    tagged_state = set(map(str, output_dataset.filter_by_attrs(**{ROLE_ATTRIBUTE: "state"})))
    assert tagged_state == state
    for name in ("ocean_heat_flux_up", "ocean_frazil_heating", "ice_energy_tendency", "ice_surface_temperature"):
        assert output_dataset[name].attrs[ROLE_ATTRIBUTE] == "derived", name


def test_output_follows_the_shared_coordinate_conventions(output_dataset):
    assert output_dataset.time.dtype == np.dtype("datetime64[ms]")
    assert output_dataset.ice_thickness.dims == ("time", "lon", "lat")
    np.testing.assert_array_equal(
        output_dataset.time.values,
        np.array(["2001-01-01T12:00", "2001-01-02T12:00", "2001-01-03T12:00"], dtype="datetime64[ms]"),
    )
    land = np.asarray(output_dataset.ice_thickness.isel(lon=[2, 3]))
    assert not land.any()
    np.testing.assert_allclose(
        output_dataset.ice_surface_temperature.isel(lon=[2, 3]).values, MASKED_SURFACE_TEMPERATURE, rtol=1e-6
    )


# ---------------------------------------------------------------------------
# Under a real Coupler
# ---------------------------------------------------------------------------

COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")


def build_coupler(model, exchangers=None, extra=None):
    components = {model.name: model, **(extra or {})}
    return Coupler(
        components, exchangers or {},
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE,
    )


@pytest.mark.parametrize("transport", [False, True])
def test_two_coupled_steps_thread_the_carry(transport):
    """Two steps through Coupler.generate_trajectory_function(2), with and without transport."""
    grid = global_grid(nx=12, ny=8, land_blocks=False)
    model = WintonSeaiceModel(grid, WintonSeaiceParameters(initial_ice_thickness=1.0), transport=transport)
    coupler = build_coupler(model)
    initial = coupler.initialize()
    final, diagnostics = coupler.generate_trajectory_function(2)(initial)

    assert jax.eval_shape(lambda: final) == jax.eval_shape(lambda: initial)
    assert int(final.step) == 2
    for leaf in jax.tree_util.tree_leaves(diagnostics["seaice"]):
        assert leaf.shape[0] == 2 and bool(jnp.all(jnp.isfinite(leaf)))
    assert float(final.components["seaice"]["state"].ice_thickness.max()) > 0.0


def test_a_parameter_gradient_through_a_coupled_trajectory():
    grid = global_grid(nx=12, ny=8, land_blocks=False)
    model = WintonSeaiceModel(grid, WintonSeaiceParameters(initial_ice_thickness=1.0))
    coupler = build_coupler(model)
    trajectory = coupler.generate_trajectory_function(2)
    warm = make_forcing(grid, rsds=300.0, rlds=300.0, air_temperature=272.0)

    def loss(lead_closing_thickness):
        carry = coupler.initialize()
        seaice = dict(
            carry.components["seaice"],
            forcing=warm,
            params=carry.components["seaice"]["params"].replace(
                lead_closing_thickness=lead_closing_thickness
            ),
        )
        carry = carry.replace(components={"seaice": seaice})
        final, _ = trajectory(carry)
        state = final.components["seaice"]["state"]
        return jnp.sum(state.ice_fraction * state.ice_thickness)

    gradient = jax.grad(loss)(jnp.float32(0.5))
    assert bool(jnp.isfinite(gradient))


def supply_atmosphere(components, time):
    """Stand in for an atmosphere: cold air over the ice, and a cooling flux for the ocean.

    The Winton ice reads its atmosphere-side inputs from an exchanger of the
    coupled model's own; no default row carries them (see
    ``docs/source/design/winton_seaice.md``), so a test that wants the ice to
    be forced writes them here. Nothing is mutated: new sections are built with
    ``.replace`` and new carries with ``dict``.
    """
    del time
    ocean, seaice = components["ocn"], components["seaice"]
    zeros = jnp.zeros_like(seaice["forcing"].rsds)
    forcing = seaice["forcing"].replace(
        rlds=zeros + 160.0, air_temperature=zeros + 240.0, air_specific_humidity=zeros + 3e-4,
    )
    return dict(
        components,
        seaice=dict(seaice, forcing=forcing),
        ocn=dict(ocean, forcing=ocean["forcing"].replace(total_heat_flux=zeros + 300.0)),
    )


def test_the_default_coupling_grows_ice_on_the_oceans_freeze_potential(caplog):
    """Ocean and Winton ice through default_exchanges plus a stand-in atmosphere.

    The ocean is cooled below freezing by the stand-in's flux, reports the
    freeze potential, and the default rows carry it (and the SST) into the ice.
    """
    grid = global_grid(nx=12, ny=8, land_blocks=False)
    # The ocean starts a tenth of a kelvin above freezing, so the stand-in's
    # cooling takes it below freezing within the first coupled steps.
    components = {
        "ocn": SlabOceanModel(grid, SlabOceanParameters(initial_sst=FREEZING_K + 0.1)),
        "seaice": WintonSeaiceModel(grid),
    }
    with caplog.at_level(logging.WARNING, logger="jem.exchangers"):
        specs = default_exchanges(components)
    sources = {(spec.src, spec.dst) for spec in specs}
    assert (
        "ocn.derived.ice_frazil_melt_energy", "seaice.forcing.ice_frazil_melt_energy"
    ) in sources
    assert (
        "ocn.state.sea_surface_temperature", "seaice.forcing.sea_surface_temperature"
    ) in sources
    # The fields the default coupling leaves to the caller are named, once.
    warning = " ".join(record.getMessage() for record in caplog.records)
    for field in ("rsds", "rlds", "air_temperature", "atm_sea_heat_flux", "ice_velocity_u"):
        assert field in warning
    assert "ice_frazil_melt_energy" not in warning
    assert "sea_surface_temperature" not in warning

    table = Exchange(specs)
    coupler = Coupler(
        components,
        {"exchange": table, "atmosphere": supply_atmosphere},
        workflow=["atmosphere", "exchange", "ocn", "seaice"],
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=START_DATE,
    )
    initial = coupler.initialize()
    table.validate(initial.components)
    final, diagnostics = coupler.generate_trajectory_function(3)(initial)

    thickness = final.components["seaice"]["state"].ice_thickness
    assert float(thickness.max()) > 0.0
    assert bool(jnp.all(jnp.isfinite(thickness)))
    frazil = np.asarray(diagnostics["seaice"]["forcing"].ice_frazil_melt_energy)
    np.testing.assert_array_equal(
        frazil[1:], np.asarray(diagnostics["ocn"]["derived"].ice_frazil_melt_energy)[:-1]
    )

    datasets = coupler.to_xarray(diagnostics)
    merged = xr.merge(list(datasets.values()), join="exact", compat="no_conflicts")
    assert merged.sizes["time"] == 3
    assert "ice_thickness" in merged and "sea_surface_temperature" in merged
    assert "forcing_ice_frazil_melt_energy" in merged


@pytest.mark.parametrize("overrides", [[], ["+seaice.transport=true"]])
def test_the_winton_ice_is_built_from_the_seaice_config_group(overrides):
    """`seaice=winton` instantiates the model on the runner's grid, wiring only."""
    from jem.runners import GROUP_TO_NAME, build_component
    from tests.unit.test_config import composed

    cfg = composed(["seaice=winton", *overrides])
    assert cfg.seaice._target_ == "jem.components.WintonSeaiceModel"
    grid = global_grid(nx=12, ny=8, land_blocks=False)
    model = build_component(cfg.seaice, grid=grid)
    assert isinstance(model, WintonSeaiceModel)
    assert model.name == GROUP_TO_NAME["seaice"]
    assert (model.transport is not None) == bool(overrides)
    # The YAML carries no parameter values: they are the class's defaults.
    defaults = WintonSeaiceParameters()
    assert jax.tree_util.tree_structure(model.params) == jax.tree_util.tree_structure(defaults)
    for got, want in zip(jax.tree_util.tree_leaves(model.params), jax.tree_util.tree_leaves(defaults)):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want))


@pytest.mark.slow
def test_winton_ice_under_jcm_and_a_slab_ocean_merges_with_the_atmosphere_dataset(caplog):
    """The real atmosphere, the slab ocean and the Winton ice, two coupled steps.

    The default coupling wires what it can (the ocean's freeze potential and
    SST into the ice, the ice fraction into the atmosphere) and the ice runs on
    its neutral initial forcing for the rest. The point is the seams: the carry
    survives ``lax.scan`` with a real atmosphere beside it, and all three
    datasets share one exact time axis and grid.
    """
    from jcm.model import Model
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.terrain import TerrainData

    from jem import default_exchangers
    from jem.components.jcm import JCMComponent

    start = jdt.to_datetime("2000-01-01")
    coords = get_speedy_coords(layers=5, spectral_truncation=21)
    atmosphere = Model(coords=coords, terrain=TerrainData.aquaplanet(coords), start_time=start)
    grid = SlabGrid.from_coords(atmosphere.coords.horizontal)
    components = {
        "atm": JCMComponent(atmosphere),
        "ocn": SlabOceanModel(grid),
        "seaice": WintonSeaiceModel(grid, WintonSeaiceParameters(initial_ice_thickness=1.0)),
    }
    with caplog.at_level(logging.WARNING, logger="jem.exchangers"):
        exchangers = default_exchangers(components)
    assert "atm_sea_heat_flux" in caplog.text
    coupler = Coupler(
        components, exchangers, coupling_timestep=COUPLING_TIMESTEP, start_date=start
    )
    initial = coupler.initialize()
    final, diagnostics = coupler.generate_trajectory_function(2)(initial)

    assert jax.eval_shape(lambda: final) == jax.eval_shape(lambda: initial)
    for leaf in jax.tree_util.tree_leaves(diagnostics["seaice"]):
        assert leaf.shape[0] == 2 and bool(jnp.all(jnp.isfinite(leaf)))
    # The atmosphere saw the ice: its sea-ice boundary condition is the ice's
    # fraction from the end of the previous step.
    sice = np.asarray(final.components["atm"]["forcing"].sice_am)
    assert sice.shape == grid.shape and sice.max() > 0.0

    datasets = coupler.to_xarray(diagnostics)
    merged = xr.merge(list(datasets.values()), join="exact", compat="no_conflicts")
    assert merged.sizes["time"] == 2
    assert "ice_thickness" in merged and "sea_surface_temperature" in merged
