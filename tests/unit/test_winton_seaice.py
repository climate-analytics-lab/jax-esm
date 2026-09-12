"""Checks for the Winton three-layer sea-ice thermodynamics.

Oracle for the temperature solve: a line-by-line numpy transcription of MITgcm
``thsice_solve4temp.F`` (enthalpy-based formulation), run with the same
prescribed surface flux and derivative.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jem.components.slab.winton_seaice_model.winton_seaice_model import (  # noqa: E402
    C_ICE, K_ICE, K_SNOW, L_ICE, MU, RHO_ICE, S_ICE, T_FREEZE, T_MELT,
    T1_from_q, T2_from_q, column_enthalpy_to_melt, q_from_T1, q_from_T2,
    winton_mass_step, winton_temperature_step,
)

DAY = 86400.0


def thsice_solve4temp_numpy(h, hs, T1, T2, Ts, flux_at, dflux, sw_abs, dt, i0=0.3, ksolar=1.5):
    """Transcription of thsice_solve4temp.F for one column, external fluxes, single pass.

    flux_at(Ts) gives the non-solar downward flux at Ts; dflux its derivative.
    thsice with external fluxes does not iterate (iterMax = 1).
    Returns (T1, T2, Ts, sHeat, flxCnB, sw_to_ocean).
    """
    cpIce, Lfresh, rhoi, kIce, kSnow, Tmlt1, tFrz = C_ICE, L_ICE, RHO_ICE, K_ICE, K_SNOW, T_MELT, T_FREEZE
    fswpen = 0.0 if hs > 0.0 else sw_abs * i0
    fswocn = fswpen * np.exp(-ksolar * h)
    fswint = fswpen - fswocn
    sHeat = sw_abs - fswpen
    k12 = 4.0 * kIce * kSnow / (kSnow * h + 4.0 * kIce * hs)
    k32 = 2.0 * kIce / h
    a10 = rhoi * cpIce * h / (2 * dt) + k32 * (4 * dt * k32 + rhoi * cpIce * h) / (6 * dt * k32 + rhoi * cpIce * h)
    b10 = -h * (rhoi * cpIce * T1 + rhoi * Lfresh * Tmlt1 / T1) / (2 * dt) \
          - k32 * (4 * dt * k32 * tFrz + rhoi * cpIce * h * T2) / (6 * dt * k32 + rhoi * cpIce * h) - fswint
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

    tIc2 = (2 * dt * k32 * (tIc1 + 2 * tFrz) + rhoi * cpIce * h * T2) / (6 * dt * k32 + rhoi * cpIce * h)
    flxFinal = flux_at(tSrf)
    fct = k12 * (tSrf - tIc1)
    sHeat_out = (sHeat + flxFinal - fct) if melting else 0.0
    flxCnB = 4 * kIce * (tIc2 - tFrz) / h
    return tIc1, tIc2, tSrf, sHeat_out, flxCnB, fswocn


def pinned_flux(T_target, gain=1e4):
    """Surface flux that pins the surface temperature: F = gain (T_target - Ts)."""
    return (lambda Ts: (gain * (T_target - Ts), -gain * jnp.ones_like(Ts)))


def test_temperature_step_matches_thsice_transcription():
    h, T1, T2, Ts, dt = 1.5, -8.0, -4.0, -15.0, 6 * 3600.0
    for hs in (0.0, 0.2):
        for name, F0, dF, sw in [("cold", -60.0, -20.0, 0.0), ("warm", 80.0, -20.0, 200.0), ("mild", 10.0, -25.0, 50.0)]:
            Ts_ref = Ts
            fj = lambda T: (F0 + dF * (T - Ts_ref), dF * jnp.ones_like(T))
            fn = lambda T: F0 + dF * (T - Ts_ref)
            got = winton_temperature_step(jnp.array(h), jnp.array(hs), jnp.array(T1), jnp.array(T2), jnp.array(Ts),
                                          fj, jnp.array(sw), dt, n_iter=1)
            ref = thsice_solve4temp_numpy(h, hs, T1, T2, Ts, fn, lambda T: dF, sw, dt)
            for g, r, what in zip(got[:5], ref[:5], ["T1", "T2", "Ts", "M_s", "F_cb"]):
                assert abs(float(g) - r) < 1e-9 * max(1.0, abs(r)), (hs, name, what, float(g), r)


def test_stefan_growth_law():
    """Cold pinned surface, no ocean flux: h^2 - h0^2 ~ 2 K dT t / (rho L)."""
    dt = 3 * 3600.0
    T_s = -30.0
    h, hs = jnp.array(1.0), jnp.array(0.0)
    T1, T2, Ts = jnp.array(-15.0), jnp.array(-8.0), jnp.array(T_s)
    days = 200
    for _ in range(int(days * DAY / dt)):
        T1, T2, Ts, M_s, F_cb, _ = winton_temperature_step(h, hs, T1, T2, Ts, pinned_flux(T_s), jnp.array(0.0), dt, n_iter=1)
        q1, q2 = q_from_T1(T1), q_from_T2(T2)
        h, hs, q1, q2, _, _ = winton_mass_step(h, hs, q1, q2, M_s, jnp.array(0.0), F_cb, jnp.array(0.0), dt)
        T1, T2 = T1_from_q(q1), T2_from_q(q2)
    # Stefan with effective latent heat of the lower ice (enthalpy at Tf)
    L_eff = float(q_from_T2(T_FREEZE))
    h_stefan = np.sqrt(1.0 + 2 * K_ICE * (T_FREEZE - T_s) * days * DAY / (RHO_ICE * L_eff))
    assert abs(float(h) - h_stefan) / h_stefan < 0.03, (float(h), h_stefan)
    assert float(Ts) < T_s + 0.05


def test_snow_insulates_growth():
    """Same pinned cold surface: an insulating snow layer slows Stefan-type growth."""
    dt = 3 * 3600.0
    T_s = -30.0
    days = 60

    def grow(hs0):
        h, hs = jnp.array(1.0), jnp.array(hs0)
        T1, T2, Ts = jnp.array(-15.0), jnp.array(-8.0), jnp.array(T_s)
        for _ in range(int(days * DAY / dt)):
            T1, T2, Ts, M_s, F_cb, _ = winton_temperature_step(h, hs, T1, T2, Ts, pinned_flux(T_s), jnp.array(0.0), dt, n_iter=1)
            q1, q2 = q_from_T1(T1), q_from_T2(T2)
            h, hs, q1, q2, _, _ = winton_mass_step(h, hs, q1, q2, M_s, jnp.array(0.0), F_cb, jnp.array(0.0), dt)
            T1, T2 = T1_from_q(q1), T2_from_q(q2)
        return float(h)

    h_no_snow = grow(0.0)
    h_snow = grow(0.3)
    assert h_snow < h_no_snow, (h_snow, h_no_snow)


def test_energy_conservation_over_melt_and_growth():
    """Change in column enthalpy equals time-integrated (atm flux + basal flux - energy to ocean)."""
    dt = 3 * 3600.0
    h, hs = jnp.array(3.0), jnp.array(0.0)
    T1, T2, Ts = jnp.array(-10.0), jnp.array(-5.0), jnp.array(-20.0)

    def column_energy(h, hs, q1, q2):
        return -float(column_enthalpy_to_melt(h, hs, q1, q2))

    q1, q2 = q_from_T1(T1), q_from_T2(T2)
    E0 = column_energy(h, hs, q1, q2)
    net_in = 0.0
    for step in range(600):
        warm = (step // 100) % 2 == 1
        fl = pinned_flux(1.0 if warm else -25.0, gain=50.0)
        sw = jnp.array(50.0 if warm else 0.0)
        F_b = jnp.array(10.0 if warm else 0.0)
        T1, T2, Ts, M_s, F_cb, sw_ocn = winton_temperature_step(h, hs, T1, T2, Ts, fl, sw, dt, n_iter=1)
        F_atm, _ = fl(Ts)
        q1, q2 = q_from_T1(T1), q_from_T2(T2)
        hn, hsn, q1, q2, e_ocn, vanished = winton_mass_step(h, hs, q1, q2, M_s, F_b, F_cb, jnp.array(0.0), dt)
        net_in += float((F_atm + sw - sw_ocn + F_b) * dt - e_ocn)
        h, hs = hn, hsn
        T1, T2 = T1_from_q(q1), T2_from_q(q2)
        assert float(h) > 0.05, "ice vanished; test forcing too warm"
    E1 = column_energy(h, hs, q1, q2)
    assert abs((E1 - E0) - net_in) < 2e-3 * abs(net_in) + 1e3, ((E1 - E0), net_in)


def test_complete_melt_returns_excess():
    dt = DAY / 4
    h, hs = jnp.array(0.3), jnp.array(0.0)
    T1, T2, Ts = jnp.array(-2.0), jnp.array(-1.9), jnp.array(-1.0)
    total_e_ocn = 0.0
    for _ in range(40):
        T1, T2, Ts, M_s, F_cb, _ = winton_temperature_step(h, hs, T1, T2, Ts, pinned_flux(10.0, gain=100.0),
                                                            jnp.array(300.0), dt)
        q1, q2 = q_from_T1(T1), q_from_T2(T2)
        h, hs, q1, q2, e_ocn, vanished = winton_mass_step(h, hs, q1, q2, M_s, jnp.array(50.0), F_cb, jnp.array(0.0), dt)
        T1, T2 = T1_from_q(q1), T2_from_q(q2)
        total_e_ocn += float(e_ocn)
        if bool(vanished) or float(h) < 1e-9:
            break
    assert np.isfinite(total_e_ocn)
    assert float(h) == 0.0
    assert total_e_ocn > 0.0


def test_enthalpy_inversion_roundtrip():
    for T in [-30.0, -10.0, -2.0, -0.2]:
        assert abs(float(T1_from_q(q_from_T1(jnp.array(T)))) - T) < 1e-9
        assert abs(float(T2_from_q(q_from_T2(jnp.array(T)))) - T) < 1e-9


def test_gradient_is_finite():
    def final_thickness(T_s):
        h, hs = jnp.array(1.0), jnp.array(0.0)
        T1, T2, Ts = jnp.array(-10.0), jnp.array(-5.0), T_s
        for _ in range(20):
            T1, T2, Ts, M_s, F_cb, _ = winton_temperature_step(h, hs, T1, T2, Ts, pinned_flux(T_s), jnp.array(20.0), DAY / 4)
            q1, q2 = q_from_T1(T1), q_from_T2(T2)
            h, hs, q1, q2, _, _ = winton_mass_step(h, hs, q1, q2, M_s, jnp.array(5.0), F_cb, jnp.array(0.0), DAY / 4)
            T1, T2 = T1_from_q(q1), T2_from_q(q2)
        return h
    g = jax.grad(final_thickness)(jnp.array(-20.0))
    assert jnp.isfinite(g) and float(g) < 0.0   # colder surface -> thicker ice


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)


def test_ice_transport_conserves_mass_and_energy():
    """Transport moves ice between cells but changes neither the global ice/snow mass nor the enthalpy."""
    import numpy as np
    from jem.components.slab.grid import SlabGrid
    from jem.components.slab.winton_seaice_model.winton_seaice_model import (
        WintonSeaiceModel, WintonState, WintonForcing, KELVIN, T_FREEZE, RHO_ICE, RHO_SNOW,
        column_enthalpy_to_melt, q_from_T1, q_from_T2)
    nx, ny = 24, 12
    rng = np.random.default_rng(0)
    land = np.zeros((nx, ny)); land[:, 0] = land[:, -1] = 1.0; land[5:8, 4:7] = 1.0
    lat = np.deg2rad(np.linspace(-80, 80, ny))[None, :].repeat(nx, 0); lon = np.deg2rad(np.linspace(0, 345, nx))[:, None].repeat(ny, 1)
    grid = SlabGrid(fractional_mask=jnp.asarray(land), latitude_radian=jnp.asarray(lat), longitude_radian=jnp.asarray(lon), threshold=0.5)
    dx = 6371e3 * np.cos(lat) * np.deg2rad(15.0); dy = np.full((nx, ny), 6371e3 * np.deg2rad(160 / (ny - 1)))
    model = WintonSeaiceModel(grid=grid, timestep=86400.0, transport=dict(dx=dx, dy=dy, diffusivity=5e4))
    ocean = np.asarray(grid.binary_mask == model.mask_value)
    area = dx * dy * ocean
    h = np.where(ocean, rng.uniform(0.5, 3.0, (nx, ny)) * (rng.uniform(size=(nx, ny)) > 0.4), 0.0)
    f = np.where(h > 0, rng.uniform(0.3, 1.0, (nx, ny)), 0.0)
    hs = np.where(h > 0, rng.uniform(0.0, 0.3, (nx, ny)), 0.0)
    T1 = np.where(h > 0, rng.uniform(-20, -3, (nx, ny)), T_FREEZE); T2 = np.where(h > 0, rng.uniform(-10, -2, (nx, ny)), T_FREEZE)
    Ts = np.where(h > 0, rng.uniform(-30, -1, (nx, ny)), T_FREEZE)
    s = WintonState(jnp.zeros(()), *(jnp.asarray(a) for a in (h, hs, f, T1, T2, Ts)))
    z = jnp.zeros((nx, ny))
    u = jnp.asarray(rng.uniform(-0.3, 0.3, (nx, ny))); v = jnp.asarray(rng.uniform(-0.3, 0.3, (nx, ny)))
    fc = WintonForcing(z, z, z + 250.0, z + 1e-3, z + 5.0, z + 1.0, z, z + KELVIN + T_FREEZE, z, z, u, v)
    step = model._create_step_function_body()
    # isolate transport: zero forcing keeps thermodynamics quiet only approximately, so compare against a
    # thermodynamics-only model with the same state and forcing
    model0 = WintonSeaiceModel(grid=grid, timestep=86400.0)
    out_t = step({"state": s, "forcing": fc}, 0)[0]["state"]
    out_0 = model0._create_step_function_body()({"state": s, "forcing": fc}, 0)[0]["state"]
    mass = lambda st: float((np.asarray(st.ice_fraction) * (RHO_ICE * np.asarray(st.ice_thickness) + RHO_SNOW * np.asarray(st.snow_thickness)) * area).sum())
    energy = lambda st: float((np.asarray(st.ice_fraction) * np.asarray(column_enthalpy_to_melt(
        st.ice_thickness, st.snow_thickness, q_from_T1(st.upper_ice_temperature), q_from_T2(st.lower_ice_temperature))) * area).sum())
    moved = float(np.abs(np.asarray(out_t.ice_fraction * out_t.ice_thickness) - np.asarray(out_0.ice_fraction * out_0.ice_thickness)).sum())
    assert moved > 0.0, "transport did nothing"
    assert abs(mass(out_t) - mass(out_0)) < 1e-6 * mass(out_0)
    assert abs(energy(out_t) - energy(out_0)) < 1e-6 * abs(energy(out_0))
    assert float(out_t.ice_fraction.max()) <= 1.0 and float(out_t.ice_thickness.min()) >= 0.0
    assert not np.any(np.asarray(out_t.ice_thickness)[~ocean] > 0)
    icy = np.asarray(out_t.ice_thickness) > 0.05
    assert np.all(np.asarray(out_t.ice_surface_temperature)[icy] < -0.5), "surface temperature lost its sign in transport"
    assert np.all(np.asarray(out_t.upper_ice_temperature)[icy] < -0.5)


def test_ice_transport_limiter_positivity_and_conservation():
    """With Courant numbers above one and a huge diffusivity the outflow limiter must keep every field
    non-negative and conserve it exactly (no clipping): the property the scheme advertises."""
    import numpy as np
    from jem.components.slab.winton_seaice_model.ice_transport import transport_fields
    rng = np.random.default_rng(3)
    nx, ny = 20, 10
    ocean = np.ones((nx, ny), bool); ocean[:, 0] = ocean[:, -1] = False; ocean[6:9, 3:6] = False
    dx = np.full((nx, ny), 2.0e5); dy = np.full((nx, ny), 2.0e5); area = dx * dy
    X1 = np.where(ocean, rng.uniform(0, 3, (nx, ny)) * (rng.uniform(size=(nx, ny)) > 0.5), 0.0)
    X2 = np.where(ocean, rng.uniform(0, 1, (nx, ny)), 0.0)
    u = rng.uniform(-3, 3, (nx, ny)); v = rng.uniform(-3, 3, (nx, ny))           # Courant ~1.3 per day
    out = transport_fields((jnp.asarray(X1), jnp.asarray(X2)), jnp.asarray(u), jnp.asarray(v), jnp.asarray(dx),
                           jnp.asarray(dy), jnp.asarray(ocean), 86400.0, diffusivity=1e6, n_substeps=1)
    for X, Y in zip((X1, X2), out):
        Y = np.asarray(Y)
        assert Y.min() >= 0.0
        assert abs((Y * area).sum() - (X * area).sum()) <= 1e-12 * (X * area).sum()
        assert not np.any(Y[~ocean] != 0.0)
        assert np.abs(Y - X).sum() > 0.0     # transport did happen


def test_winton_two_step_trajectory_through_coupler():
    """Integration: an initialized carry runs through Coupler.generate_trajectory_function(2), with transport."""
    import numpy as np
    from jem.base.coupler import Coupler
    from jem.components.slab.grid import SlabGrid
    from jem.components.slab.winton_seaice_model import WintonSeaiceModel
    nx, ny = 12, 8
    land = np.zeros((nx, ny)); land[:, 0] = land[:, -1] = 1.0
    lat = np.deg2rad(np.linspace(-80, 80, ny))[None, :].repeat(nx, 0); lon = np.deg2rad(np.linspace(0, 330, nx))[:, None].repeat(ny, 1)
    grid = SlabGrid(fractional_mask=jnp.asarray(land), latitude_radian=jnp.asarray(lat), longitude_radian=jnp.asarray(lon), threshold=0.5)
    dx = 6371e3 * np.cos(lat) * np.deg2rad(30.0); dy = np.full((nx, ny), 6371e3 * np.deg2rad(160 / (ny - 1)))
    model = WintonSeaiceModel(grid=grid, timestep=86400.0, initial_ice_thickness=1.0, transport=dict(dx=dx, dy=dy, diffusivity=2e4))
    coupler = Coupler(components=dict(ice=model))
    init = coupler.initialize()
    traj = coupler.generate_trajectory_function(workflow=["ice"], iterations=2, jitted=True, show_progress=False)
    final, preds = traj(init)
    assert set(final["ice"]) == {"state", "forcing", "derived"}
    for leaf in jax.tree_util.tree_leaves(preds["ice"]):
        assert leaf.shape[0] == 2 and bool(jnp.all(jnp.isfinite(leaf)))
    assert float(final["ice"]["state"].ice_thickness.max()) > 0.0
