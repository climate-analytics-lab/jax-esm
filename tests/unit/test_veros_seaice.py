"""A sea ice coupled to the Veros ocean (jax-esm#160).

Everything here builds a real (small) Veros model, the ``acc_basic`` example of
``test_veros_component.py``, with a Winton sea ice on its grid and a stub
atmosphere that carries JCM's exchange field names. Importing Veros flips
``jax_enable_x64`` for the whole process, so **run this file in a process of
its own** (``test_veros_earth_masks_veros.py`` gives the reasoning; the "slow
unit tests" step of ``.github/workflows/tests.yml`` runs the Veros files one
invocation each). It is skipped when Veros is not installed.

The cold case drives the budget tests. The ocean's whole column starts just
above the freezing point and the stub atmosphere takes heat out of it, so the
clamp acts and the ice forms; the flux is then reversed so the ice melts.
Because the exchange runs first and every component steps on what the previous
step left (``docs/source/design/exchange.md``), the ice's and the ocean's
budgets are lagged by one step against each other, and the closure of the pair
is exact only once the energy in transit -- the heat the ice has released but
the ocean has not yet applied, and the frazil the ocean has clamped but the ice
has not yet been handed -- is counted (:func:`_budget`).
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

pytest.importorskip("veros")

from jem.base.component import CouplingTime  # noqa: E402
from jem.base.coupler import Coupler  # noqa: E402
from jem.components.slab import SlabGrid, SlabSeaiceModel  # noqa: E402
from jem.components.slab.winton_seaice_model.winton_seaice_model import (  # noqa: E402
    RHO_ICE,
    RHO_SNOW,
    WintonSeaiceModel,
    WintonSeaiceParameters,
    column_enthalpy_to_melt,
    q_from_T1,
    q_from_T2,
)
from jem.components.veros_component import (  # noqa: E402
    SEAWATER_HEAT_CAPACITY,
    VerosComponent,
)
from jem.exchangers import default_exchanges  # noqa: E402
from jem.fluxes import VerosExchange  # noqa: E402
from tests.unit.test_veros_component import (  # noqa: E402
    COUPLING_TIMESTEP,
    START_DATE,
    _acc_basic_model,
)

DAY = 86400.0
#: Celsius temperature the cold column starts at: 0.1 K above the freezing
#: point the coupled model uses (-1.8 degC), so a modest cooling puts it below.
COLD_START = -1.7
#: Upward heat flux the stub atmosphere takes out of the sea part of a cell
#: while the ice grows (W m-2), reversed for the melt.
COOLING = 1000.0
COOLING_STEPS = 4
MELTING_STEPS = 4
N_STEPS = COOLING_STEPS + MELTING_STEPS

# ---------------------------------------------------------------------------
# The stub atmosphere
# ---------------------------------------------------------------------------


@struct.dataclass
class _AtmDerived:
    sea_eastward_wind_stress: jnp.ndarray
    sea_northward_wind_stress: jnp.ndarray
    sea_total_heat_flux: jnp.ndarray
    sea_total_freshwater_flux: jnp.ndarray
    sea_fraction: jnp.ndarray
    downward_shortwave_flux: jnp.ndarray
    downward_longwave_flux: jnp.ndarray
    sea_air_temperature: jnp.ndarray
    sea_air_specific_humidity: jnp.ndarray
    wind_speed: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray
    sea_evaporation: jnp.ndarray
    rainfall: jnp.ndarray
    snowfall: jnp.ndarray


@struct.dataclass
class _AtmForcing:
    sea_surface_temperature: jnp.ndarray
    sice_am: jnp.ndarray
    sea_ice_temperature: jnp.ndarray


class StubAtmosphere:
    """An atmosphere with JCM's exchange field names and a scripted weather.

    The sea heat flux is ``COOLING`` upward for ``COOLING_STEPS`` steps and
    ``-COOLING`` after; the air is cold with snow while it cools and warm and
    sunny while it warms. It is a script, not a model: what is under test is
    the exchange between the ocean and the ice, which only needs the fields
    to be there and the budget identities to hold whatever they are.
    """

    name = "atm"

    def __init__(self, shape, wet, phase=0):
        """Remember the grid shape and the ocean's wet mask (the sea part).

        ``phase`` shifts the script: ``COOLING_STEPS`` starts it in the warm,
        sunny half.
        """
        self.shape = shape
        self.wet = wet
        self.phase = phase

    def _derived(self, step) -> _AtmDerived:
        shape = self.shape
        cooling = (step + self.phase) < COOLING_STEPS
        full = lambda value: jnp.full(shape, value)  # noqa: E731
        flux = jnp.where(cooling, COOLING, -COOLING)
        snow = jnp.where(cooling, 1e-5, 0.0)
        evaporation = 1e-5
        rain = jnp.where(cooling, 0.0, 2e-5)
        return _AtmDerived(
            sea_eastward_wind_stress=full(0.05),
            sea_northward_wind_stress=full(0.0),
            sea_total_heat_flux=flux * jnp.ones(shape),
            sea_total_freshwater_flux=(evaporation - rain - snow) * jnp.ones(shape),
            sea_fraction=self.wet,
            downward_shortwave_flux=jnp.where(cooling, 0.0, 200.0) * jnp.ones(shape),
            downward_longwave_flux=jnp.where(cooling, 180.0, 300.0) * jnp.ones(shape),
            sea_air_temperature=jnp.where(cooling, 255.0, 276.0) * jnp.ones(shape),
            sea_air_specific_humidity=full(1e-3),
            wind_speed=full(6.0),
            normalized_surface_pressure=full(1.0),
            sea_evaporation=full(evaporation),
            rainfall=rain * jnp.ones(shape),
            snowfall=snow * jnp.ones(shape),
        )

    def initialize(self):
        shape = self.shape
        return {
            "derived": self._derived(0),
            "forcing": _AtmForcing(
                sea_surface_temperature=jnp.full(shape, 271.0),
                sice_am=jnp.zeros(shape),
                sea_ice_temperature=jnp.full(shape, 271.0),
            ),
        }

    def step(self, carry, time):
        derived = self._derived(time.step)
        return (
            {"derived": derived, "forcing": carry["forcing"]},
            {"sea_total_heat_flux": derived.sea_total_heat_flux,
             "sice_am": carry["forcing"].sice_am,
             "sea_ice_temperature": carry["forcing"].sea_ice_temperature,
             "sea_surface_temperature": carry["forcing"].sea_surface_temperature},
        )


# ---------------------------------------------------------------------------
# Building the coupled model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def veros_model(tmp_path_factory):
    yield from _acc_basic_model(tmp_path_factory.mktemp("veros_seaice"))


def _grid_for(component: VerosComponent) -> SlabGrid:
    """Return a SlabGrid on the ocean's cells whose land is the ocean's dry columns.

    ``acc_basic`` is a Cartesian box with no longitudes or latitudes; the ice
    needs only a grid of the right shape and a land fraction that is Veros'
    binary mask, so the coordinates are stand-ins.
    """
    wet = np.asarray(component.mask_T[:, :, -1])
    nx, ny = wet.shape
    longitude = np.deg2rad(np.linspace(0.0, 360.0, nx, endpoint=False))
    latitude = np.deg2rad(np.linspace(-60.0, 60.0, ny))
    return SlabGrid(
        fractional_mask=jnp.asarray(1.0 - wet),
        latitude_radian=jnp.asarray(np.broadcast_to(latitude[None, :], (nx, ny))),
        longitude_radian=jnp.asarray(np.broadcast_to(longitude[:, None], (nx, ny))),
        longitude_axis_radian=longitude,
        latitude_axis_radian=latitude,
    )


def _coupler(veros_model, *, clamp=True, ice="winton", exchanger=None, phase=0):
    """Return a coupler on the module's ``acc_basic`` model with ``ice`` on its grid."""
    ocean = VerosComponent(veros_model, clamp_at_freezing=clamp)
    atmosphere = StubAtmosphere(
        ocean.horizontal_shape, np.asarray(ocean.mask_T[:, :, -1]), phase=phase)
    components = {"atm": atmosphere, "ocn": ocean}
    if ice == "winton":
        components["seaice"] = WintonSeaiceModel(_grid_for(ocean))
    elif ice == "slab":
        components["seaice"] = SlabSeaiceModel(_grid_for(ocean), name="seaice")
    coupler = Coupler(
        components,
        {"exchange": exchanger if exchanger is not None else VerosExchange()},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    return coupler, components


def _cold_column(carry, celsius=COLD_START):
    """Return ``carry`` with the ocean's whole column at ``celsius`` on every time level.

    On a flatten-and-rebuild copy of the state, so the module's shared model
    is not modified.
    """
    from veros.core.operators import at, update

    state = jax.tree_util.tree_map(lambda leaf: leaf, carry["state"])
    variables = state.variables
    with variables.unlock():
        variables.temp = update(
            variables.temp, at[...], celsius * variables.maskT[..., None])
    derived = carry["derived"].replace(
        sea_surface_temperature=jnp.full_like(
            carry["derived"].sea_surface_temperature, celsius + 273.15))
    return dict(carry, state=state, derived=derived)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def _cell_geometry(component):
    """Return ``(area, mask, dz)`` of the interior: m2, 3-D wet mask, m."""
    interior = slice(2, -2)
    variables = component.model.state.variables
    area = np.asarray(variables.area_t[interior, interior])
    mask = np.asarray(variables.maskT[interior, interior, :])
    return area, mask, np.asarray(variables.dzt)


def _ocean_heat(component, temp):
    """Heat content of the wet ocean, J, of Celsius ``temp`` (interior, 3-D)."""
    area, mask, dz = _cell_geometry(component)
    rho_0 = component.model.state.settings.rho_0
    return float(np.sum(
        rho_0 * SEAWATER_HEAT_CAPACITY * np.asarray(temp) * mask
        * area[..., None] * dz))


def _ocean_salt(component, salt):
    """Salt content of the wet ocean, in g/kg times m3."""
    area, mask, dz = _cell_geometry(component)
    return float(np.sum(np.asarray(salt) * mask * area[..., None] * dz))


def _ice_energy(state):
    """Ice and snow enthalpy per unit sea area, J m-2 (negative: ice is below liquid water)."""
    return -state.ice_fraction * column_enthalpy_to_melt(
        state.ice_thickness, state.snow_thickness,
        q_from_T1(state.upper_ice_temperature),
        q_from_T2(state.lower_ice_temperature))


def _ice_mass(state):
    """Ice and snow mass per unit sea area, kg m-2."""
    return state.ice_fraction * (
        RHO_ICE * state.ice_thickness + RHO_SNOW * state.snow_thickness)


class _ColdRun:
    """One cooling-then-melting run of the coupled model, with everything it produced."""

    def __init__(self, veros_model):
        coupler, components = _coupler(veros_model)
        self.coupler = coupler
        self.ocean = components["ocn"]
        self.ice = components["seaice"]
        initial = coupler.initialize()
        parts = dict(initial.components)
        parts["ocn"] = _cold_column(parts["ocn"])
        self.initial = initial.replace(components=parts)
        self.final, self.diagnostics = coupler.generate_trajectory_function(
            N_STEPS)(self.initial)
        self.area, self.mask, self.dz = _cell_geometry(self.ocean)
        self.surface_area = self.area * self.mask[..., -1]
        # The ocean's state is a Veros state whose time level `tau` is the
        # current one; the diagnostics already hold each step's `temp` at it.
        variables = self.initial.components["ocn"]["state"].variables
        interior = slice(2, -2)
        self.initial_temp = np.asarray(
            variables.temp[interior, interior, :, variables.tau])
        self.initial_salt = np.asarray(
            variables.salt[interior, interior, :, variables.tau])


@pytest.fixture(scope="module")
def cold_run(veros_model):
    """Run the cold case through ``Coupler.generate_trajectory_function``, once."""
    return _ColdRun(veros_model)


# ---------------------------------------------------------------------------
# The coupled step under scan
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_coupled_carry_is_stable_under_scan_and_the_atmosphere_is_written(cold_run):
    """Two-plus coupled steps keep the carry's structure; the exchange writes the atmosphere.

    ``lax.scan`` rejects a step whose carry differs from what it received, so
    the run having finished at all is the structural check; what remains is
    that the atmosphere's forcing, which the stub only ever copies forward, was
    written by the exchanger: the ice's fraction and skin temperature, and the
    ocean's SST.
    """
    initial = cold_run.initial
    assert jax.tree_util.tree_structure(cold_run.final) == jax.tree_util.tree_structure(initial)
    atm = cold_run.diagnostics["atm"]
    # The stub starts at 271 K SST and 271 K ice temperature and zero ice; the
    # exchanger replaced all three.
    assert np.all(np.isfinite(np.asarray(atm["sice_am"])))
    wet = np.asarray(cold_run.ocean.mask_T[:, :, -1]) > 0
    final_sst = np.asarray(atm["sea_surface_temperature"])[-1]
    ocean_sst = np.asarray(cold_run.diagnostics["ocn"]["sea_surface_temperature"])[-2]
    np.testing.assert_allclose(final_sst[wet], ocean_sst[wet], rtol=1e-12)
    assert np.max(np.asarray(atm["sice_am"])) > 0.0, "ice formed, and the atmosphere was told"
    ice = np.asarray(cold_run.diagnostics["seaice"]["derived"].ice_fraction)
    np.testing.assert_allclose(np.asarray(atm["sice_am"])[1:], ice[:-1], atol=1e-12)
    ice_temperature = np.asarray(
        cold_run.diagnostics["seaice"]["derived"].ice_surface_temperature_K)
    covered = ice[:-1] > 0.0
    np.testing.assert_allclose(
        np.asarray(atm["sea_ice_temperature"])[1:][covered], ice_temperature[:-1][covered],
        rtol=1e-12)


@pytest.mark.slow
def test_the_clamp_holds_the_ocean_at_freezing_and_the_ice_forms(cold_run):
    """The surface never goes below freezing, and frazil ice appears from the deficit."""
    ocean = cold_run.diagnostics["ocn"]
    wet = np.asarray(cold_run.ocean.mask_T[:, :, -1]) > 0
    sst = np.asarray(ocean["sea_surface_temperature"])[:, wet]
    assert sst.min() >= 271.35 - 1e-9
    frazil = np.asarray(ocean["ice_frazil_melt_energy"])
    assert frazil.max() > 0.0, "the cold case must reach the clamp"
    assert np.all(frazil[:, ~wet] == 0.0), "land has no freeze/melt potential"
    ice = cold_run.diagnostics["seaice"]
    assert np.asarray(ice["state"].ice_thickness).max() > 0.0
    # The melting phase thins what the cooling phase grew.
    volume = np.sum(np.asarray(ice["derived"].ice_volume), axis=(1, 2))
    assert volume.max() > 0.0
    assert volume[-1] < volume.max()


@pytest.mark.slow
def test_the_heat_the_clamp_adds_is_the_frazil_the_ice_receives(cold_run, veros_model):
    """Every step, the clamp's heat (from the temperature change) is what the ice is handed.

    The clamp's heat is measured, not recomputed: the same ocean stepped from
    the same carry with and without the clamp differs only in the surface
    cells it raised, and that difference times the top cell's heat capacity is
    what it added. The ice receives the positive part of the published
    freeze/melt energy one exchange later, in its forcing.
    """
    ocean = cold_run.ocean
    rho_0 = veros_model.state.settings.rho_0
    heat_capacity = rho_0 * SEAWATER_HEAT_CAPACITY * float(ocean.dzt[-1])
    # One step from the cold start, once clamped and once not.
    carry = cold_run.initial.components["ocn"]
    cooling = carry["forcing"].replace(
        heat_flux=jnp.full(ocean.horizontal_shape, 4.0 * COOLING))
    carry = dict(carry, forcing=cooling)
    time = CouplingTime(step=jnp.int32(0), time=START_DATE, dt=DAY)
    plain = VerosComponent(veros_model, clamp_at_freezing=False)
    plain.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    clamped = VerosComponent(veros_model, clamp_at_freezing=True)
    clamped.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    free, _ = jax.jit(plain.step)(carry, time)
    held, _ = jax.jit(clamped.step)(carry, time)

    interior = slice(2, -2)
    wet = np.asarray(ocean.mask_T[:, :, -1]) > 0
    free_v, held_v = free["state"].variables, held["state"].variables
    added = (np.asarray(held_v.temp[interior, interior, -1, held_v.tau])
             - np.asarray(free_v.temp[interior, interior, -1, free_v.tau])) * heat_capacity
    published = np.asarray(held["derived"].ice_frazil_melt_energy)
    assert published[wet].max() > 0.0
    np.testing.assert_allclose(
        added[wet], np.maximum(published[wet], 0.0), rtol=1e-10, atol=1e-3)
    np.testing.assert_array_equal(
        np.asarray(free["derived"].ice_frazil_melt_energy), published)

    # ... and what the ice is told in the coupled run is that same published
    # energy, a step later.
    sent = np.asarray(cold_run.diagnostics["ocn"]["ice_frazil_melt_energy"])
    received = np.asarray(cold_run.diagnostics["seaice"]["forcing"].ice_frazil_melt_energy)
    np.testing.assert_array_equal(received[1:], sent[:-1])
    np.testing.assert_array_equal(received[0], 0.0)


def _budget(run):
    """Return the ocean+ice energy change and the heat the atmosphere took, J.

    The atmosphere's heat is the flux each exchange handed the ice, which is
    the previous atmosphere step's (the first exchange sees the stub's initial
    value, the same as its step 0), integrated over the sea area. The energy
    still in transit at the end -- the ice's last ``ocean_heat_flux_up``,
    which the ocean applies a step later, and the last clamp's frazil, which
    the ice is handed a step later -- is returned separately: with it the
    budget closes, without it the totals differ by exactly it.
    """
    dt = DAY
    surface_area = run.surface_area
    ocean = run.diagnostics["ocn"]
    ice_diag = run.diagnostics["seaice"]
    sea = np.asarray(run.ocean.mask_T[:, :, -1]).astype(float)

    d_ocean = (_ocean_heat(run.ocean, ocean["temp"][-1])
               - _ocean_heat(run.ocean, run.initial_temp))

    ice_initial = run.initial.components["seaice"]["state"]
    ice_final = run.final.components["seaice"]["state"]
    d_ice = float(np.sum(
        (np.asarray(_ice_energy(ice_final)) - np.asarray(_ice_energy(ice_initial)))
        * surface_area))

    atm_flux = np.asarray(run.diagnostics["atm"]["sea_total_heat_flux"])
    initial_flux = np.asarray(
        run.initial.components["atm"]["derived"].sea_total_heat_flux)
    flux_seen_by_ice = np.concatenate([initial_flux[None], atm_flux[:-1]])
    taken = dt * float(np.sum(flux_seen_by_ice * (sea * surface_area)))

    in_transit = (
        dt * float(np.sum(np.asarray(ice_diag["derived"].ocean_heat_flux_up)[-1] * surface_area))
        + float(np.sum(np.maximum(np.asarray(ocean["ice_frazil_melt_energy"])[-1], 0.0)
                       * surface_area))
    )
    return d_ocean, d_ice, -taken, in_transit


@pytest.mark.slow
def test_energy_is_conserved_across_the_ocean_the_ice_and_the_atmosphere(cold_run):
    """The ocean's heat plus the ice's enthalpy change by what the atmosphere took.

    Over four cooling steps (the clamp acts and the ice forms) and four melting
    steps (it thins), in float64. The totals are of order 1e22 J; the
    residual, after the energy in transit is counted, is Veros' own
    conservation of heat plus round-off -- 4e-13 of the heat the atmosphere
    exchanged, as measured -- and is asserted below 1e-9 of it.
    """
    d_ocean, d_ice, given, in_transit = _budget(cold_run)
    scale = abs(given)
    assert scale > 1e12
    residual = (d_ocean + d_ice) - (given + in_transit)
    assert abs(residual) < 1e-9 * scale, (
        f"residual {residual:.3e} J of {scale:.3e} J exchanged")
    # The ice took a real share of it, or the test would pass without one.
    assert abs(d_ice) > 1e-3 * scale or np.max(
        np.asarray(cold_run.diagnostics["seaice"]["state"].ice_thickness)) > 0.0


@pytest.mark.slow
def test_the_oceans_own_heat_budget_closes_with_the_clamp_counted(cold_run):
    """The ocean alone: its heat changed by what it was given plus what the clamp added.

    The separation of the two halves of the previous test: this is Veros'
    conservation (and the clamp's arithmetic), without the ice in it.
    """
    dt = DAY
    ocean = cold_run.diagnostics["ocn"]
    applied = np.asarray(ocean["heat_flux"])
    clamp = np.maximum(np.asarray(ocean["ice_frazil_melt_energy"]), 0.0)
    surface_area = cold_run.surface_area
    expected = (-dt * float(np.sum(applied * surface_area))
                + float(np.sum(clamp * surface_area)))
    actual = (_ocean_heat(cold_run.ocean, ocean["temp"][-1])
              - _ocean_heat(cold_run.ocean, cold_run.initial_temp))
    assert abs(actual - expected) < 1e-9 * abs(expected)


@pytest.mark.slow
def test_the_ocean_gets_the_ice_s_fluxes_in_place_of_the_atmosphere_s(cold_run):
    """The ocean's heat and freshwater forcing are the ice's, one exchange later."""
    ocean = cold_run.diagnostics["ocn"]
    ice = cold_run.diagnostics["seaice"]["derived"]
    wet = np.asarray(cold_run.ocean.mask_T[:, :, -1]) > 0
    np.testing.assert_array_equal(
        np.asarray(ocean["heat_flux"])[1:][:, wet],
        np.asarray(ice.ocean_heat_flux_up)[:-1][:, wet])
    np.testing.assert_array_equal(
        np.asarray(ocean["freshwater_flux"])[1:][:, wet],
        np.asarray(ice.ocean_freshwater_flux_up)[:-1][:, wet])
    # Not the atmosphere's flux once there is ice: the swamp mask is off and
    # the flux that reaches the ocean is the ice's.
    atm_flux = np.asarray(cold_run.diagnostics["atm"]["sea_total_heat_flux"])[:-1]
    assert not np.allclose(np.asarray(ocean["heat_flux"])[1:], atm_flux)


@pytest.mark.slow
def test_the_ocean_s_salt_changes_by_the_freshwater_the_ice_hands_it(cold_run, veros_model):
    """Salt content follows the ice's freshwater flux; the ice's water budget closes.

    ``forc_salt_surface = fw / rho_0 * 35`` (psu m s-1) adds that much salt
    per unit area per second to the top cell, so the salt content (g/kg times
    m3) changes by ``dt * 35 / rho_0 * sum(fw * area)``. The ice's own water
    budget is checked against it: the water an ice-covered ocean loses is the
    atmosphere's ``E - P`` plus the change of the ice and snow mass.
    """
    dt = DAY
    rho_0 = veros_model.state.settings.rho_0
    ocean = cold_run.diagnostics["ocn"]
    applied = np.asarray(ocean["freshwater_flux"])
    surface_area = cold_run.surface_area
    expected = dt * 35.0 / rho_0 * float(np.sum(applied * surface_area))
    actual = (_ocean_salt(cold_run.ocean, ocean["salt"][-1])
              - _ocean_salt(cold_run.ocean, cold_run.initial_salt))
    assert abs(expected) > 0.0
    assert abs(actual - expected) < 1e-9 * abs(expected)

    # The ice's water: its mass change is what its flux says beyond E - P.
    ice_diag = cold_run.diagnostics["seaice"]
    forcing = ice_diag["forcing"]
    e_minus_p = (np.asarray(forcing.evaporation) - np.asarray(forcing.liquid_precipitation)
                 - np.asarray(forcing.frozen_precipitation))
    fw_up = np.asarray(ice_diag["derived"].ocean_freshwater_flux_up)
    initial = _ice_mass(cold_run.initial.components["seaice"]["state"])
    masses = np.concatenate([
        np.asarray(initial)[None],
        np.stack([np.asarray(m) for m in _per_step_mass(ice_diag)])])
    wet = np.asarray(cold_run.ocean.mask_T[:, :, -1]) > 0
    np.testing.assert_allclose(
        (masses[1:] - masses[:-1])[:, wet], (dt * (fw_up - e_minus_p))[:, wet],
        rtol=1e-8, atol=1e-8)


def _per_step_mass(ice_diag):
    """Return the ice mass after each step, from the stacked state diagnostics."""
    state = ice_diag["state"]
    return [_ice_mass(jax.tree_util.tree_map(lambda leaf, k=k: leaf[k], state))
            for k in range(N_STEPS)]


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_winton_parameter_has_a_finite_gradient_through_the_coupled_run(veros_model):
    """``jax.grad`` of the final surface temperature with respect to the ice's albedo.

    Through two coupled steps: the albedo sets how much of the stub's sunshine
    the ice absorbs on step 1, which changes the heat the ice hands the ocean,
    which is the ocean's forcing on step 2. The cold, ice-covered start makes
    the sensitivity real; a gradient that is finite but zero would mean the
    ice's parameter never reached the ocean.
    """
    coupler, _ = _coupler(veros_model, phase=COOLING_STEPS)
    base = coupler.initialize(params={
        "seaice": WintonSeaiceParameters(initial_ice_thickness=1.0)})
    parts = dict(base.components)
    parts["ocn"] = _cold_column(parts["ocn"])
    base = base.replace(components=parts)
    trajectory = coupler.generate_trajectory_function(2)

    def final_surface_temperature(albedo):
        seaice = dict(base.components["seaice"])
        seaice["params"] = seaice["params"].replace(ice_albedo=albedo)
        start = base.replace(components=dict(base.components, seaice=seaice))
        final, _ = trajectory(start)
        return jnp.mean(final.components["ocn"]["derived"].sea_surface_temperature)

    value, gradient = jax.value_and_grad(final_surface_temperature)(
        jnp.asarray(WintonSeaiceParameters().ice_albedo))
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert float(gradient) != 0.0


# ---------------------------------------------------------------------------
# What a clamp that is off does not change
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_without_the_clamp_a_step_is_the_unclamped_integration(veros_model):
    """``clamp_at_freezing=False`` leaves the surface to cool below freezing; nothing else differs.

    The clamped and unclamped components are stepped from one carry. Every
    Veros variable is bit-identical between them except the temperature at the
    surface cells the clamp raised, which proves the clamp touches nothing
    else (the other levels, the salinity, the velocities, the older time
    levels), and that with it off the step is Veros' own.
    """
    plain = VerosComponent(veros_model)
    plain.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    clamped = VerosComponent(veros_model, clamp_at_freezing=True)
    clamped.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    carry = _cold_column(plain.initialize())
    carry = dict(carry, forcing=carry["forcing"].replace(
        heat_flux=jnp.full(plain.horizontal_shape, 4.0 * COOLING)))
    time = CouplingTime(step=jnp.int32(0), time=START_DATE, dt=DAY)
    free, _ = jax.jit(plain.step)(carry, time)
    held, _ = jax.jit(clamped.step)(carry, time)

    free_vars = {k: np.asarray(v) for k, v in free["state"].variables.items()}
    held_vars = {k: np.asarray(v) for k, v in held["state"].variables.items()}
    different = sorted(
        k for k in free_vars
        if not np.array_equal(free_vars[k], held_vars[k], equal_nan=True))
    assert different == ["temp"], different
    tau = int(free["state"].variables.tau)
    differing_cells = free_vars["temp"] != held_vars["temp"]
    assert not np.any(differing_cells[:, :, :-1, :]), "no level below the surface changed"
    assert not np.any(np.delete(differing_cells[:, :, -1, :], tau, axis=-1)), (
        "no time level but the current one changed")
    assert np.any(differing_cells[:, :, -1, tau]), "the cold case reached the clamp"
    wet = np.pad(np.asarray(plain.mask_T[:, :, -1]) > 0, 2)
    assert free_vars["temp"][..., -1, tau][wet].min() < -1.8, (
        "unclamped, the surface is allowed below freezing")
    assert held_vars["temp"][..., -1, tau][wet].min() >= -1.8 - 1e-12


@pytest.mark.slow
def test_the_clamp_acts_on_the_level_the_next_step_reads(veros_model):
    """Only ``tau`` matters: perturbing ``taum1`` changes nothing the next step makes.

    This is the evidence for which level the clamp is applied to
    (``VerosComponent._clamp_surface_at_freezing``): after a step the level
    the next one reads is ``tau``, and its older neighbour's *temperature* is
    never read, so a clamp that touched ``taum1`` would be doing nothing.
    """
    component = VerosComponent(veros_model, clamp_at_freezing=True)
    component.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    carry = _cold_column(component.initialize())
    carry = dict(carry, forcing=carry["forcing"].replace(
        heat_flux=jnp.full(component.horizontal_shape, 4.0 * COOLING)))
    step = jax.jit(component.step)
    first, _ = step(carry, CouplingTime(step=jnp.int32(0), time=START_DATE, dt=DAY))

    from veros.core.operators import at, update
    perturbed_state = jax.tree_util.tree_map(lambda leaf: leaf, first["state"])
    variables = perturbed_state.variables
    taum1 = variables.taum1
    with variables.unlock():
        variables.temp = update(variables.temp, at[..., taum1], 37.0)
        variables.salt = update(variables.salt, at[..., taum1], 12.0)
    perturbed = dict(first, state=perturbed_state)

    time = CouplingTime(step=jnp.int32(1), time=START_DATE, dt=DAY)
    second, _ = step(first, time)
    second_perturbed, _ = step(perturbed, time)
    tau = int(second["state"].variables.tau)
    # The level the step made current is identical whatever taum1 held.
    for name in ("temp", "salt", "u", "v"):
        np.testing.assert_array_equal(
            np.asarray(getattr(second["state"].variables, name))[..., tau],
            np.asarray(getattr(second_perturbed["state"].variables, name))[..., tau],
            err_msg=name)


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_freeze_melt_energy_survives_a_save_and_load(veros_model, tmp_path):
    """The new derived leaves come back from a checkpoint exactly."""
    component = VerosComponent(veros_model, clamp_at_freezing=True)
    component.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    carry = _cold_column(component.initialize())
    carry = dict(carry, forcing=carry["forcing"].replace(
        heat_flux=jnp.full(component.horizontal_shape, 4.0 * COOLING)))
    carry, _ = jax.jit(component.step)(
        carry, CouplingTime(step=jnp.int32(0), time=START_DATE, dt=DAY))
    assert float(jnp.max(carry["derived"].ice_frazil_melt_energy)) > 0.0

    directory = tmp_path / "ocn"
    component.save_carry(carry, directory)
    loaded = component.load_carry(directory)
    for name in ("ice_frazil_melt_energy", "clamps_at_freezing",
                 "sea_surface_temperature", "sea_fraction"):
        np.testing.assert_array_equal(
            np.asarray(getattr(loaded["derived"], name)),
            np.asarray(getattr(carry["derived"], name)), err_msg=name)
    assert float(loaded["derived"].clamps_at_freezing) == 1.0


# ---------------------------------------------------------------------------
# Construction and the refusals
# ---------------------------------------------------------------------------


def test_the_clamp_is_off_by_default_and_published_as_a_flag(veros_model):
    plain = VerosComponent(veros_model)
    assert plain.clamp_at_freezing is False
    assert float(plain.initialize()["derived"].clamps_at_freezing) == 0.0
    assert float(VerosComponent(veros_model, clamp_at_freezing=True)
                 .initialize()["derived"].clamps_at_freezing) == 1.0


def test_from_setup_takes_the_clamp_and_does_not_pass_it_to_the_setup(monkeypatch):
    """``clamp_at_freezing`` is the wrapper's argument; the setup factory never sees it."""
    seen = {}

    class _Model:
        state = None

        def setup(self):
            seen["setup"] = True

    def factory(**kwargs):
        seen["kwargs"] = kwargs
        return _Model()

    import types

    module = types.ModuleType("_fake_veros_setup")
    module.factory = factory
    monkeypatch.setitem(__import__("sys").modules, "_fake_veros_setup", module)
    monkeypatch.setattr(
        VerosComponent, "__init__",
        lambda self, model, *, clamp_at_freezing=False: seen.update(
            clamp=clamp_at_freezing))
    VerosComponent.from_setup("_fake_veros_setup.factory", clamp_at_freezing=True, depth=3)
    assert seen["kwargs"] == {"depth": 3}
    assert seen["clamp"] is True


def test_an_initial_surface_below_freezing_is_raised_when_the_ocean_clamps(
    veros_model, caplog,
):
    """A clamping ocean's initial condition has no deficit; the setup's own state is untouched."""
    from veros.core.operators import at, update

    state = veros_model.state
    cold = jax.tree_util.tree_map(lambda leaf: leaf, state)
    variables = cold.variables
    with variables.unlock():
        variables.temp = update(variables.temp, at[:, :, -1, :], -3.0 * variables.maskT[:, :, -1, None])
    component = VerosComponent(veros_model, clamp_at_freezing=True)
    component.model = type("_M", (), {"state": cold, "set_forcing": None})()
    component.dzt = VerosComponent(veros_model).dzt
    with caplog.at_level(logging.INFO, logger="jem.components.veros_component"):
        carry = component.initialize()
    interior = slice(2, -2)
    wet = np.asarray(veros_model.state.variables.maskT[interior, interior, -1]) > 0
    tau = int(carry["state"].variables.tau)
    surface = np.asarray(carry["state"].variables.temp[interior, interior, -1, tau])
    assert surface[wet].min() >= -1.8 - 1e-12
    assert float(np.max(carry["derived"].ice_frazil_melt_energy)) == 0.0
    assert any("raised to freezing" in record.getMessage() for record in caplog.records)
    # The model's own state, which `initialize` copied, still holds -3 degC.
    assert float(np.min(np.asarray(cold.variables.temp[interior, interior, -1, tau])[wet])) == -3.0


def _initial_components(veros_model, **kwargs):
    coupler, _ = _coupler(veros_model, **kwargs)
    return coupler, dict(coupler.initialize().components)


def test_a_winton_ice_over_a_veros_ocean_that_does_not_clamp_is_refused(veros_model):
    coupler, carries = _initial_components(veros_model, clamp=False)
    with pytest.raises(ValueError, match=r"clamp_at_freezing=True \(\+ocean\.clamp_at_freezing=true\)"):
        coupler.exchangers["exchange"].validate(carries)
    components = coupler.components
    with pytest.raises(ValueError, match=r"clamp_at_freezing"):
        default_exchanges(components)


def test_a_clamping_veros_ocean_with_no_ice_is_refused(veros_model):
    coupler, carries = _initial_components(veros_model, clamp=True, ice=None)
    with pytest.raises(ValueError, match="no sea ice to take the deficit"):
        coupler.exchangers["exchange"].validate(carries)
    with pytest.raises(ValueError, match="no exchange takes the deficit"):
        default_exchanges(coupler.components)


def test_a_clamping_veros_ocean_with_the_slab_ice_is_refused(veros_model):
    coupler, carries = _initial_components(veros_model, clamp=True, ice="slab")
    with pytest.raises(ValueError, match="no energy budget"):
        coupler.exchangers["exchange"].validate(carries)


def test_an_ice_whose_land_is_not_veros_wet_mask_is_refused(veros_model):
    """The ice's sea fraction must be the ocean's wet mask exactly."""
    coupler, carries = _initial_components(veros_model)
    ice = carries["seaice"]
    shifted = dict(carries, seaice=dict(
        ice, derived=ice["derived"].replace(
            sea_fraction=jnp.where(ice["derived"].sea_fraction > 0, 0.7, 0.0))))
    with pytest.raises(ValueError, match="differs from Veros' wet mask"):
        coupler.exchangers["exchange"].validate(shifted)
    coupler.exchangers["exchange"].validate(carries)


def test_the_slab_ice_is_coupled_through_the_exchanger(veros_model):
    """No silently unforced ice: the slab ice gets the SST, the atmosphere its fraction.

    Two coupled steps through ``generate_trajectory_function``; the swamp
    mask is still the ocean's insulation, so the ocean keeps the atmosphere's
    flux except where the surface is at or below freezing.
    """
    coupler, _ = _coupler(veros_model, clamp=False, ice="slab")
    carry = coupler.initialize()
    parts = dict(carry.components)
    parts["ocn"] = _cold_column(parts["ocn"], celsius=-1.9)
    carry = carry.replace(components=parts)
    final, diagnostics = coupler.generate_trajectory_function(2)(carry)
    wet = np.asarray(final.components["ocn"]["derived"].sea_fraction) > 0
    ice_fraction = np.asarray(final.components["seaice"]["derived"].ice_fraction)
    assert ice_fraction[wet].min() == 1.0, "-1.9 degC is below freezing: full cover"
    np.testing.assert_array_equal(
        np.asarray(final.components["atm"]["forcing"].sice_am)[wet],
        np.asarray(diagnostics["seaice"]["derived"].ice_fraction)[0][wet])
    # Swamp mask: the atmosphere's cooling is blocked at freezing.
    np.testing.assert_array_equal(
        np.asarray(final.components["ocn"]["forcing"].heat_flux)[wet], 0.0)


# ---------------------------------------------------------------------------
# The named configuration and the command-line compositions
# ---------------------------------------------------------------------------


def _compose_and_build(overrides):
    from hydra import compose, initialize_config_module

    import jem.config  # noqa: F401
    import jem.runners as runners

    with initialize_config_module(config_module="jem.config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return runners.build_coupler(cfg)


@pytest.mark.slow
def test_veros_earth_builds_with_the_ice_on_the_oceans_wet_mask():
    """``+configuration=veros-earth``: a clamping ocean, the ice on its grid and mask."""
    coupler = _compose_and_build(["+configuration=veros-earth"])
    assert coupler.components["ocn"].clamp_at_freezing is True
    carry = coupler.initialize().components
    np.testing.assert_array_equal(
        np.asarray(carry["seaice"]["derived"].sea_fraction),
        np.asarray(carry["ocn"]["derived"].sea_fraction))
    # What the build's own `validate` accepted, said again: the pairing is
    # matched and the ice is on one sea with the ocean.
    coupler.exchangers["exchange"].validate(carry)


@pytest.mark.slow
def test_a_veros_earth_ice_that_does_not_match_the_ocean_is_refused_with_the_fix():
    """The ice needs the clamp and the ocean's wet mask; each is refused with the fix named."""
    with pytest.raises(ValueError, match=r"\+ocean\.clamp_at_freezing=true"):
        _compose_and_build(["+configuration=veros-earth", "ocean.clamp_at_freezing=false"])
    # The packaged land fraction is fractional, not Veros' thresholded wet mask.
    with pytest.raises(ValueError, match="differs from Veros' wet mask"):
        _compose_and_build([
            "+configuration=veros-earth",
            "seaice.land_fraction_file=${jem_data:landsea_mask_fraction_RotatedGaussianLatLon.nc}"])


@pytest.mark.slow
def test_a_clamping_veros_earth_with_an_ice_that_takes_no_deficit_is_refused():
    """The SST-diagnosed ice has no energy budget, so a clamp under it would create heat."""
    with pytest.raises(ValueError, match="no energy budget and takes no deficit"):
        _compose_and_build(["+configuration=veros-earth", "seaice=slab"])


@pytest.mark.slow
def test_the_packaged_ice_mask_is_the_mask_the_earth_setup_builds():
    """``GridInfo``'s thresholded mask is the complement of the packaged ice land fraction."""
    from importlib import resources

    from jem.components.veros.setups.earth import GridInfo
    from jem.runners import _land_fraction
    from jem.tools.veros_earth_ice_mask import PACKAGED_NAME
    from jem.tools.veros_earth_terrain import LANDSEA_MASK_THRESHOLD

    data = resources.files("jem.data")
    grid_info = GridInfo(
        str(data / "RotatedGaussianLatLon.SCRIP.nc"),
        str(data / "landsea_mask_fraction_RotatedGaussianLatLon.nc"),
        LANDSEA_MASK_THRESHOLD)
    land = _land_fraction(str(data / PACKAGED_NAME))
    np.testing.assert_array_equal(grid_info.landsea_mask, 1.0 - land)
