"""The Veros half of the mask-agreement tests (jax-esm#162).

``test_veros_earth_masks.py`` holds everything that needs no Veros. These
tests build the real ``+configuration=veros-earth`` coupler, so they import
Veros, which flips ``jax_enable_x64`` for the whole process: **run this file
in a process of its own** (see ``test_veros_setups.py`` for the same rule and
the reasoning; the "slow unit tests" step of ``.github/workflows/tests.yml``
lists the Veros files one invocation each). They are skipped when Veros is
not installed.
"""

from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("veros")

from jem.components import veros_component  # noqa: E402, F401

DATA = resources.files("jem.data")
MASK_FILE = str(DATA / "landsea_mask_fraction_RotatedGaussianLatLon.nc")
SCRIP_FILE = str(DATA / "RotatedGaussianLatLon.SCRIP.nc")
PACKAGED = str(DATA / "terrain_veros_earth_T31.nc")
A2O_CONSERVE = str(DATA / "weight_algo-conserve_JCM_T31_to_RotatedGaussianLatLon.nc")

_JAX_X64_BEFORE_ANY_VEROS_IMPORT = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True, scope="module")
def _veros_x64():
    """Veros needs ``jax_enable_x64`` for this module; restore it once, after."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _JAX_X64_BEFORE_ANY_VEROS_IMPORT)


@pytest.fixture(scope="module")
def coupler():
    """Build the real ``veros-earth`` coupled model once; nothing is stepped."""
    from hydra import compose, initialize_config_module

    import jem.config  # noqa: F401
    import jem.runners as runners

    with initialize_config_module(config_module="jem.config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["+configuration=veros-earth"])
    return runners.build_coupler(cfg)


@pytest.mark.slow
def test_the_wet_mask_the_terrain_is_derived_from_is_the_one_veros_builds():
    """``jem.tools.veros_earth_terrain.wet_mask`` is ``GridInfo``'s mask.

    The generator cannot import ``GridInfo`` (it needs Veros), so it reads the
    same file with the same threshold; this holds the two together.
    """
    from jem.components.veros.setups.earth import GridInfo
    from jem.tools.veros_earth_terrain import LANDSEA_MASK_THRESHOLD, wet_mask

    grid_info = GridInfo(SCRIP_FILE, MASK_FILE, LANDSEA_MASK_THRESHOLD)
    np.testing.assert_array_equal(
        grid_info.landsea_mask, wet_mask(MASK_FILE, grid_info.nx, grid_info.ny))


@pytest.mark.slow
def test_the_veros_earth_coastlines_agree(coupler):
    """The atmosphere's sea fraction is the wet Veros area under each cell.

    Read off the built components, not the files: what the atmosphere carries
    as ``derived.sea_fraction`` against the wet mask Veros itself publishes,
    mapped with the conservative ocean-to-atmosphere weights, to JCM's own 0.1
    rounding of the land fraction (`jcm.terrain.get_terrain`).
    """
    from jem.tools.veros_earth_terrain import apply_weights

    carry = coupler.initialize()
    sea = np.asarray(carry.components["atm"]["derived"].sea_fraction, np.float64)
    wet = np.asarray(carry.components["ocn"]["derived"].sea_fraction, np.float64)
    ocean_fraction = apply_weights(
        str(DATA / "weight_algo-conserve_RotatedGaussianLatLon_to_JCM_T31.nc"), wet)
    assert np.abs(sea - ocean_fraction).max() <= 0.1 + 1e-6
    # The cells JCM does not round are exact.
    unrounded = (ocean_fraction > 0.1) & (ocean_fraction < 0.9)
    np.testing.assert_allclose(sea[unrounded], ocean_fraction[unrounded], atol=1e-6)


@pytest.mark.slow
def test_one_exchange_conserves_the_sea_tile_heat_and_freshwater(coupler):
    """What the atmosphere's sea tile gives up, the sea ice receives.

    The ocean takes its heat and freshwater from the Winton ice, which takes
    the atmosphere's sea-tile heat flux and evaporation through the same
    conservative map the ocean would have used (the ice is on the ocean's grid
    and wet mask), so the atmosphere-to-ice leg is what this checks; the ice's
    own budget is the subject of ``test_veros_seaice.py``.

    The real exchanger on the initial carry with a *non-uniform* synthetic
    heat flux (zonal and meridional structure plus a wave) and evaporation
    (standing for the freshwater flux).
    The atmosphere's integral is ``sum(area_atm * sea_fraction * flux)``, the
    ice's ``sum(area_t * wet * flux)`` with Veros' own ``area_t``, both in
    m2 (the atmosphere's cell areas are the weight file's, in steradians,
    times the radius squared that makes Veros' grid sum to the sphere). The
    residual is the second-order one the sea-fraction normalisation of a
    coastal ocean column leaves, 5e-4 of the integral for this flux (1.6e-4
    for a uniform one, which is JCM's 0.1 rounding alone); the tolerance is
    1e-3. The original terrain, which this test does not use, is 3e-3 out.
    """
    carry = coupler.initialize()
    atm, ocn = carry.components["atm"], carry.components["ocn"]
    latitude = np.broadcast_to(
        np.asarray(xr.open_dataset(PACKAGED)["lat"])[None, :], (96, 48))
    longitude = np.broadcast_to(3.75 * np.arange(96)[:, None], (96, 48))
    heat = (150.0 * np.cos(np.radians(latitude))
            * (1.0 + 0.5 * np.sin(np.radians(3.0 * longitude))) - 40.0)
    freshwater = 3e-5 * (1.0 + 0.5 * np.cos(np.radians(2.0 * latitude))) * (
        1.0 + 0.5 * np.cos(np.radians(5.0 * longitude)))
    dtype = atm["derived"].sea_total_heat_flux.dtype
    atm = dict(atm, derived=atm["derived"].replace(
        sea_total_heat_flux=jnp.asarray(heat, dtype),
        sea_evaporation=jnp.asarray(freshwater, dtype)))
    components = dict(carry.components, atm=atm)

    exchange = next(iter(coupler.exchangers.values()))
    time = coupler.coupling_time(carry.step, carry.time)
    exchanged = exchange(components, time)

    area_t = np.asarray(ocn_area(coupler))
    with xr.open_dataset(A2O_CONSERVE) as weights:
        atmosphere_area = weights["area_a"].to_numpy().reshape(96, 48, order="F")
    atmosphere_area = atmosphere_area * (area_t.sum() / (4 * np.pi))
    sea = np.asarray(atm["derived"].sea_fraction, np.float64)
    wet = np.asarray(ocn["derived"].sea_fraction, np.float64)
    forcing = exchanged["seaice"]["forcing"]
    for given_field, received_field in ((heat, forcing.total_heat_flux),
                                        (freshwater, forcing.evaporation)):
        given = np.sum(atmosphere_area * sea * given_field)
        received = np.sum(area_t * wet * np.asarray(received_field, np.float64))
        assert abs(received - given) / abs(given) < 1e-3, (received, given)

    # No atmosphere cell with sea is handed the land fill SST (273.15 K): the
    # regridding weights are convex, so every sea cell's SST lies within the
    # range of the wet ocean's own (which, from an observed initial state, runs
    # from the freezing point to 30 degC and so includes the fill value; the
    # bound is what shows that no land column's value leaked into a sea cell).
    sst = np.asarray(exchanged["atm"]["forcing"].sea_surface_temperature, np.float64)
    wet_sst = np.asarray(ocn["derived"].sea_surface_temperature, np.float64)[wet > 0]
    sea_sst = sst[sea > 0]
    assert wet_sst.min() - 1e-3 <= sea_sst.min() and sea_sst.max() <= wet_sst.max() + 1e-3


def ocn_area(coupler):
    """Veros' ``area_t`` on the interior ``(96, 48)`` columns, m2."""
    from jem.components.veros_component import GHOST_CELLS

    interior = slice(GHOST_CELLS, -GHOST_CELLS)
    return coupler.components["ocn"].model.state.variables.area_t[interior, interior]


@pytest.mark.slow
def test_two_coupled_steps_run_with_the_mask_check_in_the_scan(coupler):
    """The first-step mask check traces inside ``lax.scan`` and passes."""
    carry = coupler.initialize()
    final, _ = coupler.generate_trajectory_function(2)(carry)
    assert int(final.step) == 2
    for name, component_carry in final.components.items():
        for leaf in jax.tree_util.tree_leaves(component_carry):
            leaf = jnp.asarray(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.inexact):
                assert bool(jnp.all(jnp.isfinite(leaf))), name
