"""The atmosphere's land-sea mask and the Veros ocean's wet mask agree (jax-esm#162).

``+configuration=veros-earth`` runs the atmosphere on
``terrain_veros_earth_T31.nc``, whose land fraction is derived from the
ocean's wet mask by :mod:`jem.tools.veros_earth_terrain`. These tests need no
Veros: the wet mask is read from the packaged file the ocean reads, and the
exchange is :class:`jem.fluxes.VerosExchange` on fake carries over the real
packaged ESMF weights. The tests that need a Veros model (the mask the ocean
itself builds, the real configuration's exchange) are in
``test_veros_earth_masks_veros.py``, which runs in a process of its own.

Areas. The atmosphere's cell areas are the weight file's ``area_a`` (ESMF's
own spherical areas, which the conservative weights are normalised with; the
Gaussian quadrature weights differ from them by up to 6 % in a polar cell,
and an integral taken with those would not close however good the map is).
The ocean's are its ``area_b``.
"""

from importlib import resources

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr
from flax import struct

from jem.base.component import CouplingTime
from jem.exchangers import regrid_weighted
from jem.fluxes import VerosExchange
from jem.tools.veros_earth_terrain import (
    apply_weights,
    gaussian_latitudes,
    veros_earth_terrain,
    wet_mask,
)
from jem.utils.esmf_regrid import ESMFRegridder

DATA = resources.files("jem.data")
PACKAGED = str(DATA / "terrain_veros_earth_T31.nc")
ORIGINAL = str(DATA / "terrain_JCM_T31.nc")
MASK_FILE = str(DATA / "landsea_mask_fraction_RotatedGaussianLatLon.nc")
O2A_CONSERVE = str(DATA / "weight_algo-conserve_RotatedGaussianLatLon_to_JCM_T31.nc")
O2A_BILINEAR = str(DATA / "weight_algo-bilinear_RotatedGaussianLatLon_to_JCM_T31.nc")
A2O_CONSERVE = str(DATA / "weight_algo-conserve_JCM_T31_to_RotatedGaussianLatLon.nc")

#: JCM rounds a land fraction within this of 0 or 1 to 0 or 1 when it reads a
#: terrain file (`jcm.terrain.get_terrain`, `fmask_threshold`).
JCM_FMASK_THRESHOLD = 0.1

TIME = CouplingTime(
    step=jnp.int32(0), time=jdt.to_datetime("2001-01-01"), dt=86400.0)


def jcm_sea_fraction(land_fraction):
    """Return the sea fraction JCM derives from a file's ``lsm`` (``1 - fmask``)."""
    fmask = np.where(
        land_fraction <= JCM_FMASK_THRESHOLD, 0.0,
        np.where(land_fraction >= 1.0 - JCM_FMASK_THRESHOLD, 1.0, land_fraction))
    return 1.0 - fmask


@pytest.fixture(scope="module")
def wet():
    """Veros' wet mask as the ocean reads it, ``(96, 48)``."""
    return wet_mask(MASK_FILE, 96, 48)


@pytest.fixture(scope="module")
def areas():
    """``(atmosphere, ocean)`` cell areas, ``(96, 48)`` each, steradians."""
    with xr.open_dataset(A2O_CONSERVE) as weights:
        atmosphere = weights["area_a"].to_numpy().reshape(96, 48, order="F")
        ocean = weights["area_b"].to_numpy().reshape(96, 48, order="F")
    return atmosphere, ocean


# ---------------------------------------------------------------------------
# the packaged file
# ---------------------------------------------------------------------------


def test_the_conservative_weights_map_ones_to_one():
    """The weights' normalisation, which `lsm` staying in [0, 1] rests on."""
    mapped = apply_weights(O2A_CONSERVE, np.ones((96, 48)))
    np.testing.assert_allclose(mapped, 1.0, rtol=0, atol=1e-12)


def test_the_land_fraction_is_one_minus_the_wet_mask_mapped_to_the_atmosphere(wet):
    """The atmosphere's sea area is exactly the wet Veros area under each cell."""
    land_fraction = xr.open_dataset(PACKAGED)["lsm"].to_numpy()
    sea_fraction = apply_weights(O2A_CONSERVE, wet)
    np.testing.assert_allclose(1.0 - land_fraction, sea_fraction, rtol=0, atol=1e-12)
    assert land_fraction.min() >= 0.0 and land_fraction.max() <= 1.0


def test_the_wet_area_is_the_same_on_both_grids(wet, areas):
    """Conservation of the map itself: the atmosphere's sea area is the ocean's."""
    atmosphere_area, ocean_area = areas
    land_fraction = xr.open_dataset(PACKAGED)["lsm"].to_numpy()
    atmosphere_sea = np.sum(atmosphere_area * (1.0 - land_fraction))
    ocean_sea = np.sum(ocean_area * wet)
    np.testing.assert_allclose(atmosphere_sea, ocean_sea, rtol=1e-12)


def test_regenerating_the_terrain_reproduces_the_packaged_file(tmp_path):
    """The generator and the shipped data cannot drift apart."""
    regenerated = xr.open_dataset(veros_earth_terrain(tmp_path / "terrain.nc"))
    packaged = xr.open_dataset(PACKAGED)
    np.testing.assert_allclose(
        regenerated["lsm"].to_numpy(), packaged["lsm"].to_numpy(), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(
        regenerated["orog"].to_numpy(), packaged["orog"].to_numpy())
    np.testing.assert_array_equal(regenerated["lat"], packaged["lat"])
    assert regenerated["lsm"].dtype == packaged["lsm"].dtype
    assert regenerated["orog"].dtype == packaged["orog"].dtype
    for name in ("source_orography", "source_ocean_mask",
                 "ocean_to_atmosphere_weights", "ocean_land_fraction_threshold"):
        assert packaged.attrs[name] == regenerated.attrs[name]


def test_the_orography_is_the_original_files_and_the_layout_is_jcm_canonical():
    original = xr.open_dataset(ORIGINAL)
    packaged = xr.open_dataset(PACKAGED)
    np.testing.assert_array_equal(packaged["orog"].to_numpy(), original["orog"].to_numpy())
    assert packaged["lsm"].dims == ("lon", "lat")
    np.testing.assert_array_equal(packaged["lon"], original["lon"])


def test_the_latitude_labels_are_jcms_own_gaussian_latitudes():
    """jax-gcm reads a terrain file by coordinate, so the labels are the grid's.

    `jcm.data.bc.interpolate` interpolates the file onto the model's latitudes
    even when the file is on that grid, so a label that is off by half a
    degree (as the original file's first and last rows are) mixes a row with
    its neighbour.
    """
    from dinosaur.spherical_harmonic import Grid

    jcm_latitudes = np.asarray(Grid.T31().latitudes) * 180.0 / np.pi
    packaged = xr.open_dataset(PACKAGED)["lat"].to_numpy()
    np.testing.assert_allclose(packaged, jcm_latitudes, rtol=0, atol=1e-10)
    np.testing.assert_allclose(packaged, gaussian_latitudes(48), rtol=0, atol=0)
    original = xr.open_dataset(ORIGINAL)["lat"].to_numpy()
    assert abs(original[0] - jcm_latitudes[0]) > 0.4


def test_jcm_reads_the_packaged_file_without_resampling_it():
    """The land fraction JCM gets is the file's, to its own 0.1 rounding."""
    from dinosaur.spherical_harmonic import Grid
    from jcm.terrain import get_terrain

    _, fmask = get_terrain(terrain_file=PACKAGED, grid=Grid.T31())
    land_fraction = xr.open_dataset(PACKAGED)["lsm"].to_numpy()
    np.testing.assert_allclose(
        1.0 - np.asarray(fmask), jcm_sea_fraction(land_fraction), atol=1e-5)


def test_the_atmosphere_and_ocean_sea_areas_differ_only_by_jcms_rounding(wet, areas):
    """JCM's 0.1 rounding of the land fraction is the only disagreement left."""
    atmosphere_area, ocean_area = areas
    land_fraction = xr.open_dataset(PACKAGED)["lsm"].to_numpy()
    jcm_sea = jcm_sea_fraction(land_fraction)
    difference = np.sum(atmosphere_area * jcm_sea) - np.sum(ocean_area * wet)
    assert abs(difference) / np.sum(ocean_area * wet) < 5e-4
    # What the rounding moves, gross: sea in cells with under 10 % sea, and
    # the land in cells with over 90 %.
    moved = np.abs(jcm_sea - (1.0 - land_fraction))
    assert np.sum(atmosphere_area * moved) / np.sum(ocean_area * wet) < 5e-3


def test_the_generator_refuses_weights_that_do_not_normalise(tmp_path, wet):
    """A weight file whose rows do not sum to one cannot give a sea fraction."""
    with xr.open_dataset(O2A_CONSERVE) as weights:
        broken = weights.load()
    broken["S"] = broken["S"] * 1.01
    path = tmp_path / "broken_weights.nc"
    broken.to_netcdf(path)
    with pytest.raises(ValueError, match="does not normalise"):
        veros_earth_terrain(tmp_path / "out.nc", ocean_to_atmosphere_weights=path)


# ---------------------------------------------------------------------------
# no atmosphere cell with sea is left without ocean
# ---------------------------------------------------------------------------


def _sst_taken_by_sea_cells(land_fraction, wet, regridders):
    """Return the SST each atmosphere cell is handed, for a wet 300 K ocean
    whose land columns hold Veros' 273.15 K fill.
    """
    sst = jnp.asarray(np.where(wet > 0, 300.0, 273.15), dtype=jnp.float32)
    components = _components(land_fraction, wet, sst)
    exchanged = VerosExchange(regrid=regridders)(components, TIME)
    return np.asarray(exchanged["atm"]["forcing"].sea_surface_temperature)


def test_no_atmosphere_cell_with_sea_is_handed_the_land_fill_sst(wet):
    """Before: 110 cells (the bilinear map found no wet column); after: none."""
    packaged = xr.open_dataset(PACKAGED)["lsm"].to_numpy()
    sea = jcm_sea_fraction(packaged) > 0
    taken = _sst_taken_by_sea_cells(packaged, wet, _regridders())
    assert taken[sea].min() > 299.0 - 1e-3
    assert np.all(np.isfinite(taken))

    # The same count for the original terrain, by the unweighted fallback the
    # bilinear map alone would use: this is what the regression looked like.
    original = xr.open_dataset(ORIGINAL)["lsm"].to_numpy()
    original_sea = jcm_sea_fraction(original) > 0
    bilinear = ESMFRegridder(O2A_BILINEAR)
    fill = np.where(wet > 0, 300.0, 273.15)
    old_sst = np.asarray(regrid_weighted(
        bilinear, jnp.asarray(fill, dtype=jnp.float32),
        jnp.asarray(wet, dtype=jnp.float32)))
    assert int(np.sum(original_sea & (old_sst < 290.0))) == 110


def test_the_bilinear_map_alone_leaves_a_few_cells_the_conservative_map_serves(wet):
    """Why VerosExchange backs the bilinear SST up with the conservative map.

    A bilinear map reads the four nearest columns, so a cell with a sliver of
    wet ocean can have none of them wet even when the masks agree.
    """
    sea = jcm_sea_fraction(xr.open_dataset(PACKAGED)["lsm"].to_numpy()) > 0
    bilinear = apply_weights(O2A_BILINEAR, wet)
    conservative = apply_weights(O2A_CONSERVE, wet)
    assert int(np.sum(sea & (bilinear <= 0))) == 10
    assert int(np.sum(sea & (conservative <= 0))) == 0


def test_check_masks_raises_when_an_atmosphere_cell_has_sea_but_no_ocean():
    """The disagreement is named, counted and has a remedy."""
    sea = np.array([1.0, 0.4, 0.0, 0.2])
    reached = np.array([1.0, 0.0, 0.0, 0.5])
    with pytest.raises(ValueError, match=r"1 atmosphere cell.*veros_earth_terrain"):
        VerosExchange.check_masks(sea, reached)
    VerosExchange.check_masks(sea, np.array([1.0, 0.3, 0.0, 0.5]))


def test_the_exchange_raises_on_its_first_step_for_a_coastline_that_disagrees(wet):
    """The original terrain with the ocean's mask: 85 cells have sea, no ocean."""
    original = xr.open_dataset(ORIGINAL)["lsm"].to_numpy()
    sst = jnp.full((96, 48), 300.0, dtype=jnp.float32)
    components = _components(original, wet, sst)
    exchange = VerosExchange(regrid=_regridders())
    with pytest.raises(Exception, match="no wet ocean column"):
        jax.block_until_ready(exchange(components, TIME))


def test_the_exchange_does_not_check_after_its_first_step(wet):
    """The check is a first-step guard, so a later step does not call back."""
    original = xr.open_dataset(ORIGINAL)["lsm"].to_numpy()
    sst = jnp.full((96, 48), 300.0, dtype=jnp.float32)
    components = _components(original, wet, sst)
    later = CouplingTime(step=jnp.int32(7), time=TIME.time, dt=TIME.dt)
    jax.block_until_ready(VerosExchange(regrid=_regridders())(components, later))


# ---------------------------------------------------------------------------
# conservation of what crosses
# ---------------------------------------------------------------------------


def _synthetic_flux():
    """Return a non-uniform surface flux, W m-2 (upward-positive): zonal and
    meridional structure plus a wave, as large as a real heat flux.
    """
    latitude = np.broadcast_to(gaussian_latitudes(48)[None, :], (96, 48))
    longitude = np.broadcast_to(3.75 * np.arange(96)[:, None], (96, 48))
    return (150.0 * np.cos(np.radians(latitude))
            * (1.0 + 0.5 * np.sin(np.radians(3.0 * longitude))) - 40.0)


def test_the_heat_the_ocean_receives_is_the_heat_the_atmosphere_gives(wet, areas):
    """Area-integrated sea-tile heat flux, atmosphere out against ocean in.

    The atmosphere's integral is ``sum(area_atm * sea_fraction * flux)`` with
    the sea fraction JCM uses; the ocean's is ``sum(area_ocn * wet * flux)``.
    The two maps differ in how a coastal ocean column is normalised (by the
    sea fraction mapped onto it), which is a second-order residual in the
    flux's variation across a coastal cell; it measures 5e-4 of the integral
    here (1.6e-4 for a uniform flux, which is JCM's 0.1 rounding alone) and
    the tolerance is 1e-3. With the original terrain the same integral is 3e-3
    out, which is what this test would see without the derived coastline.
    """
    atmosphere_area, ocean_area = areas
    land_fraction = xr.open_dataset(PACKAGED)["lsm"].to_numpy()
    sea_fraction = jcm_sea_fraction(land_fraction)
    flux = _synthetic_flux()

    components = _components(
        land_fraction, wet, jnp.full((96, 48), 300.0, dtype=jnp.float32),
        heat_flux=flux)
    exchanged = VerosExchange(regrid=_regridders())(components, TIME)
    received = np.asarray(exchanged["ocn"]["forcing"].heat_flux, dtype=np.float64)

    given = np.sum(atmosphere_area * sea_fraction * flux)
    taken = np.sum(ocean_area * wet * received)
    assert abs(taken - given) / abs(given) < 1e-3

    # The original coastline, through the same a2o map and weighting.
    original_sea = jcm_sea_fraction(xr.open_dataset(ORIGINAL)["lsm"].to_numpy())
    a2o = ESMFRegridder(A2O_CONSERVE)
    old_received = np.asarray(regrid_weighted(
        a2o, jnp.asarray(flux, dtype=jnp.float32),
        jnp.asarray(original_sea, dtype=jnp.float32)), dtype=np.float64)
    old_given = np.sum(atmosphere_area * original_sea * flux)
    old_taken = np.sum(ocean_area * wet * old_received)
    assert abs(old_taken - old_given) / abs(old_given) > 2e-3


# ---------------------------------------------------------------------------
# fakes: carries carrying exactly the fields VerosExchange reads and writes
# ---------------------------------------------------------------------------


@struct.dataclass
class _AtmDerived:
    sea_eastward_wind_stress: jax.Array
    sea_northward_wind_stress: jax.Array
    sea_total_heat_flux: jax.Array
    sea_total_freshwater_flux: jax.Array
    sea_fraction: jax.Array


@struct.dataclass
class _AtmForcing:
    sea_surface_temperature: jax.Array


@struct.dataclass
class _OcnForcing:
    surface_taux: jax.Array
    surface_tauy: jax.Array
    heat_flux: jax.Array
    freshwater_flux: jax.Array


@struct.dataclass
class _OcnDerived:
    sea_surface_temperature: jax.Array
    sea_fraction: jax.Array


def _regridders():
    """Return the regridders `regrid=esmf` gives `VerosExchange`, by role."""
    return {
        "a2o_flux": ESMFRegridder(A2O_CONSERVE),
        "o2a_state": ESMFRegridder(O2A_BILINEAR),
        "o2a_flux": ESMFRegridder(O2A_CONSERVE),
    }


def _components(land_fraction, wet, sea_surface_temperature, heat_flux=None):
    """Fake ``atm``/``ocn`` carries on the real grids."""
    zeros = jnp.zeros((96, 48), dtype=jnp.float32)
    flux = zeros if heat_flux is None else jnp.asarray(heat_flux, dtype=jnp.float32)
    return {
        "atm": {
            "derived": _AtmDerived(
                zeros, zeros, flux, zeros,
                jnp.asarray(jcm_sea_fraction(land_fraction), dtype=jnp.float32)),
            "forcing": _AtmForcing(jnp.full((96, 48), 280.0, dtype=jnp.float32)),
        },
        "ocn": {
            "forcing": _OcnForcing(zeros, zeros, zeros, zeros),
            "derived": _OcnDerived(
                sea_surface_temperature, jnp.asarray(wet, dtype=jnp.float32)),
        },
    }
