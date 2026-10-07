"""Closed-form lat-lon regridding weights, read back through ESMFRegridder."""

import jax.numpy as jnp
import numpy as np
import pytest

from jem.utils.esmf_regrid import ESMFRegridder
from jem.utils.latlon_weights import (
    conservative_weights,
    gaussian_latitude_bounds,
    masked_bilinear_weights,
    regular_bounds,
    write_esmf_weights,
)

#: A small Gaussian "atmosphere" (T21's nodal grid) and a regular 5-degree
#: "ocean" that starts at a different longitude and stops short of the poles,
#: as the 1-degree Veros grid does.
ATM_LON = np.arange(64) * 360.0 / 64
ATM_LAT = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(32)[0]))
OCN_LON = 92.5 + 5.0 * np.arange(72)
OCN_LAT = -77.5 + 5.0 * np.arange(32)


def cell_areas(lon_edges, lat_edges):
    return np.outer(np.diff(lon_edges), np.diff(np.sin(np.radians(lat_edges))))


def test_gaussian_bounds_give_each_cell_its_quadrature_weight():
    """A Gaussian cell's area is its Gauss-Legendre weight: global integrals agree."""
    edges = gaussian_latitude_bounds(ATM_LAT)
    assert edges[0] == -90.0 and edges[-1] == 90.0
    assert np.all((edges[:-1] < ATM_LAT) & (ATM_LAT < edges[1:]))
    np.testing.assert_allclose(
        np.diff(np.sin(np.radians(edges))), np.polynomial.legendre.leggauss(32)[1], atol=1e-12)


def test_gaussian_bounds_refuse_latitudes_that_are_not_gaussian():
    with pytest.raises(ValueError):
        gaussian_latitude_bounds(np.linspace(-80, 80, 32))


def test_conservative_weights_preserve_constants_and_integrals(tmp_path):
    """Every destination row sums to one; the area integral survives the map."""
    a_lon_e, a_lat_e = regular_bounds(ATM_LON), gaussian_latitude_bounds(ATM_LAT)
    o_lon_e, o_lat_e = regular_bounds(OCN_LON), regular_bounds(OCN_LAT)
    path = write_esmf_weights(
        str(tmp_path / "a2o.nc"), conservative_weights(a_lon_e, a_lat_e, o_lon_e, o_lat_e),
        src_lon=ATM_LON, src_lat=ATM_LAT, dst_lon=OCN_LON, dst_lat=OCN_LAT, method="conservative")
    regrid = ESMFRegridder(path)
    np.testing.assert_allclose(np.asarray(regrid(jnp.ones((64, 32)))), 1.0, rtol=1e-6)

    rng = np.random.default_rng(0)
    field = rng.normal(size=(64, 32))
    mapped = np.asarray(regrid(jnp.asarray(field)), dtype=float)
    # The destination covers only part of the sphere, so compare with the
    # source integral over exactly that latitude band: re-map the band's own
    # cells' fractions back through the overlap.
    s, row, col = conservative_weights(a_lon_e, a_lat_e, o_lon_e, o_lat_e)
    dst_area = cell_areas(o_lon_e, o_lat_e).ravel(order="F")
    covered_source = np.zeros(64 * 32)
    np.add.at(covered_source, col, s * dst_area[row])
    np.testing.assert_allclose(
        (mapped * cell_areas(o_lon_e, o_lat_e)).sum(),
        (field.ravel(order="F") * covered_source).sum(), rtol=1e-5)
    # ...and the covered area is the band's true area.
    np.testing.assert_allclose(covered_source.sum(), dst_area.sum(), rtol=1e-12)


def test_conservative_weights_between_identical_grids_are_the_identity():
    edges = regular_bounds(OCN_LON), regular_bounds(OCN_LAT)
    s, row, col = conservative_weights(*edges, *edges)
    np.testing.assert_array_equal(row, col)
    np.testing.assert_allclose(s, 1.0)


def test_masked_bilinear_never_reads_a_masked_point(tmp_path):
    """Land values cannot leak into the destination, and nothing is left at zero."""
    mask = np.ones((72, 32), bool)
    mask[10:30, 5:25] = False  # a continent
    sst = np.where(mask, 280.0 + np.cos(np.radians(OCN_LAT))[None, :] * 20.0, -999.0)
    path = write_esmf_weights(
        str(tmp_path / "o2a.nc"), masked_bilinear_weights(OCN_LON, OCN_LAT, mask, ATM_LON, ATM_LAT),
        src_lon=OCN_LON, src_lat=OCN_LAT, dst_lon=ATM_LON, dst_lat=ATM_LAT,
        method="bilinear", src_mask=mask)
    mapped = np.asarray(ESMFRegridder(path)(jnp.asarray(sst)))
    assert mapped.min() >= sst[mask].min() - 1e-3
    assert mapped.max() <= sst[mask].max() + 1e-3


def test_masked_bilinear_is_exact_bilinear_away_from_the_mask():
    """With nothing masked, a field linear in lon and lat is reproduced exactly inside the grid."""
    mask = np.ones((72, 32), bool)
    s, row, col = masked_bilinear_weights(OCN_LON, OCN_LAT, mask, ATM_LON, ATM_LAT)
    lat2 = np.broadcast_to(OCN_LAT[None, :], (72, 32))
    mapped = np.zeros(64 * 32)
    np.add.at(mapped, row, s * lat2.ravel(order="F")[col])
    mapped = mapped.reshape(64, 32, order="F")
    inside = (ATM_LAT > OCN_LAT[0]) & (ATM_LAT < OCN_LAT[-1])
    np.testing.assert_allclose(mapped[:, inside], np.broadcast_to(ATM_LAT[inside], (64, inside.sum())), atol=1e-9)
