"""Regridding weights between two lat-lon grids, written as ESMF weight files.

JAX-ESM applies ESMF weights (:class:`jem.utils.esmf_regrid.ESMFRegridder`)
and in general does not compute them: for an arbitrary pair of grids -- a
rotated or displaced-pole ocean, say -- that is ``ESMF_RegridWeightGen``'s
job, and the packaged weights were made with it. Between two grids whose
cells are all bounded by meridians and parallels (a regular lat-lon ocean, a
Gaussian spectral atmosphere) the weights have a closed form, though, and
this module computes them exactly, so a new resolution pair needs no ESMF
installation:

- :func:`conservative_weights` -- first-order conservative: each destination
  cell receives the area-weighted mean of the source cells it overlaps, the
  overlap of two such cells being separable into a longitude overlap and an
  overlap in ``sin(latitude)``, so it is exact on the sphere;
- :func:`masked_bilinear_weights` -- bilinear interpolation from the *wet*
  source points only, renormalised over the ones present, and the nearest wet
  source point wherever none of the four is (a coast, or beyond the source
  grid's latitude range). This is what ``ESMF_RegridWeightGen`` produces with
  a source mask and ``--extrap_method neareststod``, and what a sea surface
  temperature sent to an atmosphere needs: land points of an ocean model
  carry no temperature, so they must not be averaged into the coastal
  atmosphere cells that are partly ocean.

:func:`write_esmf_weights` writes either in the ESMF/SCRIP format
(``S``/``row``/``col``, 1-based, fields flattened with longitude fastest)
that :class:`~jem.utils.esmf_regrid.ESMFRegridder` reads.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr


def regular_bounds(centres: Sequence[float]) -> np.ndarray:
    """Cell edges of a uniformly spaced axis given its centres (``n + 1``)."""
    c = np.asarray(centres, dtype=float)
    step = np.diff(c).mean()
    if not np.allclose(np.diff(c), step):
        raise ValueError("regular_bounds needs uniformly spaced centres")
    return np.concatenate([c - step / 2, [c[-1] + step / 2]])


def gaussian_latitude_bounds(latitudes: Sequence[float]) -> np.ndarray:
    """Cell edges (degrees) of a Gaussian latitude axis, south to north.

    A Gaussian grid's cell boundaries are where the running sum of the
    Gauss-Legendre quadrature weights reaches each node: ``sin`` of the edge
    below node ``j`` is ``-1 + sum(w[:j])``. That makes each cell's area
    exactly its quadrature weight, which is the area a spectral model's own
    global integrals assign it.

    Parameters
    ----------
    latitudes : Sequence[float]
        The Gaussian latitudes in degrees, south to north; checked against
        the Gauss-Legendre nodes of the same length.

    """
    lat = np.asarray(latitudes, dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(lat.size)
    if not np.allclose(np.degrees(np.arcsin(nodes)), lat, atol=1e-6):
        raise ValueError("latitudes are not the Gaussian latitudes of their length")
    sin_edges = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    # The weights sum to 2 only to rounding; the last edge is the pole exactly.
    sin_edges[-1] = 1.0
    edges: np.ndarray = np.degrees(np.arcsin(np.clip(sin_edges, -1.0, 1.0)))
    return edges


def _periodic_overlap(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """``(n_dst, n_src)`` longitude overlaps (degrees) of two periodic axes."""
    src_lo, src_hi = src_edges[:-1], src_edges[1:]
    dst_lo, dst_hi = dst_edges[:-1, None], dst_edges[1:, None]
    overlap: np.ndarray = np.zeros((dst_lo.size, src_lo.size))
    # The two axes may start at different longitudes (0E and 90.5E, say), so
    # the source is compared at its own position and one period either side.
    for shift in (-360.0, 0.0, 360.0):
        overlap += np.clip(
            np.minimum(dst_hi, src_hi + shift) - np.maximum(dst_lo, src_lo + shift),
            0.0, None,
        )
    return overlap


def _latitude_overlap(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """``(n_dst, n_src)`` overlaps in ``sin(latitude)`` -- proportional to area."""
    s_src, s_dst = np.sin(np.radians(src_edges)), np.sin(np.radians(dst_edges))
    overlap: np.ndarray = np.clip(
        np.minimum(s_dst[1:, None], s_src[None, 1:])
        - np.maximum(s_dst[:-1, None], s_src[None, :-1]),
        0.0, None,
    )
    return overlap


def conservative_weights(
    src_lon_edges: np.ndarray,
    src_lat_edges: np.ndarray,
    dst_lon_edges: np.ndarray,
    dst_lat_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-order conservative weights between two lat-lon grids.

    Each destination cell gets the source values weighted by the fraction of
    its own area each source cell covers (ESMF's ``conserve`` with
    destination-area normalisation). Wherever the source grid covers a
    destination cell completely the weights of that cell sum to one, and the
    area integral of any field over the destination grid equals the integral
    over the part of the source grid it covers.

    Parameters
    ----------
    src_lon_edges, src_lat_edges, dst_lon_edges, dst_lat_edges : numpy.ndarray
        Cell edges in degrees, ``n + 1`` each; latitudes ascending.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S, row, col)``: weights and **0-based** destination/source indices
        into fields flattened with longitude varying fastest.

    """
    w_lon = _periodic_overlap(np.asarray(src_lon_edges, float), np.asarray(dst_lon_edges, float))
    w_lon /= np.diff(dst_lon_edges)[:, None]
    w_lat = _latitude_overlap(np.asarray(src_lat_edges, float), np.asarray(dst_lat_edges, float))
    w_lat /= np.diff(np.sin(np.radians(dst_lat_edges)))[:, None]

    n_dst_lon, n_src_lon = w_lon.shape
    di, si = np.nonzero(w_lon)
    dj, sj = np.nonzero(w_lat)
    # Outer product of the two sparse 1-D overlaps.
    rows = (di[:, None] + n_dst_lon * dj[None, :]).ravel()
    cols = (si[:, None] + n_src_lon * sj[None, :]).ravel()
    weights = (w_lon[di, si][:, None] * w_lat[dj, sj][None, :]).ravel()
    keep = weights > 0
    return weights[keep], rows[keep], cols[keep]


def _unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    return np.stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)], axis=-1
    )


def masked_bilinear_weights(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    src_mask: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear weights from the unmasked source points, with a nearest fill.

    Parameters
    ----------
    src_lon, src_lat : numpy.ndarray
        Source cell centres, degrees: a periodic, uniformly spaced longitude
        axis and an ascending latitude axis.
    src_mask : numpy.ndarray
        ``(n_lon, n_lat)``, true where the source holds a value (ocean).
    dst_lon, dst_lat : numpy.ndarray
        Destination cell-centre axes, degrees.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S, row, col)`` as in :func:`conservative_weights`. Every
        destination point's weights sum to one, so a constant is preserved
        everywhere and no destination point is ever left at zero.

    """
    from scipy.spatial import cKDTree

    src_lon = np.asarray(src_lon, float)
    src_lat = np.asarray(src_lat, float)
    mask = np.asarray(src_mask, bool)
    n_src_lon, n_src_lat = mask.shape
    dlon = 360.0 / n_src_lon

    lon2, lat2 = np.meshgrid(np.asarray(dst_lon, float), np.asarray(dst_lat, float), indexing="ij")
    lon_flat, lat_flat = lon2.ravel(order="F"), lat2.ravel(order="F")
    n_dst = lon_flat.size

    # Longitude: fractional index into the periodic source axis.
    x = ((lon_flat - src_lon[0]) % 360.0) / dlon
    i0 = np.floor(x).astype(int) % n_src_lon
    i1 = (i0 + 1) % n_src_lon
    fx = x - np.floor(x)
    # Latitude: bracketing rows; outside the source range there are none.
    j1 = np.searchsorted(src_lat, lat_flat)
    inside = (j1 > 0) & (j1 < n_src_lat)
    j1c = np.clip(j1, 1, n_src_lat - 1)
    j0c = j1c - 1
    fy = (lat_flat - src_lat[j0c]) / (src_lat[j1c] - src_lat[j0c])

    corners = [
        (i0, j0c, (1 - fx) * (1 - fy)),
        (i1, j0c, fx * (1 - fy)),
        (i0, j1c, (1 - fx) * fy),
        (i1, j1c, fx * fy),
    ]
    weight = np.stack([w * mask[i, j] * inside for i, j, w in corners], axis=1)
    col = np.stack([i + n_src_lon * j for i, j, _ in corners], axis=1)
    total = weight.sum(axis=1)
    has_wet = total > 1e-12
    weight[has_wet] /= total[has_wet, None]

    rows = [np.repeat(np.nonzero(has_wet)[0], 4)]
    cols = [col[has_wet].ravel()]
    vals = [weight[has_wet].ravel()]

    # Nearest wet source point (great-circle) for everything else.
    need = np.nonzero(~has_wet)[0]
    if need.size:
        src_lon2, src_lat2 = np.meshgrid(src_lon, src_lat, indexing="ij")
        wet_index = np.nonzero(mask.ravel(order="F"))[0]
        tree = cKDTree(_unit_vectors(src_lon2.ravel(order="F")[wet_index],
                                     src_lat2.ravel(order="F")[wet_index]))
        _, nearest = tree.query(_unit_vectors(lon_flat[need], lat_flat[need]))
        rows.append(need)
        cols.append(wet_index[nearest])
        vals.append(np.ones(need.size))

    row, colv, val = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    keep = val > 0
    assert np.unique(row[keep]).size == n_dst
    return val[keep], row[keep], colv[keep]


def write_esmf_weights(
    path: str,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
    method: str,
    src_mask: np.ndarray | None = None,
) -> str:
    """Write ``(S, row, col)`` as an ESMF weight file and return ``path``.

    Parameters
    ----------
    path : str
        Output netCDF path.
    weights : tuple of numpy.ndarray
        ``(S, row, col)`` with 0-based indices, as the functions above return.
    src_lon, src_lat, dst_lon, dst_lat : numpy.ndarray
        Each grid's cell-centre axes, degrees. They fix ``grid_dims``
        (``(n_lon, n_lat)``) and are written as ESMF's ``xc_a``/``yc_a``/
        ``xc_b``/``yc_b`` centre coordinates.
    method : str
        Recorded as the file's ``map_method`` attribute.
    src_mask : numpy.ndarray, optional
        ``(n_lon, n_lat)`` source mask, written as ``mask_a`` (1 = valid).

    """
    def flat_centres(lon, lat):
        lon2, lat2 = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float), indexing="ij")
        return lon2.ravel(order="F"), lat2.ravel(order="F")

    s, row, col = weights
    xc_a, yc_a = flat_centres(src_lon, src_lat)
    xc_b, yc_b = flat_centres(dst_lon, dst_lat)
    variables = {
        "S": ("n_s", np.asarray(s, dtype=np.float64)),
        "row": ("n_s", np.asarray(row, dtype=np.int32) + 1),
        "col": ("n_s", np.asarray(col, dtype=np.int32) + 1),
        "src_grid_dims": ("src_grid_rank", np.array([len(src_lon), len(src_lat)], dtype=np.int32)),
        "dst_grid_dims": ("dst_grid_rank", np.array([len(dst_lon), len(dst_lat)], dtype=np.int32)),
        "xc_a": ("n_a", xc_a),
        "yc_a": ("n_a", yc_a),
        "xc_b": ("n_b", xc_b),
        "yc_b": ("n_b", yc_b),
    }
    if src_mask is not None:
        variables["mask_a"] = ("n_a", np.asarray(src_mask, dtype=np.int32).ravel(order="F"))
    xr.Dataset(variables, attrs={"map_method": method, "title": __name__}).to_netcdf(path)
    return path
