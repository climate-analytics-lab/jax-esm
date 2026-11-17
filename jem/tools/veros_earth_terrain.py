"""Generate the terrain file ``+configuration=veros-earth`` runs the atmosphere on.

The two components of ``veros-earth`` draw their coastlines from independent
files: JCM reads a *fractional* land mask (``lsm``), while Veros thresholds an
ERA5 land fraction on its own rotated Gaussian grid into a *binary* wet mask.
Left alone they disagree on about 4 % of the ocean area each way, and every
disagreement is a flux that is either computed over water the ocean does not
have or is handed to water the atmosphere never exchanged with (see
``docs/source/design/exchange.md``).

:func:`veros_earth_terrain` makes the ocean's mask authoritative and derives
the atmosphere's land fraction from it, the way CESM derives its surface
fractions (``lfrac = 1 - ofrac``):

1. the wet mask is read exactly as the ocean reads it
   (:func:`wet_mask`: the same file, the same threshold, the same index
   layout as ``jem.components.veros.setups.earth.GridInfo``);
2. it is mapped onto the atmosphere's grid with the packaged *conservative*
   ocean-to-atmosphere ESMF weights -- the very weights ``regrid=esmf`` maps
   the sea surface temperature's weight field with -- so each atmosphere
   cell's sea fraction ``ofrac`` is exactly the wet Veros area under it
   divided by the cell's area;
3. ``lsm = 1 - ofrac``, and the orography is carried over from a reference
   terrain file.

The output is JCM-canonical (:meth:`jcm.terrain.TerrainData.from_file`'s
layout: dims ``(lon, lat)``, variables ``lsm`` and ``orog``).

Latitude labels
---------------
:meth:`jcm.terrain.TerrainData.from_file` does not read a file by index: it
interpolates the file's fields *by coordinate* (``jcm.data.bc.interpolate``)
onto the model's Gaussian latitudes, even when the file is already on that
grid. The file's ``lat`` labels therefore have to be the model's own
latitudes -- anything else is a (small) smoothing in the rows whose label is
off -- so the output is labelled with the true Gaussian latitudes
(:func:`gaussian_latitudes`), in degrees computed as ``radians * 180 / pi``
like every other coordinate in JEM. The reference file's fields are taken by
index -- they are on JCM's grid, whatever its own labels say -- and only the
labels are written here.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: The ocean's land-fraction threshold: a column is dry where the mask file's
#: land fraction is at least this. It is ``earth_setup``'s default
#: ``landsea_mask_threshold``, and the value ``veros-earth`` uses.
LANDSEA_MASK_THRESHOLD = 0.5

#: Round-off the conservative weights carry: their row sums are 1 to about
#: 3e-14, so a mapped fraction can leave [0, 1] by that much and no more.
_WEIGHT_ROUNDOFF = 1e-12


def _data_file(name: str) -> str:
    """Return the path of a file packaged in ``jem.data``."""
    return str(resources.files("jem.data") / name)


def wet_mask(
    landsea_mask_file: str | Path,
    n_lon: int,
    n_lat: int,
    threshold: float = LANDSEA_MASK_THRESHOLD,
) -> np.ndarray:
    """Return Veros' binary wet mask, ``1`` = ocean, ``(n_lon, n_lat)``.

    The same computation as ``GridInfo.get_grid_info``
    (``jem.components.veros.setups.earth``), which cannot be imported without
    Veros: the file's ``lsm`` is read flat in SCRIP order and reshaped to
    ``(n_lat, n_lon)``, a column is land where it is at least ``threshold``,
    and the result is transposed to Veros' ``(xt, yt)`` layout.
    ``tests/unit/test_veros_earth_masks.py`` holds the two to each other when
    Veros is installed.

    Parameters
    ----------
    landsea_mask_file : str or pathlib.Path
        A land-fraction file with variable ``lsm`` (``1`` = land) on the
        ocean grid, e.g. ``landsea_mask_fraction_RotatedGaussianLatLon.nc``.
    n_lon, n_lat : int
        The ocean grid's shape.
    threshold : float
        Land fraction at and above which a column is dry.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of ``0.0`` and ``1.0``, shape ``(n_lon, n_lat)``.

    """
    with xr.open_dataset(landsea_mask_file) as mask_ds:
        land_fraction = mask_ds["lsm"].to_numpy().reshape(n_lat, n_lon)
    is_land = land_fraction >= threshold
    return (1 - is_land.astype(int)).transpose().astype(np.float64)


def apply_weights(weight_file: str | Path, field: np.ndarray) -> np.ndarray:
    """Apply an ESMF weight file to a ``(n_lon, n_lat)`` field, in float64.

    The same sparse product :class:`jem.utils.esmf_regrid.ESMFRegridder`
    applies, but in double precision on NumPy arrays, so the result can be
    compared with the weights' own normalisation to round-off.
    """
    with xr.open_dataset(weight_file) as weights:
        src_shape = tuple(int(n) for n in weights["src_grid_dims"].to_numpy())
        dst_shape = tuple(int(n) for n in weights["dst_grid_dims"].to_numpy())
        row = weights["row"].to_numpy() - 1
        col = weights["col"].to_numpy() - 1
        value = weights["S"].to_numpy()
    if field.shape != src_shape:
        raise ValueError(
            f"{weight_file} maps a {src_shape} grid but the field is "
            f"{field.shape}."
        )
    flat = field.reshape(-1, order="F")
    mapped = np.bincount(
        row, weights=value * flat[col], minlength=int(np.prod(dst_shape)))
    return mapped.reshape(dst_shape, order="F")


def gaussian_latitudes(n_lat: int) -> np.ndarray:
    """Return the ``n_lat`` Gaussian latitudes in degrees, south to north.

    The Gauss-Legendre nodes' arcsine, which is what ``dinosaur`` (and so
    JCM) uses for its Gaussian grids; degrees are ``radians * 180 / pi`` in
    float64, as :func:`jem.components.slab.grid.to_degrees` computes them.
    """
    sine_latitude, _ = np.polynomial.legendre.leggauss(n_lat)
    return np.arcsin(sine_latitude) * 180.0 / np.pi


def veros_earth_terrain(
    output_file: str | Path,
    *,
    reference_file: str | Path | None = None,
    landsea_mask_file: str | Path | None = None,
    ocean_to_atmosphere_weights: str | Path | None = None,
    threshold: float = LANDSEA_MASK_THRESHOLD,
) -> Path:
    """Write a JCM terrain file whose sea is exactly Veros' wet area.

    Parameters
    ----------
    output_file : str or pathlib.Path
        Path to write; its parent directory is created if needed.
    reference_file : str or pathlib.Path, optional
        A JCM-canonical terrain file supplying the orography (taken by
        index; see the module docstring). Default the packaged
        ``terrain_JCM_T31.nc``.
    landsea_mask_file : str or pathlib.Path, optional
        The ocean's land-fraction file. Default the packaged
        ``landsea_mask_fraction_RotatedGaussianLatLon.nc``, the one
        ``+configuration=veros-earth`` gives the ocean.
    ocean_to_atmosphere_weights : str or pathlib.Path, optional
        The conservative ocean-to-atmosphere ESMF weights. Default the
        packaged ``weight_algo-conserve_RotatedGaussianLatLon_to_JCM_T31.nc``,
        the one ``regrid=esmf`` names ``o2a_conserve``.
    threshold : float
        The ocean's land-fraction threshold for a dry column; see
        :func:`wet_mask`.

    Returns
    -------
    pathlib.Path
        ``output_file``.

    Raises
    ------
    ValueError
        If the weights' source grid is not the mask's grid or their
        destination grid is not the reference terrain's, or if a mapped sea
        fraction leaves [0, 1] by more than weight round-off (weights that
        do not normalise to one over the destination cell they cover).

    Notes
    -----
    ``lsm`` is stored in float64: it is ``1 - ofrac`` to round-off, and the
    consistency of the two masks is the point of the file. A cell whose sea
    fraction is within 1e-12 of 0 or 1 is stored as exactly 0 or 1 (the
    weights' own normalisation error). JCM itself then rounds an ``lsm``
    within 0.1 of 0 or 1 to exactly 0 or 1 when it reads the file
    (``jcm.terrain.get_terrain``, ``fmask_threshold``); that rounding is
    JCM's and is not undone here -- see ``docs/source/design/exchange.md``
    for its size.

    """
    reference_file = reference_file or _data_file("terrain_JCM_T31.nc")
    landsea_mask_file = landsea_mask_file or _data_file(
        "landsea_mask_fraction_RotatedGaussianLatLon.nc")
    ocean_to_atmosphere_weights = ocean_to_atmosphere_weights or _data_file(
        "weight_algo-conserve_RotatedGaussianLatLon_to_JCM_T31.nc")

    with xr.open_dataset(ocean_to_atmosphere_weights) as weights:
        ocean_shape = tuple(int(n) for n in weights["src_grid_dims"].to_numpy())
        atmosphere_shape = tuple(
            int(n) for n in weights["dst_grid_dims"].to_numpy())
    with xr.open_dataset(reference_file) as reference:
        reference = reference.load()
    if (reference.sizes["lon"], reference.sizes["lat"]) != atmosphere_shape:
        raise ValueError(
            f"{reference_file} is on a "
            f"{(reference.sizes['lon'], reference.sizes['lat'])} grid but the "
            f"weights map onto {atmosphere_shape}."
        )

    wet = wet_mask(
        landsea_mask_file, ocean_shape[0], ocean_shape[1], threshold=threshold)
    ocean_fraction = apply_weights(ocean_to_atmosphere_weights, wet)
    excursion = max(-ocean_fraction.min(), ocean_fraction.max() - 1.0, 0.0)
    if excursion > _WEIGHT_ROUNDOFF:
        raise ValueError(
            f"The mapped sea fraction leaves [0, 1] by {excursion:.3e}, more "
            f"than the weights' round-off ({_WEIGHT_ROUNDOFF:.0e}): "
            f"{ocean_to_atmosphere_weights} does not normalise to one over "
            "each cell it covers."
        )
    ocean_fraction = np.where(
        ocean_fraction < _WEIGHT_ROUNDOFF, 0.0,
        np.where(ocean_fraction > 1.0 - _WEIGHT_ROUNDOFF, 1.0, ocean_fraction))
    land_fraction = 1.0 - ocean_fraction

    output = xr.Dataset(
        data_vars=dict(
            lsm=(("lon", "lat"), land_fraction, dict(
                long_name="land fraction, 1 - ocean fraction",
                units="1")),
            orog=(("lon", "lat"), reference["orog"].to_numpy(),
                  reference["orog"].attrs),
        ),
        coords=dict(
            lon=("lon", reference["lon"].to_numpy()),
            lat=("lat", gaussian_latitudes(atmosphere_shape[1])),
        ),
        attrs=dict(
            grid_type=reference.attrs.get("grid_type", "gaussian"),
            title="JCM T31 terrain whose land-sea mask is the Veros ocean's",
            comment=(
                "lsm = 1 - ofrac, where ofrac is Veros' binary wet mask "
                "mapped onto the atmosphere grid with conservative ESMF "
                "weights, so each atmosphere cell's sea area is exactly the "
                "wet ocean area under it. orog is the reference file's, "
                "taken by index; lat is the true Gaussian latitude."),
            source_orography=Path(reference_file).name,
            source_ocean_mask=Path(landsea_mask_file).name,
            ocean_to_atmosphere_weights=Path(ocean_to_atmosphere_weights).name,
            ocean_land_fraction_threshold=float(threshold),
            generated_by="jem.tools.veros_earth_terrain.veros_earth_terrain",
        ),
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(output_file)
    logger.info("Wrote %s", output_file)
    return output_file
