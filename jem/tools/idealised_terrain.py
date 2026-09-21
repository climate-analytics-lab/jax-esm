"""Generate a JCM terrain file whose geography is an idealised shape.

A real terrain file is ERA5-derived (see the packaged
``terrain_JCM_T31.nc``): its land-sea mask is the actual coastline, at
whatever resolution and grid it was regridded to. An idealised experiment
(an aquaplanet, a two-continent "double-drake" ocean, ...) wants a
land-sea mask that is a simple, reproducible *geometric shape* on the same
grid instead, at whatever resolution the atmosphere happens to run at --
which is not a file worth shipping once per shape per resolution, but is
cheap to *generate* from a reference terrain file (for its ``lon``/``lat``
axes and grid metadata) whenever it is needed.

:func:`idealised_terrain` is that generator. Its ``reference_file`` must be
JCM-canonical (:func:`jcm.terrain.TerrainData.from_file`'s layout: dims
``(lon, lat)``, data variables ``lsm`` and ``orog``, e.g. the packaged
``jem/data/terrain_JCM_T31.nc``) -- it supplies the axes and the
``grid_type`` attribute the output copies, and, for ``"capped_earth"``, its
own land mask. The resulting ``orog`` is always zero: an idealised planet
has no real orography, only a land-sea mask.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr

#: Latitude poleward of which every idealised geography is land -- a polar
#: ice cap, whatever the ocean basins below it look like.
CAP_LATITUDE = 75.0


@dataclasses.dataclass(frozen=True)
class _Basin:
    """One meridional land divider: land from `lat_edge` to the pole, in a
    longitude band `width` degrees wide starting at `lon_beg`.
    """

    lon_beg: float
    lat_edge: float
    width: float


@dataclasses.dataclass(frozen=True)
class _Geography:
    """The land-mask geometry for one ``planet_type``.

    ``include_reference_land`` additionally ORs in ``reference_file``'s own
    land mask (``"capped_earth"``: real continents, plus a capped ice edge).
    ``basins`` are meridional land dividers on top of the polar cap.
    """

    include_reference_land: bool = False
    basins: tuple[_Basin, ...] = ()


#: One entry per supported `planet_type`, with every geometry constant
#: `modify_jcm_terrain.py` (the driver this replaces) hard-coded, unchanged.
_GEOGRAPHIES: dict[str, _Geography] = {
    "aquaplanet": _Geography(),
    "capped_earth": _Geography(include_reference_land=True),
    "toy_earth": _Geography(basins=(
        _Basin(lon_beg=0.0, lat_edge=-10.0, width=3.75),
        _Basin(lon_beg=120.0, lat_edge=-50.0, width=3.75),
    )),
    "double_drake": _Geography(basins=(
        _Basin(lon_beg=0.0, lat_edge=-45.0, width=3.85),
        _Basin(lon_beg=67.5, lat_edge=-45.0, width=3.85),
    )),
    "double_drake_equal_basin_width": _Geography(basins=(
        _Basin(lon_beg=0.0, lat_edge=-45.0, width=3.85),
        _Basin(lon_beg=180.0, lat_edge=-45.0, width=3.85),
    )),
}


def idealised_terrain(
    reference_file: str | Path, planet_type: str, output_file: str | Path
) -> Path:
    """Write a JCM terrain file whose land-sea mask is an idealised geography.

    Parameters
    ----------
    reference_file : str or pathlib.Path
        A JCM-canonical terrain file (dims ``(lon, lat)``, vars ``lsm``/
        ``orog``, e.g. ``jem/data/terrain_JCM_T31.nc``) supplying the ``lon``/
        ``lat`` axes, the ``grid_type`` attribute, and -- for
        ``planet_type="capped_earth"`` only -- its own land mask.
    planet_type : str
        One of ``"aquaplanet"``, ``"capped_earth"``, ``"toy_earth"``,
        ``"double_drake"``, ``"double_drake_equal_basin_width"``.
    output_file : str or pathlib.Path
        Full path to write to; its parent directory is created if needed.

    Returns
    -------
    pathlib.Path
        ``output_file``.

    Raises
    ------
    ValueError
        If ``planet_type`` is not one of the geographies above; the message
        names the value given and the valid ones.

    """
    geography = _GEOGRAPHIES.get(planet_type)
    if geography is None:
        raise ValueError(
            f"Unknown planet_type {planet_type!r}; expected one of "
            f"{sorted(_GEOGRAPHIES)!r}."
        )

    reference = xr.open_dataset(reference_file)
    lon = reference["lon"].to_numpy()
    lat = reference["lat"].to_numpy()
    # Longitudes are wrapped to [0, 360) for the basin geometry below only --
    # a basin's bounds are given in that convention -- the output keeps the
    # reference file's own (unwrapped) coordinate values.
    mesh_lon, mesh_lat = np.meshgrid(lon % 360, lat, indexing="ij")

    is_land = np.abs(mesh_lat) >= CAP_LATITUDE
    if geography.include_reference_land:
        is_land = is_land | (reference["lsm"].to_numpy() == 1)
    for basin in geography.basins:
        in_basin = (
            (mesh_lat >= basin.lat_edge)
            & (mesh_lon >= basin.lon_beg)
            & (mesh_lon < basin.lon_beg + basin.width)
        )
        is_land = is_land | in_basin
    mask = is_land.astype(np.float64)

    output = xr.Dataset(
        data_vars=dict(
            lsm=(("lon", "lat"), mask),
            orog=(("lon", "lat"), np.zeros_like(mask)),
        ),
        coords=dict(lon=("lon", lon), lat=("lat", lat)),
        attrs=dict(grid_type=reference.attrs.get("grid_type", "gaussian")),
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(output_file)
    return output_file
