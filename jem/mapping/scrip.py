from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
import xarray as xr

DEG_TO_RAD = jnp.pi / 180.0


@dataclass
class ScripGridData:
    """
    Generic contents of a SCRIP-style grid/mask netCDF file.

    Conventions (radians unless noted):
        longitude, latitude: 1D logical axes of the (possibly rotated) grid, shape (ni,), (nj,).
        true_longitude, true_latitude: actual geographic location of each cell, shape (nj, ni).
            Equal to a broadcast of longitude/latitude when the grid is not rotated.
        fmask: fraction of each cell's area occupied by land, shape (nj, ni).
        bmask: binary land (1) / sea (0) mask, shape (nj, ni).
    """

    longitude: Array
    latitude: Array
    true_longitude: Array
    true_latitude: Array
    fmask: Array
    bmask: Array


def read_scrip_grid_file(
    scrip_file: str,
    longitude_name: str = "i",
    latitude_name: str = "j",
    true_longitude_name: str = "true_lon",
    true_latitude_name: str = "true_lat",
    fmask_name: str = "land_fraction",
    bmask_name: str = "land_sea_mask",
) -> ScripGridData:
    """
    Read a SCRIP-style grid/mask netCDF file.

    The file is expected to describe a logically rectangular (j, i) grid via 1D
    logical longitude/latitude axes, a 2D true (geographic) longitude/latitude
    per cell, and a fractional and binary land-sea mask. Variable names can be
    overridden to accommodate different SCRIP-style naming conventions.

    Args:
        scrip_file: Path to the SCRIP-style netCDF file.
        longitude_name: Name of the 1D logical longitude variable.
        latitude_name: Name of the 1D logical latitude variable.
        true_longitude_name: Name of the 2D geographic longitude variable.
        true_latitude_name: Name of the 2D geographic latitude variable.
        fmask_name: Name of the fractional land mask variable.
        bmask_name: Name of the binary land-sea mask variable.

    Returns:
        :code:`ScripGridData` with all angles in radians.
    """

    ds = xr.open_dataset(scrip_file, engine="netcdf4")

    longitude = jnp.asarray(ds[longitude_name]) * DEG_TO_RAD
    latitude = jnp.asarray(ds[latitude_name]) * DEG_TO_RAD
    true_longitude = jnp.asarray(ds[true_longitude_name]) * DEG_TO_RAD
    true_latitude = jnp.asarray(ds[true_latitude_name]) * DEG_TO_RAD

    fmask = jnp.asarray(ds[fmask_name])
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), (
        "Land fraction mask must be between 0 and 1"
    )

    bmask = jnp.asarray(ds[bmask_name])
    assert jnp.all((bmask == 0.0) | (bmask == 1.0)), (
        "Binary land-sea mask must be 0 or 1"
    )
    bmask = bmask.astype(fmask.dtype)

    return ScripGridData(
        longitude=longitude,
        latitude=latitude,
        true_longitude=true_longitude,
        true_latitude=true_latitude,
        fmask=fmask,
        bmask=bmask,
    )
