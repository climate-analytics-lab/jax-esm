from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
import xarray as xr

DEG_TO_RAD = jnp.pi / 180.0


@dataclass
class CurvilinearGridData:
    """
    Generic contents of a curvilinear grid/mask netCDF file (e.g. rotated-pole,
    displaced-pole): a logically rectangular (i, j) grid whose true geographic
    location differs from its logical index space.

    Conventions (radians unless noted):
        longitude, latitude: 1D logical axes of the grid, shape (ni,), (nj,).
        true_longitude, true_latitude: actual geographic location of each cell, shape (ni, nj).
            Equal to a broadcast of longitude/latitude when the grid is not distorted.
        fmask: fraction of each cell's area occupied by land, shape (ni, nj).
        bmask: binary land (1) / sea (0) mask, shape (ni, nj).
    """

    longitude: Array
    latitude: Array
    true_longitude: Array
    true_latitude: Array
    fmask: Array
    bmask: Array


def read_curvilinear_grid_file(
    curvilinear_grid_file: str,
    longitude_name: str = "i",
    latitude_name: str = "j",
    true_longitude_name: str = "true_lon",
    true_latitude_name: str = "true_lat",
    fmask_name: str = "land_fraction",
    bmask_name: str = "land_sea_mask",
) -> CurvilinearGridData:
    """
    Read a curvilinear grid/mask netCDF file (e.g. rotated-pole, displaced-pole).

    The file is expected to describe a logically rectangular (i, j) grid via 1D
    logical longitude/latitude axes, a 2D true (geographic) longitude/latitude
    per cell, and a fractional and binary land-sea mask. Variable names can be
    overridden to accommodate different curvilinear-grid naming conventions.

    Args:
        curvilinear_grid_file: Path to the curvilinear grid netCDF file.
        longitude_name: Name of the 1D logical longitude variable.
        latitude_name: Name of the 1D logical latitude variable.
        true_longitude_name: Name of the 2D geographic longitude variable.
        true_latitude_name: Name of the 2D geographic latitude variable.
        fmask_name: Name of the fractional land mask variable.
        bmask_name: Name of the binary land-sea mask variable.

    Returns:
        :code:`CurvilinearGridData` with all angles in radians.
    """

    ds = xr.open_dataset(curvilinear_grid_file, engine="netcdf4")

    # 2D fields may be stored as (longitude_name, latitude_name) or
    # (latitude_name, longitude_name) on disk -- pin down (lon, lat) order here
    # so downstream shape (ni, nj) doesn't depend on file layout.
    def as_lon_lat(name):
        return jnp.asarray(ds[name].transpose(longitude_name, latitude_name))

    longitude = jnp.asarray(ds[longitude_name]) * DEG_TO_RAD
    latitude = jnp.asarray(ds[latitude_name]) * DEG_TO_RAD
    true_longitude = as_lon_lat(true_longitude_name) * DEG_TO_RAD
    true_latitude = as_lon_lat(true_latitude_name) * DEG_TO_RAD

    fmask = as_lon_lat(fmask_name)
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), (
        "Land fraction mask must be between 0 and 1"
    )

    bmask = as_lon_lat(bmask_name)
    assert jnp.all((bmask == 0.0) | (bmask == 1.0)), (
        "Binary land-sea mask must be 0 or 1"
    )
    bmask = bmask.astype(fmask.dtype)

    return CurvilinearGridData(
        longitude=longitude,
        latitude=latitude,
        true_longitude=true_longitude,
        true_latitude=true_latitude,
        fmask=fmask,
        bmask=bmask,
    )
