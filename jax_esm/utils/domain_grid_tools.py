from typing import List, Tuple, Optional, Dict
from jax import Array
import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re
import jax_esm
from pathlib import Path

from jax_esm import GridSpecification

import coordax as cx

def generate_coordinate_from_latitude_longitude(
    cls,
    latitude: List[float] | jnp.ndarray,
    longitude: List[float] | jnp.ndarray,
    order: str = "latitude_longitude",
) -> cx.Coordinate:
    
    axis_latitude = cx.LabeledAxis('latitude', jnp.array(latitude))
    axis_longitude = cx.LabeledAxis('longitude', jnp.array(longitude))

    args = None

    if order == "latitude_longitude":
        args = (axis_latitude, axis_longitude)

    elif order == "longitude_latitude":
        args = (axis_longitude, axis_latitude)
    
    else:
        raise ValueError(
            f"Error: `order` has to be either `longitude_latitude` or `latitude_longitude`. User here input `{str(order):s}`"
        )
    
    return cx.coords.compose(*args)



def load_jcm_mask(mask_file):
    ds = xr.open_dataset(mask_file, engine="netcdf4")

    # land-sea mask
    fmask = jnp.asarray(ds["lsm"])

    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), (
        "Land-sea mask must be between 0 and 1"
    )

    # It is land (mask = 1) only if fmask == 1
    # If there is a bit of water ( fmask < 1 ), then bmask = 0
    bmask = jnp.where(fmask > 0.95, 1.0, 0.0)

    return fmask, bmask

def load_jcm_topography_file(
    topography_file: str,
):
    return jnp.asarray(xr.open_dataset(topography_file, engine="netcdf4")["orog"])


def get_jcm_domain(
    horizontal_resolution: int,
    mask_file: Optional[str] = None,
    topography_file: Optional[str] = None,
) -> Domain:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
    """

    grid_family = f"T{horizontal_resolution:d}"

    try:
        horizontal_grid = getattr(
            dinosaur.spherical_harmonic.Grid, f"T{horizontal_resolution:d}"
        )
    except AttributeError:
        raise ValueError(
            f"Invalid horizontal grid name: {horizontal_resolution:s}. Must be one of: T21, T31, T42, T85, T106, T119, T170, T213, T340, or T425."
        )

    one_layer_coords = dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal_grid(radius=1.0),  # PHYSICS_SPECS.radius),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates([0.0, 1.0]),
    )

    hgrid = one_layer_coords.horizontal
    
    coordinate_T = generate_coordinate_from_latitude_longitude(
        latitude=hgrid.latitudes,
        longitude=hgrid.longitudes,
        order="longitude_latitude",
    )

    nodal_shape = ( len(_shape) for _shape in coordinate_T.shapes )

    if mask_file is None:
        fmask = jnp.ones(nodal_shape)
        bmask = jnp.ones(nodal_shape)
    else:
        fmask, bmask = load_jcm_mask(mask_file)

    if topography_file is None:
        topography = jnp.zeros(nodal_shape)
    else:
        topography = load_jcm_topography_file(topography_file)

    return Domain(
        grids = dict(
            T = Grid(
                coordinate = coordinate_T,
                grid_type = "T",
                grid_specification=GridSpecification(grid_universe="JCM", grid_family=grid_family),
                bmask = bmask,
                fmask = fmask,
            ),
        ),
        topography=topography,
    )


def load_veros_mask(mask_file):
    ds = xr.open_dataset(mask_file, engine="netcdf4")
    surface_grid_idx = int(jnp.argmax(ds["zt"].to_numpy()))
    proxy = jnp.array(ds["temp"].isel(zt=surface_grid_idx, Time=0).to_numpy())

    # land-sea mask
    fmask = jnp.where(jnp.isnan(proxy), 1.0, 0.0)

    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), (
        "Land-sea mask must be between 0 and 1"
    )

    # It is land (mask = 1) only if fmask == 1
    # If there is a bit of water ( fmask < 1 ), then bmask = 0
    bmask = jnp.where(fmask == 1.0, 1.0, 0.0)

    return fmask, bmask


def get_veros_domain(
    grid_family: str,
    mask_file: Optional[str],
    topography_file: Optional[str],
) -> Domain:
    grids = None
    try:
        ds = xr.open_dataset(
            Path(jax_esm.__file__).parent / "data" / "veros" / f"veros_{grid_family:s}.nc"
        )
        longitude = jnp.array(ds["xt"]) * jnp.pi / 180.0
        latitude = jnp.array(ds["yt"]) * jnp.pi / 180.0

        grids = dict(
            T=Grid.from_latitude_longitude(
                latitude=latitude,
                longitude=longitude,
                order="latitude_longitude",
            )
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e

    coordinate_T = generate_coordinate_from_latitude_longitude(
        latitude=hgrid.latitudes,
        longitude=hgrid.longitudes,
        order="longitude_latitude",
    )

    nodal_shape = ( len(_shape) for _shape in coordinate_T.shapes )

    if mask_file is None:
        fmask = jnp.zeros(nodal_shape)
        bmask = jnp.zeros(nodal_shape)
    else:
        fmask, bmask = load_veros_mask(mask_file)

    if topography_file is None:
        topography = jnp.zeros(nodal_shape)
    else:
        # When veros provide its own topography file, it will be
        # topography = load_veros_topography_file(topography_file)
        pass

    return Domain(
        grids = dict(
            T = Grid(
                coordinate = coordinate_T,
                grid_type = "T",
                grid_specification=GridSpecification(grid_universe="Veros", grid_family=grid_family),
                bmask = bmask,
                fmask = fmask,
            ),
        ),
        topography=topography,
    )
