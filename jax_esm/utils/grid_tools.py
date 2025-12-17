from typing import List, Tuple, Optional, Dict
from jax import Array
import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re
import jax_esm
from pathlib import Path

from jax_esm import GridSpecification, GridType



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
    
    coordinate_T = coordinate_from_latitude_longitude(
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

    grid_T = Grid(
        coordinate = coordinate_T,
        grid_type = "T",
        grid_specification=GridSpecification(grid_universe="JCM", grid_family=grid_family),
        bmask = bmask,
        fmask = fmask,

    )

    return Domain(
        grids = {GridType("T"): grid_T},
        topography=topography,
    )
