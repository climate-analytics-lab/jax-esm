from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

from jem.mapping.grid import Grid, GridSpecification
from jem.utils.domain_grid_tools import generate_coordinate_from_latitude_longitude

import coordax as cx
import dinosaur
from jax import Array
import jax.numpy as jnp
import xarray as xr

def generate_grids_from_grid_specification(
    grid_specification: str,
    mask_file: Optional[str] = None,
    topography_file: Optional[str] = None,
) -> Dict[str, Grid]:
    """
    Construct :code:`Domain` based on known :code:`GridSpecification`
    strings.
   
    Args:
        grid_specification: The string representation of grid 
            specification. 
        mask_file: The path to the mask file.
        topography_file: The path to the topography file.

    Returns:
        :code:`Domain` object.
    """

    grids = None

    parsed_grid_specification = GridSpecification.parse_grid_specification(
        grid_specification
    )
    if parsed_grid_specification.grid_universe == "JCM":
        grids = get_jcm_grids(
            horizontal_resolution=int(parsed_grid_specification.grid_family[1:]),
            mask_file=mask_file,
        )

    elif parsed_grid_specification.grid_universe == "Veros":
        grids = get_veros_grids(
            parsed_grid_specification.grid_family,
            mask_file=mask_file,
        )

    if grids is None:
        raise Exception("Error: grids is not created.")

    return grids


def generate_coordinate_from_latitude_longitude(
    latitude: List[float] | jnp.ndarray,
    longitude: List[float] | jnp.ndarray,
    order: str = "latitude_longitude",
) -> cx.Coordinate:
    axis_latitude = cx.LabeledAxis("latitude", jnp.array(latitude))
    axis_longitude = cx.LabeledAxis("longitude", jnp.array(longitude))

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


def get_jcm_grids(
    horizontal_resolution: int,
    mask_file: Optional[str] = None,
) -> Dict[str, Grid]:
    """
    Returns a CoordinateSystem object for the given number of layers and 
    horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
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

    if mask_file is None:
        print("Notice: No mask_file given. Set fmask = 0.")
        fmask = jnp.zeros(coordinate_T.shape)
        bmask = jnp.zeros(coordinate_T.shape)
    else:
        fmask, bmask = load_jcm_mask(mask_file)

    return dict(
        T=Grid(
            coordinate=coordinate_T,
            grid_type="T",
            grid_specification=GridSpecification(
                grid_universe="JCM", grid_family=grid_family
            ),
            bmask=bmask,
            fmask=fmask,
        ),
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


def get_veros_grids(
    grid_family: str,
    mask_file: Optional[str],
) -> Dict[str, Grid]:
    grids = None
    try:
        ds = xr.open_dataset(
            Path(jem.__file__).parent
            / "data"
            / "veros"
            / f"veros_{grid_family:s}.nc"
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

    if mask_file is None:
        fmask = jnp.zeros(coordinate_T.shape)
        bmask = jnp.zeros(coordinate_T.shape)
    else:
        fmask, bmask = load_veros_mask(mask_file)

    if topography_file is None:
        topography = jnp.zeros(coordinate_T.shape)
    else:
        # When veros provide its own topography file, it will be
        # topography = load_veros_topography_file(topography_file)
        pass

    return dict(
        T=Grid(
            coordinate=coordinate_T,
            grid_type="T",
            grid_specification=GridSpecification(
                grid_universe="Veros", grid_family=grid_family
            ),
            bmask=bmask,
            fmask=fmask,
        ),
    )
