from typing import List, Tuple, Optional, Dict
from jax import Array
import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re

from jax_esm.base.grid import Grid

from pathlib import Path

@dataclass
class Domain:
    """
        Domain is a collection of horizontal_grids plus other meta data such as topography
    """
    horizontal_grids: Dict[str, Grid]
    topography: Dict[str, Array | None]

    @classmethod
    def from_grid_specification(
        cls,
        grid_specification: str,
        mask_file: Optional[str] = None,
        topography_file: Optional[str] = None,
    ) -> "Domain":
        """
        Returns a coordinate object based on grid specification.
        """

        d = None

        parsed_grid_specification = parse_grid_specification(grid_specification)
        if parsed_grid_specification["root_name"] == "JCM":
            d = get_jcm_domain(
                horizontal_resolution=int(parsed_grid_specification["grid_family"][1:]),
                mask_file=mask_file,
                topography_file=topography_file,
            )

        elif parsed_grid_specification["root_name"] == "Veros":
            d = get_veros_domain(
                parsed_grid_specification["grid_family"],
                mask_file=mask_file,
                topography_file=topography_file,
            )

        if d is None:
            raise Exception("Error: domain is not created.")

        return d


def parse_grid_specification(grid_specification: str) -> Dict[str, str]:
    """
    Parse a grid specification string of format "<root_name>::<grid_family>".

    For root_name == "JCM", grid_family should be "T<truncation_number>"
    where truncation_number is an integer.

    For root_name == "Veros", grid_family should be "<resolution>"
    where resolution is a float.

    Args:
        grid_specification (str): String in format "<root_name>::<grid_family>"

    Returns:
        dict: Dictionary with keys 'root_name', 'grid_family', and if applicable,
              'truncation_number'

    Raises:
        ValueError: If the format is invalid
    """
    # Parse the basic format: <root_name>::<grid_family>
    match = re.match(r"^([^:]+)::(.+)$", grid_specification)

    if not match:
        raise ValueError(
            f"Invalid grid specification format: '{grid_specification}'. "
            f"Expected format: '<root_name>::<grid_family>'"
        )

    root_name = match.group(1)
    grid_family = match.group(2)

    return {
        "root_name": root_name,
        "grid_family": grid_family,
    }


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

    return None


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
    horizontal_grids = dict(
        T=Grid.from_latitude_longitude(
            latitude=hgrid.latitudes,
            longitude=hgrid.longitudes,
            order="longitude_latitude",
        )
    )

    if mask_file is None:
        fmask = jnp.zeros(horizontal_grids["T"].nodal_shape)
        bmask = jnp.zeros(horizontal_grids["T"].nodal_shape)
    else:
        fmask, bmask = load_jcm_mask(mask_file)

    if topography_file is None:
        topography = jnp.zeros(horizontal_grids["T"].nodal_shape)
    else:
        topography = load_jcm_topography_file(topography_file)

    return Domain(
        grid_specification=GridSpecification(root_name="JCM", grid_family=grid_family),
        horizontal_grids=horizontal_grids,
        fmask=fmask,
        bmask=bmask,
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
    horizontal_grids = None
    try:
        ds = xr.open_dataset(
            Path(jax_esm.__file__).parent / "data" / "veros" / f"veros_{grid_family:s}.nc"
        )
        longitude = jnp.array(ds["xt"]) * jnp.pi / 180.0
        latitude = jnp.array(ds["yt"]) * jnp.pi / 180.0

        horizontal_grids = dict(
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

    if mask_file is None:
        fmask = jnp.zeros(horizontal_grids["T"].nodal_shape)
        bmask = jnp.zeros(horizontal_grids["T"].nodal_shape)
    else:
        fmask, bmask = load_veros_mask(mask_file)

    if topography_file is None:
        topography = jnp.zeros(horizontal_grids["T"].nodal_shape)
    else:
        pass
        # When veros provide its own topography file, it will be
        # topography = load_veros_topography_file(topography_file)

    return Domain(
        grid_specification=GridSpecification(root_name="Veros", grid_family=grid_family),
        horizontal_grids=horizontal_grids,
        fmask=fmask,
        bmask=bmask,
        topography=topography,
    )
