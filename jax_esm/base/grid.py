from typing import List, Tuple, Optional, Dict
from jax import Array
import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re
import jax_esm
from pathlib import Path

class GridType(str):
    """
        "T" for tracer grid (most common), "U" for U grid (arakawa-grid context), ... and such.
    """
    pass

@dataclass
class GridSpecification:
    """
        grid_universe : Top name for classification. Such as JCM, GFDL, CESM, ... , and such.
        grid_family   : Such as T31, FV45, ..., gx1v6 and such.
    """
    grid_family    : str
    grid_universe  : str

    def parse_grid_specification(grid_specification_string: str) -> Dict[str, str]:
        """
        Parse a grid specification string of format "<grid_universe>::<grid_family>".

        For grid_universe == "JCM", grid_family should be "T<truncation_number>"
        where truncation_number is an integer.

        For grid_universe == "Veros", grid_family should be "<resolution>"
        where resolution is a float.

        Args:
            grid_specification (str): String in format "<grid_universe>::<grid_family>"

        Returns:
            dict: Dictionary with keys 'grid_universe', 'grid_family', and if applicable,
                  'truncation_number'

        Raises:
            ValueError: If the format is invalid
        """
        # Parse the basic format: <grid_universe>::<grid_family>
        match = re.match(r"^([^:]+)::(.+)$", grid_specification_string)

        if not match:
            raise ValueError(
                f"Invalid grid specification format: '{grid_specification}'. "
                f"Expected format: '<grid_universe>::<grid_family>'"
            )

        return GridSpecification(
            grid_universe = match.group(1),
            grid_family = match.group(2),
        )


    @property
    def full_name(self):
        return f"{self.grid_universe}::{self.grid_family}"

    def __str__(self):
        return self.full_name

@dataclass
class Grid:
    """
        Grid specifies the coordinate (shape the most important) and additionally weights and masks.
    """

    coordinate: cx.Coordinate
    grid_type: Optional[GridType] = None
    grid_specification: Optional[GridSpecification] = None
    weights: Optional[Array] = None
    bmask: Optional[Array] = None
    fmask: Optional[Array] = None
 
    def __post_init__(self):
        if self.grid_weights is not None:
            assert self.grid_weights.shape == self.shape, \
                "Area weights must match grid shape"
        if self.bmask is not None:
            assert self.bmask.shape == self.shape, \
                "Binary mask must match grid shape"
        if self.fmask is not None:
            assert self.fmask.shape == self.shape, \
                "Fractional mask must match grid shape"

    @property
    def full_name(self):
        grid_specification_full_name = "" if self.grid_specification is None else self.grid_specification.full_name
        return f"{grid_specification_full_name}{self.grid_type}"


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
    grids = dict(
        T=Grid.from_latitude_longitude(
            latitude=hgrid.latitudes,
            longitude=hgrid.longitudes,
            order="longitude_latitude",
        )
    )

    if mask_file is None:
        fmask = jnp.zeros(grids["T"].nodal_shape)
        bmask = jnp.zeros(grids["T"].nodal_shape)
    else:
        fmask, bmask = load_jcm_mask(mask_file)

    if topography_file is None:
        topography = jnp.zeros(grids["T"].nodal_shape)
    else:
        topography = load_jcm_topography_file(topography_file)

    return Domain(
        grid_specification=GridSpecification(grid_universe="JCM", grid_family=grid_family),
        grids=grids,
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

    if mask_file is None:
        fmask = jnp.zeros(grids["T"].nodal_shape)
        bmask = jnp.zeros(grids["T"].nodal_shape)
    else:
        fmask, bmask = load_veros_mask(mask_file)

    if topography_file is None:
        topography = jnp.zeros(grids["T"].nodal_shape)
    else:
        pass
        # When veros provide its own topography file, it will be
        # topography = load_veros_topography_file(topography_file)

    return Domain(
        grid_specification=GridSpecification(grid_universe="Veros", grid_family=grid_family),
        grids=grids,
        fmask=fmask,
        bmask=bmask,
        topography=topography,
    )
