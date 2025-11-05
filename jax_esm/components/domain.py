from typing import List, Tuple, Optional, Dict
import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re
import jax_esm
from pathlib import Path

@dataclass
class GridSpecification:
    grid_type : str
    grid_name : str

    def __str__(self):
        return f"{self.grid_type:s}::{self.grid_name:s}"


@dataclass
class Grid:
    nodal_shape : Tuple[int]
    axis_names  : Tuple[str]
    axis_values : Tuple[List[float]]

    @classmethod
    def from_latitude_longitude(
        cls, 
        latitude : List[float],
        longitude : List[float],
        order : str ="latitude_longitude",
    ) -> "Grid":
        
        if order == "latitude_longitude":
            return cls(
                nodal_shape = (len(latitude), len(longitude)),
                axis_names = ("latitude", "longitude"),
                axis_values = (latitude, longitude),
            )
        elif order == "longitude_latitude":
            return cls(
                nodal_shape = (len(longitude), len(latitude)),
                axis_names = ("longitude", "latitude"),
                axis_values = (longitude, latitude),
            )
        else:
            raise ValueError(f"Error: `order` has to be either `longitude_latitude` or `latitude_longitude`. User here input `{str(order):s}`")

@dataclass
class Domain:
    grid_specification    : GridSpecification 
    grids                 : Dict[str, Grid]
    fmask                 : jnp.array   # fractional mask
    bmask                 : jnp.array   # binary mask
    topography            : jnp.array
    meta                  : dict        # Component dependent information

    @classmethod
    def from_grid_specification(
        cls,
        grid_specification: str,
        mask_file : Optional[str] = None,
        topography_file : Optional[str] = None,
    ) -> "Domain":
        """
        Returns a coordinate object based on grid specification.
        """
        
        grid_specification = parse_grid_specification(grid_specification)
        if grid_specification["grid_type"] == "JCM":    
            return get_jcm_domain(
                horizontal_resolution=int(grid_specification["grid_name"][1:]),
                mask_file = mask_file,
                topography_file = topography_file,
            )

        elif grid_specification["grid_type"] == "Veros": 
            return get_veros_domain(
                grid_specification["grid_name"],
                mask_file = mask_file,
                topography_file = topography_file,
            )

def parse_grid_specification(grid_specification):
    """
    Parse a grid specification string of format "<grid_type>::<grid_name>".
    
    For grid_type == "JCM", grid_name should be "T<truncation_number>" 
    where truncation_number is an integer.
 
    For grid_type == "Veros", grid_name should be "<resolution>" 
    where resolution is a float.
    
    Args:
        grid_specification (str): String in format "<grid_type>::<grid_name>"
    
    Returns:
        dict: Dictionary with keys 'grid_type', 'grid_name', and if applicable,
              'truncation_number'
    
    Raises:
        ValueError: If the format is invalid
    """
    # Parse the basic format: <grid_type>::<grid_name>
    match = re.match(r'^([^:]+)::(.+)$', grid_specification)
    
    if not match:
        raise ValueError(f"Invalid grid specification format: '{grid_specification}'. "
                        f"Expected format: '<grid_type>::<grid_name>'")
    
    grid_type = match.group(1)
    grid_name = match.group(2)
    
    return {
        'grid_type': grid_type,
        'grid_name': grid_name,
    }


def load_jcm_mask(mask_file):
    
    ds = xr.open_dataset(mask_file, engine="netcdf4")

    # land-sea mask
    fmask = jnp.asarray(ds["lsm"])
    
    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"

    # It is land (mask = 1) only if fmask == 1
    # If there is a bit of water ( fmask < 1 ), then bmask = 0
    bmask = jnp.where(fmask == 1, 1.0, 0.0)
   
    return fmask, bmask 


def load_jcm_topography_file(
    topography_file : str,
):
    return jnp.asarray(xr.open_dataset(topography_file, engine="netcdf4")["orog"])


    return None

def get_jcm_domain(
    horizontal_resolution,
    mask_file: Optional[str],
    topography_file: Optional[str],
) -> Domain:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
    """

    grid_name = f"T{horizontal_resolution:d}"

    try:
        horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f"T{horizontal_resolution:d}")
    except AttributeError:
        raise ValueError(f"Invalid horizontal grid name: {horizontal_resolution:s}. Must be one of: T21, T31, T42, T85, T106, T119, T170, T213, T340, or T425.")
 
    meta = dict(
        one_layer_coords = dinosaur.coordinate_systems.CoordinateSystem(
            horizontal=horizontal_grid(radius=1.0),#PHYSICS_SPECS.radius),
            vertical=dinosaur.sigma_coordinates.SigmaCoordinates([0.0, 1.0])
        ),
        horizontal_resolution=horizontal_resolution,
    )

    hgrid = meta["one_layer_coords"].horizontal 
    grids = dict(
        T = Grid.from_latitude_longitude(
            latitude = hgrid.latitudes,
            longitude = hgrid.longitudes,
            order = "longitude_latitude",
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
        grid_specification = GridSpecification(grid_type = "JCM", grid_name = grid_name),
        grids = grids,
        fmask = fmask,
        bmask = bmask,
        topography = topography,
        meta = meta,
    )

def load_veros_mask(mask_file):
    
    ds = xr.open_dataset(mask_file, engine="netcdf4")
    surface_grid_idx = int(jnp.argmax(ds["zt"].to_numpy()))
    proxy = jnp.array(ds["temp"].isel(zt=surface_grid_idx, Time=0).to_numpy())

    # land-sea mask
    fmask = jnp.where(jnp.isnan(proxy), 1.0, 0.0)
    
    # Apply some sanity checks -- might want to check this shape against the model shape?
    assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"

    # It is land (mask = 1) only if fmask == 1
    # If there is a bit of water ( fmask < 1 ), then bmask = 0
    bmask = jnp.where(fmask == 1.0, 1.0, 0.0)
   
    return fmask, bmask 


def get_veros_domain(
    grid_name: str,
    mask_file: Optional[str],
    topography_file: Optional[str],
) -> Domain:


    grids = None
    try:

        ds = xr.open_dataset(Path(jax_esm.__file__).parent / "data" / "veros" / f"veros_{grid_name:s}.nc")
        longitude = ds["xt"].to_numpy() * jnp.pi / 180.0
        latitude = ds["yt"].to_numpy() * jnp.pi / 180.0

        grids = dict(
            T = Grid.from_latitude_longitude(
                latitude = latitude,
                longitude = longitude,
                order = "latitude_longitude",
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
        topography = load_veros_topography_file(topography_file)


    return Domain(
        grid_specification = GridSpecification(grid_type = "Veros", grid_name = grid_name),
        grids = grids,
        fmask = fmask,
        bmask = bmask,
        topography = topography,
        meta = dict(),
    )


