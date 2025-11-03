from typing import List, Tuple, Optional, Self, Dict
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

@dataclass
class Grid:
    nodal_shape : Tuple[int]
    axis_names  : Tuple[str]
    axis_values : Tuple[List[float]]

    @classmethod
    def from_latitude_longitude(
        cls, 
        lat : List[float],
        lon : List[float],
        order : str ="lat_lon",
    ) -> Self:
        
        if order == "lat_lon":
            return cls(
                nodal_shape = (len(lat), len(lon)),
                axis_names = ("lat", "lon"),
                axis_values = (lat, lon),
            )
        elif order == "lon_lat":
            return cls(
                nodal_shape = (len(lon), len(lat)),
                axis_names = ("lon", "lat"),
                axis_values = (lon, lat),
            )
        else:
            raise ValueError(f"Error: `order` has to be either `lon_lat` or `lat_lon`. User here input `{str(order):s}`")

@dataclass
class Domain:
    grid_specification    : GridSpecification 
    grids                 : Dict[str, Grid]
    fmask                 : jnp.array   # fractional mask
    bmask                 : jnp.array   # binary mask
    topography            : jnp.array
    meta                  : dict        # Component dependent information


    @classmethod
    def from_file_and_resolution(cls, filename, horizontal_resolution: int):
        
        coords = get_coords(horizontal_resolution)

        ds = xr.open_dataset(filename, engine="netcdf4")

        # land-sea mask
        fmask = jnp.asarray(ds["lsm"])
        topography = jnp.asarray(ds["orog"])
        
        # Apply some sanity checks -- might want to check this shape against the model shape?
        assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"

        # It is land (mask = 1) only if fmask == 1
        # If there is a bit of water ( fmask < 1 ), then bmask = 0
        bmask = jnp.where(fmask == 1, 1.0, 0.0)

        for shape in [ fmask.shape, topography.shape ]:
            if coords.horizontal.nodal_shape != shape:
                raise Exception("The shape from file and coords must be identical.")
        
        return Domain(
            coords = coords,
            fmask = fmask,
            bmask = bmask,
            topography = topography,
            meta = dict(horizontal_resolution=horizontal_resolution),
            horizontal_grid_name = f"T{horizontal_resolution:d}",
        )

    @classmethod
    def from_resolution_all_ocean(cls, horizontal_resolution: int):
        
        coords = get_coords(horizontal_resolution)

        fmask = jnp.zeros(coords.horizontal.nodal_shape)
        topography = jnp.zeros_like(fmask) - 4000.0
        
        # Apply some sanity checks -- might want to check this shape against the model shape?
        assert jnp.all((0.0 <= fmask) & (fmask <= 1.0)), "Land-sea mask must be between 0 and 1"
        
        bmask = jnp.where(fmask < 1.0, 0.0, 1.0)

        for shape in [ fmask.shape, topography.shape ]:
            if coords.horizontal.nodal_shape != shape:
                raise Exception("The shape from file and coords must be identical.")
        
        return Domain(
            coords = coords,
            fmask = fmask,
            bmask = bmask,
            topography = topography,
            meta = dict(horizontal_resolution=horizontal_resolution),
            horizontal_grid_name = f"T{horizontal_resolution:d}",
        )


    @classmethod
    def from_grid_specification(
        cls,
        grid_specification: str,
        mask_file : Optional[str] = None,
        topography_file : Optional[str] = None,
    ) -> Self:
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
            lat = hgrid.longitudes,
            lon = hgrid.latitudes,
            order = "lat_lon",
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
        grid_specification = "JCM::T{horizontal_resolution:s}",
        grids = grids,
        fmask = fmask,
        bmask = bmask,
        topography = topography,
        meta = meta,
    )

def get_veros_coords(
    grid_name: str,
    mask_file: Optional[str],
    topography_file: Optional[str],
) -> Domain:

    try:

        ds = Path(jax_esm.__file__).parent / "data" / "veros" / f"veros_{grid_name:s}.nc"
        lon = ds["xt"].to_numpy()
        lat = ds["yt"].to_numpy()
            
    except Exception as e:

        import traceback
        traceback.print_exc()    

    if mask_file is None:
        fmask = jnp.zeros(grid.nodal_shape["T"])
        bmask = jnp.zeros(grid.nodal_shape["T"])
    else:
        fmask, bmask = load_veros_mask(mask_file)
 
    if topography_file is None:
        topography = jnp.zeros(grid.nodal_shape["T"])
    else:
        topography = load_veros_topography_file(topography_file)


    return Domain(
        grid = dict(
            T = Grid.from_latitude_longitude(
                lat = lat,
                lon = lon,
                order = "lat_lon",
            )
        ),
        fmask = fmask,
        bmask = bmask,
        topography = topography,
    )


