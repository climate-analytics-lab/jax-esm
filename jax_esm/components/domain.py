import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re
import jax_esm
from pathlib import Path

def get_jcm_coords(horizontal_resolution) -> dinosaur.coordinate_systems.CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
    """
    try:
        horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, horizontal_resolution)
    except AttributeError:
        raise ValueError(f"Invalid horizontal grid name: {horizontal_resolution:s}. Must be one of: T21, T31, T42, T85, T106, T119, T170, T213, T340, or T425.")
    
    return dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal_grid(radius=1.0),#PHYSICS_SPECS.radius),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates([0.0, 1.0])
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
    
    result = {
        'grid_type': grid_type,
        'grid_name': grid_name,
    }

def get_veros_coords(grid_name):

    coords = dict()
    try:
        ds = Path(jax_esm.__file__).parent / "data" / "veros" / f"veros_{grid_name:s}.nc"
        coords["lon"] = ds["xt"].to_numpy()
        coords["lat"] = ds["yt"].to_numpy()
        setattr(coords, nodal_shape, (1, len(coords["lat"]), len(coords["lon"]))
            
    except Exception as e:

        import traceback
        traceback.print_exc()    

    return coords


def get_coords(grid_specification: str) -> Any:
    """
    Returns a coordinate object based on grid specification.
    """

    grid_specification = parse_grid_specification(grid_specification)
    
    if grid_specification["grid_type"] == "JCM":    

        return get_jcm_coords(grid_specification["grid_name"])

    elif grid_specification["grid_type"] == "Veros": 
        
        return get_veros_coords(grid_specification["grid_name"])
        













@dataclass
class Domain:

    fmask                 : jnp.array   # fractional mask
    bmask                 : jnp.array   # binary mask
    topography            : jnp.array
    horizontal_grid_name  : str
    coords                : any         # This is just a one layer coordinate describing horizontal lat lon
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

