import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass

fmask_threshold = 0.5


def get_coords(horizontal_resolution) -> dinosaur.coordinate_systems.CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
    """
    try:
        horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f"T{horizontal_resolution:d}")
    except AttributeError:
        raise ValueError(f"Invalid horizontal resolution: {horizontal_resolution}. Must be one of: 21, 31, 42, 85, 106, 119, 170, 213, 340, or 425.")
    
    return dinosaur.coordinate_systems.CoordinateSystem(
        horizontal=horizontal_grid(radius=1.0),#PHYSICS_SPECS.radius),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates([0.0, 1.0])
    )

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
        
        # Set values close to 0 or 1 to exactly 0 or 1
        bmask = jnp.where(fmask <= fmask_threshold, 0.0, jnp.where(fmask >= 1.0 - fmask_threshold, 1.0, fmask))

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
        
        # Set values close to 0 or 1 to exactly 0 or 1
        bmask = jnp.where(fmask <= fmask_threshold, 0.0, jnp.where(fmask >= 1.0 - fmask_threshold, 1.0, fmask))

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

