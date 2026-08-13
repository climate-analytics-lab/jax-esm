from dataclasses import dataclass

import coordax as cx
import jax.numpy as jnp
from jax import Array


@dataclass
class Grid:
    """
    Grid specifies the coordinate (shape the most important) and additionally area and masks.
    The binary mask `bmask` values adopt the convention that 1 means land, and 0 means ocean
    The fraction mask `fmask` means the fraction of grid area occupied by land.
    """

    coordinate: cx.Coordinate
    grid_type: str | None = (
        None  # "T" for tracer grid (most common), "U" for U grid (arakawa-grid context), ... and such.
    )
    grid_specification: str | None = None
    area: Array | None = None
    bmask: Array | None = None
    fmask: Array | None = None

    def __post_init__(self):
        if self.area is not None:
            assert self.area.shape == self.shape, (
                "Area must match grid shape"
            )
        if self.bmask is not None:
            assert self.bmask.shape == self.shape, "Binary mask must match grid shape"
        if self.fmask is not None:
            assert self.fmask.shape == self.shape, (
                "Fractional mask must match grid shape"
            )

    @property
    def full_name(self):
        grid_specification_full_name = self.grid_specification or ""
        return f"{grid_specification_full_name}{self.grid_type}"

    @property
    def shape(self):
        return self.coordinate.shape

    @property
    def dimension_names(self):
        return self.coordinate.dims

    def get_info(self):
        
        bmask_info = None
        if self.bmask is not None:
            count_ones  = jnp.sum(self.bmask == 1)
            count_zeros = jnp.sum(self.bmask == 0)
            total = count_zeros + count_ones
            bmask_info = {
                'shape' : str(self.bmask.shape),
                'count 1' : f"{count_ones:d} / {total:d} ({100*count_ones/total:.1f}%)",
                'count 0' : f"{count_zeros:d} / {total:d} ({100*count_zeros/total:.1f}%)",
            }
 
        area_info = None
        if self.area is not None:
            area_info = {
                'shape' : str(self.area.shape),
                'sum of area' : f"{jnp.sum(self.area):f}",
            }
        
        return {
            'grid_specification' : str(self.grid_specification),
            'bmask' : bmask_info,
            'area' : area_info,
        }
