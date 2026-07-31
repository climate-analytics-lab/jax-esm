from typing import Optional
import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass
import re
import coordax as cx

@dataclass
class GridSpecification:
    """
    grid_universe : Top name for classification. Such as JCM, GFDL, CESM, ... , and such.
    grid_family   : Such as T31, FV45, ..., gx1v6 and such.
    """

    grid_universe: str
    grid_family: str

    @classmethod
    def parse_grid_specification(cls, grid_specification_string: str) -> "GridSpecification":
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
                f"Invalid grid specification format: '{grid_specification_string}'. "
                f"Expected format: '<grid_universe>::<grid_family>'"
            )

        return GridSpecification(
            grid_universe=match.group(1),
            grid_family=match.group(2),
        )

    @property
    def full_name(self):
        return f"{self.grid_universe}::{self.grid_family}"

    def __str__(self):
        return self.full_name


@dataclass
class Grid:
    """
    Grid specifies the coordinate (shape the most important) and additionally area and masks.
    The binary mask `bmask` values adopt the convention that 1 means land, and 0 means ocean
    The fraction mask `fmask` means the fraction of grid area occupied by land.
    """

    coordinate: cx.Coordinate
    grid_type: Optional[str] = (
        None  # "T" for tracer grid (most common), "U" for U grid (arakawa-grid context), ... and such.
    )
    grid_specification: Optional[GridSpecification] = None
    area: Optional[Array] = None
    bmask: Optional[Array] = None
    fmask: Optional[Array] = None
    true_latitude: Optional[Array] = (
        None  # Actual geographic latitude per cell, radians. Differs from `coordinate` on rotated/curvilinear grids.
    )
    true_longitude: Optional[Array] = (
        None  # Actual geographic longitude per cell, radians. Differs from `coordinate` on rotated/curvilinear grids.
    )

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
        if self.true_latitude is not None:
            assert self.true_latitude.shape == self.shape, (
                "True latitude must match grid shape"
            )
        if self.true_longitude is not None:
            assert self.true_longitude.shape == self.shape, (
                "True longitude must match grid shape"
            )

    @property
    def full_name(self):
        grid_specification_full_name = (
            "" if self.grid_specification is None else self.grid_specification.full_name
        )
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
