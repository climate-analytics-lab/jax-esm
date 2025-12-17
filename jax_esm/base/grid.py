from typing import List, Tuple, Optional, Dict
from jax import Array
import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re
import jax_esm
from pathlib import Path

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
    grid_type: Optional[str] = None # "T" for tracer grid (most common), "U" for U grid (arakawa-grid context), ... and such.
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

    @property
    def shape(self):
        return self.coordinate.shape

    @property
    def dimension_names(self):
        return self.coordinate.dims



