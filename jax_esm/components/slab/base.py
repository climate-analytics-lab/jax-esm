"""Base class for slab models (ocean, land, atmosphere).

This module provides a common base class that extracts shared functionality
from SlabOceanModel, SlabLandModel, and SlabAtmosphereModel to reduce
code duplication.
"""

from abc import abstractmethod
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr

from jax_esm.components.base import Component, CoupledComponentConfig
from jax_esm.components.domain import Domain


class SlabModelBase(Component):
    """Base class for slab models providing shared infrastructure.

    This base class handles:
    - Domain initialization from grid specification
    - Lat/lon grid construction
    - Time offset calculations for climatology lookup
    - Common xarray coordinate creation for predictions

    Subclasses must implement:
    - _create_state_and_forcing_classes(): Define state and forcing structure
    - _initialize_fields(): Set up initial field values
    - _create_step_function_body(): Implement the physics for each timestep
    - _create_xarray_data_vars(): Define xarray data variables for output
    """

    def __init__(
        self,
        name: str,
        grid_specification: str = "JCM::T31",
        start_datetime: jdt.Datetime = jdt.to_datetime("2001-01-01"),
        timestep: float = 86400.0,
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
    ):
        """Initialize slab model base.

        Args:
            name: Component name (e.g., "SlabOceanModel")
            grid_specification: Grid spec string (e.g., "JCM::T31")
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds
            topography_file: Optional path to topography NetCDF file
            mask_file: Optional path to land/ocean mask NetCDF file
        """
        super().__init__(CoupledComponentConfig(name=name, timestep=timestep))

        self.start_datetime = start_datetime
        self.timestep = timestep
        self.topography_file = topography_file
        self.mask_file = mask_file

        # Initialize domain
        self.domain = Domain.from_grid_specification(
            grid_specification,
            topography_file=topography_file,
            mask_file=mask_file,
        )

        # Get grid shape for state/forcing class creation
        self.grid_shape = self.domain.grids["T"].nodal_shape

        # Subclass creates state and forcing classes
        self._create_state_and_forcing_classes()

        # Lat/lon grids will be set during initialize()
        self.llat_rad = None
        self.llon_rad = None

    @abstractmethod
    def _create_state_and_forcing_classes(self) -> None:
        """Create component_state_class and component_forcing_class.

        Subclasses must set:
        - self.component_state_class
        - self.component_forcing_class
        """
        pass

    def _setup_lat_lon_grids(self) -> None:
        """Set up 2D latitude and longitude grids in radians.

        Creates self.llat_rad and self.llon_rad as 2D arrays matching
        the T-grid shape.
        """
        T_grid = self.domain.grids["T"]
        D2_nodal_shape = T_grid.nodal_shape

        lat_dim_idx = next(
            i
            for i, axis_name in enumerate(T_grid.axis_names)
            if axis_name == "latitude"
        )
        lon_dim_idx = next(
            i
            for i, axis_name in enumerate(T_grid.axis_names)
            if axis_name == "longitude"
        )

        self.llat_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[lat_dim_idx],
                axis=lon_dim_idx,
            ),
            repeats=D2_nodal_shape[lon_dim_idx],
            axis=lon_dim_idx,
        )

        self.llon_rad = jnp.repeat(
            jnp.expand_dims(
                T_grid.axis_values[lon_dim_idx],
                axis=lat_dim_idx,
            ),
            repeats=D2_nodal_shape[lat_dim_idx],
            axis=lat_dim_idx,
        )

    def _compute_start_day_offset(self) -> float:
        """Compute the day offset from start of year for climatology lookup.

        Returns:
            Number of seconds from Jan 1 of start year to start_datetime
        """
        ref_year = self.start_datetime.to_pydatetime().year
        ref_dt = jdt.to_datetime(f"{ref_year:d}-01-01")
        return (self.start_datetime - ref_dt) / jdt.to_timedelta(1, "second")

    def _get_climatology_indices(
        self,
        t: float,
        start_day_offset: float,
        cycle_length: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute climatology indices for begin and end of timestep.

        Args:
            t: Current simulation time in seconds
            start_day_offset: Offset from start of year in seconds
            cycle_length: Number of days in the climatology cycle

        Returns:
            Tuple of (begin_index, end_index) for climatology lookup
        """
        clim_beg_idx = jnp.mod(
            jnp.int_(jnp.floor((start_day_offset + t) / 86400.0)),
            cycle_length,
        )
        clim_end_idx = jnp.mod(
            jnp.int_(jnp.floor((start_day_offset + t + self.timestep) / 86400.0)),
            cycle_length,
        )
        return clim_beg_idx, clim_end_idx

    def initialize(self):
        """Initialize the slab model state.

        Sets up lat/lon grids and delegates field initialization to subclass.

        Returns:
            Initial component state
        """
        self._setup_lat_lon_grids()
        return self._initialize_fields()

    @abstractmethod
    def _initialize_fields(self):
        """Initialize model-specific fields.

        Subclasses implement this to set up initial conditions,
        climatology data, and compute time factors.

        Returns:
            Initial component state
        """
        pass

    def generate_step_function(self, jitted: bool = True):
        """Generate the step function for time integration.

        Args:
            jitted: If True, JIT-compile the step function

        Returns:
            Step function with signature (state, forcing, t) -> (new_state, predictions)
        """
        step_fn = self._create_step_function_body()
        return jax.jit(step_fn) if jitted else step_fn

    @abstractmethod
    def _create_step_function_body(self):
        """Create the uncompiled step function body.

        Subclasses implement the physics equations here.

        Returns:
            Step function with signature (state, forcing, t) -> (new_state, predictions)
        """
        pass

    def validate(self):
        """Validate the model configuration."""
        pass

    def predictions_to_xarray(self, predictions) -> xr.Dataset:
        """Convert predictions to xarray Dataset.

        Args:
            predictions: Predictions dict from step function

        Returns:
            xarray Dataset with model output
        """
        T_grid_axis_names = self.domain.grids["T"].axis_names
        start_datetime_str = self.start_datetime.to_pydatetime().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Get time coordinate from predictions
        sim_time = self._get_sim_time_from_predictions(predictions)

        # Build coordinates dict
        coords = dict(
            time=(
                ["time"],
                sim_time / 3600.0,
                {"units": f"hours since {start_datetime_str:s}"},
            ),
            latitude2D=(T_grid_axis_names, self.llat_rad * 180 / jnp.pi),
            longitude2D=(T_grid_axis_names, self.llon_rad * 180 / jnp.pi),
        )

        # Get model-specific data variables from subclass
        data_vars = self._create_xarray_data_vars(predictions)

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def _get_sim_time_from_predictions(self, predictions) -> jnp.ndarray:
        """Extract simulation time from predictions.

        Args:
            predictions: Predictions dict from step function

        Returns:
            Array of simulation times
        """
        return predictions["prog"].sim_time

    @abstractmethod
    def _create_xarray_data_vars(self, predictions) -> Dict[str, Any]:
        """Create model-specific xarray data variables.

        Args:
            predictions: Predictions dict from step function

        Returns:
            Dict of data variables for xarray Dataset
        """
        pass

    def _get_grid_dims(self) -> Tuple[str, ...]:
        """Get the dimension names for grid variables.

        Returns:
            Tuple of dimension names including time
        """
        return ("time",) + self.domain.grids["T"].axis_names
