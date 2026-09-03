"""Base class for slab models.

This module provides a common base class that extracts shared functionality
from SlabOceanModel, SlabLandModel, and SlabAtmosphereModel to reduce
code duplication.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr
from jcm.date import days_per_year as jcm_days_per_year

from jem.components.slab.grid import SlabGrid
from jem.utils.cycles import evaluate_cyclic_linear

_DEFAULT_START_DATETIME = jdt.to_datetime("2001-01-01")


class SlabModelBase(ABC):
    """Base class for slab models providing shared infrastructure.

    This base class handles:
    - Storing the model's SlabGrid (fractional_mask, latitude_radian,
      longitude_radian fully specify it -- there is no separate grid
      specification)
    - Time offset calculations for climatology lookup
    - Common xarray coordinate creation for predictions

    Subclasses must implement:
    - initialize(): Build the initial state/forcing/derived carry
    - _create_step_function_body(): Implement the physics for each timestep
    - _create_xarray_data_vars(): Define xarray data variables for output
    """

    grid: SlabGrid


    def __init__(
        self,
        name: str,
        grid: SlabGrid,
        start_datetime: jdt.Datetime = _DEFAULT_START_DATETIME,
        timestep: float = 86400.0,
        calendar: str = "365_day",
    ):
        """Initialize slab model base.

        Args:
            name: Component name (e.g., "SlabOceanModel")
            grid: The model's grid. See
                `jem.components.slab.grid.SlabGrid`, and
                `jem.components.slab.grid.generate_slab_grid` to build one
                from one of JEM's canonical grid specifications.
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds

        """
        self.name = name
        self.grid = grid
        self.start_datetime = start_datetime
        self.timestep = timestep
        self.calendar = calendar
        self.days_per_year = jcm_days_per_year(calendar)

    def _compute_start_day_offset(self) -> float:
        """Seconds from Jan 1 of start year to start_datetime."""
        ref_year = self.start_datetime.to_pydatetime().year
        ref_dt = jdt.to_datetime(f"{ref_year:d}-01-01")
        return float((self.start_datetime - ref_dt) / jdt.to_timedelta(1, "second"))

    def _year_fraction(self, t: float, start_day_offset: float) -> jnp.ndarray:
        """Return cycle position in [0, 1) for simulation time t.

        Args:
            t: Simulation time in seconds since start
            start_day_offset: Seconds from Jan 1 of start year to start_datetime

        """
        return jnp.mod(
            (start_day_offset + t) / (86400.0 * self.days_per_year), 1.0
        )

    def _interpolate_cyclic(
        self,
        t: float,
        start_day_offset: float,
        data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Linearly interpolate cyclic climatology data at simulation time t.

        Records in data (last axis) are assumed equally spaced over one year.
        Interpolation is continuous and periodic across year boundaries.

        Args:
            t: Simulation time in seconds since start
            start_day_offset: Seconds from Jan 1 of start year to start_datetime
            data: Array of shape (..., n_records)

        Returns:
            Interpolated array of shape (...)

        """
        return evaluate_cyclic_linear(self._year_fraction(t, start_day_offset), data)

    @abstractmethod
    def initialize(self):
        """Initialize the slab model state.

        Returns:
            Initial component state and forcing

        """

    def generate_step_function(self):
        """Generate the step function for time integration.

        Returns:
            Step function with signature (state, forcing, t) -> (new_state, predictions)

        """
        step_fn = self._create_step_function_body()
        return step_fn

    @abstractmethod
    def _create_step_function_body(self):
        """Create the uncompiled step function body.

        Subclasses implement the physics equations here.

        Returns:
            Step function with signature (state, forcing, t) -> (new_state, predictions)

        """

    def validate(self):
        """Validate the model configuration."""

    def predictions_to_xarray(self, predictions) -> xr.Dataset:
        """Convert predictions to xarray Dataset.

        Args:
            predictions: Predictions dict from step function

        Returns:
            xarray Dataset with model output

        """
        T_grid_axis_names = self.grid.dims
        start_datetime_str = self.start_datetime.to_pydatetime().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        coords = {
            "time": (
                ["time"],
                predictions["state"].sim_time / 3600.0,
                {"units": f"hours since {start_datetime_str:s}"},
            ),
            "latitude2D": (T_grid_axis_names, self.grid.latitude_radian * 180 / jnp.pi),
            "longitude2D": (T_grid_axis_names, self.grid.longitude_radian * 180 / jnp.pi),
        }

        data_vars = self._create_xarray_data_vars(predictions)

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self._create_xarray_global_attributes(),
        )

    @abstractmethod
    def _create_xarray_data_vars(self, predictions) -> dict[str, Any]:
        """Create model-specific xarray data variables.

        Args:
            predictions: Predictions dict from step function

        Returns:
            Dict of data variables for xarray Dataset

        """
    
    def _create_xarray_global_attributes(self) -> dict[str, Any]:
        """Create model-specific xarray Dataset global attributes.

        Returns:
            Dict of global attributes for xarray Dataset

        """
        return {}

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
        }
