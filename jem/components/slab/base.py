"""Base class for slab models.

This module provides a common base class that extracts shared functionality
from SlabOceanModel, SlabLandModel, and SlabAtmosphereModel to reduce
code duplication.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr
from jcm.date import days_per_year as jcm_days_per_year

from jem.components.slab.grid import SlabGrid
from jem.utils.cycles import evaluate_cyclic_linear

_DEFAULT_START_DATETIME = jdt.to_datetime("2001-01-01")

# Dimension names a monthly climatology file may use, per axis, in the order
# the loader normalises them to: (longitude, latitude, time).
_CLIMATOLOGY_DIM_ALIASES = (
    ("lon", "longitude"),
    ("lat", "latitude"),
    ("time",),
)

_MONTHS_PER_YEAR = 12

# Tolerance for matching a climatology file's coordinates against the grid.
# The jcm T30 climatology that ships with jax-gcm stores its Gaussian
# latitudes rounded to three decimal places -- up to ~5e-4 degrees away from
# the exact Gaussian abscissae the grid computes -- so the tolerance has to be
# looser than that rounding. It is still two orders of magnitude tighter than
# the ~0.4 degree spacing of the finest grid JEM supports, so a file written
# on a genuinely different grid is rejected rather than silently regridded.
_COORDINATE_TOLERANCE_DEGREES = 1e-3


def _resolve_dim_name(
    dims: tuple[str, ...],
    aliases: tuple[str, ...],
    path,
    var: str,
) -> str:
    """Return which of `aliases` names one of `dims`."""
    for alias in aliases:
        if alias in dims:
            return alias
    raise ValueError(
        f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has dimensions "
        f"{dims!r}, none of which is one of {aliases!r}."
    )


def _grid_axes_degrees(grid: SlabGrid, path) -> tuple[np.ndarray, np.ndarray]:
    """Return the grid's 1-D (longitude, latitude) axes in degrees.

    A SlabGrid only stores 2-D radian fields, so the 1-D axes a climatology
    file is written on have to be recovered from them. That is only
    meaningful for a separable lat-lon grid, which is checked here rather
    than left to surface later as a confusing coordinate mismatch.
    """
    longitude = np.rad2deg(np.asarray(grid.longitude_radian, dtype=np.float64))
    latitude = np.rad2deg(np.asarray(grid.latitude_radian, dtype=np.float64))
    longitude_axis = longitude[:, 0]
    latitude_axis = latitude[0, :]

    separable = np.allclose(
        longitude, longitude_axis[:, None], atol=_COORDINATE_TOLERANCE_DEGREES
    ) and np.allclose(
        latitude, latitude_axis[None, :], atol=_COORDINATE_TOLERANCE_DEGREES
    )
    if not separable:
        raise ValueError(
            f"Climatology file \"{path!s:s}\" cannot be matched against this grid: "
            "the grid is curvilinear (longitude/latitude are not separable), so it "
            "has no 1-D longitude/latitude axes to compare the file's coordinates to."
        )

    return longitude_axis, latitude_axis


def _check_axis_matches(
    file_values: np.ndarray,
    grid_values: np.ndarray,
    axis_name: str,
    path,
    var: str,
    periodic: bool = False,
) -> None:
    """Raise if a file coordinate axis does not match the grid's."""
    if file_values.shape != grid_values.shape:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has "
            f"{file_values.size:d} {axis_name:s} points but the grid has "
            f"{grid_values.size:d}."
        )

    difference = file_values - grid_values
    if periodic:
        # Longitudes may be written on 0-360 or -180-180; compare modulo 360.
        difference = (difference + 180.0) % 360.0 - 180.0

    largest = float(np.max(np.abs(difference)))
    if largest > _COORDINATE_TOLERANCE_DEGREES:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" is on a "
            f"different {axis_name:s} axis than the grid (largest difference "
            f"{largest:.6g} degrees, tolerance {_COORDINATE_TOLERANCE_DEGREES:g}). "
            "The file must already be on the model grid; this loader does not regrid."
        )


def load_monthly_climatology(path, var: str, grid: SlabGrid) -> jnp.ndarray:
    """Load a monthly climatology field onto the model grid.

    The array is transposed BY NAME rather than by position, so a file written
    in any axis order loads correctly and a file written on the wrong grid is
    rejected instead of being reinterpreted: a bare ``jnp.array(ds[var])`` would
    silently accept any array whose shape happened to fit.

    Lives on the slab base module because every slab component reads its
    boundary conditions as ``(n_lon, n_lat, 12)`` monthly climatologies on the
    model grid; the ocean uses it for SST and Q-flux, and the land model's
    positional loader is scheduled to move onto it (api_hardening_plan T1.4).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a netCDF file.
    var : str
        Name of the variable to read.
    grid : SlabGrid
        The model grid the file must already be on.

    Returns
    -------
    jnp.ndarray
        The field with shape ``(n_lon, n_lat, 12)``.

    Raises
    ------
    ValueError
        If the variable is missing, its dimensions are not recognisable as
        longitude/latitude/time, it does not have exactly 12 time records, or
        its coordinates do not match the grid's. The message names the file
        and the check that failed.

    """
    dataset = xr.open_dataset(path)

    if var not in dataset:
        raise ValueError(
            f"Climatology file \"{path!s:s}\" has no variable \"{var:s}\" "
            f"(it has {sorted(map(str, dataset.data_vars))!r})."
        )
    field = dataset[var]

    longitude_name, latitude_name, time_name = (
        _resolve_dim_name(tuple(str(d) for d in field.dims), aliases, path, var)
        for aliases in _CLIMATOLOGY_DIM_ALIASES
    )

    unexpected = set(field.dims) - {longitude_name, latitude_name, time_name}
    if unexpected:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has extra "
            f"dimensions {sorted(map(str, unexpected))!r}; only longitude, latitude and time "
            "are supported."
        )

    n_records = field.sizes[time_name]
    if n_records != _MONTHS_PER_YEAR:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has "
            f"{n_records:d} \"{time_name:s}\" records; exactly "
            f"{_MONTHS_PER_YEAR:d} monthly records are required."
        )

    for name in (longitude_name, latitude_name):
        if name not in field.coords:
            raise ValueError(
                f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has no "
                f"\"{name:s}\" coordinate variable, so its grid cannot be verified."
            )

    grid_longitude, grid_latitude = _grid_axes_degrees(grid, path)
    _check_axis_matches(
        np.asarray(field.coords[longitude_name].values, dtype=np.float64),
        grid_longitude,
        "longitude",
        path,
        var,
        periodic=True,
    )
    _check_axis_matches(
        np.asarray(field.coords[latitude_name].values, dtype=np.float64),
        grid_latitude,
        "latitude",
        path,
        var,
    )

    return jnp.asarray(
        field.transpose(longitude_name, latitude_name, time_name).values
    )


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
