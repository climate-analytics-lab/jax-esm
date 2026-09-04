"""Shared infrastructure for the slab models.

What lives here is what every slab component needs and none of them should own
a copy of: the grid, the monthly-climatology loader, and the conversion of a
run's stacked diagnostics into a CF-labelled :class:`xarray.Dataset`.

What deliberately does *not* live here any more is the clock. The coupler owns
the one clock of a coupled run and hands it to every component as a
:class:`~jem.base.component.CouplingTime`; a slab model holds no start date, no
timestep and no calendar, so two components cannot disagree about the date and
the seasonal cycle survives a chunked or restarted run unbroken. The one thing
``initialize()`` still needs the date for -- which month of a climatology the
run starts in -- reaches the model through
:meth:`SlabModelBase.bind`, which the coupler calls at registration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr

from jem.base.component import (
    Carry,
    CouplingTime,
    Diagnostics,
    TimeAxis,
    start_year_fraction,
)
from jem.components.slab.grid import SlabGrid, to_degrees
from jem.utils.time import time_coordinate

logger = logging.getLogger(__name__)

#: Temperature (K) reported for cells a component does not integrate -- ocean
#: temperature over land, land temperature over ocean. It is a fill value for
#: output and for the masked branch of an update, never a prognostic: nothing
#: in any slab model reads it back. 288.15 K (15 degC) is what the models have
#: always used.
MASKED_SURFACE_TEMPERATURE = 288.15

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

#: CF attributes for the horizontal coordinates, copied from
#: ``jcm.cf_metadata._COORD_ATTRS`` so a slab dataset and a JCM dataset
#: describe their shared axes identically.
_LONGITUDE_ATTRS = {
    "standard_name": "longitude",
    "units": "degrees_east",
    "long_name": "longitude",
}
_LATITUDE_ATTRS = {
    "standard_name": "latitude",
    "units": "degrees_north",
    "long_name": "latitude",
}


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

    Only a separable lat-lon grid has 1-D axes for a climatology file's
    coordinates to be compared against, which is checked here rather than left
    to surface later as a confusing coordinate mismatch.
    """
    if not grid.is_separable:
        raise ValueError(
            f"Climatology file \"{path!s:s}\" cannot be matched against this grid: "
            "the grid is curvilinear (longitude/latitude are not separable), so it "
            "has no 1-D longitude/latitude axes to compare the file's coordinates to."
        )
    return (
        to_degrees(grid.longitude_axis_radian),
        to_degrees(grid.latitude_axis_radian),
    )


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


def first_present_variable(path, candidates: tuple[str, ...]) -> str | None:
    """Return the first of `candidates` present in the netCDF file at `path`.

    Boundary files written by different tools spell the same field differently
    (jax-gcm writes ``snowc`` and ``soilw_am`` where SPEEDY writes ``snowd``
    and ``soilw``). Resolving the name *before* loading keeps
    :func:`load_monthly_climatology` strict -- it still fails loudly on a
    variable that is genuinely absent -- instead of each component reaching
    into the dataset itself and losing the grid check.

    Returns
    -------
    str or None
        The first candidate that exists, or None if the file has none of them.

    """
    with xr.open_dataset(path) as dataset:
        for candidate in candidates:
            if candidate in dataset:
                return candidate
    return None


def load_monthly_climatology(path, var: str, grid: SlabGrid) -> jnp.ndarray:
    """Load a monthly climatology field onto the model grid.

    The array is transposed BY NAME rather than by position, so a file written
    in any axis order loads correctly and a file written on the wrong grid is
    rejected instead of being reinterpreted: a bare ``jnp.array(ds[var])`` would
    silently accept any array whose shape happened to fit.

    Lives on the slab base module because every slab component reads its
    boundary conditions as ``(n_lon, n_lat, 12)`` monthly climatologies on the
    model grid: the ocean uses it for SST and Q-flux, the land model for its
    surface temperature, snow and soil water.

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


def end_of_step(time: CouplingTime) -> CouplingTime:
    """Return the clock as it will read at the *end* of ``time``'s step.

    Several slab models need a boundary condition at both ends of a step (the
    climatology an anomaly is measured against at the start, and added back to
    at the end). ``CouplingTime.end_of_step`` is the one definition of what
    "one step later" means; this alias keeps the slab call sites short.
    """
    return time.end_of_step()


class SlabModelBase(ABC):
    """Base class for slab models providing shared infrastructure.

    A subclass is a :class:`~jem.base.component.Component`: it holds its
    configuration (grid, parameters, boundary data) on ``self`` and its
    evolving state in the carry it returns from :meth:`initialize` and threads
    through :meth:`step`. The carry always contains a ``"params"`` entry, so
    the model's tunables are pytree leaves of the coupled state and
    ``jax.grad`` with respect to them works through the coupler with no special
    casing.

    Subclasses must implement:

    - ``initialize()`` -- build the initial carry. Pure with respect to
      ``self``: boundary data is loaded in ``__init__``, so calling it twice
      gives the same answer and calling it never mutates the component.
    - ``step(carry, time)`` -- advance one coupling step.
    - ``_create_xarray_data_vars(diagnostics)`` -- name the output variables.

    Attributes
    ----------
    name : str
        The component's name in the coupler's workflow and carry.
    grid : SlabGrid
        The model's grid.

    """

    name: str
    grid: SlabGrid

    def __init__(self, name: str, grid: SlabGrid):
        """Initialize the shared slab-model state.

        Parameters
        ----------
        name : str
            Component name, unique within a coupler.
        grid : SlabGrid
            The model's grid. Build one with
            :meth:`~jem.components.slab.grid.SlabGrid.from_coords` (from the
            atmosphere's horizontal grid) or
            :meth:`~jem.components.slab.grid.SlabGrid.from_scrip`.

        """
        self.name = name
        self.grid = grid
        # Where the run starts in the annual cycle; see `bind` and
        # `start_year_fraction`. An unbound model reads its climatologies at
        # 1 January.
        self._start_year_fraction = 0.0

    def bind(
        self,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str,
    ) -> None:
        """Adopt the coupler's clock (:class:`~jem.base.component.SupportsBind`).

        A slab model has no internal timestep to reconcile -- it advances by
        exactly the ``dt`` on the :class:`~jem.base.component.CouplingTime` it
        is handed, so ``coupling_timestep`` is accepted (the coupler passes the
        same three facts to every bindable component) and not used. What a slab
        model does need is the *date*: :meth:`initialize` samples monthly
        climatologies for the initial condition, and it receives no clock,
        because the clock lives in the carry and the carry does not exist yet.

        Parameters
        ----------
        coupling_timestep : jax_datetime.Timedelta
            The coupled timestep. Unused; see above.
        start_date : jax_datetime.Datetime
            The run's start date. Its position in the annual cycle is what
            :attr:`start_year_fraction` reports.
        calendar : str
            The run's calendar, which fixes the length of the year.

        """
        del coupling_timestep
        self._start_year_fraction = start_year_fraction(start_date, calendar)

    @property
    def start_year_fraction(self) -> float:
        """Position of the run's start date in the annual cycle, in ``[0, 1)``.

        What :meth:`initialize` samples its climatologies at. It is ``0.0``
        (1 January) for a model that was never registered with a
        :class:`~jem.base.coupler.Coupler`, which is what a bare
        ``model.initialize()`` in a test or a notebook gets.
        """
        return self._start_year_fraction

    @abstractmethod
    def initialize(self) -> Carry:
        """Build the initial carry. Must not integrate the model."""

    @abstractmethod
    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance one coupling timestep, returning the new carry and this step's output."""

    def to_xarray(self, diagnostics: Diagnostics, time: TimeAxis) -> xr.Dataset:
        """Convert a run's stacked diagnostics to a CF-labelled Dataset.

        The coupler stacks each step's diagnostics into a leading time axis
        before calling this, so every field is ``(time, *grid.dims)``.

        Coordinates follow JCM's, so ``xr.merge`` of an atmosphere dataset and
        a slab dataset from the same run aligns instead of producing an outer
        join: ``time`` is the absolute ``datetime64[ns]`` axis built by
        :func:`jem.utils.time.time_coordinate`, and a separable grid writes 1-D
        ``lon``/``lat`` in degrees with the same values JCM writes for the same
        coordinate system. A curvilinear grid cannot: it writes 2-D auxiliary
        ``lat``/``lon`` coordinates over index-space dimensions, and each data
        variable gets the CF ``coordinates`` attribute that points at them.

        Parameters
        ----------
        diagnostics : Diagnostics
            The stacked per-step output of :meth:`step`.
        time : jem.base.component.TimeAxis
            The records' place in the run, from ``Coupler.time_axis``.

        Returns
        -------
        xarray.Dataset

        """
        data_vars = self._create_xarray_data_vars(diagnostics)
        self._check_time_axis_length(data_vars, time)

        time_values, time_attrs = time_coordinate(time)
        coords: dict[str, Any] = {"time": ("time", time_values, time_attrs)}

        grid_dims = self.grid.dims
        if self.grid.is_separable:
            coords["lon"] = (
                "lon",
                to_degrees(self.grid.longitude_axis_radian),
                {**_LONGITUDE_ATTRS, "axis": "X"},
            )
            coords["lat"] = (
                "lat",
                to_degrees(self.grid.latitude_axis_radian),
                {**_LATITUDE_ATTRS, "axis": "Y"},
            )
        else:
            # CF forbids ``axis`` on an auxiliary coordinate: it marks a true
            # coordinate variable, and these are 2-D fields over index-space
            # dimensions.
            coords["lon"] = (
                grid_dims,
                to_degrees(self.grid.longitude_radian),
                dict(_LONGITUDE_ATTRS),
            )
            coords["lat"] = (
                grid_dims,
                to_degrees(self.grid.latitude_radian),
                dict(_LATITUDE_ATTRS),
            )
            data_vars = {
                name: (dims, values, {**attrs, "coordinates": "lat lon"})
                for name, (dims, values, attrs) in data_vars.items()
            }

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self._create_xarray_global_attributes(),
        )

    def _check_time_axis_length(self, data_vars: dict[str, Any], time: TimeAxis) -> None:
        """Raise if the diagnostics and the time axis describe different runs.

        Mismatched lengths otherwise surface as an opaque xarray broadcasting
        error naming a dimension the caller never wrote.
        """
        for name, (_dims, values, *_rest) in data_vars.items():
            n_records = np.shape(values)[0]
            if n_records != len(time):
                raise ValueError(
                    f"{self.name}: output variable \"{name}\" has {n_records:d} time "
                    f"records but the time axis has {len(time):d}. The diagnostics "
                    "handed to to_xarray must be the coupler's stacked trajectory."
                )

    @abstractmethod
    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Return the output variables as ``{name: (dims, values, attrs)}``."""

    def _create_xarray_global_attributes(self) -> dict[str, Any]:
        """Return the Dataset's global attributes (empty unless overridden)."""
        return {}
