"""Shared helpers for the slab-component unit tests.

The slab tests all need the same three things: a tiny grid they own, a
:class:`~jem.base.component.CouplingTime` built by hand (the coupler is not
under test here), and a way to write a climatology file onto that grid. They
live here rather than in a ``conftest.py`` so that each test file names what it
imports.
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr

from jem.base.component import CouplingTime, TimeAxis
from jem.components.slab.grid import SlabGrid

LONGITUDE_DEGREES = np.array([0.0, 90.0, 180.0, 270.0])
LATITUDE_DEGREES = np.array([-60.0, 0.0, 60.0])
TIMESTEP = 86400.0
DAYS_PER_YEAR = 365.0


def coupling_time(
    step: int,
    dt: float = TIMESTEP,
    year_offset_seconds: float = 0.0,
    days_per_year: float = DAYS_PER_YEAR,
) -> CouplingTime:
    """Build the clock the coupler would hand a component at `step`."""
    return CouplingTime(
        step=jnp.int32(step),
        sim_time=jnp.float32(step * dt),
        dt=dt,
        year_offset_seconds=year_offset_seconds,
        days_per_year=days_per_year,
    )


def time_axis(n_records: int, dt: float = TIMESTEP) -> TimeAxis:
    """Build the output time axis the coupler would hand `to_xarray`."""
    return TimeAxis(
        start_date=jdt.to_datetime("2001-01-01"),
        steps=np.arange(n_records),
        dt=jdt.to_timedelta(int(dt), "second"),
        calendar="365_day",
    )


def make_grid(fractional_mask=None, threshold: float = 0.5) -> SlabGrid:
    """Return a 4x3 lon-lat grid, all ocean unless a land fraction is given."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    longitude_axis = np.deg2rad(LONGITUDE_DEGREES)
    latitude_axis = np.deg2rad(LATITUDE_DEGREES)
    if fractional_mask is None:
        fractional_mask = jnp.zeros(shape)
    return SlabGrid(
        fractional_mask=jnp.asarray(fractional_mask),
        latitude_radian=jnp.asarray(np.broadcast_to(latitude_axis[None, :], shape)),
        longitude_radian=jnp.asarray(np.broadcast_to(longitude_axis[:, None], shape)),
        threshold=threshold,
        longitude_axis_radian=longitude_axis,
        latitude_axis_radian=latitude_axis,
    )


def write_climatology(
    path,
    var: str,
    values_time_lat_lon: np.ndarray,
    longitude_degrees: np.ndarray = LONGITUDE_DEGREES,
    latitude_degrees: np.ndarray = LATITUDE_DEGREES,
) -> str:
    """Write a 12-month climatology in (time, lat, lon) order and return its path."""
    dataset = xr.Dataset(
        data_vars={var: (("time", "lat", "lon"), values_time_lat_lon)},
        coords={
            "time": np.arange(12),
            "lat": latitude_degrees,
            "lon": longitude_degrees,
        },
    )
    dataset.to_netcdf(path)
    return str(path)


def monthly_ramp() -> np.ndarray:
    """Return a (12, n_lat, n_lon) field whose every element is distinct."""
    return np.arange(
        12 * len(LATITUDE_DEGREES) * len(LONGITUDE_DEGREES), dtype=np.float32
    ).reshape(12, len(LATITUDE_DEGREES), len(LONGITUDE_DEGREES))


def run_steps(model, carry, n_steps: int, dt: float = TIMESTEP):
    """Thread `n_steps` steps by hand and stack the diagnostics like the coupler does.

    The coupler stacks each step's diagnostics into a leading time axis with
    ``lax.scan``; doing the same here with ``jnp.stack`` keeps these tests
    independent of the coupler while producing exactly what ``to_xarray``
    expects.
    """
    diagnostics = []
    for step in range(n_steps):
        carry, output = model.step(carry, coupling_time(step, dt=dt))
        diagnostics.append(output)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *diagnostics)
    return carry, stacked


def tree_signature(tree):
    """Structure, shapes and dtypes of a pytree -- what ``lax.scan`` compares."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return treedef, [(jnp.shape(leaf), jnp.asarray(leaf).dtype) for leaf in leaves]
