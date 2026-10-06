"""Tests for `jem.components.slab.slab_bucket_land_model`."""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jem.base.component import TimeAxis
from jem.components.slab.slab_bucket_land_model import SlabBucketLandModel
from tests.unit.slab_test_utils import coupling_time, make_grid

START_DATE = jdt.to_datetime("2000-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")
PRECIPITATION = 3.0e-5  # kg m-2 s-1, about 2.6 mm/day


def _all_land_model():
    """Return a bucket land model on a tiny all-land grid, bound to a clock."""
    grid = make_grid()
    model = SlabBucketLandModel(
        make_grid(fractional_mask=jnp.ones(grid.shape)))
    model.bind(coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE)
    return model


def test_precipitation_forcing_is_written_to_the_output():
    """The dataset reports the precipitation the soil was forced with.

    Under the `forcing_` prefix and the `forcing` role, so it merges with the
    atmosphere's own `precipitation` instead of colliding with it, and with
    the value the exchanger put in the carry rather than a recomputed one.
    """
    model = _all_land_model()
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        precipitation=jnp.full(model.grid.shape, PRECIPITATION))
    _, diagnostics = model.step(carry, coupling_time(0))

    stacked = jax.tree.map(lambda leaf: leaf[None], diagnostics)
    dataset = model.to_xarray(stacked, TimeAxis(
        start_date=START_DATE, steps=np.array([0]), dt=COUPLING_TIMESTEP))

    assert set(dataset.data_vars) == {
        "land_surface_temperature", "snowc", "soilw",
        "forcing_total_heat_flux", "forcing_precipitation",
    }
    precipitation = dataset["forcing_precipitation"]
    assert precipitation.dims == ("time",) + model.grid.dims
    assert precipitation.attrs["jem_role"] == "forcing"
    assert precipitation.attrs["units"] == "kg m-2 s-1"
    assert precipitation.attrs["positive"] == "downward"
    np.testing.assert_allclose(precipitation.values, PRECIPITATION, rtol=1e-6)
    # And it is the forcing the step actually used: the top soil layer, dry
    # at the start, holds water after one step of rain.
    assert float(dataset["soilw"].isel(layer=0).min()) > 0.0
