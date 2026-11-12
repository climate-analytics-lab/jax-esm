"""Tests for the bucket land (:mod:`jem.components.slab.bucket_land_model`)."""

import dataclasses

import jax
import jax.numpy as jnp
import jcm.constants as jcm_constants
import numpy as np
import pytest

from jem.components.slab.bucket_land_model import (
    BucketLandModel,
    BucketLandParameters,
    soil_water_availability,
)
from jem.components.slab.bucket_land_model.bucket_land_model import WATER_DENSITY
from tests.unit.slab_test_utils import (
    TIMESTEP,
    coupling_time,
    make_grid,
    run_steps,
    time_axis,
    tree_signature,
    write_climatology,
)

# Land on the two eastern columns, a coast column of 0.4 land, ocean west.
LAND_FRACTION = np.array([
    [0.0, 0.0, 0.0],
    [0.4, 0.4, 0.4],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
])


@pytest.fixture
def grid():
    return make_grid(LAND_FRACTION)


@pytest.fixture
def model(grid):
    return BucketLandModel(grid)


def _forced(carry, precipitation, evaporation):
    shape = carry["forcing"].precipitation.shape
    return dict(carry, forcing=carry["forcing"].replace(
        precipitation=jnp.full(shape, precipitation),
        evaporation=jnp.full(shape, evaporation),
    ))


def _storage(carry, params):
    """Water the soil holds, kg m-2 of land."""
    return WATER_DENSITY * jnp.sum(
        carry["state"].soil_water_content * params.soil_bucket_depth, axis=-1)


def test_only_cells_with_land_hold_water(model, grid):
    carry = model.initialize()
    water = np.asarray(carry["state"].soil_water_content)
    assert np.all(water[LAND_FRACTION == 0.0] == 0.0)
    assert np.all(water[LAND_FRACTION > 0.0] > 0.0)
    # The coast column is land too: it carries the soil of its land part.
    assert np.all(water[1] > 0.0)


def test_without_a_climatology_the_soil_starts_half_full(model):
    params = model.params
    water = model.initialize()["state"].soil_water_content[2, 0]
    expected = params.soil_water_wilting_point + 0.5 * (
        params.soil_water_field_capacity - params.soil_water_wilting_point)
    np.testing.assert_allclose(water, expected, rtol=1e-6)


def test_the_soil_starts_from_the_climatologys_availability(tmp_path, grid):
    """The initial availability is the observed one, where it is reachable."""
    availability = np.full((12, 3, 4), 0.3, dtype=np.float32)
    path = write_climatology(tmp_path / "soilw.nc", "soilw_am", availability)
    model = BucketLandModel(grid, soil_water_clim_file=path)
    derived = model.initialize()["derived"]
    np.testing.assert_allclose(
        np.asarray(derived.soil_water_availability)[LAND_FRACTION > 0.0], 0.3, rtol=1e-5)
    assert np.all(np.asarray(derived.soil_water_availability)[LAND_FRACTION == 0.0] == 0.0)


def test_availability_is_speedys_formula():
    """``(d1 max(W1 - wp1, 0) + veg d2 max(W2 - wp2, 0)) / (d1 (fc1 - wp1) +
    d2 (fc2 - wp2))`` -- with SPEEDY's depths, ``rsw (W1 + veg idep2
    max(W2 - swwil, 0))`` with ``rsw = 1 / (swcap + idep2 (swcap - swwil))``.
    """
    params = BucketLandParameters.default()
    water = jnp.array([[0.12, 0.25]])
    swcap, swwil, idep2, veg = 0.30, 0.17, 3.0, 0.8
    rsw = 1.0 / (swcap + idep2 * (swcap - swwil))
    expected = rsw * (0.12 + veg * idep2 * max(0.25 - swwil, 0.0))
    np.testing.assert_allclose(soil_water_availability(water, params), [expected], rtol=1e-6)


@pytest.mark.parametrize("precipitation, evaporation", [
    (3e-5, 1e-5),     # wet: drainage and overflow
    (0.0, 4e-5),      # drying: evaporation the soil cannot all give
    (1e-6, -2e-6),    # dew
])
def test_the_water_budget_closes(model, precipitation, evaporation):
    """Per unit of land area, ``d(storage)/dt = P - (E - unmet) - runoff``."""
    params = model.params
    carry = model.initialize()
    for step in range(20):
        before = _storage(carry, params)
        carry, _ = model.step(_forced(carry, precipitation, evaporation),
                              coupling_time(step))
        derived = carry["derived"]
        change = (_storage(carry, params) - before) / TIMESTEP
        balance = (precipitation - (evaporation - derived.unmet_evaporation)
                   - derived.runoff)
        land = LAND_FRACTION > 0.0
        np.testing.assert_allclose(
            np.asarray(change)[land], np.asarray(balance)[land], rtol=1e-4, atol=1e-10)


def test_dry_soil_reports_the_evaporation_it_cannot_give(model):
    carry = model.initialize()
    carry, _ = run_steps(model, _forced(carry, 0.0, 1e-3), 3)
    unmet = np.asarray(carry["derived"].unmet_evaporation)
    assert np.all(unmet[LAND_FRACTION > 0.0] > 0.0)
    assert np.all(unmet[LAND_FRACTION == 0.0] == 0.0)


def test_the_soil_stays_between_wilting_point_and_field_capacity(model):
    params = model.params
    carry = model.initialize()
    for precipitation, evaporation in ((1e-3, 0.0), (0.0, 1e-3)):
        carry, _ = run_steps(model, _forced(carry, precipitation, evaporation), 5)
        water = np.asarray(carry["state"].soil_water_content)[LAND_FRACTION > 0.0]
        assert np.all(water >= np.asarray(params.soil_water_wilting_point) - 1e-7)
        assert np.all(water <= np.asarray(params.soil_water_field_capacity) + 1e-7)


def test_drainage_is_stable_for_any_coupling_step(grid):
    """Implicit drainage never takes a layer below its wilting point, even
    with a time scale far shorter than the coupling step.
    """
    params = BucketLandParameters.default().replace(
        drainage_timescale=jnp.asarray((3600.0, 3600.0)))
    model = BucketLandModel(grid, params)
    carry, _ = run_steps(model, _forced(model.initialize(), 2e-4, 0.0), 4)
    water = np.asarray(carry["state"].soil_water_content)[LAND_FRACTION > 0.0]
    assert np.all(np.isfinite(water))
    assert np.all(water >= np.asarray(params.soil_water_wilting_point) - 1e-7)


def test_dew_goes_into_the_surface_layer(model):
    """Dew adds its water, all of it, and all through the surface layer:
    against a step without it, the surface layer holds more, the deep
    layer only what drained down from that excess, and the water kept plus
    the extra runoff it caused is exactly the dew.
    """
    params = model.params
    carry = model.initialize()
    dew, _ = model.step(_forced(carry, 0.0, -1e-6), coupling_time(0))
    dry, _ = model.step(_forced(carry, 0.0, 0.0), coupling_time(0))
    land = LAND_FRACTION == 1.0
    difference = (dew["state"].soil_water_content
                  - dry["state"].soil_water_content)
    assert np.all(np.asarray(difference[..., 0])[land] > 0.0)
    assert np.all(np.asarray(difference[..., 1])[land] >= 0.0)
    assert np.all(np.asarray(dew["derived"].unmet_evaporation)[land] == 0.0)
    kept = _storage(dew, params) - _storage(dry, params)
    extra_runoff = (dew["derived"].runoff - dry["derived"].runoff) * TIMESTEP
    np.testing.assert_allclose(
        np.asarray(kept + extra_runoff)[land], 1e-6 * TIMESTEP, rtol=1e-3)


def test_the_runoff_mass_rate_sums_runoff_over_the_land_area(model, grid):
    carry, _ = run_steps(model, _forced(model.initialize(), 1e-3, 0.0), 2)
    derived = carry["derived"]
    expected = np.sum(np.asarray(grid.cell_solid_angle) * jcm_constants.rearth ** 2
                      * LAND_FRACTION * np.asarray(derived.runoff))
    assert float(derived.runoff_mass_rate) > 0.0
    np.testing.assert_allclose(float(derived.runoff_mass_rate), expected, rtol=1e-5)


def test_two_steps_keep_the_carry_structure(model):
    """What ``lax.scan`` compares: structure, shapes and dtypes."""
    carry0 = model.initialize()
    carry1, _ = model.step(carry0, coupling_time(0))
    carry2, _ = model.step(carry1, coupling_time(1))
    assert tree_signature(carry0) == tree_signature(carry1) == tree_signature(carry2)


def test_the_bucket_is_differentiable(model):
    """``jax.grad`` of the availability after a step reaches the drainage
    time scale and the precipitation, finitely.
    """
    carry = _forced(model.initialize(), 2e-5, 1e-5)

    def availability(tau, precipitation):
        params = carry["params"].replace(drainage_timescale=tau)
        forced = dict(carry, params=params, forcing=carry["forcing"].replace(
            precipitation=jnp.full(carry["forcing"].precipitation.shape, precipitation)))
        stepped, _ = model.step(forced, coupling_time(0))
        return jnp.sum(stepped["derived"].soil_water_availability)

    grads = jax.grad(availability, argnums=(0, 1))(
        carry["params"].drainage_timescale, 2e-5)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
    assert float(grads[1]) > 0.0


def test_output_is_labelled(model):
    _, stacked = run_steps(model, model.initialize(), 2)
    dataset = model.to_xarray(stacked, time_axis(2))
    assert dataset["soil_water_content"].dims == ("time", "lon", "lat", "soil_layer")
    assert dataset["runoff_mass_rate"].dims == ("time",)
    assert dataset["forcing_evaporation"].attrs["jem_role"] == "forcing"
    assert dataset["soil_water_availability"].attrs["jem_role"] == "derived"


@pytest.mark.parametrize("field, value", [
    ("soil_bucket_depth", jnp.asarray((0.0, 0.21))),
    ("drainage_timescale", jnp.asarray((-1.0, 86400.0))),
    ("soil_water_field_capacity", jnp.asarray((0.30, 0.10))),
    ("vegetation_fraction", 1.5),
    ("initial_soil_water_fill", -0.1),
])
def test_invalid_parameters_are_refused(grid, field, value):
    params = BucketLandParameters.default().replace(**{field: value})
    with pytest.raises(ValueError, match=field):
        BucketLandModel(grid, params)


def test_a_grid_without_cell_areas_is_refused():
    grid = dataclasses.replace(make_grid(LAND_FRACTION), cell_solid_angle=None)
    with pytest.raises(ValueError, match="cell areas"):
        BucketLandModel(grid)
