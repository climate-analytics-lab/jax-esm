"""Tests for `jem.components.slab.slab_seaice_model`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import jcm.constants as jcm_constants
from jem import constants
from jem.components.slab.base import MASKED_SURFACE_TEMPERATURE
from jem.components.slab.slab_seaice_model import SlabSeaiceModel, SlabSeaiceParameters
from tests.unit.slab_test_utils import (
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    coupling_time,
    make_grid,
    tree_signature,
)

#: Energy (J/m2) that freezes one metre of ice, for the constants JCM owns.
ENERGY_PER_METRE = jcm_constants.rhoi * jcm_constants.alhf


@pytest.fixture
def uniform_grid():
    """Return a tiny all-ocean 4x3 lon-lat grid the tests own."""
    return make_grid()


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    fractional_mask = jnp.where(
        jnp.arange(shape[0])[:, None] >= shape[0] // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return make_grid(fractional_mask=fractional_mask)


def test_positive_frazil_energy_grows_ice(uniform_grid):
    """The freeze/melt potential is an energy per coupling step, applied as-is."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(uniform_grid.shape, 0.5 * ENERGY_PER_METRE)
    )

    stepped, _ = model.step(carry, coupling_time(0))

    np.testing.assert_allclose(
        np.asarray(stepped["state"].ice_thickness), 0.5, rtol=1e-5
    )
    # Half a metre against the 0.5 m fill-in scale: 1 - exp(-1).
    np.testing.assert_allclose(
        np.asarray(stepped["derived"].ice_fraction), 1.0 - np.exp(-1.0), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(stepped["state"].ice_surface_temperature), jcm_constants.tmelt
    )


def test_melt_cannot_drive_thickness_negative(uniform_grid):
    """Surplus ocean heat melts ice, and stops at open water."""
    model = SlabSeaiceModel(
        uniform_grid, SlabSeaiceParameters(initial_ice_thickness=0.1)
    )
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(uniform_grid.shape, -1.0 * ENERGY_PER_METRE)
    )

    stepped, _ = model.step(carry, coupling_time(0))

    np.testing.assert_allclose(np.asarray(stepped["state"].ice_thickness), 0.0)
    np.testing.assert_allclose(
        np.asarray(stepped["state"].ice_surface_temperature),
        constants.seawater_freezing_point_K,
    )


def test_land_cells_carry_no_ice(half_land_grid):
    """Only ocean cells are integrated; land reports the masked temperature."""
    model = SlabSeaiceModel(
        half_land_grid, SlabSeaiceParameters(initial_ice_thickness=1.0)
    )
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(half_land_grid.shape, ENERGY_PER_METRE)
    )
    stepped, _ = model.step(carry, coupling_time(0))

    land = np.asarray(half_land_grid.binary_mask) == 1.0
    thickness = np.asarray(stepped["state"].ice_thickness)
    temperature = np.asarray(stepped["state"].ice_surface_temperature)
    assert np.all(thickness[land] == 0.0)
    assert np.all(thickness[~land] > 0.0)
    np.testing.assert_allclose(temperature[land], MASKED_SURFACE_TEMPERATURE)


def test_invalid_parameters_are_rejected(uniform_grid):
    """A thickness scale of zero would make the closures undefined."""
    with pytest.raises(ValueError, match="min_ice_thickness"):
        SlabSeaiceModel(uniform_grid, SlabSeaiceParameters(min_ice_thickness=0.0))
    with pytest.raises(ValueError, match="ice_fraction_thickness_scale"):
        SlabSeaiceModel(
            uniform_grid, SlabSeaiceParameters(ice_fraction_thickness_scale=0.0)
        )
    with pytest.raises(ValueError, match="initial_ice_thickness"):
        SlabSeaiceModel(uniform_grid, SlabSeaiceParameters(initial_ice_thickness=-1.0))


def test_params_default_equivalence(uniform_grid):
    """Constructing with no params is constructing with the defaults."""
    implicit = SlabSeaiceModel(uniform_grid).initialize()
    explicit = SlabSeaiceModel(
        uniform_grid, SlabSeaiceParameters.default()
    ).initialize()

    assert jax.tree_util.tree_structure(implicit) == jax.tree_util.tree_structure(
        explicit
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_state_has_no_clock(uniform_grid):
    """State carries no time, and the step does not depend on the date.

    The sea-ice model integrates an energy that already has the coupling step
    folded into it (the ocean's ``frzmlt``), so unlike the ocean and land
    models it reads nothing from the clock at all -- which is only checkable
    now that the clock arrives as an argument rather than living in the state.
    """
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    assert "sim_time" not in carry["state"].asdict()

    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(uniform_grid.shape, 0.25 * ENERGY_PER_METRE)
    )
    first, _ = model.step(carry, coupling_time(0))
    later, _ = model.step(carry, coupling_time(182))

    np.testing.assert_array_equal(
        np.asarray(first["state"].ice_thickness),
        np.asarray(later["state"].ice_thickness),
    )


def test_step_shapes_and_dtypes_stable(uniform_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_step_is_differentiable(uniform_grid):
    """Reverse-mode gradients of one step agree with finite differences."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()

    def mean_ice_fraction(energy_scale):
        # One unit is the energy that freezes a metre of ice, so the ice
        # fraction responds at O(1) to a unit change.
        forced = carry["forcing"].replace(
            ice_frazil_melt_energy=jnp.full(
                uniform_grid.shape, ENERGY_PER_METRE * energy_scale
            )
        )
        stepped, _ = model.step({**carry, "forcing": forced}, coupling_time(0))
        return jnp.mean(stepped["derived"].ice_fraction)

    check_grads(
        mean_ice_fraction, (0.5,), order=1, modes=["rev"], eps=1e-3, atol=1e-3, rtol=1e-3
    )


def test_thickness_scale_is_differentiable(uniform_grid):
    """The ice-fraction closure's scale is a pytree leaf of the carry."""
    model = SlabSeaiceModel(uniform_grid)

    def mean_ice_fraction(scale):
        carry = model.initialize()
        carry["params"] = carry["params"].replace(ice_fraction_thickness_scale=scale)
        carry["forcing"] = carry["forcing"].replace(
            ice_frazil_melt_energy=jnp.full(uniform_grid.shape, ENERGY_PER_METRE)
        )
        carry, _ = model.step(carry, coupling_time(0))
        return jnp.mean(carry["derived"].ice_fraction)

    gradient = jax.grad(mean_ice_fraction)(jnp.float32(0.5))
    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 0.0
