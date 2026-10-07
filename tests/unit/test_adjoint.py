"""The checkpointed gradient is the gradient.

:func:`jem.adjoint.checkpointed_value_and_grad` re-runs blocks of a coupled
run and differentiates them a step at a time; none of that may change the
answer. These tests compare it, leaf for leaf, with :func:`jax.grad` of the
same objective computed over the whole run in one piece, on a real JCM
atmosphere (T21L5, the smallest SPEEDY runs) coupled to a slab ocean -- so the
cotangent crosses an exchanger and two components' carries every step, which
is where a mistake in chaining it would show.
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
from jcm.initial_states import jw_state
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

from jem.adjoint import checkpointed_value_and_grad
from jem.base.coupler import Coupler
from jem.components.jcm import JCMComponent
from jem.components.slab import SlabGrid, SlabOceanModel

START_DATE = jdt.to_datetime("2000-01-01")


def exchange(components, time):
    """Heat flux down, sea surface temperature up."""
    del time
    atmosphere, ocean = components["atm"], components["ocn"]
    return dict(
        components,
        atm=dict(atmosphere, forcing=atmosphere["forcing"].replace(
            sea_surface_temperature=ocean["state"].sea_surface_temperature)),
        ocn=dict(ocean, forcing=ocean["forcing"].replace(
            total_heat_flux=atmosphere["derived"].total_heat_flux)),
    )


def precipitation_in_a_box(new_carry, diagnostics):
    """Mean precipitation over a patch of the tropics, kg m-2 s-1."""
    del diagnostics
    precipitation = new_carry.components["atm"]["derived"].precipitation
    return jnp.mean(precipitation[10:20, 12:20])


@pytest.fixture(scope="module")
def coupler():
    """T21L5 SPEEDY aquaplanet over a slab ocean, coupled every six hours.

    The atmosphere starts from a moist Jablonowski-Williamson state rather
    than at rest, so it rains from the first step: an objective that is zero
    throughout would make every gradient zero and the comparison vacuous.
    """
    coords = get_speedy_coords(layers=5, spectral_truncation=21)
    model = Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                  start_time=START_DATE, time_step=60)
    atmosphere = JCMComponent(model, initial_state=jw_state(model, rh=0.8))
    atmosphere.set_exchanged_forcing(["sea_surface_temperature"])
    return Coupler(
        {"atm": atmosphere, "ocn": SlabOceanModel(SlabGrid.from_coords(coords.horizontal))},
        {"exchange": exchange},
        coupling_timestep=jdt.to_timedelta(6, "h"),
        start_date=START_DATE,
    )


@pytest.fixture(scope="module")
def reference(coupler):
    """``J`` and ``jax.grad`` of it over the whole five-step run at once."""
    step = coupler.generate_step_function()
    weights = (0.0, 0.25, 0.0, 0.5, 0.25)

    def objective(carry):
        total = 0.0
        for weight in weights:
            carry, diagnostics = step(carry)
            total = total + weight * precipitation_in_a_box(carry, diagnostics)
        return total

    initial = coupler.initialize()
    value, gradient = jax.jit(jax.value_and_grad(objective, allow_int=True))(initial)
    return initial, weights, float(value), gradient


def assert_gradients_match(actual, expected):
    for (path, a), e in zip(jax.tree_util.tree_leaves_with_path(actual), jax.tree.leaves(expected)):
        if e.dtype == jax.dtypes.float0:
            # An integer leaf: no gradient, returned as zeros of its own dtype.
            assert not np.any(np.asarray(a)), jax.tree_util.keystr(path)
            continue
        a, e = np.asarray(a), np.asarray(e)
        # Single precision, reassociated differently in the two programs and
        # accumulated over five steps: elements a thousandth of their leaf's
        # largest value carry a few per cent of rounding. A cotangent chained
        # wrongly is wrong at the scale of the leaf itself.
        scale = np.max(np.abs(e)) if e.size else 1.0
        np.testing.assert_allclose(a, e, rtol=1e-3, atol=1e-3 * scale,
                                   err_msg=jax.tree_util.keystr(path))


@pytest.mark.parametrize("block_size", [1, 2, 5])
@pytest.mark.parametrize("offload", [True, False])
def test_checkpointed_gradient_equals_the_one_piece_gradient(coupler, reference, block_size, offload):
    """Any block size, on host or device, gives ``jax.grad``'s answer."""
    initial, weights, expected_value, expected_gradient = reference
    assert expected_value > 0  # it rains in the box
    value, gradient = checkpointed_value_and_grad(
        coupler.generate_step_function(), precipitation_in_a_box, initial, weights,
        block_size=block_size, offload=offload,
    )
    assert value == pytest.approx(expected_value, rel=1e-5)
    assert_gradients_match(gradient, expected_gradient)
    # The sensitivity this exists for -- to the initial ocean -- is not zero.
    sst_sensitivity = gradient.components["ocn"]["state"].sea_surface_temperature
    assert np.any(np.asarray(sst_sensitivity) != 0)


def test_trailing_zero_weights_are_not_run(coupler, reference):
    """Steps after the last non-zero weight cannot change ``J``: same answer."""
    initial, weights, expected_value, expected_gradient = reference
    value, gradient = checkpointed_value_and_grad(
        coupler.generate_step_function(), precipitation_in_a_box, initial,
        (*weights, 0.0, 0.0), block_size=2,
    )
    assert value == pytest.approx(expected_value, rel=1e-5)
    assert_gradients_match(gradient, expected_gradient)


@pytest.mark.parametrize("weights, block_size", [((0.0, 0.0), 1), ((1.0,), 0)])
def test_rejects_an_objective_or_block_that_cannot_work(coupler, weights, block_size):
    """All-zero weights and an empty block are errors, not a silent zero."""
    with pytest.raises(ValueError):
        checkpointed_value_and_grad(
            coupler.generate_step_function(), precipitation_in_a_box,
            coupler.initialize(), weights, block_size=block_size,
        )
