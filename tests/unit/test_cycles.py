"""Tests for the periodic interpolators in ``jem.utils.cycles``."""

import jax.numpy as jnp
import numpy as np
import pytest

from jem.utils.cycles import evaluate_cyclic_linear, evaluate_periodic


def test_evaluate_cyclic_linear_wraps_around():
    """Records at 0, 1/4, 1/2, 3/4 interpolate linearly and wrap from the last back to the first."""
    data = jnp.array([0.0, 1.0, 2.0, 3.0])
    assert float(evaluate_cyclic_linear(0.125, data)) == pytest.approx(0.5)
    # Halfway between record 3 (at 0.75) and record 0 (at 1.0 == 0.0).
    assert float(evaluate_cyclic_linear(0.875, data)) == pytest.approx(1.5)


def test_evaluate_periodic_matches_linear_interpolation_on_irregular_ticks():
    """``evaluate_periodic`` handles unevenly spaced records, including the wrap-around interval."""
    cx = pytest.importorskip("coordax")
    ticks = np.array([0.1, 0.3, 0.7, 0.9])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    field = cx.field(jnp.asarray(values), cx.LabeledAxis("time", ticks))
    # Inside the first interval: 0.2 sits halfway between 10 and 20.
    assert float(evaluate_periodic(0.2, field)) == pytest.approx(15.0)
    # Exactly on a tick returns that record.
    assert float(evaluate_periodic(0.7, field)) == pytest.approx(30.0)
    # Wrap-around interval from tick 0.9 to tick 0.1 + 1 (length 0.2), entered
    # from above the last tick: 0.95 is a quarter of the way along it.
    assert float(evaluate_periodic(0.95, field)) == pytest.approx(32.5)
    # ... and from below the first tick: 0.05 is three quarters along it.
    assert float(evaluate_periodic(0.05, field)) == pytest.approx(17.5)
    # x outside [0, 1) is wrapped first.
    assert float(evaluate_periodic(1.95, field)) == pytest.approx(32.5)
    assert float(evaluate_periodic(-0.95, field)) == pytest.approx(17.5)


def test_vmap_evaluate_periodic_matches_scalar_calls():
    """The vmapped variant agrees with scalar calls, so the function is trace-safe."""
    cx = pytest.importorskip("coordax")
    from jem.utils.cycles import vmap_evaluate_periodic

    ticks = np.array([0.1, 0.3, 0.7, 0.9])
    field = cx.field(jnp.array([10.0, 20.0, 30.0, 40.0]), cx.LabeledAxis("time", ticks))
    xs = jnp.array([0.2, 0.7, 0.95, 0.05])
    expected = [float(evaluate_periodic(float(x), field)) for x in xs]
    np.testing.assert_allclose(vmap_evaluate_periodic(xs, field, "time"), expected)
