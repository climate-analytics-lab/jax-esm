"""Gradients of a coupled run too large to differentiate in one piece.

``jax.grad`` of :meth:`Coupler.generate_trajectory_function` (with
``remat=True``) is the right tool while the coupled carry is small: the scan
stores one carry per coupling step on the device and recomputes the rest.
At high resolution that one carry per step is what does not fit. A T255
atmosphere coupled to a one-degree, 25-layer ocean carries about 2.7 GB, and
a twelve-day sensitivity with hourly coupling has 288 steps -- some 780 GB of
checkpoints, against an 80 GB GPU.

:func:`checkpointed_value_and_grad` computes the same gradient by two-level
checkpointing with the outer level held in host memory:

1. a forward run that copies the carry to the host at the start of every
   block of ``block_size`` steps, accumulating the objective;
2. a reverse sweep over the blocks: each block's starting carry is copied
   back to the device, the block is re-run forward keeping its
   ``block_size`` carries on the device, and the steps are differentiated one
   at a time, last first, each with :func:`jax.vjp` of one coupled step,
   chaining the cotangent of the carry back to the start.

Device memory is then ``block_size`` carries plus what one step's
:func:`jax.vjp` needs, whatever the length of the run; host memory is one
carry per block. The cost is about three forward runs and one reverse run.
Each step is one compiled function called repeatedly, so a long run compiles
exactly two programs (a step, and a step's vector-Jacobian product).

The objective is a weighted sum over steps of a scalar computed from each
step's output, ``J = sum_k weights[k] * objective(carry_k+1, diagnostics_k)``
-- a window mean, an accumulated total, or a single step's value are all of
that form -- and the gradient is with respect to the whole initial
:class:`~jem.base.component.CoupledCarry`, so the sensitivity to any
component's initial state *and* to any parameter in a component's
``carry["params"]`` comes out of the one reverse sweep.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def _is_inexact(leaf: Any) -> bool:
    return bool(jnp.issubdtype(jnp.result_type(leaf), jnp.inexact))


def checkpointed_value_and_grad(
    step: Callable[[Any], tuple[Any, Any]],
    objective: Callable[[Any, Any], jax.Array],
    carry: Any,
    weights: Sequence[float],
    *,
    block_size: int,
    offload: bool = True,
    on_block_gradient: Callable[[int, Any], None] | None = None,
) -> tuple[float, Any]:
    """Return ``J`` and ``dJ/d(carry)`` for a weighted sum over coupled steps.

    Parameters
    ----------
    step : callable
        One coupled step, ``carry -> (new_carry, diagnostics)`` --
        :meth:`jem.base.coupler.Coupler.generate_step_function`'s result. It
        is compiled here; do not pass it already jitted.
    objective : callable
        ``(new_carry, diagnostics) -> scalar``, evaluated after every step.
        Only what it reads of ``diagnostics`` is computed in the
        differentiated program, so a step's full diagnostics cost nothing.
    carry : CoupledCarry
        The initial carry (``coupler.initialize()``).
    weights : sequence of float
        One weight per step; the length is the number of steps run.
        Steps after the last non-zero weight cannot influence ``J`` and are
        not run at all.
    block_size : int
        Steps per block: the device holds this many carries at once during
        the reverse sweep, the host one per block.
    offload : bool
        Hold the block-start carries in host memory (the point of this
        function). ``False`` keeps them on the device, for a model small
        enough for that.
    on_block_gradient : callable, optional
        Called as ``on_block_gradient(k, gradient)`` once per block during the
        reverse sweep, with ``k`` the step at which the block starts and
        ``gradient`` the gradient of ``J`` with respect to the carry *at that
        step* -- the sensitivity to the state ``k`` steps into the run, which
        the reverse sweep passes through anyway. It is how the sensitivity's
        evolution with lead time is read off one sweep; copy out (to the
        host) only the parts needed, since the device memory is reused.

    Returns
    -------
    tuple
        ``(J, gradient)``. ``gradient`` has the structure of ``carry``;
        integer and boolean leaves (step counters, masks), which have no
        gradient, hold zeros of their own dtype.

    """
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1; got {block_size}")
    step_weights = np.asarray(weights, dtype=float)
    nonzero = np.nonzero(step_weights)[0]
    if nonzero.size == 0:
        raise ValueError("every weight is zero: the objective does not depend on the run")
    n_steps = int(nonzero[-1]) + 1

    # Everything below works on the carry's flat list of leaves rather than on
    # the carry pytree itself, for two reasons that both come from component
    # pytrees that do more than store their leaves when they are rebuilt.
    # Veros' `VerosVariables` casts every leaf back to its declared dtype with
    # `jnp.asarray` in `tree_unflatten`: a host copy made by
    # `jax.device_get(carry)` would be put straight back on the device (so
    # nothing would be offloaded), and a `float0` cotangent for an integer
    # leaf cannot be cast at all. So the host holds plain lists of numpy
    # leaves, and the step is differentiated with respect to the
    # floating-point leaves only, with the integer and boolean ones (step
    # counters, masks) passed through as constants.
    leaves, treedef = jax.tree.flatten(carry)
    inexact = [_is_inexact(leaf) for leaf in leaves]

    def merge(floats: Sequence[Any], others: Sequence[Any]) -> list[Any]:
        """Interleave the floating-point leaves with the rest, in carry order."""
        floats_iter, others_iter = iter(floats), iter(others)
        return [next(floats_iter) if keep else next(others_iter) for keep in inexact]

    def split(flat: Sequence[Any]) -> tuple[list[Any], list[Any]]:
        return ([leaf for leaf, keep in zip(flat, inexact) if keep],
                [leaf for leaf, keep in zip(flat, inexact) if not keep])

    def weighted_step(flat: list[Any], weight: jax.Array) -> tuple[list[Any], jax.Array]:
        new, diagnostics = step(jax.tree.unflatten(treedef, flat))
        return jax.tree.leaves(new), weight * objective(new, diagnostics)

    forward = jax.jit(weighted_step)

    @jax.jit
    def reverse(flat: list[Any], weight: jax.Array, carry_cotangent: list[Any]) -> list[Any]:
        floats, others = split(flat)

        def float_step(floats_: list[Any]) -> tuple[list[Any], jax.Array]:
            new_flat, contribution = weighted_step(merge(floats_, others), weight)
            return split(new_flat)[0], contribution

        _, vjp = jax.vjp(float_step, floats)
        (cotangent,) = vjp((carry_cotangent, jnp.ones((), dtype=jnp.result_type(weight))))
        return list(cotangent)

    def gradient_tree(floats: list[Any]) -> Any:
        """Rebuild a carry-shaped gradient, with zeros of their own dtype for the integer leaves."""
        return jax.tree.unflatten(treedef, merge(
            floats, [np.zeros(np.shape(leaf), dtype=jnp.result_type(leaf))
                     for leaf, keep in zip(leaves, inexact) if not keep]))

    store = jax.device_get if offload else (lambda flat: flat)
    starts = list(range(0, n_steps, block_size))

    t0 = time.perf_counter()
    block_carries: list[Any] = []
    value = 0.0
    current = leaves
    for k in range(n_steps):
        if k % block_size == 0:
            block_carries.append(store(current))
        current, contribution = forward(current, step_weights[k])
        value += float(contribution)
    logger.info("Forward run: %d steps, J = %.6g (%.0f s).", n_steps, value, time.perf_counter() - t0)

    t0 = time.perf_counter()
    cotangent = [jnp.zeros_like(leaf) for leaf in split(current)[0]]
    del current
    for b in reversed(range(len(starts))):
        steps = range(starts[b], min(starts[b] + block_size, n_steps))
        flat = jax.device_put(block_carries[b]) if offload else block_carries[b]
        block_carries[b] = None  # the host copy is no longer needed
        carries = [flat]
        for k in steps[:-1]:
            flat, _ = forward(flat, step_weights[k])
            carries.append(flat)
        for k in reversed(steps):
            cotangent = reverse(carries.pop(), step_weights[k], cotangent)
        if on_block_gradient is not None:
            on_block_gradient(starts[b], gradient_tree(cotangent))
        logger.debug("Reverse sweep: block %d of %d done.", len(starts) - b, len(starts))
    logger.info("Reverse sweep: %d blocks (%.0f s).", len(starts), time.perf_counter() - t0)
    return value, gradient_tree(cotangent)
