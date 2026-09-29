"""How far a wrapped model's own clock may drift from the coupler's.

A component that wraps an externally-developed model (JCM, Veros) lets that
model keep the counter of elapsed simulated time it advances itself. The
coupler owns the run's clock, so the two counters are redundant -- and that
redundancy is the one cheap check that a carry belongs to this run: a
checkpoint restored into a coupler with a different start date, a restart
state paired with the wrong ``CoupledCarry.step``, or a carry threaded into
the wrong component all show up as a model clock that has parted from the
coupler's.

The tolerance for that comparison lives here, rather than in either wrapper,
so the two cannot answer the question differently and a third wrapper has one
definition to reach for. What a wrapper does with the answer stays with the
wrapper: what a drift means for the run, and what has to be said about it, is
specific to the model being wrapped.
"""

from __future__ import annotations

import numpy as np

# Floor on how far a component's own clock may drift from the coupler's
# before it is reported. One second is below the shortest plausible timestep
# of any model JEM couples, so any real disagreement (a checkpoint from
# another run, a carry threaded into the wrong component) is orders of
# magnitude larger than this.
CLOCK_TOLERANCE_SECONDS = 1.0

# ... but a fixed floor cannot hold for a long run. Both clocks are float32
# unless `jax_enable_x64` is on, and a float32's spacing at 3e9 s (a century
# of simulated time) is 256 s, so the two counters differ by hundreds of
# seconds from rounding alone -- an ERROR line per step reporting nothing but
# arithmetic. The tolerance therefore also scales with the magnitude of the
# time being compared, at a few float32 ulps: large enough that accumulated
# rounding never trips it, and far below the interval (one coupling step at
# least) any genuine mismatch is off by.
CLOCK_TOLERANCE_FLOAT32_ULPS = 8.0


def clock_tolerance_seconds(sim_time: float) -> float:
    """Return the drift, in seconds, tolerated at simulation time ``sim_time``.

    Parameters
    ----------
    sim_time : float
        The coupler's simulation time in seconds, i.e. the magnitude at which
        the two clocks are being compared.

    Returns
    -------
    float
        ``max(CLOCK_TOLERANCE_SECONDS, CLOCK_TOLERANCE_FLOAT32_ULPS * eps32 *
        |sim_time|)`` -- the constant floor for short runs, growing with the
        float32 resolution of the clock for long ones.

    """
    relative = (
        CLOCK_TOLERANCE_FLOAT32_ULPS
        * float(np.finfo(np.float32).eps)
        * abs(float(sim_time))
    )
    return max(CLOCK_TOLERANCE_SECONDS, relative)
