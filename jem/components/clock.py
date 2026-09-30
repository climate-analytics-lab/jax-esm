"""How far a wrapped model's own clock may drift from the coupler's.

A component that wraps an externally-developed model (Veros) lets that
model keep the counter of elapsed simulated time it advances itself. The
coupler owns the run's clock -- a calendar datetime, and so exact -- so the
two clocks are redundant, and that redundancy is the one cheap check that a
carry belongs to this run: a checkpoint restored into a coupler with a
different start date, a restart state paired with the wrong coupled ``time``,
or a carry threaded into the wrong component all show up as a model clock that
has parted from the coupler's. (The JCM wrapper's own clock is a datetime too,
so it compares exactly and needs no tolerance.)

The tolerance for that comparison lives here, rather than in a wrapper, so a
second wrapper around a model with a float clock has one definition to reach
for and cannot answer the question differently. What a wrapper does with the answer stays with the
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

# ... but a fixed floor cannot hold for a wrapped model whose own clock is a
# float32 count of seconds. The coupler's side of the comparison is exact (it
# is taken from the datetime clock as whole days and seconds), so the drift is
# the model counter's own rounding: a float32's spacing at 3e9 s (a century
# of simulated time) is 256 s, an ERROR line per step reporting nothing but
# arithmetic. The tolerance therefore also scales with the magnitude of the
# time being compared, at a few float32 ulps: large enough that the counter's
# rounding never trips it, and far below the interval (one coupling step at
# least) any genuine mismatch is off by. A model whose counter is float64
# (Veros' default) is held to the same, looser, bound, which is still orders
# of magnitude below a genuine mismatch.
CLOCK_TOLERANCE_FLOAT32_ULPS = 8.0


def clock_tolerance_seconds(elapsed_seconds: float) -> float:
    """Return the drift, in seconds, tolerated after ``elapsed_seconds`` of a run.

    Parameters
    ----------
    elapsed_seconds : float
        Seconds the coupler's clock has advanced since the run's start date,
        i.e. the magnitude at which the two clocks are being compared.

    Returns
    -------
    float
        ``max(CLOCK_TOLERANCE_SECONDS, CLOCK_TOLERANCE_FLOAT32_ULPS * eps32 *
        |elapsed_seconds|)`` -- the constant floor for short runs, growing
        with the float32 resolution of the clock for long ones.

    """
    relative = (
        CLOCK_TOLERANCE_FLOAT32_ULPS
        * float(np.finfo(np.float32).eps)
        * abs(float(elapsed_seconds))
    )
    return max(CLOCK_TOLERANCE_SECONDS, relative)
