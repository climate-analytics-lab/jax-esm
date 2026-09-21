"""Tests for the shared clock-drift tolerance (:mod:`jem.components.clock`).

The tolerance is one definition used by every wrapper that compares an
external model's own clock with the coupler's, so it is tested here rather
than with any one of them.
"""

import pytest

from jem.components.clock import CLOCK_TOLERANCE_SECONDS, clock_tolerance_seconds


def test_clock_tolerance_floor_applies_to_a_young_run():
    """Early in a run the tolerance is the fixed floor, well under one step."""
    # 8 float32 ulps of one day is ~0.08 s, so the floor wins.
    assert clock_tolerance_seconds(86400.0) == CLOCK_TOLERANCE_SECONDS
    assert clock_tolerance_seconds(0.0) == CLOCK_TOLERANCE_SECONDS


def test_clock_tolerance_grows_with_float32_resolution():
    """After decades the tolerance follows float32's spacing, by a hand value.

    At 3e9 s the float32 eps is 1.1920928955078125e-07, so eight ulps of the
    simulation time is 8 * 1.1920928955078125e-07 * 3e9 = 2861.02294921875 s
    -- above the ~256 s spacing of a float32 there, and so above the drift
    that rounding alone can produce, while still far below the one-day
    coupling step a real mismatch would be off by.
    """
    assert clock_tolerance_seconds(3.0e9) == pytest.approx(2861.02294921875)
    # Sign of the simulation time cannot shrink the window.
    assert clock_tolerance_seconds(-3.0e9) == clock_tolerance_seconds(3.0e9)
