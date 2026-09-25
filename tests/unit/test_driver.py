"""Tests for :mod:`jem.driver` -- the chunked run loop.

Almost everything here runs on a two-slab coupler with **no atmosphere**: a
4x3 grid, an ocean and a sea ice, coupled by the default wiring. That is
deliberate, and not only because it is fast. The loop's job is arithmetic on
the coupled step counter -- how many chunks, where each one starts, what is
left after a resume -- and a toy coupler makes the answers exact, so
``test_continuous_chunked_resumed_agree`` can insist on 1e-12 rather than on
"close enough for an atmosphere". The same comparison with the real
atmosphere is the slow test at the bottom, which is what proves the toy is
not the only thing this works on.
"""

import logging
import pathlib
import shutil

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr

from jem.accumulate import monthly_mean
from jem.base.coupler import Coupler
from jem.checkpoint import CARRY_FILENAME
from jem.components.slab import SlabOceanModel, SlabSeaiceModel
from jem.driver import RunResult, default_health_check, run_chunked
from jem.exchangers import default_exchangers
from tests.unit.slab_test_utils import make_grid

START_DATE = jdt.to_datetime("2001-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")


def two_slabs() -> Coupler:
    """Return an ocean and a sea ice on the 4x3 grid, with the default wiring."""
    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    return Coupler(
        components,
        default_exchangers(components),
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )


def doubly_nested_coupler() -> Coupler:
    """Return a doubly nested 24x6x5 toy coupler with three compounding rates.

    Three nested rates compound into one raw counter: an ``atm`` sub-stepped
    5 times inside a 10-minute-inside-1-hour nesting (rate 6), itself nested
    24x inside an outer daily coupler -- so the fastest raw counter anywhere
    in the hierarchy advances ``24 * 6 * 5 = 720`` times per outer coupled
    step. Built on the toy ``Counter``/exchanger fixtures
    :mod:`tests.unit.test_nested_coupler` already defines for exactly this
    kind of test, so a resume bug in
    ``_check_step_counters_fit_int32`` (which only shows up once
    ``first_step`` is non-zero) has a coupler with a small enough raw-counter
    limit to name a resume point inside it without an astronomical run.
    """
    from tests.unit.test_nested_coupler import (
        Counter,
        atm_lnd_exchange,
        srf_ocn_exchange_nested,
    )

    ten_minutes = jdt.to_timedelta(10, "minute")
    hour = jdt.to_timedelta(1, "hour")
    day = jdt.to_timedelta(1, "day")
    inner = Coupler(
        {"atm": Counter("atm"), "lnd": Counter("lnd")},
        {"atm_lnd_exchange": atm_lnd_exchange},
        coupling_timestep=ten_minutes,
        start_date=START_DATE,
        name="atm_lnd",
        workflow=["atm_lnd_exchange"] + ["atm"] * 5 + ["lnd"],
    )
    mid = Coupler(
        {"atm_lnd": inner, "ocn": Counter("ocn")},
        {"srf_ocn_exchange": srf_ocn_exchange_nested},
        coupling_timestep=hour,
        start_date=START_DATE,
        name="mid",
        workflow=["srf_ocn_exchange", "atm_lnd", "ocn"],
    )
    return Coupler(
        {"mid": mid, "sea": Counter("sea")},
        {},
        coupling_timestep=day,
        start_date=START_DATE,
        workflow=["mid", "sea"],
    )


@pytest.fixture
def coupler() -> Coupler:
    return two_slabs()


def carry_leaves(carry):
    """Return the carry's leaves as numpy, for an exact comparison."""
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(carry)]


def assert_carries_agree(left, right, atol):
    """Assert two carries hold the same numbers to ``atol``, leaf by leaf."""
    left_leaves, right_leaves = carry_leaves(left), carry_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for index, (one, other) in enumerate(zip(left_leaves, right_leaves)):
        np.testing.assert_allclose(one, other, atol=atol, rtol=0, err_msg=f"leaf {index}")


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


def test_run_chunked_python_api(coupler, tmp_path):
    """Four days in two chunks: two files per component, and the clock advanced."""
    result = run_chunked(
        coupler, total_time="4 days", chunk="2 days", output_dir=tmp_path
    )

    assert isinstance(result, RunResult)
    assert result.completed
    assert result.steps_completed == 4
    assert int(result.final_carry.step) == 4

    # One file per component per chunk, named after the coupled step each
    # chunk starts at: steps 0 and 2 of a four-step run in two-day chunks.
    names = sorted(path.name for path in result.paths)
    assert names == [
        "ocn-00000000.nc", "ocn-00000002.nc",
        "seaice-00000000.nc", "seaice-00000002.nc",
    ]
    assert all(path.exists() for path in result.paths)

    # The default health check has no atmosphere to look at, so it abstains --
    # once per chunk, and without stopping the run.
    assert [report["skipped"] for report in result.reports] == ["no atmosphere"] * 2
    assert [report["chunk"] for report in result.reports] == [0, 1]

    # Each chunk is labelled with its own dates rather than the first chunk's.
    first = xr.open_dataset(tmp_path / "ocn-00000000.nc")
    second = xr.open_dataset(tmp_path / "ocn-00000002.nc")
    assert second["time"].values[0] > first["time"].values[-1]


def test_run_chunked_accepts_days_as_numbers(coupler, tmp_path):
    """A duration may be a number of days as well as a string."""
    result = run_chunked(coupler, total_time=2, chunk=1, output_dir=tmp_path)
    assert result.steps_completed == 2
    assert len(result.paths) == 4


def test_run_chunked_rejects_partial_chunk(coupler, tmp_path):
    """A run that is not a whole number of chunks is refused, naming both."""
    with pytest.raises(ValueError, match="not a whole number of chunks"):
        run_chunked(
            coupler, total_time="5 days", chunk="2 days", output_dir=tmp_path
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_time": "10 days", "chunk": "12 hours"}, "chunk="),
        ({"total_time": "36 hours", "chunk": "1 day"}, "total_time="),
    ],
)
def test_run_chunked_rejects_a_duration_that_is_not_whole_steps(
    coupler, tmp_path, kwargs, message
):
    """Both durations must be whole multiples of the coupling timestep."""
    with pytest.raises(ValueError, match=message):
        run_chunked(coupler, output_dir=tmp_path, **kwargs)


@pytest.mark.parametrize("subsample", [0, -1, 1.0, True])
def test_run_chunked_rejects_a_bad_subsample_before_it_integrates(
    coupler, tmp_path, subsample
):
    """`subsample` is checked up front, not after a chunk has been integrated.

    It is only read once a chunk's output exists, so a run configured with a
    nonsensical stride would otherwise compile a trajectory and integrate a
    whole chunk -- for an atmosphere, hours -- before raising. The trajectory
    factory is replaced with one that explodes, so the test fails if the
    check ever moves back behind it.
    """
    def must_not_be_called(iterations):
        raise AssertionError(
            f"a {iterations}-step trajectory was built despite subsample="
            f"{subsample!r}"
        )

    coupler.generate_trajectory_function = must_not_be_called
    with pytest.raises(ValueError, match="subsample must be a positive integer"):
        run_chunked(
            coupler, total_time="2 days", chunk="2 days",
            output_dir=tmp_path, subsample=subsample,
        )
    assert list(tmp_path.iterdir()) == []


def test_max_element_rate_is_one_with_no_multiplicity_or_nesting(coupler):
    """The floor: nothing runs faster than the coupled step itself."""
    from jem.driver import _max_element_rate

    assert _max_element_rate(coupler) == 1


def test_max_element_rate_follows_a_sub_stepped_element():
    """An element run `n` times a coupled step counts `n` times as fast."""
    from jem.driver import _max_element_rate

    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    exchangers = default_exchangers(components)
    coupler = Coupler(
        components, exchangers, coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=[list(exchangers), ["ocn"] * 24, "seaice"],
    )
    assert _max_element_rate(coupler) == 24


class _FakeJCMLikeComponent:
    """A minimal component with its own internal, sub-cycled counter.

    Stands in for JCM's real ``JCMComponent`` (``self._inner_steps()``,
    ``RunState.step``): a component with an internal timestep of its own,
    faster than the coupled step calling it, entirely invisible to
    ``jem.driver`` unless it reports it via ``internal_steps_per_call``
    (:class:`~jem.base.component.SupportsInternalStepping`). Used here rather
    than a real ``JCMComponent`` so the boundary tests below stay fast and
    exact -- they are about the arithmetic ``jem.driver`` does with the rate
    a component reports, not about jax-gcm itself (that is
    ``tests/unit/test_jcm_component.py``'s job).
    """

    def __init__(self, internal_rate: int, name: str = "atm", starting_counter: int = 0):
        """Name the component, fix the internal rate, and seed its own counter.

        ``starting_counter`` stands in for a component whose own internal
        counter does not start at zero when a run begins -- a model
        integrated before it was wrapped or registered (Veros), or a
        resumed run's carry (JCM, Veros, and this fake alike).
        """
        self.name = name
        self._internal_rate = internal_rate
        self._starting_counter = starting_counter

    def initialize(self):
        return {"value": jnp.float32(0.0)}

    def step(self, carry, time):
        del time
        return {"value": carry["value"] + 1.0}, {"value": carry["value"]}

    def internal_steps_per_call(self) -> int:
        """Report the internal rate, exactly as ``JCMComponent._inner_steps`` does."""
        return self._internal_rate

    def internal_counter(self, carry) -> int:
        """Report this fake's own seeded starting counter, ignoring ``carry``.

        A real component (JCM, Veros) reads its counter from its own carry
        (``carry["step"]``, ``carry["state"].variables.itt``); this fake's
        own carry (``{"value": ...}``) has no such field, so the counter it
        reports is fixed at construction instead, which is enough to
        exercise ``jem.driver``'s arithmetic without needing a real one.
        """
        del carry
        return self._starting_counter


def test_max_element_rate_includes_a_components_own_internal_stepping_rate():
    """A component may opt in to reporting its own internal stepping rate.

    A component that implements ``SupportsInternalStepping`` reports how many
    of its own internal timesteps happen inside one ``step()`` call; a plain
    (non-nested, multiplicity-1) component contributing that rate directly is
    the base case nesting and multiplicity build on.
    """
    from jem.driver import _max_element_rate

    component = _FakeJCMLikeComponent(48)
    coupler = Coupler(
        {"atm": component}, {}, coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    assert _max_element_rate(coupler) == 48


def test_max_element_rate_multiplies_internal_stepping_by_workflow_multiplicity():
    """A component sub-stepped by the workflow AND internally is both, multiplied.

    An element called ``m`` times by the workflow, each call itself making
    ``k`` internal steps, advances its own internal counter ``m * k`` times
    per outer coupled step -- the same composition multiplicity and nesting
    already get, just with a component's own reported rate as one more
    factor.
    """
    from jem.driver import _max_element_rate

    component = _FakeJCMLikeComponent(48)
    coupler = Coupler(
        {"atm": component}, {}, coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE, workflow=["atm"] * 3,
    )
    assert _max_element_rate(coupler) == 3 * 48


def test_max_element_rate_defaults_to_one_for_a_component_with_no_internal_stepping():
    """A component that does not implement the capability is assumed rate 1.

    The same as every component before this capability existed (e.g.
    ``SlabOceanModel``, which has no internal timestep of its own to report).
    """
    from jem.driver import _max_element_rate

    coupler = Coupler(
        {"atm": _FakeJCMLikeComponent(48, name="atm"), "ocn": SlabOceanModel(make_grid())},
        {}, coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE,
    )
    # Dominated by "atm"'s internal rate, but "ocn" (no `internal_steps_per_call`)
    # must not raise or silently count as anything other than 1.
    assert _max_element_rate(coupler) == 48


def test_check_step_counters_accepts_exactly_at_and_refuses_one_past_a_components_internal_limit(
    tmp_path,
):
    """A JCM-like component's own internal rate is refused exactly at its boundary.

    Without a way for `_max_element_rate` to see a component's own internal
    counter, a run long enough to overflow ``time.step * internal_rate``
    (JCM's own ``expected_step``, computed in
    ``_report_authoritative_clock_drift``) would never be refused by
    `run_chunked`. This is checked directly against
    `_check_step_counters_fit_int32` (not the whole of `run_chunked`, which
    would then have to build and run a multi-million-step trajectory) --
    exactly the pattern `test_run_chunked_accepts_a_run_at_exactly_the_day_count_limit`
    already uses for the coupler-hierarchy-only case.
    """
    from jem.driver import _check_step_counters_fit_int32, _max_safe_coupled_steps

    # A large internal rate, like the sub-stepped-element test above, so the
    # limit is small enough to name in this test's own assertions.
    component = _FakeJCMLikeComponent(2880)
    coupler = Coupler(
        {"atm": component}, {}, coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    limit = _max_safe_coupled_steps(coupler)
    assert limit < 10**7  # sanity: the internal-rate limit, not the day one

    carries = {"atm": component.initialize()}
    _check_step_counters_fit_int32(coupler, 0, limit + 1, carries)  # last step == limit: fine
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        _check_step_counters_fit_int32(coupler, 0, limit + 2, carries)  # last step == limit + 1

    def must_not_be_called(iterations, **kwargs):
        raise AssertionError(
            f"a {iterations}-step trajectory was built for a run past a "
            "component's own internal-stepping limit"
        )

    coupler.generate_trajectory_function = must_not_be_called
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        run_chunked(
            coupler, total_time=f"{limit + 2} days", chunk=f"{limit + 2} days",
            output_dir=tmp_path,
        )


def test_check_step_counters_refuses_a_run_that_wraps_a_components_own_starting_counter():
    """A component's own internal counter is refused exactly where IT would wrap.

    `_max_element_rate`'s own rate assumes an internal counter starts at
    zero and stays in lockstep with the coupled step -- true only until a
    component's own counter starts somewhere else, which its rate alone
    cannot know. Here the fake's counter is already close to int32's own
    range before this run even starts (standing in for a `VerosComponent`
    wrapping a model that was integrated before it was bound -- see
    `VerosComponent.bind`'s own docstring -- or any component's carry coming
    from elsewhere), so the OLD, rate-only check would have accepted a run
    this one refuses.
    """
    from jem.driver import _check_step_counters_fit_int32

    rate = 24
    # Chosen so the boundary is exact: `starting_counter + 10 * rate ==
    # 2**31 - 1` precisely, with no remainder to obscure the "one step past
    # is refused" edge.
    starting_counter = 2**31 - 1 - 10 * rate
    component = _FakeJCMLikeComponent(rate, starting_counter=starting_counter)
    coupler = Coupler(
        {"atm": component}, {}, coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    carries = {"atm": component.initialize()}

    _check_step_counters_fit_int32(coupler, 0, 10, carries)  # counter reaches 2**31 - 1: fine
    with pytest.raises(ValueError, match="own internal counter"):
        _check_step_counters_fit_int32(coupler, 0, 11, carries)  # one step past


def test_check_step_counters_accepts_a_resume_whose_counter_matches_first_step_times_rate():
    """An ordinary resume's counter is exactly what its own rate predicts.

    A resumed run's carry already holds an advanced internal counter -- the
    normal case, not the pre-stepped-before-binding one above -- and the new
    per-component check must accept it exactly as the old, rate-only check
    did: this is `total_steps` being absolute, not relative to `first_step`
    (the same distinction the coupler-level check already makes), now
    checked for a component's own counter too, so it must not be
    double-counted here either.
    """
    from jem.driver import _check_step_counters_fit_int32

    rate = 48
    first_step = 1_000
    # Exactly what an uninterrupted run from step 0 would have left this
    # component's own counter at by the time it reached `first_step`.
    component = _FakeJCMLikeComponent(rate, starting_counter=first_step * rate)
    coupler = Coupler(
        {"atm": component}, {}, coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )
    carries = {"atm": component.initialize()}

    _check_step_counters_fit_int32(coupler, first_step, first_step + 10, carries)


def test_run_chunked_refuses_a_run_past_a_sub_stepped_elements_own_counter(
    tmp_path,
):
    """A sub-stepped element's own `step*multiplicity+call` overflows first.

    `_check_step_counters_fit_int32` is checked before any trajectory is
    built (the trajectory factory is replaced with one that explodes, as
    `test_run_chunked_rejects_a_bad_subsample_before_it_integrates` does for
    `subsample`), so this is a real up-front refusal, not one discovered
    partway through a run that happened to be interrupted.
    """
    from jem.driver import _max_safe_coupled_steps

    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    exchangers = default_exchangers(components)
    # 2880 divides the 86400 s coupling timestep exactly (30 s sub-steps),
    # and is large enough that the resulting limit is reached at a coupled
    # step count small enough to name in this test's own assertions, not
    # (like the calendar's own day-count limit) in the millions of years.
    coupler = Coupler(
        components, exchangers, coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        workflow=[list(exchangers), ["ocn"] * 2880, "seaice"],
    )
    limit = _max_safe_coupled_steps(coupler)
    assert limit < 10**7  # sanity: this is the sub-step limit, not the day one

    def must_not_be_called(iterations, **kwargs):
        raise AssertionError(
            f"a {iterations}-step trajectory was built for a run past the "
            "sub-stepped element's own counter limit"
        )

    coupler.generate_trajectory_function = must_not_be_called
    # `total_time` of `limit + 1` days reaches coupled step `limit` exactly
    # (steps `0` through `limit`, `limit + 1` of them) -- still within
    # bounds; `limit + 2` reaches `limit + 1`, one past it.
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        run_chunked(
            coupler, total_time=f"{limit + 2} days", chunk=f"{limit + 2} days",
            output_dir=tmp_path,
        )


def test_run_chunked_refuses_a_run_past_the_gregorian_day_count_limit(
    coupler, tmp_path,
):
    """The int32 day-count limit of ``gregorian`` calendar math is checked too.

    Without this check, nothing outside `jem.accumulate._midpoint_month_rule`
    (which only ever runs on the fixed calendars) would check this at all,
    so a ``"gregorian"`` run past it -- which `gregorian_instant` is exact
    for up to about 5.87 million years, but not beyond, an inherent int32
    limit no algorithm can move -- would silently derive a wrong seasonal
    phase (`CouplingTime.year_fraction`) with no error. This coupler has no
    sub-stepped element, so `_max_safe_coupled_steps` here is exactly
    `jem.base.calendar.max_safe_record` of its own coupling timestep and
    start date, checked at `offset_seconds=dt_seconds` (conservative enough
    to cover every offset within one record a caller downstream -- a
    midpoint or end label -- actually uses) -- checked directly, so this
    test does not have to construct (or wait out) a multi-million-year run
    to prove the refusal fires exactly where that limit is.
    """
    from jem.base.calendar import max_safe_record
    from jem.driver import _max_safe_coupled_steps

    start = coupler.start_date
    dt_seconds = int(round(coupler.dt_seconds))
    expected = max_safe_record(
        dt_seconds, offset_seconds=dt_seconds,
        start_seconds=int(start.delta.seconds),
        start_days=int(start.delta.days),
    )
    limit = _max_safe_coupled_steps(coupler)
    assert limit == expected
    assert limit > 5 * 10**8  # sanity: millions of years, not a small bound

    def must_not_be_called(iterations, **kwargs):
        raise AssertionError(
            f"a {iterations}-step trajectory was built for a run past the "
            "day-count limit"
        )

    coupler.generate_trajectory_function = must_not_be_called
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        run_chunked(
            coupler, total_time=f"{limit + 2} days", chunk=f"{limit + 2} days",
            output_dir=tmp_path,
        )


def test_max_safe_coupled_steps_day_limit_covers_an_end_of_interval_offset(coupler):
    """The day limit must cover more than offset 0.

    Checking the day-count limit only at ``offset_seconds = 0`` -- the
    record's own START -- would miss that `jem.accumulate`'s gregorian
    monthly-mean rules bin at the record's MIDPOINT
    (``offset_seconds=dt_seconds // 2``), and a caller is free to label at
    the record's END too (``offset_seconds=dt_seconds``). Since
    `max_safe_record`'s bound only ever shrinks as ``offset_seconds`` grows,
    a check at offset 0 would accept a coupled-step count that a midpoint-
    or end-labelled caller downstream could silently get wrong: this
    coupler's own last-safe-at-offset-0 record already wraps to a negative
    day count once `gregorian_instant` is asked for its END instead
    (``offset_seconds=dt_seconds``).
    """
    from jem.base.calendar import gregorian_instant, max_safe_record
    from jem.driver import _max_safe_coupled_steps

    start = coupler.start_date
    dt_seconds = int(round(coupler.dt_seconds))
    start_seconds = int(start.delta.seconds)
    start_days = int(start.delta.days)
    offset_0_bound = max_safe_record(
        dt_seconds, start_seconds=start_seconds, start_days=start_days,
    )
    # The reproduction: at offset 0's own bound, the END of that same
    # interval (one full `dt_seconds` later) already wraps.
    end_days, _ = gregorian_instant(
        jnp.int32(offset_0_bound), dt_seconds, start_days, start_seconds,
        offset_seconds=dt_seconds,
    )
    assert int(end_days) < 0  # wrapped -- confirms this is a genuine gap

    limit = _max_safe_coupled_steps(coupler)
    assert limit < offset_0_bound  # the fix must be strictly more conservative
    # Checking at `offset_seconds=dt_seconds` (a full record later than record
    # 0's own start) is exactly one record's worth stricter here, since
    # `max_safe_record`'s bound moves by a whole record for a whole
    # `record_seconds` of extra offset.
    end_of_interval_bound = max_safe_record(
        dt_seconds, offset_seconds=dt_seconds,
        start_seconds=start_seconds, start_days=start_days,
    )
    assert limit == end_of_interval_bound


def test_run_chunked_accepts_a_run_at_exactly_the_day_count_limit(coupler):
    """The refusal's own boundary is exact: reaching `limit` itself is not refused.

    ``_check_step_counters_fit_int32(coupler, first_step, total_steps, carries)``
    checks the LAST coupled step the run would reach, ``total_steps - 1`` --
    ``total_steps`` is the run's own ABSOLUTE target step count (see the
    function's own docstring), so it alone (not ``first_step``) says how far
    the run goes. This only exercises the check directly (not the whole of
    `run_chunked`, which would then have to build and run a multi-million-step
    trajectory) -- the boundary is what is under test, not the run.
    """
    from jem.driver import _check_step_counters_fit_int32, _max_safe_coupled_steps

    limit = _max_safe_coupled_steps(coupler)
    # No `SupportsInternalStepping` component here, so `carries`' own content
    # feeds no counter check -- it still has to be a real starting carry
    # (structurally, for the nested-coupler walk), not an empty placeholder.
    carries = coupler.initialize().components
    _check_step_counters_fit_int32(coupler, 0, limit + 1, carries)  # last step == limit: fine
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        _check_step_counters_fit_int32(coupler, 0, limit + 2, carries)  # last step == limit + 1
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        # A resumed run whose absolute target is past the limit: `total_steps`
        # is the run's absolute target step count (see `remaining_batches`'s
        # own docstring), so a resume at `limit` itself plus `limit + 2` more
        # is expressed as a target of `limit + 2`, not `limit + 2` on its own.
        _check_step_counters_fit_int32(coupler, limit, limit + 2, carries)


def test_check_step_counters_accepts_a_realistic_resume_of_a_deeply_nested_coupler():
    """`total_steps` is absolute, not relative to `first_step`.

    `run_chunked` passes `_check_step_counters_fit_int32` the same
    `total_steps` it passes `remaining_batches` -- the ABSOLUTE coupled-step
    count the *whole run* is asked to reach, never a count of steps still to
    integrate from `first_step` (`remaining_batches`'s own docstring: "Coupled
    steps the whole run is asked for"). Computing
    ``last_step = first_step + total_steps - 1`` instead would silently
    double-count `first_step` -- refusing a real, legitimate 6000-year run
    of the doubly nested 24x6x5 coupler above, resumed at coupled step
    1,000,000, even though its true last step (`total_steps - 1`, about
    2.19 million) sits well inside the raw-counter limit (about 2.98
    million here, since `rate = 720`).
    """
    from jem.driver import _check_step_counters_fit_int32, _max_safe_coupled_steps

    coupler = doubly_nested_coupler()
    limit = _max_safe_coupled_steps(coupler)
    total_steps = 2_191_455  # ~6000 years of daily coupled steps
    assert total_steps - 1 <= limit  # sanity: this run is genuinely in bounds

    # No `SupportsInternalStepping` component in this coupler, so `carries`
    # feeds no counter check -- it still has to be a real starting carry
    # (structurally, for the nested-coupler walk), not an empty placeholder.
    carries = coupler.initialize().components

    # Resumed at coupled step 1,000,000: legitimate, must be accepted.
    _check_step_counters_fit_int32(coupler, 1_000_000, total_steps, carries)

    # One coupled step past the limit -- an absolute target one past `limit`
    # -- is still refused, resumed or not.
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        _check_step_counters_fit_int32(coupler, 1_000_000, limit + 2, carries)


def test_max_safe_coupled_steps_leaves_room_for_carry_steps_own_post_increment():
    """`carry.step` itself must survive its own +1.

    A coupled step's own counter is incremented and persisted AFTER it runs
    (`Coupler.step`'s own body: ``step=carry.step + 1``), so the largest safe
    coupled step to reach is not simply the largest one a raw counter can be
    COMPUTED at -- it is one less than that, so the resulting ``carry.step``
    (the computed step's index plus one) is itself still representable.
    ``"365_day"`` has no calendar-derived (day-count) limit at all, so with
    no sub-stepped element (``rate == 1``) taking its raw-counter limit as
    exactly ``2**31 - 1`` would be wrong: reaching that coupled step is fine
    on its own, but the ``carry.step`` this run would then persist,
    ``2**31``, silently wraps -- an int32 cannot hold it.
    """
    from jem.driver import _max_element_rate, _max_safe_coupled_steps

    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    coupler = Coupler(
        components, default_exchangers(components), coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE, calendar="365_day",
    )
    assert _max_element_rate(coupler) == 1  # no sub-stepping: the floor case

    limit = _max_safe_coupled_steps(coupler)
    assert limit == 2**31 - 2  # one short of int32's own max, not the max itself
    assert limit + 1 <= 2**31 - 1  # the post-increment `carry.step` still fits


def test_run_chunked_writes_nothing_when_the_run_is_already_done(coupler, tmp_path):
    """A carry already at `total_time` completes with no chunks and no files."""
    carry = coupler.initialize()
    run = coupler.generate_trajectory_function(3)
    carry, _ = run(carry)

    result = run_chunked(
        coupler,
        total_time="3 days",
        chunk="3 days",
        initial_carry=carry,
        output_dir=tmp_path,
    )
    assert result.completed
    assert result.steps_completed == 3
    assert result.paths == []
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The health gate
# ---------------------------------------------------------------------------


def test_unhealthy_chunk_stops_the_run(coupler, tmp_path):
    """The gate stops the run at the chunk it rejects, keeping that chunk's files."""
    def fails_on_the_second_chunk(datasets, chunk_index, elapsed_days):
        return chunk_index < 1, {"chunk": chunk_index, "elapsed_days": elapsed_days}

    result = run_chunked(
        coupler,
        total_time="6 days",
        chunk="2 days",
        output_dir=tmp_path,
        health_check=fails_on_the_second_chunk,
    )
    assert not result.completed
    # Two chunks ran: the good one and the one that failed. The third never did.
    assert result.steps_completed == 4
    assert len(result.reports) == 2
    assert sorted(path.name for path in result.paths) == [
        "ocn-00000000.nc", "ocn-00000002.nc",
        "seaice-00000000.nc", "seaice-00000002.nc",
    ]


def test_unhealthy_chunk_can_be_logged_and_ignored(coupler, tmp_path):
    """`bail_on_unhealthy=False` integrates an unhealthy state and says so.

    It also keeps checkpointing: the run is carrying on, so it has to stay
    resumable from where it has got to -- which is the opposite of what
    bailing does.
    """
    checkpoint = tmp_path / "checkpoint"
    result = run_chunked(
        coupler,
        total_time="4 days",
        chunk="2 days",
        output_dir=tmp_path / "output",
        checkpoint_path=checkpoint,
        health_check=lambda datasets, index, days: (False, {"chunk": index}),
        bail_on_unhealthy=False,
    )
    assert result.completed
    assert result.steps_completed == 4
    assert len(result.reports) == 2
    assert int(two_slabs().load_carry(checkpoint).step) == 4


def test_a_rejected_chunk_does_not_overwrite_the_last_good_checkpoint(
    coupler, tmp_path
):
    """Bailing leaves the restart point at the last chunk that passed.

    There is one checkpoint directory and it is overwritten in place, so
    checkpointing a chunk the gate has just rejected would destroy the last
    healthy state: the resume would start from the broken one, fail again and
    have nothing left to go back to. The gate therefore runs before the save.
    """
    def fails_on_the_second_chunk(datasets, chunk_index, elapsed_days):
        return chunk_index < 1, {"chunk": chunk_index}

    checkpoint = tmp_path / "checkpoint"
    result = run_chunked(
        coupler,
        total_time="6 days",
        chunk="2 days",
        output_dir=tmp_path / "output",
        checkpoint_path=checkpoint,
        health_check=fails_on_the_second_chunk,
    )

    assert not result.completed
    # Two chunks were integrated, and the second one's output was kept ...
    assert result.steps_completed == 4
    assert len(result.paths) == 4
    # ... but only the first was checkpointed, so a resume repeats the chunk
    # that failed instead of starting from its state.
    assert int(two_slabs().load_carry(checkpoint).step) == 2


def test_no_health_check_collects_no_reports(coupler, tmp_path):
    """`health_check=None` is no gate at all, not a gate that always passes."""
    result = run_chunked(
        coupler,
        total_time="2 days",
        chunk="2 days",
        output_dir=tmp_path,
        health_check=None,
    )
    assert result.completed
    assert result.reports == []


def test_default_health_check_skips_without_an_atmosphere():
    """A coupled model with no `atm` dataset gets an abstention, not a pass."""
    ok, report = default_health_check({"ocn": xr.Dataset()}, 3, 12.0)
    assert ok
    assert report == {"chunk": 3, "elapsed_days": 12.0, "skipped": "no atmosphere"}


class BlowsUpInTheLastRecord(Coupler):
    """A two-slab coupler whose ocean's chunk ends with one NaN point.

    A slab ocean integrated for a few days does not blow up, and tuning one
    until it did would stop it being a slab ocean. What a health gate has to
    cope with is the *shape* of a blow-up rather than its cause -- a bad
    value in the chunk's final record -- so that is injected here, at the one
    place the driver takes a chunk's datasets from.
    """

    def to_xarray(self, diagnostics, time=None, *, first_step=0):
        datasets = super().to_xarray(diagnostics, time, first_step=first_step)
        ocean = datasets["ocn"]["sea_surface_temperature"]
        values = np.asarray(ocean.values).copy()
        values[-1, 0, 0] = np.nan
        datasets["ocn"]["sea_surface_temperature"] = ocean.copy(data=values)
        return datasets


def two_slabs_blowing_up() -> Coupler:
    """Return :func:`two_slabs`' model, with a NaN at the end of every chunk."""
    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    return BlowsUpInTheLastRecord(
        components,
        default_exchangers(components),
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )


def rejects_a_nan_in_the_last_record(datasets, chunk_index, elapsed_days):
    """Ask of the ocean what `jcm.diagnostics.check_health` asks of the atmosphere.

    The real gate reads `isel(time=-1)` and fails on a NaN there; this is the
    same question, put to the one component a two-slab coupler has.
    """
    last = datasets["ocn"]["sea_surface_temperature"].isel(time=-1)
    unhealthy = bool(np.isnan(np.asarray(last)).any())
    return not unhealthy, {"chunk": chunk_index, "nan_in_last_record": unhealthy}


@pytest.mark.parametrize(
    ("options", "records"),
    [({}, 4), ({"output_averages": True}, 1), ({"subsample": 2}, 2)],
)
def test_health_gate_sees_the_chunk_unreduced(tmp_path, options, records):
    """A state that goes bad at the end of a chunk is caught however output is reduced.

    Both reductions destroy the evidence a gate reads: `output_averages`
    averages the chunk, and xarray's mean skips NaNs and dilutes a finite
    extreme, while `subsample=2` drops the last of four records outright. So
    the gate is given the chunk as integrated, and only the copy written to
    disk is reduced -- which the file's own record count and the NaN's
    absence from it check, so the fix cannot be "stop reducing the output".
    """
    result = run_chunked(
        two_slabs_blowing_up(),
        total_time="4 days",
        chunk="4 days",
        output_dir=tmp_path,
        health_check=rejects_a_nan_in_the_last_record,
        **options,
    )

    assert not result.completed
    assert result.reports == [{"chunk": 0, "nan_in_last_record": True}]

    with xr.open_dataset(tmp_path / "ocn-00000000.nc") as written:
        assert written.sizes["time"] == records
        # The file is the reduced form, and in both reduced forms the bad
        # point is no longer in it -- which is exactly what the gate used to
        # be shown.
        nan_in_file = bool(np.isnan(written["sea_surface_temperature"].values).any())
        assert nan_in_file == (not options)


# ---------------------------------------------------------------------------
# Chunking and resuming give the same run
# ---------------------------------------------------------------------------


def test_continuous_chunked_resumed_agree(tmp_path):
    """Ten steps in one chunk, in two chunks, and across a restart, all agree.

    This is the property the whole loop exists to have: how a run is *divided*
    -- by chunk, by checkpoint, by process -- must not change the trajectory.
    The step counter lives in the carry, so the seasonal cycle and every
    component's clock continue across a boundary rather than restarting.
    """
    continuous = run_chunked(
        two_slabs(), total_time="10 days", chunk="10 days",
        output_dir=tmp_path / "continuous",
    )
    chunked = run_chunked(
        two_slabs(), total_time="10 days", chunk="5 days",
        output_dir=tmp_path / "chunked",
    )

    checkpoint = tmp_path / "checkpoint"
    first_half = run_chunked(
        two_slabs(), total_time="5 days", chunk="5 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )
    # A second, independently built coupler, resuming from the file alone:
    # what a new process does, without the carry the first run returned.
    resumed = run_chunked(
        two_slabs(), total_time="10 days", chunk="5 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )

    assert first_half.steps_completed == 5
    for result in (chunked, resumed):
        assert result.steps_completed == continuous.steps_completed == 10
        assert_carries_agree(result.final_carry, continuous.final_carry, atol=1e-12)

    # The resumed run wrote the second chunk's file, not the first one again.
    assert sorted(path.name for path in resumed.paths) == [
        "ocn-00000005.nc", "seaice-00000005.nc"
    ]
    assert (tmp_path / "restarted" / "ocn-00000000.nc").exists()


def test_checkpoint_is_one_directory_rewritten_each_chunk(coupler, tmp_path):
    """The checkpoint is a single directory holding the newest state only."""
    checkpoint = tmp_path / "checkpoint"
    result = run_chunked(
        coupler, total_time="4 days", chunk="2 days",
        output_dir=tmp_path, checkpoint_path=checkpoint,
    )
    assert (checkpoint / CARRY_FILENAME).exists()
    assert sorted(p.name for p in checkpoint.iterdir()) == [CARRY_FILENAME]

    restored = two_slabs().load_carry(checkpoint)
    assert int(restored.step) == 4
    assert_carries_agree(restored, result.final_carry, atol=1e-12)


def test_checkpointing_is_on_by_default_inside_the_output_directory(
    coupler, tmp_path
):
    """A run with no `checkpoint_path` still leaves a restart point behind.

    Checkpointing is on, and the default path is relative, so it lands in the
    run's own output directory. That is what makes an on-by-default checkpoint
    safe: every run's output directory is its own, so two runs launched from
    the same shell cannot write over each other's restart state.
    """
    result = run_chunked(
        coupler, total_time="4 days", chunk="2 days", output_dir=tmp_path
    )

    checkpoint = tmp_path / "checkpoint"
    assert (checkpoint / CARRY_FILENAME).exists()
    restored = two_slabs().load_carry(checkpoint)
    assert int(restored.step) == 4
    assert_carries_agree(restored, result.final_carry, atol=1e-12)


def test_a_rerun_into_the_same_output_directory_resumes(tmp_path, caplog):
    """Pointing a second run at the same `output_dir` continues the first.

    With a relative default resolved against `output_dir`, the action that
    resumes a run is the same one that would otherwise overwrite its output --
    and the provenance line says which happened, so it is never a silent
    choice.
    """
    run_chunked(two_slabs(), total_time="2 days", chunk="2 days", output_dir=tmp_path)

    with caplog.at_level(logging.INFO, logger="jem.driver"):
        resumed = run_chunked(
            two_slabs(), total_time="4 days", chunk="2 days", output_dir=tmp_path
        )

    assert resumed.steps_completed == 4
    assert (
        f"Resumed from checkpoint {tmp_path / 'checkpoint'} at coupled step 2."
        in caplog.text
    )
    # Only the second chunk was integrated, so only its file was written.
    assert sorted(path.name for path in resumed.paths) == [
        "ocn-00000002.nc", "seaice-00000002.nc"
    ]


def test_repeating_a_finished_run_warns_that_it_did_nothing(tmp_path, caplog):
    """The same call twice resumes rather than repeating, and says so loudly.

    `output_dir` is a fixed relative path for a plain Python caller, so with
    checkpointing on the second of two identical calls -- a script re-run, a
    notebook cell run again -- restores the first call's final state and has
    nothing left to integrate. That is correct, and it is also not what
    someone re-running a script expects, so it is a WARNING rather than a
    note.
    """
    run_chunked(two_slabs(), total_time="2 days", chunk="2 days", output_dir=tmp_path)

    with caplog.at_level(logging.INFO, logger="jem.driver"):
        again = run_chunked(
            two_slabs(), total_time="2 days", chunk="2 days", output_dir=tmp_path
        )

    assert again.completed
    assert again.steps_completed == 2
    assert again.paths == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "a run that integrated nothing must say so at WARNING"
    assert "Nothing to integrate" in caplog.text
    # The same line names where the state came from, so the reason it had
    # nothing to do is in the message that reports it.
    assert f"Resumed from checkpoint {tmp_path / 'checkpoint'}" in caplog.text


def test_checkpointing_can_be_turned_off(coupler, tmp_path):
    """`checkpoint_path=None` writes no restart state at all."""
    run_chunked(
        coupler, total_time="2 days", chunk="2 days",
        output_dir=tmp_path, checkpoint_path=None,
    )
    assert not (tmp_path / "checkpoint").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "ocn-00000000.nc", "seaice-00000000.nc"
    ]


def test_an_absolute_checkpoint_path_is_used_as_given(coupler, tmp_path):
    """An absolute path is not resolved against `output_dir`.

    That is how a run checkpoints to scratch while writing its output
    somewhere else -- and how a run keeps one restart directory across
    several output directories.
    """
    output = tmp_path / "output"
    elsewhere = tmp_path / "scratch" / "restart"
    run_chunked(
        coupler, total_time="2 days", chunk="2 days",
        output_dir=output, checkpoint_path=elsewhere,
    )

    assert (elsewhere / CARRY_FILENAME).exists()
    assert not (output / "checkpoint").exists()


def test_resume_skips_incomplete_checkpoint(coupler, tmp_path, caplog):
    """A checkpoint directory with no carry file is stepped over, not loaded.

    `jem.checkpoint` publishes the carry file last, so a directory without one
    is what an interrupted save leaves behind: its component states belong to
    a step nothing records. The run starts from the initial carry instead, and
    says so.
    """
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )
    (checkpoint / CARRY_FILENAME).unlink()

    with caplog.at_level(logging.WARNING, logger="jem.driver"):
        result = run_chunked(
            coupler, total_time="2 days", chunk="2 days",
            output_dir=tmp_path / "second", checkpoint_path=checkpoint,
        )
    assert "not a complete checkpoint" in caplog.text
    # Started from step 0, so the run integrated its two days again.
    assert result.steps_completed == 2


def test_the_documented_long_run_durations_are_a_whole_number_of_chunks(coupler):
    """The long-run snippet the docs show is one `run_chunked` accepts.

    `total_time` must be a whole multiple of `chunk`. The docs spell the
    duration in days (`"2190 days"`), not `"6 years"`: on the coupler's
    default `"gregorian"` calendar a calendar-averaged year is 365.2425
    days, so `"6 years"` is 2191.455 days -- not even a whole number of
    days, let alone a whole multiple of a 30-day chunk -- which is why the
    docs spell this duration in days instead of relying on a duration whose
    length depends on the calendar. This pins both facts: the day-count
    example the docs use works, and the `"6 years"` spelling would not.
    """
    from jem.driver import _whole_steps

    coupling_days = coupler.dt_seconds / 86400.0
    total = _whole_steps("2190 days", coupling_days, coupler, "total_time")
    per_chunk = _whole_steps("30 days", coupling_days, coupler, "chunk")
    assert total == 2190
    assert total % per_chunk == 0

    with pytest.raises(ValueError, match="total_time='6 years'"):
        _whole_steps("6 years", coupling_days, coupler, "total_time")


# ---------------------------------------------------------------------------
# How often the checkpoint is written
# ---------------------------------------------------------------------------


def checkpoint_step(checkpoint):
    """Return the coupled step a checkpoint holds, or None if there is none."""
    if not (checkpoint / CARRY_FILENAME).exists():
        return None
    return int(two_slabs().load_carry(checkpoint).step)


def watch_the_checkpoint(checkpoint, seen, rejects=()):
    """Return a health check that records what the checkpoint holds as it runs.

    The gate runs after a chunk has been integrated and **before** that chunk
    is saved, so the step it records is the one the chunks *before* it left
    behind -- which is exactly the restart point the run would fall back to if
    it died there. `rejects` names the chunk indices the gate fails.
    """
    def health_check(datasets, chunk_index, elapsed_days):
        seen.append(checkpoint_step(checkpoint))
        return chunk_index not in rejects, {"chunk": chunk_index}

    return health_check


def test_checkpoint_interval_saves_every_nth_chunk(tmp_path):
    """Five one-day chunks with a two-chunk interval save at steps 2, 4 and 5.

    `seen[i]` is what the checkpoint held when chunk `i` had just been
    integrated: nothing for the first two, then the step-2 boundary, then the
    step-4 one. The 5 at the end is the other rule -- the last chunk of a
    completed run is saved whatever the interval says, so a finished run
    always leaves its final restart state.
    """
    checkpoint = tmp_path / "checkpoint"
    seen = []
    result = run_chunked(
        two_slabs(),
        total_time="5 days",
        chunk="1 day",
        checkpoint_interval="2 days",
        output_dir=tmp_path / "output",
        checkpoint_path=checkpoint,
        health_check=watch_the_checkpoint(checkpoint, seen),
    )

    assert result.completed
    assert result.steps_completed == 5
    assert seen == [None, None, 2, 2, 4]
    assert checkpoint_step(checkpoint) == 5


def test_without_an_interval_every_chunk_is_still_saved(tmp_path):
    """The default is unchanged: no interval, a checkpoint after every chunk."""
    checkpoint = tmp_path / "checkpoint"
    seen = []
    run_chunked(
        two_slabs(),
        total_time="4 days",
        chunk="1 day",
        output_dir=tmp_path / "output",
        checkpoint_path=checkpoint,
        health_check=watch_the_checkpoint(checkpoint, seen),
    )
    assert seen == [None, 1, 2, 3]
    assert checkpoint_step(checkpoint) == 4


def test_the_interval_counts_from_the_start_of_the_run_not_of_the_call(tmp_path):
    """A resumed run checkpoints where an uninterrupted one would.

    Six one-day chunks with a two-chunk interval save at steps 2, 4 and 6.
    Stopping after three days and resuming must not move those points: the
    second call integrates the chunks ending at 4, 5 and 6 and saves at 4 --
    the *run's* second boundary -- and not at 5, which is where an interval
    counted from the start of the call would have landed.
    """
    checkpoint = tmp_path / "checkpoint"
    settings = {
        "chunk": "1 day",
        "checkpoint_interval": "2 days",
        "checkpoint_path": checkpoint,
    }

    first = []
    run_chunked(
        two_slabs(), total_time="3 days", output_dir=tmp_path / "first",
        health_check=watch_the_checkpoint(checkpoint, first), **settings,
    )
    # Step 2 is the interval boundary; step 3 is there because a completed run
    # always checkpoints its last chunk.
    assert first == [None, None, 2]
    assert checkpoint_step(checkpoint) == 3

    resumed = []
    run_chunked(
        two_slabs(), total_time="6 days", output_dir=tmp_path / "second",
        health_check=watch_the_checkpoint(checkpoint, resumed), **settings,
    )
    assert resumed == [3, 4, 4]
    assert checkpoint_step(checkpoint) == 6


def test_a_bail_out_keeps_the_interval_checkpoint_it_had_already_written(tmp_path):
    """The last accepted chunk was the interval boundary: nothing to add.

    One-day chunks with a two-chunk interval, and the gate rejects the chunk
    ending at step 3. The chunk before it ended on the boundary and was saved,
    so the restart point is already the last healthy state and bailing writes
    nothing further -- the rejected chunk itself is never checkpointed.
    """
    checkpoint = tmp_path / "checkpoint"
    seen = []
    result = run_chunked(
        two_slabs(),
        total_time="5 days",
        chunk="1 day",
        checkpoint_interval="2 days",
        output_dir=tmp_path / "output",
        checkpoint_path=checkpoint,
        health_check=watch_the_checkpoint(checkpoint, seen, rejects={2}),
    )

    assert not result.completed
    assert result.steps_completed == 3
    assert seen == [None, None, 2]
    assert checkpoint_step(checkpoint) == 2


def test_a_bail_out_saves_the_accepted_chunk_the_interval_had_skipped(
    tmp_path, caplog
):
    """Bailing must not lose the healthy chunks the interval had not saved.

    One-day chunks with a three-chunk interval: the run saves at step 3, then
    integrates the chunk ending at 4 (accepted, unsaved because of the
    interval) and the chunk ending at 5, which the gate rejects. Left alone
    the restart point would fall back to step 3 and the resume would
    re-integrate a healthy chunk already paid for, so the last accepted carry
    -- step 4 -- is written on the way out, and named in the log. The rejected
    chunk is still not checkpointed.
    """
    checkpoint = tmp_path / "checkpoint"
    seen = []
    with caplog.at_level(logging.INFO):
        result = run_chunked(
            two_slabs(),
            total_time="6 days",
            chunk="1 day",
            checkpoint_interval="3 days",
            output_dir=tmp_path / "output",
            checkpoint_path=checkpoint,
            health_check=watch_the_checkpoint(checkpoint, seen, rejects={4}),
        )

    assert not result.completed
    assert result.steps_completed == 5
    # Step 3 is all the interval had written by the time the chunk was rejected.
    assert seen == [None, None, None, 3, 3]
    assert checkpoint_step(checkpoint) == 4
    assert (
        "Checkpointed the last chunk the health gate accepted, at coupled step 4"
        in caplog.text
    )


def test_a_resume_that_cannot_reach_the_interval_says_so(tmp_path, caplog):
    """A resume part-way through a chunk warns that the interval cannot land.

    The interval is counted from the start of the run and the loop stops only
    at a chunk boundary, so a checkpoint written under a *different* chunk
    length leaves an offset that no chunk end of this run can turn into a
    multiple of the interval: the run would checkpoint only when it finished.
    That is a real loss of restart points, so it is said out loud rather than
    left to be discovered after a job was killed.
    """
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="3 days", chunk="3 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )
    with caplog.at_level(logging.WARNING):
        result = run_chunked(
            two_slabs(), total_time="8 days", chunk="2 days",
            checkpoint_interval="4 days",
            output_dir=tmp_path / "second", checkpoint_path=checkpoint,
        )

    assert result.steps_completed == 8
    assert "starts at coupled step 3" in caplog.text
    assert "not a whole number of the 2-step chunks" in caplog.text
    assert "no chunk it integrates before the last" in caplog.text
    # This run has a checkpoint whose chunk length it could match, so it is
    # told how to get the interval back.
    assert "the chunk the checkpoint was written under" in caplog.text
    # It still leaves its final restart state, which is the other guarantee.
    assert checkpoint_step(checkpoint) == 8


def test_a_total_time_that_is_not_whole_intervals_warns_but_runs(tmp_path, caplog):
    """Five chunks with a two-chunk interval run; the shorter last gap is said.

    The interval need not divide `total_time`, because a completed run
    checkpoints its last chunk whatever the interval says -- so this is a
    warning and not a refusal. It is still worth one: the interval is what
    someone sizing a requeue reasons with, and the last gap between saves
    (step 4 to step 5 here) is shorter than it.
    """
    checkpoint = tmp_path / "checkpoint"
    with caplog.at_level(logging.WARNING):
        result = run_chunked(
            two_slabs(), total_time="5 days", chunk="1 day",
            checkpoint_interval="2 days",
            output_dir=tmp_path / "output", checkpoint_path=checkpoint,
        )

    assert result.completed
    assert "total_time ('5 days', 5 coupled steps)" in caplog.text
    assert "checkpoint_interval ('2 days', 2 coupled steps)" in caplog.text
    assert checkpoint_step(checkpoint) == 5


def test_a_total_time_that_is_whole_intervals_says_nothing(tmp_path, caplog):
    """The warning is about a short final gap, so a run without one is silent."""
    with caplog.at_level(logging.WARNING):
        run_chunked(
            two_slabs(), total_time="4 days", chunk="1 day",
            checkpoint_interval="2 days",
            output_dir=tmp_path / "output",
            checkpoint_path=tmp_path / "checkpoint",
        )
    assert "checkpoint_interval" not in caplog.text


def test_an_initial_carry_part_way_through_a_chunk_warns_without_a_remedy(
    tmp_path, caplog
):
    """The same warning for a carry handed in, minus the advice that would lie.

    A run also starts part-way through a chunk when it is given an
    `initial_carry` at such a step -- it never resumed, and there is no
    checkpoint whose chunk length it could match -- so the fact is stated and
    the remedy is not.
    """
    coupler = two_slabs()
    carry, _ = coupler.generate_trajectory_function(3)(coupler.initialize())

    with caplog.at_level(logging.WARNING):
        result = run_chunked(
            coupler, total_time="8 days", chunk="2 days",
            checkpoint_interval="4 days", initial_carry=carry,
            output_dir=tmp_path / "output",
            checkpoint_path=tmp_path / "checkpoint",
        )

    assert result.steps_completed == 8
    warnings = [
        record.getMessage() for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    assert len(warnings) == 1
    assert "starts at coupled step 3" in warnings[0]
    assert "the chunk the checkpoint was written under" not in warnings[0]


def test_an_accumulated_run_checkpoints_on_the_interval_too(tmp_path):
    """The interval applies to a run reducing inside the scan.

    An accumulated run writes no files and can have no health check, so there
    is no hook to watch the checkpoint through: `save_carry` itself is
    wrapped, and what it records is the saves the interval allowed -- the
    step-2 and step-4 boundaries, and the last chunk of the completed run.
    """
    coupler = two_slabs()
    saved = []
    save_carry = coupler.save_carry

    def record(carry, path):
        saved.append(int(carry.step))
        return save_carry(carry, path)

    coupler.save_carry = record
    result = run_chunked(
        coupler,
        total_time="5 days",
        chunk="1 day",
        checkpoint_interval="2 days",
        output_dir=tmp_path / "output",
        checkpoint_path=tmp_path / "checkpoint",
        health_check=None,
        accumulate=monthly_mean(two_slabs()),
    )

    assert result.completed
    assert result.paths == []
    assert saved == [2, 4, 5]


def test_an_ignored_health_failure_keeps_the_interval_and_bails_out_of_nothing(
    tmp_path, caplog
):
    """`bail_on_unhealthy=False` never reaches the bail-out checkpoint.

    The run carries on, so every chunk is "accepted" as far as the interval is
    concerned and the saves fall exactly where a healthy run's would. The
    extra save a bail-out makes is for the state a stopping run would
    otherwise lose, and nothing stops here.
    """
    checkpoint = tmp_path / "checkpoint"
    seen = []
    with caplog.at_level(logging.INFO):
        result = run_chunked(
            two_slabs(),
            total_time="5 days",
            chunk="1 day",
            checkpoint_interval="2 days",
            output_dir=tmp_path / "output",
            checkpoint_path=checkpoint,
            health_check=watch_the_checkpoint(
                checkpoint, seen, rejects={0, 1, 2, 3, 4}
            ),
            bail_on_unhealthy=False,
        )

    assert result.completed
    assert seen == [None, None, 2, 2, 4]
    assert checkpoint_step(checkpoint) == 5
    assert "Checkpointed the last chunk the health gate accepted" not in caplog.text


@pytest.mark.parametrize(
    ("interval", "message"),
    [
        # Longer than a chunk but not a multiple of one.
        ("3 days", "not a whole number of chunks"),
        # Shorter than a chunk: the same refusal, since the run cannot stop
        # in the middle of one to write a checkpoint.
        ("1 day", "not a whole number of chunks"),
        # Not even a whole number of coupled steps.
        ("36 hours", "checkpoint_interval="),
    ],
)
def test_run_chunked_rejects_a_checkpoint_interval_it_cannot_honour(
    coupler, tmp_path, interval, message
):
    """An interval that is not a whole number of chunks is refused, naming both."""
    with pytest.raises(ValueError, match=message):
        run_chunked(
            coupler, total_time="10 days", chunk="2 days",
            checkpoint_interval=interval, output_dir=tmp_path,
        )


def test_a_bad_checkpoint_interval_is_refused_before_anything_is_compiled(
    coupler, tmp_path
):
    """The interval is checked with the other durations, not at the first save.

    It is not read until a chunk has been integrated and passed the gate, so
    a run configured with an interval the driver cannot honour would otherwise
    compile a trajectory and integrate a whole chunk before raising.
    """
    interval = "3 days"

    def must_not_be_called(iterations, **kwargs):
        raise AssertionError(
            f"a {iterations}-step trajectory was built despite "
            f"checkpoint_interval={interval!r}"
        )

    coupler.generate_trajectory_function = must_not_be_called
    with pytest.raises(ValueError, match="not a whole number of chunks"):
        run_chunked(
            coupler, total_time="10 days", chunk="2 days",
            checkpoint_interval=interval, output_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_a_checkpoint_interval_without_a_checkpoint_path_is_refused(
    coupler, tmp_path
):
    """Spacing out saves that were switched off is a contradiction, not a no-op.

    Ignoring it would leave a run that asked to checkpoint less often
    checkpointing not at all, and finding out when it tried to resume.
    """
    with pytest.raises(ValueError, match="checkpoint_path=None"):
        run_chunked(
            coupler, total_time="4 days", chunk="2 days",
            checkpoint_interval="4 days", checkpoint_path=None,
            output_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# What the run says about the state it starts from
# ---------------------------------------------------------------------------


def test_a_run_with_checkpointing_off_says_it_started_from_initialize(
    coupler, tmp_path, caplog
):
    """No checkpoint, no carry: the log names `coupler.initialize()` and step 0.

    A run's starting state decides what its output means, and a reader of the
    log cannot see it in any other line -- so there is exactly one, always.
    """
    with caplog.at_level(logging.INFO, logger="jem.driver"):
        run_chunked(
            coupler, total_time="2 days", chunk="2 days",
            output_dir=tmp_path, checkpoint_path=None,
        )

    assert (
        "Starting from coupler.initialize() at coupled step 0 "
        "(no checkpoint was given)." in caplog.text
    )


def test_a_first_run_names_the_empty_checkpoint_path_it_looked_at(
    coupler, tmp_path, caplog
):
    """Checkpointing is on by default, so a first run says what it did not find.

    It is reported at INFO, not WARNING: with a fresh output directory per run
    this is what *every* first run sees, and a warning nobody can avoid is a
    warning nobody reads. The path is named, so a mistyped one is still
    visible in the one line the run always prints.
    """
    with caplog.at_level(logging.INFO, logger="jem.driver"):
        run_chunked(coupler, total_time="2 days", chunk="2 days", output_dir=tmp_path)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert f"There is no checkpoint at {tmp_path / 'checkpoint'}" in caplog.text
    assert (
        f"Starting from coupler.initialize() at coupled step 0 "
        f"({tmp_path / 'checkpoint'} holds no complete checkpoint)."
        in caplog.text
    )


def test_an_initial_carry_is_named_as_the_source(coupler, tmp_path, caplog):
    """A carry handed in is reported as such, at the step it is already at.

    This is the case that used to be silent: an in-process continuation and a
    cold start produced identical logs, and the second is a run that repeats
    simulated time already paid for.
    """
    carry, _ = coupler.generate_trajectory_function(3)(coupler.initialize())

    with caplog.at_level(logging.INFO, logger="jem.driver"):
        run_chunked(
            coupler, total_time="5 days", chunk="1 day",
            initial_carry=carry, output_dir=tmp_path,
        )

    assert "Starting from the initial_carry argument at coupled step 3." in caplog.text


def test_a_resumed_run_names_the_checkpoint_it_came_from(tmp_path, caplog):
    """A real resume says so, with the path and the step it restored."""
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )

    with caplog.at_level(logging.INFO, logger="jem.driver"):
        run_chunked(
            two_slabs(), total_time="4 days", chunk="2 days",
            output_dir=tmp_path / "second", checkpoint_path=checkpoint,
        )

    assert f"Resumed from checkpoint {checkpoint} at coupled step 2." in caplog.text


def test_the_wreckage_of_an_interrupted_save_warns(coupler, tmp_path, caplog):
    """A checkpoint directory with no carry file is a WARNING, not a note.

    That is the one failure to resume that is genuinely abnormal: a run died
    mid-save, and the chunk it was writing is gone. Unlike an empty path it is
    not what every first run sees, so it is worth a warning -- which says both
    what could not be read and that every component therefore starts from its
    initial state rather than from a restart.
    """
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )
    (checkpoint / CARRY_FILENAME).unlink()

    with caplog.at_level(logging.INFO, logger="jem.driver"):
        run_chunked(
            coupler, total_time="2 days", chunk="2 days",
            output_dir=tmp_path / "second", checkpoint_path=checkpoint,
        )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "an interrupted save must warn"
    assert "starts from its initial state rather than from a restart" in caplog.text
    # And the provenance line still says where the state did come from, naming
    # the path so the two lines cannot be read as being about different runs.
    assert (
        f"Starting from coupler.initialize() at coupled step 0 ({checkpoint} "
        "holds no complete checkpoint)." in caplog.text
    )


def test_a_failed_resume_does_not_claim_an_initial_carry_was_discarded(
    coupler, tmp_path, caplog
):
    """A spun-up `initial_carry` is what the run falls back to, and is said so.

    Starting from a spun-up state while checkpointing as you go is the normal
    way to begin a production run, and on its first call there is no
    checkpoint to resume. The warning must not then assert that the model is
    back at its initial state at the start date -- that would contradict the
    provenance line printed immediately after it, and a warning that is
    routinely wrong is a warning nobody reads.
    """
    carry, _ = coupler.generate_trajectory_function(3)(coupler.initialize())

    with caplog.at_level(logging.INFO, logger="jem.driver"):
        run_chunked(
            coupler, total_time="5 days", chunk="1 day",
            initial_carry=carry, output_dir=tmp_path,
            checkpoint_path=tmp_path / "checkpoint",
        )

    assert "There is no checkpoint at" in caplog.text
    assert "rather than from a restart" in caplog.text
    assert "begins again at the start date" not in caplog.text
    assert "Starting from the initial_carry argument at coupled step 3." in caplog.text


def test_a_resume_says_the_initial_carry_was_not_used(tmp_path, caplog):
    """A checkpoint wins over `initial_carry`, and the log does not hide it."""
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )

    coupler = two_slabs()
    unused, _ = coupler.generate_trajectory_function(1)(coupler.initialize())
    with caplog.at_level(logging.INFO, logger="jem.driver"):
        run_chunked(
            coupler, total_time="4 days", chunk="2 days",
            initial_carry=unused, output_dir=tmp_path / "second",
            checkpoint_path=checkpoint,
        )

    assert f"Resumed from checkpoint {checkpoint} at coupled step 2." in caplog.text
    assert "The initial_carry argument was not used." in caplog.text


def test_the_load_names_every_component_and_where_it_came_from(tmp_path, caplog):
    """`load_carry` reports each component's source and the restored step.

    A coupled checkpoint is a mixture of two storage mechanisms -- the shared
    carry file and the subdirectories components write themselves -- and which
    a component used is invisible from the outside. The log says it, so
    "which of my components actually came off disk?" is answered without
    reading `jem.checkpoint`.
    """
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )

    with caplog.at_level(logging.INFO, logger="jem.checkpoint"):
        two_slabs().load_carry(checkpoint)

    assert f"Loaded checkpoint {checkpoint} at coupled step 2" in caplog.text
    # Neither slab checkpoints itself, so both are in the shared file and the
    # delegated list is empty -- and says so rather than being blank.
    assert f"ocn, seaice restored from {CARRY_FILENAME}" in caplog.text
    assert "(none) restored by their own load_carry" in caplog.text


def test_resume_with_a_different_chunk_length_still_stops_on_time(tmp_path):
    """A checkpoint part-way through a chunk is finished off in a short batch.

    The chunk length is a choice of the run, not a property of the checkpoint,
    so a run resumed with a longer chunk starts mid-chunk. What is left is
    computed from the restored step counter, so the run still stops exactly at
    `total_time` -- and, because `remaining_batches` puts the short batch
    LAST, the eight days are integrated as 3 + 4 + 1. The control run is what
    makes that a statement about the trajectory rather than about the counter:
    the separately-compiled short final batch has to produce the same numbers
    as one continuous eight-day integration, which is the property a bug in
    `remaining_batches` or in the driver's per-length trajectory cache would
    break while still stopping at step 8.
    """
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="3 days", chunk="3 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
    )
    resumed = run_chunked(
        two_slabs(), total_time="8 days", chunk="4 days",
        output_dir=tmp_path / "second", checkpoint_path=checkpoint,
    )
    assert resumed.completed
    assert resumed.steps_completed == 8

    continuous = run_chunked(
        two_slabs(), total_time="8 days", chunk="8 days",
        output_dir=tmp_path / "continuous",
    )
    assert_carries_agree(resumed.final_carry, continuous.final_carry, atol=1e-12)


def test_resume_with_a_different_chunk_length_keeps_the_earlier_files(tmp_path):
    """A different chunk length on resume must not write over earlier output.

    Three days in one chunk, then a resume with four-day chunks into the SAME
    output directory. Every file is named after the coupled step its chunk
    starts at -- 0 for the first run, then 3 and 7 (a full four-day chunk and
    then the short one that stops the run exactly at eight days) -- so all
    three survive. Under a chunk *index* the resumed run's first chunk would
    have been index `3 // 4 == 0` again, and the first run's three days of
    output would have been silently replaced.
    """
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    first = run_chunked(
        two_slabs(), total_time="3 days", chunk="3 days",
        output_dir=output, checkpoint_path=checkpoint,
    )
    resumed = run_chunked(
        two_slabs(), total_time="8 days", chunk="4 days",
        output_dir=output, checkpoint_path=checkpoint,
    )
    assert resumed.steps_completed == 8

    assert [path.name for path in first.paths] == [
        "ocn-00000000.nc", "seaice-00000000.nc"
    ]
    assert sorted(path.name for path in resumed.paths) == [
        "ocn-00000003.nc", "ocn-00000007.nc",
        "seaice-00000003.nc", "seaice-00000007.nc",
    ]
    # The first run's files are still there, and still hold its three days.
    assert all(path.exists() for path in first.paths)
    with xr.open_dataset(output / "ocn-00000000.nc") as written:
        assert written.sizes["time"] == 3

    # The whole run reads back as one continuous eight-day series.
    with xr.open_mfdataset(
        sorted(output.glob("ocn-*.nc")), combine="by_coords"
    ) as combined:
        assert combined.sizes["time"] == 8


# ---------------------------------------------------------------------------
# Output left past the restart point by a killed run
# ---------------------------------------------------------------------------


def killed_run_state(tmp_path):
    """Build the directory and checkpoint a killed run leaves behind.

    The state a `checkpoint_interval` makes ordinary, and the one the resume
    check exists for: output written for coupled steps the checkpoint does not
    reach. Four one-day chunks with a four-day interval put the checkpoint at
    step 4 and files at steps 0-3; the run then carries on to six days,
    writing files at steps 4 and 5, and is "killed" by putting the step-4
    checkpoint back. Building it by running rather than by hand is what makes
    the files real output of this coupler, so a test can compare them with
    what a resume writes.

    Returns the output directory and the checkpoint directory.
    """
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    settings = {
        "chunk": "1 day",
        "checkpoint_interval": "4 days",
        "output_dir": output,
        "checkpoint_path": checkpoint,
    }
    run_chunked(two_slabs(), total_time="4 days", **settings)
    assert checkpoint_step(checkpoint) == 4
    at_four = tmp_path / "checkpoint-at-step-4"
    shutil.copytree(checkpoint, at_four)

    run_chunked(two_slabs(), total_time="6 days", **settings)
    assert checkpoint_step(checkpoint) == 6
    assert (output / "ocn-00000005.nc").exists()

    shutil.rmtree(checkpoint)
    shutil.copytree(at_four, checkpoint)
    return output, checkpoint


def output_names(output):
    """Return the names of the netCDF files in an output directory, sorted."""
    return sorted(path.name for path in output.glob("*.nc"))


def test_a_resume_that_would_orphan_output_is_refused(tmp_path):
    """The Codex scenario: a rechunked resume over a killed run's files raises.

    A killed run leaves a checkpoint at step 4 and one-day files at steps 4
    and 5. Resuming with two-day chunks would write a single file at step 4
    holding steps 5 and 6 -- and step 5 is not a name this run writes, so the
    old file would stay behind holding step 6 a second time and the directory
    would read back with a duplicate nothing downstream could tell from a real
    one. The run is refused instead, before anything is compiled, and the
    message names the file that cannot be rewritten, the step the run resumed
    from, the chunk it was asked for and the three ways out.
    """
    output, checkpoint = killed_run_state(tmp_path)
    before = output_names(output)

    with pytest.raises(ValueError) as raised:
        run_chunked(
            two_slabs(), total_time="6 days", chunk="2 days",
            output_dir=output, checkpoint_path=checkpoint,
        )

    message = str(raised.value)
    assert "ocn-00000005.nc" in message
    assert "seaice-00000005.nc" in message
    # The files the resume WOULD have rewritten are not part of the complaint.
    assert "ocn-00000004.nc" not in message
    assert "coupled step 4" in message
    assert "chunk='2 days'" in message
    assert "Resume with the chunk those files were written under" in message

    # Nothing was written, and nothing was deleted: the refusal leaves the
    # killed run's output exactly as it found it, for the user to decide
    # about.
    assert output_names(output) == before
    with xr.open_dataset(output / "ocn-00000004.nc") as untouched:
        assert untouched.sizes["time"] == 1
    assert checkpoint_step(checkpoint) == 4


def test_a_resume_under_the_same_chunk_rewrites_and_is_allowed(
    tmp_path, caplog
):
    """The ordinary killed-run resume still works, and rewrites identically.

    Under the chunk the killed run used, every file at or after the restart
    point starts on one of this run's chunk boundaries, so the resume writes
    each of them again under the same name from the same starting state -- the
    property that makes a `checkpoint_interval` safe. That must not be
    collateral damage of refusing the rechunked resume, so it is asserted down
    to the contents of the rewritten files, and the run says at INFO how many
    it will rewrite.
    """
    output, checkpoint = killed_run_state(tmp_path)
    before = {}
    for name in ("ocn-00000004.nc", "ocn-00000005.nc"):
        # Read into memory and close: netCDF4 will not let the resume
        # overwrite a file this test still holds open.
        with xr.open_dataset(output / name) as dataset:
            before[name] = dataset.load()

    with caplog.at_level(logging.INFO):
        result = run_chunked(
            two_slabs(), total_time="6 days", chunk="1 day",
            output_dir=output, checkpoint_path=checkpoint,
        )

    assert result.steps_completed == 6
    assert (
        "4 existing output file(s) at or after coupled step 4 start on this "
        "run's chunk boundaries" in caplog.text
    )
    assert output_names(output) == [
        f"{name}-{step:08d}.nc"
        for name in ("ocn", "seaice")
        for step in range(6)
    ]
    for name, original in before.items():
        with xr.open_dataset(output / name) as rewritten:
            xr.testing.assert_identical(rewritten, original)

    # And the directory as a whole -- the killed run's files, the rewritten
    # ones and the chunk that finished the run -- reads back as the same six
    # days an uninterrupted run writes: every label once, in order, with the
    # same values. That is the property the refusal above protects.
    uninterrupted = tmp_path / "uninterrupted"
    run_chunked(
        two_slabs(), total_time="6 days", chunk="1 day",
        output_dir=uninterrupted, checkpoint_path=None,
    )
    with xr.open_mfdataset(
        sorted(output.glob("ocn-*.nc")), combine="by_coords"
    ) as combined, xr.open_mfdataset(
        sorted(uninterrupted.glob("ocn-*.nc")), combine="by_coords"
    ) as reference:
        assert combined.sizes["time"] == 6
        assert len(np.unique(combined["time"].values)) == 6
        np.testing.assert_array_equal(
            combined["time"].values, reference["time"].values
        )
        np.testing.assert_allclose(
            combined["sea_surface_temperature"].values,
            reference["sea_surface_temperature"].values,
            atol=1e-12, rtol=0,
        )


def test_a_rechunked_resume_removes_a_file_it_keeps_no_record_for(
    tmp_path, caplog
):
    """A chunk this pass keeps nothing for takes its name's file with it.

    The one way the skip could leave a duplicate behind. An earlier pass with
    six-day chunks and `subsample=5` wrote files at steps 0, 6, 12 and 18,
    the step-12 one holding coupled step 15. Rewound to step 12 and resumed
    with three-day chunks, this run's grid is 12, 15, 18, 21 -- so the
    step-12 and step-18 files are on it and the resume check declares them
    this run's to rewrite. But this run's chunk at step 12 covers steps 12-14
    and the stride keeps none of them, so it has nothing to put at that name
    while it writes step 15 into `ocn-00000015.nc`: leaving the old file
    there would leave step 15 in the directory twice. It is removed instead.
    """
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    settings = {
        "chunk": "6 days",
        "subsample": 5,
        "output_dir": output,
        "checkpoint_path": checkpoint,
    }
    run_chunked(two_slabs(), total_time="12 days", **settings)
    at_twelve = tmp_path / "checkpoint-at-step-12"
    shutil.copytree(checkpoint, at_twelve)
    run_chunked(two_slabs(), total_time="24 days", **settings)
    assert output_names(output) == [
        f"{name}-{step:08d}.nc"
        for name in ("ocn", "seaice")
        for step in (0, 6, 12, 18)
    ]
    with xr.open_dataset(output / "ocn-00000012.nc") as stale:
        assert list(stale["time"].values) == step_labels(15)

    shutil.rmtree(checkpoint)
    shutil.copytree(at_twelve, checkpoint)
    with caplog.at_level(logging.INFO, logger="jem.output"):
        run_chunked(
            two_slabs(), total_time="24 days", chunk="3 days",
            output_dir=output, checkpoint_path=checkpoint, subsample=5,
        )

    assert "ocn-00000012.nc was not written and the file already there was " \
        "removed" in caplog.text
    assert not (output / "ocn-00000012.nc").exists()
    with xr.open_dataset(output / "ocn-00000015.nc") as written:
        assert list(written["time"].values) == step_labels(15)

    # The directory holds each kept step once: steps 0 and 5 from the first
    # pass's first file, 10 from its second, and 15 and 20 from this one.
    with xr.open_mfdataset(sorted(output.glob("ocn-*.nc"))) as combined:
        np.testing.assert_array_equal(
            combined["time"].values, np.array(step_labels(0, 5, 10, 15, 20))
        )


def test_a_resume_ignores_files_it_did_not_write(tmp_path, caplog):
    """Only this coupler's own output is examined, never anything else.

    An output directory can hold more than one run's files -- another
    component's, another model's, something a user put there -- and a file
    this coupler would never write is neither a reason to refuse a resume nor
    something the resume touches. `jem.output.output_file_step` is what draws
    the line: it returns a step only for a name one of the run's components
    would have been written under. The foreign file here sits at step 5, off
    the two-day grid, so it would refuse this resume if it were counted.
    """
    output, checkpoint = killed_run_state(tmp_path)
    # The run's own off-grid files, removed as the refusal's message suggests;
    # what is left to test is the foreign file at the same step.
    for name in ("ocn-00000005.nc", "seaice-00000005.nc"):
        (output / name).unlink()
    foreign = output / "atm-00000005.nc"
    foreign.write_bytes(b"not this coupler's output")
    unpatterned = output / "ocn.nc"
    unpatterned.write_bytes(b"nor this")

    with caplog.at_level(logging.INFO):
        result = run_chunked(
            two_slabs(), total_time="6 days", chunk="2 days",
            output_dir=output, checkpoint_path=checkpoint,
        )

    assert result.steps_completed == 6
    assert foreign.read_bytes() == b"not this coupler's output"
    assert unpatterned.exists()
    assert "atm-00000005.nc" not in caplog.text
    # The run's own file at the restart point is on its chunk grid, so it was
    # rewritten -- with the two-day chunk's two records.
    with xr.open_dataset(output / "ocn-00000004.nc") as rewritten:
        assert rewritten.sizes["time"] == 2


def test_a_fresh_run_is_not_checked_against_the_directory(tmp_path, caplog):
    """A run that did not resume is not refused, whatever is in the directory.

    The check is justified only by a restart point: the files after one are
    the ones this run's chunks are about to interleave with. With no
    checkpoint to resume from there is no such point and no earlier pass of
    *this* run to overlap, so an existing file is somebody's business and not
    the driver's -- `write_chunk`'s overwrite warning remains the only thing
    said about a directory that already holds output.
    """
    output = tmp_path / "output"
    output.mkdir()
    stray = output / "ocn-00000003.nc"
    stray.write_bytes(b"an earlier run's file, at a step this run never writes")

    with caplog.at_level(logging.INFO):
        run_chunked(
            two_slabs(), total_time="1 day", chunk="1 day",
            output_dir=output, checkpoint_path=None,
        )

    assert stray.exists()
    assert "chunk boundaries" not in caplog.text


def test_a_resume_with_nothing_past_the_restart_point_says_nothing(
    tmp_path, caplog
):
    """The common resume has no files to rewrite, and reports none.

    Checkpointed after every chunk, the last file written is the one before
    the restart point -- so there is nothing at or after it, nothing to
    rewrite and nothing to say. The INFO line has to stay rare enough to mean
    something when it appears.
    """
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    run_chunked(
        two_slabs(), total_time="2 days", chunk="1 day",
        output_dir=output, checkpoint_path=checkpoint,
    )
    with caplog.at_level(logging.INFO):
        resumed = run_chunked(
            two_slabs(), total_time="4 days", chunk="1 day",
            output_dir=output, checkpoint_path=checkpoint,
        )

    assert resumed.steps_completed == 4
    assert "chunk boundaries" not in caplog.text
    assert output_names(output) == [
        f"{name}-{step:08d}.nc"
        for name in ("ocn", "seaice")
        for step in range(4)
    ]


def test_a_resume_with_nothing_to_integrate_is_not_refused(tmp_path, caplog):
    """A call that writes nothing cannot overlap anything, so it is allowed.

    The files past the restart point are only a problem because this call
    would write that simulated time again. A call with nothing left to
    integrate -- the checkpoint is already at `total_time` -- writes no files
    at all, so refusing it would block a harmless re-invocation of a finished
    run and, worse, make the killed run's last output impossible to look at
    from the same directory.
    """
    output, checkpoint = killed_run_state(tmp_path)

    with caplog.at_level(logging.WARNING):
        result = run_chunked(
            two_slabs(), total_time="4 days", chunk="2 days",
            output_dir=output, checkpoint_path=checkpoint,
        )

    assert result.steps_completed == 4
    assert result.paths == []
    assert "Nothing to integrate" in caplog.text
    assert (output / "ocn-00000005.nc").exists()


def test_an_accumulated_resume_is_not_refused(tmp_path, caplog):
    """An accumulated resume writes no files, so no file can be orphaned.

    `accumulate` reduces inside the scan and `run_chunked` writes nothing, so
    the chunk length it uses cannot interleave anything with a killed run's
    output -- and refusing it would stop a legitimate reduction over a
    directory that merely happens to hold files.
    """
    output, checkpoint = killed_run_state(tmp_path)

    result = run_chunked(
        two_slabs(), total_time="6 days", chunk="2 days",
        output_dir=output, checkpoint_path=checkpoint,
        accumulate=monthly_mean(two_slabs()), health_check=None,
    )

    assert result.paths == []
    assert result.steps_completed == 6
    assert (output / "ocn-00000005.nc").exists()


def test_a_resume_is_refused_over_on_grid_output_past_its_end(tmp_path):
    """Being on the chunk grid is not enough: the run has to reach the file.

    The killed run got to six days; this resume keeps its one-day chunk but
    asks for five. The step-4 file is on the grid and inside the run, so it is
    rewritten -- but the step-5 file is on the grid and *past the end*, so
    nothing this run writes reaches it and it would be left beyond this run's
    output as the longer pass's. That is a different complaint from an
    overlap, and the message says which it is.
    """
    output, checkpoint = killed_run_state(tmp_path)
    before = output_names(output)

    with pytest.raises(ValueError) as raised:
        run_chunked(
            two_slabs(), total_time="5 days", chunk="1 day",
            output_dir=output, checkpoint_path=checkpoint,
        )

    message = str(raised.value)
    assert "ocn-00000005.nc" in message
    assert "at or past coupled step 5, where this run ends" in message
    # It is not the overlap case, and the file the run does rewrite is not
    # part of the complaint.
    assert "under a different chunk" not in message
    assert "ocn-00000004.nc" not in message
    assert "Resume with the chunk those files were written under" in message
    assert output_names(output) == before


def test_a_resume_is_refused_over_off_grid_output_past_its_end(tmp_path):
    """A file both off the grid and past the end is reported as past the end.

    An earlier pass under a different chunk reached step 7; this run resumes
    at step 4 with two-day chunks and stops at step 6. The step-7 file is off
    this run's grid *and* beyond its end, and the second is the reason that
    matters: no chunk of this run can reach step 7 whatever its length, so the
    message must not tell the modeller that the file overlaps output this run
    is about to write.
    """
    output, checkpoint = killed_run_state(tmp_path)
    # The killed run's own off-grid files, removed as the message suggests;
    # what is left is output from a pass that got further than this run goes.
    for name in ("ocn-00000005.nc", "seaice-00000005.nc"):
        (output / name).unlink()
    shutil.copy(output / "ocn-00000004.nc", output / "ocn-00000007.nc")
    before = output_names(output)

    with pytest.raises(ValueError) as raised:
        run_chunked(
            two_slabs(), total_time="6 days", chunk="2 days",
            output_dir=output, checkpoint_path=checkpoint,
        )

    message = str(raised.value)
    assert "ocn-00000007.nc" in message
    assert "at or past coupled step 6, where this run ends" in message
    assert "under a different chunk" not in message
    assert output_names(output) == before


class OutputlessCounter:
    """A component that steps and writes no output: it has no ``to_xarray``.

    The coupler steps it and stacks its diagnostics like any other, and
    ``Coupler.to_xarray`` then skips it -- so no file is ever named after it.
    """

    def __init__(self, name="atm"):
        """Name the component."""
        self.name = name

    def initialize(self):
        return {"value": jnp.float32(0.0)}

    def step(self, carry, time):
        del time
        new_carry = {"value": carry["value"] + 1.0}
        return new_carry, {"value": new_carry["value"]}


def test_a_component_that_writes_no_output_is_not_one_of_the_names(tmp_path):
    """A file named after an output-less component is not this run's.

    `Coupler.to_xarray` skips a component without `to_xarray`, so the run
    never writes a file under its name -- and a file that happens to be called
    after it therefore belongs to something else. Counting the name anyway
    would have that stranger's file refuse a resume the run is perfectly able
    to make, so `_output_names` applies the same `SupportsXarray` test the
    output path does.
    """
    from jem.driver import _output_names

    def coupler():
        return Coupler(
            {"ocn": SlabOceanModel(make_grid()), "atm": OutputlessCounter()},
            {},
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=START_DATE,
        )

    assert _output_names(coupler()) == ["ocn"]

    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    run_chunked(
        coupler(), total_time="2 days", chunk="1 day",
        output_dir=output, checkpoint_path=checkpoint,
    )
    assert output_names(output) == ["ocn-00000000.nc", "ocn-00000001.nc"]
    # Off this resume's chunk grid, and named after the component that writes
    # nothing: not this run's file, so not this run's business.
    foreign = output / "atm-00000003.nc"
    foreign.write_bytes(b"somebody else's output")

    resumed = run_chunked(
        coupler(), total_time="4 days", chunk="2 days",
        output_dir=output, checkpoint_path=checkpoint,
    )

    assert resumed.steps_completed == 4
    assert foreign.read_bytes() == b"somebody else's output"


def test_a_nested_couplers_files_are_named_after_its_inner_components(tmp_path):
    """The check knows the names a nested coupler's output is written under.

    `Coupler.to_xarray` flattens a nested coupler's datasets into the result
    under *its* components' names, so those -- not the name the inner coupler
    is registered under -- are what its files are called. `_output_names`
    follows the nesting for the same reason, or a nested run's files would
    match nothing and an overlapping resume would go unnoticed.
    """
    from jem.driver import _output_names

    grid = make_grid()
    inner = Coupler(
        {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)},
        {},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        name="surface",
    )
    outer = Coupler(
        {"surface": inner},
        {},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
    )

    assert _output_names(outer) == ["ocn", "seaice"]
    # And the names really are what the files are called.
    result = run_chunked(
        outer, total_time="1 day", chunk="1 day", output_dir=tmp_path,
        checkpoint_path=None,
    )
    assert sorted(path.name for path in result.paths) == [
        "ocn-00000000.nc", "seaice-00000000.nc"
    ]


# ---------------------------------------------------------------------------
# Reducing inside the scan instead of writing every step
# ---------------------------------------------------------------------------


def test_an_accumulated_run_equals_one_long_accumulated_trajectory(tmp_path):
    """Three chunks of an accumulated run are one accumulated trajectory.

    This is the property that makes `accumulate` usable from the driver at
    all: the accumulator is threaded from chunk to chunk, so where the chunk
    boundaries fall must not touch the means. The comparison is against the
    same reduction taken in a single call, which is the answer the driver has
    to reproduce.
    """
    monthly = monthly_mean(two_slabs())
    result = run_chunked(
        two_slabs(),
        total_time="9 days",
        chunk="3 days",
        output_dir=tmp_path,
        health_check=None,
        accumulate=monthly,
    )

    reference = two_slabs()
    _, expected = reference.generate_trajectory_function(9, accumulate=monthly)(
        reference.initialize()
    )

    assert result.completed
    assert result.steps_completed == 9
    for got, want in zip(
        jax.tree_util.tree_leaves(monthly.finalize(result.accumulator)),
        jax.tree_util.tree_leaves(monthly.finalize(expected)),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-12)


def test_an_accumulated_run_compiles_one_trajectory_for_every_chunk(
    coupler, tmp_path
):
    """The accumulator is seeded before the first chunk, not left as None.

    `jax.jit` keys its cache on the argument treedefs, so a first chunk called
    with None and a second with the accumulator pytree would compile the same
    scan twice -- minutes, for an atmosphere. The trajectory factory is
    wrapped to count how many distinct functions the driver builds and how
    many times each is called with what, which is what catches a
    reintroduction of the None.
    """
    seen = []
    build = coupler.generate_trajectory_function

    def counting(iterations, **kwargs):
        trajectory = build(iterations, **kwargs)

        def call(carry, accumulator):
            seen.append(accumulator is None)
            return trajectory(carry, accumulator)

        return call

    coupler.generate_trajectory_function = counting
    run_chunked(
        coupler, total_time="6 days", chunk="2 days", output_dir=tmp_path,
        health_check=None, accumulate=monthly_mean(two_slabs()),
    )

    assert len(seen) == 3
    assert not any(seen), "a chunk was handed None instead of an accumulator"


def test_an_accumulated_run_with_nothing_to_do_still_returns_an_accumulator(
    coupler, tmp_path
):
    """Re-running a finished job must not hand `finalize` a None.

    Pointing the same command at a checkpoint already at `total_time` is the
    documented way to confirm a run is done, and it integrates no chunks. The
    accumulator it returns is the reduction's own empty one -- every bin NaN,
    which is what "no steps were accumulated" means -- rather than a None that
    `finalize` cannot unpack.
    """
    monthly = monthly_mean(two_slabs())
    carry, _ = coupler.generate_trajectory_function(2)(coupler.initialize())

    result = run_chunked(
        coupler, total_time="2 days", chunk="2 days", output_dir=tmp_path,
        initial_carry=carry, health_check=None, accumulate=monthly,
    )

    assert result.completed
    assert result.paths == []
    means = monthly.finalize(result.accumulator)
    assert np.all(
        np.isnan(np.asarray(means["ocn"]["state"].sea_surface_temperature))
    )


def test_an_accumulated_run_writes_no_files(coupler, tmp_path, caplog):
    """The reduction is the output: no per-chunk files, and `paths` is empty."""
    with caplog.at_level(logging.INFO, logger="jem.driver"):
        result = run_chunked(
            coupler,
            total_time="4 days",
            chunk="2 days",
            output_dir=tmp_path,
            health_check=None,
            accumulate=monthly_mean(coupler),
        )

    assert result.paths == []
    assert list(tmp_path.glob("*.nc")) == []
    assert "no per-chunk files are written" in caplog.text
    assert "reduced into the accumulator, no files written" in caplog.text


def test_a_run_without_accumulate_has_no_accumulator(coupler, tmp_path):
    """`RunResult.accumulator` is None unless the run was given a reduction."""
    result = run_chunked(
        coupler, total_time="2 days", chunk="2 days", output_dir=tmp_path
    )
    assert result.accumulator is None


def test_accumulate_with_a_health_check_is_refused(coupler, tmp_path):
    """The gate has nothing to look at, so it is refused rather than skipped.

    The gate is on by default, so a run that simply passed `accumulate=` would
    otherwise lose it silently -- and a long accumulated run of an atmosphere
    is exactly the run that needs it. `health_check=None` makes going without
    the gate something the caller decided.
    """
    def must_not_be_called(iterations, **kwargs):
        raise AssertionError("a trajectory was built despite the bad pairing")

    coupler.generate_trajectory_function = must_not_be_called
    with pytest.raises(ValueError, match="health_check=None"):
        run_chunked(
            coupler,
            total_time="2 days",
            chunk="2 days",
            output_dir=tmp_path,
            accumulate=monthly_mean(two_slabs()),
        )


def test_an_accumulated_run_says_the_accumulator_is_not_checkpointed(
    coupler, tmp_path, caplog
):
    """The carry is still checkpointed; the accumulator deliberately is not.

    The checkpoint is the model's restart state and the accumulator is an
    analysis product; putting one in the other would make the checkpoint
    format depend on which reduction a run happened to choose. A first run has
    lost nothing yet, so it is told at INFO -- warning here would fire on
    every accumulated run, which is how a warning stops being read.
    """
    checkpoint = tmp_path / "checkpoint"
    with caplog.at_level(logging.INFO, logger="jem.driver"):
        result = run_chunked(
            coupler,
            total_time="4 days",
            chunk="2 days",
            output_dir=tmp_path,
            checkpoint_path=checkpoint,
            health_check=None,
            accumulate=monthly_mean(coupler),
        )

    assert "The accumulator is not part of the checkpoint" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert result.accumulator is not None
    # The restart state itself is written as usual, and holds no accumulator:
    # the checkpoint of an accumulated run is loadable by a run that asks for
    # no reduction at all, or for a different one.
    assert sorted(p.name for p in checkpoint.iterdir()) == [CARRY_FILENAME]
    assert int(two_slabs().load_carry(checkpoint).step) == 4


def test_a_resumed_accumulated_run_warns_that_its_means_are_partial(
    tmp_path, caplog
):
    """The warning fires where the damage is: the run whose means are partial.

    A run that resumed has an accumulator covering only the chunks this call
    integrated, and the means it returns are not the means of the simulation
    they appear to describe. That is worth a warning, and it is actionable --
    unlike the same sentence on a first run, which has lost nothing.
    """
    checkpoint = tmp_path / "checkpoint"
    monthly = monthly_mean(two_slabs())
    run_chunked(
        two_slabs(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "first", checkpoint_path=checkpoint,
        health_check=None, accumulate=monthly,
    )

    with caplog.at_level(logging.WARNING, logger="jem.driver"):
        resumed = run_chunked(
            two_slabs(), total_time="4 days", chunk="2 days",
            output_dir=tmp_path / "second", checkpoint_path=checkpoint,
            health_check=None, accumulate=monthly,
        )

    assert "covers only the chunks this call integrates" in caplog.text
    # Two of the run's four days were accumulated, and it is the second two.
    _, counts = resumed.accumulator
    assert int(np.sum(np.asarray(counts))) == 2


def test_an_accumulated_run_warns_that_output_reductions_do_nothing(
    coupler, tmp_path, caplog
):
    """`output_averages` and `subsample` reduce files, and there are none."""
    with caplog.at_level(logging.WARNING, logger="jem.driver"):
        run_chunked(
            coupler,
            total_time="2 days",
            chunk="2 days",
            output_dir=tmp_path,
            health_check=None,
            output_averages=True,
            subsample=2,
            accumulate=monthly_mean(coupler),
        )

    assert "they do nothing here" in caplog.text


# ---------------------------------------------------------------------------
# Output options reach the files
# ---------------------------------------------------------------------------


def written_labels(output_dir, name="ocn"):
    """Return every record label a run left in ``output_dir``, in run order.

    The files are named after the coupled step their chunk starts at and
    zero-padded, so sorting them by name is sorting them by run time.
    """
    labels = []
    for path in sorted(pathlib.Path(output_dir).glob(f"{name}-*.nc")):
        with xr.open_dataset(path) as dataset:
            labels += list(dataset["time"].values)
    return labels


def step_labels(*steps):
    """Return the output label of each coupled step, as a record carries it.

    jax-gcm PR 878 labels an averaged record at its interval's MIDPOINT, not
    its end (``docs/source/v2_to_v3.rst``, "One real datetime clock"), and
    ``TimeAxis.datetimes`` follows suit: record ``k`` covers
    ``[start + k day, start + (k+1) day)`` and is labelled at
    ``start + k day + 12h``.
    """
    day = np.timedelta64(1, "D").astype("timedelta64[ns]")
    half_day = np.timedelta64(12, "h").astype("timedelta64[ns]")
    start = np.datetime64("2001-01-01", "ns")
    return [start + step * day + half_day for step in steps]


def chunk_midpoint_labels(*step_ranges):
    """Return the label ``postprocess(output_averages=True)`` gives a chunk mean.

    A chunk's own true midpoint is the average of its first and last
    record's own (midpoint) label -- exact for equal-length, contiguous
    records regardless of the record length itself (``jem.output.postprocess``'s
    own derivation, in its body, for the no-``time_bounds`` case every
    non-JCM component's dataset takes). ``step_ranges`` is ``(first_step,
    last_step)`` pairs of the coupled steps each chunk covers.
    """
    first_steps, last_steps = zip(*step_ranges, strict=True)
    first_labels = step_labels(*first_steps)
    last_labels = step_labels(*last_steps)
    return [
        first + (last - first) // 2
        for first, last in zip(first_labels, last_labels, strict=True)
    ]


def test_subsample_keeps_the_same_records_however_the_run_is_chunked(tmp_path):
    """The stride is the run's, so `chunk` is free to be chosen for memory.

    Six steps with `subsample=2`, run whole and in three-step chunks: both
    keep coupled steps 0, 2 and 4. A stride restarted at each chunk would
    give 0, 2, 3, 5 -- an irregular cadence, and four records where three
    were asked for.
    """
    whole = run_chunked(
        two_slabs(), total_time="6 days", chunk="6 days",
        output_dir=tmp_path / "whole", subsample=2,
    )
    chunked = run_chunked(
        two_slabs(), total_time="6 days", chunk="3 days",
        output_dir=tmp_path / "chunked", subsample=2,
    )

    assert whole.steps_completed == chunked.steps_completed == 6
    assert written_labels(tmp_path / "whole") == step_labels(0, 2, 4)
    assert written_labels(tmp_path / "chunked") == step_labels(0, 2, 4)


def test_subsample_survives_a_resume(tmp_path):
    """A run stopped and resumed writes the records the whole run writes.

    The phase comes from the coupled step in the carry, which the checkpoint
    holds, so the resumed pass continues the stride rather than starting it
    again at its own first chunk -- here mid-period, since it restarts at
    step 3 with `subsample=2`.
    """
    uninterrupted = run_chunked(
        two_slabs(), total_time="6 days", chunk="1 day",
        output_dir=tmp_path / "whole", subsample=2,
    )
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        two_slabs(), total_time="3 days", chunk="1 day",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
        subsample=2,
    )
    resumed = run_chunked(
        two_slabs(), total_time="6 days", chunk="1 day",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
        subsample=2,
    )

    assert uninterrupted.steps_completed == resumed.steps_completed == 6
    assert written_labels(tmp_path / "whole") == step_labels(0, 2, 4)
    assert written_labels(tmp_path / "restarted") == step_labels(0, 2, 4)
    # A one-day chunk whose step the stride drops has nothing to write, so it
    # writes no file: three chunks of the six kept a step, across both passes.
    assert sorted(
        path.name for path in (tmp_path / "restarted").glob("ocn-*.nc")
    ) == ["ocn-00000000.nc", "ocn-00000002.nc", "ocn-00000004.nc"]
    assert sorted(path.name for path in resumed.paths) == [
        "ocn-00000004.nc", "seaice-00000004.nc"
    ]


def test_a_thinned_run_reads_back_as_one_series(tmp_path):
    """The directory a thinned run leaves opens with `open_mfdataset`.

    This is what the output is *for*, and it is why a chunk that keeps no
    step writes no file rather than an empty one: a zero-length time
    dimension makes the default `combine="by_coords"` fail on the whole
    directory, so one skipped chunk would cost the run all of its output.
    """
    run_chunked(
        two_slabs(), total_time="6 days", chunk="1 day",
        output_dir=tmp_path, subsample=2,
    )

    with xr.open_mfdataset(sorted(tmp_path.glob("ocn-*.nc"))) as combined:
        assert combined.sizes["time"] == 3
        np.testing.assert_array_equal(
            combined["time"].values, np.array(step_labels(0, 2, 4))
        )


def test_a_chunk_mean_is_labelled_at_the_chunk_midpoint_however_it_is_thinned(
    tmp_path,
):
    """`output_averages` with `subsample` keeps one evenly spaced mean a chunk.

    Twenty days in four-day chunks with `subsample=3`: the kept coupled steps
    are 0, 3, 6, 9, 12, 15 and 18, which fall 2, 1, 1, 2, 1 to a chunk -- so
    the means are over different numbers of records, which the same run
    without the averaging shows file by file. Each mean still covers its own
    chunk -- steps 0-3, 4-7, 8-11, 12-15, 16-19 -- and is labelled at that
    chunk's own true MIDPOINT (the average of the chunk's first and last
    record's own label; see `jem.output.postprocess`'s module docstring and
    `chunk_midpoint_labels` above), four days apart; labelling with the last
    record the stride happened to keep would make the series jump about
    instead, and labelling with the chunk's own last record's label would be
    neither the chunk's end nor its midpoint once every record's own label
    is itself a midpoint (jax-gcm PR 878).
    """
    settings = {"total_time": "20 days", "chunk": "4 days", "subsample": 3}
    thinned = tmp_path / "thinned"
    averaged = tmp_path / "averaged"
    run_chunked(two_slabs(), output_dir=thinned, **settings)
    run_chunked(two_slabs(), output_dir=averaged, output_averages=True, **settings)

    # Unequal weighting, from the data: the records that went into each mean.
    kept = []
    for path in sorted(thinned.glob("ocn-*.nc")):
        with xr.open_dataset(path) as records:
            kept.append(records["sea_surface_temperature"].load())
    assert [records.sizes["time"] for records in kept] == [2, 1, 1, 2, 1]

    # And each mean is over exactly those records, labelled at its chunk's
    # own true midpoint -- five means, four days apart, whatever went into
    # them (the stride chooses what is AVERAGED, never what interval the
    # mean's own label covers).
    assert written_labels(averaged) == chunk_midpoint_labels(
        (0, 3), (4, 7), (8, 11), (12, 15), (16, 19)
    )
    for path, records in zip(sorted(averaged.glob("ocn-*.nc")), kept, strict=True):
        with xr.open_dataset(path) as mean:
            assert mean.sizes["time"] == 1
            assert (
                "time: mean"
                in mean["sea_surface_temperature"].attrs["cell_methods"]
            )
            np.testing.assert_allclose(
                mean["sea_surface_temperature"].values[0],
                records.mean(dim="time").values,
                rtol=1e-6,
            )


def test_a_sub_stepped_component_is_thinned_by_coupled_step(tmp_path):
    """All of a kept step's records are written, and none of a dropped one's."""
    grid = make_grid()
    components = {"ocn": SlabOceanModel(grid), "seaice": SlabSeaiceModel(grid)}
    exchangers = default_exchangers(components)
    weaved = Coupler(
        components,
        exchangers,
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        # The ocean runs twice per coupled step, so it records twice a step
        # while the sea ice records once.
        workflow=[list(exchangers), ["ocn"] * 2, "seaice"],
    )

    run_chunked(
        weaved, total_time="4 days", chunk="2 days",
        output_dir=tmp_path, subsample=2,
    )

    day = np.timedelta64(1, "D").astype("timedelta64[ns]")
    start = np.datetime64("2001-01-01", "ns")
    quarter_day = np.timedelta64(6, "h").astype("timedelta64[ns]")
    # Coupled steps 0 and 2: the ocean's two half-day (12h) sub-step records
    # for each of them, each labelled at ITS OWN midpoint -- 6h and 18h into
    # the coupled day -- not derived from the daily `step_labels` (jax-gcm PR
    # 878; see that function's docstring). The sea ice records once a day, so
    # it uses `step_labels` unchanged.
    assert written_labels(tmp_path, "ocn") == sorted(
        [start + step * day + quarter_day for step in (0, 2)]
        + [start + step * day + 3 * quarter_day for step in (0, 2)]
    )
    assert written_labels(tmp_path, "seaice") == step_labels(0, 2)


def test_output_options_reach_the_files(coupler, tmp_path):
    """`output_averages` and `subsample` are passed through to the postprocessing."""
    averaged = run_chunked(
        coupler, total_time="4 days", chunk="4 days",
        output_dir=tmp_path / "averaged", output_averages=True,
    )
    dataset = xr.open_dataset(averaged.paths[0])
    assert dataset.sizes["time"] == 1
    assert "time: mean" in dataset["sea_surface_temperature"].attrs["cell_methods"]

    thinned = run_chunked(
        two_slabs(), total_time="4 days", chunk="4 days",
        output_dir=tmp_path / "thinned", subsample=2,
    )
    assert xr.open_dataset(thinned.paths[0]).sizes["time"] == 2


# ---------------------------------------------------------------------------
# The same properties with the real atmosphere
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_continuous_chunked_resumed_agree_with_jcm(tmp_path):
    """The chunk/restart invariance holds for a real coupled model too.

    The toy coupler above pins the arithmetic exactly; this pins the thing the
    arithmetic is for -- an atmosphere whose cross-step physics carry, dycore
    state and forcing all have to survive a chunk boundary and a round trip
    through a checkpoint file. The tolerance is 1e-6 because the atmosphere
    integrates in float32.
    """
    import jcm
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.terrain import TerrainData

    from jem.components import JCMComponent
    from jem.components.slab import SlabGrid

    def build() -> Coupler:
        coords = get_speedy_coords(layers=5, spectral_truncation=21)
        model = jcm.model.Model(
            coords=coords,
            terrain=TerrainData.aquaplanet(coords),
            start_time=START_DATE,
        )
        atm = JCMComponent(model)
        components = {
            "atm": atm,
            "ocn": SlabOceanModel(SlabGrid.from_coords(coords.horizontal)),
        }
        return Coupler(
            components,
            default_exchangers(components),
            coupling_timestep=COUPLING_TIMESTEP,
            start_date=START_DATE,
            calendar="gregorian",
        )

    continuous = run_chunked(
        build(), total_time="4 days", chunk="4 days",
        output_dir=tmp_path / "continuous",
    )
    checkpoint = tmp_path / "checkpoint"
    run_chunked(
        build(), total_time="2 days", chunk="2 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )
    resumed = run_chunked(
        build(), total_time="4 days", chunk="2 days",
        output_dir=tmp_path / "restarted", checkpoint_path=checkpoint,
    )

    assert resumed.steps_completed == continuous.steps_completed == 4
    assert_carries_agree(resumed.final_carry, continuous.final_carry, atol=1e-6)
    # The atmosphere is there, so the default gate actually looked at it.
    assert all("skipped" not in report for report in continuous.reports)


@pytest.mark.slow
def test_run_smoke_cli(tmp_path):
    """`python -m jem.main` runs the shipped `short_run` option end to end.

    A subprocess, in a scratch working directory, because that is what a user
    types: it exercises Hydra's composition from the installed package, the
    resolvers the shipped configurations use, the run directory Hydra makes
    and the driver's own logging -- none of which an in-process call of
    `runners.run` would touch.
    """
    import os
    import subprocess
    import sys

    repository = pathlib.Path(__file__).resolve().parents[2]
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    finished = subprocess.run(
        [sys.executable, "-m", "jem.main",
         "+configuration=aquaplanet-slab", "coupled_run=short_run"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=1800,
    )
    assert finished.returncode == 0, finished.stderr[-4000:]

    run_directories = sorted((tmp_path / "outputs").glob("*/*"))
    assert len(run_directories) == 1, run_directories
    written = sorted(path.name for path in run_directories[0].glob("*.nc"))
    assert written == [
        "atm-00000000.nc", "ocn-00000000.nc", "seaice-00000000.nc",
    ]

    # The run said what it did, at INFO, through the logger rather than print.
    log = (run_directories[0] / "main.log").read_text()
    assert "Chunk 0:" in log
    assert "Finished: 2 coupled steps, completed=True, 3 file(s) written." in log
