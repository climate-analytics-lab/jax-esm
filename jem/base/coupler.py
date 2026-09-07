"""The :class:`Coupler`: the coupled-model definition, its clock and its step.

A ``Coupler`` is the whole definition of a coupled model:

- **what is coupled** -- the named components and the named exchangers that
  move information between them;
- **in what order** -- the ``workflow``, one ordered list over the single
  namespace shared by exchangers and components. It may be nested, and a name
  may appear more than once, which is how a component runs on a faster clock
  than the coupled one;
- **on what clock** -- the coupling timestep, start date and calendar. The
  coupler owns the only clock in the system. Components hold no time state of
  their own; each ``step`` is handed a :class:`~jem.base.component.CouplingTime`
  built from the step counter that lives in the carry, so two components can
  never disagree about the date, and the date survives chunked runs and
  checkpoint restarts (a ``lax.scan`` index restarts at zero on every call; the
  carry does not).

The coupler produces *functions*, not runs: :meth:`Coupler.step_function` and
:meth:`Coupler.generate_trajectory_function` return pure functions of the
carry, which the caller composes with ``jax.jit``, ``jax.grad`` or a chunked
run loop. Nothing here logs per step, holds a trajectory or mutates the
coupler, because all of that would either break under tracing or make the
returned function impure.

See ``docs/source/design/architecture.md`` and, for the task numbering
(T1.1, T1.3, T1.7), the API hardening plan, which lives on the review
branch rather than in this repository:
https://github.com/climate-analytics-lab/jax-esm/blob/claude/jax-esm-api-review-jv7j7u/docs/source/design/api_hardening_plan.md
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr
from jcm.date import days_per_year as jcm_days_per_year

from jem.base.component import (
    Carry,
    Component,
    CoupledCarry,
    CouplingTime,
    Diagnostics,
    Exchanger,
    SupportsBind,
    SupportsXarray,
    TimeAxis,
    seconds_since_new_year,
)

logger = logging.getLogger(__name__)

# The attributes the Component protocol requires. `isinstance(x, Component)`
# is the check that decides acceptance; this tuple exists only so the error
# message can say *which* part of the contract is missing, which
# `isinstance` on its own cannot report.
_REQUIRED_COMPONENT_ATTRIBUTES = ("name", "initialize", "step")

_SECONDS_PER_DAY = 86400.0


def _missing_component_attributes(component: Any) -> list[str]:
    """Return the names of the :class:`Component` members ``component`` lacks."""
    return [
        attribute
        for attribute in _REQUIRED_COMPONENT_ATTRIBUTES
        if not hasattr(component, attribute)
    ]


def _flatten_workflow(workflow: Any) -> Iterator[str]:
    """Yield the names in an arbitrarily nested sequence of workflow entries.

    A workflow is written the way a coupling scheme is described on paper --
    ``[["atm_lnd_exchange", "atm", "lnd"] * 24, "atm_ocn_exchange", "ocn"]``
    is a day of hourly land-atmosphere coupling followed by one ocean
    exchange -- and the nesting is purely notation: the coupler runs the flat
    sequence. Flattening here, once, keeps every other part of the coupler
    (validation, multiplicity, the step loop) working on one flat tuple.

    Strings are the leaves. Any other leaf is a ``TypeError``, because the
    alternative -- iterating it -- would silently turn a stray object into a
    sequence of characters or of whatever it happens to contain.
    """
    if isinstance(workflow, str):
        yield workflow
        return
    if isinstance(workflow, Sequence):
        for entry in workflow:
            yield from _flatten_workflow(entry)
        return
    raise TypeError(
        f"Workflow entries must be component or exchanger names (strings), or "
        f"sequences of them; got {workflow!r}."
    )


def _compressed_workflow(workflow: Sequence[str]) -> str:
    """Render a flat workflow with its repeated blocks collapsed, for a repr.

    A workflow with multiplicity is long -- 24 hourly land-atmosphere blocks
    is 72 entries -- and printing it flat buries the two entries that are not
    part of the cycle. The shortest block that repeats at each position is
    collapsed to ``[...] * n``, which is how the workflow was written in the
    first place.
    """
    parts: list[str] = []
    index = 0
    total = len(workflow)
    while index < total:
        for length in range(1, (total - index) // 2 + 1):
            block = list(workflow[index : index + length])
            repeats = 1
            while (
                list(workflow[index + repeats * length : index + (repeats + 1) * length])
                == block
            ):
                repeats += 1
            if repeats > 1:
                parts.append(f"{block!r} * {repeats}")
                index += repeats * length
                break
        else:
            parts.append(repr(workflow[index]))
            index += 1
    return ", ".join(parts)


def _timedelta_seconds(delta: jdt.Timedelta) -> int:
    """Return a ``jdt.Timedelta`` as a whole number of seconds.

    ``jdt.Timedelta`` is integer-backed (whole days plus whole seconds within
    the day), so this is exact -- which is what the multiplicity arithmetic
    needs: a sub-step of ``coupling_timestep / n`` has to be expressible as a
    ``Timedelta`` again, and only whole seconds are.
    """
    return int(np.asarray(delta.days)) * int(_SECONDS_PER_DAY) + int(
        np.asarray(delta.seconds)
    )


def _leading_axis_length(diagnostics: Diagnostics, name: str) -> int:
    """Return the number of output records in a component's stacked diagnostics.

    ``lax.scan`` gives every leaf a leading axis of length ``iterations``, so
    any leaf answers the question; the first one is used.
    """
    leaves = jax.tree_util.tree_leaves(diagnostics)
    if not leaves:
        raise ValueError(
            f"Component {name!r} returned diagnostics with no arrays, so the "
            "number of output records cannot be determined."
        )
    shape = jnp.shape(leaves[0])
    if not shape:
        raise ValueError(
            f"Component {name!r} returned a scalar diagnostics leaf; stacked "
            "diagnostics must have a leading time axis."
        )
    return int(shape[0])


def _merge_leading_axes(diagnostics: Diagnostics, name: str, expected: int) -> Diagnostics:
    """Fold a ``(records, expected, ...)`` diagnostics pytree to ``(records * expected, ...)``.

    A component that runs ``expected`` times per coupled step has its
    per-call diagnostics stacked on a new leading axis, and ``lax.scan`` then
    stacks *that* over the coupled steps, so a trajectory hands back
    ``(steps, expected, ...)``. The records are consecutive in time -- call
    ``k`` of step ``s`` covers sub-step ``s * expected + k`` -- so flattening
    the two axes in C order is exactly the run's own time order.
    """

    def merge(leaf: Any) -> Any:
        array = jnp.asarray(leaf)
        if array.ndim < 2 or array.shape[1] != expected:
            raise ValueError(
                f"Component {name!r} runs {expected} times per coupled step, so its "
                f"stacked diagnostics must have shape (steps, {expected}, ...); got "
                f"{array.shape}. (These diagnostics did not come from a trajectory "
                "of this coupler.)"
            )
        return array.reshape((array.shape[0] * expected, *array.shape[2:]))

    return jax.tree_util.tree_map(merge, diagnostics)


class Coupler:
    """A coupled model: named components, named exchangers, an order and a clock.

    Parameters
    ----------
    components : dict[str, Component]
        The components to couple. The **dict key is the component's name**
        throughout: it keys the carries in :class:`CoupledCarry`, the
        diagnostics returned by a step, and the entries of ``workflow``. A
        value that does not satisfy the :class:`~jem.base.component.Component`
        protocol raises ``TypeError``.
    exchangers : dict[str, Exchanger], optional
        The functions that move information between components. They share
        one namespace with the components, so a name may not be used twice.
    coupling_timestep : jdt.Timedelta
        The coupled timestep: how far the whole system advances per step.
        A ``jdt.Timedelta`` holds whole seconds, so the shortest step
        expressible today is one second -- fine for every geoscience
        configuration, but a wall for a non-geoscience component (the spring
        example in ``examples/03_non_geoscience/`` has to be rescaled because
        of it). Only ``dt_seconds`` is used downstream, so accepting a float
        number of seconds would lift the limit: jax-esm#110.
    start_date : jdt.Datetime
        Date of coupled step 0.
    calendar : str
        Calendar name as JCM spells it; determines the length of the year
        used for the annual cycle (``jcm.date.days_per_year``).
    workflow : Sequence, optional
        The order in which exchangers and components run within one coupled
        step. Defaults to every exchanger (in insertion order) followed by
        every component (in insertion order), i.e. information is exchanged
        first and the components then all see the same exchanged state.

        The sequence may be **nested** to any depth -- it is flattened at
        construction, and :attr:`workflow` is the flat tuple -- so a coupling
        scheme can be written the way it is described::

            workflow=[["atm_lnd_exchange", "atm", "lnd"] * 24,
                      "atm_ocn_exchange", "ocn"]

        Any name must be a registered exchanger or component, and a name may
        appear more than once: an element listed ``n`` times runs ``n`` times
        per coupled step, on a clock ``n`` times faster (see :meth:`bind` on
        the components and :meth:`coupling_time_at_substep`).

    Notes
    -----
    Coupling is **lagged**: with the default workflow the exchanger at step
    *n* moves the fields each component produced during step *n-1*, so the
    first step of a run exchanges the values from ``initialize()``.

    **Multiplicity.** A component or exchanger listed ``n > 1`` times in the
    workflow lives on a clock of ``coupling_timestep / n``, which must be a
    whole number of seconds. A bindable component is bound with that
    sub-timestep rather than with the coupled one, so a component with an
    internal timestep sub-cycles the right number of times, and each of its
    ``n`` calls in a coupled step is handed the clock of its own sub-step.
    Its diagnostics come back stacked on a new leading axis of length ``n``
    and :meth:`to_xarray` labels them at the sub-rate, so a repeated
    component writes ``n`` output records per coupled step. Everything about
    ``n == 1`` -- the clock a component sees, the shape of its diagnostics,
    its time axis -- is exactly as it is for a coupler with no multiplicity
    at all.

    """

    def __init__(
        self,
        components: dict[str, Component],
        exchangers: dict[str, Exchanger] | None = None,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str = "365_day",
        workflow: Sequence[Any] | None = None,
    ):
        """Build a coupled model; see the class docstring for the parameters."""
        self.components: dict[str, Component] = {}
        self.exchangers: dict[str, Exchanger] = {}

        self._coupling_timestep = coupling_timestep
        self._start_date = start_date
        self._calendar = calendar
        # Resolved once, in the constructor: these are the static facts every
        # CouplingTime is built from, and recomputing them per step would put
        # calendar arithmetic inside a traced function.
        self._dt_seconds = float(coupling_timestep / jdt.to_timedelta(1, "second"))
        # Only components with an internal timestep (JCM, Veros) check the
        # coupling timestep in `bind`; a slab-only coupler would otherwise
        # accept zero (year_fraction divides by dt; every record gets the
        # same timestamp) or a negative value (slab physics integrated
        # backwards) without complaint.
        if not self._dt_seconds > 0.0:
            raise ValueError(
                f"coupling_timestep must be positive; got {coupling_timestep!r} "
                f"({self._dt_seconds:g} s)."
            )
        # The exact integer form of the same quantity: dividing it by a
        # multiplicity is what decides whether an element's sub-timestep is
        # expressible as a `jdt.Timedelta` at all.
        self._dt_total_seconds = _timedelta_seconds(coupling_timestep)
        self._year_offset_seconds = seconds_since_new_year(start_date, calendar)
        self._days_per_year = float(jcm_days_per_year(calendar))

        for name, exchanger in (exchangers or {}).items():
            self.add_exchanger(name, exchanger)
        for name, component in components.items():
            self._register_component(name, component)

        # A workflow given explicitly is validated now, so a typo is a
        # construction error rather than a trace-time one; it is validated
        # again by `step_function`, because components may be added or
        # removed after construction.
        self._explicit_workflow: tuple[str, ...] | None = None
        if workflow is not None:
            self._explicit_workflow = self._validated_workflow(workflow)

        # Binding happens only now, because a component's clock is the
        # coupled timestep divided by how many times the workflow runs it,
        # and that is not known until the workflow is resolved. A component
        # the explicit workflow never names is not bound at all: it will
        # never be stepped, so there is no clock to give it.
        for name, component in self.components.items():
            multiplicity = self._multiplicity(name)
            if multiplicity:
                self._bind_component(name, component, multiplicity)

        logger.debug("Coupled workflow: %s", ", ".join(self.workflow))

    # -- the coupled model definition --------------------------------------

    def _register_component(self, name: str, component: Component) -> None:
        """Check ``component`` against the contract and store it under ``name``."""
        if not isinstance(component, Component):
            missing = _missing_component_attributes(component)
            raise TypeError(
                f"Component {name!r} ({type(component).__name__}) does not satisfy the "
                f"Component protocol: missing {', '.join(missing) or 'nothing (see jem.base.component)'}. "
                "A component must have a `name`, an `initialize()` and a `step(carry, time)`."
            )
        if name in self.exchangers:
            raise ValueError(
                f"The name {name!r} is already used by an exchanger; components and "
                "exchangers share one namespace."
            )
        self.components[name] = component

    def _bind_component(
        self, name: str, component: Component, multiplicity: int
    ) -> None:
        """Hand ``component`` the clock it runs on, if it asks for one.

        Components with an internal timestep (JCM, Veros) need the coupling
        timestep and must agree with the coupler about the start date and
        calendar. The timestep they are given is the one *they* advance by,
        ``coupling_timestep / multiplicity``, not the coupled one: a
        component the workflow runs 24 times per coupled step advances an
        hour at a time in a daily coupler, and binding it with the day would
        make it integrate 24 days per coupled step.
        """
        if isinstance(component, SupportsBind):
            component.bind(
                coupling_timestep=self._element_timestep(name, multiplicity),
                start_date=self._start_date,
                calendar=self._calendar,
            )

    def add_component(self, name: str, component: Component) -> None:
        """Register ``component`` under ``name``.

        The object itself is stored -- there is no wrapper -- so
        ``coupler.components[name] is component``.

        A component added after construction is bound with the timestep the
        *current* workflow implies for it: the coupled timestep divided by
        how many times an explicit workflow names it, or the coupled timestep
        itself when nothing names it yet -- which is the case for the default
        workflow, since that is derived from the components and this one is
        not registered yet. Binding happens **before** registering, so a
        component that rejects the clock never enters the coupler: a caller
        that catches the ``ValueError`` keeps a usable coupler, and a
        previously valid component under the same name is not replaced.

        Raises
        ------
        TypeError
            If ``component`` does not satisfy the :class:`Component`
            protocol; the message names the missing attributes.
        ValueError
            If ``name`` is already an exchanger, or the component refuses the
            clock.

        """
        if not isinstance(component, Component):
            missing = _missing_component_attributes(component)
            raise TypeError(
                f"Component {name!r} ({type(component).__name__}) does not satisfy the "
                f"Component protocol: missing {', '.join(missing) or 'nothing (see jem.base.component)'}. "
                "A component must have a `name`, an `initialize()` and a `step(carry, time)`."
            )
        if name in self.exchangers:
            raise ValueError(
                f"The name {name!r} is already used by an exchanger; components and "
                "exchangers share one namespace."
            )
        self._bind_component(name, component, max(self._multiplicity(name), 1))
        self.components[name] = component

    def remove_component(self, name: str) -> None:
        """Remove the component registered under ``name`` if there is one."""
        self.components.pop(name, None)

    def add_exchanger(self, name: str, exchanger: Exchanger) -> None:
        """Register ``exchanger`` under ``name``.

        Raises
        ------
        ValueError
            If ``name`` is already a component.

        """
        if name in self.components:
            raise ValueError(
                f"The name {name!r} is already used by a component; components and "
                "exchangers share one namespace."
            )
        self.exchangers[name] = exchanger

    def remove_exchanger(self, name: str) -> None:
        """Remove the exchanger registered under ``name`` if there is one."""
        self.exchangers.pop(name, None)

    @property
    def workflow(self) -> tuple[str, ...]:
        """The flat order in which exchangers and components run in a coupled step.

        A nested ``workflow=`` argument is flattened at construction; this is
        always the flat tuple actually executed, in which a repeated element
        appears once per run.
        """
        if self._explicit_workflow is not None:
            return self._explicit_workflow
        return (*self.exchangers, *self.components)

    def multiplicities(self) -> dict[str, int]:
        """Return how many times each element runs per coupled step."""
        return dict(collections.Counter(self.workflow))

    def _multiplicity(self, name: str) -> int:
        """Return how many times ``name`` runs per coupled step (0 if never)."""
        return collections.Counter(self.workflow)[name]

    def _sub_timestep(self, multiplicity: int) -> jdt.Timedelta | None:
        """Return ``coupling_timestep / multiplicity``, or None if it is not whole.

        ``jdt.Timedelta`` is integer-backed, so a sub-timestep that is not a
        whole number of seconds cannot be represented and would have to be
        silently rounded -- which would desynchronise the sub-stepped
        component from the coupled clock a little more every step. The
        callers refuse it instead; this one only reports it, so that each can
        name what it was dividing.
        """
        if multiplicity == 1:
            # The coupler's own object, not an equal one: a component that
            # compares the timestep it is bound to sees exactly what a
            # coupler without multiplicity would give it.
            return self._coupling_timestep
        if multiplicity < 1:
            return None
        seconds, remainder = divmod(self._dt_total_seconds, multiplicity)
        if remainder or seconds < 1:
            return None
        return jdt.to_timedelta(seconds, "second")

    def _element_timestep(self, name: str, multiplicity: int) -> jdt.Timedelta:
        """Return the timestep an element listed ``multiplicity`` times advances by."""
        sub_timestep = self._sub_timestep(multiplicity)
        if sub_timestep is None:
            raise ValueError(
                f"Workflow element {name!r} appears {multiplicity} times per coupled "
                f"step, so it runs on a timestep of {self._dt_total_seconds}/"
                f"{multiplicity} s, which is not a whole number of seconds. A "
                "coupling timestep must divide exactly by the number of times an "
                "element runs."
            )
        return sub_timestep

    def _validated_workflow(self, workflow: Sequence[Any]) -> tuple[str, ...]:
        """Flatten ``workflow`` and check every name is known and can be sub-stepped."""
        flat = tuple(_flatten_workflow(workflow))
        for name in flat:
            if name not in self.components and name not in self.exchangers:
                raise ValueError(
                    f"Workflow entry {name!r} is neither a component "
                    f"({sorted(self.components)}) nor an exchanger "
                    f"({sorted(self.exchangers)})."
                )
        for name, multiplicity in collections.Counter(flat).items():
            # Raises if the implied sub-timestep is not a whole number of
            # seconds, so an impossible multiplicity is a construction error
            # rather than something the first traced step discovers.
            self._element_timestep(name, multiplicity)
        return flat

    # -- the clock ---------------------------------------------------------

    @property
    def coupling_timestep(self) -> jdt.Timedelta:
        """The coupled timestep."""
        return self._coupling_timestep

    @property
    def start_date(self) -> jdt.Datetime:
        """The date of coupled step 0."""
        return self._start_date

    @property
    def calendar(self) -> str:
        """The run's calendar, as JCM spells it."""
        return self._calendar

    @property
    def dt_seconds(self) -> float:
        """The coupled timestep in seconds."""
        return self._dt_seconds

    @property
    def year_offset_seconds(self) -> float:
        """Seconds from 1 January of the start year to :attr:`start_date`."""
        return self._year_offset_seconds

    @property
    def days_per_year(self) -> float:
        """Length of the year in days for :attr:`calendar`."""
        return self._days_per_year

    def coupling_time(self, step: Any) -> CouplingTime:
        """Return the clock as a component sees it at coupled step ``step``.

        Parameters
        ----------
        step : int or jax.Array
            Number of coupled steps completed before the step in question.

        """
        step_array = jnp.asarray(step, dtype=jnp.int32)
        return CouplingTime(
            step=step_array,
            sim_time=step_array * self._dt_seconds,
            dt=self._dt_seconds,
            year_offset_seconds=self._year_offset_seconds,
            days_per_year=self._days_per_year,
        )

    def coupling_time_at_substep(
        self, step: Any, call: int, multiplicity: int
    ) -> CouplingTime:
        """Return the clock for call ``call`` of an element that runs ``multiplicity`` times.

        An element listed ``multiplicity`` times in the workflow runs on a
        clock ``multiplicity`` times faster than the coupled one, so its
        ``dt`` is the sub-timestep and its step counter counts sub-steps:
        call ``k`` of coupled step ``s`` is sub-step ``s * multiplicity + k``.
        The arithmetic is on the int32 step counter, so it is exact, and
        ``CouplingTime.year_fraction`` keeps its exact integer reduction at
        the sub-rate too (an hourly sub-step still divides a 365-day year).

        ``multiplicity == 1`` returns exactly :meth:`coupling_time`, so a
        workflow without multiplicity is byte for byte the model it was
        before multiplicity existed.

        Parameters
        ----------
        step : int or jax.Array
            Number of *coupled* steps completed before the step in question.
        call : int
            Which of the element's calls within the coupled step, from 0.
        multiplicity : int
            How many times the element runs per coupled step.

        """
        if multiplicity == 1:
            return self.coupling_time(step)
        sub_dt = self._dt_seconds / multiplicity
        substep = jnp.asarray(step, dtype=jnp.int32) * multiplicity + call
        return CouplingTime(
            step=substep,
            sim_time=substep * sub_dt,
            dt=sub_dt,
            year_offset_seconds=self._year_offset_seconds,
            days_per_year=self._days_per_year,
        )

    def time_axis(self, first_step: int, n: int, *, multiplicity: int = 1) -> TimeAxis:
        """Return the datetimes of ``n`` output records starting at ``first_step``.

        Parameters
        ----------
        first_step : int
            Index of the first record's step, counted in the steps of the
            axis being built -- coupled steps for ``multiplicity == 1``,
            sub-steps otherwise.
        n : int
            Number of records.
        multiplicity : int
            How many records the component writes per coupled step; the axis
            is spaced at ``coupling_timestep / multiplicity``.

        """
        dt = self._sub_timestep(multiplicity)
        if dt is None:
            raise ValueError(
                f"A time axis at {multiplicity} records per coupled step needs a "
                f"record interval of {self._dt_total_seconds}/{multiplicity} s, "
                "which is not a whole number of seconds."
            )
        return TimeAxis(
            start_date=self._start_date,
            steps=np.arange(first_step, first_step + n),
            dt=dt,
            calendar=self._calendar,
        )

    # -- the coupled model as a function -----------------------------------

    def initialize(self) -> CoupledCarry:
        """Build the initial coupled carry: every component's carry, and step 0."""
        return CoupledCarry(
            components={
                name: component.initialize()
                for name, component in self.components.items()
            },
            step=jnp.int32(0),
        )

    def step_function(
        self,
    ) -> Callable[[CoupledCarry], tuple[CoupledCarry, dict[str, Diagnostics]]]:
        """Return the pure function that advances the coupled model one step.

        The returned function takes a :class:`CoupledCarry` and returns the
        new carry (with ``step`` incremented) and one diagnostics pytree per
        component that ran. It never mutates its argument: the carries dict
        is rebuilt, not updated in place, so the caller's carry remains
        valid, which is what makes re-running a step or differentiating
        through it safe.

        Notes
        -----
        The loop over the workflow is ordinary Python, run once at trace
        time, so which call of a repeated element this is -- ``k`` of ``n``
        -- is a static number and can select that call's clock. A component
        that runs ``n > 1`` times has its ``n`` diagnostics stacked on a new
        leading axis of length ``n``, in the order they were produced; for
        ``n == 1`` the diagnostics are handed back exactly as the component
        returned them, so nothing about a workflow without multiplicity
        changes.

        After each workflow element the pytree structure of the carries dict
        is compared with the structure it had on entry, and a change is a
        ``RuntimeError``. The comparison happens at trace time, so it costs
        nothing per step, and it turns what would otherwise be an opaque
        ``lax.scan`` structure error into one that names the element
        responsible.

        """
        workflow = self._validated_workflow(self.workflow)
        multiplicity = collections.Counter(workflow)
        # Snapshot the coupled model: the returned function is a description
        # of the model as it stands now, so registering a component later
        # cannot silently change what an already-generated (and possibly
        # already-compiled) step does.
        components_by_name = dict(self.components)
        exchangers_by_name = dict(self.exchangers)

        def step(carry: CoupledCarry) -> tuple[CoupledCarry, dict[str, Diagnostics]]:
            # Built once and shared by every element of multiplicity 1, so a
            # workflow without multiplicity traces exactly the operations it
            # traced before there was any.
            coupled_time = self.coupling_time(carry.step)
            components: dict[str, Carry] = dict(carry.components)
            # Typed as Any because `jax.tree_util.tree_structure` returns an
            # opaque PyTreeDef that static analysis cannot compare.
            expected_structure: Any = jax.tree_util.tree_structure(components)
            collected: dict[str, list[Diagnostics]] = {}
            calls: collections.Counter = collections.Counter()

            for name in workflow:
                runs = multiplicity[name]
                call = calls[name]
                calls[name] += 1
                time = (
                    coupled_time
                    if runs == 1
                    else self.coupling_time_at_substep(carry.step, call, runs)
                )

                if name in exchangers_by_name:
                    # The exchanger gets a fresh dict, so whatever it does
                    # with it cannot reach the caller's carry.
                    exchanged = exchangers_by_name[name](dict(components), time)
                    if not isinstance(exchanged, dict):
                        raise TypeError(
                            f"Exchanger {name!r} returned {type(exchanged).__name__}; "
                            "an exchanger must return the mapping of component carries."
                        )
                    components = exchanged
                else:
                    new_carry, component_diagnostics = components_by_name[name].step(
                        components[name], time
                    )
                    components = dict(components, **{name: new_carry})
                    collected.setdefault(name, []).append(component_diagnostics)

                structure: Any = jax.tree_util.tree_structure(components)
                if structure != expected_structure:
                    raise RuntimeError(
                        f"Workflow element {name!r} changed the structure of the "
                        f"component carries, which `lax.scan` cannot carry.\n"
                        f"  before: {expected_structure}\n"
                        f"  after:  {structure}"
                    )

            diagnostics = {
                name: (
                    per_call[0]
                    if len(per_call) == 1
                    else jax.tree_util.tree_map(
                        lambda *arrays: jnp.stack(arrays), *per_call
                    )
                )
                for name, per_call in collected.items()
            }

            # `dataclasses.replace` rather than a bare constructor so that a
            # field added to CoupledCarry later is carried through untouched.
            new_carry = dataclasses.replace(
                carry, components=components, step=carry.step + 1
            )
            return new_carry, diagnostics

        return step

    def generate_trajectory_function(
        self,
        iterations: int,
        *,
        remat: bool = False,
        jit: bool = True,
    ) -> Callable[[CoupledCarry], tuple[CoupledCarry, dict[str, Diagnostics]]]:
        """Return the function that runs ``iterations`` coupled steps.

        Parameters
        ----------
        iterations : int
            Number of coupled steps per call.
        remat : bool
            Wrap the coupled step in ``jax.checkpoint``, trading recomputation
            for memory when differentiating through a long trajectory.
        jit : bool
            Wrap the trajectory in ``jax.jit``.

        Returns
        -------
        Callable
            ``carry -> (final_carry, diagnostics)``, where every diagnostics
            leaf has a leading axis of length ``iterations`` (``lax.scan``
            stacks them); a component the workflow runs ``n > 1`` times per
            coupled step has a second axis of length ``n`` after it. The
            clock is the carry's own ``step``, not the scan index, so calling
            the function twice continues the run rather than restarting it.

        """
        step = self.step_function()

        def scan_body(
            carry: CoupledCarry, _: None
        ) -> tuple[CoupledCarry, dict[str, Diagnostics]]:
            return step(carry)

        body = jax.checkpoint(scan_body) if remat else scan_body

        def trajectory(
            carry: CoupledCarry,
        ) -> tuple[CoupledCarry, dict[str, Diagnostics]]:
            # No `xs`: the steps are identical and the only per-step input,
            # the clock, is derived from the carry.
            return jax.lax.scan(body, carry, xs=None, length=iterations)

        return jax.jit(trajectory) if jit else trajectory

    # -- output ------------------------------------------------------------

    def to_xarray(
        self,
        diagnostics: dict[str, Diagnostics],
        *,
        first_step: int = 0,
    ) -> dict[str, xr.Dataset]:
        """Convert stacked diagnostics to one dataset per component that can.

        Components that do not implement
        :class:`~jem.base.component.SupportsXarray` are skipped, so a coupled
        model with an output-less component still produces output. Every
        dataset is labelled with a :class:`TimeAxis` built from the same
        start date and coupled timestep, so the results can be merged.

        A component the workflow runs ``n > 1`` times per coupled step wrote
        ``n`` records per step, arriving as ``(steps, n, ...)``: the two
        leading axes are folded into one -- they are already in time order --
        and the component is handed a time axis spaced at
        ``coupling_timestep / n`` and starting at sub-step
        ``first_step * n``, so its records are stamped at the end of each
        sub-step rather than all at the end of the coupled step.

        Parameters
        ----------
        diagnostics : dict[str, Diagnostics]
            The diagnostics returned by a trajectory function.
        first_step : int
            The coupled step the first record covers; the step counter of the
            carry the trajectory started from.

        """
        datasets: dict[str, xr.Dataset] = {}
        for name, component in self.components.items():
            if name not in diagnostics:
                continue
            if not isinstance(component, SupportsXarray):
                logger.debug("Component %r has no to_xarray; skipping its output.", name)
                continue
            component_diagnostics = diagnostics[name]
            runs = max(self._multiplicity(name), 1)
            n_records = _leading_axis_length(component_diagnostics, name) * runs
            if runs > 1:
                component_diagnostics = _merge_leading_axes(
                    component_diagnostics, name, runs
                )
            datasets[name] = component.to_xarray(
                component_diagnostics,
                self.time_axis(first_step * runs, n_records, multiplicity=runs),
            )
        return datasets

    def __repr__(self) -> str:
        """Return a summary naming the components, exchangers, order and clock.

        The workflow is printed with its repeated blocks collapsed
        (``['exchange', 'atm', 'lnd'] * 24``), which is both shorter than the
        flat sequence and the way the workflow was written.
        """
        return (
            f"{type(self).__name__}("
            f"components={list(self.components)}, "
            f"exchangers={list(self.exchangers)}, "
            f"workflow=[{_compressed_workflow(self.workflow)}], "
            f"coupling_timestep={self._dt_seconds / _SECONDS_PER_DAY:g} days, "
            f"start_date={self._start_date.to_pydatetime().isoformat()}, "
            f"calendar={self._calendar!r})"
        )
