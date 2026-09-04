"""The :class:`Coupler`: the coupled-model definition, its clock and its step.

A ``Coupler`` is the whole definition of a coupled model:

- **what is coupled** -- the named components and the named exchangers that
  move information between them;
- **in what order** -- the ``workflow``, one ordered list over the single
  namespace shared by exchangers and components;
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

import dataclasses
import logging
from collections.abc import Callable, Sequence
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
    workflow : Sequence[str], optional
        The order in which exchangers and components run within one coupled
        step. Defaults to every exchanger (in insertion order) followed by
        every component (in insertion order), i.e. information is exchanged
        first and the components then all see the same exchanged state. Any
        name must be a registered exchanger or component and may appear only
        once.

    Notes
    -----
    Coupling is **lagged**: with the default workflow the exchanger at step
    *n* moves the fields each component produced during step *n-1*, so the
    first step of a run exchanges the values from ``initialize()``.

    """

    def __init__(
        self,
        components: dict[str, Component],
        exchangers: dict[str, Exchanger] | None = None,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str = "365_day",
        workflow: Sequence[str] | None = None,
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
        self._year_offset_seconds = seconds_since_new_year(start_date, calendar)
        self._days_per_year = float(jcm_days_per_year(calendar))

        for name, exchanger in (exchangers or {}).items():
            self.add_exchanger(name, exchanger)
        for name, component in components.items():
            self.add_component(name, component)

        # A workflow given explicitly is validated now, so a typo is a
        # construction error rather than a trace-time one; it is validated
        # again by `step_function`, because components may be added or
        # removed after construction.
        self._explicit_workflow: tuple[str, ...] | None = None
        if workflow is not None:
            self._explicit_workflow = self._validated_workflow(workflow)

        logger.debug("Coupled workflow: %s", ", ".join(self.workflow))

    # -- the coupled model definition --------------------------------------

    def add_component(self, name: str, component: Component) -> None:
        """Register ``component`` under ``name``.

        The object itself is stored -- there is no wrapper -- so
        ``coupler.components[name] is component``.

        Raises
        ------
        TypeError
            If ``component`` does not satisfy the :class:`Component`
            protocol; the message names the missing attributes.
        ValueError
            If ``name`` is already an exchanger.

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
        # Components with an internal timestep (JCM, Veros) need the coupling
        # timestep and must agree with the coupler about the start date and
        # calendar. Binding here rather than only in `__init__` means a
        # component added later is bound too; binding BEFORE registering
        # means a component that rejects the clock never enters the coupler,
        # so a caller that catches the ValueError keeps a usable coupler and
        # a previously valid component under the same name is not replaced.
        if isinstance(component, SupportsBind):
            component.bind(
                coupling_timestep=self._coupling_timestep,
                start_date=self._start_date,
                calendar=self._calendar,
            )
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
        """The order in which exchangers and components run within a coupled step."""
        if self._explicit_workflow is not None:
            return self._explicit_workflow
        return (*self.exchangers, *self.components)

    def _validated_workflow(self, workflow: Sequence[str]) -> tuple[str, ...]:
        """Return ``workflow`` as a tuple, checking every name is known and unique."""
        seen: set[str] = set()
        for name in workflow:
            if not isinstance(name, str):
                raise TypeError(
                    f"Workflow entries must be component or exchanger names (strings); "
                    f"got {name!r}."
                )
            if name not in self.components and name not in self.exchangers:
                raise ValueError(
                    f"Workflow entry {name!r} is neither a component "
                    f"({sorted(self.components)}) nor an exchanger "
                    f"({sorted(self.exchangers)})."
                )
            if name in seen:
                # A repeated component would have to write two diagnostics
                # under one key; a repeated exchanger is more likely a typo
                # than an intent, so both are rejected.
                raise ValueError(
                    f"Workflow entry {name!r} appears more than once; each component "
                    "and exchanger runs exactly once per coupled step."
                )
            seen.add(name)
        return tuple(workflow)

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

    def time_axis(self, first_step: int, n: int) -> TimeAxis:
        """Return the datetimes of ``n`` output records starting at ``first_step``."""
        return TimeAxis(
            start_date=self._start_date,
            steps=np.arange(first_step, first_step + n),
            dt=self._coupling_timestep,
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
        After each workflow element the pytree structure of the carries dict
        is compared with the structure it had on entry, and a change is a
        ``RuntimeError``. The comparison happens at trace time, so it costs
        nothing per step, and it turns what would otherwise be an opaque
        ``lax.scan`` structure error into one that names the element
        responsible.

        """
        workflow = self._validated_workflow(self.workflow)
        # Snapshot the coupled model: the returned function is a description
        # of the model as it stands now, so registering a component later
        # cannot silently change what an already-generated (and possibly
        # already-compiled) step does.
        components_by_name = dict(self.components)
        exchangers_by_name = dict(self.exchangers)

        def step(carry: CoupledCarry) -> tuple[CoupledCarry, dict[str, Diagnostics]]:
            time = self.coupling_time(carry.step)
            components: dict[str, Carry] = dict(carry.components)
            # Typed as Any because `jax.tree_util.tree_structure` returns an
            # opaque PyTreeDef that static analysis cannot compare.
            expected_structure: Any = jax.tree_util.tree_structure(components)
            diagnostics: dict[str, Diagnostics] = {}

            for name in workflow:
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
                    diagnostics[name] = component_diagnostics

                structure: Any = jax.tree_util.tree_structure(components)
                if structure != expected_structure:
                    raise RuntimeError(
                        f"Workflow element {name!r} changed the structure of the "
                        f"component carries, which `lax.scan` cannot carry.\n"
                        f"  before: {expected_structure}\n"
                        f"  after:  {structure}"
                    )

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
            stacks them). The clock is the carry's own ``step``, not the scan
            index, so calling the function twice continues the run rather
            than restarting it.

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
        dataset is labelled with the same :class:`TimeAxis`, so the results
        can be merged.

        Parameters
        ----------
        diagnostics : dict[str, Diagnostics]
            The diagnostics returned by a trajectory function.
        first_step : int
            The coupled step the first record covers; the step counter of the
            carry the trajectory started from.

        """
        datasets = {}
        for name, component in self.components.items():
            if name not in diagnostics:
                continue
            if not isinstance(component, SupportsXarray):
                logger.debug("Component %r has no to_xarray; skipping its output.", name)
                continue
            component_diagnostics = diagnostics[name]
            n_records = _leading_axis_length(component_diagnostics, name)
            datasets[name] = component.to_xarray(
                component_diagnostics, self.time_axis(first_step, n_records)
            )
        return datasets

    def __repr__(self) -> str:
        """Return a summary naming the components, exchangers, order and clock."""
        return (
            f"{type(self).__name__}("
            f"components={list(self.components)}, "
            f"exchangers={list(self.exchangers)}, "
            f"workflow={list(self.workflow)}, "
            f"coupling_timestep={self._dt_seconds / _SECONDS_PER_DAY:g} days, "
            f"start_date={self._start_date.to_pydatetime().isoformat()}, "
            f"calendar={self._calendar!r})"
        )
