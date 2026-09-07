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

A ``Coupler`` is itself a :class:`~jem.base.component.Component`: it has a
``name``, an ``initialize()`` and a ``step(carry, time)``, and it binds to a
slower clock, writes its output and checkpoints itself like any other
component that provides those capabilities. A coupled model can therefore be
a component of a slower coupled model -- the GFDL pattern of a fast
atmosphere/land loop inside a daily ocean coupling -- with no wrapper class.
The alternative for the same model is one coupler with a repeated workflow
(see :class:`Coupler`); the two are equivalent, and which reads better
depends on whether the fast loop is a thing in its own right.

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
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

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
    SupportsCheckpoint,
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
                f"{name!r} runs {expected} times per step, so its stacked "
                f"diagnostics must have shape (steps, {expected}, ...); got "
                f"{array.shape}. (These diagnostics did not come from a trajectory "
                "of this coupler.)"
            )
        return array.reshape((array.shape[0] * expected, *array.shape[2:]))

    return jax.tree_util.tree_map(merge, diagnostics)


def nested_carry(
    carries: dict[str, Carry], outer_name: str, inner_name: str
) -> Carry:
    """Return the carry of ``inner_name`` inside the nested coupler ``outer_name``.

    An exchanger in an outer coupler sees a nested :class:`Coupler` as one
    entry in the carries mapping, holding a whole :class:`CoupledCarry` rather
    than a component carry. Reaching an inner component means going through
    ``.components``, and this is that step written once, with an error that
    says what was actually there when it is not a nested coupler.

    Parameters
    ----------
    carries : dict[str, Carry]
        The mapping an exchanger was handed.
    outer_name : str
        Name the nested coupler is registered under in the outer coupler.
    inner_name : str
        Name of the component inside it.

    Returns
    -------
    Carry

    """
    return _inner_carries(carries, outer_name)[inner_name]


def with_nested_carry(
    carries: dict[str, Carry],
    outer_name: str,
    inner_name: str,
    new_inner_carry: Carry,
) -> dict[str, Carry]:
    """Return ``carries`` with one component of a nested coupler replaced.

    The counterpart of :func:`nested_carry`, and the only safe way for an
    exchanger to write into a nested model: nothing is mutated in place -- a
    new inner carries dict, a new :class:`CoupledCarry` (via
    ``dataclasses.replace``, so a field added to it later is carried through)
    and a new outer mapping -- because the carries an exchanger is handed are
    the ones a ``lax.scan`` is carrying.

    Parameters
    ----------
    carries : dict[str, Carry]
        The mapping an exchanger was handed.
    outer_name : str
        Name the nested coupler is registered under in the outer coupler.
    inner_name : str
        Name of the component inside it to replace.
    new_inner_carry : Carry
        The carry to put there. It must have the pytree structure of the one
        it replaces, like any other carry an exchanger returns.

    Returns
    -------
    dict[str, Carry]

    """
    coupled = carries[outer_name]
    inner = dict(_inner_carries(carries, outer_name), **{inner_name: new_inner_carry})
    return dict(carries, **{outer_name: dataclasses.replace(coupled, components=inner)})


def _inner_carries(carries: dict[str, Carry], outer_name: str) -> dict[str, Carry]:
    """Return the component carries of the nested coupler registered as ``outer_name``."""
    coupled = carries[outer_name]
    if not isinstance(coupled, CoupledCarry):
        raise TypeError(
            f"{outer_name!r} does not hold a nested coupled model: its carry is a "
            f"{type(coupled).__name__}, not a CoupledCarry. Only a component that "
            "is itself a Coupler has components inside it."
        )
    return coupled.components


def _named_datasets(
    name: str, written: xr.Dataset | Mapping[str, xr.Dataset]
) -> dict[str, xr.Dataset]:
    """Return what a component's ``to_xarray`` produced, keyed by output name.

    A component normally returns one dataset, which is keyed by the name it
    is registered under. A component that is itself a coupled model returns a
    mapping of its own components' names to their datasets -- they have
    different variables and, if the inner workflow repeats one of them,
    different sampling rates, so there is nothing to concatenate them into --
    and those names are used as they are.
    """
    if isinstance(written, xr.Dataset):
        return {name: written}
    if isinstance(written, Mapping):
        # An `xarray.Dataset` is itself a Mapping (of variable name to
        # DataArray), so the isinstance above cannot narrow the union for
        # static analysis; the Dataset case has already returned.
        return dict(cast(Mapping[str, xr.Dataset], written))
    raise TypeError(
        f"Component {name!r} returned {type(written).__name__} from to_xarray; "
        "it must return an xarray.Dataset, or a mapping of name to Dataset."
    )


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
    name : str
        The coupler's own name, as the :class:`Component` protocol requires
        it of anything a coupler steps -- a ``Coupler`` is a component (see
        the Notes), so it has one. It is *not* what an outer coupler keys it
        by: that is the dict key it is registered under, as for every other
        component.
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

    **A coupler is a component.** It has a ``name``, an ``initialize()``
    returning its :class:`CoupledCarry` and a ``step(carry, time)``, and it
    implements :class:`~jem.base.component.SupportsBind`,
    :class:`~jem.base.component.SupportsXarray` and
    :class:`~jem.base.component.SupportsCheckpoint`, so it can be registered
    in a slower coupler with no wrapper class::

        fast = Coupler({"atm": atm, "lnd": lnd}, {"atm_lnd_exchange": ...},
                       coupling_timestep=jdt.to_timedelta(1, "hour"),
                       start_date=start_date)
        model = Coupler({"atm_lnd": fast, "ocn": ocn}, {"srf_ocn_exchange": ...},
                        coupling_timestep=jdt.to_timedelta(1, "day"),
                        start_date=start_date,
                        workflow=["srf_ocn_exchange", "atm_lnd", "ocn"])

    The outer timestep must be a whole multiple of the inner one, and the two
    must share a start date and calendar; :meth:`bind` refuses anything else.
    One outer step runs ``r = outer / inner`` inner coupled steps, driven by
    the inner carry's own step counter, so the inner clock is continuous
    across outer steps. Its diagnostics come back stacked on a leading axis
    of length ``r`` (none for ``r == 1``), and its :meth:`to_xarray` returns
    one dataset **per inner component**, which the outer coupler flattens
    into its own result under those names -- the nested coupler's own
    registered name does not appear in the output.

    An exchanger in the outer coupler sees the inner coupled carry under that
    registered name and reaches inside it with :func:`nested_carry` and
    :func:`with_nested_carry`.

    :meth:`save_state` / :meth:`load_state` checkpoint the whole coupled
    model in one call, deriving each component's writer from the component
    itself; because a coupler implements the capability too, a nested model
    checkpoints by recursion into a subdirectory of its own.

    The same model can be written either as a nested pair of couplers or as
    one coupler with a repeated workflow, and the two produce identical
    numbers. Prefer the nested form when the fast loop is a thing in its own
    right -- built, tested, checkpointed or run on its own -- and the flat
    form when it is only a rate: one coupler, one carry and one workflow to
    read.

    """

    def __init__(
        self,
        components: dict[str, Component],
        exchangers: dict[str, Exchanger] | None = None,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str = "365_day",
        name: str = "coupled",
        workflow: Sequence[Any] | None = None,
    ):
        """Build a coupled model; see the class docstring for the parameters."""
        self.name = name
        # Set only by `bind`, when this coupler is registered as a component
        # of a slower one: the number of coupled steps of this model that
        # make one step of that one. None means "not nested".
        self._outer_ratio: int | None = None
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

    # -- the coupled model as a component of a slower one -------------------

    @property
    def outer_ratio(self) -> int | None:
        """Coupled steps of this model per step of the coupler it is nested in.

        ``None`` until :meth:`bind` is called, i.e. for a coupler that is not
        a component of another one.
        """
        return self._outer_ratio

    def bind(
        self,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str,
    ) -> None:
        """Adopt an outer coupler's clock (:class:`~jem.base.component.SupportsBind`).

        A nested coupled model keeps its own timestep and runs several of its
        own coupled steps per outer step, so what it needs from the outer
        coupler is the *ratio*. The two clocks must agree exactly otherwise:
        a nested model that started on a different date, or counted a
        different year, would date its own forcing and output differently
        from every other component of the outer model, which is the one thing
        the coupler's single clock exists to prevent.

        Parameters
        ----------
        coupling_timestep : jax_datetime.Timedelta
            The outer coupled timestep. It must be a whole multiple of this
            coupler's own, because this coupler advances by whole steps of
            its own and a fractional outer step could only be rounded.
        start_date : jax_datetime.Datetime
            The outer run's start date; must equal :attr:`start_date`.
        calendar : str
            The outer run's calendar; must equal :attr:`calendar`.

        Raises
        ------
        ValueError
            If the timestep does not divide, if either clock setting differs,
            or if this coupler is already bound to a different outer
            timestep. Binding it again to the same clock is a no-op.

        """
        if str(calendar) != str(self._calendar):
            raise ValueError(
                f"Calendar mismatch: the outer coupler runs {calendar!r} but the "
                f"nested coupler {self.name!r} runs {self._calendar!r}."
            )
        if start_date != self._start_date:
            raise ValueError(
                f"Start-date mismatch: the outer coupler starts at {start_date!r} "
                f"but the nested coupler {self.name!r} starts at "
                f"{self._start_date!r}."
            )
        outer_seconds = _timedelta_seconds(coupling_timestep)
        ratio, remainder = divmod(outer_seconds, self._dt_total_seconds)
        if remainder or ratio < 1:
            raise ValueError(
                f"The outer coupling timestep ({outer_seconds} s) is not a whole "
                f"multiple of the nested coupler {self.name!r}'s own "
                f"({self._dt_total_seconds} s), so one outer step is not a whole "
                "number of its coupled steps."
            )
        if self._outer_ratio is not None and ratio != self._outer_ratio:
            # One instance belongs to one coupled model: `step` runs
            # `_outer_ratio` inner steps, so a second outer coupler with
            # another timestep would silently run the wrong number for the
            # first one.
            raise ValueError(
                f"The nested coupler {self.name!r} is already bound to an outer "
                f"step of {self._outer_ratio} of its own steps and cannot also be "
                f"bound to one of {ratio}. Build a separate instance per coupled "
                "model."
            )
        self._outer_ratio = ratio

    def _require_outer_ratio(self, what: str) -> int:
        """Return the bound outer ratio, or explain that there is not one."""
        if self._outer_ratio is None:
            raise RuntimeError(
                f"{type(self).__name__} {self.name!r} cannot be {what} as a "
                "component because it has not been bound to an outer coupler. "
                "Register it in one (which binds it), or drive it directly with "
                "`step_function()` / `generate_trajectory_function()`."
            )
        return self._outer_ratio

    def _check_outer_clock(self, time: CouplingTime, ratio: int) -> None:
        """Check the outer clock is the one this coupler was bound to.

        Only the static fields of a :class:`CouplingTime` can be checked: the
        step counter is a traced array, and the inner steps are driven by the
        inner carry's own counter anyway. The static fields are enough to
        catch the mistake that matters -- a clock from a different coupler,
        or a hand-built one at the wrong rate -- at trace time.
        """
        expected_dt = self._dt_seconds * ratio
        if not math.isclose(time.dt, expected_dt, rel_tol=1e-12):
            raise ValueError(
                f"The nested coupler {self.name!r} was bound to an outer step of "
                f"{expected_dt:g} s but was handed a clock with dt={time.dt:g} s."
            )
        if time.days_per_year != self._days_per_year or (
            time.year_offset_seconds != self._year_offset_seconds
        ):
            raise ValueError(
                f"The nested coupler {self.name!r} was handed a clock from a "
                "different calendar or start date than the one it was bound to."
            )

    def step(
        self, carry: CoupledCarry, time: CouplingTime
    ) -> tuple[CoupledCarry, dict[str, Diagnostics]]:
        """Advance this coupled model by one step of the coupler it is nested in.

        Runs ``r`` of this coupler's own coupled steps, where ``r`` is the
        ratio :meth:`bind` recorded. The inner clock comes from the inner
        carry's own step counter, exactly as it does in a standalone run, so
        it is continuous across outer steps and survives a checkpoint; the
        outer ``time`` is only checked against it (see
        :meth:`_check_outer_clock`).

        Parameters
        ----------
        carry : CoupledCarry
            This coupler's own carry, which is what :meth:`initialize`
            returns and what the outer coupler stores under this coupler's
            registered name.
        time : CouplingTime
            The outer coupler's clock for this outer step.

        Returns
        -------
        carry : CoupledCarry
            The inner carry after ``r`` inner steps.
        diagnostics : dict[str, Diagnostics]
            One entry per inner component that ran, stacked on a leading axis
            of length ``r``. For ``r == 1`` there is no extra axis and the
            inner step is run directly rather than through a length-1 scan,
            mirroring what multiplicity does for a component that runs once.

        """
        ratio = self._require_outer_ratio("stepped")
        self._check_outer_clock(time, ratio)
        if ratio == 1:
            return self.step_function()(carry)
        # An unjitted trajectory: `lax.scan` keeps one copy of the inner step
        # in the outer jaxpr instead of `r` unrolled ones, and stacks the
        # per-step diagnostics on the leading axis this method promises. It
        # is left unjitted so that it composes into whatever the outer
        # coupler's trajectory is wrapped in.
        trajectory = self.generate_trajectory_function(ratio, jit=False)
        return trajectory(carry)

    # -- output ------------------------------------------------------------

    def to_xarray(
        self,
        diagnostics: dict[str, Diagnostics],
        time: TimeAxis | None = None,
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

        A component that is itself a :class:`Coupler` returns one dataset per
        *its* components, and they are flattened into this coupler's result
        under those names; a name that collides with one already there is a
        ``ValueError``, because silently overwriting one component's output
        with another's is not a merge.

        There are two ways to call this, one method so that the labelling
        rules cannot drift apart:

        - ``to_xarray(diagnostics, first_step=0)`` -- how a *run* is written
          out. ``first_step`` is the coupled step the first record covers:
          the ``step`` of the carry the trajectory started from. **Pass it
          for a chunked run**, or every chunk is labelled with the first
          chunk's dates.
        - ``to_xarray(diagnostics, time)`` -- the
          :class:`~jem.base.component.SupportsXarray` signature, used when
          this coupler is nested in a slower one. ``time`` is the outer
          coupler's axis: its first step, times the ratio :meth:`bind`
          recorded, is where this model's own records start, and the outer
          coupler's extra leading axis of length ``r`` (one entry per inner
          coupled step) is folded into the records first. The datasets come
          back on the inner, faster axis.

        Parameters
        ----------
        diagnostics : dict[str, Diagnostics]
            The diagnostics returned by a trajectory function, or -- in the
            nested form -- the ones an outer coupler stacked for this one.
        time : TimeAxis, optional
            The outer coupler's time axis; only for a nested coupler, and
            mutually exclusive with ``first_step``.
        first_step : int
            The coupled step the first record covers.

        """
        if time is not None:
            if first_step:
                raise ValueError(
                    "Pass either `time` (the outer coupler's axis, when this "
                    "coupler is nested) or `first_step` (the coupled step a run "
                    "starts at), not both."
                )
            ratio = self._require_outer_ratio("written out")
            # The outer axis counts ITS steps; this model's records start at
            # the inner step that outer step corresponds to.
            first_step = int(np.asarray(time.steps)[0]) * ratio
            if ratio > 1:
                diagnostics = {
                    name: _merge_leading_axes(component_diagnostics, name, ratio)
                    for name, component_diagnostics in diagnostics.items()
                }

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
            written = component.to_xarray(
                component_diagnostics,
                self.time_axis(first_step * runs, n_records, multiplicity=runs),
            )
            for dataset_name, dataset in _named_datasets(name, written).items():
                if dataset_name in datasets:
                    raise ValueError(
                        f"Component {name!r} wrote a dataset named "
                        f"{dataset_name!r}, which another component of this "
                        "coupler has already written. Rename the component "
                        "inside it, or register it under another name."
                    )
                datasets[dataset_name] = dataset
        return datasets

    # -- checkpointing -----------------------------------------------------

    def _component_savers(self) -> dict[str, Callable[[Carry, Path], None]]:
        """Return the ``save_state`` of every component that has one.

        Which components need writing by hand rather than pickling is a
        property of the components, and the coupler is the one object that
        knows them all -- so it is the one object that can assemble this
        mapping. A caller enumerating it instead would have to know that the
        ocean is Veros and that Veros needs its HDF5 restart path, and would
        have to know it again for every model it builds.
        """
        return {
            name: component.save_state
            for name, component in self.components.items()
            if isinstance(component, SupportsCheckpoint)
        }

    def _component_loaders(self) -> dict[str, Callable[[Path], Carry]]:
        """Return the ``load_state`` of every component that has one.

        The inverse of :meth:`_component_savers`, derived from the same
        capability, so a carry written by a component's ``save_state`` is
        always read back by that component's ``load_state``.
        """
        return {
            name: component.load_state
            for name, component in self.components.items()
            if isinstance(component, SupportsCheckpoint)
        }

    def save_state(self, carry: CoupledCarry, directory: Path) -> None:
        """Write the coupled carry to ``directory`` (:class:`~jem.base.component.SupportsCheckpoint`).

        This is what a driver calls: one line that checkpoints the whole
        coupled model, however it is put together. Every component that
        implements :class:`~jem.base.component.SupportsCheckpoint` writes its
        own carry into ``directory / <its registered name>``; every other
        component's carry is pickled as ``<name>_carry.pkl``; and the coupled
        step counter is written last, as the completion marker (see
        :func:`jem.utils.checkpoints.save_coupled_carry`).

        A :class:`Coupler` implements the capability itself, so a **nested**
        coupled model is checkpointed by recursion: the outer coupler hands
        the inner one the subdirectory named after it, and the inner one
        writes its own components and its own marker there. Without that,
        the outer save would treat the inner :class:`CoupledCarry` as a plain
        pytree and pickle it -- which silently bypasses the HDF5 restart path
        a component like Veros requires.

        :func:`jem.utils.checkpoints.save_coupled_carry` still takes an
        explicit ``component_savers`` mapping, for a caller that wants to
        override or supply a saver for something that is not a component
        capability. This method is the answer for the ordinary case.

        Parameters
        ----------
        carry : CoupledCarry
            The carry a trajectory function (or a nested step) returned.
        directory : pathlib.Path
            Directory to write into; created if absent.

        """
        # Imported here rather than at module scope: `jem.utils.checkpoints`
        # imports the component contract from `jem.base`, so the dependency
        # runs the other way round and a module-level import would make the
        # two modules' import order load-bearing.
        from jem.utils.checkpoints import save_coupled_carry

        save_coupled_carry(carry, directory, component_savers=self._component_savers())

    def load_state(self, directory: Path) -> CoupledCarry:
        """Read back a coupled carry written by :meth:`save_state`.

        The exact inverse: the loaders are derived from the same components,
        so a nested coupler reads its own subdirectory back and a component
        with a custom format is read by the code that wrote it. The result is
        a :class:`CoupledCarry` that can be handed straight to a trajectory
        function, which continues the run from the step the checkpoint holds.

        The components read are the ones registered *now*: a checkpoint is
        loaded into the model that is meant to continue it, and a component
        added or removed since it was written is a mismatch the load reports
        (a missing file) rather than papering over.

        Parameters
        ----------
        directory : pathlib.Path
            A directory written by :meth:`save_state`.

        Returns
        -------
        CoupledCarry

        Raises
        ------
        ValueError
            If ``directory`` holds no completion marker, i.e. it is not a
            complete checkpoint.

        """
        # See `save_state` for why this import is not at module scope.
        from jem.utils.checkpoints import load_coupled_carry

        return load_coupled_carry(
            directory, self.components, component_loaders=self._component_loaders()
        )

    def __repr__(self) -> str:
        """Return a summary naming the components, exchangers, order and clock.

        The workflow is printed with its repeated blocks collapsed
        (``['exchange', 'atm', 'lnd'] * 24``), which is both shorter than the
        flat sequence and the way the workflow was written.
        """
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"components={list(self.components)}, "
            f"exchangers={list(self.exchangers)}, "
            f"workflow=[{_compressed_workflow(self.workflow)}], "
            f"coupling_timestep={self._dt_seconds / _SECONDS_PER_DAY:g} days, "
            f"start_date={self._start_date.to_pydatetime().isoformat()}, "
            f"calendar={self._calendar!r})"
        )
