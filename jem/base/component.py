"""The component contract and the coupled-model state types.

This module is the whole of the interface between the :class:`Coupler` and
the things it couples. It is deliberately small:

- :class:`Component` is a :class:`typing.Protocol`: a component is any object
  with a ``name``, an ``initialize()`` and a ``step(carry, time)``. There is
  no base class to inherit from, so an external model (JCM, Veros) is adapted
  by a thin wrapper class rather than by monkey-patching methods onto it.
- :class:`SupportsXarray`, :class:`SupportsCheckpoint` and
  :class:`SupportsBind` are *optional* capabilities. The coupler tests for
  them with ``isinstance`` (the protocols are runtime-checkable, which for a
  Protocol means "has these attributes"), never with ``hasattr`` at random
  call sites.
- :class:`CoupledCarry` is the scanned state of the coupled model: one carry
  per component plus the authoritative step counter. The counter lives in the
  carry, not in the ``lax.scan`` index, so the clock survives chunked runs and
  checkpoint restarts (the scan index restarts at zero on every call; the
  carry does not).
- :class:`CouplingTime` is what every ``Component.step`` receives instead of
  a bare step index: the coupler's current ``jax_datetime.Datetime`` and the
  coupling ``dt``. Components therefore hold **no clock state of their own**;
  the coupler owns the one clock -- a carried, incrementally advanced
  ``Datetime``, not a step count multiplied out -- and two components can
  never disagree about the date.
- :data:`Exchanger` is the type of the functions that move information
  between components. They were called "mappers" before v1.0; the name was
  changed because "mapper" reads as a regridding operation, whereas an
  exchanger may regrid, compute fluxes, convert units or simply copy a field.
  It is the *only* place where one component's carry is read by another.

The design is recorded in ``docs/source/design/carry_and_clock.md`` and, for
the task numbering (T1.1, T1.3), in the API hardening plan, which lives on
the review branch rather than in this repository:
https://github.com/climate-analytics-lab/jax-esm/blob/claude/jax-esm-api-review-jv7j7u/docs/source/design/api_hardening_plan.md
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast, get_args, runtime_checkable

import jax
import jax_datetime as jdt
import numpy as np
import xarray as xr
from flax import struct
from jcm.date import fraction_of_year_elapsed
from jcm.predictions import output_time_labels

# A component's carry is an arbitrary pytree; by convention the slab models
# and the JCM wrapper use a dict with "state", "forcing" and "derived" keys
# (see carry_and_clock.md), but the coupler never looks inside it.
Carry = Any
# What a component returns per step for output; also an arbitrary pytree.
# The coupler stacks it over the scanned steps, so every leaf gains a leading
# time axis of length ``iterations``.
Diagnostics = Any

SECONDS_PER_DAY = 86400.0

#: Prefix on the output name of a field a component was *given*, as opposed to
#: one it computed. See :func:`forcing_variable`.
FORCING_VARIABLE_PREFIX = "forcing_"


def forcing_variable(name: str) -> str:
    """Return the output-variable name for a field the component was forced with.

    A coupled run writes one dataset per component and they are meant to be
    read -- and merged -- together. A field a component *received* through the
    coupler is already written, unprefixed, by the component that produced it,
    so writing the received copy under the same name puts two different
    variables with one name into the merged dataset: different because coupling
    is lagged, so the copy is one step behind the original. ``xr.merge`` then
    refuses the two datasets outright.

    Prefixing the received copy with ``forcing_`` fixes that and says the more
    accurate thing anyway: ``forcing_total_heat_flux`` in the land model's
    output is the flux the land was driven with, not a flux the land computed.
    A component's own state and its derived diagnostics keep their plain names
    -- they are that component's output, and match what JCM calls them.

    This is a convention of the *packaged* output, not part of the component
    protocol: the coupler never inspects a dataset, and a wrapper around an
    external model may name its variables however that model does. The slab
    models and the Veros adapter follow it so that their datasets merge with
    each other's and with JCM's; a component added later only needs to follow
    it if its output is meant to merge the same way. It lives here, next to
    the rest of the contract, so that there is one definition of the prefix.

    A name that already carries the prefix is returned unchanged, so a model
    whose own field is called ``forcing_shortwave_flux`` does not come out as
    ``forcing_forcing_shortwave_flux``.

    Parameters
    ----------
    name : str
        The physical field's name, as the component that produces it writes it.

    Returns
    -------
    str

    """
    if name.startswith(FORCING_VARIABLE_PREFIX):
        return name
    return f"{FORCING_VARIABLE_PREFIX}{name}"


#: Name of the variable attribute that records which part of a component's
#: carry an output variable came from. See :func:`role_attrs`.
ROLE_ATTRIBUTE = "jem_role"

#: What a variable's role may be: the three sections of the carry layout the
#: packaged components share (``jem.exchangers`` addresses fields by them).
Role = Literal["state", "derived", "forcing"]

#: The roles, as a tuple, for validation and for iterating in a test.
ROLES: tuple[str, ...] = get_args(Role)


def role_attrs(role: Role) -> dict[str, str]:
    """Return the variable attributes marking an output variable's role.

    A packaged component's output says which part of its carry a variable
    came from in **two** ways, and they are not redundant:

    - The ``forcing_`` name prefix (:func:`forcing_variable`) exists to stop
      an ``xr.merge`` collision. A field one component computed and the copy
      another was given through the coupler are *different* variables --
      coupling is lagged, so the copy is a step behind -- and under one name
      ``xr.merge`` refuses the two datasets outright. Renaming is the only
      thing that fixes that, so the prefix stays.
    - This attribute exists so that nothing has to *parse* names to find out
      what a variable is. ``ds.filter_by_attrs(jem_role="forcing")`` is the
      whole query; the alternative, matching a prefix, cannot tell a received
      ``forcing_q_flux`` from a model whose own field happens to be called
      ``forcing_shortwave_flux``, and says nothing at all about the rest --
      whether ``total_heat_flux`` in the ocean's output is state the ocean
      integrated or a diagnostic it computed.

    So the prefix is a naming rule and this is metadata; every packaged
    component sets both. The roles are the sections of the carry layout the
    packaged components share and that :mod:`jem.exchangers` addresses:
    ``state`` is what the component integrates, ``derived`` what it diagnosed
    for others to read, ``forcing`` what it was given. A variable that is
    none of those -- a grid mask, a layer thickness, anything time-invariant
    that came from the component's configuration rather than its carry -- is
    left untagged, which is a meaningful answer and not an omission.

    A fresh dict is returned on every call, because xarray keeps the dict it
    is handed: two variables sharing one attrs dict would share any later
    edit to it.

    Parameters
    ----------
    role : {"state", "derived", "forcing"}
        Which section of the carry the variable was read from.

    Returns
    -------
    dict[str, str]
        ``{"jem_role": role}``, ready to merge into a variable's attributes.

    Raises
    ------
    ValueError
        If ``role`` is not one of the three.

    """
    if role not in ROLES:
        raise ValueError(
            f"Unknown variable role {role!r}; it must be one of {list(ROLES)!r}."
        )
    return {ROLE_ATTRIBUTE: role}


def start_year_fraction(start_date: jdt.Datetime) -> float:
    """Return the position of ``start_date`` in the annual cycle, in ``[0, 1)``.

    The same quantity :attr:`CouplingTime.year_fraction` reports at step 0,
    computed by the same function (``jcm.date.fraction_of_year_elapsed``), so
    a component that samples a climatology in ``initialize()`` and one that
    samples it in ``step()`` cannot disagree about where the run starts.
    """
    return float(fraction_of_year_elapsed(start_date))


@struct.dataclass
class CouplingTime:
    """The coupler's clock as seen by one component step.

    There is one clock: a carried ``jax_datetime.Datetime`` that the coupler
    advances by ``dt`` every step (:meth:`end_of_step`), never a step count
    multiplied out. ``step`` and ``sim_time`` remain for the sub-step
    indexing a component with an internal timestep or a health-check log line
    needs; nothing derives ``time`` from either of them.

    Attributes
    ----------
    step : jax.Array
        int32 scalar; number of coupling steps completed before this one
        (0 on the first step). Copied from :attr:`CoupledCarry.step`.
    time : jax_datetime.Datetime
        The current model time -- the coupler's start date, advanced by
        ``dt`` once per step it has taken.
    sim_time : jax.Array
        Seconds since the run's start; equals ``step * dt``. Float64 when
        ``jax_enable_x64`` is on, float32 otherwise. A convenience for a
        clock-drift check against a wrapped model's own elapsed-seconds
        counter (:mod:`jem.components.clock`); ``time`` is authoritative.
    dt : float
        Coupling timestep in seconds. Static (not a pytree leaf).

    """

    step: jax.Array
    time: jdt.Datetime
    sim_time: jax.Array
    dt: float = struct.field(pytree_node=False)

    def end_of_step(self) -> "CouplingTime":
        """Return the clock as it reads at the end of this step (one step later).

        A component that needs a boundary condition at both ends of a step
        (the slab models measure an anomaly against the climatology at the
        start and add it back at the end) uses this rather than adding ``dt``
        to ``time`` by hand.
        """
        advanced: CouplingTime = self.replace(  # type: ignore[attr-defined]
            step=self.step + 1,
            time=self.time + jdt.to_timedelta(int(self.dt), "second"),
            sim_time=self.sim_time + self.dt,
        )
        return advanced

    @property
    def year_fraction(self) -> jax.Array:
        """Position in the annual cycle in ``[0, 1)`` at the *start* of this step.

        Zero is 00:00 on 1 January. ``jcm.date.fraction_of_year_elapsed`` --
        the same function the atmosphere's own seasonal physics uses -- so a
        component's seasonal cycle and JCM's agree by construction. This is
        what a monthly climatology is interpolated with
        (``jem.utils.cycles.evaluate_cyclic_linear``).
        """
        return cast(jax.Array, fraction_of_year_elapsed(self.time))


@struct.dataclass
class CoupledCarry:
    """The scanned state of the whole coupled model.

    Attributes
    ----------
    components : dict[str, Carry]
        One carry per component, keyed by component name.
    time : jax_datetime.Datetime
        The model's current time -- the coupler's one clock. Set to the
        coupler's start date at :meth:`~jem.base.coupler.Coupler.initialize`
        and advanced by the coupling ``dt`` every step; every component's
        :class:`CouplingTime` is built from it.
    step : jax.Array
        int32 scalar; number of coupling steps completed. Kept for the
        sub-step indexing workflow multiplicity needs and for a checkpoint's
        step count; ``time`` is what the clock is.

    """

    components: dict[str, Carry]
    time: jdt.Datetime
    step: jax.Array


@dataclasses.dataclass(frozen=True)
class TimeAxis:
    """The output records of a run and how they are labelled in time.

    Built by ``Coupler.time_axis(first_step, n)``; handed to
    :meth:`SupportsXarray.to_xarray` so every component labels its output
    with the same ``time`` coordinate and ``xr.merge`` of two components'
    datasets is an N-long join rather than a 2N-long union.

    The labelling convention is JCM's: record ``k`` covers the interval
    ``[start_date + k dt, start_date + (k+1) dt)`` and is labelled with its
    **midpoint**, as ``datetime64[ms]`` -- the same convention and the same
    arithmetic as ``jcm.predictions.output_time_labels`` and
    ``ModelPredictions.to_xarray``, computed here in whole milliseconds on the
    host so the two are identical bit for bit and ``xr.merge`` joins them on
    one time axis rather than unioning two.

    Attributes
    ----------
    start_date : jdt.Datetime
        The run's start date.
    steps : numpy.ndarray
        int array of coupled-step indices, one per record.
    dt : jdt.Timedelta
        Coupling timestep.

    """

    start_date: jdt.Datetime
    steps: Any
    dt: jdt.Timedelta

    def __len__(self) -> int:
        """Return the number of output records."""
        return len(self.steps)

    def _bounds_ms(self) -> np.ndarray:
        """Return each record's ``[start, end)`` bounds, as int64 milliseconds."""
        start_ms = int(self.start_date.to_datetime64().astype("datetime64[ms]").astype(np.int64))
        dt_seconds = int(np.asarray(self.dt.days)) * int(SECONDS_PER_DAY) + int(
            np.asarray(self.dt.seconds)
        )
        dt_ms = dt_seconds * 1000
        steps = np.asarray(self.steps, dtype=np.int64)
        lower = start_ms + steps * dt_ms
        return np.stack([lower, lower + dt_ms], axis=-1)

    def bounds(self) -> np.ndarray:
        """Return each record's interval bounds, as ``datetime64[ms]`` ``(n, 2)``."""
        return self._bounds_ms().astype("datetime64[ms]")

    def datetimes(self) -> np.ndarray:
        """Return the record labels as ``datetime64[ms]`` (each interval's midpoint).

        ``lower + (upper - lower) // 2``, exactly
        ``jcm.predictions.ModelPredictions.time_labels`` -- floor division in
        integer milliseconds, so an odd-length interval's half-millisecond
        midpoint matches JCM's rather than a naive float mean rounding it
        differently.
        """
        bounds = self._bounds_ms()
        midpoints = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) // 2
        return cast(np.ndarray, output_time_labels(midpoints.astype("datetime64[ms]")))

    @property
    def attrs(self) -> dict[str, str]:
        """CF attributes JCM writes on its ``time`` coordinate.

        ``units`` is deliberately absent: xarray owns it through the datetime
        encoding it chooses on write, and setting it here collides with that.
        """
        return {"standard_name": "time", "axis": "T", "long_name": "time"}


@runtime_checkable
class Component(Protocol):
    """What the coupler requires of anything it steps.

    A component holds its *configuration* (grid, parameters, boundary data)
    on ``self`` and its *evolving state* in the carry it returns from
    ``initialize`` and threads through ``step``. ``step`` must be a pure
    function of ``(carry, time)`` and must return a carry with exactly the
    pytree structure, shapes and dtypes it received, or ``lax.scan`` rejects
    it.
    """

    name: str

    def initialize(self) -> Carry:
        """Build the initial carry. Must not integrate the model."""
        ...

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance one coupling timestep; return the new carry and the step's output."""
        ...


@runtime_checkable
class SupportsXarray(Protocol):
    """Optional: convert stacked diagnostics to ``xarray``.

    The return value is normally one :class:`xarray.Dataset`, keyed in the
    coupler's output under the component's registered name. A component that
    is itself a coupled model -- a :class:`~jem.base.coupler.Coupler` nested
    inside a slower one -- has no single dataset to return: it holds several
    components of its own, each with its own variables and its own sampling
    rate. It may therefore return a **mapping** of name to dataset instead,
    which the outer coupler flattens into its own result under those names
    (and refuses if one of them collides with a name already there). A
    wrapper around any other multi-model system can do the same.
    """

    def to_xarray(
        self, diagnostics: Diagnostics, time: TimeAxis
    ) -> xr.Dataset | Mapping[str, xr.Dataset]: ...


@runtime_checkable
class SupportsCheckpoint(Protocol):
    """Optional: components whose carry is not a plain pytree of arrays (Veros)."""

    def save_carry(self, carry: Carry, directory: Path) -> None: ...

    def load_carry(self, directory: Path) -> Carry: ...


@runtime_checkable
class SupportsBind(Protocol):
    """Optional: receive the coupler's clock definition at registration.

    A component that has its own internal timestep (JCM, Veros) needs to know
    the coupling timestep to decide how many internal steps make one coupled
    step, and needs to agree with the coupler about the start date. The
    coupler calls ``bind`` once, from its constructor, for every component
    that provides it. Raise ``ValueError`` on a mismatch.
    """

    def bind(
        self,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
    ) -> None: ...


# An exchanger moves information between components. It receives the mapping
# of component carries (a fresh dict, so adding or replacing entries never
# mutates the coupler's input) and the current clock, and returns the mapping
# to continue with. It must not mutate the carries it receives in place;
# build new ones with ``dataclasses.replace`` / ``.replace`` and return them.
# The clock is passed so an exchanger can implement time-dependent coupling
# (lagged exchange, ramped forcing) without keeping state of its own.
Exchanger = Callable[[dict[str, Carry], CouplingTime], dict[str, Carry]]
