"""The JCM atmosphere as a JEM :class:`~jem.base.component.Component`.

:class:`JCMComponent` wraps a :class:`jcm.model.Model` rather than
monkey-patching methods onto it, so the atmosphere JEM drives is the same
object the user configured and nothing in JCM has to know JEM exists.

Four things this wrapper exists to get right:

**The exact clock is threaded.** jax-gcm v3 (PR 878) replaced the model's
float ``sim_time`` -- exact only to float32 rounding, which stops being
enough after a few decades of simulated time -- with an exact ``RunState``
clock (``time``, an absolute :class:`jax_datetime.Datetime`, and ``step``, an
integer JCM-timestep count), and ``Model.run_from_state_with_carry`` now
requires both explicitly rather than inferring them from the incoming dycore
state. So they live in the component carry under ``"time"``/``"step"``,
exactly like ``"physics"`` below, and are passed straight back in as
``initial_time``/``initial_step``. They must be threaded rather than
recomputed from the coupler's own step counter each call: JCM's own
``RunState`` is now the **authoritative** clock (jax-gcm v3, PR 878), and
jax-gcm's own migration guide (``docs/source/v2_to_v3.rst``, "One real
datetime clock") is explicit that a caller should continue threading all of
``RunState`` rather than deriving ``time``/``step`` from a step counter kept
elsewhere -- not because recomputing it would overflow (an int32-safe
reduce-before-multiply decomposition, :func:`jem.base.calendar
.gregorian_instant`, computes exactly this instant from the coupler's own
step counter for :meth:`_report_authoritative_clock_drift`'s own drift check
below, so recomputing it is not the problem). Threading it is simply what
keeps two clocks -- JCM's and the coupler's -- from ever being able to
disagree about what instant a step is at, which recomputing one from scratch
every call cannot guarantee once a checkpoint or a differently-configured
coupler enters the picture (see :meth:`JCMComponent.initialize`).

**The physics carry is threaded.** JCM's operator-split integration keeps
cross-step physics state — sub-cycled radiation, prior-step TKE, the
tendencies a term hands to the next step — in a carry that
:meth:`jcm.model.Model.run_from_state_with_carry` takes in and hands back.
Dropping it between coupling steps resets that memory once per coupling
interval, which is a silent, systematic error in the coupled run. So it
lives in the component carry under ``"physics"`` and is passed straight
back in. Its pytree structure is identical before and after a step, which
is what lets the whole coupled step scan; it contains integer and boolean
leaves, so it must never be cast wholesale to a float dtype.

**The forcing an exchanger writes is a plain array.** With
``forcing=from_file`` jax-gcm builds the surface boundary conditions as
:class:`jcm.forcing.TimeSeries` leaves — three pytree leaves each — which
the model slices by date on every internal timestep. An exchanger writes a
single ``(ix, il)`` array into those same fields, so a carry that held a
time series on the way in holds a bare array on the way out: a change of
pytree structure that ``lax.scan`` cannot carry, and that the coupler
refuses. :meth:`set_exchanged_forcing` names the fields the coupled model
supplies, and :meth:`initialize` collapses exactly those to the
climatology at the run's start date. The carry's ``"forcing"`` section
therefore has, from ``initialize()`` onward, the structure an exchange
preserves; every field no component supplies keeps its time series and
goes on being sliced by jax-gcm, one step at a time.

**Nothing integrates at initialization.** :meth:`initialize` builds the
initial pytrees with :meth:`jcm.model.Model.bootstrap_state` and a
structural template of the diagnostics dict. The previous adapter ran a
whole coupling interval just to discover the shape of the diagnostics it
would later store, which both cost a full model step per run and started
the atmosphere one interval ahead of the coupler's clock.

Every JCM *attribute* this wrapper touches is public at the pinned revision
(``jem.components.jcm.contract``), apart from one underscore-prefixed
diagnostics key: SPEEDY's private ``_surface_flux.u0``/``.v0``, which
``jem.components.jcm.exchange_fields`` still reads directly for the
near-surface wind *vector* jax-gcm#754's package-independent contract does
not publish (see that module's docstring). Every other field of the surface
exchange comes from the published ``diagnostics["surface_exchange"]``
struct. That is why the initial state comes from ``bootstrap_state``'s
return value and a stacked prediction is repaired with
``ModelPredictions.with_context``: an adapter that reached into JCM's
internals would break on a JCM refactor that broke nothing else.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import tree_math
import xarray as xr
from jcm.date import DateData
from jcm.forcing import ForcingData, TimeSeries, default_forcing
from jcm.model import Model
from jcm.predictions import ModelPredictions

from jem.base.calendar import gregorian_instant
from jem.base.component import (
    Carry,
    CouplingTime,
    Diagnostics,
    TimeAxis,
    role_attrs,
)
from jem.components.clock import clock_tolerance_seconds
from jem.components.jcm import exchange_fields

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0

#: Fields of :class:`jcm.forcing.ForcingData` that a coupled run's exchangers
#: write -- the surface boundary conditions an uncoupled JCM run prescribes
#: and a coupled one receives from the surface components (see
#: :func:`jem.exchangers.default_exchanges`). Where one of these appears in
#: JCM's own output dataset it is tagged ``jem_role = "forcing"``, so the
#: atmosphere's output can be queried for its forcing the same way every
#: other component's can. The rest of JCM's variables are left untagged:
#: those names are JCM's, and the roles of its diagnostics are not JEM's to
#: assert.
#:
#: This is the *standard* set, and it is used for output tagging only. Which
#: of these a given run actually receives depends on which surface components
#: it was built with, so the set that decides the carry's structure is the one
#: passed to :meth:`JCMComponent.set_exchanged_forcing`.
FORCING_VARIABLE_NAMES = (
    "sea_surface_temperature",
    "sice_am",
    "stl_am",
    "snowc_am",
    "soilw_am",
)


@tree_math.struct
class JCMDerived:
    """What the atmosphere publishes for the other components to read.

    ``physics`` is JCM's own per-step diagnostics dict, carried through
    opaquely so an exchanger can reach any field JCM computes without this
    module having to enumerate them; its keys depend on which physics terms
    the model was built with. The named fields are the surface exchange in
    JEM's conventions (see
    :class:`~jem.components.jcm.exchange_fields.SurfaceExchange`).

    ``total_freshwater_flux`` is ``evaporation - precipitation``: positive
    upward, ``kg m-2 s-1``. It is stored rather than recomputed by each
    exchanger because that is the quantity an ocean or land surface takes,
    and one definition of the sign is safer than several.
    """

    physics: Any
    total_heat_flux: jnp.ndarray
    total_freshwater_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    u0: jnp.ndarray
    v0: jnp.ndarray

    @classmethod
    def zeros(cls, shape, physics, **overrides):
        """Zero-filled derived fields on a ``shape`` horizontal grid.

        Parameters
        ----------
        shape : tuple of int
            Horizontal nodal shape ``(ix, il)``.
        physics : Any
            Structural template for the opaque ``physics`` passthrough; it
            must have the pytree structure, shapes and dtypes a real step
            produces, or the coupled ``lax.scan`` rejects the carry after
            the first step.
        **overrides
            Named fields to use instead of zeros.

        """
        fields = {
            name: overrides.get(name, jnp.zeros(shape))
            for name in (
                "total_heat_flux",
                "total_freshwater_flux",
                "evaporation",
                "precipitation",
                "u0",
                "v0",
            )
        }
        return cls(physics, **fields)


def _with_model_context(predictions: ModelPredictions,
                        model: Model) -> ModelPredictions:
    """Re-attach the coords/physics/dycore a pytree round-trip dropped.

    ``ModelPredictions`` is registered as a pytree whose only children are
    the raw prediction arrays, so everything JEM's ``lax.scan`` hands back
    has ``coords``, ``physics`` and ``dycore`` set to ``None`` and cannot
    serialize itself. ``ModelPredictions.with_context(model)`` is jax-gcm's
    own spelling of the repair; the model-bound form is used rather than the
    ``(coords, physics)`` one so that the dycore -- which owns the
    trajectory-to-Dataset conversion for non-separable grids -- comes along
    with them. The one-line wrapper earns its place by putting that reason
    next to JEM's ``lax.scan``, which is what creates the need.

    One visible consequence: the dataset's ``jcm_prov_params`` global
    attribute gains jax-gcm's ``parameters_rederived_from_live_context``
    note (and therefore a different ``jcm_prov_params_sha``). That is
    accurate and wanted. A coupled trajectory is traced once and scanned, so
    the parameters recorded here really are read from the live physics
    afterwards rather than captured at trace time, and a reader of a coupled
    atmosphere file should be told so rather than be shown a provenance
    record that claims more than it knows.
    """
    return predictions.with_context(model)


def _diagnostics_template(model: Model) -> Any:
    """Structural template matching one step's saved physics diagnostics.

    JCM's averaged output path accumulates the per-step diagnostics dict into
    a zero template built from ``Physics.get_empty_data(coords)``, minus the
    ``_sampler_state`` entry, which stays in the integration carry but is
    never saved. Reproducing both transforms here is what lets
    :meth:`JCMComponent.initialize` seed ``JCMDerived.physics`` with the exact
    structure, shapes and dtypes step 1 will produce — without integrating a
    step to find out, which is what the previous adapter did.

    Only *inexact* (float) leaves are cast to float: a categorical (integer or
    boolean) diagnostic is not meaningfully averaged, so jax-gcm's own
    accumulator leaves it at its native dtype and simply keeps the latest
    step's raw value (``jcm.model._op_split_trajectory``'s ``inner_step``,
    which JCM's own release notes describe as "categorical" diagnostics --
    see ``ModelPredictions.to_xarray``'s ``omitted_interval_mean_variables``).
    Every other leaf is float-cast because it *is* divided by the number of
    inner steps. This mirrors JCM's own accumulator seed
    (``jcm.model._get_op_split_integrate_fn``'s ``empty_diag_sum``) leaf for
    leaf; blanket-casting every leaf to float here (as before jax-gcm PR 878,
    when every leaf — categorical ones included — WAS promoted) now produces
    a template whose categorical leaves disagree in dtype with what a real
    step actually returns, which ``lax.scan`` catches as a carry-structure
    error on the first coupled step.

    A mismatch would surface as a ``lax.scan`` carry-structure error on the
    first coupled step, so it is checked directly by the component's tests
    rather than assumed.
    """
    template = model.physics.get_empty_data(model.coords)
    template = {k: v for k, v in template.items() if k != "_sampler_state"}
    return jax.tree.map(
        lambda leaf: (jnp.zeros_like(leaf, dtype=float)
                      if jnp.issubdtype(leaf.dtype, jnp.inexact)
                      else jnp.zeros_like(leaf)),
        template,
    )


def _forcing_field_names() -> frozenset[str]:
    """Return the field names of :class:`jcm.forcing.ForcingData`.

    Read from the struct rather than listed here, so a jax-gcm release that
    adds a boundary condition needs no change on this side for it to be
    nameable as an exchanged field.
    """
    return frozenset(field.name for field in dataclasses.fields(ForcingData))


def _collapse_exchanged_forcing(
    forcing: ForcingData,
    names: tuple[str, ...],
    date: jdt.Datetime,
) -> ForcingData:
    """Return ``forcing`` with the ``names`` fields taken at ``date``.

    The named fields are the ones a coupled run's exchangers overwrite every
    coupling step. Left as :class:`jcm.forcing.TimeSeries` leaves they would
    make the atmosphere's carry change pytree structure the first time an
    exchanger wrote a plain array into one, which ``lax.scan`` cannot carry;
    collapsed here they are the ``(ix, il)`` arrays an exchange writes, and
    the structure is the same before and after.

    The value they are collapsed *to* is the climatology at ``date``, which
    is what the atmosphere would have seen at that date in an uncoupled run.
    It survives only until the first exchange under the default workflow
    (``exchange`` runs before ``atm``), and is what the atmosphere integrates
    its first step on under a workflow that steps the atmosphere first.

    Only the named fields are taken from the slice: everything else --
    including :attr:`~jcm.forcing.ForcingData.solar`, which
    :meth:`~jcm.forcing.ForcingData.select` fills in as a by-product -- is
    left exactly as jax-gcm built it, because jax-gcm slices the forcing
    again on every internal timestep and is the right place for that to
    happen.

    Parameters
    ----------
    forcing : jcm.forcing.ForcingData
        The boundary conditions as jax-gcm built them.
    names : tuple of str
        Fields of ``ForcingData`` the coupled model supplies.
    date : jax_datetime.Datetime
        The run's start date.

    Returns
    -------
    jcm.forcing.ForcingData

    Notes
    -----
    jax-gcm PR 878 dropped ``calendar`` from both ``DateData.set_date`` and
    ``ForcingData.select``: a wrap-year climatology is now always replayed on
    the real Gregorian calendar (``docs/source/v2_to_v3.rst``, "One real
    datetime clock"), so there is no longer a calendar to pass here.

    """
    if not names:
        return forcing
    at_date = forcing.select(DateData.set_date(date))
    # ``tree_math.struct`` generates ``replace`` at runtime, so mypy cannot
    # see it on the struct.
    return forcing.replace(  # type: ignore[no-any-return]
        **{name: getattr(at_date, name) for name in names}
    )


def _collapse_save_axis(leaf: jnp.ndarray) -> jnp.ndarray:
    """Merge a stacked leaf's ``(coupling step, save)`` axes into one time axis.

    Each coupled step runs JCM for exactly one save interval, so every leaf
    the coupler stacks is ``(iterations, 1, ...)``; JCM's own serialization
    wants a single leading time axis.
    """
    if getattr(leaf, "ndim", 0) < 2:
        return leaf
    return leaf.reshape((-1, *leaf.shape[2:]))


def _collapse_time_cell_method(predictions: ModelPredictions) -> ModelPredictions:
    """Collapse a stacked ``time_cell_method`` flag back to jax-gcm's expected scalar.

    jax-gcm PR 878 added ``time_cell_method`` -- a single JAX boolean on the
    whole prediction frame (true for an interval mean), read by
    ``ModelPredictions.time_labels``/``.to_xarray`` as a bare Python ``bool``.
    It has no per-record axis of its own -- one ``run_from_state_with_carry``
    call produces one flag for its whole trajectory, not one per saved frame
    -- but :meth:`JCMComponent.step` always calls it with the same fixed
    ``output_averages=True``, and the coupler's ``lax.scan`` necessarily
    stacks this leaf to shape ``(iterations,)`` like every other one: to JAX
    it is structurally identical to a genuine per-record leaf. jax-gcm's own
    ``bool(np.asarray(...))`` then raises on more than one element. Every
    stacked copy is identical (the fixed argument never varies), so this
    keeps only the first.

    There is no public accessor for this field, so this reaches
    ``ModelPredictions._predictions.time_cell_method`` -- the same private
    attribute jax-gcm's own ``time_labels``/``.to_xarray`` read internally
    (``getattr(self._predictions, "time_cell_method", None)``) -- rather than
    reimplementing the read; ``jem/components/jcm/contract.py`` pins it as a
    private, watched integration point so a future rename is caught by
    ``test_jcm_contract.py``, not by this function returning silently wrong
    output. Tracked upstream as jax-gcm#907, "A stacked ModelPredictions
    can't be labelled or serialised: time_cell_method has no public
    accessor" -- a public way to rebuild a stacked trajectory's scalar
    metadata (or a setter) would let this function go away.
    """
    raw = getattr(predictions, "_predictions", None)
    cell_method = getattr(raw, "time_cell_method", None) if raw is not None else None
    if cell_method is None or getattr(cell_method, "ndim", 0) == 0:
        return predictions
    return ModelPredictions(
        raw.replace(time_cell_method=cell_method[0]),  # type: ignore[union-attr]
        None, None,
        observations=getattr(predictions, "_observations", None),
    )


class JCMComponent:
    """The JCM atmosphere, driven one coupling timestep at a time.

    Satisfies :class:`~jem.base.component.Component`,
    :class:`~jem.base.component.SupportsBind` and
    :class:`~jem.base.component.SupportsXarray`.

    Parameters
    ----------
    model : jcm.model.Model
        A fully configured JCM model. Its ``start_time`` must match the
        coupler's ``start_date``, and the coupler must run on the
        ``"gregorian"`` calendar -- jax-gcm's own clock is unconditionally
        Gregorian since v3 (PR 878) and no longer has a ``calendar`` of its
        own to check against. :meth:`bind` checks both.
    forcing : jcm.forcing.ForcingData, optional
        Boundary conditions for the atmosphere. Defaults to JCM's
        :func:`~jcm.forcing.default_forcing` (prescribed SSTs) on the
        model's grid. It lives in the carry, not on ``self``, because
        exchangers overwrite parts of it (the SST an ocean component
        computes) every coupling step.
    exchanged_forcing : iterable of str, optional
        Names of ``forcing`` fields the coupled model supplies -- the ones
        an exchanger writes into ``atm.forcing`` every coupling step. They
        are collapsed to plain per-step arrays by :meth:`initialize` so the
        carry's structure survives the first exchange; see
        :meth:`set_exchanged_forcing`, which is how
        :func:`jem.runners.build_coupler` sets them from the coupling table
        once the exchangers are known. Empty by default: an atmosphere no
        exchanger writes to keeps every boundary condition exactly as
        jax-gcm built it.

    Attributes
    ----------
    name : str
        ``"atm"``.

    """

    name = "atm"

    def __init__(
        self,
        model: Model,
        *,
        forcing: ForcingData | None = None,
        exchanged_forcing: Iterable[str] = (),
    ) -> None:
        """Wrap ``model``; see the class docstring for the parameters."""
        self.model = model
        self.forcing = (forcing if forcing is not None
                        else default_forcing(model.coords.horizontal))
        self._exchanged_forcing: tuple[str, ...] = ()
        self.set_exchanged_forcing(exchanged_forcing)
        # Horizontal nodal shape (ix, il); the leading axis of
        # ``coords.nodal_shape`` is the vertical.
        self.nodal_shape = tuple(model.coords.nodal_shape[1:])
        # Coupling interval in days, as a Python float. Static on purpose:
        # ``run_from_state_with_carry`` takes ``save_interval`` and
        # ``total_time`` as static arguments of a jit, so they cannot be
        # traced values. ``None`` until bind() has run.
        self._coupling_days: float | None = None

    @property
    def exchanged_forcing(self) -> tuple[str, ...]:
        """Forcing fields the coupled model supplies, in the order declared."""
        return self._exchanged_forcing

    @property
    def time_varying_forcing(self) -> tuple[str, ...]:
        """Forcing fields jax-gcm built as time series, in ``ForcingData`` order.

        These are the fields that carry three pytree leaves rather than one,
        and so the ones that have to be named to
        :meth:`set_exchanged_forcing` if a coupler is going to write to them.
        A caller that cannot read its coupling off a table -- a hand-written
        exchanger -- uses this to check what it has to declare.

        Top-level fields only: a boundary condition jax-gcm keeps inside a
        mapping (prescribed emissions, oxidants) is not a field an exchange
        spec can address, so it is not reported here.

        Returns
        -------
        tuple[str, ...]

        """
        return tuple(
            field.name for field in dataclasses.fields(self.forcing)
            if isinstance(getattr(self.forcing, field.name), TimeSeries)
        )

    def set_exchanged_forcing(self, names: Iterable[str]) -> None:
        """Declare which forcing fields an exchanger writes every coupling step.

        A coupled atmosphere does not prescribe the surface it stands on --
        the surface components do -- so the fields named here stop being
        boundary conditions the atmosphere reads from a file and become
        per-step values it is handed. :meth:`initialize` collapses them to
        the climatology at the run's start date, which is what gives the
        carry's ``"forcing"`` section the same pytree structure before and
        after an exchange. Fields that are *not* named keep whatever jax-gcm
        built them as: a :class:`jcm.forcing.TimeSeries` stays a time series
        and goes on being sliced per internal timestep, so an unexchanged
        climatology -- the land surface in a run without a land model -- still
        varies through the year.

        Which fields those are is a property of the coupled model, not of the
        atmosphere, which is why it is set from outside rather than assumed
        here: an aquaplanet with no land model must keep the file's land
        climatology, while :func:`jem.runners.build_coupler` derives the set
        from the coupling table it just built
        (:func:`jem.exchangers.exchanged_fields`).

        Call it before :meth:`initialize`; the carry is built there, so a
        later change does not reach a carry already made.

        Parameters
        ----------
        names : iterable of str
            Field names of :class:`jcm.forcing.ForcingData`. Duplicates
            are collapsed; an empty iterable clears the declaration.

        Raises
        ------
        ValueError
            If a name is not a field of ``ForcingData``. Raised here, while
            the model is being built, rather than as a ``TypeError`` from
            ``replace()`` inside the traced coupled step.

        """
        declared = tuple(dict.fromkeys(names))
        known = _forcing_field_names()
        unknown = [name for name in declared if name not in known]
        if unknown:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: exchanged_forcing names "
                f"{', '.join(repr(name) for name in unknown)}, which "
                "jcm.forcing.ForcingData does not have -- so no exchanger can "
                f"write to '{self.name}.forcing.<that name>' either. "
                f"ForcingData's fields are {sorted(known)!r}."
            )
        self._exchanged_forcing = declared

    def bind(
        self,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str,
    ) -> None:
        """Adopt the coupler's clock, or refuse if the model disagrees with it.

        Parameters
        ----------
        coupling_timestep : jax_datetime.Timedelta
            The coupled model's timestep. It must be an exact multiple of
            the model's own timestep, because JCM advances by whole
            timesteps and a coupling interval that is not a multiple of one
            would silently be rounded.
        start_date : jax_datetime.Datetime
            The run's start date; must equal ``model.start_time``.
        calendar : str
            The run's calendar; must be ``"gregorian"``.

        Raises
        ------
        ValueError
            If the timestep does not divide, or either clock setting
            differs. The message names both values: a mismatch here means
            the atmosphere would date its own forcing and output
            differently from every other component. Also if the component
            is already bound to a different coupling timestep (binding it
            again to the same clock is a no-op).

        """
        # jax-gcm v3 (PR 878) removed ``Model.calendar``: the atmosphere's
        # own clock -- physics seasonal phase, forcing alignment, output
        # labelling -- is unconditionally proleptic Gregorian now, with no
        # configuration knob to check this against. So the check runs the
        # other way: the coupler itself must be on the one calendar jax-gcm
        # still understands. Any other choice would not fail loudly -- the
        # atmosphere would simply run Gregorian regardless -- it would
        # silently put the atmosphere's seasonal cycle out of phase with
        # every other component's, which reads its calendar from the
        # coupler (:func:`jem.base.component.days_per_year`).
        if str(calendar) != "gregorian":
            raise ValueError(
                f"{self.name!r} needs the coupler's calendar to be "
                f"'gregorian', not {calendar!r}: jax-gcm v3's atmosphere "
                "clock (physics seasonal phase, forcing alignment, output "
                "labelling) is unconditionally proleptic Gregorian and has "
                "no calendar of its own to match. Build the Coupler with "
                "calendar='gregorian'."
            )
        if start_date != self.model.start_time:
            raise ValueError(
                f"Start-date mismatch: the coupler starts at {start_date!r}"
                f" but {self.name!r} was built with start_time="
                f"{self.model.start_time!r}. Rebuild the model with"
                " start_time=<coupler start date>."
            )
        model_timestep = jdt.to_timedelta(
            int(self.model.dt_si.to_timedelta().total_seconds()), "second")
        n_steps = float(coupling_timestep / model_timestep)
        if n_steps != int(n_steps) or n_steps < 1:
            raise ValueError(
                f"Coupling timestep {coupling_timestep!r} is not a whole"
                f" multiple of {self.name!r}'s model timestep"
                f" {model_timestep!r}."
            )
        coupling_days = float(coupling_timestep / jdt.to_timedelta(1, "day"))
        if self._coupling_days is not None and coupling_days != self._coupling_days:
            # One instance belongs to one coupled model: `step` advances the
            # atmosphere by `_coupling_days`, so a second coupler with another
            # timestep would silently desynchronise the first coupler's runs
            # from its own step counter.
            raise ValueError(
                f"{type(self).__name__} {self.name!r} is already bound to a "
                f"coupling timestep of {self._coupling_days:g} days and cannot "
                f"also be bound to {coupling_days:g} days. Build a separate "
                "instance per coupled model."
            )
        self._coupling_days = coupling_days

    def initialize(self) -> Carry:
        """Build the initial carry without integrating the model.

        The ``"forcing"`` entry is the boundary conditions this component was
        built with, with the fields :meth:`set_exchanged_forcing` names
        collapsed to their value at the model's start date. That collapse is
        what fixes the carry's pytree structure: an exchanger writes a plain
        ``(ix, il)`` array into each of those fields every coupling step, and
        a ``lax.scan`` carry may not change structure between steps.

        The ``"time"``/``"step"`` entries are JCM's own exact clock
        (``jcm.model.RunState.time`` / ``.step``): jax-gcm PR 878 requires
        ``run_from_state_with_carry`` to be given this pair explicitly on
        every call rather than inferring it from the incoming dycore state
        (see ``docs/source/v2_to_v3.rst``, "One real datetime clock"), so
        :meth:`step` threads them the same way it already threads
        ``"physics"``: read from ``carry``, passed as ``initial_time`` /
        ``initial_step``, and replaced with the exact ``RunState`` the call
        returns. JCM's ``RunState`` is the **authoritative** clock now (jax-gcm
        v3, PR 878), and jax-gcm's own migration guide
        (``docs/source/v2_to_v3.rst``, "One real datetime clock") says
        explicitly to keep threading it rather than deriving it from a step
        counter kept elsewhere -- threading it is what guarantees JCM's clock
        and the coupler's can never disagree about what instant a step is at,
        which recomputing one from the coupler's own step count every call
        would not, once a checkpoint or a differently-configured coupler is
        involved (:meth:`_report_authoritative_clock_drift` is the check that
        catches exactly that disagreement, and does so with the same
        int32-safe arithmetic -- :func:`jem.base.calendar.gregorian_instant`
        -- that recomputing the clock itself would use, so overflow was never
        the reason to thread it).

        Returns
        -------
        dict
            ``{"state": dycore state, "physics": cross-step physics carry,
            "time": exact jax_datetime.Datetime, "step": exact JCM step
            count, "derived": JCMDerived, "forcing": ForcingData}``.

        """
        dycore_state, physics_carry = self.model.bootstrap_state()
        return {
            "state": dycore_state,
            "physics": physics_carry,
            "time": self.model.start_time,
            "step": jnp.int32(0),
            "derived": JCMDerived.zeros(
                self.nodal_shape, _diagnostics_template(self.model)),
            "forcing": _collapse_exchanged_forcing(
                self.forcing,
                self._exchanged_forcing,
                self.model.start_time,
            ),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the atmosphere by one coupling timestep.

        Parameters
        ----------
        carry : dict
            The carry :meth:`initialize` produced, as last returned.
        time : jem.base.component.CouplingTime
            The coupler's clock for this step.

        Returns
        -------
        tuple
            The new carry and JCM's :class:`~jcm.predictions.ModelPredictions`
            for the interval. The predictions object is returned whole so
            the coupler can stack it and :meth:`to_xarray` can hand it back
            to JCM's own serialization.

        Raises
        ------
        RuntimeError
            If the component has not been bound to a coupler clock.

        """
        if self._coupling_days is None:
            raise RuntimeError(
                f"{type(self).__name__} {self.name!r} has no coupling"
                " timestep: register it with a Coupler (which calls bind())"
                " before stepping it."
            )
        self._report_clock_drift(carry["state"], time)
        self._report_authoritative_clock_drift(carry["time"], carry["step"], time)

        run_state, predictions = self.model.run_from_state_with_carry(
            initial_state=carry["state"],
            forcing=carry["forcing"],
            save_interval=self._coupling_days,
            total_time=self._coupling_days,
            output_averages=True,
            initial_physics_state=carry["physics"],
            initial_time=carry["time"],
            initial_step=carry["step"],
        )
        # One coupling step is exactly one save interval, so the saved
        # trajectory has a length-1 leading axis; the derived fields are
        # per-step maps, not trajectories.
        diagnostics = jax.tree.map(lambda leaf: leaf[0], predictions.physics)
        exchange = exchange_fields.from_diagnostics(diagnostics)
        # ``tree_math.struct`` builds the dataclass at runtime, so mypy
        # cannot see the generated __init__ signature.
        derived = JCMDerived(  # type: ignore[call-arg]
            diagnostics,
            total_heat_flux=exchange.total_heat_flux,
            total_freshwater_flux=exchange.evaporation - exchange.precipitation,
            evaporation=exchange.evaporation,
            precipitation=exchange.precipitation,
            u0=exchange.u0,
            v0=exchange.v0,
        )
        return (
            {
                "state": run_state.dynamics,
                "physics": run_state.physics,
                "time": run_state.time,
                "step": run_state.step,
                "derived": derived,
                "forcing": carry["forcing"],
            },
            predictions,
        )

    def to_xarray(self, diagnostics: Diagnostics, time: TimeAxis) -> xr.Dataset:
        """Serialize the stacked per-step predictions through JCM.

        Parameters
        ----------
        diagnostics : jcm.predictions.ModelPredictions
            The per-step predictions stacked by the coupler, so every leaf
            carries a ``(iterations, 1, ...)`` pair of leading axes.
        time : jem.base.component.TimeAxis
            The coupler's time axis, used to check the record count.

        Returns
        -------
        xarray.Dataset
            Whatever JCM's own ``ModelPredictions.to_xarray`` produces.

        Notes
        -----
        Every variable in the dataset is JCM's, named as JCM names it, so the
        ``forcing_`` prefix the other components use is deliberately not
        applied here -- renaming JCM's output would make a coupled run's
        atmosphere files disagree with an uncoupled run's. The variables that
        *are* recognisably the surface forcing an exchanger writes
        (:data:`FORCING_VARIABLE_NAMES`) are marked with the ``jem_role``
        attribute where they appear; everything else is left untagged rather
        than guessed at, because JCM owns those names and their meaning.

        The ``time`` coordinate is JCM's, not the coupler's: JCM labels each
        averaged record with the **midpoint** of the interval it covers
        (exact ``datetime64[ms]``, absolute, from the model's own
        ``start_time`` -- jax-gcm v3's ``jcm.predictions.output_time_labels``,
        see ``docs/source/v2_to_v3.rst``, "One real datetime clock"), and JEM
        does not relabel it, because a coupled dataset in which the
        atmosphere's time axis disagrees with the atmosphere's own output
        files would be worse than one where two components label the same
        interval differently. The labels the other components carry come from
        ``TimeAxis.datetimes``, which since jax-gcm#862 (closed by PR 878)
        calls the same ``output_time_labels`` conversion directly rather than
        reimplementing it -- see :class:`jem.base.component.TimeAxis`.

        """
        collapsed = jax.tree.map(_collapse_save_axis, diagnostics)
        collapsed = _collapse_time_cell_method(collapsed)
        predictions = _with_model_context(collapsed, self.model)
        n_records = int(jnp.shape(predictions.times)[0])
        if len(time) != n_records:
            raise ValueError(
                f"{self.name!r} produced {n_records} output records but the"
                f" coupler's time axis has {len(time)}; the diagnostics"
                " passed here are not the ones this run produced."
            )
        dataset: xr.Dataset = predictions.to_xarray()
        for name in FORCING_VARIABLE_NAMES:
            if name in dataset.variables:
                # A fresh dict: xarray keeps the one it is handed, and the
                # attrs on JCM's variable are JCM's to own.
                dataset[name].attrs = {
                    **dataset[name].attrs, **role_attrs("forcing")
                }
        return dataset

    def _report_clock_drift(self, state: Any, time: CouplingTime) -> None:
        """Log at ERROR if the dycore state's own clock has left the coupler's.

        The dycore state carries its own ``sim_time`` in seconds and JCM
        advances it independently of the coupler's step counter, so the two
        can only disagree if the carry did not come from this run: a
        checkpoint restored into a coupler with a different start date, or a
        carry threaded into the wrong component.

        The tolerance grows with simulation time
        (:func:`jem.components.clock.clock_tolerance_seconds`, shared with the
        other wrappers that make this check), because both counters are
        float32 by default and would otherwise disagree by float32 rounding
        alone after a few decades.

        Reported rather than raised, and through ``jax.debug.callback``
        rather than ``checkify``: the check runs inside the coupled
        ``lax.scan``, where a Python exception cannot fire on a traced value
        and where aborting the scan would throw away a run that may still be
        salvageable. The message is loud enough to find in a log.
        """
        state_sim_time = self.model.dycore.sim_time(state)
        name = self.name

        def _report(model_seconds, coupler_seconds) -> None:
            drift = float(model_seconds) - float(coupler_seconds)
            # The tolerance is set by the COUPLER's time: it is the one that
            # is right by construction, so a model clock that is wildly wrong
            # cannot widen the window that would catch it.
            if abs(drift) > clock_tolerance_seconds(coupler_seconds):
                logger.error(
                    "%s: model clock is %.6g s from the coupler's"
                    " (model %.6g s, coupler %.6g s). The atmosphere will"
                    " date its forcing and output differently from the rest"
                    " of the coupled model.",
                    name, drift, float(model_seconds), float(coupler_seconds),
                )

        jax.debug.callback(_report, state_sim_time, time.sim_time)

    def _report_authoritative_clock_drift(
        self, carry_time: jdt.Datetime, carry_step: Any, time: CouplingTime
    ) -> None:
        """Log at ERROR if the carry's own exact clock has left the coupler's.

        :meth:`_report_clock_drift` checks the dycore state's own ``sim_time``
        -- a *derived* quantity JCM keeps for its own physics -- against the
        coupler's; it does not touch ``carry["time"]``/``carry["step"]``,
        which are the **authoritative** clock since jax-gcm v3 (PR 878):
        every call to :meth:`step` reads them from the carry and threads back
        exactly the ``RunState`` the call returns (see :meth:`initialize`),
        and nothing else in this class recomputes them. A checkpoint restored
        into a coupler with a different start date silently mismatches
        ``carry["time"]`` against what the coupler's own clock says this step
        should be -- and nothing before this check ever looked, so such a
        mismatch would go undetected all the way to wrong forcing dates and
        wrong output labels.

        The exact expected time is computed the same int32-safe way item
        A/B's calendar arithmetic is (:func:`jem.base.calendar
        .gregorian_instant`, a limb multiply-then-divide exact for any
        traced step counter an int32 can hold -- see that function's own
        docstring), from the coupler's own step count and coupling timestep
        rather than from anything JCM derives -- so this check does not
        depend on the very clock it is checking. Both quantities being
        compared are exact integers (whole
        days and seconds, and a step count), so unlike
        :meth:`_report_clock_drift` this needs no float32 tolerance: any
        difference at all is a real one.

        Reported rather than raised, and through ``jax.debug.callback`` for
        the same reason as :meth:`_report_clock_drift` -- this runs inside
        the coupled ``lax.scan``.
        """
        expected_step = time.step * self._inner_steps()
        record_seconds = round(time.dt)
        start_days = int(np.asarray(self.model.start_time.delta.days))
        start_seconds = int(np.asarray(self.model.start_time.delta.seconds))
        expected_days, expected_seconds = gregorian_instant(
            time.step, record_seconds, start_days, start_seconds
        )
        name = self.name

        def _report(
            carry_days, carry_seconds, exp_days, exp_seconds, carry_step_, exp_step
        ) -> None:
            if (
                int(carry_days) != int(exp_days)
                or int(carry_seconds) != int(exp_seconds)
                or int(carry_step_) != int(exp_step)
            ):
                logger.error(
                    "%s: the carry's own clock (time=%s+%ss, step=%d) does not "
                    "match the coupler's (time=%s+%ss, step=%d) -- a checkpoint "
                    "restored into a coupler with a different start date, "
                    "calendar or coupling timestep, or a carry threaded into "
                    "the wrong component. The atmosphere will date its forcing "
                    "and output differently from the rest of the coupled "
                    "model.",
                    name, int(carry_days), int(carry_seconds), int(carry_step_),
                    int(exp_days), int(exp_seconds), int(exp_step),
                )

        jax.debug.callback(
            _report,
            carry_time.delta.days, carry_time.delta.seconds,
            expected_days, expected_seconds,
            carry_step, expected_step,
        )

    def _inner_steps(self) -> int:
        """Return how many of the model's own timesteps make one coupled step.

        Static (Python int, never traced): resolved once from the facts
        :meth:`bind` already validated (the coupling timestep is a whole
        multiple of the model's own), not stored separately at bind time
        because :attr:`_coupling_days` (days) and the model's own timestep are
        already enough to recompute it exactly.
        """
        # Only called from `step`, which already refuses to run at all
        # (`RuntimeError`, before this) when the component has not been
        # bound -- the same condition that leaves `_coupling_days` `None`.
        assert self._coupling_days is not None
        model_timestep_seconds = int(
            self.model.dt_si.to_timedelta().total_seconds()
        )
        coupling_seconds = round(self._coupling_days * 86400.0)
        return coupling_seconds // model_timestep_seconds
