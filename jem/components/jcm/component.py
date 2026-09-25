"""The JCM atmosphere as a JEM :class:`~jem.base.component.Component`.

:class:`JCMComponent` wraps a :class:`jcm.model.Model` rather than
monkey-patching methods onto it, so the atmosphere JEM drives is the same
object the user configured and nothing in JCM has to know JEM exists.

Four things this wrapper exists to get right:

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

**The published surface exchange is put on the atmosphere's own grid.**
A physics package built with ``ComposablePhysics(vectorize_columns=True)``
(ECHAM) flattens the horizontal ``(ix, il)`` grid to a single ``ncols`` axis
before iterating its terms, and every diagnostic it writes -- including the
published surface exchange -- stays flattened; jax-gcm reshapes a
column-vectorized diagnostic back to the grid only inside its own xarray
serialization, never before. SPEEDY (``vectorize_columns=False``) never
flattens anything, so this went unnoticed until an ECHAM-composed coupled
model was actually run (jax-esm#129): every named field of
:class:`JCMDerived` is documented and consumed as an ``(ix, il)`` map, and a
flattened field copied into a plain ``(ix, il)`` component (a slab ocean, in
the default exchange table) fails that component's own step with an opaque
shape-mismatch error. :func:`_unflatten_to_nodal_shape` reshapes it back --
a no-op for SPEEDY, the fix for ECHAM -- and both :meth:`JCMDerived.zeros`
(the initial carry) and :meth:`JCMComponent.step` go through it via the one
shared :func:`_surface_exchange_on_nodal_grid`, so the two agree on shape
from the first step onward.

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
from typing import Any, overload

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import tree_math
import xarray as xr
from jcm.date import DateData
from jcm.forcing import ForcingData, TimeSeries, default_forcing
from jcm.model import Model
from jcm.predictions import ModelPredictions

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


@overload
def _unflatten_to_nodal_shape(
    value: jnp.ndarray, nodal_shape: tuple[int, int], field_name: str
) -> jnp.ndarray: ...
@overload
def _unflatten_to_nodal_shape(
    value: None, nodal_shape: tuple[int, int], field_name: str
) -> None: ...
def _unflatten_to_nodal_shape(
    value: jnp.ndarray | None, nodal_shape: tuple[int, int], field_name: str
) -> jnp.ndarray | None:
    """Reshape a column-vectorized ``(ncols,)`` field back to ``(ix, il)``.

    A physics package built with ``ComposablePhysics(vectorize_columns=True)``
    (ECHAM) flattens the horizontal ``(ix, il)`` grid to a single ``ncols``
    axis before iterating its terms (jax-gcm's
    ``jcm/physics/composable_physics.py``, ``_compute_tendencies_columns``),
    and every diagnostic it writes -- including the published
    ``surface_exchange`` struct :mod:`~jem.components.jcm.exchange_fields`
    reads -- stays on that flattened axis: jax-gcm only reshapes a
    column-vectorized diagnostic back to the grid inside its own xarray
    serialization (``ComposablePhysics.data_struct_to_dict``, used by
    ``ModelPredictions.to_xarray``), never before. A physics package built
    with ``vectorize_columns=False`` (SPEEDY) keeps every diagnostic on
    ``(ix, il)`` throughout, so nothing here ever needs reshaping for it.

    Not reshaping here was a real, if narrow, bug this function closes:
    every named field of :class:`JCMDerived` is documented and consumed as an
    ``(ix, il)`` map (see :class:`~jem.components.jcm.exchange_fields.
    SurfaceExchange`'s own docstring), which every exchanger and every other
    component's grid assumes -- and a coupled step that copied a flattened
    ``(ncols,)`` field into a component built on the real ``(ix, il)`` grid
    (:func:`jem.exchangers.default_exchanges`'s ``atm`` -> ``ocn`` heat-flux
    row, applied by a plain slab ocean) failed with an opaque
    ``ValueError: Incompatible shapes for broadcasting`` deep inside that
    component's own step, not a message naming the field or the package that
    caused it. Discovered running the jax-esm#129 regression test
    (``tests/unit/test_coupled.py::
    test_echam_coupled_to_a_slab_ocean_completes_several_steps``) against a
    real ECHAM model -- no fabricated diagnostics fixture reproduces it,
    because a fixture built with already-gridded arrays never exercises the
    flattened case a real column-vectorized step actually produces.

    Reshaping with a plain ``.reshape(nodal_shape)`` (default, row-major
    order) is not a guess: it is the exact inverse of jax-gcm's own flatten,
    which merges ``(ix, il)`` into ``ncols`` the same way (a plain
    ``.reshape``, lon-major -- ``_flattened_column_sharding``'s "lon-major
    flatten" comment), and it is character-for-character the same reshape
    jax-gcm's own ``data_struct_to_dict`` applies before xarray ever sees a
    column-vectorized diagnostic. So this mirrors an existing, validated
    jax-gcm convention rather than inventing a new one.

    Parameters
    ----------
    value : jax.Array or None
        One field of a translated :class:`~jem.components.jcm.
        exchange_fields.SurfaceExchange`. ``None`` (no wind vector, see that
        module's docstring) passes through unchanged.
    nodal_shape : tuple of int
        The atmosphere's horizontal nodal shape, ``(ix, il)``
        (:attr:`JCMComponent.nodal_shape`).
    field_name : str
        The field's name, used only to name it in the error message below.

    Returns
    -------
    jax.Array or None
        ``value`` reshaped to ``nodal_shape`` if it was a flattened
        ``(ncols,)`` array, else ``value`` unchanged (already ``(ix, il)``,
        as every SPEEDY field already is).

    Raises
    ------
    ValueError
        If ``value`` is neither shape ``(ncols,)`` (the column-vectorized
        case this function exists to fix) nor already ``nodal_shape`` (the
        SPEEDY case, a no-op) -- e.g. a future package publishing a
        per-column field with an extra trailing axis, such as ``(ncols, 1)``.
        Silently passing such a shape through, as the pre-review version of
        this function did, would leave it to fail downstream as an opaque
        broadcast error naming neither this field nor why it is malformed.

    """
    if value is None:
        return None
    ncols = nodal_shape[0] * nodal_shape[1]
    if value.shape == nodal_shape:
        return value
    if value.ndim == 1 and value.shape[0] == ncols:
        return value.reshape(nodal_shape)
    raise ValueError(
        f"{field_name!r} has shape {value.shape}, which is neither the "
        f"atmosphere's nodal shape {nodal_shape!r} nor a flattened "
        f"(ncols,) = ({ncols},) column-vectorized field. "
        "_unflatten_to_nodal_shape only knows how to reshape those two "
        "cases (jax-esm#129); a physics package publishing surface exchange "
        "on some other layout needs this function taught the new shape "
        "explicitly, not a silent pass-through that fails later as an "
        "opaque broadcast error."
    )


def _surface_exchange_on_nodal_grid(
    diagnostics: dict[str, Any], nodal_shape: tuple[int, int]
) -> exchange_fields.SurfaceExchange:
    """:func:`~jem.components.jcm.exchange_fields.from_diagnostics`, gridded.

    The single seam :meth:`JCMDerived.zeros` and
    :meth:`JCMComponent.step` both go through, so the exchange a coupled
    model's initial (all-zero) carry is built from and the one a real step
    later produces are put on the same grid the same way -- see
    :func:`_unflatten_to_nodal_shape`.
    """
    exchange = exchange_fields.from_diagnostics(diagnostics)
    return exchange_fields.SurfaceExchange(
        total_heat_flux=_unflatten_to_nodal_shape(
            exchange.total_heat_flux, nodal_shape, "total_heat_flux"),
        evaporation=_unflatten_to_nodal_shape(
            exchange.evaporation, nodal_shape, "evaporation"),
        precipitation=_unflatten_to_nodal_shape(
            exchange.precipitation, nodal_shape, "precipitation"),
        u0=_unflatten_to_nodal_shape(exchange.u0, nodal_shape, "u0"),
        v0=_unflatten_to_nodal_shape(exchange.v0, nodal_shape, "v0"),
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

    ``u0``/``v0`` (the near-surface wind's true-east/true-north components)
    are ``None`` for a composed physics package that publishes no wind
    *vector* -- today, everything but SPEEDY (see
    :mod:`jem.components.jcm.exchange_fields`'s module docstring). This is a
    **static** property of the composed physics, decided once by
    :meth:`zeros` from a structural template and never revisited by
    :meth:`~jem.components.jcm.component.JCMComponent.step` (jax-esm#129):
    ``None`` is an empty JAX pytree node, contributing no leaf, so a
    ``JCMDerived`` whose ``u0``/``v0`` are ``None`` on step 0 has exactly the
    same pytree structure on every later step, which is what lets it be
    carried through ``jit`` and the coupled ``lax.scan``, saved and restored
    by a checkpoint (``jem/checkpoint.py``), and merged into a coupled
    model's output the same way any other absent field of a carry already
    is. Absence is never faked as a zero, a NaN, or a value derived from
    ``wind_speed`` or the published stress (see the module docstring cited
    above for why) -- a consumer that actually needs the wind vector
    (:class:`jem.fluxes.VerosExchange`) is instead refused at composition
    time, before this ever reaches a coupled step.
    """

    physics: Any
    total_heat_flux: jnp.ndarray
    total_freshwater_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    u0: jnp.ndarray | None
    v0: jnp.ndarray | None

    @classmethod
    def zeros(cls, physics, nodal_shape, **overrides):
        """Zero-filled derived fields shaped like a real step's, off a template.

        Every named field's shape is read off ``physics`` itself, through
        :func:`~jem.components.jcm.exchange_fields.from_diagnostics` -- the
        same reader a real step's diagnostics go through -- and then put on
        ``nodal_shape`` by :func:`_unflatten_to_nodal_shape`, rather than
        this method assuming zeros of ``nodal_shape`` directly the way it
        used to. Composed physics that keeps its state on the atmosphere's
        own grid (SPEEDY) already agrees with ``nodal_shape``, so this is a
        no-op for it; one that vectorizes columns
        (``ComposablePhysics(vectorize_columns=True)``, e.g. ECHAM) publishes
        the surface exchange on a flattened ``(ncols,)`` axis instead, and a
        template built with the wrong shape here would only be discovered
        wrong on step 1, as a ``lax.scan`` shape-mismatch error naming a flat
        pytree index rather than a field name.

        Parameters
        ----------
        physics : Any
            Structural (all-zero) template of one step's diagnostics dict,
            e.g. ``Physics.get_empty_data(coords)`` (as
            :meth:`JCMComponent.initialize` builds it) -- it must have the
            pytree structure, shapes and dtypes a real step produces, or the
            coupled ``lax.scan`` rejects the carry after the first step. Also
            what :func:`~jem.components.jcm.exchange_fields.from_diagnostics`
            decides ``u0``/``v0``'s presence from: ``None`` on this template
            reads back as ``None`` here too (see the class docstring).
        nodal_shape : tuple of int
            The atmosphere's horizontal nodal shape, ``(ix, il)``
            (:attr:`JCMComponent.nodal_shape`) -- the grid every named field
            is put on, whatever grid the composed physics happened to
            publish it on.
        **overrides
            Named fields to use instead of the template-derived defaults.

        Raises
        ------
        TypeError
            If ``physics`` is a tuple -- the signature was ``zeros(shape,
            physics, **overrides)`` before jax-esm#129 swapped the argument
            order (and what the first one means); a legacy positional call
            passes its old ``shape`` tuple where ``physics`` now goes, which
            otherwise fails several calls deep, as an opaque ``TypeError:
            tuple indices must be integers or slices, not str`` out of
            ``exchange_fields.from_diagnostics``'s ``dict.get`` -- naming
            neither this method nor the argument order that actually changed.

        """
        if isinstance(physics, tuple):
            raise TypeError(
                "JCMDerived.zeros(physics, nodal_shape, **overrides) takes "
                "the diagnostics template first and the atmosphere's nodal "
                f"shape second; got a tuple ({physics!r}) as the first "
                "argument. jax-esm#129 swapped both the order and the "
                "meaning of zeros()'s first two arguments (it used to be "
                "zeros(shape, physics, **overrides)) -- swap them at this "
                "call site."
            )
        exchange = _surface_exchange_on_nodal_grid(physics, nodal_shape)
        # `zeros_like`, not the exchange's own values: `physics` is an
        # all-zero template, so every field below is already mathematically
        # zero, but `total_heat_flux = -net_heat_flux`'s negation turns a
        # template's `+0.0` into `-0.0` (a distinct float bit pattern, code
        # review finding) -- and a SPEEDY run's `zeros()` used to give `+0.0`
        # unconditionally (`jnp.zeros(shape)`, no negation involved), so a
        # signed zero here would be a real, if invisible, regression against
        # the "SPEEDY bit-for-bit unchanged" guarantee. `zeros_like` keeps
        # every field's shape (already put on `nodal_shape` above) and dtype
        # while canonicalising the value to positive zero.
        defaults = {
            "total_heat_flux": jnp.zeros_like(exchange.total_heat_flux),
            "total_freshwater_flux": jnp.zeros_like(exchange.evaporation),
            "evaporation": jnp.zeros_like(exchange.evaporation),
            "precipitation": jnp.zeros_like(exchange.precipitation),
            "u0": None if exchange.u0 is None else jnp.zeros_like(exchange.u0),
            "v0": None if exchange.v0 is None else jnp.zeros_like(exchange.v0),
        }
        fields = {name: overrides.get(name, value) for name, value in defaults.items()}
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

    JCM's averaged output path accumulates the per-step diagnostics dict
    into a float-cast zero template built from
    ``Physics.get_empty_data(coords)``, minus the ``_sampler_state`` entry,
    which stays in the integration carry but is never saved. Reproducing
    both transforms here is what lets :meth:`JCMComponent.initialize` seed
    ``JCMDerived.physics`` with the exact structure, shapes and dtypes step
    1 will produce — without integrating a step to find out, which is what
    the previous adapter did.

    A mismatch would surface as a ``lax.scan`` carry-structure error on the
    first coupled step, so it is checked directly by the component's tests
    rather than assumed.
    """
    template = model.physics.get_empty_data(model.coords)
    template = {k: v for k, v in template.items() if k != "_sampler_state"}
    # ``dtype=float`` (the default float type, so float64 under
    # jax_enable_x64) exactly mirrors JCM's own accumulator, which promotes
    # every leaf — integer and boolean ones included — because it divides by
    # the number of inner steps.
    return jax.tree.map(lambda leaf: jnp.zeros_like(leaf, dtype=float), template)


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
    calendar: str,
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
    calendar : str
        The model's calendar, which decides how a wrap-year climatology is
        indexed.

    Returns
    -------
    jcm.forcing.ForcingData

    """
    if not names:
        return forcing
    at_date = forcing.select(DateData.set_date(date), calendar=calendar)
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

    The merged size is computed explicitly (``leaf.shape[0] * leaf.shape[1]``)
    rather than left for a ``-1`` placeholder to infer, because some composed
    physics diagnostics are legitimately zero-sized in their default
    configuration -- ECHAM's aerosol struct carries a per-species axis of
    length 0 with no aerosol species configured (jax-gcm's own uncoupled
    ``to_xarray`` drops these; jax-esm#129's regression test is what
    surfaced one going through this collapse first, in a coupled run). A
    ``-1`` placeholder asks JAX to divide the leaf's total size by the
    product of the other axes to recover it, and a zero-sized leaf makes that
    product zero too, raising ``ZeroDivisionError`` before jax-gcm's own
    xarray conversion ever gets a chance to skip the field. Multiplying the
    two known axis lengths directly needs no division and gives the identical
    result whenever ``-1`` would have resolved cleanly (the save axis is
    always exactly 1), so this is a strict generalisation, not a special case
    -- SPEEDY runs, whose diagnostics never happen to include a zero-sized
    array, reshape exactly as before.
    """
    if getattr(leaf, "ndim", 0) < 2:
        return leaf
    return leaf.reshape((leaf.shape[0] * leaf.shape[1], *leaf.shape[2:]))


class JCMComponent:
    """The JCM atmosphere, driven one coupling timestep at a time.

    Satisfies :class:`~jem.base.component.Component`,
    :class:`~jem.base.component.SupportsBind` and
    :class:`~jem.base.component.SupportsXarray`.

    Parameters
    ----------
    model : jcm.model.Model
        A fully configured JCM model. Its ``start_date`` and ``calendar``
        must match the coupler's; :meth:`bind` checks that.
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
            The run's start date; must equal ``model.start_date``.
        calendar : str
            The run's calendar; must equal ``model.calendar``.

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
        if str(calendar) != str(self.model.calendar):
            raise ValueError(
                f"Calendar mismatch: the coupler runs {calendar!r} but"
                f" {self.name!r} was built with"
                f" {self.model.calendar!r}. Rebuild the model with"
                " calendar=<coupler calendar>."
            )
        if start_date != self.model.start_date:
            raise ValueError(
                f"Start-date mismatch: the coupler starts at {start_date!r}"
                f" but {self.name!r} was built with"
                f" {self.model.start_date!r}. Rebuild the model with"
                " start_date=<coupler start date>."
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

        Returns
        -------
        dict
            ``{"state": dycore state, "physics": cross-step physics carry,
            "derived": JCMDerived, "forcing": ForcingData}``.

        """
        dycore_state, physics_carry = self.model.bootstrap_state()
        return {
            "state": dycore_state,
            "physics": physics_carry,
            "derived": JCMDerived.zeros(
                _diagnostics_template(self.model), self.nodal_shape),
            "forcing": _collapse_exchanged_forcing(
                self.forcing,
                self._exchanged_forcing,
                self.model.start_date,
                self.model.calendar,
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

        state, physics_carry, predictions = self.model.run_from_state_with_carry(
            initial_state=carry["state"],
            forcing=carry["forcing"],
            save_interval=self._coupling_days,
            total_time=self._coupling_days,
            output_averages=True,
            initial_physics_state=carry["physics"],
        )
        # One coupling step is exactly one save interval, so the saved
        # trajectory has a length-1 leading axis; the derived fields are
        # per-step maps, not trajectories.
        diagnostics = jax.tree.map(lambda leaf: leaf[0], predictions.physics)
        exchange = _surface_exchange_on_nodal_grid(diagnostics, self.nodal_shape)
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
                "state": state,
                "physics": physics_carry,
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
        averaged record with the **end** of the interval it covers
        (``datetime64[ns]``, absolute, from the model's own ``start_date``),
        and JEM does not relabel it, because a coupled dataset in which the
        atmosphere's time axis disagrees with the atmosphere's own output
        files would be worse than one where two components label the same
        interval differently. The labels the other components carry come from
        ``TimeAxis.datetimes``, which reproduces JCM's *output* arithmetic
        rather than calling ``Model.date_from_sim_time`` -- public since
        jax-gcm#824, but a different conversion, for the reason set out on
        :class:`jem.base.component.TimeAxis`. Publishing the labelling
        itself is jax-gcm#862.

        """
        collapsed = jax.tree.map(_collapse_save_axis, diagnostics)
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
