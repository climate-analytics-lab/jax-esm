"""The JCM atmosphere as a JEM :class:`~jem.base.component.Component`.

:class:`JCMComponent` wraps a :class:`jcm.model.Model` rather than
monkey-patching methods onto it, so the atmosphere JEM drives is the same
object the user configured and nothing in JCM has to know JEM exists.

Two things this wrapper exists to get right:

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

**Nothing integrates at initialization.** :meth:`initialize` builds the
initial pytrees with :meth:`jcm.model.Model.bootstrap_state` and a
structural template of the diagnostics dict. The previous adapter ran a
whole coupling interval just to discover the shape of the diagnostics it
would later store, which both cost a full model step per run and started
the atmosphere one interval ahead of the coupler's clock.

JCM private attributes are still read in three places, each isolated in one
helper below and tagged with the jax-gcm issue that will remove it.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import tree_math
import xarray as xr
from jcm.forcing import ForcingData, default_forcing
from jcm.model import Model
from jcm.predictions import ModelPredictions

from jem.base.component import Carry, CouplingTime, Diagnostics, TimeAxis
from jem.components.jcm import exchange_fields

logger = logging.getLogger(__name__)

# Floor on how far the atmosphere's own clock may drift from the coupler's
# before it is reported. One second is below JCM's shortest plausible
# timestep, so any real disagreement (a checkpoint from another run, a carry
# threaded into the wrong component) is orders of magnitude larger than this.
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

SECONDS_PER_DAY = 86400.0


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


def _bootstrapped_dycore_state(model: Model) -> Any:
    """Return the dycore state ``bootstrap_state`` just built.

    TODO(jax-gcm#755): ``bootstrap_state`` is public but only publishes its
    result through the private ``_final_dycore_state`` attribute; a public
    ``Model.initial_state()`` would remove this read.
    """
    return model._final_dycore_state


def _bootstrapped_physics_carry(model: Model) -> Any:
    """Return the initial cross-step physics carry ``bootstrap_state`` built.

    TODO(jax-gcm#755): same gap as :func:`_bootstrapped_dycore_state`; a
    public ``Model.physics_carry`` would remove this read.
    """
    return model._final_physics_state


def _with_model_context(predictions: ModelPredictions,
                        model: Model) -> ModelPredictions:
    """Re-attach the coords/physics/dycore a pytree round-trip dropped.

    ``ModelPredictions`` is registered as a pytree whose only children are
    the raw prediction arrays, so everything JEM's ``lax.scan`` hands back
    has ``coords``, ``physics`` and ``dycore`` set to ``None`` and cannot
    serialize itself.

    TODO(jax-gcm#756): a public
    ``ModelPredictions.with_context(coords, physics, dycore)`` would remove
    this read of the private ``_predictions`` payload.
    """
    return ModelPredictions(
        predictions._predictions, model.coords, model.physics,
        dycore=model.dycore,
    )


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


def _collapse_save_axis(leaf: jnp.ndarray) -> jnp.ndarray:
    """Merge a stacked leaf's ``(coupling step, save)`` axes into one time axis.

    Each coupled step runs JCM for exactly one save interval, so every leaf
    the coupler stacks is ``(iterations, 1, ...)``; JCM's own serialization
    wants a single leading time axis.
    """
    if getattr(leaf, "ndim", 0) < 2:
        return leaf
    return leaf.reshape((-1, *leaf.shape[2:]))


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

    Attributes
    ----------
    name : str
        ``"atm"``.

    """

    name = "atm"

    def __init__(self, model: Model, *, forcing: ForcingData | None = None) -> None:
        """Wrap ``model``; see the class docstring for the parameters."""
        self.model = model
        self.forcing = (forcing if forcing is not None
                        else default_forcing(model.coords.horizontal))
        # Horizontal nodal shape (ix, il); the leading axis of
        # ``coords.nodal_shape`` is the vertical.
        self.nodal_shape = tuple(model.coords.nodal_shape[1:])
        # Coupling interval in days, as a Python float. Static on purpose:
        # ``run_from_state_with_carry`` takes ``save_interval`` and
        # ``total_time`` as static arguments of a jit, so they cannot be
        # traced values. ``None`` until bind() has run.
        self._coupling_days: float | None = None

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
            differently from every other component.

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
        self._coupling_days = float(
            coupling_timestep / jdt.to_timedelta(1, "day"))

    def initialize(self) -> Carry:
        """Build the initial carry without integrating the model.

        Returns
        -------
        dict
            ``{"state": dycore state, "physics": cross-step physics carry,
            "derived": JCMDerived, "forcing": ForcingData}``.

        """
        self.model.bootstrap_state()
        return {
            "state": _bootstrapped_dycore_state(self.model),
            "physics": _bootstrapped_physics_carry(self.model),
            "derived": JCMDerived.zeros(
                self.nodal_shape, _diagnostics_template(self.model)),
            "forcing": self.forcing,
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
        exchange = exchange_fields.detect(diagnostics)(diagnostics)
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
        The ``time`` coordinate is JCM's, not the coupler's: JCM labels each
        averaged record with the **end** of the interval it covers
        (``datetime64[ns]``, absolute, from the model's own ``start_date``),
        and JEM does not relabel it, because a coupled dataset in which the
        atmosphere's time axis disagrees with the atmosphere's own output
        files would be worse than one where two components label the same
        interval differently. JEM cannot reproduce JCM's calendar
        arithmetic itself while ``Model._date_from_sim_time`` is private —
        TODO(jax-gcm#758).

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
        return dataset

    def _report_clock_drift(self, state: Any, time: CouplingTime) -> None:
        """Log at ERROR if the dycore state's own clock has left the coupler's.

        The dycore state carries its own ``sim_time`` in seconds and JCM
        advances it independently of the coupler's step counter, so the two
        can only disagree if the carry did not come from this run: a
        checkpoint restored into a coupler with a different start date, or a
        carry threaded into the wrong component.

        The tolerance grows with simulation time
        (:func:`clock_tolerance_seconds`), because both counters are float32
        by default and would otherwise disagree by float32 rounding alone
        after a few decades.

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
