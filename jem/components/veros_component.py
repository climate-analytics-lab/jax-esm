"""Adapter for the Veros ocean GCM (optional dependency).

:class:`VerosComponent` wraps a ``VerosSetup`` so the JEM ``Coupler`` can
drive it, exposing the :class:`~jem.base.component.Component` contract plus
the optional bind / xarray / checkpoint capabilities. The wrapper holds the
grid metadata and the coupling arithmetic; the evolving ``VerosState`` lives
in the carry, like every other component's state.

One monkey-patch survives, and has to: Veros calls its setup's
``set_forcing`` from inside every ``step``, so a coupled run must replace it
with a no-op or Veros overwrites the surface forcing the coupler just
handed it. That is done once, in the constructor, and said out loud there.
"""

import importlib
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import tree_math
import xarray as xr
from veros import runtime_settings

from jem.base.component import (
    Carry,
    CouplingTime,
    Diagnostics,
    TimeAxis,
    forcing_variable,
    role_attrs,
)
from jem.checkpoint import CARRY_FILENAME
from jem.checkpoint import load as load_pytree
from jem.checkpoint import save as save_pytree
from jem.components.clock import clock_tolerance_seconds

logger = logging.getLogger(__name__)

# Veros pads every horizontal field with two halo cells on each side; the
# interior a coupled component exchanges is the [2:-2, 2:-2] slice. Hard
# coded in Veros itself, so it is hard coded here too rather than guessed
# from array shapes.
GHOST_CELLS = 2

# Reference values used to turn the surface fluxes the coupler supplies into
# the temperature/salinity surface forcings Veros integrates. Taken from
# Veros' own ``setups/global_1deg``.
SEAWATER_HEAT_CAPACITY = 3991.86795711963  # J kg-1 K-1
REFERENCE_SALINITY = 35.0  # PSU

# ``sqrt`` has an infinite derivative at 0, so AD through ``sqrt(x)`` blows
# up to NaN as x -> 0 even though the primal value stays finite. The TKE
# surface forcing floors the *squared* stress magnitude -- sqrt's argument --
# at this value squared, which bounds sqrt and its derivative and caps the
# resulting magnitude from below.
MIN_STRESS_MAGNITUDE = 1e-3  # N m-2

#: Name of the HDF5 restart file :meth:`VerosComponent.save_carry` writes
#: inside its checkpoint directory. Veros owns the format; JEM only chooses
#: where it goes, and fixes the name so that the loader finds it.
VEROS_RESTART_FILENAME = "veros.restart.h5"


def configure_veros_runtime() -> None:
    """Point Veros at the JAX backend before any of its operators are bound.

    Veros binds its array backend when its core modules are first imported,
    reading the process-global ``runtime_settings`` at that moment and then
    locking them: any later assignment raises ``RuntimeError``. The settings
    therefore have to be applied before *anything* in the process imports
    ``veros.core`` -- including a user's own ``VerosSetup`` module. That is
    why this module calls it at import time: importing
    ``jem.components.veros_component`` (or ``jem.components.Veros``) is the
    documented way to make Veros JAX-backed, and it must come before the
    setup module is imported.

    Calling it again after the core modules are imported is safe: the
    locked settings are inspected instead of assigned, and an error is
    raised only if they were bound to a non-JAX backend.

    Raises
    ------
    RuntimeError
        If Veros already bound its operators to a non-JAX backend. The
        operators cannot be re-pointed, so the only fix is to import this
        module before the Veros setup module.

    """
    try:
        runtime_settings.backend = "jax"
        runtime_settings.force_overwrite = True
        runtime_settings.linear_solver = "scipy_jax"
    except RuntimeError as exc:
        # Settings are locked because veros.core was already imported. If
        # they were locked with the JAX backend (this module was imported
        # first, as documented) there is nothing left to do.
        if runtime_settings.backend != "jax":
            raise RuntimeError(
                "Veros bound its operators to the"
                f" {runtime_settings.backend!r} backend before jem could select"
                " JAX, and the settings are now locked. Import"
                " jem.components.veros_component (or call"
                " configure_veros_runtime()) before importing any Veros setup"
                " module or veros.core."
            ) from exc


@contextmanager
def _veros_runtime_setting(name: str, value: object) -> Iterator[None]:
    """Set one locked Veros runtime setting for the duration of a block.

    ``runtime_settings`` locks itself once ``veros.core`` is imported, and
    reading a restart needs ``force_overwrite`` off while the rest of a
    coupled run needs it on. Veros itself offers no supported way to flip a
    locked setting, so the lock flag is cleared and restored around each
    assignment.

    It is a context manager, and the restore is in a ``finally``, because the
    settings are **process-global**: a failed ``read_restart`` -- a missing or
    mismatched HDF5 file -- would otherwise leave the whole process with
    ``force_overwrite`` off, and the next thing that tried to write an output
    or a restart would fail for a reason with no connection to the one that
    actually went wrong. The previous value is put back rather than a
    hard-coded one, so this makes no assumption about who set it.
    """
    previous = getattr(runtime_settings, name)
    was_locked = getattr(runtime_settings, "__locked__", False)

    def assign(to: object) -> None:
        object.__setattr__(runtime_settings, "__locked__", False)
        try:
            setattr(runtime_settings, name, to)
        finally:
            object.__setattr__(runtime_settings, "__locked__", was_locked)

    assign(value)
    try:
        yield
    finally:
        assign(previous)


# Deliberate import-time side effect: see configure_veros_runtime(). This is the
# only way to guarantee the setting precedes the operator import that binds it.
configure_veros_runtime()


@tree_math.struct
class VerosForcing:
    """Surface forcing an exchanger writes for the ocean to integrate."""

    heat_flux: jnp.ndarray
    freshwater_flux: jnp.ndarray
    surface_taux: jnp.ndarray
    surface_tauy: jnp.ndarray
    surface_air_temperature: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        heat_flux=None,
        freshwater_flux=None,
        surface_taux=None,
        surface_tauy=None,
        surface_air_temperature=None,
    ):
        """Zero-filled forcing on a ``shape`` horizontal grid."""
        return cls(
            heat_flux if heat_flux is not None else jnp.zeros(shape),
            freshwater_flux if freshwater_flux is not None else jnp.zeros(shape),
            surface_taux if surface_taux is not None else jnp.zeros(shape),
            surface_tauy if surface_tauy is not None else jnp.zeros(shape),
            surface_air_temperature if surface_air_temperature is not None else jnp.zeros(shape),
        )


@tree_math.struct
class VerosDerived:
    """What the ocean publishes for the other components to read."""

    sea_surface_temperature: jnp.ndarray
    sea_surface_u: jnp.ndarray
    sea_surface_v: jnp.ndarray

    @classmethod
    def zeros(cls, shape, sea_surface_temperature=None, sea_surface_u=None, sea_surface_v=None):
        """Zero-filled derived fields on a ``shape`` horizontal grid."""
        return cls(
            sea_surface_temperature if sea_surface_temperature is not None else jnp.zeros(shape) + 273.15,
            sea_surface_u if sea_surface_u is not None else jnp.zeros(shape),
            sea_surface_v if sea_surface_v is not None else jnp.zeros(shape),
        )


class VerosComponent:
    """The Veros ocean GCM, driven one coupling timestep at a time.

    Satisfies :class:`~jem.base.component.Component`,
    :class:`~jem.base.component.SupportsBind`,
    :class:`~jem.base.component.SupportsXarray` and
    :class:`~jem.base.component.SupportsCheckpoint`.

    Parameters
    ----------
    model : veros.VerosSetup
        A Veros setup on which ``setup()`` has already been called, so
        ``model.state`` holds the allocated grid and initial conditions
        this wrapper reads its geometry from.

    Attributes
    ----------
    name : str
        ``"ocn"``.
    mask_T, mask_U : jax.Array
        Land-sea masks on the T and u grids, halo cells removed.
    mask_surface_Z : jax.Array
        Surface land-sea mask on the zeta (corner) points -- where the
        barotropic streamfunction lives -- halo cells removed.
    longitude, latitude : jax.Array
        T-grid cell centres, halo cells removed.
    dlatitude : jax.Array
        T-grid cell heights, halo cells removed: Veros' ``dyt``, a true
        meridional **distance in metres** whichever coordinates are in use
        (``calc_grid`` converts it with ``degtom`` when they are degrees).
    dlongitude : jax.Array
        T-grid cell widths, halo cells removed: Veros' ``dxt``. In degree
        coordinates this is the *nominal*, equatorial-equivalent zonal
        spacing, **not** a distance: Veros keeps the spherical metric
        separately in ``cost``/``cosu`` (``area_t = cost * dyt * dxt``), so a
        true zonal distance is ``dlongitude * cos(latitude)``. In Cartesian
        coordinates the two are the same thing and it is metres, like
        ``dlatitude``.
    enable_streamfunction : bool
        Whether the wrapped setup solves the external mode for a barotropic
        streamfunction; it decides where the ``psi`` output comes from (see
        :meth:`_barotropic_streamfunction`) and whether there is an ``ssh``
        output at all -- Veros carries a sea surface height only under the
        linear free surface.

    """

    name = "ocn"

    def __init__(self, model: Any) -> None:
        """Wrap ``model``; see the class docstring for the parameters."""
        # The settings were applied when this module was imported; re-checking
        # here catches the case where Veros operators were bound to another
        # backend before that import happened (a setup module imported first).
        configure_veros_runtime()

        self.model = model
        settings = model.state.settings
        variables = model.state.variables

        interior = slice(GHOST_CELLS, -GHOST_CELLS)
        self.horizontal_shape = (
            model.state.dimensions["xt"], model.state.dimensions["yt"],
        )
        self.mask_T = jnp.array(variables.maskT)[interior, interior]
        # `maskU` as well as `maskT`: the barotropic streamfunction is
        # diagnosed from the depth-integrated *zonal* transport, which lives
        # on the u grid (see `_barotropic_streamfunction`).
        self.mask_U = jnp.array(variables.maskU)[interior, interior]
        # `psi` is published on the zeta points, and over land it is carried
        # through the integration rather than computed, so the mask that
        # blanks it has to be published with it.
        self.mask_surface_Z = jnp.array(variables.maskZ)[interior, interior, -1]
        self.dzt = jnp.array(variables.dzt)
        self.longitude = jnp.array(variables.xt)[interior]
        self.latitude = jnp.array(variables.yt)[interior]
        self.dlongitude = jnp.array(variables.dxt)[interior]
        self.dlatitude = jnp.array(variables.dyt)[interior]
        # A Cartesian setup (``coord_degree=False``) gives its grid
        # spacings in metres -- the dynamics divides by ``dxt``/``dyt`` as
        # lengths and keeps the spherical metric separately in
        # ``cost``/``cosu``, which it sets to 1 in that case -- so the
        # coordinates those spacings accumulate into are metres too.
        self.longitude_units = "degrees_east" if settings.coord_degree else "m"
        self.latitude_units = "degrees_north" if settings.coord_degree else "m"

        # Which external mode the setup solves is fixed for the whole run, so
        # it is read once here as a Python bool and the `psi` branch in
        # `step` is taken at trace time rather than on a traced value.
        self.enable_streamfunction = bool(settings.enable_streamfunction)

        # Number of Veros tracer timesteps per coupling step; set by bind().
        self._steps_per_coupling_step: int | None = None
        # Veros' own `variables.time` at the moment the coupler adopted this
        # ocean, i.e. the reading that corresponds to the coupler's
        # `start_date`; set by bind(). See `_report_clock_drift`.
        self._veros_time_zero: float | None = None

        # Static-configuration checks. Reported here rather than inside
        # ``step`` because they read Python-level settings that cannot change
        # during a run, and a message emitted from inside the traced step
        # function fires once at trace time, not once per step -- which reads
        # as a per-step warning and is not one.
        if not settings.enable_tempsalt_sources:
            logger.warning(
                "%s: settings.enable_tempsalt_sources is False, so Veros'"
                " `temp_source` term is inactive. The coupled surface heat"
                " flux is applied through `forc_temp_surface` and is"
                " unaffected, but a setup relying on `temp_source` will not"
                " see it.", self.name,
            )
        if settings.enable_tke:
            logger.info(
                "%s: settings.enable_tke is True; the coupled wind stress"
                " drives `forc_tke_surface`.", self.name,
            )
        if not self.enable_streamfunction:
            logger.info(
                "%s: settings.enable_streamfunction is False, so Veros'"
                " `variables.psi` holds the surface pressure rather than a"
                " streamfunction. The `psi` output variable is diagnosed"
                " from the depth-integrated zonal transport instead"
                " (see VerosComponent._barotropic_streamfunction), and"
                " that surface pressure over `grav` is published as `ssh`.",
                self.name,
            )

        # Veros calls its setup's ``set_forcing`` from inside every ``step``.
        # In a coupled run the forcing comes from the exchangers, so the
        # setup's own routine has to be disabled or it overwrites what the
        # coupler just applied. This is the one place JEM still mutates a
        # component object, and there is no way around it short of a Veros
        # change: ``step`` looks the method up on the setup instance.
        logger.info(
            "%s: replacing the VerosSetup's set_forcing() with a no-op; in a"
            " coupled run the surface forcing comes from the coupler.",
            self.name,
        )
        model.set_forcing = lambda state: None

    @classmethod
    def from_setup(cls, setup: str, **setup_kwargs: Any) -> "VerosComponent":
        """Build the component from an importable Veros setup.

        The constructor takes an already-built Veros model, which is a live
        Python object no configuration file can name. This is the door a
        config comes through (``jem/config/ocean/veros.yaml``): it imports
        ``setup``, builds it with ``setup_kwargs``, runs the setup's own
        ``setup()`` -- which is what allocates the grid and the initial
        conditions this wrapper reads its geometry from -- and wraps it.

        Parameters
        ----------
        setup : str
            Importable dotted path of either a ``VerosSetup`` subclass or a
            factory returning one. It is never a file path: the module has to
            be importable like any other, so a setup that lives in an example
            directory needs that directory on ``sys.path``. Both spellings
            are accepted because the setups shipped with the examples are
            factories -- that is how a case is parameterised by its grid file
            and timesteps -- and which one a path names is only discoverable
            by calling it.
        **setup_kwargs
            Passed to the class or factory.

        Returns
        -------
        VerosComponent
            Wrapping a setup whose ``setup()`` has been called.

        Raises
        ------
        ImportError
            If ``setup`` is not a dotted path, its module cannot be imported,
            or the module has no such attribute.

        """
        module_path, _, attribute = setup.rpartition(".")
        if not module_path:
            raise ImportError(
                f"{setup!r} is not an importable dotted path to a Veros setup"
                " (expected something like"
                " 'my_package.my_case.MySetup')."
            )
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import {module_path!r} for the Veros setup {setup!r}."
                " A setup that lives outside an installed package (the ones"
                " under examples/ do) needs its directory on PYTHONPATH."
            ) from exc
        try:
            factory = getattr(module, attribute)
        except AttributeError as exc:
            raise ImportError(
                f"{module_path!r} has no attribute {attribute!r}"
                f" (from the Veros setup {setup!r})."
            ) from exc

        model = factory(**setup_kwargs)
        if isinstance(model, type):
            # ``factory`` was a factory returning the setup *class*, not the
            # class itself; the shipped example cases are written that way so
            # that the class can close over the case's grid and settings.
            model = model()
        model.setup()
        return cls(model)

    def bind(
        self,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str,
    ) -> None:
        """Adopt the coupler's clock and work out the internal step count.

        Veros has no calendar: ``variables.time`` counts seconds from
        whatever the setup treats as its own start, so there is no start date
        to check against the coupler's the way the JCM wrapper does. What
        makes the two clocks comparable at all is this: the Veros time the
        setup holds **when the coupler adopts it** is the reading that
        corresponds to the coupler's ``start_date``, and it is recorded here
        as the zero point of :meth:`_report_clock_drift`. A setup that was
        already integrated before it was wrapped therefore starts the coupled
        run at its own current time rather than being declared wrong -- JEM
        cannot know which absolute date that state belongs to -- while any
        *later* disagreement between the two clocks is caught.

        The zero point is recorded once, on the first bind, so re-binding an
        ocean that has since been stepped cannot silently move it.

        Parameters
        ----------
        coupling_timestep : jax_datetime.Timedelta
            The coupled model's timestep. It must be an exact multiple of
            Veros' tracer timestep ``dt_tracer``.
        start_date : jax_datetime.Datetime
            The run's start date. Recorded for the output metadata, and (see
            above) taken to be the date Veros' current ``variables.time``
            stands for.
        calendar : str
            The run's calendar. Recorded for the output metadata; Veros has
            no calendar of its own to reconcile with.

        Raises
        ------
        ValueError
            If the coupling timestep is not a whole multiple of
            ``dt_tracer``, or the component is already bound to a
            different coupling timestep (the same one again is a no-op).

        """
        self.start_date = start_date
        self.calendar = calendar

        model_timestep = jdt.to_timedelta(
            int(self.model.state.settings.dt_tracer), "second")
        n_steps = float(coupling_timestep / model_timestep)
        if n_steps != int(n_steps) or n_steps < 1:
            raise ValueError(
                f"Coupling timestep {coupling_timestep!r} is not a whole"
                f" multiple of {self.name!r}'s tracer timestep"
                f" {model_timestep!r}."
            )
        steps_per_coupling_step = int(n_steps)
        if (
            self._steps_per_coupling_step is not None
            and steps_per_coupling_step != self._steps_per_coupling_step
        ):
            # One instance belongs to one coupled model: `step` runs this
            # many tracer steps, so a second coupler with another timestep
            # would silently advance the ocean by the wrong interval in the
            # first coupler's runs.
            raise ValueError(
                f"{type(self).__name__} {self.name!r} is already bound to a "
                f"coupling timestep of {self._steps_per_coupling_step:d} tracer "
                f"steps and cannot also be bound to {steps_per_coupling_step:d}. "
                "Build a separate instance per coupled model."
            )
        self._steps_per_coupling_step = steps_per_coupling_step
        if self._veros_time_zero is None:
            self._veros_time_zero = float(self.model.state.variables.time)

    def internal_steps_per_call(self) -> int:
        """Report how many of Veros' own tracer steps happen inside one ``step()`` call.

        Implements :class:`~jem.base.component.SupportsInternalStepping`
        (2026-09 review, round 3 follow-up, finding 7, extended to Veros).
        Veros' own iteration counter, ``state.variables.itt``, is declared
        ``dtype="int32"`` in ``veros.variables`` and is a genuine traced
        pytree leaf carried through :meth:`step`'s ``jax.lax.fori_loop`` (not
        a Python int or a float outside the carry) -- confirmed directly
        (2026-09 review): ``model.state.variables.itt.dtype`` is
        ``int32``, and ``veros.veros.py`` increments it with ``vs.itt =
        vs.itt + 1`` once per internal Veros step, ``self
        ._steps_per_coupling_step`` times per call to this method's caller.
        That is exactly the same shape of raw-counter risk as JCM's
        ``RunState.step`` (see ``JCMComponent.internal_steps_per_call``'s own
        docstring): for a representative double-drake configuration (this
        package's shipped coupled ocean setup, ``dt_tracer=3600`` s) under a
        1 day coupling timestep, ``self._steps_per_coupling_step == 24``, so
        ``itt`` wraps at a coupled step count of about ``2**31 / 24``,
        roughly 245,000 simulated years -- a real, finite limit exactly like
        JCM's, just at a different scale.

        Reporting it here is what lets ``jem.driver._max_element_rate``
        cover this counter (multiplied in for every element clock that
        calls this component, the same as a workflow multiplicity or a
        nested coupler's own substep rate), so
        ``jem.driver.run_chunked``'s up-front int32 check refuses a run
        before ``itt`` could overflow, exactly as it already does for the
        coupler hierarchy's own counters.
        """
        assert self._steps_per_coupling_step is not None
        return self._steps_per_coupling_step

    def _derived_fields(self, state: Any) -> "VerosDerived":
        """Extract the fields ``VerosDerived`` publishes, from any state.

        This is the surface-extraction convention -- the interior slice,
        the ``tau`` time index, the Kelvin offset and the land-column
        substitution -- defined exactly once so :meth:`step` (after
        integrating) and :meth:`initialize` (before any step) cannot drift
        apart on it. ``state.variables.tau`` is well-defined even before
        the model has taken a single step (Veros initializes it to ``1``
        and a setup's ``setup()`` fills all three time levels of
        ``temp``/``u``/``v`` with the same initial condition), so calling
        this on the freshly built ``model.state`` reads the setup's actual
        initial surface fields rather than a placeholder.

        Parameters
        ----------
        state : veros.state.VerosState
            Either the setup's freshly initialized state, or the state
            :meth:`step` has just advanced.

        Returns
        -------
        VerosDerived
            The sea surface temperature, u and v this state implies.

        """
        interior = slice(GHOST_CELLS, -GHOST_CELLS)
        variables = state.variables
        tau = variables.tau

        sea_surface_temperature = (
            variables.temp[interior, interior, -1, tau] + 273.15
        )
        # Intended to replace a land column's fill value with a plausible
        # constant, so downstream components never read an unphysical SST
        # through the mask. It does not fire for either shipped setup: both
        # build their cold start as `... * vs.maskT`, so a land column holds
        # 0 degC and reaches `273.15` here rather than the large negative
        # sentinel this threshold assumes, and land is published as 273.15 K.
        # Left as-is deliberately -- what a land column should publish is a
        # contract decision for the exchange, not a change to make in passing;
        # see jax-esm#127. Whatever is decided, `step` and `initialize` share
        # this helper, so they cannot disagree about it.
        sea_surface_temperature = jnp.where(
            sea_surface_temperature < 100, 288.15, sea_surface_temperature)
        zonal_velocity = variables.u[interior, interior, :, tau]
        sea_surface_u = zonal_velocity[:, :, -1]
        sea_surface_v = variables.v[interior, interior, -1, tau]

        return VerosDerived(  # type: ignore[call-arg]
            sea_surface_temperature, sea_surface_u, sea_surface_v,
        )

    def initialize(self) -> Carry:
        """Build the initial carry without integrating the model.

        ``derived`` is seeded from the Veros setup's own initial condition
        via :meth:`_derived_fields`, not from ``VerosDerived.zeros``'
        placeholder. Every shipped coupling workflow runs the exchangers
        before any component has stepped, so whatever ``derived`` holds
        here is exactly what the atmosphere integrates over for the whole
        first coupling interval; publishing ``VerosDerived.zeros``'s
        uniform 273.15 K here would mean that interval runs over a
        fictitious freezing-point ocean instead of the setup's actual
        (e.g. ~288 K) surface temperature.

        Returns
        -------
        dict
            ``{"state": VerosState, "derived": VerosDerived,
            "forcing": VerosForcing}``.

        """
        return {
            "state": self.model.state,
            "derived": self._derived_fields(self.model.state),
            "forcing": VerosForcing.zeros(self.horizontal_shape),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the ocean by one coupling timestep.

        Applies the surface forcing the exchangers wrote into the carry,
        then runs ``dt_tracer``-sized Veros steps until one coupling
        interval has elapsed.

        Parameters
        ----------
        carry : dict
            The carry :meth:`initialize` produced, as last returned.
        time : jem.base.component.CouplingTime
            The coupler's clock for this step. Veros integrates on its own
            ``variables.time`` counter and reads no date-dependent forcing of
            its own once ``set_forcing`` has been disabled, so the coupler's
            clock does not enter the integration -- but the two are compared
            before it, and a disagreement is reported
            (:meth:`_report_clock_drift`).

        Returns
        -------
        tuple
            The new carry and this step's diagnostics as a dict of
            ``(lon, lat[, depth])`` maps. Which keys it holds is fixed at
            construction, not per step: ``ssh`` is among them only for a run
            that solves the linear free surface, the only regime in which
            Veros carries a sea surface height.

        Raises
        ------
        RuntimeError
            If the component has not been bound to a coupler clock.

        """
        if self._steps_per_coupling_step is None:
            raise RuntimeError(
                f"{type(self).__name__} {self.name!r} has no coupling"
                " timestep: register it with a Coupler (which calls bind())"
                " before stepping it."
            )
        self._report_clock_drift(carry, time)

        # Imported here rather than at module scope: `veros.core.operators`
        # binds its array backend (numpy or jax) at import time from
        # `runtime_settings.backend`, so importing it before
        # `configure_veros_runtime()` has run would silently pin the coupler
        # to numpy operators.
        from veros.core.operators import at, update
        from veros.core.operators import numpy as npx

        interior = slice(GHOST_CELLS, -GHOST_CELLS)
        state = carry["state"]
        forcing = carry["forcing"]
        variables = state.variables
        settings = state.settings

        with variables.unlock():
            variables.surface_taux = update(
                variables.surface_taux, at[interior, interior],
                forcing.surface_taux,
            )
            variables.surface_tauy = update(
                variables.surface_tauy, at[interior, interior],
                forcing.surface_tauy,
            )
            if settings.enable_tke:
                # Follows Veros' own `setups/global_1deg`.
                surface_stress_squared = (
                    (0.5 * (variables.surface_taux[1:-1, 1:-1]
                            + variables.surface_taux[:-2, 1:-1])
                     / settings.rho_0) ** 2
                    + (0.5 * (variables.surface_tauy[1:-1, 1:-1]
                              + variables.surface_tauy[1:-1, :-2])
                       / settings.rho_0) ** 2
                )
                surface_stress_magnitude = npx.sqrt(
                    npx.maximum(surface_stress_squared,
                                MIN_STRESS_MAGNITUDE ** 2)
                )
                variables.forc_tke_surface = update(
                    variables.forc_tke_surface, at[1:-1, 1:-1],
                    surface_stress_magnitude ** 1.5,
                )

            # W m-2 * (kg K J-1) * (m3 kg-1) = K m s-1. The coupler's heat
            # flux is positive upward (out of the ocean), Veros' surface
            # forcing warms the top cell, hence the negation.
            surface_mask = variables.maskT[interior, interior, -1]
            variables.forc_temp_surface = update(
                variables.forc_temp_surface, at[interior, interior],
                -forcing.heat_flux * surface_mask
                / SEAWATER_HEAT_CAPACITY / settings.rho_0,
            )
            # Freshwater flux is positive upward, so a positive flux removes
            # fresh water and must increase salinity.
            variables.forc_salt_surface = update(
                variables.forc_salt_surface, at[interior, interior],
                forcing.freshwater_flux * surface_mask
                / settings.rho_0 * REFERENCE_SALINITY,
            )

        def _sub_step(_, inner_state):
            self.model.step(inner_state)
            return inner_state

        state = jax.lax.fori_loop(
            0, self._steps_per_coupling_step, _sub_step, state)
        # `fori_loop` reconstructs the carry into fresh VerosState/
        # VerosVariables instances (via tree_unflatten), so the `variables`
        # bound above is now stale; rebind it to the evolved state before
        # reading diagnostics from it.
        variables = state.variables
        tau = variables.tau

        derived = self._derived_fields(state)
        sea_surface_temperature = derived.sea_surface_temperature
        sea_surface_u = derived.sea_surface_u
        sea_surface_v = derived.sea_surface_v
        sea_surface_salinity = variables.salt[interior, interior, -1, tau]
        zonal_velocity = variables.u[interior, interior, :, tau]

        # The barotropic streamfunction, from whichever of the two this run
        # actually has: Veros only carries a real one when it solves the
        # external mode for it, and `_barotropic_streamfunction` explains
        # what stands in for it when it does not.
        if self.enable_streamfunction:
            psi = variables.psi[interior, interior, tau]
        else:
            psi = self._barotropic_streamfunction(zonal_velocity)

        diagnostics = {
            "sea_surface_temperature": sea_surface_temperature,
            "sea_surface_salinity": sea_surface_salinity,
            "sea_surface_u": sea_surface_u,
            "sea_surface_v": sea_surface_v,
            "temp": variables.temp[interior, interior, :, tau],
            "salt": variables.salt[interior, interior, :, tau],
            "u": zonal_velocity,
            "v": variables.v[interior, interior, :, tau],
            "psi": psi,
            "surface_air_temperature": forcing.surface_air_temperature,
            "surface_taux": forcing.surface_taux,
            "surface_tauy": forcing.surface_tauy,
            "heat_flux": forcing.heat_flux,
            "freshwater_flux": forcing.freshwater_flux,
        }
        # Under the linear free surface Veros solves for a surface pressure,
        # and the sea surface height that goes with it is that pressure over
        # `grav` -- the relation Veros' own `barotropic_velocity_update`
        # applies when it sets `variables.ssh`. In streamfunction mode there
        # is no sea surface height at all (Veros deactivates the variable),
        # so the key exists only in the regime that has one. The branch is on
        # the Python bool read at construction, so a component's diagnostics
        # keys are fixed for the whole run -- which is what `jax.eval_shape`
        # of the step and the coupler's stacking of per-call diagnostics rely
        # on.
        #
        # The relation is applied here rather than `variables.ssh` being read
        # back, because Veros writes that field *before* it permutes its time
        # indices at the end of the step: after a step `variables.ssh` is the
        # surface pressure of the time level that has just become `taum1`,
        # one Veros timestep behind the `psi`, `u`, `v` and tracers published
        # in the same record (in the acc_basic free-surface case, a ~27%
        # difference while the free surface spins up). Reading `psi` at `tau`
        # like every other field here keeps one output record internally
        # consistent.
        if not self.enable_streamfunction:
            diagnostics["ssh"] = (
                variables.psi[interior, interior, tau] / state.settings.grav
            )

        return (
            {
                "state": state,
                "derived": derived,
                "forcing": forcing,
            },
            diagnostics,
        )

    def _barotropic_streamfunction(self, zonal_velocity: jnp.ndarray) -> jnp.ndarray:
        """Diagnose the barotropic streamfunction from the zonal transport.

        Veros carries a barotropic streamfunction only when the setup solves
        the external mode for one (``settings.enable_streamfunction``). Under
        the linear free surface -- what every Veros setup shipped with JEM
        chooses -- the *same* array ``variables.psi`` holds the surface
        pressure instead: a different quantity, in m^2 s^-2, on the T grid
        rather than the corner (zeta) points. Publishing it as ``psi`` would
        therefore be wrong rather than merely approximate, and this diagnosis
        stands in for it.

        The discrete relation it inverts is the one Veros uses **when it
        does solve for a streamfunction**, in
        ``veros.core.external.solve_stream.barotropic_velocity_update``: that
        routine strips the vertical mean from the baroclinic velocity and
        then adds ``-maskU (psi[i, j] - psi[i, j-1]) / dyt[j] * hur`` to every
        level, with ``hur = 1 / sum_k dzt maskU`` from
        ``veros.core.numerics.calc_topo_kernel``. The depth-integrated zonal
        transport

            U[i, j] = sum_k u[i, j, k] dzt[k] maskU[i, j, k]

        is then, exactly,

            U[i, j] = -(psi[i, j] - psi[i, j-1]) / dyt[j].

        A free-surface run never reaches that routine -- it solves for a
        surface pressure and the barotropic mode enters the momentum
        equation as a pressure gradient instead
        (``veros.core.external.solve_pressure``) -- so what carries over is
        the *definition*, not that run's own arithmetic: the same discrete
        relation, applied to the transports the free-surface run produced.
        That the definition is the right one, with the sign and the metric
        Veros uses, is what the streamfunction-mode cross-check in the tests
        pins down, by making this diagnosis reproduce Veros' own psi.

        No ``cos`` metric factor enters: the difference is meridional, and
        ``dyt`` is already a distance in metres (``calc_grid`` converts the
        spacings with ``degtom`` when ``coord_degree``). Inverting the
        recurrence northwards from a boundary where psi vanishes gives the
        cumulative sum this method computes,

            psi[i, j] = -sum_{j' <= j} U[i, j'] dyt[j'],

        with psi = 0 on the boundary row immediately south of the first
        emitted one.

        Two caveats, both recorded in the output's ``comment`` attribute:

        - Under a free surface the barotropic flow is not exactly
          non-divergent, so this is the standard "meridionally integrated
          zonal transport" diagnostic rather than an exact streamfunction.
        - A streamfunction is defined only up to a constant. Veros' solver
          fixes that constant by holding its first island at zero; this
          diagnosis fixes it at the southern boundary. In a domain whose
          southern and northern boundaries belong to one land mass the two
          agree outright; where a zonal channel carries a net throughflow
          (an ACC) they differ by that transport, a constant, while the
          gradients -- the transports the field is read for -- agree.

        Parameters
        ----------
        zonal_velocity : jax.Array
            ``u`` at the current time level on the exchanged interior grid,
            shaped ``(lon, lat, depth)``.

        Returns
        -------
        jax.Array
            ``(lon, lat)`` streamfunction in m^3 s^-1, on the zeta points
            that ``u``'s meridional differences sit between.

        """
        # Depth integral over the trailing axis, then the meridional
        # integral over the latitude axis that leaves: shape-static, no
        # branching, so it traces the same way inside the coupled scan.
        transport = jnp.sum(zonal_velocity * self.mask_U * self.dzt, axis=-1)
        return -jnp.cumsum(transport * self.dlatitude, axis=-1)

    def _report_clock_drift(self, carry: Carry, time: CouplingTime) -> None:
        """Log at ERROR if the ocean's own clock has left the coupler's.

        Veros advances ``variables.time`` by ``dt_tracer`` per internal step,
        independently of the coupler's step counter, so the two can only
        disagree if the carry did not come from this run: a Veros restart
        state paired with a ``CoupledCarry.step`` from a different point in
        the run, a setup integrated behind the coupler's back, or a carry
        threaded into the wrong component. The coupler and every other
        component would go on dating this ocean's fields by ``time.sim_time``,
        silently assigning them to the wrong simulated date.

        Veros' counter is not seconds since the coupler's ``start_date`` but
        seconds since the setup's own start, so it is compared in the
        coupler's frame: ``variables.time`` minus the reading :meth:`bind`
        recorded when the coupler adopted this ocean. It is a no-op before
        that has happened, which only a component ``step`` has already
        refused can be.

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
        zero = self._veros_time_zero
        if zero is None:
            return
        model_seconds = carry["state"].variables.time - zero
        name = self.name

        def _report(model_seconds, coupler_seconds) -> None:
            drift = float(model_seconds) - float(coupler_seconds)
            # The tolerance is set by the COUPLER's time: it is the one that
            # is right by construction, so a model clock that is wildly wrong
            # cannot widen the window that would catch it.
            if abs(drift) > clock_tolerance_seconds(coupler_seconds):
                logger.error(
                    "%s: model clock is %.6g s from the coupler's"
                    " (model %.6g s, coupler %.6g s, both counted from the"
                    " coupler's start date). The ocean's state will be dated"
                    " differently from the rest of the coupled model.",
                    name, drift, float(model_seconds), float(coupler_seconds),
                )

        jax.debug.callback(_report, model_seconds, time.sim_time)

    def to_xarray(self, diagnostics: Diagnostics, time: TimeAxis) -> xr.Dataset:
        """Label the stacked per-step diagnostics as an ``xarray.Dataset``.

        Parameters
        ----------
        diagnostics : dict
            The per-step diagnostics stacked by the coupler, so every field
            carries a leading time axis of length ``iterations``.
        time : jem.base.component.TimeAxis
            The coupler's time axis, used to check the record count.

        Returns
        -------
        xarray.Dataset
            The ocean state and the forcing it was driven with. The fields the
            ocean was *given* -- everything that came out of ``carry["forcing"]``
            -- carry the ``forcing_`` prefix
            :func:`~jem.base.component.forcing_variable` applies, exactly as the
            slab models' output does, so merging this dataset with the
            atmosphere's does not collide on a name two components both hold.
            Each variable that came out of the carry also carries the
            ``jem_role`` attribute (:func:`~jem.base.component.role_attrs`),
            which says the same thing without a name to parse; the grid
            fields (the ``mask_*`` masks and ``dzt``) carry none, because
            they are configuration rather than carry. The masks of all three
            staggerings the output uses are published, because a reader
            cannot rebuild them: ``mask_T`` for the tracers, ``mask_U`` for
            ``u`` and for the depth integral behind ``psi``, and
            ``mask_surface_Z`` for the zeta points ``psi`` itself sits on,
            where its values over land are an artefact of the integration.
            The ``time`` coordinate is the absolute ``datetime64[ms]`` axis
            :meth:`~jem.base.component.TimeAxis.datetimes` builds from ``time``,
            the same one every other component labels its output with, so
            ``xr.merge`` joins the records instead of unioning two axes -- or,
            as before this coordinate was written at all, leaving the ocean's
            ``time`` as a bare 0..n-1 index that means nothing.
            ``ssh`` is present only for a run that solves the linear free
            surface, because that is the only regime in which Veros carries
            a sea surface height; ``step`` emits it on the same static
            branch, so the two always agree.

        """
        n_records = int(jnp.shape(diagnostics["sea_surface_temperature"])[0])
        if len(time) != n_records:
            raise ValueError(
                f"{self.name!r} produced {n_records} output records but the"
                f" coupler's time axis has {len(time)}; the diagnostics"
                " passed here are not the ones this run produced."
            )

        dataset = xr.Dataset(
            data_vars={
                "temp": (["time", "lon", "lat", "depth"], diagnostics["temp"]),
                "salt": (["time", "lon", "lat", "depth"], diagnostics["salt"]),
                "u": (["time", "lon", "lat", "depth"], diagnostics["u"]),
                "v": (["time", "lon", "lat", "depth"], diagnostics["v"]),
                "sea_surface_temperature": (["time", "lon", "lat"], diagnostics["sea_surface_temperature"]),
                "sea_surface_u": (["time", "lon", "lat"], diagnostics["sea_surface_u"]),
                "sea_surface_v": (["time", "lon", "lat"], diagnostics["sea_surface_v"]),
                "sea_surface_salinity": (["time", "lon", "lat"], diagnostics["sea_surface_salinity"]),
                "psi": (["time", "lon", "lat"], diagnostics["psi"]),
                forcing_variable("surface_air_temperature"): (
                    ["time", "lon", "lat"], diagnostics["surface_air_temperature"]),
                forcing_variable("surface_taux"): (
                    ["time", "lon", "lat"], diagnostics["surface_taux"]),
                forcing_variable("surface_tauy"): (
                    ["time", "lon", "lat"], diagnostics["surface_tauy"]),
                forcing_variable("heat_flux"): (
                    ["time", "lon", "lat"], diagnostics["heat_flux"]),
                forcing_variable("freshwater_flux"): (
                    ["time", "lon", "lat"], diagnostics["freshwater_flux"]),
                "mask_T": (["lon", "lat", "depth"], self.mask_T),
                "mask_surface_T": (["lon", "lat"], self.mask_T[:, :, -1]),
                "mask_U": (["lon", "lat", "depth"], self.mask_U),
                "mask_surface_U": (["lon", "lat"], self.mask_U[:, :, -1]),
                "mask_surface_Z": (["lon", "lat"], self.mask_surface_Z),
                "dzt": (["depth"], self.dzt),
            },
            coords={
                # ``attrs`` is a fresh dict per access, so xarray -- which
                # keeps the dict it is handed -- gets one of its own.
                "time": (["time"], time.datetimes(), time.attrs),
                "lon": (["lon"], self.longitude),
                "lat": (["lat"], self.latitude),
            },
        )

        dataset.lon.attrs = {"long_name": "T-grid longitude", "units": self.longitude_units}
        dataset.lat.attrs = {"long_name": "T-grid latitude", "units": self.latitude_units}

        # Nothing in the numbers says whether `psi` is Veros' own prognostic
        # streamfunction or the diagnosis that stands in for it, so the
        # attribute does. See `_barotropic_streamfunction`.
        if self.enable_streamfunction:
            psi_comment = (
                "Veros' own prognostic barotropic streamfunction"
                " (`variables.psi` at the current time level): this run"
                " solves the external mode for it"
                " (settings.enable_streamfunction)."
            )
        else:
            psi_comment = (
                "diagnosed as the meridionally integrated depth-integrated"
                " zonal transport, fixed to zero at the southern boundary:"
                " this run solves the external mode for a linear free"
                " surface (settings.enable_streamfunction is False), where"
                " Veros' `variables.psi` holds the surface pressure instead;"
                " that solve's own sea surface height is published here as"
                " `ssh`."
                " The barotropic flow is then not exactly non-divergent, so"
                " this is the standard `meridionally integrated zonal"
                " transport` diagnostic rather than an exact streamfunction."
                " Its values over land are carried through the integration"
                " rather than computed, so blank them with `mask_surface_Z`"
                " before reading them as transports; the depth integral it"
                " comes from uses `mask_U` and `dzt`, both published here,"
                " so it can be reproduced from this file."
            )
        psi_comment += (
            " Like `u` and `v` it lives on Veros' staggered grid -- here the"
            " zeta (corner) points, whose surface land-sea mask is published"
            " as `mask_surface_Z` -- but is labelled with the T-grid"
            " `lon`/`lat` coordinates this dataset uses throughout."
        )

        # `jem_role` records which section of the carry each variable came
        # from, so a reader does not have to parse the `forcing_` prefix.
        # The masks and `dzt` at the end are the grid itself -- time-invariant
        # configuration, not state, diagnostics or forcing -- so they carry no
        # role.
        var_attrs = {
            "temp": {"long_name": "ocean potential temperature", "units": "deg C",
                     **role_attrs("state")},
            "salt": {"long_name": "ocean salinity", "units": "g/kg",
                     **role_attrs("state")},
            "u": {"long_name": "zonal ocean velocity", "units": "m/s",
                  **role_attrs("state")},
            "v": {"long_name": "meridional ocean velocity", "units": "m/s",
                  **role_attrs("state")},
            "sea_surface_temperature": {
                "long_name": "sea surface temperature", "units": "K",
                "comment": "unlike `temp`, this field is shifted by +273.15 to Kelvin",
                **role_attrs("derived"),
            },
            "sea_surface_u": {"long_name": "sea surface zonal velocity", "units": "m/s",
                              **role_attrs("derived")},
            "sea_surface_v": {"long_name": "sea surface meridional velocity", "units": "m/s",
                              **role_attrs("derived")},
            "sea_surface_salinity": {"long_name": "sea surface salinity", "units": "g/kg",
                                     **role_attrs("derived")},
            "psi": {"long_name": "barotropic streamfunction", "units": "m^3/s",
                    "comment": psi_comment, **role_attrs("derived")},
            forcing_variable("surface_air_temperature"): {
                "long_name": "surface air temperature forcing", "units": "K",
                "comment": "unit inferred by convention; not dimensionally enforced anywhere in this module",
                **role_attrs("forcing"),
            },
            forcing_variable("surface_taux"): {
                "long_name": "zonal surface wind stress forcing", "units": "N/m^2",
                **role_attrs("forcing")},
            forcing_variable("surface_tauy"): {
                "long_name": "meridional surface wind stress forcing", "units": "N/m^2",
                **role_attrs("forcing")},
            forcing_variable("heat_flux"): {
                "long_name": "net surface heat flux forcing (upward positive)", "units": "W/m^2",
                **role_attrs("forcing")},
            forcing_variable("freshwater_flux"): {
                "long_name": "net surface freshwater flux forcing (upward positive)",
                "units": "kg/m^2/s",
                **role_attrs("forcing")},
            "mask_T": {"long_name": "land-sea mask on T grid", "units": "1"},
            "mask_surface_T": {"long_name": "land-sea mask on T grid, surface level", "units": "1"},
            "mask_U": {"long_name": "land-sea mask on u grid", "units": "1",
                       "comment": "the mask `u` lives on, and the one `psi`'s depth integral uses"},
            "mask_surface_U": {"long_name": "land-sea mask on u grid, surface level", "units": "1"},
            "mask_surface_Z": {"long_name": "land-sea mask on zeta (corner) points, surface level",
                               "units": "1",
                               "comment": "the points `psi` itself lives on"},
            "dzt": {"long_name": "vertical grid spacing (T)", "units": "m"},
        }

        # The sea surface height exists only where `step` emitted one, so it
        # is added rather than sitting in the literals above. Unlike `psi` it
        # needs no staggering note: it is a T-grid field, the grid this
        # dataset's `lon`/`lat` already label.
        if not self.enable_streamfunction:
            dataset["ssh"] = (["time", "lon", "lat"], diagnostics["ssh"])
            var_attrs["ssh"] = {
                "long_name": "sea surface height", "units": "m",
                "comment": (
                    "the sea surface height of Veros' surface-pressure"
                    " solve, `ssh = psi / grav` -- the relation Veros itself"
                    " applies when it sets `variables.ssh` -- evaluated on"
                    " the `variables.psi` of this record's own time level,"
                    " which `variables.ssh` is one Veros timestep behind"
                    " because Veros writes it before permuting its time"
                    " indices. Present only for a run that solves the linear"
                    " free surface (settings.enable_streamfunction is"
                    " False), where `variables.psi` holds the surface"
                    " pressure; a streamfunction run has no sea surface"
                    " height to publish."
                ),
                **role_attrs("derived"),
            }

        for name, attrs in var_attrs.items():
            dataset[name].attrs = attrs

        return dataset

    def save_carry(self, carry: Carry, directory: Path) -> None:
        """Write the carry to ``directory`` (:class:`~jem.base.component.SupportsCheckpoint`).

        The ``VerosState`` goes through Veros' own HDF5 restart writer
        because it is not a plain pytree -- it is a mutable object holding
        settings, dimensions and its own array backend. What is left of the
        carry, the ``derived`` and ``forcing`` structs, is an ordinary pytree
        and is written beside it by :func:`jem.checkpoint.save`, under the
        same file name a coupled checkpoint uses, so one directory has one
        carry file however deep in the model it sits.

        Parameters
        ----------
        carry : Carry
            ``{"state": VerosState, "derived": ..., "forcing": ...}``.
        directory : pathlib.Path
            Directory to write into; created if absent.

        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        from veros.restart import write_restart

        state = carry["state"]
        with state.settings.unlock():
            state.settings.restart_output_filename = str(
                directory / VEROS_RESTART_FILENAME
            )
            logger.info(
                "Saving ocean restart file to %s",
                state.settings.restart_output_filename,
            )
        write_restart(state, force=True)

        # The restart file first, the pytree carry file last: the coupled
        # checkpoint treats the carry file as the completion marker, and this
        # component keeps the same promise for its own directory.
        save_pytree(
            {"derived": carry["derived"], "forcing": carry["forcing"]},
            directory / CARRY_FILENAME,
        )

    def load_carry(self, directory: Path) -> Carry:
        """Read back a carry written by :meth:`save_carry`.

        Veros' restart reader mutates ``model.state`` in place, so the
        returned carry shares that object -- as :meth:`initialize` does. The
        ``derived`` and ``forcing`` structs are poured back into the templates
        :meth:`initialize` builds, so a restart written on another grid is
        refused by shape rather than silently adopted.

        Parameters
        ----------
        directory : pathlib.Path
            A directory written by :meth:`save_carry`.

        Returns
        -------
        Carry

        """
        directory = Path(directory)

        from veros.restart import read_restart

        state = self.model.state
        with state.settings.unlock():
            state.settings.restart_input_filename = str(
                directory / VEROS_RESTART_FILENAME
            )
        # Veros refuses to read a restart while `force_overwrite` is on, which
        # this module turns on so that a coupled run may rewrite its own
        # outputs; the context manager puts it back however the read ends.
        with _veros_runtime_setting("force_overwrite", False):
            read_restart(state)

        template = {
            "derived": VerosDerived.zeros(self.horizontal_shape),
            "forcing": VerosForcing.zeros(self.horizontal_shape),
        }
        stored = load_pytree(template, directory / CARRY_FILENAME)
        return {"state": state, **stored}


def make_jem_compatible(
    model: Any,
    coupling_timestep: jdt.Timedelta,
) -> VerosComponent:
    """Return a :class:`VerosComponent` for ``model``.

    .. deprecated::
        Construct ``VerosComponent(model)`` directly and register it with a
        ``Coupler``, which binds the coupling timestep.

    Parameters
    ----------
    model : veros.VerosSetup
        The ocean to wrap.
    coupling_timestep : jax_datetime.Timedelta
        Ignored. The coupler now supplies the coupling timestep, together
        with the start date and calendar, through
        :meth:`VerosComponent.bind`.

    Returns
    -------
    VerosComponent
        A new wrapper. Constructing it still replaces the setup's
        ``set_forcing`` with a no-op, which a coupled run requires.

    """
    warnings.warn(
        "make_jem_compatible() is deprecated: use"
        " jem.components.veros_component.VerosComponent(model) and register"
        " it with a Coupler. The returned component is a wrapper object --"
        " unlike the old function it does not inject"
        " initialize()/generate_step_function()/predictions_to_xarray()"
        " onto the VerosSetup, so anything calling those on the model must"
        " call them on the component instead. The coupling_timestep argument"
        " is ignored; the Coupler passes it to VerosComponent.bind().",
        DeprecationWarning,
        stacklevel=2,
    )
    return VerosComponent(model)
