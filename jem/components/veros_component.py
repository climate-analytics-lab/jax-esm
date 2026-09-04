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

import logging
import warnings
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
)
from jem.utils.checkpoints import load_veros_carry, save_veros_carry
from jem.utils.time import time_coordinate

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
    mask_T : jax.Array
        Land-sea mask on the T grid, halo cells removed.
    longitude, latitude : jax.Array
        T-grid cell centres, halo cells removed.
    dlongitude, dlatitude : jax.Array
        T-grid cell widths, halo cells removed.

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
        self.dzt = jnp.array(variables.dzt)
        self.longitude = jnp.array(variables.xt)[interior]
        self.latitude = jnp.array(variables.yt)[interior]
        self.dlongitude = jnp.array(variables.dxt)[interior]
        self.dlatitude = jnp.array(variables.dyt)[interior]
        self.longitude_units = "degrees_east" if settings.coord_degree else "km"
        self.latitude_units = "degrees_north" if settings.coord_degree else "km"

        # Number of Veros tracer timesteps per coupling step; set by bind().
        self._steps_per_coupling_step: int | None = None

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

    def bind(
        self,
        *,
        coupling_timestep: jdt.Timedelta,
        start_date: jdt.Datetime,
        calendar: str,
    ) -> None:
        """Adopt the coupler's clock and work out the internal step count.

        Parameters
        ----------
        coupling_timestep : jax_datetime.Timedelta
            The coupled model's timestep. It must be an exact multiple of
            Veros' tracer timestep ``dt_tracer``.
        start_date : jax_datetime.Datetime
            The run's start date. Recorded for the output metadata; Veros
            keeps its own seconds-since-start counter (``variables.time``)
            and has no calendar of its own to reconcile with.
        calendar : str
            The run's calendar. Recorded for the same reason.

        Raises
        ------
        ValueError
            If the coupling timestep is not a whole multiple of
            ``dt_tracer``.

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
        self._steps_per_coupling_step = int(n_steps)

    def initialize(self) -> Carry:
        """Build the initial carry without integrating the model.

        Returns
        -------
        dict
            ``{"state": VerosState, "derived": VerosDerived,
            "forcing": VerosForcing}``.

        """
        return {
            "state": self.model.state,
            "derived": VerosDerived.zeros(self.horizontal_shape),
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
            The coupler's clock for this step. Unused: Veros advances its
            own ``variables.time`` counter and reads no date-dependent
            forcing of its own once ``set_forcing`` has been disabled.

        Returns
        -------
        tuple
            The new carry and this step's diagnostics as a dict of
            ``(lon, lat[, depth])`` maps.

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

        sea_surface_temperature = (
            variables.temp[interior, interior, -1, tau] + 273.15
        )
        # Land columns carry a fill value rather than a temperature; replace
        # them with a plausible constant so downstream components never see
        # an unphysical SST through the mask.
        sea_surface_temperature = jnp.where(
            sea_surface_temperature < 100, 288.15, sea_surface_temperature)
        sea_surface_salinity = variables.salt[interior, interior, -1, tau]
        sea_surface_u = variables.u[interior, interior, -1, tau]
        sea_surface_v = variables.v[interior, interior, -1, tau]

        diagnostics = {
            "sea_surface_temperature": sea_surface_temperature,
            "sea_surface_salinity": sea_surface_salinity,
            "sea_surface_u": sea_surface_u,
            "sea_surface_v": sea_surface_v,
            "temp": variables.temp[interior, interior, :, tau],
            "salt": variables.salt[interior, interior, :, tau],
            "u": variables.u[interior, interior, :, tau],
            "v": variables.v[interior, interior, :, tau],
            "surface_air_temperature": forcing.surface_air_temperature,
            "surface_taux": forcing.surface_taux,
            "surface_tauy": forcing.surface_tauy,
            "heat_flux": forcing.heat_flux,
            "freshwater_flux": forcing.freshwater_flux,
        }
        return (
            {
                "state": state,
                # ``tree_math.struct`` builds the dataclass at runtime, so
                # mypy cannot see the generated __init__ signature.
                "derived": VerosDerived(  # type: ignore[call-arg]
                    sea_surface_temperature, sea_surface_u, sea_surface_v,
                ),
                "forcing": forcing,
            },
            diagnostics,
        )

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
            The ``time`` coordinate is the absolute ``datetime64[ns]`` axis
            :func:`jem.utils.time.time_coordinate` builds from ``time``, the
            same one every other component labels its output with, so
            ``xr.merge`` joins the records instead of unioning two axes -- or,
            as before this coordinate was written at all, leaving the ocean's
            ``time`` as a bare 0..n-1 index that means nothing.

        """
        n_records = int(jnp.shape(diagnostics["sea_surface_temperature"])[0])
        if len(time) != n_records:
            raise ValueError(
                f"{self.name!r} produced {n_records} output records but the"
                f" coupler's time axis has {len(time)}; the diagnostics"
                " passed here are not the ones this run produced."
            )

        time_values, time_attrs = time_coordinate(time)

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
                "dzt": (["depth"], self.dzt),
            },
            coords={
                "time": (["time"], time_values, time_attrs),
                "lon": (["lon"], self.longitude),
                "lat": (["lat"], self.latitude),
            },
        )

        dataset.lon.attrs = {"long_name": "T-grid longitude", "units": self.longitude_units}
        dataset.lat.attrs = {"long_name": "T-grid latitude", "units": self.latitude_units}

        var_attrs = {
            "temp": {"long_name": "ocean potential temperature", "units": "deg C"},
            "salt": {"long_name": "ocean salinity", "units": "g/kg"},
            "u": {"long_name": "zonal ocean velocity", "units": "m/s"},
            "v": {"long_name": "meridional ocean velocity", "units": "m/s"},
            "sea_surface_temperature": {
                "long_name": "sea surface temperature", "units": "K",
                "comment": "unlike `temp`, this field is shifted by +273.15 to Kelvin",
            },
            "sea_surface_u": {"long_name": "sea surface zonal velocity", "units": "m/s"},
            "sea_surface_v": {"long_name": "sea surface meridional velocity", "units": "m/s"},
            "sea_surface_salinity": {"long_name": "sea surface salinity", "units": "g/kg"},
            forcing_variable("surface_air_temperature"): {
                "long_name": "surface air temperature forcing", "units": "K",
                "comment": "unit inferred by convention; not dimensionally enforced anywhere in this module",
            },
            forcing_variable("surface_taux"): {
                "long_name": "zonal surface wind stress forcing", "units": "N/m^2"},
            forcing_variable("surface_tauy"): {
                "long_name": "meridional surface wind stress forcing", "units": "N/m^2"},
            forcing_variable("heat_flux"): {
                "long_name": "net surface heat flux forcing (upward positive)", "units": "W/m^2"},
            forcing_variable("freshwater_flux"): {
                "long_name": "net surface freshwater flux forcing (upward positive)",
                "units": "kg/m^2/s"},
            "mask_T": {"long_name": "land-sea mask on T grid", "units": "1"},
            "mask_surface_T": {"long_name": "land-sea mask on T grid, surface level", "units": "1"},
            "dzt": {"long_name": "vertical grid spacing (T)", "units": "m"},
        }
        for name, attrs in var_attrs.items():
            dataset[name].attrs = attrs

        return dataset

    def save_state(self, carry: Carry, directory: Path) -> None:
        """Write the carry to ``directory``.

        The ``VerosState`` goes through Veros' own HDF5 restart writer
        because it is not a plain pytree; the derived and forcing structs
        are pickled alongside it.
        """
        save_veros_carry(carry, directory)

    def load_state(self, directory: Path) -> Carry:
        """Read back a carry written by :meth:`save_state`.

        Veros' restart reader mutates ``model.state`` in place, so the
        returned carry shares that object -- as :meth:`initialize` does.
        """
        return load_veros_carry(directory, self.model)


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
