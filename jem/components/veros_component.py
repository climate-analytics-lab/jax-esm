"""Veros adapter to JEM"""
from typing import Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import tree_math
import xarray as xr
from veros import runtime_settings

from jem.utils.bulk_op import stack_objects


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

def copy_veros_state(state):
    return jax.tree_util.tree_map(lambda x: x, state)

def check_before_setattr(target, attribute_name, value, *, raise_exception=True):
    if hasattr(target, attribute_name):
        message = f"Attribute name `{attribute_name:s}` already exists."
        if raise_exception:
            raise AttributeError(message)
        else:
            print(f"Warning: {message:s}")

    setattr(target, attribute_name, value)


@tree_math.struct
class VerosForcing:
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
        return cls(
            heat_flux if heat_flux is not None else jnp.zeros(shape),
            freshwater_flux if freshwater_flux is not None else jnp.zeros(shape),
            surface_taux if surface_taux is not None else jnp.zeros(shape),
            surface_tauy if surface_tauy is not None else jnp.zeros(shape),
            surface_air_temperature if surface_air_temperature is not None else jnp.zeros(shape),
        )


@tree_math.struct
class VerosDerived:
    sea_surface_temperature: jnp.ndarray
    sea_surface_u: jnp.ndarray
    sea_surface_v: jnp.ndarray

    @classmethod
    def zeros(cls, shape, sea_surface_temperature=None, sea_surface_u=None, sea_surface_v=None):
        return cls(
            sea_surface_temperature if sea_surface_temperature is not None else jnp.zeros(shape) + 273.15,
            sea_surface_u if sea_surface_u is not None else jnp.zeros(shape),
            sea_surface_v if sea_surface_v is not None else jnp.zeros(shape),
        )


def make_jem_compatible(
    model: Any,
    coupling_timestep: jdt.Timedelta,
) -> Any:
    """Wrap a Veros model so the JEM `Coupler` can drive it."""
    # The settings were applied when this module was imported; re-checking
    # here catches the case where Veros operators were bound to another
    # backend before that import happened (a setup module imported first).
    configure_veros_runtime()
    # Imported here rather than at module scope: `veros.core.operators` binds
    # its array backend (numpy or jax) at import time from
    # `runtime_settings.backend`, so importing it before
    # `configure_veros_runtime()` has run would silently pin the coupler to
    # numpy operators.
    from veros.core.operators import at, update
    from veros.core.operators import numpy as npx

    timestep = jdt.to_timedelta(int(model.state.settings.dt_tracer), "second")
    if timestep * jnp.floor(coupling_timestep / timestep) != coupling_timestep:
        raise ValueError("Coupling timestep should be a multiple of timestep.")
    steps_per_coupling_timestep = int(coupling_timestep / timestep)
    
    nxt = model.state.dimensions["xt"]
    nyt = model.state.dimensions["yt"]
    ghost_cell = 2 # Veros hard-coded ghost cell number
    
    horizontal_T_shape = (nxt, nyt)

    mask_T = jnp.array(model.state.variables.maskT)
    mask_T = mask_T[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell]

    dzt = jnp.array(model.state.variables.dzt)
    longitude  = jnp.array(model.state.variables.xt)[ghost_cell:-ghost_cell]
    latitude   = jnp.array(model.state.variables.yt)[ghost_cell:-ghost_cell]
    dlongitude = jnp.array(model.state.variables.dxt)[ghost_cell:-ghost_cell]
    dlatitude  = jnp.array(model.state.variables.dyt)[ghost_cell:-ghost_cell]

    lon_units = "degrees_east" if model.state.settings.coord_degree else "km"
    lat_units = "degrees_north" if model.state.settings.coord_degree else "km"

    model.__JEM_TOOL__ = {
        "mask_T": mask_T,
        "latitude": latitude,
        "longitude": longitude,
        "dlatitude": dlatitude,
        "dlongitude": dlongitude,
    }

    def set_forcing(state):
        print("The original set_forcing in the VerosSetup object is replaced "
              "by this empty set_forcing function. JEM-veros will set the "
              "forcing in the step_function.")
        
    def initialize():
        initial_state = model.state
        initial_derived = VerosDerived.zeros(horizontal_T_shape)
        initial_forcing = VerosForcing.zeros(horizontal_T_shape)
        print(f"initial_forcing.surface_taux.shape = {initial_forcing.surface_taux.shape!s}")
        return {"state": initial_state, "derived": initial_derived, "forcing": initial_forcing}
    
    def generate_step_function():

        def step_function(carry, step):
            
            state = carry["state"]
            forcing = carry["forcing"]
            cp_0 = 3991.86795711963
            salinity_ref = 35.0 # PSU
            vs = state.variables
            settings = state.settings
             
            with vs.unlock():
                
                if not settings.enable_tempsalt_sources:
                    print("Warning: settings.enable_tempsalt_sources = False. This is for the `temp_source` variable, "
                          "which is very similar to forc_temp_surface."
                    )
                vs.surface_taux = update(vs.surface_taux, at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell], forcing.surface_taux)
                vs.surface_tauy = update(vs.surface_tauy, at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell], forcing.surface_tauy)
                # The following computation is learned from
                # `veros/setups/global_1deg/global_1deg.py`
                if settings.enable_tke:
                    print("Veros: settings.enable_tke is set true")
                    surface_stress_squared = (
                        (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                        + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
                    )
                    # `sqrt` has an infinite derivative at 0, so AD through `sqrt(x)`
                    # blows up to NaN as x -> 0, even though the primal value (0)
                    # stays finite. Floor the *squared* magnitude -- sqrt's argument
                    # -- at min_stress_magnitude**2 so sqrt and its derivative stay
                    # bounded; this naturally caps the resulting magnitude from below
                    # at min_stress_magnitude.
                    min_stress_magnitude = 1e-3
                    surface_stress_magnitude = npx.sqrt(
                        npx.maximum(surface_stress_squared, min_stress_magnitude ** 2)
                    )
                    vs.forc_tke_surface = update(
                        vs.forc_tke_surface,
                        at[1:-1, 1:-1],
                        surface_stress_magnitude ** 1.5,
                    )

                # W/m^2 K kg/J m^3/kg = K m/s
                vs.forc_temp_surface = update(
                    vs.forc_temp_surface,
                    at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell],
                    - forcing.heat_flux * vs.maskT[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1] / cp_0 / settings.rho_0
                )

                # freshwater_flux is upward positive. Therefore, positive freshwater_flux should increase salinity
                vs.forc_salt_surface = update(
                    vs.forc_salt_surface,
                    at[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell],
                     forcing.freshwater_flux * vs.maskT[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1] / settings.rho_0 * salinity_ref
                )

            def _sub_step_function(_, state):
                model.step(state)
                return state

            state = jax.lax.fori_loop(0, steps_per_coupling_timestep, _sub_step_function, state)
            # `fori_loop` reconstructs the carry into fresh VerosState/VerosVariables
            # instances (via tree_unflatten), so the `vs` bound above is now stale;
            # rebind it to the evolved state before reading diagnostics from it.
            vs = state.variables

            sea_surface_temperature = vs.temp[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau] + 273.15
            sea_surface_temperature = jnp.where( sea_surface_temperature < 100, 288.15, sea_surface_temperature )
            sea_surface_salinity = vs.salt[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau]
            
            temp = vs.temp[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]
            salt = vs.salt[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]
            u = vs.u[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]
            v = vs.v[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, :, vs.tau]

            sea_surface_u = vs.u[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau]
            sea_surface_v = vs.v[ghost_cell:-ghost_cell, ghost_cell:-ghost_cell, -1, vs.tau]

            return {
                "state": state,
                "derived": VerosDerived(sea_surface_temperature, sea_surface_u, sea_surface_v),
                "forcing": forcing,
            }, stack_objects([{
                    "sea_surface_temperature": sea_surface_temperature,
                    "sea_surface_salinity": sea_surface_salinity,
                    "sea_surface_u": sea_surface_u,
                    "sea_surface_v": sea_surface_v,
                    "temp": temp,
                    "salt": salt,
                    "u": u,
                    "v": v,
                    "surface_air_temperature": forcing.surface_air_temperature,
                    "surface_taux": forcing.surface_taux,
                    "surface_tauy": forcing.surface_tauy,
                    "heat_flux": forcing.heat_flux,
                    "freshwater_flux": forcing.freshwater_flux,
            }])


        return step_function

    def predictions_to_xarray(predictions):
        ds = xr.Dataset(
            data_vars={
                "temp": (["time", "lon", "lat", "depth"], predictions["temp"]),
                "salt": (["time", "lon", "lat", "depth"], predictions["salt"]),
                "u": (["time", "lon", "lat", "depth"], predictions["u"]),
                "v": (["time", "lon", "lat", "depth"], predictions["v"]),
                "sea_surface_temperature": (["time", "lon", "lat"], predictions["sea_surface_temperature"]),
                "sea_surface_u": (["time", "lon", "lat"], predictions["sea_surface_u"]),
                "sea_surface_v": (["time", "lon", "lat"], predictions["sea_surface_v"]),
                "sea_surface_salinity": (["time", "lon", "lat"], predictions["sea_surface_salinity"]),
                "surface_air_temperature": (["time", "lon", "lat"], predictions["surface_air_temperature"]),
                "surface_taux": (["time", "lon", "lat"], predictions["surface_taux"]),
                "surface_tauy": (["time", "lon", "lat"], predictions["surface_tauy"]),
                "heat_flux": (["time", "lon", "lat"], predictions["heat_flux"]),
                "freshwater_flux": (["time", "lon", "lat"], predictions["freshwater_flux"]),
                "mask_T": (["lon", "lat", "depth"], mask_T),
                "mask_surface_T": (["lon", "lat"], mask_T[:, :, -1]),
                "dzt": (["depth"], dzt),
            },
            coords={
                "lon": (["lon"], longitude),
                "lat": (["lat"], latitude),
            },
        )

        ds.lon.attrs = {"long_name": "T-grid longitude", "units": lon_units}
        ds.lat.attrs = {"long_name": "T-grid latitude", "units": lat_units}

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
            "surface_air_temperature": {
                "long_name": "surface air temperature forcing", "units": "K",
                "comment": "unit inferred by convention; not dimensionally enforced anywhere in this module",
            },
            "surface_taux": {"long_name": "zonal surface wind stress forcing", "units": "N/m^2"},
            "surface_tauy": {"long_name": "meridional surface wind stress forcing", "units": "N/m^2"},
            "heat_flux": {"long_name": "net surface heat flux forcing (upward positive)", "units": "W/m^2"},
            "freshwater_flux": {"long_name": "net surface freshwater flux forcing (upward positive)", "units": "kg/m^2/s"},
            "mask_T": {"long_name": "land-sea mask on T grid", "units": "1"},
            "mask_surface_T": {"long_name": "land-sea mask on T grid, surface level", "units": "1"},
            "dzt": {"long_name": "vertical grid spacing (T)", "units": "m"},
        }
        for name, attrs in var_attrs.items():
            ds[name].attrs = attrs

        return ds

    def get_info():
        return {
            key: str(value) for key, value in model.state.settings.items()
        }

    check_before_setattr(model, "initialize", initialize)
    check_before_setattr(model, "predictions_to_xarray", predictions_to_xarray)
    check_before_setattr(model, "generate_step_function", generate_step_function)
    check_before_setattr(model, "get_info", get_info)
    check_before_setattr(model, "set_forcing", set_forcing, raise_exception=False)
    
    return model
