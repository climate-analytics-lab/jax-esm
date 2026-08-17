from veros.core.operators import update, at

# Veros hard-coded ghost cell number; see jem/components/Veros.py, where
# `derived["sea_surface_temperature"]` is diagnosed from this same slice
# of the prognostic temperature field.
_OCEAN_GHOST_CELL = 2

def get_ocean_temperature(ocn_state):
    """Read the ocean's temperature [K] and salinity [PSU] from its prognostic state."""
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    return vs.temp[g:-g, g:-g, :, vs.tau] + 273.15

def get_ocean_salinity(ocn_state):
    """Read the ocean's salinity [PSU] from its prognostic state."""
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    return vs.salt[g:-g, g:-g, :, vs.tau]

def set_ocean_temperature(ocn_state, temp):
    """Functionally overwrite the ocean's prognostic temperature [K] at all depths."""
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    with vs.unlock():
        vs.temp = update(vs.temp, at[g:-g, g:-g, :, vs.tau], temp - 273.15)
    return ocn_state

def set_ocean_salinity(ocn_state, salt):
    """Functionally overwrite the ocean's prognostic salinity [PSU]."""
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    with vs.unlock():
        vs.salt = update(vs.salt, at[g:-g, g:-g, -1, vs.tau], salt)
    return ocn_state

def get_ocean_surface_temperature(ocn_state):
    """Read the ocean's sea surface temperature [K] from its prognostic state.

    Mirrors the diagnostic computed in `jem.components.Veros.make_jem_compatible`,
    so callers can read/perturb exactly the field the coupled dynamics use.
    """
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    return vs.temp[g:-g, g:-g, -1, vs.tau] + 273.15

def set_ocean_surface_temperature(ocn_state, sst):
    """Functionally overwrite the ocean's prognostic surface temperature [K].

    Unlike `coupled_carry["ocn"]["derived"]["sea_surface_temperature"]`, which
    is recomputed from the prognostic field every sub-step, this writes into
    `state.variables.temp` itself, so a perturbation here actually propagates
    through the ocean dynamics.
    """
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    with vs.unlock():
        vs.temp = update(vs.temp, at[g:-g, g:-g, -1, vs.tau], sst - 273.15)
    return ocn_state


