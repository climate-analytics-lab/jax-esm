# The Winton sea ice

`WintonSeaiceModel` is a thermodynamic sea ice with a snow layer, two ice layers and a Hibler (1979) lead closure for the ice fraction. Its thermodynamics are a port of MITgcm's `pkg/thsice` (Winton 2000), checked against the Fortran at machine precision (`tests/reference/thsice`). The numerics, constants and sign conventions are in the module docstring of `jem.components.slab.winton_seaice_model`; this page is its coupling contract.

## What the ice reads

Everything the ice needs arrives in `forcing`, in these units:

| field | meaning |
| --- | --- |
| `rsds`, `rlds` | downwelling shortwave and longwave at the surface (W m-2) |
| `air_temperature`, `air_specific_humidity`, `wind_speed`, `normalized_surface_pressure` | the near-surface state the bulk fluxes over ice are computed from |
| `evaporation` | the atmosphere's evaporation over the sea part of the cell (kg m-2 s-1, upward) |
| `liquid_precipitation`, `frozen_precipitation` | rain and snow (kg m-2 s-1, downward) |
| `atm_sea_heat_flux` | the atmosphere's heat flux over the sea part of the cell (W m-2, downward) |
| `sea_surface_temperature` | the ocean's SST (K) |
| `ice_frazil_melt_energy` | the ocean's freeze/melt potential per coupling step (J m-2, CESM's `frzmlt`) |
| `ice_velocity_u`, `ice_velocity_v` | ice drift, used only when the model is built with transport |

The ice computes its own surface fluxes, with SPEEDY's bulk formulae, from the near-surface state, so that the flux and its derivative are available inside the implicit surface-temperature solve. The atmosphere sees the ice only through the ice fraction.

## What the ice publishes

- `ice_fraction`, which the standard coupling copies to JCM's `sice_am`.
- `ocean_heat_flux_up` (W m-2, upward): the flux the ocean under the cell should receive. It is the atmosphere's flux over the sea part, minus what the ice absorbed, plus what the ice passes down, minus the basal heat it draws, minus the latent heat of the frazil ice formed from the ocean's freezing potential (`ocean_frazil_heating`). That last term returns the frazil heat to an ocean that does not clamp its own temperature at freezing (Veros); an ocean that clamps has already removed the deficit and would count it twice.
- `ocean_freshwater_flux_up` (kg m-2 s-1, upward): the atmosphere's E - P over the sea part, plus the snow held on ice until it melts (`snowfall_on_ice`), plus the water ice growth takes. The ice has no sublimation term, so the evaporation over the ice-covered part is still charged to the ocean.
- Budget diagnostics: ice and snow volume, surface melt, basal growth, the thermodynamic and transport enthalpy tendencies. These are exact for the thermodynamic state, so a coupler can close the ice-ocean-atmosphere budget.

## What the standard coupling fills

The standard table (see {doc}`exchange`) was written for the SST-diagnosed `SlabSeaiceModel`. Registered as `"seaice"`, the Winton ice gets only its SST and passes its ice fraction to the atmosphere. The atmospheric forcing it reads, the freeze/melt potential and the routing of `ocean_heat_flux_up` and `ocean_freshwater_flux_up` back to the ocean are not in the table: `examples/02_experimental/02_winton_seaice.py` writes that exchange out by hand. Both gaps are tracked in [jax-esm#141](https://github.com/climate-analytics-lab/jax-esm/issues/141).
