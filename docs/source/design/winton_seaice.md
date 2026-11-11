# The Winton sea ice

`WintonSeaiceModel` is a thermodynamic sea ice with a snow layer, two ice layers and a Hibler (1979) lead closure for the ice fraction. Its thermodynamics are a port of MITgcm's `pkg/thsice` (Winton 2000), checked against the Fortran at machine precision (`tests/reference/thsice`). The numerics, constants and sign conventions are in the module docstring of `jem.components.slab.winton_seaice_model`; this page is its coupling contract. The component was written by Dhruv Balwada (jax-esm#114).

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
- `ocean_heat_flux_up` (W m-2, upward): the flux the ocean under the cell should receive. It is the atmosphere's flux over the sea part, minus what the ice absorbed, plus what the ice passes down, minus the basal heat it draws.

## Frazil ice and the ocean's clamp

The ocean clamps its own temperature at the freezing point, and the ice only forms ice. When a coupling step takes the ocean below freezing, the ocean puts its temperature back at freezing and reports the heat that took as `ice_frazil_melt_energy`: the heat deficit of that step alone. The ice turns that energy into frazil ice. The heat the clamp added is the latent heat that freezing releases, so `ocean_heat_flux_up` does not return it a second time; `ocean_frazil_heating` reports it as a diagnostic.

Both of JEM's oceans do this: the slab ocean clamps its mixed layer, and the Veros wrapper clamps Veros' top cell once per coupling step. We chose the clamp over leaving the ocean below freezing for the ice to warm back, because the coupling is lagged: the heat the ice returned would arrive a step after the ocean had already reported its next deficit, which would still contain the part the ice had just frozen, and the ice would freeze it twice. A clamping ocean reports each deficit once, whatever the lag.

When the ocean is above freezing, the energy is negative and is not an energy exchange: it caps the basal heat the ice may draw in one step, `F_b = min(F_b_raw, -E / dt)`, so the ice cannot take more than the mixed layer holds above freezing. The heat it does draw is part of `ocean_heat_flux_up`.
- `ocean_freshwater_flux_up` (kg m-2 s-1, upward): the atmosphere's E - P over the sea part, plus the snow held on ice until it melts (`snowfall_on_ice`), plus the water ice growth takes. The ice has no sublimation term, so the evaporation over the ice-covered part is still charged to the ocean.
- Budget diagnostics: ice and snow volume, surface melt, basal growth, the thermodynamic and transport enthalpy tendencies. These are exact for the thermodynamic state, so a coupler can close the ice-ocean-atmosphere budget.

## What the standard coupling fills

The standard table (see {doc}`exchange`) was written for the SST-diagnosed `SlabSeaiceModel`. Registered as `"seaice"`, the Winton ice gets only its SST and passes its ice fraction to the atmosphere. The atmospheric forcing it reads, the freeze/melt potential and the routing of `ocean_heat_flux_up` and `ocean_freshwater_flux_up` back to the ocean are not in the table: `examples/02_experimental/02_winton_seaice.py` writes that exchange out by hand. There the atmosphere's evaporation and rain and snow go to the ice, and the ice's resulting `ocean_freshwater_flux_up` is all the water the ocean receives, through its `sea_ice_freshwater_flux` (sign-flipped to positive into the ocean); the ocean takes no precipitation or evaporation of its own. A land's runoff goes to the ocean directly. Both gaps are tracked in [jax-esm#141](https://github.com/climate-analytics-lab/jax-esm/issues/141).
