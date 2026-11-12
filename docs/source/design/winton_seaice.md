# The Winton sea ice

`WintonSeaiceModel` (`seaice=winton`) is a thermodynamic sea ice with a snow
layer, two ice layers and a Hibler (1979) lead closure for the ice fraction.
Its thermodynamics are a port of MITgcm's `pkg/thsice` (Winton 2000), checked
against the Fortran at machine precision (`tests/reference/thsice`, built and
run in CI). The numerics, constants and sign conventions are in the module
docstring of `jem.components.slab.winton_seaice_model`; this page is its
coupling contract. The component was written by Dhruv Balwada (jax-esm#114).

Unlike `SlabSeaiceModel`, which diagnoses its cover from the SST, the Winton
ice has an energy budget: it grows from the ocean's freezing deficit and from
conduction through the ice, melts at its surface and base, and holds snow. The
atmosphere sees its fraction and its skin temperature.

## Areas

Every flux and amount is per unit area of the **sea part** of the cell, and
the ice fraction is a fraction of the sea part. That is SPEEDY's convention for
`sice_am`, and the slab ocean's for its mixed layer, so on a coastal cell the
atmosphere's sea tile, the ice and the ocean under it share one area.

## What the ice reads

| field | from | meaning |
| --- | --- | --- |
| `rsds`, `rlds` | `atm.derived.downward_shortwave_flux`, `downward_longwave_flux` | downward shortwave and incident downward longwave at the surface (W m⁻²); the ice absorbs `emissivity` of the longwave |
| `air_temperature`, `air_specific_humidity` | `atm.derived.sea_air_temperature`, `sea_air_specific_humidity` | the air the atmosphere's sea fluxes were computed against (SPEEDY's sea surface-layer air) |
| `wind_speed`, `normalized_surface_pressure` | `atm.derived.wind_speed`, `normalized_surface_pressure` | the rest of the bulk formula's state |
| `total_heat_flux` | `atm.derived.sea_total_heat_flux` | the atmosphere's net heat flux over the sea part (W m⁻², upward) |
| `evaporation` | `atm.derived.sea_evaporation` | the atmosphere's evaporation over the sea part (kg m⁻² s⁻¹, upward) |
| `liquid_precipitation`, `frozen_precipitation` | `atm.derived.rainfall`, `snowfall` | rain and snow (kg m⁻² s⁻¹, downward) |
| `sea_surface_temperature` | `ocn.state.sea_surface_temperature` | the ocean's SST (K) |
| `ice_frazil_melt_energy` | `ocn.derived.ice_frazil_melt_energy` | the ocean's freeze/melt potential per coupling step (J m⁻², CESM's `frzmlt`) |
| `ice_velocity_u`, `ice_velocity_v` | (none) | ice drift, read only when the model is built with transport |

The ice computes its own surface fluxes, with SPEEDY's bulk formulae, from the
atmosphere's state, so that the flux and its derivative are available inside
the implicit surface-temperature solve.

**Rain and snow.** SPEEDY condenses liquid only, so the JCM adapter splits its
precipitation as MITgcm's AIM (SPEEDY's physics in MITgcm) splits it for its
thermodynamic sea ice: all of a cell's precipitation is snow where the lowest
level's potential temperature brought to the surface pressure is below
273.15 K (`pkg/aim_v23/phy_snow_precip.F`). AIM also corrects its
condensation heating for the latent heat of fusion of that snow; SPEEDY in
jax-gcm does not, so the ice charges that heat to the ocean when it stores
the snow (`frozen_precipitation_latent_heat_release_mode=True`, thsice's
convention). A package that splits its own precipitation by phase is taken at
its word.

## What the ice publishes

- `ice_fraction` → `atm.forcing.sice_am`, and `ice_surface_temperature_K` →
  `atm.forcing.sea_ice_surface_temperature`. SPEEDY blends the two with the
  SST into one sea-surface temperature, `tsea = (1 − f) SST + f T_ice`, as its
  own sea model blends its ice (`sea_model.f90`), and evaluates its sea fluxes
  there. Without an ice model SPEEDY takes `T_ice` as the SST capped at the
  freezing point; under a clamped ocean that is a 271 K surface in the polar
  winter.
- `ocean_heat_flux_up` → `ocn.forcing.total_heat_flux` (W m⁻², upward): the
  atmosphere's downward flux over the sea part, minus what the ice absorbed,
  plus what it passes down (transmitted shortwave, melt surplus), minus the
  basal heat it draws. The ice and the ocean together receive exactly the flux
  the atmosphere lost. Where the atmosphere's flux over the ice part differs
  from the ice's own (SPEEDY evaluates it at the blended temperature and with
  its own ice albedo), the difference goes to the ocean.
- `ocean_freshwater_flux_up` (kg m⁻² s⁻¹, upward): the atmosphere's E − P over
  the sea part, plus the snow held on ice until it melts (`snowfall_on_ice`),
  plus the water ice growth takes. The ice has no sublimation term, so the
  evaporation over the ice-covered part is charged to the ocean. A slab ocean
  has no salinity, so this is a diagnostic.
- Budget diagnostics: ice and snow volume, surface melt, basal growth, the
  thermodynamic and transport enthalpy tendencies. They are exact for the
  thermodynamic state, so the ice–ocean–atmosphere budget can be closed.
- `sea_fraction` and `ice_area_fraction` (the ice's share of the whole cell),
  which weight regridded rows back onto the atmosphere: the ice fraction by
  the sea part, the skin temperature by the ice area, so that an atmosphere
  cell's `sice_am · T_ice` is the ice-area mean of the ocean cells under it.

## Frazil ice and the ocean's clamp

The ocean clamps its mixed layer at the freezing point
(`SlabOceanParameters.clamp_at_freezing=True`). When a coupling step takes the
mixed layer below freezing, the ocean puts it back at freezing and reports the
heat that took as `ice_frazil_melt_energy`, the deficit of that step alone.
The ice turns that energy into frazil ice. The heat the clamp added is the
latent heat that freezing releases, so `ocean_heat_flux_up` does not return it
a second time; `ocean_frazil_heating` reports it as a diagnostic.

We chose the clamp over leaving the ocean below freezing for the ice to warm
back, because the coupling is lagged: the heat the ice returned would arrive a
step after the ocean had already reported its next deficit, which would still
contain the part the ice had just frozen, and the ice would freeze it twice. A
clamping ocean reports each deficit once, whatever the lag.

When the mixed layer is above freezing, the energy is negative and is not an
exchange: it caps the basal heat the ice may draw in one step,
`F_b = min(F_b_raw, −E / dt)`, so the ice cannot take more than the mixed
layer holds above freezing. The heat it does draw is part of
`ocean_heat_flux_up`.

## What the standard coupling does

With a Winton ice registered as `"seaice"`, `default_exchanges` uses the slab
table with its ocean heat-flux row replaced by `WINTON_SEAICE_EXCHANGES`: the
rows above. It refuses three compositions it cannot couple faithfully:

- **an ocean that does not clamp** (`clamp_at_freezing=False`): it keeps its
  deficit, and the ice would freeze it a second time;
- **the Veros ocean**: Veros neither clamps its top cell nor publishes the
  deficit;
- **an atmosphere with no sea flux tile** (ECHAM): the ice would be forced by
  the cell mean, and ECHAM integrates its own sea ice on its sea-ice tile.

Without an atmosphere (a forced or standalone ice) the fields the atmosphere
would supply are the caller's to write. `+configuration=earth-slab-winton` is
the Earth-like configuration: `earth-slab` with a clamping slab ocean and the
Winton ice started from the observed cover of the start month.

## A year on Earth

`earth-slab-winton` against `earth-slab` (the SST-diagnosed slab ice), one
year from 1 January, both from the atmosphere's isothermal rest. The first
year is therefore partly spin-up. Ice area in 10⁶ km², months January to
December, against the observed concentration in the configuration's own
forcing file:

| | NH | SH |
| --- | --- | --- |
| observed | 12.7 13.5 13.6 12.9 11.5 9.5 6.9 5.3 5.1 7.0 9.4 11.3 | 3.7 2.3 3.0 5.5 8.5 11.3 13.7 15.2 15.8 15.3 12.7 7.7 |
| slab ice | 16.8 17.8 17.3 15.1 10.3 1.0 0.1 0.3 3.9 10.2 13.8 16.4 | 3.0 3.2 6.9 12.1 17.5 21.6 24.1 25.0 24.3 21.7 15.9 7.3 |
| Winton | 15.0 17.8 18.0 16.1 13.2 8.4 3.2 0.6 0.3 4.3 12.9 16.3 | 3.0 1.8 2.5 8.1 14.8 20.0 24.0 25.7 25.4 23.4 20.4 16.0 |

**What the coupling does.**
- The atmosphere sees the ice's own skin. Over the Arctic (north of 66°N)
  its February surface is 240.7 K with the Winton ice, against 261.5 K with
  the slab ice, which SPEEDY takes to be at the freezing point. Over the
  Antarctic (south of 66°S) in August it is 229.5 K, against 242.8 K.
- The Winton ice holds an energy budget, so the NH cover declines over the
  summer and reaches its minimum in September, as observed. The slab ice
  follows the SST and is nearly gone by June (1.0×10⁶ km²).
- The ice's budget closes on every cell it integrates: to 0.15 W m⁻² in a
  float32 run, against fluxes of about 110 W m⁻². The residual is float32
  round-off.
- Snow accumulates on the NH ice from February, reaching 2×10³ km³ in April.

**What it does not fix.** Both runs share the atmosphere's biases:
- an Arctic July surface of 278 K, which melts the NH ice out in summer;
- a cold Antarctic winter, which over-extends the SH ice. Under the Winton
  ice this cover is thick: 46×10³ km³ in October, about 2 m on average, and
  it retreats late (16×10⁶ km² in December).

These are not the coupling's to fix. The ice's untuned levers are its
albedos (snow takes the bare-ice 0.60 unless `snow_albedo` is given) and the
absence of ice dynamics (`transport=False` by default).

## Initial condition

With `ice_clim_file` (a monthly `icec`, such as the atmosphere's forcing file)
the ice starts from the observed cover of the month the run starts in, at
`params.initial_ice_thickness`. Without one, every ocean cell starts fully
covered at that thickness, or ice-free if it is zero (the default).
