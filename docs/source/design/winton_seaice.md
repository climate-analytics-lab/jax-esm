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
jax-gcm does not, so the ice credits the ocean with that heat (`+f S L`, heat to the ocean)
when it stores the snow (`frozen_precipitation_latent_heat_release_mode=True`, thsice's
convention). A package that splits its own precipitation by phase is taken at
its word.

## What the ice publishes

- `ice_fraction` → `atm.forcing.sice_am`, and `ice_surface_temperature_K` →
  `atm.forcing.sea_ice_temperature`. SPEEDY blends the two with the
  SST into one sea-surface temperature, `tsea = (1 − f) SST + f T_ice`, as its
  own sea model blends its ice (`sea_model.f90`), and evaluates its sea fluxes
  there. Without an ice model SPEEDY takes `T_ice` as the SST capped at the
  freezing point; under a clamped ocean that is a 271 K surface in the polar
  winter.
- `ocean_heat_flux_up` → `ocn.forcing.total_heat_flux` (W m⁻², upward): the
  atmosphere's downward flux over the sea part, minus what the ice absorbed,
  plus what it passes down (transmitted shortwave, melt surplus), minus the
  basal heat it draws. The ice and the ocean together receive the flux the
  atmosphere lost, to round-off (see "Energy budget" below). The ice is
  charged the flux its implicit solve conducted, the linearised flux thsice
  reports as `flxAtm`. The ice never hands that flux to the atmosphere: SPEEDY
  evaluates its own sea flux at the blended temperature and with its own ice
  albedo, and the difference between that flux and the ice's goes to the
  ocean.
- `ocean_freshwater_flux_up` (kg m⁻² s⁻¹, upward): the atmosphere's E − P over
  the sea part, plus the snow held on ice until it melts (`snowfall_on_ice`),
  plus the water ice growth takes. The ice has no sublimation term, so the
  evaporation over the ice-covered part is charged to the ocean. A slab ocean
  has no salinity, so this is a diagnostic.
- Budget diagnostics: ice and snow volume, surface melt, basal growth, the
  thermodynamic and transport enthalpy tendencies. They describe the
  thermodynamic state, so the ice–ocean–atmosphere budget can be closed.
- `sea_fraction` and `ice_area_fraction` (the ice's share of the whole cell),
  which weight regridded rows back onto the atmosphere: the ice fraction by
  the sea part, the skin temperature by the ice area, so that an atmosphere
  cell's `sice_am · T_ice` is the ice-area mean of the ocean cells under it.
  That holds only if both are mapped by the same operator, so the skin
  temperature, a state, is mapped conservatively like the fraction rather
  than bilinearly like the SST.

## Energy budget

The per-cell residual of one coupling step,
`ice_energy_tendency − ocean_heat_flux_up + ocean_frazil_heating + total_heat_flux`,
is zero to round-off. It was measured (float64, per-cell random states on a
coastal global grid, eight coupling steps of a nonlinear, time-varying forcing
that takes cells across the melting clamp in both directions) at 4e-12,
8e-12 and 7e-12 W m⁻² for `(n_flux_iterations, n_substeps)` of (1, 1), (1, 4)
and (3, 4); `test_energy_budget_closes_to_roundoff_when_the_surface_solve_has_not_converged`
asserts it below 1e-6. In float32, the dtype of a JEM run, the same
experiment gives at most 4e-3 W m⁻² (2e-3, 4e-3 and 3e-3), float32 round-off
of the enthalpy of columns a few metres thick.

Three choices make that hold, the first two for a non-converged solve:

- The ice-atmosphere flux is the one the implicit solve conducted: the flux and
  its slope at the surface temperature the last iteration started from, carried
  to the new temperature along that slope (thsice's
  `flxAtm += flxTexSW + dFlxdT*dTsrf`). A melting surface sits at 0 °C and takes
  the flux evaluated there. The nonlinear flux re-evaluated at the new surface
  temperature is not what the ice absorbed; the two differ by the
  linearisation error, tens of W m⁻² after one iteration of a long substep,
  and charging the ocean the second leaks exactly that difference.
- A negative surface residual (the melting clamp overshot, so the surface
  conducts more than the atmosphere gave) is a deficit the ocean covers, as
  thsice's `esurp`, rather than being clipped to zero.

- The enthalpy the ice state holds is always one a temperature can represent.
  The state is a temperature per layer, so an enthalpy beyond the melting point
  of layer 1 cannot be stored: clipping its temperature would drop the excess
  from the ice without the ocean receiving it. It arises when a basal melt takes
  layer 2 first and the layers are then re-equalised keeping `q2` fixed (thsice's
  rule, which keeps `T2 <= 0`): the thicker, warm layer 1 would be left warmer
  than its melting point. The excess warmth melts ice instead, at constant
  column enthalpy, down to the thickness whose layers hold the column's
  enthalpy at (`q1` at `T_MELT`, `q2`); that equals the unmelted thickness
  exactly where the rule stops overshooting, so the branch is continuous, and
  the melt water reaches the ocean through the water budget. thsice has no such
  branch: it stops with a "Tice(1) > Tmlt" error in `THSICE_SOLVE4TEMP` on the
  next call. A complete melt-out is where it matters most, because it takes
  several substeps of strong melting to get there, but it is not specific to a
  column that vanishes: the leak is in the partial-melt substeps before it.

The same holds through a complete melt-out, from above (strong surface heating)
and from below (a large ocean melt potential), bare or snow-covered, at partial
and full cover: when the column vanishes its last ice and snow enthalpy is
charged to the ocean in `e_ocn` as thsice's `esurp`/`flx2oc`, the oracle
comparison includes such cases (`complete melt from above/below`) and agrees
with thsice to round-off, and `test_energy_and_water_budgets_close_when_a_column_melts_out_completely`
asserts both the energy residual (below 1e-6 W m⁻²) and the water residual
(`ocean_freshwater_flux_up − (E − P) − d(ice and snow mass)/dt`, below 1e-12
kg m⁻² s⁻¹, with the whole initial mass returned to the ocean) for
`(n_flux_iterations, n_substeps)` of (1, 1), (1, 4) and (3, 4).
A coupled 30-day run (`+configuration=earth-slab-winton`, float32 output)
gives, over every ocean cell and record, a largest residual of 3.5e-3 W m⁻², which
is float32 round-off of the output, and 5e-4 W m⁻² over the 78 cell-steps in
which the ice vanished. Clipping layer 1's temperature at its melting point
instead of melting the excess leaks up to 4.6 W m⁻² in that run (0.8 W m⁻² in the
vanishing steps), by an amount that does not depend on the size of the ocean's
melt potential.

`n_flux_iterations` therefore changes the surface temperature and the fluxes
(the solve is closer to the nonlinear one with more iterations), not whether the
budget closes. The oracle comparison is unchanged: it drives a linear flux, for
which one iteration is exact, and it now also compares `flxAtm` and the
negative-`sHeat` case.

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
rows above. It refuses two compositions it cannot couple faithfully:

- **an ocean that does not clamp** (`params.clamp_at_freezing=False` for a slab
  ocean, `clamp_at_freezing=False` for Veros): it keeps its deficit, and the
  ice would freeze it a second time;
- **an atmosphere with no sea flux tile** (ECHAM): the ice would be forced by
  the cell mean, and ECHAM integrates its own sea ice on its sea-ice tile.

### Over the Veros ocean

The Veros ocean is coupled to a Winton ice by `jem.fluxes.VerosExchange`
(which every Veros configuration uses), through the same table rows weighted
and regridded by the same code (`jem.exchangers.table_specs`,
`VEROS_WINTON_SEAICE_EXCHANGES`). Three things differ from the slab ocean:

- **The clamp is the Veros wrapper's.** `VerosComponent(model,
  clamp_at_freezing=True)` (`+ocean.clamp_at_freezing=true`; off by default)
  raises the top cell to the freezing point once per coupling step and
  publishes the deficit as `derived.ice_frazil_melt_energy`,
  `(T_f - T_top) rho_0 c_p dz_top` in J m⁻², evaluated before the clamp, with
  the slab ocean's sign (positive: the clamp added that heat and the ice must
  freeze it; negative: heat above freezing, which only caps the ice's basal
  heat draw). It is applied to the temperature at the time level `tau` the step
  has just made current, which is the level the next step reads (Veros reads
  `temp` only at `tau`, and its Adams-Bashforth step reads the *tendencies* at
  `tau` and `taum1`; `taum1` temperature is never read), and to wet surface
  cells only. `rho_0` and the heat capacity are those the wrapper divides the
  surface heat flux by, so the energy a clamp adds equals the energy the ice
  is told it received to round-off whatever the seawater constants are
  (`jem.constants`' slab values, 1025 kg m⁻³ and 3985 J kg⁻¹ K⁻¹, differ from
  Veros' by about 0.2 %: that changes the temperature at which an energy
  leaves the ocean, not the conservation between the two). The freezing point
  is `jem.constants.seawater_freezing_point_K`, the slab ocean's and the
  Winton ice's own `T_FREEZE`; not the saline freezing point of Veros'
  equation of state (about -1.9 degC) nor the atmosphere's
  `SEA_ICE_FREEZING_TEMPERATURE`, which only applies where no ice is coupled.
- **The ocean takes both fluxes from the ice**, in place of the atmosphere's
  sea-tile fluxes: `ocean_heat_flux_up` (`forcing.heat_flux`, W m⁻², upward) and
  `ocean_freshwater_flux_up` (`forcing.freshwater_flux`, kg m⁻² s⁻¹, upward),
  which `VerosComponent.step` turns into the salinity forcing
  `forc_salt_surface = fw / rho_0 * 35`. A slab ocean has no salinity, so for it
  the freshwater flux is a diagnostic; for Veros it is part of the coupling.
  Ice growth is water the ocean loses, so it concentrates the top cell (a
  virtual salt flux at the reference salinity; the ice carries no salt in this
  budget).
- **The swamp mask is not applied.** It is the ocean's insulation when there is
  no ice taking the deficit; with the ice, the ice is the insulation.

`VerosExchange.validate` (which `jem.runners` calls on the initial carries)
refuses a Winton ice over a Veros ocean that does not clamp, a clamping ocean
with an ice that takes no deficit (the SST-diagnosed `SlabSeaiceModel`) or with
no ice, and an ice whose sea fraction is not Veros' wet mask. The SST-diagnosed
slab ice is coupled too (the ocean's SST in, its fraction out, the swamp mask
kept) so that no composed ice is silently unforced.

Limits, which are the clamp's and the ice's rather than the exchange's:

- the clamp acts on the **top cell only**. Sub-freezing water below it is not
  frazil ice here; it reaches the surface through the ocean's own mixing and is
  clamped then. A column initialised or driven below freezing in the interior
  keeps that heat deficit until it is mixed up;
- the ice sees the wind stress the atmosphere delivered over the sea tile
  unchanged (there is no ice-ocean stress), and does not drift unless built
  with transport;
- the coupling is lagged by one step like every other, and the ice sits between
  the atmosphere and the ocean, so the atmosphere's flux reaches the ocean two
  steps after the atmosphere computed it (one to the ice, one to the ocean),
  where `seaice=none` takes one. With no ice the ocean's forcing is therefore
  exactly `seaice=none`'s, a step later: over 30 days of `veros-earth-winton`
  the ocean's heat and freshwater forcing are the ice's passthrough to 0.0
  (bit for bit), and equal to `seaice=none`'s of the step before, but the ocean's
  first step is unforced (the ice's initial flux is zero) and the SST trajectory
  differs from `seaice=none`'s by up to 0.9 K at day 30 for that reason alone.
  The pair's energy budget closes exactly only once the energy in transit at the
  end of a run is counted (`tests/unit/test_veros_seaice.py`: 4e-13 of the heat
  exchanged);
- the ice lives on the ocean's grid, and its `sea_fraction` has to be Veros'
  wet mask. `veros-earth-winton` builds it on the ocean's SCRIP grid with
  `jem.tools.veros_earth_ice_mask`, the binary mask the ocean setup thresholds
  from the packaged land fraction.

`+configuration=veros-earth-winton` is the Earth-like Veros configuration with
the ice. The Veros ocean there starts from a horizontally uniform 15 degC state,
so no ice forms in it for months, and the ice starts ice-free: it demonstrates
the coupling (the budgets close, the ice and ocean exchange what they should)
but a seasonal cycle of the ice needs an ocean with a realistic initial state,
which the configuration does not have.

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
- The ice's budget closes on every cell it integrates (see "Energy budget"
  below).
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
