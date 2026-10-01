# The Winton sea ice

`jem.components.WintonSeaiceModel` is a three-layer thermodynamic sea-ice
component: one snow layer and two ice layers per cell, an implicit
surface-temperature solve, a Hibler sub-grid lead closure, exact water and
enthalpy diagnostics, and an optional conservative transport step. It is a
sibling of the single-layer `SlabSeaiceModel` and is chosen with
`seaice=winton` in the configuration or built directly in Python
({doc}`../python_api`).

## Scheme and provenance

The thermodynamic core is a port of MITgcm `pkg/thsice`
(`THSICE_SOLVE4TEMP`, `THSICE_CALC_THICKN`), which implements Winton (2000),
*J. Atmos. Oceanic Technol.* 17, 525-531. Where thsice departs from the paper
the component follows thsice: layer enthalpies relative to liquid water at
0 degC with a brine melting point, a surface clamped at 0 degC while melting,
surface melt applied before basal growth, and re-equalisation of the two ice
layers with the `q2 > L` guard. The sub-grid ice fraction uses a Hibler (1979)
lead closure: frazil ice closes leads over `lead_closing_thickness`, melt
shrinks the fraction by `-(f / 2h) dh`, and the thermodynamics act on the
ice-covered part of the cell, of which `h` is the thickness. Bulk surface
fluxes over ice are SPEEDY's, so the flux and its derivative are available to
the implicit solve. The package `NOTICE` carries the MITgcm licence and the
references, and `tests/reference/thsice` rebuilds an oracle from the unmodified
Fortran that the port is compared with to round-off.

### Constants

The scheme's own constants (ice density `RHO_ICE = 905`, heat of fusion
`L_ICE = 3.34e5`, freezing point `T_FREEZE = -1.8` degC, the layer
conductivities, ...) are the MITgcm `thsice` values, and they are deliberately
**module-local** to
`jem.components.slab.winton_seaice_model.winton_seaice_model` rather than read
from `jcm.constants`, where the two model families chose differently (ice
density 905 against 917, heat of fusion 3.34e5 against 3.33e5). The reason is
the oracle: the comparison against the unmodified Fortran is exact only
against the `thsice` parameter set, so substituting the `jcm.constants` values
would turn a round-off comparison into an approximate one. The ice is also a
closed enthalpy budget in those units: the ocean is charged in joules, so the
coupled budget closes whichever density or heat of fusion the atmosphere
uses. What the ice shares with the atmosphere's
surface-flux scheme -- `cpd`, `rd`, `p0`, `sbc` and `alhc` -- is read from
`jcm.constants` when the step is traced, so `set_constants(...)` reaches the
bulk fluxes over ice as it reaches the atmosphere. The freezing point of the
water under the ice is `T_FREEZE = -1.8` degC, the value of
`jem.constants.seawater_freezing_point_K`, fixed in the ice model's degC.

## The component

The carry is the shared layout:

| Section | Contents |
|---|---|
| `params` | a `WintonSeaiceParameters` (below) |
| `state` | `WintonState`: ice and snow thickness of the ice-covered part, the ice fraction, the two layer temperatures and the surface temperature, in degC |
| `forcing` | `WintonForcing`: what other components supply |
| `derived` | `WintonDerived`: budgets and diagnostics, rebuilt every step |

The component holds no clock and no timestep. `step` advances the ice by the
`time.dt` of the `CouplingTime` it is handed, divided into
`params.n_substeps` thermodynamic substeps, so the coupling timestep can be
changed without touching the component.

`WintonSeaiceParameters` is a `flax.struct.dataclass`: every numeric tunable
(albedos, penetrating-shortwave fraction, extinction coefficient,
lead-closing thickness, thresholds, emissivity, the transport diffusivity and
the nested JCM `SurfaceFluxParameters` used for the fluxes over ice) is a leaf
that `jax.grad` reaches through the coupler. The substep and iteration counts
and the mask convention are static. The bulk-flux parameters are held in an
`IceSurfaceFluxParameters` (`chs`, `vgust`, `dtheta`, `fstab`, all leaves, and
the static flag `lscasym`) rather than as JCM's `SurfaceFluxParameters`, which
also has boolean fields that make `jax.grad` over the whole parameter pytree
fail without `allow_int`; passing JCM's object converts it, so the values the
atmosphere runs with can still be handed over. Every leaf of the parameter
pytree is a float. `snow_albedo` and `snow_melt_albedo`
default to `None`, which makes snow take the ice albedo (and melting snow the
dry-snow albedo), so a gradient with respect to `ice_albedo` also moves the
snow albedo; giving a value decouples them. `initial_ice_thickness` is an
initial-condition parameter: it is read by `initialize` only and varied by
passing parameters to `initialize` ({doc}`carry_and_clock`, *Parameters*).
Parameters are validated at construction, where they are concrete.

## Coupling

Fields move only through exchangers. Signs and units are the project's: heat
flux positive upward at the ocean interface, `ice_frazil_melt_energy` in the
CESM `frzmlt` convention (positive forms ice), temperatures in K at the
interface and degC inside the ice.

### What the ice reads

| `WintonForcing` field | Source in the default coupling |
|---|---|
| `ice_frazil_melt_energy` | `ocn.derived.ice_frazil_melt_energy` (standard row) |
| `sea_surface_temperature` | `ocn.state.sea_surface_temperature` (a row added when `seaice` is a Winton model; `ocn.derived...` for Veros) |
| `rsds`, `rlds`, `air_temperature`, `air_specific_humidity`, `normalized_surface_pressure`, `wind_speed` | none |
| `snowfall` | none |
| `atm_sea_heat_flux` | none |
| `ice_velocity_u`, `ice_velocity_v` | none; zero, so transport is pure diffusion |

Fields with no source stay at the value `initialize` gave them: air and sea at
the freezing point, saturation humidity, 5 m/s wind, unit normalized pressure,
and a downward longwave flux equal to what the ice itself emits there,
`emissivity * sbc * T**4` with the parameters' emissivity. At that seed the net
surface flux is zero, so ice at the freezing point has no surface forcing to
melt or grow it for want of a supplier (the first step also runs on this
forcing, because coupling is lagged).

**The default coupling does not force the ice with an atmosphere.** No default
row supplies the fields marked "none" above, so an ice coupled to an
atmosphere through the default table would run on the seed values for the
whole run. This gap is tracked in
[issue #141](https://github.com/climate-analytics-lab/jax-esm/issues/141),
which is where the atmospheric-forcing exchange (and the choice of which
atmosphere flux stands in for the open-sea flux) will be settled. Until then:

- **From Python**, `default_exchanges` / `default_exchangers` stay permissive,
  because a forced or standalone ice is legitimate; they log a warning naming
  the fields left unsupplied.
- **From Hydra**, building the coupling raises an error naming those fields
  when a Winton ice and an `atm` component are configured and neither
  `coupling.exchanger` nor `coupling.exchangers` is set. Supply the forcing
  with an exchanger of your own, reading whatever your atmosphere publishes,
  and the check is out of the way.

Three points of the contract belong to whoever writes that exchanger:

- **`atm_sea_heat_flux`** is the atmosphere's net flux over the *open-sea*
  fraction, downward positive. The ocean is then charged that flux minus what
  the ice absorbed, so the ice-ocean-atmosphere energy budget closes. The JCM
  adapter publishes a grid-cell mean over land and sea only
  ({doc}`jcm_adapter`); which quantity stands in for the open-sea flux is not
  settled by the adapter (issue #141).
- **`snowfall`** is a liquid-phase mass flux (kg m-2 s-1) whose latent heat of
  fusion the supplier has **not** released: the ice hands the ocean the fusion
  enthalpy of the snow that lands on it, as thsice does.
- **`ice_velocity_u/v`** are in the grid's own x and y directions.

### What the ice publishes

`derived` carries `ice_fraction`, `effective_sea_surface_temperature` (the
ice-fraction-weighted surface temperature), `ocean_heat_flux_up`,
`ocean_freshwater_flux_up`, the ice-atmosphere flux, the surface-melt and basal
fluxes, volumes, the albedo and the enthalpy tendencies.

**The atmosphere sees the ice through the sea-ice fraction.** The default table
consumes `ice_fraction` and writes it to the atmosphere's `sice_am`, which is
how JCM's surface scheme accounts for ice; the atmosphere is *not* handed an
effective sea surface temperature. The other ocean- and atmosphere-facing
fields are published for an exchanger to use, and the default table consumes
none of them:

- `ocean_heat_flux_up` is the net heat flux leaving the ocean surface,
  positive upward, in the sign of the ocean's `total_heat_flux`: the
  atmosphere's flux over the sea cell less what the ice absorbed, plus what the
  ice passes down to the ocean (transmitted shortwave, melt surplus, the fusion
  enthalpy of snow) less the basal heat the ice draws, all reversed in sign. It is
  the flux an ocean under ice is meant to be forced with in place of the
  atmosphere's grid-mean flux. **The default coupling does not do this**: the
  ocean receives the atmosphere's grid-mean `total_heat_flux`, and
  `ocean_heat_flux_up` is published but consumed by nothing (issue #141).
- `effective_sea_surface_temperature` is the ice-fraction-weighted surface
  temperature, a diagnostic for an atmosphere that resolves one surface
  temperature per cell. The default coupling does not use it: the atmosphere's
  sea surface temperature is the ocean's, and it sees the ice through
  `sice_am`. An atmosphere that were handed this as its sea surface
  temperature would already see the ice and must not also be handed the ice
  fraction.
- `ocean_freshwater_flux_up` is positive when ice growth removes water from
  the ocean.

### Energy and water budgets

The water and enthalpy budgets are exact for the thermodynamic state: the
ice diagnoses what it returns to the ocean (`ocean_heat_flux_up`,
`ocean_freshwater_flux_up`), and snowfall, frazil ice, disposed slivers and
melt-out are all charged so nothing is discarded without its energy. Whether
the ocean *receives* that flux depends on the coupling: under the default
table it does not (see above, issue #141), so the closure is a property of the
ice's own diagnostics, not yet of the coupled run. `derived.ice_energy_tendency` is the thermodynamic rate of change of
the ice and snow enthalpy and `derived.ice_energy_transport` the change by
transport convergence, so
`ice_energy_tendency - ocean_heat_flux_up + ocean_frazil_heating -
atm_sea_heat_flux` is the per-cell residual a conservation check reads. The
residual of the implicit surface solve is the linearisation error of the
surface flux, which falls as `n_flux_iterations` rises; the oracle validates
one iteration and coupled runs use three.

## Transport

`WintonSeaiceModel(grid, transport=True)` adds a flux-form advection-diffusion
step after the thermodynamics: first-order upwind advection by
`forcing.ice_velocity_u/v` and lateral diffusion of the conserved ice
quantities (area, ice and snow volume, layer enthalpies, area-weighted
surface temperature), with no-flux land faces and a cyclic first axis. Every
face flux is split into the outgoing parts of its two cells and cells whose
outgoing fraction would exceed 0.9 per substep are scaled down, so every field
stays non-negative and is conserved to round-off for any velocity or
diffusivity. Ice does not converge into compact cells (a cavitating-fluid
limiter). The step exchanges nothing with the ocean: the budgets are computed
on the thermodynamic state and the ice is then moved between cells.

The metrics (`dx`, `dy` in metres at the cell centres) are properties of the grid:
`transport=True` derives them for a separable lon/lat grid
(`IceTransportGrid.from_grid`), and a curvilinear grid passes an
`IceTransportGrid` built from its own metrics. The diffusivity and the substep
count are parameters. Advection by a velocity needs a source for it; until a
component publishes one the step is diffusion of the ice thickness, which is
what Ferreira, Marshall and Rose (2011) used as a proxy for ice dynamics.

## Output

Every variable is tagged with its `jem_role`; the three that come from the
forcing carry the `forcing_` prefix (`forcing_ice_frazil_melt_energy`,
`forcing_ice_velocity_u`, `forcing_ice_velocity_v`). Cells the model does not
integrate report `MASKED_SURFACE_TEMPERATURE` for the ice surface, as every slab
model does. Dimensions, coordinates and time follow {doc}`output`, so the
dataset merges with the atmosphere's and the ocean's under `join="exact"`.
