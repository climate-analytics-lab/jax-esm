# The slab land

`SlabLandModel` (`land=slab`) is JAX-ESM's own land: it evolves the land's
temperature, its snow, its soil water and its vegetation fraction, and
diagnoses a surface albedo, on the land part of each cell. It is the
alternative to the land the atmosphere carries inside its physics (SPEEDY's
slab land, ECHAM's SLIM), and a run has exactly one of the two evolving the
land's temperature. The soil water is the bucket of {doc}`bucket_land`, the
same code, so a land with a temperature differs from the bucket land only in
what it adds to it. The model was written by Tien-Yiao Hsu. This page is its
coupling contract and what it does and does not model.

## Choosing the land

Which land evolves the land's temperature, and which the soil water, is the
`land` group:

| `land=` | land temperature | soil water |
| --- | --- | --- |
| `none` (the default) | the atmosphere's own land: SPEEDY's slab land, inside jax-gcm | prescribed climatology (`soilw_am`) |
| `bucket` | the atmosphere's own land | JEM's bucket (`BucketLandModel`) |
| `slab` | JEM's `SlabLandModel` | the same bucket, inside it |

Two things have to agree for `land=slab`, and they are set in two places: the
component has to be in the run, and the atmosphere's own land has to be
switched off. SPEEDY's surface-flux term integrates a slab land unless it is
built with `land_model="prescribed"`; with it on, the atmosphere reads
`forcing.stl_am` as a climatology and adds its own anomaly to it, so the land
tile would see the component's temperature plus a second slab's response to
the same flux. `jem/config/land/slab.yaml` therefore sets both with one switch,
the component and
`atmosphere.physics.terms.speedy_surface_flux.land_model: prescribed`:

    python -m jem.main +configuration=earth-slab land=slab

The file's comments give the reason for every setting. It reads the
atmosphere's own boundary file for the temperature (`stl`), snow (`snowc`) and
soil-water availability (`soilw_am`) climatologies, so both lands are tied to
the same observations; relaxes the temperature to that climatology, which is
SPEEDY's scheme and what makes this the like-for-like alternative to the
built-in land; reads the snow from it, since there is no snow budget; starts
from it; and marks as glacier the land cells whose background albedo (`alb`)
is at or above 0.4, which is how SPEEDY's own land tells land ice from soil
(`glacier_fraction_from_albedo`). The soil water is not prescribed: the bucket
evolves it and hands the atmosphere the availability it diagnoses, as
`land=bucket` does. Appending
`+land.soil_water_availability_from_clim_file=true` prescribes it instead,
which is the built-in land's soil water (see the comparison at the end of
this page).

Because the land reads the atmosphere's boundary file
(`${atmosphere.forcing.file}`), `land=slab` composes beside an atmosphere that
has one (`forcing=from_file`, as in every configuration with land). Beside one
that does not, name a file with `land.land_clim_file=<file>`.

From Python the same two things are explicit. The atmosphere is built with
`jcm.physics.speedy.speedy_terms.speedy_physics(land_model="prescribed")`
passed as `jcm.model.Model(..., physics=...)`, and a `SlabLandModel` is
registered as `"lnd"` ({doc}`../python_api` has a two-step example).

**The pairing is checked.** `jem.exchangers.check_land_coupling`, called by
`default_exchanges` and by `VerosExchange`, raises a `ValueError` before
anything is integrated for a `SlabLandModel` under an atmosphere that
integrates its own land temperature ("Two lands would integrate the land
temperature ..."). `JCMComponent.integrates_land_temperature` is that
property: true when any term of the atmosphere's physics has a `land_model`
other than `"prescribed"`, which is SPEEDY's default slab land and ECHAM's
SLIM, and read off the physics' static attributes, so nothing is integrated to
find out. The same function refuses any land under a physics with no land flux
tile (ECHAM delivers its fluxes untiled), since it would otherwise be charged
with the grid-cell mean, the sea's share included.

**The run says which land it has.** `jem.runners.build_coupler` logs, at
INFO, `Land temperature: ...` with the phrase
`jem.exchangers.land_temperature_source` returns: "evolved by the
atmosphere's own land, inside its physics.", "evolved by the land component
(SlabLandModel); the atmosphere's own land is prescribed and reads it.", or,
for an atmosphere whose land is prescribed with no land component,
"prescribed: the atmosphere's boundary condition, which neither its physics
nor a land component evolves."

## The equations

The model integrates every cell with any land (`SlabGrid.land_mask`,
`fractional_mask > 0`), forced by the atmosphere's flux over the land part of
the cell. Other cells hold `MASKED_SURFACE_TEMPERATURE` and zero soil water,
snow, vegetation and albedo.

**Temperature.** A single slab whose heat capacity mixes soil and glacier ice
by the glacier fraction `f` (`params.glacier_fraction`, a scalar or a map):

    C = (1 − f) · soil_thermal_depth · soil_volumetric_heat_capacity
        + f · glacier_depth · glacier_volumetric_heat_capacity

with the defaults 1 m and 2.5×10⁶ J m⁻³ K⁻¹ for soil, 5 m and 1.93×10⁶ J m⁻³
K⁻¹ for ice, which are SPEEDY's (`depth_soil`, `hcap_soil`, `depth_lice`,
`hcap_lice`). The slab is warmed by the heat flux into it, `F = −total_heat_flux`
(the coupler's flux is positive upward). Run freely it steps

    T_new = T + dt · F / C.

With `land_surface_temperature_from_clim_file=True` it follows SPEEDY's
scheme: the anomaly `T' = T − T_clim` is forced by `F` and damped on
`params.temperature_relaxation_timescale` `tau` (40 days), explicit in the
flux and implicit in the relaxation,

    T'_new = tau / (tau + dt) · (T' + dt · F / C),
    T_new  = T'_new + T_clim(end of step).

The relaxation is a source and a sink of energy of `C / tau` per kelvin of
anomaly, as it is in SPEEDY's land. Its step is the coupling step, where
SPEEDY's is the physics step, so the two schemes agree in the limit of a step
short against `tau` and not otherwise.

**Snow.** The snow depth (mm of water) is the climatology at the current time
with `snow_depth_from_clim_file=True`, and otherwise stays at its initial
value. Cover is diagnosed, `min(1, depth / params.snow_depth_to_cover_scale)`
with the scale 60 mm (SPEEDY's `sd2sc`). jax-gcm stores `snowc` as depth over
that scale, so a climatology from its files is read with
`snow_depth_units_in_clim_file="cover"` and multiplied by the scale.

**Soil water.** The two-layer bucket of {doc}`bucket_land`: precipitation,
evaporation shared between the layers, implicit drainage, overflow, all by
`step_soil_water`, which `BucketLandModel` calls as well. The soil water
parameters are the bucket's (`SlabLandParameters` extends
`BucketLandParameters`), with the bucket's defaults: drainage time scales of
(10, 365) days, and a soil that starts at `initial_soil_water_fill` of its
range, or, with `initialize_from_clim_file=True`, from the availability
climatology of the start month. The deep layer's vegetation weighting is this
model's `vegetation_fraction`, a field of its state. Water is conserved to
rounding per unit of land area; there is no snow budget, so sublimation comes
out of the soil too.

**Soil water availability.** The bucket's diagnosis, SPEEDY's `soilw_am` (the
formula is on {doc}`bucket_land`), or the climatology with
`soil_water_availability_from_clim_file=True`. A prescribed availability does
not feed back on the soil water: the bucket runs either way, and over a soil
that is drier than the prescribed availability says, the atmosphere evaporates
water the soil does not hold, which is `derived.unmet_evaporation`.

**Vegetation fraction.** `params.vegetation_fraction` (0.8, uniform, or a map)
at the start; with `vegetation_fraction_from_clim_file=True` the climatology
at the current time. It weights the albedo blend and the deep layer's share of
the soil water.

**Albedo.** The bare soil and vegetation albedos blended by the vegetation
fraction, `baresoil_albedo (1 − v) + vegetation_albedo v` (0.2 and 0.1), or the
climatology with `surface_albedo_from_clim_file=True`.

Every part that can be read from the climatology file has its own
`*_from_clim_file` flag and its own variable name argument, and a field that is
missing from the file (or every field, with no file) falls back to a stated
idealised value: an idealised temperature, no snow, an availability of 0.5, an
albedo of 0.2, a vegetation fraction of 0.8. A fallback for a part the model
was asked to read from the file is a warning. A variable with no time
dimension in the file is the same map in every month
(`load_monthly_climatology`); `glacier_fraction_from_clim_file=True` reads a
time-invariant map (`load_static_field`) into the parameters. For a boundary
file that carries a background albedo and no glacier map,
`glacier_fraction_from_albedo=True` derives the map instead: a land cell
whose annual-mean albedo (the variable `surface_albedo_varname_in_clim_file`
names) is at or above `params.glacier_albedo_threshold` (0.4) is all glacier,
and the rest all soil. The component logs at construction which mode each
part runs in.

## The carry

`LandState`, `LandForcing` and `LandDerived`, as `@tree_math.struct`s; the
parameters travel in `carry["params"]` and every numeric one is a
differentiable leaf. The grid is the atmosphere's, so no regridding is
involved.

| section | field | meaning |
| --- | --- | --- |
| state | `land_surface_temperature` | K |
| state | `snow_depth` | mm of water |
| state | `snow_cover` | fraction of the land, diagnosed from the depth |
| state | `soil_water_content` | `(lon, lat, 2)`, m³ m⁻³, surface and deep layer |
| state | `vegetation_fraction` | fraction of the land |
| forcing | `total_heat_flux` | net heat flux over the land part, W m⁻², positive upward |
| forcing | `precipitation` | kg m⁻² s⁻¹, positive down; one value per cell |
| forcing | `evaporation` | over the land part, kg m⁻² s⁻¹, positive up |
| derived | `surface_albedo` | output only |
| derived | `soil_water_availability` | SPEEDY's `soilw_am` |
| derived | `deep_drainage`, `soil_water_overflow`, `runoff`, `unmet_evaporation` | per unit land area, kg m⁻² s⁻¹; the bucket's, with the same meanings |
| derived | `runoff_mass_rate` | the runoff over the whole grid, kg s⁻¹ |

The derived fields have the names and meanings of the bucket's, so an exchange
that serves a bucket land serves this land too. The runoff leaves as
`runoff_mass_rate`, the route a Veros ocean takes (`ocn.forcing.runoff_mass_rate`,
spread over its wet surface); a slab ocean carries no salinity and has no
freshwater budget, so with one the runoff leaves the coupled model. The land
needs a grid with cell areas for it (`SlabGrid.cell_solid_angle`).

## What crosses the exchange

A `SlabLandModel` registered as `"lnd"` adds `jem.exchangers.SLAB_LAND_EXCHANGES`
to either standard table (slab or Veros), on top of the three soil-water rows
every land has ({doc}`bucket_land`):

| source | destination |
| --- | --- |
| `atm.derived.land_total_heat_flux` | `lnd.forcing.total_heat_flux` |
| `lnd.state.land_surface_temperature` | `atm.forcing.stl_am` |
| `lnd.state.snow_cover` | `atm.forcing.snowc_am` |

`jem.fluxes.VerosExchange` makes the same rows, read off the land's carry (its
forcing has a `total_heat_flux`), and a Veros configuration has to list
`stl_am`, `snowc_am` and `soilw_am` in `coupling.exchanged_forcing`, since
that exchanger cannot be read off as a table can. The atmosphere fields the
rows write (`stl_am`, `snowc_am`, `soilw_am`) are declared to the atmosphere
with `exchanged_fields`, like every field a table writes into it.

**The tile identity.** The atmosphere's `JCMDerived.land_total_heat_flux` is
the net surface heat flux over its `land` tile, in JEM's convention (positive
upward, so jax-gcm's `net_heat_flux`, positive down, is negated once, at the
boundary). On a coastal cell it adds up with the sea part's:

    sea_fraction · sea_total_heat_flux + (1 − sea_fraction) · land_total_heat_flux
        = total_heat_flux,

so the ocean takes the sea part, the land takes the land part, and no part of
the cell's energy is applied twice or dropped. The land is charged with the
land tile and never the grid-cell mean, for the same reason the soil water is
charged with the land tile's evaporation: on a coastal cell the mean contains
the sea's share. A physics that publishes no land tile leaves the field at the
grid mean, and `check_land_coupling` refuses to force a land with it.

Water works the same way. The soil takes the cell's precipitation and the land
tile's evaporation, and its budget closes per unit of land area to rounding,
`rho_w Σ d·ΔW / dt = P − (E − unmet) − deep_drainage − overflow`; what it
could not supply is reported as `unmet_evaporation` rather than clipped away.
The exchange is lagged by one coupling step like every other
({doc}`exchange`): the atmosphere sees the temperature and snow cover the land
had at the end of its previous step, and the land is forced by the flux the
atmosphere published at the end of its own.

## How it differs from the atmosphere's own land

With the same climatology and the same relaxation, the thermal part is
SPEEDY's equation. It differs in three ways a reader comparing the two should
know:

- **Where it integrates.** SPEEDY's slab holds its anomaly at zero on cells
  whose land fraction is below `flandmin` (1/3), so those cells keep the
  climatology. `SlabLandModel` integrates every cell with any land.
- **Land ice.** SPEEDY's slab takes the ice heat capacity where the background
  albedo `alb0` is at or above 0.4. `SlabLandModel` takes it by
  `params.glacier_fraction` (default 0, every cell soil), a scalar or a map
  read from the climatology file. `glacier_fraction_from_albedo=True` builds
  that map by SPEEDY's rule, and `land=slab` sets it: land ice has about four
  times the heat capacity of soil, and without the map Greenland and
  Antarctica end a 30-day January run 11 K warmer than under the built-in
  land.
- **The coupling step.** The relaxation is applied once per coupling step, not
  once per physics step.

What the atmosphere's own land does not evolve, this one does: the soil water,
through the bucket's code. `land=bucket` keeps SPEEDY's temperature with that
same soil water, so the choice between `bucket` and `slab` is the choice of
which land evolves the temperature.

## What it does not do

- **No snow budget.** The snow depth is a climatology or a constant.
  Precipitation never builds snow and snow never melts into the soil;
  sublimation is taken from the soil water.
- **The albedo is output only.** `derived.surface_albedo` is written, and
  nothing reads it: the atmosphere keeps its own background albedo (`alb0`).
- **The vegetation fraction is state that nothing evolves.** It is held, or
  follows the climatology.
- **No rivers.** Runoff is a mass rate over the grid, as for the bucket.
- **SPEEDY physics only**, and only with SPEEDY's own land prescribed: the
  pairing with ECHAM is refused, and so is the pairing with an atmosphere that
  integrates its own land.

## Output

The land writes its state, its derived fields and the forcing it received with
the `forcing_` prefix and the `jem_role` attribute ({doc}`output`), and the
glacier fraction as a time-independent coordinate. The land tile's fluxes are
written as `forcing_land_total_heat_flux` and `forcing_land_evaporation` (the
bucket writes `forcing_land_evaporation` too), not `forcing_total_heat_flux` and
`forcing_evaporation`: a Winton sea ice is forced by the same fluxes over the
sea part and writes those two names, and the datasets of the two components
must merge. Precipitation, one value per cell, is `forcing_precipitation`.

## A 30-day comparison with the built-in land

Four runs of `earth-slab` from 1 January, 30 days each, differing only in the
land: the built-in land (`land=none`), `land=bucket`, `land=slab`, and
`land=slab` with the soil-water availability prescribed
(`+land.soil_water_availability_from_clim_file=true`). The land temperature is
the one the atmosphere's land tile saw (`surface_flux.tsfc`), on the 1253
cells that are all land, as a 30-day mean difference from the built-in run.
The fluxes are 30-day means over the land part of every cell.

| run | land evaporation | land sensible heat flux | land temperature, mean difference | RMS over cells |
| --- | --- | --- | --- | --- |
| built-in (`land=none`) | 1.35 mm/day | 8.1 W m-2 | | |
| `land=slab`, availability prescribed | 1.32 mm/day | 8.9 W m-2 | +0.11 K | 0.44 K |
| `land=bucket` | 0.80 mm/day | 17.9 W m-2 | +1.38 K | 2.65 K |
| `land=slab` | 0.79 mm/day | 18.3 W m-2 | +1.45 K | 2.73 K |

Two things follow.

**With the same soil water, the two lands agree.** The thermal slab moved from
the atmosphere to the component, stepped once per coupled day and not every 30
minutes, stays within 0.1 K of the built-in land on average: +0.19 K (RMS
0.43 K) over the 794 soil cells and -0.46 K (RMS 0.50 K) over the 459 land-ice
cells. In that run the bucket still evolves its own soil water without the
atmosphere reading it, and cannot supply 0.35 mm/day of the evaporation the
prescribed availability asks of it (`derived.unmet_evaporation`).

**What separates `land=slab` from the built-in land is the soil water, not the
temperature.** `land=slab` and `land=bucket` are within 0.1 K of each other
and both 1.4 K warmer than the built-in land, because the bucket dries: its
availability falls from 0.60 to 0.46 over the 30 days, the land evaporates 40%
less, and the surplus leaves as sensible heat. That is the bucket's own
behaviour ({doc}`bucket_land`, "Tuning the drainage"), which both options
share.

One month from one start date is a check that the coupling is right, not a
climatology: it says nothing about the seasonal cycle or about drift.
