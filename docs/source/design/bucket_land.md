# The bucket land

`BucketLandModel` (`land=bucket`) evolves the land's soil water. It does not
evolve the land's temperature: that is the atmosphere's own. jax-gcm
integrates SPEEDY's slab land (`land_model.f90`) inside its surface-flux term,
forced by the flux over the land part of each cell, and a second thermal slab
in JAX-ESM would integrate the same equation twice. What SPEEDY does not
evolve is its soil water: its soil water availability `soilw_am` is a
prescribed climatology, so the land's evaporation cannot respond to the land
drying out. The bucket closes that loop. It was written by Tien-Yiao Hsu as
the soil-water half of JAX-ESM's former thermal slab land (jax-esm#154).

## The equations

Two layers of depth `d = (0.07, 0.21)` m hold a volumetric water content `W`
between a wilting point `wp = (0, 0.17)` and a field capacity
`fc = (0.30, 0.30)`: SPEEDY's soil layers (`sdep1`, `idep2 · sdep1`), its
`swwil` for the deep layer and its `swcap`. A coupling step of length `dt`
applies, in turn:

1. **Precipitation** `P` into the surface layer.
2. **Evaporation** `E`, shared between the layers in proportion to the water
   each can give, `d1 max(W1 − wp1, 0)` and `veg d2 max(W2 − wp2, 0)`: the
   deep layer only through vegetation, whose roots reach it. Dew (`E < 0`)
   goes into the surface layer. No layer is taken below its wilting point;
   what it cannot give is `derived.unmet_evaporation`.
3. **Drainage** of each layer's water above its wilting point on a
   10-day time scale `tau`: the surface layer into the deep one, the deep
   layer out of the model. It is stepped implicitly in the drained water,
   `e_new = e / (1 + dt / tau)`, so it is stable and never drains a layer
   below its wilting point, whatever the ratio of the coupling step to `tau`.
4. **Overflow**: each layer is capped at its field capacity and the excess
   leaves the model.

Per unit of land area, water is conserved to rounding:
`rho_w Σ d·ΔW / dt = P − (E − unmet) − deep_drainage − overflow`.

The availability handed to the atmosphere is SPEEDY's own quantity,

    soilw_am = (d1 max(W1 − wp1, 0) + veg d2 max(W2 − wp2, 0))
               / (d1 (fc1 − wp1) + d2 (fc2 − wp2)),

which with SPEEDY's depths is `rsw (W1 + veg idep2 max(W2 − swwil, 0))`,
`rsw = 1 / (swcap + idep2 (swcap − swwil))`. `veg` is a uniform 0.8 by
default, a stand-in for SPEEDY's `vegh + 0.8 vegl`, which jax-gcm's packaged
T30 boundary conditions do not carry; a map can be passed instead. The drainage
time scale is this model's own choice: SPEEDY prescribes its soil water, so it
has none.

## What it is forced with

The coupling is three rows of the standard table, all on the atmosphere's own
grid, so no regridding is involved:

| source | destination |
| --- | --- |
| `atm.derived.precipitation` | `lnd.forcing.precipitation` |
| `atm.derived.land_evaporation` | `lnd.forcing.evaporation` |
| `lnd.derived.soil_water_availability` | `atm.forcing.soilw_am` |

Precipitation is one field over the cell, so the land takes the grid value.
Evaporation is the atmosphere's **land tile**, not the grid-cell mean: on a
coastal cell the mean includes the sea's evaporation, which the soil would
otherwise have to supply. `JCMDerived.land_evaporation` reads it by tile name
from jax-gcm's `SurfaceExchange` (see {doc}`jcm_adapter`). A physics package
that publishes no land flux tile (ECHAM delivers its fluxes untiled) cannot
force the bucket faithfully, so `default_exchanges` refuses the pairing rather
than charging the soil with the cell mean.

Over a step the atmosphere evaporates at the availability it was handed at the
start, so on a soil that dries within one coupling step it can ask for more
than the soil holds. That excess is `unmet_evaporation`: water the coupled
system created, reported per cell rather than clipped away, so the size of the
coupling-lag error is visible.

## Where the runoff goes

Deep drainage and overflow leave the land as `derived.runoff` (per unit land
area) and, integrated over the grid with the cells' true solid angles
(`SlabGrid.cell_solid_angle`, the atmosphere's Gaussian quadrature weights),
as `derived.runoff_mass_rate` in kg s⁻¹. There are no rivers: a Veros ocean
takes the mass rate (`ocn.forcing.runoff_mass_rate`) and spreads it uniformly
over its own wet surface, so the water the land loses is the water the ocean
gains, whatever the two grids. A slab ocean carries no salinity, so with one
the runoff leaves the coupled model and `runoff_mass_rate` is a diagnostic.

## What it does to the climate: the soil dries

The bucket conserves water and couples as intended. In a 60-day
`earth-slab` + `land=bucket` run from 1 January:
- its budget closes to 1×10⁻¹⁰ kg m⁻² s⁻¹ per cell;
- the land receives exactly the atmosphere's land-tile evaporation;
- unmet evaporation is 4×10⁻⁵ of the total.

But at the default 10-day drainage the soil empties. The runoff, 1.2–1.9 mm/day
over land, nearly equals the precipitation, and the land-mean availability
falls from the observed 0.65 to 0.21 by day 30 and 0.29 by day 60, against a
climatology of 0.63–0.65 for those months. The atmosphere responds: over days
21–30 the land evaporates 0.45 mm/day against 1.00 mm/day with the
climatological soil, and receives 1.60 against 2.01 mm/day of precipitation.

The drainage time scale is the lever. It is this model's own choice, since
SPEEDY prescribes its soil water and has none, and it has not been tuned
against observed soil moisture. Tune `drainage_timescale` before using the
bucket for a climate run.

## Initial condition

With `soil_water_clim_file` (a monthly `soilw_am`, such as the atmosphere's
own forcing file) the soil starts from the availability observed in the month
the run starts in: both layers are filled to the same fraction of their range,
the one that gives that availability. Without one, both layers start half
full (`initial_soil_water_fill`).
