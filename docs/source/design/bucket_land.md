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
3. **Drainage** of each layer's water above its wilting point on its time
   scale `tau = (10, 365)` days: the surface layer into the deep one
   (percolation), the deep layer out of the model. It is stepped implicitly
   in the drained water, `e_new = e / (1 + dt / tau)`, so it is stable and
   never drains a layer below its wilting point, whatever the ratio of the
   coupling step to `tau`.
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
has none; they are tuned below.

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

## Tuning the drainage

The two time scales do different jobs. The surface one sets how fast rain
reaches the deep layer; the deep one how long the deep layer keeps it. They
were tuned against the atmosphere's own climatology: 90-day `earth-slab` runs
from 1 January with `land=bucket` (the soil starting from the January
`soilw_am`), against a control with the prescribed `soilw_am` (`land=none`).
Both start the atmosphere from isothermal rest, so the table compares days
30–90, after its first month. Land means (land-fraction weighted), fluxes in
mm/day, the tropics 20°S–20°N:

| tau (days) | availability | E | P | runoff | of which overflow | tropical E | tropical T_sfc (K) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control (prescribed) | 0.61 (climatology) | 1.10 | 2.34 | | | 2.85 | 302.1 |
| 10, 10 | 0.27 | 0.58 | 2.32 | 1.63 | 0.96 | 1.50 | 306.7 |
| 30, 30 | 0.40 | 0.57 | 2.22 | 1.65 | 1.33 | 1.35 | 306.9 |
| 60, 60 | 0.46 | 0.55 | 2.27 | 1.66 | 1.48 | 1.29 | 307.1 |
| 120, 120 | 0.49 | 0.54 | 2.17 | 1.65 | 1.56 | 1.21 | 307.2 |
| 1, 60 | 0.36 | 0.61 | 2.36 | 1.69 | 1.35 | 1.50 | 306.6 |
| 3, 60 | 0.40 | 0.73 | 2.44 | 1.59 | 1.26 | 1.80 | 306.1 |
| 3, 180 | 0.45 | 0.76 | 2.49 | 1.65 | 1.53 | 1.86 | 305.9 |
| 3, 365 | 0.46 | 0.79 | 2.50 | 1.66 | 1.59 | 1.90 | 306.0 |
| 5, 365 | 0.47 | 0.83 | 2.45 | 1.53 | 1.47 | 1.94 | 305.9 |
| **10, 365** (default) | 0.50 | 0.83 | 2.42 | 1.54 | 1.48 | 1.91 | 306.0 |

- **Equal time scales do not work.** Lengthening both raises the
  availability only where the land barely evaporates (the northern winter)
  and leaves the evaporation where it was. The deep layer stays near its
  wilting point (0.19–0.22 against 0.17): it gives up its water to transpiration and its own drainage
  as fast as the surface layer passes water down, and the rain the surface
  layer cannot pass down overflows.
- **The deep layer has to keep its water.** With its drainage on a yearly
  time scale it fills where it rains (0.27 on average, against a field
  capacity of 0.30), and the runoff is mostly overflow, as in a Manabe (1969) bucket. The evaporation then no
  longer depends on the surface time scale between 3 and 10 days; the default
  keeps 10, whose availability is nearest the climatology.

**What tuning cannot fix.** Even tuned, the land evaporates about a quarter
less than with the climatology, and tropical land is 4 K warmer. The deficit is
in the dry season. Over 20°S–0° (the wet season, 8.2 mm/day of rain) the
availability is 0.62 against an observed 0.67; over 0–20°N (the dry season)
it is 0.34 against 0.66. The climatology carries the wet season's water
through the dry season. The bucket cannot: SPEEDY's two layers hold 48 mm
between wilting point and field capacity, which evaporation empties within
weeks however slowly the soil drains. Carrying a season would take a deeper
reservoir below SPEEDY's 28 cm, which the bucket does not have.

Each row is one 90-day realisation from one start date, and the land fields
are 3-daily snapshots, so the differences between the last few rows are
within the weather's noise; those against the control and the first rows are
not.

The bucket conserves water and couples as intended. At the default drainage:
- its budget closes on every land cell to 2×10⁻¹⁰ kg m⁻² s⁻¹ (days 30–40 of
  a run keeping every daily record, with up to 68 mm/day of rain);
- the land receives exactly the atmosphere's land-tile evaporation;
- unmet evaporation is 6×10⁻⁶ of the total over the 90-day run.

## Initial condition

With `soil_water_clim_file` (a monthly `soilw_am`, such as the atmosphere's
own forcing file) the soil starts from the availability observed in the month
the run starts in: both layers are filled to the same fraction of their range,
the one that gives that availability. Without one, both layers start half
full (`initial_soil_water_fill`).
