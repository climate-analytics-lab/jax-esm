# The slab land's soil water

`SlabLandModel` is SPEEDY's land surface: a slab whose temperature is heated by the surface heat flux and, optionally, relaxed to a climatology. Next to it runs a two-layer soil water bucket that turns the atmosphere's precipitation and evaporation into a soil water availability for the atmosphere and runoff for the ocean. This page records how the bucket is built and why. The equations themselves are in the class docstring.

## Layers and their defaults

The bucket has a surface layer of 7 cm and a deep layer of 21 cm (`soil_bucket_depth`), the depths SPEEDY's own `soilw_am` is defined over. Each layer holds a volumetric water content between its wilting point and its field capacity (`soil_water_wilting_point = (0, 0.17)`, `soil_water_field_capacity = (0.30, 0.30)`, SPEEDY's `swwil` and `swcap`). The thermal slab is a separate depth, `soil_thermal_depth = 1 m`, SPEEDY's `depth_soil`: the water depths and the heat capacity are different things and are tuned separately.

## One step

Within a coupling step the processes are applied in turn:

1. **Precipitation** goes into the surface layer.
2. **Evaporation** is taken from both layers, in proportion to the water each can supply: `a1 = d1 * max(W1 - wp1, 0)` and `a2 = vegetation_fraction * d2 * max(W2 - wp2, 0)`. These are the same terms as the soil water availability, so the deep layer gives water up only through vegetation, and the split follows SPEEDY's definition of what is available. If neither layer has any, all of it is asked of the surface layer. Dew goes into the surface layer.
3. **Drainage**: each layer drains `max(W - wp, 0) / tau` into the one below, and the deep layer's drainage leaves the model (`derived.deep_drainage`). Water at the wilting point is held by the soil.
4. **Overflow**: each layer is capped at its field capacity, and the excess leaves the model directly (`derived.soil_water_overflow`) without passing through the layer below.

Evaporation runs before drainage so that this step's rain can be evaporated before it drains away, which keeps the unmet evaporation (below) small.

## The wilting point is a floor

No layer is taken below its wilting point by evaporation. What a layer's share of the evaporation cannot supply is published as `derived.unmet_evaporation`, rather than clamped away: the atmosphere evaporated that water anyway, so it is water the coupled system created, and publishing it keeps the land's budget closed and the gap visible. The budget

    P - (E - unmet_evaporation) - deep_drainage - soil_water_overflow = d(water)/dt

holds to round-off in every step, which `tests/unit/test_slab_land_model.py` checks for wet, dry, saturated and dew conditions.

The floor is applied to the soil water itself, and so it assumes no layer starts below its wilting point: a layer that did would be lifted to it, and the water that took would be booked as unmet evaporation. `initialize` therefore starts each layer at its wilting point, the driest state the bucket allows. Nothing in the step can then take a layer below it: precipitation only adds water, drainage stops at the wilting point and the overflow only removes water above field capacity. A state loaded from elsewhere (an older checkpoint, a hand-set carry, a changed wilting point) is the caller's to keep at or above it.

## Soil water availability

The availability the atmosphere reads (`derived.soil_water_availability`, sent to JCM's `soilw_am`) is SPEEDY's:

    (d1 * max(W1 - wp1, 0) + vegetation_fraction * d2 * max(W2 - wp2, 0)) / (d1 * (fc1 - wp1) + d2 * (fc2 - wp2))

clipped to [0, 1] and zero off land. With `soil_water_availability_from_clim_file=True` it is the climatology instead, which is how SPEEDY itself runs (`land=slab_speedy`). The bucket still runs and is written out then, but it does not feed back on what the atmosphere sees, and a prescribed wet availability over a dry bucket shows up as unmet evaporation.

## Vegetation and albedo

The vegetation fraction is a uniform 0.8 or, with `vegetation_fraction_from_clim_file=True`, the climatology. It weights the deep layer's part of the availability and the evaporation split, and it blends the albedo, `baresoil_albedo * (1 - f) + vegetation_albedo * f`. The albedo is written out but nothing in the coupling reads it yet.

## Runoff

The deep drainage and the overflow together are the land's runoff. `runoff_to_ocean_mapping` routes it to ocean cells, on the land's own grid: it is an ESMF weight file or a function, and `derived.runoff_to_ocean` is the routed field. The standard coupling copies it to the ocean's `forcing.runoff` (see {doc}`exchange`), and the slab ocean counts it in its salinity budget against E - P. Without a mapping the total runoff is spread evenly over the cells with no land: the area-weighted total is kept and handed out as one uniform flux, so the ocean receives all of it, and none lands on a coast cell where it would be shared with the land part. Cell areas are taken as proportional to cos(latitude), exact on a regular lon/lat grid and close on SPEEDY's Gaussian grid. This conserves the water but puts it in the wrong place: a river-routing map belongs in `runoff_to_ocean_mapping`.

## Partly-land cells

The atmosphere publishes grid-cell means, and the land runs on every cell with any land (`grid.binary_land_mask`). In a coast cell the bucket therefore receives the cell-mean precipitation and evaporation, and the ocean in the same cell receives them too. Weighting the two surfaces by the land fraction is not done yet.
