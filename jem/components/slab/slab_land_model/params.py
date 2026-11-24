"""Tunable parameters of the slab land model."""

import jax.numpy as jnp
from flax import struct

from jem.components.slab.bucket_land_model.params import (
    SECONDS_PER_DAY,
    BucketLandParameters,
)


@struct.dataclass
class SlabLandParameters(BucketLandParameters):
    """Parameters of :class:`~jem.components.slab.slab_land_model.SlabLandModel`.

    All numeric fields are pytree leaves, so ``jax.grad`` works with respect
    to any of them. They travel in ``carry["params"]`` and are read every
    step, except the two initial-condition parameters named below.

    The soil water's parameters are the bucket's
    (:class:`~jem.components.slab.bucket_land_model.BucketLandParameters`: the
    layer depths, drainage time scales, field capacities, wilting points,
    ``vegetation_fraction`` and ``initial_soil_water_fill``, with the
    bucket's defaults), because the soil water is the bucket's own code. Two
    of them mean something a little different here:

    - ``drainage_timescale`` keeps the bucket's tuned default of (10, 365)
      days: the deep layer has to hold what percolates into it for the land
      to evaporate near the rate the observed availability gives.
    - ``vegetation_fraction`` is the uniform vegetation fraction the model
      *starts* from (a scalar or a map on the grid), which seeds
      ``state.vegetation_fraction``. After that the state evolves (it follows
      the climatology with ``vegetation_fraction_from_clim_file=True`` and is
      held otherwise), so like ``initial_soil_water_fill`` it is an
      initial-condition parameter: vary it by passing the parameters to
      ``initialize``. Default 0.8.

    The thermal parameters, the relaxation timescale and the snow-cover scale
    follow SPEEDY (``land_model.f90``).

    Attributes
    ----------
    soil_thermal_depth : float
        Thickness (m) of the thermal slab on the soil part of a cell.
        Default 1, SPEEDY's ``depth_soil``. It is separate from the water
        bucket's layers: it sets the heat capacity and nothing about how much
        water the soil holds.
    glacier_depth : float
        Thickness (m) of the thermal slab on the glacier part of a cell.
        Default 5.
    soil_volumetric_heat_capacity : float
        Volumetric heat capacity of soil (J m-3 K-1). Default 2.5e6.
    glacier_volumetric_heat_capacity : float
        Volumetric heat capacity of glacier ice (J m-3 K-1). Default 1.93e6.
    temperature_relaxation_timescale : float
        Damping timescale (s) of the land surface temperature anomaly about
        its climatology. Default 40 days. Only used when the model relaxes to
        climatology (``land_surface_temperature_from_clim_file=True``).
    snow_depth_to_cover_scale : float
        Snow depth (mm water equivalent) at which snow cover reaches 1.
        SPEEDY's ``sd2sc``. Default 60. With
        ``snow_depth_units_in_clim_file="cover"`` it also turns the file's
        values into a depth.
    baresoil_albedo : float or jnp.ndarray
        Albedo of bare soil, in [0, 1]. Either a single value for every cell
        or a map on the model grid, ``(n_lon, n_lat)``. Default 0.2. With
        ``surface_albedo_from_clim_file=False`` the surface albedo is
        ``baresoil_albedo * (1 - vegetation_fraction) + vegetation_albedo *
        vegetation_fraction``. It also fills every month of the idealized
        albedo climatology used when there is no climatology file. The albedo
        is written to the output but does not yet feed back on the land model
        or the atmosphere.
    vegetation_albedo : float or jnp.ndarray
        Albedo of vegetation, in [0, 1], blended with ``baresoil_albedo`` by
        the vegetation fraction. Either a single value for every cell or a map
        on the model grid, ``(n_lon, n_lat)``. Default 0.1.
    glacier_fraction : float or jnp.ndarray
        Fraction of the land part of each cell covered by glacier, in
        [0, 1]. The slab heat capacity is the soil and glacier capacities
        weighted by this fraction. Either a single value for every cell or
        a map on the model grid, ``(n_lon, n_lat)``, which ``SlabLandModel`` reads from its
        climatology file with ``glacier_fraction_from_clim_file=True`` or
        derives from its background albedo with
        ``glacier_fraction_from_albedo=True``. It is constant in time.
        Default 0, so every cell is all soil.
    glacier_albedo_threshold : float
        Background albedo at or above which a land cell is glacier, read once
        at construction and only with ``glacier_fraction_from_albedo=True``.
        Default 0.4, SPEEDY's test for land ice (jax-gcm's
        ``LandModelParameters.albice_threshold``). It selects the glacier map
        and is not read in ``step``, so a run's gradient with respect to it
        is zero: the map itself, ``glacier_fraction``, is the differentiable
        quantity.

    """

    soil_thermal_depth: float | jnp.ndarray = 1.0
    glacier_depth: float | jnp.ndarray = 5.0
    soil_volumetric_heat_capacity: float | jnp.ndarray = 2.50e6
    glacier_volumetric_heat_capacity: float | jnp.ndarray = 1.93e6
    temperature_relaxation_timescale: float | jnp.ndarray = 40.0 * SECONDS_PER_DAY
    snow_depth_to_cover_scale: float | jnp.ndarray = 60.0
    baresoil_albedo: float | jnp.ndarray = 0.2
    vegetation_albedo: float | jnp.ndarray = 0.1
    glacier_fraction: float | jnp.ndarray = 0.0
    glacier_albedo_threshold: float | jnp.ndarray = 0.4

    @classmethod
    def default(cls) -> "SlabLandParameters":
        """Return the default parameters."""
        return cls()
