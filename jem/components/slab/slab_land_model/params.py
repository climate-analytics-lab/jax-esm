"""Tunable parameters of the slab land model."""

import jax.numpy as jnp
from flax import struct

SECONDS_PER_DAY = 86400.0


@struct.dataclass
class SlabLandParameters:
    """Parameters of :class:`~jem.components.slab.slab_land_model.SlabLandModel`.

    All numeric fields are pytree leaves, so ``jax.grad`` works with respect
    to any of them. They travel in ``carry["params"]`` and are read every
    step.

    The thermal parameters, the relaxation timescale and the snow-cover scale
    follow SPEEDY (``land_model.f90``). SPEEDY prescribes its soil water
    rather than evolving it, so the two-layer bucket's drainage is our own
    choice; its layer depths, field capacity and wilting points are the ones
    SPEEDY uses for its soil water availability.

    Attributes
    ----------
    soil_bucket_depth : jnp.ndarray, shape (2,)
        Thickness (m) of the surface and deep layers of the soil water
        bucket. Default (0.07, 0.21), SPEEDY's soil water layers (``sdep1``
        and ``idep2 * sdep1``). It sets how much water each layer holds and
        does not affect the heat capacity.
    soil_thermal_depth : float
        Thickness (m) of the thermal slab on the soil part of a cell.
        Default 1, SPEEDY's ``depth_soil``.
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
    drainage_timescale : jnp.ndarray, shape (2,)
        Drainage timescale (s) of each soil layer. Default (60, 60) days. The
        surface layer drains into the deep layer, and the deep layer drains
        out of the model. Only used when the bucket is active
        (``soil_water_content_from_clim_file=False``).
    soil_water_field_capacity : jnp.ndarray, shape (2,)
        Field capacity of each layer (m3 m-3). Default (0.30, 0.30), SPEEDY's
        ``swcap``. Used only in the soil water availability, whose denominator
        is the water both layers hold between wilting point and field
        capacity.
    soil_water_wilting_point : jnp.ndarray, shape (2,)
        Wilting point of each layer (m3 m-3). Used only in the soil water
        availability: water below it does not count as available. Default
        (0, 0.17), which reproduces SPEEDY: the deep layer uses SPEEDY's
        ``swwil``, and the surface layer counts all of its water.
    snow_depth_to_cover_scale : float
        Snow depth (mm water equivalent) at which snow cover reaches 1.
        SPEEDY's ``sd2sc``. Default 60.
    baresoil_albedo : float or jnp.ndarray, shape (n_lon, n_lat)
        Albedo of bare soil, in [0, 1]. Either a single value for every cell
        or a map on the model grid. Default 0.2. With
        ``surface_albedo_from_clim_file=False`` the surface albedo is
        ``baresoil_albedo * (1 - vegetation_fraction) + vegetation_albedo *
        vegetation_fraction``. It also fills every month of the idealized
        albedo climatology used when there is no climatology file. The albedo
        is written to the output but does not yet feed back on the land model
        or the atmosphere.
    vegetation_albedo : float or jnp.ndarray, shape (n_lon, n_lat)
        Albedo of vegetation, in [0, 1], blended with ``baresoil_albedo`` by
        the vegetation fraction. Either a single value for every cell or a map
        on the model grid. Default 0.1.
    glacier_fraction : float or jnp.ndarray, shape (n_lon, n_lat)
        Fraction of the land part of each cell covered by glacier, in
        [0, 1]. The slab heat capacity is the soil and glacier capacities
        weighted by this fraction. Either a single value for every cell or
        a map on the model grid, which ``SlabLandModel`` reads from its
        climatology file with ``glacier_fraction_from_clim_file=True``. It
        is constant in time. Default 0, so every cell is all soil.

    """

    # Per-layer defaults are arrays, which dataclasses only accept through a
    # default_factory.
    soil_bucket_depth: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((0.07, 0.21)))
    soil_thermal_depth: float | jnp.ndarray = 1.0
    glacier_depth: float | jnp.ndarray = 5.0
    soil_volumetric_heat_capacity: float | jnp.ndarray = 2.50e6
    glacier_volumetric_heat_capacity: float | jnp.ndarray = 1.93e6
    temperature_relaxation_timescale: float | jnp.ndarray = 40.0 * SECONDS_PER_DAY
    drainage_timescale: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((60.0 * SECONDS_PER_DAY, 60.0 * SECONDS_PER_DAY)))
    soil_water_field_capacity: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((0.30, 0.30)))
    soil_water_wilting_point: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((0.0, 0.17)))
    snow_depth_to_cover_scale: float | jnp.ndarray = 60.0
    baresoil_albedo: float | jnp.ndarray = 0.2
    vegetation_albedo: float | jnp.ndarray = 0.1
    glacier_fraction: float | jnp.ndarray = 0.0

    @classmethod
    def default(cls) -> "SlabLandParameters":
        """Return the default parameters."""
        return cls()
