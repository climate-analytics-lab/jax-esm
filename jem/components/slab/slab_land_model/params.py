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
    follow SPEEDY (``land_model.f90``). The two-layer soil water bucket is not
    part of SPEEDY, and its defaults are our own choice.

    Attributes
    ----------
    soil_depth : jnp.ndarray, shape (2,)
        Thickness (m) of the surface and deep soil layers. Default (1, 2).
        The surface layer is also the thermal slab, so ``soil_depth[0]``
        sets the soil heat capacity.
    landice_depth : float
        Thickness (m) of the thermal slab on land ice. Default 5.
    soil_volumetric_heat_capacity : float
        Volumetric heat capacity of soil (J m-3 K-1). Default 2.5e6.
    landice_volumetric_heat_capacity : float
        Volumetric heat capacity of land ice (J m-3 K-1). Default 1.93e6.
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
        Field capacity of each layer (m3 m-3). Default (0.30, 0.30). Not used
        yet.
    soil_water_wilting_point : jnp.ndarray, shape (2,)
        Wilting point of each layer (m3 m-3). Default (0.17, 0.17). Not used
        yet.
    snow_depth_to_cover_scale : float
        Snow depth (mm water equivalent) at which snow cover reaches 1.
        SPEEDY's ``sd2sc``. Default 60.
    landice_albedo_threshold : float
        Cells with surface albedo at or above this value are treated as land
        ice. Default 0.4.
    surface_albedo : float
        Uniform surface albedo used when the model is built without an albedo
        field. Default 0.2, which is below ``landice_albedo_threshold``, so
        such a model has no land ice.

    """

    # Per-layer defaults are arrays, which dataclasses only accept through a
    # default_factory.
    soil_depth: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((1.0, 2.0)))
    landice_depth: float | jnp.ndarray = 5.0
    soil_volumetric_heat_capacity: float | jnp.ndarray = 2.50e6
    landice_volumetric_heat_capacity: float | jnp.ndarray = 1.93e6
    temperature_relaxation_timescale: float | jnp.ndarray = 40.0 * SECONDS_PER_DAY
    drainage_timescale: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((60.0 * SECONDS_PER_DAY, 60.0 * SECONDS_PER_DAY)))
    soil_water_field_capacity: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((0.30, 0.30)))
    soil_water_wilting_point: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((0.17, 0.17)))
    snow_depth_to_cover_scale: float | jnp.ndarray = 60.0
    landice_albedo_threshold: float | jnp.ndarray = 0.4
    surface_albedo: float | jnp.ndarray = 0.2

    @classmethod
    def default(cls) -> "SlabLandParameters":
        """Return the default parameters."""
        return cls()
