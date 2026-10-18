"""Tunable parameters of the slab land model."""

import jax.numpy as jnp
from flax import struct

SECONDS_PER_DAY = 86400.0


@struct.dataclass
class SlabLandParameters:
    """Parameters of :class:`~jem.components.slab.slab_bucket_land_model.SlabLandModel`.

    Every numeric field is a pytree leaf, so ``jax.grad`` of a coupled run with
    respect to any of them works: the parameters travel in the component's
    carry (``carry["params"]``), not in a closure over the model object.

    They are all **process** parameters, read from the carry every step, so a
    parameter study varies one by replacing that leaf in ``carry["params"]``.
    The land model has no initial-condition parameters: it starts from its
    climatology.

    Defaults are SPEEDY's (``land_model.f90``).

    Attributes
    ----------
    soil_depth : jnp.ndarray
        Per-layer soil depth (m) (default: (1.0, 2.0))
    landice_depth : jnp.ndarray
        Depth (m) of the land-ice layer, used instead of the soil layer where
        the surface albedo marks an ice sheet.
    soil_volumetric_heat_capacity : jnp.ndarray
        Volumetric heat capacity of soil (J/m3/K). With ``soil_depth`` it gives
        SPEEDY's ``hcapl``.
    landice_volumetric_heat_capacity : jnp.ndarray
        Volumetric heat capacity of land ice (J/m3/K). With ``landice_depth`` it
        gives SPEEDY's ``hcapli``.
    temperature_relaxation_timescale : jnp.ndarray
        Dissipation timescale (s) of the land-temperature anomaly about its
        climatology. Default 40 days.
    drainage_timescale: Per-layer soil moisture drainage timescale in seconds
        (default: (5, 60) days). Layer 0 drains into layer 1; layer 1
        drains out as deep drainage.
    soil_water_field_capacity: Per-layer field capacity, unitless 0-1 (default: (0.30, 0.30)).
        Currently unused by the bucket dynamics.
    soil_water_wilting_point: Per-layer wilting point, unitless 0-1 (default: (0.17, 0.17)).
        Currently unused by the bucket dynamics.
    snow_depth_to_cover_scale : jnp.ndarray
        Snow depth (mm water equivalent) at which the diagnosed snow cover
        fraction saturates at one. SPEEDY's ``sd2sc``.
    landice_albedo_threshold : jnp.ndarray
        Surface albedo at or above which a cell is treated as land ice rather
        than soil.
    surface_albedo : jnp.ndarray
        Surface albedo used when no albedo field is passed to the model. The
        default, 0.2, is below ``landice_albedo_threshold`` everywhere, so a
        model built without an albedo field is all soil.

    """

    # The per-layer fields default to arrays, which are unhashable and so are
    # rejected by ``dataclasses`` as a plain default; a factory also gives each
    # instance its own array rather than one shared at class-definition time.
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
