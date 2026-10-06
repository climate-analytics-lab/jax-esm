"""Tunable parameters of the slab land model."""

import jax.numpy as jnp
from flax import struct

SECONDS_PER_DAY = 86400.0


@struct.dataclass
class SlabBucketLandParameters:
    """Parameters of :class:`~jem.components.slab.slab_bucket_land_model.SlabBucketLandModel`.

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
    depth_soil : jnp.ndarray
        Per-layer soil depth (m) (default: (1.0, 2.0))
    depth_lice : jnp.ndarray
        Depth (m) of the land-ice layer, used instead of the soil layer where
        the surface albedo marks an ice sheet.
    soil_volumetric_heat_capacity : jnp.ndarray
        Volumetric heat capacity of soil (J/m3/K). With ``depth_soil`` it gives
        SPEEDY's ``hcapl``.
    land_ice_volumetric_heat_capacity : jnp.ndarray
        Volumetric heat capacity of land ice (J/m3/K). With ``depth_lice`` it
        gives SPEEDY's ``hcapli``.
    tdland : jnp.ndarray
        Dissipation timescale (s) of the land-temperature anomaly about its
        climatology. Default 40 days.
    flandmin : jnp.ndarray
        Minimum land fraction of a cell for its anomaly to evolve at all;
        cells below it are pinned to the climatology.
    land_threshold : jnp.ndarray
        Land fraction at or above which a cell counts as land for this model.
        Cells below it report :data:`MASKED_SURFACE_TEMPERATURE` and are not
        integrated.
    tau_drain: Per-layer soil moisture drainage timescale in seconds
        (default: (5, 60) days). Layer 0 drains into layer 1; layer 1
        drains out as deep drainage.
    swcap: Per-layer field capacity, unitless 0-1 (default: (0.30, 0.30)).
        Currently unused by the bucket dynamics.
    swwil: Per-layer wilting point, unitless 0-1 (default: (0.17, 0.17)).
        Currently unused by the bucket dynamics.
    snow_depth_to_cover_scale : jnp.ndarray
        Snow depth (mm water equivalent) at which the diagnosed snow cover
        fraction saturates at one. SPEEDY's ``sd2sc``.
    land_ice_albedo_threshold : jnp.ndarray
        Surface albedo at or above which a cell is treated as land ice rather
        than soil.
    surface_albedo : jnp.ndarray
        Surface albedo used when no albedo field is passed to the model. The
        default, 0.2, is below ``land_ice_albedo_threshold`` everywhere, so a
        model built without an albedo field is all soil.

    """

    # The per-layer fields default to arrays, which are unhashable and so are
    # rejected by ``dataclasses`` as a plain default; a factory also gives each
    # instance its own array rather than one shared at class-definition time.
    depth_soil: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((1.0, 2.0)))
    depth_lice: float | jnp.ndarray = 5.0
    soil_volumetric_heat_capacity: float | jnp.ndarray = 2.50e6
    land_ice_volumetric_heat_capacity: float | jnp.ndarray = 1.93e6
    tdland: float | jnp.ndarray = 40.0 * SECONDS_PER_DAY
    flandmin: float | jnp.ndarray = 1.0 / 3.0
    tau_drain: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((60.0 * SECONDS_PER_DAY, 60.0 * SECONDS_PER_DAY)))
    swcap: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((0.30, 0.30)))
    swwil: jnp.ndarray = struct.field(default_factory=lambda: jnp.asarray((0.17, 0.17)))
    land_threshold: float | jnp.ndarray = 0.1
    snow_depth_to_cover_scale: float | jnp.ndarray = 60.0
    land_ice_albedo_threshold: float | jnp.ndarray = 0.4
    surface_albedo: float | jnp.ndarray = 0.2

    @classmethod
    def default(cls) -> "SlabBucketLandParameters":
        """Return the default parameters."""
        return cls()
