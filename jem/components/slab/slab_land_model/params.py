"""Tunable parameters of the slab land model."""

import jax.numpy as jnp
from flax import struct

SECONDS_PER_DAY = 86400.0


@struct.dataclass
class SlabLandParameters:
    """Parameters of :class:`~jem.components.slab.slab_land_model.SlabLandModel`.

    Every numeric field is a pytree leaf, so ``jax.grad`` of a coupled run with
    respect to any of them works: the parameters travel in the component's
    carry (``carry["params"]``), not in a closure over the model object.

    Defaults are SPEEDY's (``land_model.f90``).

    Attributes
    ----------
    depth_soil : jnp.ndarray
        Depth (m) of the soil layer that carries the surface heat capacity.
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
    initial_year_fraction : jnp.ndarray
        Position in the annual cycle, in [0, 1), at which the climatologies are
        sampled for the initial condition. The coupler owns the clock and
        ``initialize()`` receives no date, so a run that starts on 1 July sets
        this to 0.5 rather than expecting the component to know.

    """

    depth_soil: float | jnp.ndarray = 1.0
    depth_lice: float | jnp.ndarray = 5.0
    soil_volumetric_heat_capacity: float | jnp.ndarray = 2.50e6
    land_ice_volumetric_heat_capacity: float | jnp.ndarray = 1.93e6
    tdland: float | jnp.ndarray = 40.0 * SECONDS_PER_DAY
    flandmin: float | jnp.ndarray = 1.0 / 3.0
    land_threshold: float | jnp.ndarray = 0.1
    snow_depth_to_cover_scale: float | jnp.ndarray = 60.0
    land_ice_albedo_threshold: float | jnp.ndarray = 0.4
    surface_albedo: float | jnp.ndarray = 0.2
    initial_year_fraction: float | jnp.ndarray = 0.0

    @classmethod
    def default(cls) -> "SlabLandParameters":
        """Return the default parameters."""
        return cls()
