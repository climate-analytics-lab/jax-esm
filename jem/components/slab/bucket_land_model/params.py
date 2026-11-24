"""Tunable parameters of the bucket land model."""

import jax.numpy as jnp
from flax import struct

SECONDS_PER_DAY = 86400.0


@struct.dataclass
class BucketLandParameters:
    """Parameters of :class:`~jem.components.slab.bucket_land_model.BucketLandModel`.

    Every field is a pytree leaf, so ``jax.grad`` reaches any of them. They
    travel in ``carry["params"]``. ``initial_soil_water_fill`` is an
    initial-condition parameter, read once by
    :meth:`~jem.components.slab.bucket_land_model.BucketLandModel.initialize`
    and only where no soil-water-availability climatology was given; the
    others are read every step.

    The layer depths, field capacity, deep wilting point and vegetation
    weighting are the ones SPEEDY diagnoses its soil water availability
    ``soilw_am`` with (``sdep1``, ``idep2 * sdep1``, ``swcap``, ``swwil``), so
    the availability this model hands the atmosphere is SPEEDY's own
    quantity. SPEEDY prescribes its soil water rather than evolving it, so the
    drainage time scales are this model's own choice.

    Attributes
    ----------
    soil_bucket_depth : jnp.ndarray
        Shape ``(2,)``: Depth (m) of the surface and deep layer. Default (0.07, 0.21).
    drainage_timescale : jnp.ndarray
        Shape ``(2,)``: e-folding time (s) on which each layer's water above its wilting point
        drains: the surface layer into the deep one (percolation), the deep
        layer out of the model. Default (10, 365) days: the deep layer has to
        keep what percolates into it for the land to evaporate near the rate
        the observed availability gives, so its drainage is slow and the
        runoff is mostly overflow. Tuned in 90-day Earth-like runs against
        the prescribed climatology (``docs/source/design/bucket_land.md``,
        "Tuning the drainage").
    soil_water_field_capacity : jnp.ndarray
        Shape ``(2,)``: Volumetric water content (m3 m-3) each layer holds at most; what
        rises above it leaves as overflow. Default (0.30, 0.30).
    soil_water_wilting_point : jnp.ndarray
        Shape ``(2,)``: Volumetric water content (m3 m-3) below which a layer's water neither
        drains nor evaporates. Default (0, 0.17): SPEEDY counts all of the
        surface layer's water and the deep layer's above ``swwil``.
    vegetation_fraction : float or jnp.ndarray
        Fraction of the land covered by vegetation, a scalar or a map on the
        model grid. The deep layer gives up water, and counts towards the
        availability, only through vegetation, whose roots reach it. Default
        0.8: a uniform stand-in for SPEEDY's ``veg = vegh + 0.8 vegl``, which
        the packaged T30 boundary conditions do not carry.
    initial_soil_water_fill : float or jnp.ndarray
        Where each layer starts between its wilting point (0) and field
        capacity (1), when there is no climatology to start from. Default 0.5.

    """

    soil_bucket_depth: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.asarray((0.07, 0.21)))
    drainage_timescale: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.asarray((10.0 * SECONDS_PER_DAY, 365.0 * SECONDS_PER_DAY)))
    soil_water_field_capacity: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.asarray((0.30, 0.30)))
    soil_water_wilting_point: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.asarray((0.0, 0.17)))
    vegetation_fraction: float | jnp.ndarray = 0.8
    initial_soil_water_fill: float | jnp.ndarray = 0.5

    def __post_init__(self) -> None:
        """Take a per-layer pair given as a list, as Hydra hands one over.

        ``+land.params.drainage_timescale=[a,b]`` reaches the constructor as
        a Python list, which would make the pair two leaves of the pytree
        instead of the one array the default is, and change what
        ``jax.grad`` returns. Only a list or tuple is converted: the arrays
        and tracers JAX rebuilds the struct from pass through untouched.
        """
        for name in ("soil_bucket_depth", "drainage_timescale",
                     "soil_water_field_capacity", "soil_water_wilting_point"):
            value = getattr(self, name)
            if isinstance(value, (list, tuple)):
                object.__setattr__(self, name, jnp.asarray(value, dtype=float))

    @classmethod
    def default(cls) -> "BucketLandParameters":
        """Return the default parameters."""
        return cls()
