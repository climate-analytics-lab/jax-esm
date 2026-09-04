"""Tunable parameters of the slab ocean model."""

import jax.numpy as jnp
from flax import struct

#: Forcing terms the mixed-layer temperature equation can carry.
FORCING_METHODS = ("none", "qflux", "relaxation")


@struct.dataclass
class SlabOceanParameters:
    """Parameters of :class:`~jem.components.slab.slab_ocean_model.SlabOceanModel`.

    Every numeric field is a pytree leaf, so ``jax.grad`` of a coupled run with
    respect to any of them works: the parameters travel in the component's
    carry (``carry["params"]``), not in a closure over the model object.
    ``forcing_method`` is the one exception -- it selects which terms are traced
    at all, so it is static aux data and is read with an ordinary ``if``.

    Attributes
    ----------
    relaxation_time : jnp.ndarray
        Timescale (s) of the relaxation of SST to its climatology. Read only
        when ``forcing_method == "relaxation"``. Default 60 days.
    mixed_layer_depth_min, mixed_layer_depth_max : jnp.ndarray
        Ends of the prescribed mixed-layer depth profile (m): the depth is
        ``max + (min - max) * cos(latitude)**3``, so ``min`` applies at the
        equator and ``max`` at the poles.
    initial_sst : jnp.ndarray
        Sea-surface temperature (K) the ocean starts from where no SST
        climatology is given -- the base of the idealized profile described in
        :class:`SlabOceanModel`.
    forcing_method : str
        One of ``"none"``, ``"qflux"`` or ``"relaxation"``; static.
    ocean_mask_value : float
        Value of the grid's binary mask that marks an ocean cell (0 = ocean,
        1 = land). Static: it selects which cells the model integrates at trace
        time, and is a mask convention rather than a physical tunable.

    """

    relaxation_time: float | jnp.ndarray = 60 * 86400.0
    mixed_layer_depth_min: float | jnp.ndarray = 40.0
    mixed_layer_depth_max: float | jnp.ndarray = 60.0
    initial_sst: float | jnp.ndarray = 288.15
    forcing_method: str = struct.field(pytree_node=False, default="none")
    ocean_mask_value: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def default(cls) -> "SlabOceanParameters":
        """Return the default parameters."""
        return cls()
