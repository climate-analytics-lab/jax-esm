"""Tunable parameters of the slab sea-ice model."""

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class SlabSeaiceParameters:
    """Parameters of :class:`~jem.components.slab.slab_seaice_model.SlabSeaiceModel`.

    The parameters travel in the component's carry (``carry["params"]``), not
    in a closure over the model object, like every slab model's, so
    ``jax.grad`` of a coupled run reaches ``ice_free_sst_excess``.

    Attributes
    ----------
    ice_free_sst_excess : float or jnp.ndarray
        Width (K) of the linear ramp from full ice cover at the seawater
        freezing point to open water: a cell is ice-free once its sea surface
        temperature is this far above freezing. Default 1.8 K.
    ocean_mask_value : float
        Value of the grid's binary mask that marks an ocean cell (0 = ocean,
        1 = land). Static: it selects which cells the model integrates at trace
        time, and is a mask convention rather than a physical tunable.

    """

    ice_free_sst_excess: float | jnp.ndarray = 1.8
    ocean_mask_value: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def default(cls) -> "SlabSeaiceParameters":
        """Return the default parameters."""
        return cls()
