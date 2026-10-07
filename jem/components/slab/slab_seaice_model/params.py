"""Tunable parameters of the slab sea-ice model."""

from flax import struct


@struct.dataclass
class SlabSeaiceParameters:
    """Parameters of :class:`~jem.components.slab.slab_seaice_model.SlabSeaiceModel`.

    The parameters travel in the component's carry (``carry["params"]``), not
    in a closure over the model object, like every slab model's.

    Attributes
    ----------
    ocean_mask_value : float
        Value of the grid's binary mask that marks an ocean cell (0 = ocean,
        1 = land). Static: it selects which cells the model integrates at trace
        time, and is a mask convention rather than a physical tunable.

    """

    ocean_mask_value: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def default(cls) -> "SlabSeaiceParameters":
        """Return the default parameters."""
        return cls()
