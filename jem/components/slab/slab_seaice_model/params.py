"""Tunable parameters of the slab sea-ice model."""

from flax import struct


@struct.dataclass
class SlabSeaiceParameters:
    """Parameters of :class:`~jem.components.slab.slab_seaice_model.SlabSeaiceModel`.

    The model has no tunable parameters yet, so this class has no fields. It
    is kept so that the sea-ice carry has a ``params`` entry like every other
    slab model, and so a future parameter has an obvious place to go.

    The width of the SST ramp is the module constant
    :data:`~jem.components.slab.slab_seaice_model.ICE_FREE_SST_EXCESS`. Being
    a constant, it cannot be differentiated.
    """

    @classmethod
    def default(cls) -> "SlabSeaiceParameters":
        """Return the default parameters."""
        return cls()
