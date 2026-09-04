"""Tunable parameters of the slab sea-ice model."""

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class SlabSeaiceParameters:
    """Parameters of :class:`~jem.components.slab.slab_seaice_model.SlabSeaiceModel`.

    Every numeric field is a pytree leaf, so ``jax.grad`` of a coupled run with
    respect to any of them works: the parameters travel in the component's
    carry (``carry["params"]``), not in a closure over the model object.

    Attributes
    ----------
    initial_ice_thickness : jnp.ndarray
        Uniform ice thickness (m) over ocean points at the start of a run.
    min_ice_thickness : jnp.ndarray
        Thickness (m) above which a cell is *diagnosed* as ice-covered. It has
        no effect on the thickness tendency; it only selects which surface
        temperature is reported.
    ice_fraction_thickness_scale : jnp.ndarray
        Thickness scale (m) of the smooth ``1 - exp(-h / scale)`` closure that
        turns thickness into an areal ice fraction.
    ocean_mask_value : float
        Value of the grid's binary mask that marks an ocean cell (0 = ocean,
        1 = land). Static: it selects which cells the model integrates at trace
        time, and is a mask convention rather than a physical tunable.

    """

    initial_ice_thickness: float | jnp.ndarray = 0.0
    min_ice_thickness: float | jnp.ndarray = 1e-3
    ice_fraction_thickness_scale: float | jnp.ndarray = 0.5
    ocean_mask_value: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def default(cls) -> "SlabSeaiceParameters":
        """Return the default parameters."""
        return cls()
