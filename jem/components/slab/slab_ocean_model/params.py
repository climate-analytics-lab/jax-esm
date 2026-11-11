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

    ``initial_sst`` is an **initial-condition** parameter: it is read once, by
    :meth:`SlabOceanModel.initialize` (and only when no SST climatology was
    given), and never by ``step``. Vary it by passing parameters to
    ``initialize`` (or to ``Coupler.initialize({"ocn": params})``); replacing
    that leaf in a carry that already exists changes nothing, because its value
    has already been copied into the state. The others are process parameters,
    read from the carry every step.

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
        :class:`SlabOceanModel`. Initial condition; read only by
        ``initialize``.
    reference_salinity : jnp.ndarray
        Salinity (g/kg) the mixed layer starts from, and the salinity the
        virtual salt flux is computed with: a freshwater flux ``F`` (kg m-2
        s-1, positive upward) changes the salinity at the rate
        ``reference_salinity * F / (rho * h)``. Using one fixed salinity
        rather than the local one is CESM's ``ocn_ref_salinity`` convention:
        it makes the salt change proportional to the water budget alone, so a
        freshwater flux that sums to zero over the ocean leaves the total salt
        unchanged. Default 34.7, CESM's value.
    forcing_method : str
        One of ``"none"``, ``"qflux"`` or ``"relaxation"``; static.

    """

    relaxation_time: float | jnp.ndarray = 60 * 86400.0
    mixed_layer_depth_min: float | jnp.ndarray = 40.0
    mixed_layer_depth_max: float | jnp.ndarray = 60.0
    initial_sst: float | jnp.ndarray = 288.15
    reference_salinity: float | jnp.ndarray = 34.7
    forcing_method: str = struct.field(pytree_node=False, default="none")
    
    @classmethod
    def default(cls) -> "SlabOceanParameters":
        """Return the default parameters."""
        return cls()
