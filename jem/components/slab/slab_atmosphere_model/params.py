"""Tunable parameters of the idealized slab atmosphere model."""

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class SlabAtmosphereParameters:
    """Parameters of :class:`~jem.components.slab.slab_atmosphere_model.SlabAtmosphereModel`.

    Every field is a pytree leaf, so ``jax.grad`` of a coupled run with respect
    to any of them works: the parameters travel in the component's carry
    (``carry["params"]``), not in a closure over the model object.

    They are all initial-condition tunables. The bulk-formula coefficients this
    model uses are physical constants of the surface layer, not per-run knobs,
    and live in :mod:`jem.constants`; the drag coefficient additionally travels
    in the model's *forcing* so an exchanger can vary it per cell.

    Attributes
    ----------
    initial_temperature_base : jnp.ndarray
        Column-mean air temperature (K) at the poles at the start of a run.
    initial_temperature_amplitude : jnp.ndarray
        Equator-to-pole range (K) of the initial column-mean air temperature.
    initial_zonal_wind : jnp.ndarray
        Initial column-mean eastward wind (m/s), uniform over the grid.
    initial_meridional_wind : jnp.ndarray
        Initial column-mean northward wind (m/s), uniform over the grid.

    """

    initial_temperature_base: float | jnp.ndarray = 273.15
    initial_temperature_amplitude: float | jnp.ndarray = 17.0
    initial_zonal_wind: float | jnp.ndarray = 10.0
    initial_meridional_wind: float | jnp.ndarray = 0.0

    @classmethod
    def default(cls) -> "SlabAtmosphereParameters":
        """Return the default parameters."""
        return cls()
