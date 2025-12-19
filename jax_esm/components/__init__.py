"""Earth system components module."""

from jax_esm.components.JCM import JCM
import jax_esm.components.slab_ocean_model as slab_ocean_model
import jax_esm.components.slab_atmosphere_model as slab_atmosphere_model
import jax_esm.components.slab_land_model as slab_land_model

__all__ = [
    "JCM",
    "slab_ocean_model",
    "slab_atmosphere_model",
    "slab_land_model",
]
