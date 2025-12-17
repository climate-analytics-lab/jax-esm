from typing import List, Tuple, Optional, Dict
from jax import Array
import jax.numpy as jnp
import xarray as xr
import dinosaur
from dataclasses import dataclass
import re
import jax_esm
from pathlib import Path

from jax_esm.base.grid import GridSpecification

import coordax as cx

def generate_coordinate_from_latitude_longitude(
    latitude: List[float] | jnp.ndarray,
    longitude: List[float] | jnp.ndarray,
    order: str = "latitude_longitude",
) -> cx.Coordinate:
    
    axis_latitude = cx.LabeledAxis('latitude', jnp.array(latitude))
    axis_longitude = cx.LabeledAxis('longitude', jnp.array(longitude))

    args = None

    if order == "latitude_longitude":
        args = (axis_latitude, axis_longitude)

    elif order == "longitude_latitude":
        args = (axis_longitude, axis_latitude)
    
    else:
        raise ValueError(
            f"Error: `order` has to be either `longitude_latitude` or `latitude_longitude`. User here input `{str(order):s}`"
        )
    
    return cx.coords.compose(*args)




