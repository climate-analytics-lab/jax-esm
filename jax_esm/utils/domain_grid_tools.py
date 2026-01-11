from typing import List
import jax.numpy as jnp


import coordax as cx


def generate_coordinate_from_latitude_longitude(
    latitude: List[float] | jnp.ndarray,
    longitude: List[float] | jnp.ndarray,
    order: str = "latitude_longitude",
) -> cx.Coordinate:
    axis_latitude = cx.LabeledAxis("latitude", jnp.array(latitude))
    axis_longitude = cx.LabeledAxis("longitude", jnp.array(longitude))

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
