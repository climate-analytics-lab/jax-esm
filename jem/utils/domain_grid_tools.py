
import coordax as cx
import jax.numpy as jnp


def generate_coordinate_from_latitude_longitude(
    latitude: list[float] | jnp.ndarray,
    longitude: list[float] | jnp.ndarray,
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
            f"Error: `order` has to be either `longitude_latitude` or `latitude_longitude`. User here input `{order!s:s}`"
        )

    return cx.coords.compose(*args)
