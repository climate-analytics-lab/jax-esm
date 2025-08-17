
from dataclasses import make_dataclass


def AddOnesZerosCopy(
    cls,
    var_info,   # A list of (varname, dtype, shape) pairs.
):
    make_dataclass(
    "SAMState",
    [("T", jnp.ndarray), ("q", jnp.ndarray), ("u", jnp.ndarray)]
)