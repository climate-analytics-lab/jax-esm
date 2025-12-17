"""Base component interface for Earth system models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Tuple, Sequence, Callable, Dict

from jax_esm.components.base import AbstractFieldGroup
import jax.numpy as jnp
import xarray as xr
import tree_math
from dataclasses import make_dataclass

def create_field_group_class(
    fields: Sequence[Tuple[str, type, Sequence[str], Sequence[int]]],
    cls_name: Optional[str] = None,
):
    """
    A tool function that creates a state class dynamically with given dimension names and shapes.
    The created class will have the following methods: `zeros`, `ones`, and `copy`.

    Args:

        cls_name : Name of the state class
        fields   : A list of (varname, data type, dimension names, shape) tuples.

    Returns:

        class : The resulting state class.

    """
    dataclass_fields = [(varname, dtype) for varname, dtype, _, _ in fields]

    cls = make_dataclass(
        cls_name=cls_name or "",
        fields=dataclass_fields,
        bases=(AbstractFieldGroup,),
    )

    @classmethod  # type: ignore
    def zeros(_cls, **kwargs):
        init_args = dict()
        for varname, dtype, _, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.zeros(shape, dtype=dtype)

        return _cls(**init_args)

    @classmethod  # type: ignore
    def ones(_cls, **kwargs):
        init_args = dict()
        for varname, dtype, _, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.ones(shape, dtype=dtype)

        return _cls(**init_args)

    def copy(self, **kwargs):
        init_args = dict()
        for varname, dtype, _, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = getattr(self, varname)

        return type(self)(**init_args)

    cls.zeros = zeros  # type: ignore
    cls.ones = ones  # type: ignore
    cls.copy = copy  # type: ignore
    cls.fields = fields

    return tree_math.struct(cls)

def create_variable_container_class(
    subgroups: Dict[str, type],
    base_class: type = object,
    cls_name: str = "",
):
    """
    A tool function that creates a container class dynamically with given dimension.
    The created class will have the following methods: `zeros`, `ones`, and `copy`.

    Args:
        subgroups: The mapping of subgroup name to a AbstractFieldGroup
        cls_name  : Name of the class.

    Returns:

        The resulting class.

    """

    new_cls = make_dataclass(
        f"{cls_name:s}",
        [ (subgroup_name, subgroup_class) for subgroup_name, subgroup_class in subgroups.items() ], 
        bases=(base_class,),
    )

    @classmethod  # type: ignore
    def zeros(_cls):
        return _cls(**{ subgroup_name : subgroup_class.zeros() for subgroup_name, subgroup_class in subgroups.items() })

    @classmethod  # type: ignore
    def ones(_cls):
        return _cls(**{ subgroup_name : subgroup_class.ones() for subgroup_name, subgroup_class in subgroups.items() })

    def copy(self, **kwargs):
        return new_cls(**{
            subgroup_name : getattr(self, subgroup_name).copy(**(kwargs[subgroup_name] if subgroup_name in kwargs else {}))
            for subgroup_name, subgroup_class in subgroups.items()
        })


    new_cls.zeros = zeros  # type: ignore
    new_cls.ones = ones  # type: ignore
    new_cls.copy = copy  # type: ignore

    new_cls = tree_math.struct(new_cls)

    return new_cls

