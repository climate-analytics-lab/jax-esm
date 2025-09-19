
#from collections import abc

from typing import Dict, Tuple, Any, List

import jax
import jax.numpy as jnp
import tree_math
from jax import tree_util

from dataclasses import make_dataclass

from jax_esm.components.PhysicsState import PhysicsState


def stack_objects(
    objs : List
):
    """
    A tool function that stack dataclasses together.

    Args:

        objs : A list of objects that need to be stacked

    Returns:

        stacked : Stacked object.
        
    """
    # objs is a list of pytrees with same structure
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked


def createStateDiagClass(
    state_cls  : type,
    diag_cls   : type,
    model_name : str = "",
):
    """
    A tool function that creates a state-diag class dynamically with given dimension.
    The created class will have the following methods: `zeros`, `ones`, and `copy`. 
    
    Args:

        state_cls  : The state class.
        diag_cls   : The diag class.
        model_name : Name of the model. The class name will be "StateDiagClass_{model_name}".

    Returns:

        stateDiagClass: The resulting state-diag class.
    
    """
    
    new_cls = make_dataclass(
        f"StateDiagClass_{model_name:s}",
        [
            ("state", state_cls),
            ("diag", diag_cls),
        ]
    )

    @classmethod
    def zeros(_cls):
        return _cls(
            state = state_cls.zeros(),
            diag = diag_cls.zeros(),
        )

    @classmethod
    def ones(_cls):
        return _cls(
            state = state_cls.ones(),
            diag = diag_cls.ones(),
        )


    def copy(self, state_kwargs = {}, diag_kwargs = {}):
        o = new_cls(
            state = self.state.copy(**state_kwargs),
            diag = self.diag.copy(**diag_kwargs),
        )

        return o
        
    new_cls.zeros = zeros
    new_cls.ones = ones
    new_cls.copy = copy
    
    new_cls = tree_math.struct(new_cls)
    
    return new_cls

def createPhysicsStateClass(
    cls_name: str,
    fields: Tuple,   
):
    
    """
    A tool function that creates a state class dynamically with given dimension.
    The created class will have the following methods: `zeros`, `ones`, and `copy`. 
    
    Args:

        cls_name : Name of the state class
        fields   : A list of (varname, data type, shape) tuples.

    Returns:

        class : The resulting state class.
    
    """
    dataclass_fields = [ (varname, dtype) for varname, dtype, _ in fields ]
    
    cls = make_dataclass(
        cls_name = cls_name,
        fields = dataclass_fields,
        bases = ( PhysicsState, ),
    )

    
    @classmethod
    def zeros(_cls, **kwargs):
        
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.zeros(shape, dtype=dtype)

        return _cls(**init_args)

    @classmethod
    def ones(_cls, **kwargs):
        
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = jnp.ones(shape, dtype=dtype)

        return _cls(**init_args)

    def copy(self, **kwargs):
        init_args = dict()
        for varname, dtype, shape in fields:
            if (varname in kwargs) and (kwargs[varname] is not None):
                init_args[varname] = kwargs[varname]
            else:
                init_args[varname] = getattr(self, varname)

        return type(self)(**init_args)

    cls.zeros = zeros
    cls.ones = ones
    cls.copy = copy
    
    return tree_math.struct(cls)
