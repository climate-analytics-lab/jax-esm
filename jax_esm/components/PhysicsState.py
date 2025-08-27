
from collections import abc

import jax.numpy as jnp
import tree_math
#from typing import Callable
from jax import tree_util
from dataclasses import make_dataclass


class PhysicsState:

    ...

def CreatePhysicsStateClass(
    cls_name,
    fields,   # A list of (varname, jnp_dtype, shape) tuples.
):

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


    
