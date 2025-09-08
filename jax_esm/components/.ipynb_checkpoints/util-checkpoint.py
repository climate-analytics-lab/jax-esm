
#from collections import abc

from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
import tree_math
from jax import tree_util

from dataclasses import make_dataclass

from jax_esm.components.PhysicsState import PhysicsState


def stack_objects(objs):
    # objs is a list of pytrees with same structure
    leaves = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return leaves


def createStateDiagClass(
    state_cls,
    diag_cls,
    model_name = "",
):
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


    def copy(self, state_kwargs = {}, diag_kwargs = {}):
        o = new_cls(
            state = self.state.copy(**state_kwargs),
            diag = self.diag.copy(**diag_kwargs),
        )

        return o
        
    new_cls.zeros = zeros
    new_cls.copy = copy
    
    new_cls = tree_math.struct(new_cls)
    
    return new_cls


def createResultClass(
    state_cls,
    model_name = "",
):
    result_cls = make_dataclass(
        f"ResultClass_{model_name:s}",
        [
            ("state", state_cls),
            ("predictions", Any),
            ("times", Any),
        ]
    )

    @classmethod
    def empty(_cls):
        
        return _cls(
            state = state_cls.zeros(),
            predictions = [],
            times = [],
        )

    result_cls.empty = empty
    result_cls = tree_math.struct(result_cls)
    
    return result_cls


def createPhysicsStateClass(
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


"""
def convertTrajectoryToXarray(
    trajectory
):

    stacked = stack_objects(trajectory)
    test_obj = trajectory[0]
    
    for super_name in ["state", "diag"]:
        super_obj = getattr(test_obj, super_name)
        for varname, _ in super_obj.__dataclass_fields__.items():
            value = getattr(x, varname)
            
    
    ds = xr.Dataset(
        data_vars = dict(
            T   = (["time", "lon", "lat"], stacked.state.T),
            mld = (["time", "lon", "lat"], stacked.state.mld),
        ),
    )
    
    return ds
"""