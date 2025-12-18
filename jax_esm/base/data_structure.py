from typing import get_type_hints, Annotated, get_origin, get_args
from typing import Any, List, Tuple
import jax.numpy as jnp
from copy import deepcopy
import tree_math
from jax.tree_util import register_pytree_node_class

def schema(cls):
    hints = get_type_hints(cls, include_extras=True)
    cls._fields = {}

    for name, hint in hints.items():
        if get_origin(hint) is Annotated:
            meta = get_args(hint)
            cls._fields[name] = dict(data_type=meta[0], dimension_names=meta[1], shape_name=meta[2])
        else:
            cls._fields[name] = hint  # nested schema

    cls.is_schema = True


    def tree_flatten(self):
        children = []
        aux_data = []  # any static info to keep

        flds = getattr(self, "_fields", {})
        for name, info in flds.items():
            obj = getattr(self, name)
            if isinstance(info, dict):  # leaf
                children.append(obj)
                aux_data.append(dict(name=name, is_leaf=True))
            elif isinstance(info, type) and hasattr(info, "_fields"):  # nested
                _children, _children_aux_data = obj.tree_flatten()
                children.append(_children) 
                aux_data.append(dict(name=name, is_leaf=False, cls=info, aux_data=_children_aux_data))
        
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)  # bypass __init__
        for metadata, data in zip(aux_data, children):
            if metadata["is_leaf"]:
                setattr(obj, metadata["name"], data)
            else:
                node_obj = metadata["cls"].tree_unflatten(metadata["aux_data"], data)
                setattr(obj, metadata["name"], node_obj)
                

        return obj


    cls.tree_unflatten = tree_unflatten
    cls.tree_flatten = tree_flatten

    return register_pytree_node_class(cls)

def build_dataclass(shape_dict: dict):

    def decorator(cls):
        
        if getattr(cls, "is_schema", None) != True:
            raise Exception("Cannot apply build_dataclass on a non-schemaed class.")
                
        cls.shape_dict = shape_dict or {}

        if "zero_dimensional" not in cls.shape_dict:
            cls.shape_dict["zero_dimensional"] = ()
        
        fields = getattr(cls, "_fields", {})
        # Recursive constructor
        def __init__(self):
            for name, info in fields.items():
                if isinstance(info, type) and hasattr(info, "_fields"):
                    setattr(self, name, info)  # nested node
                else:
                    setattr(self, name, None)    # placeholder for leaf

        # Recursive fill helper
        def _fill(obj, flds, fill_fn):
            for name, info in flds.items():
                if isinstance(info, dict):  # leaf
                    setattr(obj, name, fill_fn(cls.shape_dict[info["shape_name"]]))
                elif isinstance(info, type) and hasattr(info, "_fields"):  # nested
                    node_obj = info.__new__(info)
                    _fill(node_obj, info._fields, fill_fn)
                    setattr(obj, name, node_obj)


                else:
                    raise Exception("Should not be here. Please type check")

        # Recursive replacement helper
        def _replace(obj, replace_dict):
            for path, new_value in replace_dict.items():
                parts = path.split(".")
                target = obj
                for p in parts[:-1]:
                    target = getattr(target, p)
                setattr(target, parts[-1], new_value)

        # Class methods
        @classmethod
        def zeros(cls, replace_dict=None):
            obj = cls()
            _fill(obj, fields, jnp.zeros)
            if replace_dict:
                _replace(obj, replace_dict)
            return obj

        @classmethod
        def ones(cls, replace_dict=None):
            obj = cls()
            _fill(obj, fields, jnp.ones)
            if replace_dict:
                _replace(obj, replace_dict)
            return obj

        # Member copy with optional replacements
        def copy(self, replace_dict=None):
            obj_copy = deepcopy(self)
            if replace_dict:
                _replace(obj_copy, replace_dict)
            return obj_copy

        # Recursive introspection
        def _collect_info(obj_or_class, parent_name="") -> List[Tuple[str, Any, Any, Any]]:
            result = []
            flds = getattr(obj_or_class, "_fields", {})
            for name, info in flds.items():
                full_name = f"{parent_name}.{name}" if parent_name else name
                if isinstance(info, dict):  # leaf
                    result.append((full_name, info["data_type"], info["dimension_names"], cls.shape_dict[info["shape_name"]]))
                elif isinstance(info, type) and hasattr(info, "_fields"):  # nested
                    result.extend(_collect_info(info, parent_name=full_name))
                else:
                    raise Exception("Should not be here. Please type check")
            return result
        
        # Attribute introspection method
        @classmethod
        def schema_info(cls) -> List[Tuple[str, Any, Any, Any]]:
            """Return list of tuples: (full_name, data_type, dim_names, shape)"""
            return _collect_info(cls)

        def __getitem__(self, key: str):
            obj = self
            for part in key.split("."):
                obj = getattr(obj, part)
            return obj

        # Inject
        cls.__init__ = __init__
        cls.zeros = zeros
        cls.ones = ones
        cls.copy = copy
        cls.schema_info = schema_info
        cls.__getitem__ = __getitem__
        
        return cls

    return decorator




if __name__ == "__main__":
 
    @schema
    class PrognosticData:
        sim_time: Annotated[float, (), "zero_dimensional"]
        sea_surface_temperature: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]
        mixed_layer_depth: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]

    @schema
    class OceanState:
        prog : PrognosticData


    @schema
    class AirseaFlux:
        total_heat_flux: Annotated[float, ("latitudinal", "longitude"), "two_dimensional"]

    @schema
    class OceanForcing:
        flux : AirseaFlux

   
    decorator = build_dataclass({"two_dimensional": (2, 3)})
    forcing_class= decorator(OceanForcing)
    state_class= decorator(OceanState)

    forcing = forcing_class.zeros()
    state = state_class.zeros()

    f = state.prog.tree_flatten()
    PrognosticData.tree_unflatten(f[1], f[0])
    flattened = state.tree_flatten()
    state_class.tree_unflatten(flattened[1], flattened[0])
