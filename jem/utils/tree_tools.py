import jax
from jax import tree_util
import numpy as np
from typing import Any
from dataclasses import dataclass, fields, is_dataclass

def print_tree(tree, root=None):
    print_dict_tree(tree_to_dict(tree), root=root)

def print_dict_tree(d, indent="", is_last=True, prefix=None, root=None, node_symbol="●", leaf_symbol="○"):
    """Print dictionary in directory-style format"""
    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "
   
    if not isinstance(d, dict):
        # It's a leaf value
        if isinstance(d, list):
            if len(d) > 3:
                preview = f"[{d[0]}, {d[1]}, ..., {d[-1]}] (len={len(d)})"
            else:
                preview = str(d)
            print(f"{indent}{connector}{leaf_symbol} {prefix}{preview}")
        else:
            if isinstance(d, np.ndarray | jax.Array):
                d_str = f"{d.shape}"
            else:
                d_str = f"{d}"
            print(f"{indent}{connector}{leaf_symbol} {prefix}{d_str}")
        return
   
    if root:
        print(f"/{root}") 
    # It's a dict node
    if prefix:
        print(f"{indent}{connector}{node_symbol} {prefix}")
        indent = indent + extension
    
    items = list(d.items())
    for i, ((key, class_name), value) in enumerate(items):
        is_last_item = (i == len(items) - 1)
        
        if isinstance(value, dict):
            print_dict_tree(value, indent, is_last_item, f"{key} <{class_name}>")
        else:
            print_dict_tree(value, indent, is_last_item, f"{key} <{class_name}>: ")

def tree_to_dict(obj: Any) -> Any:
    """
    Convert a tree containing dataclasses, dicts, lists, and tuples into a pure dict tree.
    
    Handles:
    - Dataclasses (recursively converts to dicts)
    - Dicts (recursively processes values)
    - Lists (recursively processes items, converts to list)
    - Tuples (recursively processes items, converts to list)
    - NumPy arrays (converts to lists)
    - Primitives (returns as-is)
    
    Args:
        obj: Tree structure to convert
        
    Returns:
        Pure dict tree (or list/primitive)
        
    Note: Both lists and tuples are converted to lists in the output
    """
    # Dataclass -> dict
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            (f"{field.name}", type(getattr(obj, field.name)).__name__): tree_to_dict(getattr(obj, field.name))
            for field in fields(obj)
        }
    
    # Dict -> recursively process values
    if isinstance(obj, dict):
        return { (f"{key}", type(value).__name__): tree_to_dict(value) for key, value in obj.items()}
    
    # List -> recursively process items
    if isinstance(obj, list):
        return { (f"{{{i:d}}}", type(item).__name__) : tree_to_dict(item) for i, item in enumerate(obj) }
 
    # Tuple -> recursively process items
    if isinstance(obj, tuple):
        return { (f"[{i:d}]", type(item).__name__) : tree_to_dict(item) for i, item in enumerate(obj) }
    
    # NumPy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Primitive types (int, float, str, bool, None)
    return obj       
 



