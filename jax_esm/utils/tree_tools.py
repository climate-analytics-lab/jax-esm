import jax
from jax import tree_util
import numpy as np

def print_tree(tree, root=None):
    print_dict_tree(convert_tree_into_dict(tree), root=root)


def convert_key_to_string(key):

    key_str = ""
    if isinstance(key, jax.tree_util.DictKey):
        key_str = f"{key.key}" 
    elif isinstance(key, jax.tree_util.GetAttrKey):
        key_str = f"{key.name}"
    elif isinstance(key, jax.tree_util.SequenceKey):
        key_str = f"[{key.idx}]"
    elif isinstance(key, jax.tree_util.FlattenedIndexKey):
        key_str = f"<{key.key}>"
    else:
        key_str = str(key)

    return key_str

def print_dict_tree(d, indent="", is_last=True, prefix=None, root=None):
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
            print(f"{indent}{connector}📄 {prefix}{preview}")
        else:
            if isinstance(d, np.ndarray | jax.Array):
                d_str = f"<Array>"
            else:
                d_str = f"{d}"
            print(f"{indent}{connector}📄 {prefix}{d_str}")
        return
   
    if root:
        print(f"/{root}") 
    # It's a dict node
    if prefix:
        print(f"{indent}{connector}📁 {prefix}")
        indent = indent + extension
    
    items = list(d.items())
    for i, (key, value) in enumerate(items):
        is_last_item = (i == len(items) - 1)
        
        if isinstance(value, dict):
            print_dict_tree(value, indent, is_last_item, key)
        else:
            print_dict_tree(value, indent, is_last_item, f"{key}: ")

def convert_tree_into_dict(tree):
    path_leaves = tree_util.tree_flatten_with_path(tree)[0]
 
    d = {}
    for path, leaf in path_leaves:
        _d = d 
        for key in path[:-1]:
            key_str = convert_key_to_string(key)
            if key_str not in _d:
                _d[key_str] = {}
            _d = _d[key_str]

        _d[convert_key_to_string(path[-1])] = leaf

    return d

        
