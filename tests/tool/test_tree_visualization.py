import jem

import jem.components.slab_atmosphere_model as SAM
from jem.utils.tree_tools import analyze_tree, visualize_pytree_with_paths, PyTreePrinter

from jax import tree_util


if __name__ == "__main__":
    visualize_pytree_with_paths(SAM.AtmosphereState)


    PyTreePrinter.print_structure(SAM.AtmosphereForcing)
    print(tree_util.tree_flatten(SAM.AtmosphereState))
    SAM.AtmosphereState.zeros().tree_flatten()
    
    
    
