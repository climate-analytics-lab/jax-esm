"""
Generated with ClaudeAI
"""

from typing import Any, Tuple, List, Optional, Dict
import warnings


class DottedDict(dict):
    """An immutable dict that supports dot notation and nested key access, with pytree support.
    
    Once created, values cannot be modified directly. Use the copy() method to create
    modified versions, similar to Flax's FrozenDict.
    """
    
    def __init__(self, *args, metadata: Optional[List[Tuple[str, Any, Tuple[str, ...], Tuple[int, ...]]]] = None, 
                 _allow_mutations: bool = True, **kwargs):
        """
        Initialize DottedDict with optional metadata.
        
        Args:
            metadata: List of (variable_name, data_type, dimension_names, shape) tuples
            _allow_mutations: Internal flag to allow mutations during initialization
        """
        super().__init__(*args, **kwargs)
        
        # Store metadata
        object.__setattr__(self, '_metadata', {})
        if metadata:
            for var_name, dtype, dim_names, shape in metadata:
                self._metadata[var_name] = {
                    'dtype': dtype,
                    'dim_names': dim_names,
                    'shape': shape
                }
        
        # Recursively freeze nested dicts
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, DottedDict):
                super().__setitem__(key, DottedDict(value, _allow_mutations=True))
        
        # Freeze the dict
        object.__setattr__(self, '_frozen', not _allow_mutations)
    
    def _check_frozen(self):
        """Raise error if attempting to modify frozen dict."""
        if getattr(self, '_frozen', False):
            raise TypeError(
                f"DottedDict is immutable. Use .copy(**updates) to create a modified version."
            )
    
    def __setitem__(self, key, value):
        """Handle setting items - only allowed during initialization."""
        self._check_frozen()
        
        if '.' in key:
            keys = key.split('.')
            current = self
            
            # Navigate/create nested structure
            for k in keys[:-1]:
                if k not in current:
                    super(DottedDict, current).__setitem__(k, DottedDict(_allow_mutations=False))
                elif not isinstance(current[k], dict):
                    super(DottedDict, current).__setitem__(k, DottedDict(_allow_mutations=False))
                current = current[k]
            
            # Set the final value
            super(DottedDict, current).__setitem__(keys[-1], value)
        else:
            super().__setitem__(key, value)
    
    def __delitem__(self, key):
        """Prevent deletion."""
        self._check_frozen()
        super().__delitem__(key)
    
    def __getattr__(self, key):
        """Allow dot notation access."""
        try:
            value = self[key]
            # Wrap dicts in DottedDict for chaining
            if isinstance(value, dict) and not isinstance(value, DottedDict):
                value = DottedDict(value, _allow_mutations=False)
                super().__setitem__(key, value)
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        """Prevent attribute setting."""
        if key.startswith('_'):
            # Allow internal attributes like _metadata, _frozen
            object.__setattr__(self, key, value)
        else:
            self._check_frozen()
            self[key] = value
    
    def __delattr__(self, key):
        """Prevent attribute deletion."""
        self._check_frozen()
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __hash__(self):
        """Make DottedDict hashable since it's immutable."""
        try:
            return hash(tuple(sorted(self.items())))
        except TypeError:
            # If values aren't hashable, fall back to id-based hash
            return hash(id(self))
    
    def _get_all_keys(self, prefix=''):
        """Recursively get all dotted keys in the structure."""
        keys = set()
        for k, v in self.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            if isinstance(v, DottedDict):
                keys.update(v._get_all_keys(prefix=full_key))
        return keys
    
    def _set_nested(self, key: str, value: Any):
        """Set a value using dotted key notation (internal use only during init)."""
        if '.' in key:
            keys = key.split('.')
            current = self
            for k in keys[:-1]:
                if k not in current:
                    super(DottedDict, current).__setitem__(k, DottedDict(_allow_mutations=False))
                current = current[k]
            super(DottedDict, current).__setitem__(keys[-1], value)
        else:
            super().__setitem__(key, value)
    
    def _get_nested(self, key: str) -> Any:
        """Get a value using dotted key notation."""
        if '.' in key:
            keys = key.split('.')
            current = self
            for k in keys:
                current = current[k]
            return current
        else:
            return self[key]
    
    @classmethod
    def zeros(cls, metadata: List[Tuple[str, Any, Tuple[str, ...], Tuple[int, ...]]], **overrides) -> 'DottedDict':
        """
        Create a DottedDict with zero-initialized arrays based on metadata.
        
        Args:
            metadata: List of (variable_name, data_type, dimension_names, shape) tuples
            **overrides: Key-value pairs to override zeros with pre-existing data
            
        Returns:
            Immutable DottedDict with zero-initialized (or overridden) arrays
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX is required for zeros() method. Install with: pip install jax")
        
        result = cls(metadata=metadata, _allow_mutations=True)
        
        # Check for non-existent keys in overrides
        all_var_names = {var_name for var_name, _, _, _ in metadata}
        for key in overrides.keys():
            if key not in all_var_names:
                warnings.warn(f"Override key '{key}' does not match any variable in metadata. Available keys: {all_var_names}")
        
        # Create arrays
        for var_name, dtype, dim_names, shape in metadata:
            if var_name in overrides:
                # Use pre-existing data
                result._set_nested(var_name, overrides[var_name])
            else:
                # Create zeros array
                result._set_nested(var_name, jnp.zeros(shape, dtype=dtype))
        
        # Freeze the result
        object.__setattr__(result, '_frozen', True)
        return result
    
    @classmethod
    def ones(cls, metadata: List[Tuple[str, Any, Tuple[str, ...], Tuple[int, ...]]], **overrides) -> 'DottedDict':
        """
        Create a DottedDict with one-initialized arrays based on metadata.
        
        Args:
            metadata: List of (variable_name, data_type, dimension_names, shape) tuples
            **overrides: Key-value pairs to override ones with pre-existing data
            
        Returns:
            Immutable DottedDict with one-initialized (or overridden) arrays
        """
        try:
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX is required for ones() method. Install with: pip install jax")
        
        result = cls(metadata=metadata, _allow_mutations=True)
        
        # Check for non-existent keys in overrides
        all_var_names = {var_name for var_name, _, _, _ in metadata}
        for key in overrides.keys():
            if key not in all_var_names:
                warnings.warn(f"Override key '{key}' does not match any variable in metadata. Available keys: {all_var_names}")
        
        # Create arrays
        for var_name, dtype, dim_names, shape in metadata:
            if var_name in overrides:
                # Use pre-existing data
                result._set_nested(var_name, overrides[var_name])
            else:
                # Create ones array
                result._set_nested(var_name, jnp.ones(shape, dtype=dtype))
        
        # Freeze the result
        object.__setattr__(result, '_frozen', True)
        return result
    
    def copy(self, **overrides) -> 'DottedDict':
        """
        Create a deep copy of this DottedDict with optional overrides.
        
        This is the primary way to "modify" an immutable DottedDict - by creating
        a new version with changes applied.
        
        Args:
            **overrides: Key-value pairs to override in the copied structure
            
        Returns:
            A new immutable DottedDict with copied (and optionally overridden) data
        """
        try:
            import jax.numpy as jnp
            from jax.tree_util import tree_map
        except ImportError:
            raise ImportError("JAX is required for copy() method. Install with: pip install jax")
        
        # Deep copy the structure using tree_map
        result = tree_map(lambda x: jnp.array(x) if hasattr(x, 'shape') else x, self)
        
        # Ensure result is a mutable DottedDict temporarily
        if not isinstance(result, DottedDict):
            result = DottedDict(result, _allow_mutations=True)
        else:
            object.__setattr__(result, '_frozen', False)
        
        # Copy metadata if it exists
        if hasattr(self, '_metadata'):
            object.__setattr__(result, '_metadata', self._metadata.copy())
        
        # Check for non-existent keys in overrides
        all_keys = self._get_all_keys()
        for key in overrides.keys():
            if key not in all_keys:
                warnings.warn(f"Override key '{key}' does not exist in the structure. Available keys: {all_keys}")
            else:
                # Apply override
                result._set_nested(key, overrides[key])
        
        # Freeze the result
        object.__setattr__(result, '_frozen', True)
        return result
    
    # Pytree methods
    def tree_flatten(self) -> Tuple[List[Any], List[str]]:
        """Flatten the dict into (values, keys) for pytree traversal."""
        keys = sorted(self.keys())
        values = [self[k] for k in keys]
        return values, keys
    
    @classmethod
    def tree_unflatten(cls, keys: List[str], values: List[Any]) -> 'DottedDict':
        """Reconstruct DottedDict from flattened form."""
        result = cls(zip(keys, values), _allow_mutations=True)
        object.__setattr__(result, '_frozen', True)
        return result


# Register with JAX (if available)
try:
    import jax
    jax.tree_util.register_pytree_node(
        DottedDict,
        DottedDict.tree_flatten,
        DottedDict.tree_unflatten
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# Register with Optree (if available) - used by PyTorch
try:
    import optree
    optree.register_pytree_node(
        DottedDict,
        DottedDict.tree_flatten,
        DottedDict.tree_unflatten,
        namespace="dotted_dict"
    )
    OPTREE_AVAILABLE = True
except ImportError:
    OPTREE_AVAILABLE = False


# Example usage
if __name__ == "__main__":
    print("=== Immutability Test ===")
    if JAX_AVAILABLE:
        import jax.numpy as jnp
        
        # Create immutable structure
        data = DottedDict({"a": 1, "b": 2})
        print(f"Original data: {dict(data)}")
        
        # Try to modify (should fail)
        try:
            data["a"] = 100
            print("ERROR: Should have raised TypeError!")
        except TypeError as e:
            print(f"✓ Correctly prevented mutation: {e}")
        
        try:
            data.c = 3
            print("ERROR: Should have raised TypeError!")
        except TypeError as e:
            print(f"✓ Correctly prevented mutation: {e}")
        
        print("\n=== Copy with Overrides ===")
        original = DottedDict({"x": jnp.array([1.0, 2.0]), "y": jnp.array([3.0])})
        print(f"Original x: {original.x}")
        
        # Create modified copy
        modified = original.copy(x=jnp.array([10.0, 20.0]))
        print(f"Modified x: {modified.x}")
        print(f"Original x (unchanged): {original.x}")
        print(f"Are they the same object? {original is modified}")  # False
        
        print("\n=== Metadata-based Initialization ===")
        metadata = [
            ("layer1.weights", jnp.float32, ("input", "output"), (3, 4)),
            ("layer1.bias", jnp.float32, ("output",), (4,)),
        ]
        
        params = DottedDict.zeros(metadata)
        print(f"params.layer1.weights shape: {params.layer1.weights.shape}")
        print(f"params.layer1.weights:\n{params.layer1.weights}")
        
        # Try to modify (should fail)
        try:
            params.layer1.weights = jnp.ones((3, 4))
            print("ERROR: Should have raised TypeError!")
        except TypeError as e:
            print(f"✓ Correctly prevented mutation: {e}")
        
        # Create modified version
        new_weights = jnp.ones((3, 4))
        updated_params = params.copy(**{"layer1.weights": new_weights})
        print(f"\nUpdated params.layer1.weights:\n{updated_params.layer1.weights}")
        print(f"Original params.layer1.weights (unchanged):\n{params.layer1.weights}")
        
        print("\n=== Functional Programming Style ===")
        # Demonstrate functional update pattern
        def update_weights(params, new_weights):
            """Functional update - returns new params without modifying original."""
            return params.copy(**{"layer1.weights": new_weights})
        
        params_v1 = DottedDict.zeros(metadata)
        params_v2 = update_weights(params_v1, jnp.ones((3, 4)) * 2)
        params_v3 = update_weights(params_v2, jnp.ones((3, 4)) * 3)
        
        print(f"v1 weights sum: {params_v1.layer1.weights.sum()}")  # 0.0
        print(f"v2 weights sum: {params_v2.layer1.weights.sum()}")  # 24.0
        print(f"v3 weights sum: {params_v3.layer1.weights.sum()}")  # 36.0
        
        print("\n=== Hashability ===")
        # Immutable objects can be hashed
        d1 = DottedDict({"a": 1, "b": 2})
        d2 = DottedDict({"a": 1, "b": 2})
        print(f"d1 hash: {hash(d1)}")
        print(f"d2 hash: {hash(d2)}")
        
        # Can use as dict keys
        lookup = {d1: "first"}
        print(f"Can use as dict key: {lookup[d1]}")
        
    else:
        print("\n[JAX not available - install with: pip install jax]")
