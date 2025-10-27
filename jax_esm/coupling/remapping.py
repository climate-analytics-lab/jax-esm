import numpy as np
from typing import Callable, Literal
try:
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX not available. Generated functions will not be jittable.")


class Remapping:
    """
    Generic 1D interpolator class for earth system modeling coupler.
    
    Provides flux-conservative and non-conservative remapping between two grids
    using matrix multiplication. Generated interpolation functions are JAX-jittable.
    
    Parameters
    ----------
    matrix : np.ndarray
        Remapping matrix of shape (N_out, N_in) for forward remapping
        or (N_in, N_out) for backward remapping.
    n_in : int
        Size of the source/input grid.
    n_out : int
        Size of the destination/output grid.
    direction : {'forward', 'backward', 'bidirectional'}, default='forward'
        Remapping direction(s):
        - 'forward': input grid -> output grid
        - 'backward': output grid -> input grid
        - 'bidirectional': both directions supported
    conservative : bool, default=False
        Whether the remapping is flux-conservative.
    
    Raises
    ------
    ValueError
        If matrix dimensions don't match the specified grid sizes.
    
    Examples
    --------
    >>> matrix = np.array([[0.5, 0.5, 0.0],
    ...                     [0.0, 0.5, 0.5]])
    >>> remap = Remapping(matrix, n_in=3, n_out=2, direction='forward')
    >>> interpolate = remap.get_interpolator()
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> result = interpolate(data)
    """
    
    def __init__(
        self,
        matrix: np.ndarray,
        n_in: int,
        n_out: int,
        direction: Literal['forward', 'backward', 'bidirectional'] = 'forward',
        conservative: bool = False
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.direction = direction
        self.conservative = conservative
        
        # Validate matrix dimensions
        self._validate_matrix(matrix)
        
        # Store the remapping matrix as numpy array for validation
        # Will be converted to JAX array in the generated function
        self.matrix = np.asarray(matrix)
        
    def _validate_matrix(self, matrix: np.ndarray) -> None:
        """
        Validate that matrix dimensions match the grid sizes.
        
        Parameters
        ----------
        matrix : np.ndarray
            The remapping matrix to validate.
            
        Raises
        ------
        ValueError
            If matrix dimensions don't match grid sizes for the specified direction.
        """
        if matrix.ndim != 2:
            raise ValueError(
                f"Remapping matrix must be 2-dimensional, got {matrix.ndim} dimensions"
            )
        
        rows, cols = matrix.shape
        
        if self.direction == 'forward':
            if rows != self.n_out or cols != self.n_in:
                raise ValueError(
                    f"For forward remapping, matrix shape must be ({self.n_out}, {self.n_in}), "
                    f"got ({rows}, {cols})"
                )
        elif self.direction == 'backward':
            if rows != self.n_in or cols != self.n_out:
                raise ValueError(
                    f"For backward remapping, matrix shape must be ({self.n_in}, {self.n_out}), "
                    f"got ({rows}, {cols})"
                )
        elif self.direction == 'bidirectional':
            if rows != self.n_out or cols != self.n_in:
                raise ValueError(
                    f"For bidirectional remapping (forward direction), matrix shape must be "
                    f"({self.n_out}, {self.n_in}), got ({rows}, {cols})"
                )
    
    def get_interpolator(
        self,
        inverse: bool = False,
        use_jax: bool = True
    ) -> Callable:
        """
        Generate a JAX-jittable interpolation function.
        
        Parameters
        ----------
        inverse : bool, default=False
            If True and direction is 'bidirectional', return the inverse remapping
            function (output grid -> input grid).
        use_jax : bool, default=True
            If True and JAX is available, return a jitted JAX function.
            If False, return a NumPy function.
            
        Returns
        -------
        Callable
            Interpolation function that takes data on source grid and returns
            data on destination grid. The function is JAX-jittable.
            
        Raises
        ------
        ValueError
            If inverse remapping is requested but not supported by the direction.
        """
        if inverse and self.direction not in ['backward', 'bidirectional']:
            raise ValueError(
                f"Inverse remapping requested but direction is '{self.direction}'. "
                "Use 'backward' or 'bidirectional' direction."
            )
        
        # Determine which matrix to use
        if self.direction == 'forward' or (self.direction == 'bidirectional' and not inverse):
            matrix = self.matrix
            expected_input_size = self.n_in
        elif self.direction == 'backward' or (self.direction == 'bidirectional' and inverse):
            matrix = self.matrix.T
            expected_input_size = self.n_out
        
        # Convert to JAX array if using JAX
        if use_jax and JAX_AVAILABLE:
            matrix_jax = jnp.array(matrix)
            
            def interpolate_jax(data):
                """JAX-compatible interpolation function."""
                return jnp.dot(matrix_jax, data)
            
            # Return jitted function
            return jit(interpolate_jax)
        else:
            # Return NumPy function
            def interpolate_numpy(data):
                """NumPy interpolation function."""
                data = np.asarray(data)
                if data.shape[0] != expected_input_size:
                    raise ValueError(
                        f"Input data size {data.shape[0]} doesn't match expected "
                        f"source grid size {expected_input_size}"
                    )
                return np.dot(matrix, data)
            
            return interpolate_numpy
    
    def __repr__(self) -> str:
        return (
            f"Remapping(n_in={self.n_in}, n_out={self.n_out}, "
            f"direction='{self.direction}', conservative={self.conservative})"
        )


class Remapping2D:
    """
    2D interpolator class that utilizes Remapping for each dimension.
    
    Performs 2D remapping by applying 1D remapping operations along each dimension.
    Supports both 2D and 3D input (3D input treated as slices).
    
    Parameters
    ----------
    remapping_dim0 : Remapping
        Remapping object for the first dimension (rows).
    remapping_dim1 : Remapping
        Remapping object for the second dimension (columns).
    inverse_dim0 : bool, default=False
        Whether to use inverse remapping for dimension 0.
    inverse_dim1 : bool, default=False
        Whether to use inverse remapping for dimension 1.
    
    Examples
    --------
    >>> # Create remapping for each dimension
    >>> matrix_rows = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> matrix_cols = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    >>> remap_dim0 = Remapping(matrix_rows, n_in=2, n_out=2)
    >>> remap_dim1 = Remapping(matrix_cols, n_in=3, n_out=2)
    >>> remap_2d = Remapping2D(remap_dim0, remap_dim1)
    >>> interp_2d = remap_2d.get_interpolator()
    >>> data = np.random.rand(2, 3)
    >>> result = interp_2d(data)
    >>> print(result.shape)  # (2, 2)
    """
    
    def __init__(
        self,
        remapping_dim0: Remapping,
        remapping_dim1: Remapping,
        inverse_dim0: bool = False,
        inverse_dim1: bool = False
    ):
        self.remapping_dim0 = remapping_dim0
        self.remapping_dim1 = remapping_dim1
        self.inverse_dim0 = inverse_dim0
        self.inverse_dim1 = inverse_dim1
        
        # Determine input and output shapes
        if inverse_dim0:
            self.n_in_dim0 = remapping_dim0.n_out
            self.n_out_dim0 = remapping_dim0.n_in
        else:
            self.n_in_dim0 = remapping_dim0.n_in
            self.n_out_dim0 = remapping_dim0.n_out
            
        if inverse_dim1:
            self.n_in_dim1 = remapping_dim1.n_out
            self.n_out_dim1 = remapping_dim1.n_in
        else:
            self.n_in_dim1 = remapping_dim1.n_in
            self.n_out_dim1 = remapping_dim1.n_out
    
    def get_interpolator(self, use_jax: bool = True) -> Callable:
        """
        Generate a 2D interpolation function.
        
        The generated function handles both 2D and 3D input:
        - 2D input (n_in_dim0, n_in_dim1) -> 2D output (n_out_dim0, n_out_dim1)
        - 3D input (n_slices, n_in_dim0, n_in_dim1) -> 3D output (n_slices, n_out_dim0, n_out_dim1)
        
        Parameters
        ----------
        use_jax : bool, default=True
            If True and JAX is available, return a jitted JAX function.
            
        Returns
        -------
        Callable
            2D interpolation function that is JAX-jittable.
        """
        # Get 1D interpolators for each dimension
        interp_dim0 = self.remapping_dim0.get_interpolator(
            inverse=self.inverse_dim0, use_jax=use_jax
        )
        interp_dim1 = self.remapping_dim1.get_interpolator(
            inverse=self.inverse_dim1, use_jax=use_jax
        )
        
        if use_jax and JAX_AVAILABLE:
            def interpolate_2d_jax(data):
                """
                JAX-compatible 2D interpolation function.
                
                Parameters
                ----------
                data : jax.numpy.ndarray
                    Input data of shape (n_in_dim0, n_in_dim1) or 
                    (n_slices, n_in_dim0, n_in_dim1)
                    
                Returns
                -------
                jax.numpy.ndarray
                    Remapped data with shape (n_out_dim0, n_out_dim1) or
                    (n_slices, n_out_dim0, n_out_dim1)
                """
                original_ndim = data.ndim
                
                if original_ndim == 2:
                    # Apply remapping along dim1 (columns) first
                    temp = jnp.apply_along_axis(interp_dim1, 1, data)
                    # Then apply remapping along dim0 (rows)
                    result = jnp.apply_along_axis(interp_dim0, 0, temp)
                elif original_ndim == 3:
                    # Process each slice
                    def process_slice(slice_data):
                        temp = jnp.apply_along_axis(interp_dim1, 1, slice_data)
                        return jnp.apply_along_axis(interp_dim0, 0, temp)
                    
                    # Vectorize over slices
                    result = jnp.stack([process_slice(data[i]) for i in range(data.shape[0])])
                else:
                    raise ValueError(f"Input must be 2D or 3D, got {original_ndim}D")
                
                return result
            
            return jit(interpolate_2d_jax)
        else:
            def interpolate_2d_numpy(data):
                """
                NumPy 2D interpolation function.
                
                Parameters
                ----------
                data : np.ndarray
                    Input data of shape (n_in_dim0, n_in_dim1) or 
                    (n_slices, n_in_dim0, n_in_dim1)
                    
                Returns
                -------
                np.ndarray
                    Remapped data with shape (n_out_dim0, n_out_dim1) or
                    (n_slices, n_out_dim0, n_out_dim1)
                """
                data = np.asarray(data)
                original_ndim = data.ndim
                
                if original_ndim == 2:
                    if data.shape != (self.n_in_dim0, self.n_in_dim1):
                        raise ValueError(
                            f"Input shape {data.shape} doesn't match expected "
                            f"({self.n_in_dim0}, {self.n_in_dim1})"
                        )
                    # Apply remapping along dim1 (columns) first
                    temp = np.apply_along_axis(interp_dim1, 1, data)
                    # Then apply remapping along dim0 (rows)
                    result = np.apply_along_axis(interp_dim0, 0, temp)
                elif original_ndim == 3:
                    if data.shape[1:] != (self.n_in_dim0, self.n_in_dim1):
                        raise ValueError(
                            f"Input shape {data.shape[1:]} doesn't match expected "
                            f"({self.n_in_dim0}, {self.n_in_dim1})"
                        )
                    # Process each slice
                    result = np.stack([
                        np.apply_along_axis(interp_dim0, 0, 
                                          np.apply_along_axis(interp_dim1, 1, data[i]))
                        for i in range(data.shape[0])
                    ])
                else:
                    raise ValueError(f"Input must be 2D or 3D, got {original_ndim}D")
                
                return result
            
            return interpolate_2d_numpy
    
    def __repr__(self) -> str:
        return (
            f"Remapping2D(\n"
            f"  dim0: {self.remapping_dim0} (inverse={self.inverse_dim0})\n"
            f"  dim1: {self.remapping_dim1} (inverse={self.inverse_dim1})\n"
            f"  input_shape: ({self.n_in_dim0}, {self.n_in_dim1})\n"
            f"  output_shape: ({self.n_out_dim0}, {self.n_out_dim1})\n"
            f")"
        )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Basic 1D Remapping")
    print("=" * 70)
    matrix_1d = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5]
    ])
    remap_1d = Remapping(matrix_1d, n_in=3, n_out=2, direction='forward')
    interp_1d = remap_1d.get_interpolator(use_jax=JAX_AVAILABLE)
    
    data_1d = np.array([1.0, 2.0, 3.0])
    result_1d = interp_1d(data_1d)
    print(f"Input:  {data_1d}")
    print(f"Output: {result_1d}")
    print()
    
    print("=" * 70)
    print("Example 2: 2D Remapping with 2D input")
    print("=" * 70)
    # Create remapping for rows: 3 -> 2
    matrix_rows = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5]
    ])
    remap_rows = Remapping(matrix_rows, n_in=3, n_out=2)
    
    # Create remapping for cols: 4 -> 3
    matrix_cols = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5]
    ])
    remap_cols = Remapping(matrix_cols, n_in=4, n_out=3)
    
    # Create 2D remapping
    remap_2d = Remapping2D(remap_rows, remap_cols)
    interp_2d = remap_2d.get_interpolator(use_jax=JAX_AVAILABLE)
    
    # Test with 2D input
    data_2d = np.arange(12).reshape(3, 4).astype(float)
    result_2d = interp_2d(data_2d)
    print(f"Input shape: {data_2d.shape}")
    print(f"Input:\n{data_2d}")
    print(f"\nOutput shape: {result_2d.shape}")
    print(f"Output:\n{result_2d}")
    print()
    
    print("=" * 70)
    print("Example 3: 2D Remapping with 3D input (slices)")
    print("=" * 70)
    # Test with 3D input (5 slices)
    data_3d = np.arange(60).reshape(5, 3, 4).astype(float)
    result_3d = interp_2d(data_3d)
    print(f"Input shape: {data_3d.shape}")
    print(f"Output shape: {result_3d.shape}")
    print(f"\nFirst slice input:\n{data_3d[0]}")
    print(f"First slice output:\n{result_3d[0]}")
    print()
    
    if JAX_AVAILABLE:
        print("=" * 70)
        print("JAX compatibility verified!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("JAX not available - using NumPy functions")
        print("=" * 70)
