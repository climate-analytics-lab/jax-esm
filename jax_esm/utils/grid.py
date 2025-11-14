"""Grid interpolation utilities for coupling different resolution components."""

from typing import Tuple

import jax.numpy as jnp
from jax import Array


class GridInterpolator:
    """Interpolates fields between different grids."""
    
    @staticmethod
    def bilinear_interpolate(
        data: Array,
        source_lat: Array,
        source_lon: Array,
        target_lat: Array,
        target_lon: Array,
    ) -> Array:
        """Bilinear interpolation between grids.
        
        Args:
            data: Source data array (..., lat, lon)
            source_lat: Source latitude coordinates
            source_lon: Source longitude coordinates
            target_lat: Target latitude coordinates
            target_lon: Target longitude coordinates
            
        Returns:
            Interpolated data on target grid
        """
        # Get indices for interpolation
        lat_idx, lat_weights = GridInterpolator._get_interp_indices_weights(
            source_lat, target_lat
        )
        lon_idx, lon_weights = GridInterpolator._get_interp_indices_weights(
            source_lon, target_lon
        )
        
        # Perform bilinear interpolation
        # Get the four corner points
        data_00 = data[..., lat_idx[0], lon_idx[0]]
        data_01 = data[..., lat_idx[0], lon_idx[1]]
        data_10 = data[..., lat_idx[1], lon_idx[0]]
        data_11 = data[..., lat_idx[1], lon_idx[1]]
        
        # Bilinear weights
        w00 = (1 - lat_weights) * (1 - lon_weights)
        w01 = (1 - lat_weights) * lon_weights
        w10 = lat_weights * (1 - lon_weights)
        w11 = lat_weights * lon_weights
        
        # Interpolate
        interp_data = (
            w00[..., None, None] * data_00 +
            w01[..., None, None] * data_01 +
            w10[..., None, None] * data_10 +
            w11[..., None, None] * data_11
        )
        
        return interp_data
    
    @staticmethod
    def conservative_regrid(
        data: Array,
        source_lat_bounds: Array,
        source_lon_bounds: Array,
        target_lat_bounds: Array,
        target_lon_bounds: Array,
    ) -> Array:
        """Conservative regridding that preserves integrated quantities.
        
        Args:
            data: Source data array
            source_lat_bounds: Source latitude cell boundaries
            source_lon_bounds: Source longitude cell boundaries
            target_lat_bounds: Target latitude cell boundaries
            target_lon_bounds: Target longitude cell boundaries
            
        Returns:
            Regridded data preserving area integrals
        """
        # Calculate area overlap fractions
        overlap_fractions = GridInterpolator._calculate_overlap_fractions(
            source_lat_bounds, source_lon_bounds,
            target_lat_bounds, target_lon_bounds
        )
        
        # Apply conservative remapping
        # This is a simplified version - full implementation would handle
        # all edge cases and use more efficient algorithms
        n_target_lat = len(target_lat_bounds) - 1
        n_target_lon = len(target_lon_bounds) - 1
        
        regridded = jnp.zeros((n_target_lat, n_target_lon))
        
        for i in range(n_target_lat):
            for j in range(n_target_lon):
                # Sum contributions from all source cells
                regridded = regridded.at[i, j].set(
                    jnp.sum(data * overlap_fractions[i, j])
                )
        
        return regridded
    
    @staticmethod
    def _get_interp_indices_weights(
        source_coords: Array,
        target_coords: Array,
    ) -> Tuple[Tuple[Array, Array], Array]:
        """Get interpolation indices and weights.
        
        Args:
            source_coords: Source coordinate array
            target_coords: Target coordinate array
            
        Returns:
            Tuple of (indices, weights) for interpolation
        """
        # Find bracketing indices
        idx_below = jnp.searchsorted(source_coords, target_coords, side='right') - 1
        idx_above = jnp.minimum(idx_below + 1, len(source_coords) - 1)
        idx_below = jnp.maximum(idx_below, 0)
        
        # Calculate weights
        coord_below = source_coords[idx_below]
        coord_above = source_coords[idx_above]
        
        weights = jnp.where(
            coord_above > coord_below,
            (target_coords - coord_below) / (coord_above - coord_below),
            0.0
        )
        
        return (idx_below, idx_above), weights
    
    @staticmethod
    def _calculate_overlap_fractions(
        source_lat_bounds: Array,
        source_lon_bounds: Array,
        target_lat_bounds: Array,
        target_lon_bounds: Array,
    ) -> Array:
        """Calculate area overlap fractions between source and target cells.
        
        This is a placeholder for a more sophisticated implementation.
        """
        # Simplified implementation
        n_source_lat = len(source_lat_bounds) - 1
        n_source_lon = len(source_lon_bounds) - 1
        n_target_lat = len(target_lat_bounds) - 1
        n_target_lon = len(target_lon_bounds) - 1
        
        # For now, return uniform weights (not truly conservative)
        # A full implementation would calculate actual overlap areas
        fractions = jnp.zeros((n_target_lat, n_target_lon, n_source_lat, n_source_lon))
        
        # This is a placeholder - real implementation would calculate
        # actual overlap fractions based on cell boundaries
        for i in range(n_target_lat):
            for j in range(n_target_lon):
                # Find overlapping source cells
                # (simplified - just use nearest neighbor for now)
                src_i = int(i * n_source_lat / n_target_lat)
                src_j = int(j * n_source_lon / n_target_lon)
                fractions = fractions.at[i, j, src_i, src_j].set(1.0)
        
        return fractions
    
    @staticmethod
    def spectral_interpolate(
        data: Array,
        source_resolution: int,
        target_resolution: int,
    ) -> Array:
        """Interpolate spectral data between different resolutions.
        
        Args:
            data: Spectral coefficients
            source_resolution: Source spectral resolution (e.g., T63)
            target_resolution: Target spectral resolution (e.g., T127)
            
        Returns:
            Interpolated spectral coefficients
        """
        # Placeholder for spectral interpolation
        # Real implementation would handle spherical harmonics properly
        if target_resolution > source_resolution:
            # Pad with zeros for higher resolution
            pad_size = target_resolution - source_resolution
            return jnp.pad(data, ((0, pad_size), (0, pad_size)), mode='constant')
        else:
            # Truncate for lower resolution
            return data[:target_resolution, :target_resolution]