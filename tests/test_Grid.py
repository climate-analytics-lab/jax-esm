"""Tests for component interface."""

import jax
import jax.numpy as jnp
import pytest
import unittest

from jax_esm.domain.Grid import Grid


class TestGrid(unittest.TestCase):
    """Test component interface."""
    
    def test_grid_initialization(self):
      
        for horizontal_resolution in [21, 31, 42, 85, 106, 119, 170, 213, 340, 425]: 
            grid = Grid(horizontal_resolution = horizontal_resolution)
 
    
   
