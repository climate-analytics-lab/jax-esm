"""Tests for component interface."""

import jax.numpy as jnp
import unittest

from jax_esm.components.JCM import JCM

import jcm
from pathlib import Path


class TestJCM(unittest.TestCase):
    """Test component interface."""

    def test_JCM_initialization(self):
        """Test component initialization."""
        model = JCM(model=jcm.model.Model())

    def test_JCM_initialize_state(self):
        """Test state initialization."""
        model = JCM(model=jcm.model.Model())

        state = model.initialize()
        assert hasattr(state.prog, "temperature")
        assert state.prog.temperature.shape == (8, 96, 48)

