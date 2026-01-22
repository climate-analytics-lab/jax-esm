"""Unit tests for jem.components.base module."""

import pytest
import jax.numpy as jnp
import jax
from jem.base.mixin import (
    MethodNotFoundError,
    MethodSignatureNotMatchError,
    verify_functions,
)
from dataclasses import dataclass
from typing import Any, Callable

class TestMixin:
    """Tests for mixin functions """

    def test_function(self):
        """Test that the target object has the correct function and signature."""
        
        @dataclass
        class State:
            f : Callable

        def function_0args():
            pass

        def function_1args(x):
            pass

        def function_2args(x, y):
            pass

        def function_3args(x, y, z):
            pass


        s0 = State(function_0args)
        s1 = State(function_1args)
        s2 = State(function_2args)
        s3 = State(function_3args)

        verify_functions(s0, {"f" : Callable[[], Any]}, verbose=True)
        verify_functions(s1, {"f" : Callable[[Any], Any]}, verbose=True)
        verify_functions(s2, {"f" : Callable[[Any, Any], Any]}, verbose=True)
        verify_functions(s3, {"f" : Callable[[Any, Any, Any], Any]}, verbose=True)
        
        with pytest.raises(MethodNotFoundError):
            verify_functions(s0, {"g" : Callable[[Any, Any, Any], Any]})
 
        with pytest.raises(MethodSignatureNotMatchError):
            verify_functions(s0, {"f" : Callable[[Any, Any, Any], Any]})

