"""Unit tests for jem.coupling.forcing_mapper module."""

import pytest
import jax.numpy as jnp
from dataclasses import dataclass

from jem.coupling.forcing_mapper import ForcingMapper, strget, strset
from jem.components.base import (
    create_field_group_class,
    create_component_state_class,
    create_component_forcing_class,
    Component,
    CoupledComponentConfig,
)


# =============================================================================
# Test fixtures - Mock components for testing
# =============================================================================


class MockComponent(Component):
    """Mock component for testing ForcingMapper."""

    def __init__(self, name, grid_shape=(4, 8)):
        super().__init__(CoupledComponentConfig(name=name, timestep=86400.0))
        self.grid_shape = grid_shape

        # Create state class
        self.component_state_class = create_component_state_class(
            prog_cls=create_field_group_class(
                "Prog",
                [
                    ("temperature", float, grid_shape),
                    ("sim_time", float, ()),
                ],
            ),
            phydata_cls=create_field_group_class(
                "Phydata",
                [("heat_flux", float, grid_shape)],
            ),
        )

        # Create forcing class
        self.component_forcing_class = create_component_forcing_class(
            flux_cls=create_field_group_class(
                "Flux",
                [("incoming_heat", float, grid_shape)],
            ),
            scalar_cls=create_field_group_class(
                "Scalar",
                [("external_temp", float, grid_shape)],
            ),
            cls_name=f"{name}Forcing",
        )

    def initialize(self):
        return self.component_state_class.zeros()

    def generate_step_function(self, jitted=False):
        def step(state, forcing, t):
            return state, {}

        return step

    def validate(self):
        pass

    def predictions_to_xarray(self, predictions):
        return None


@pytest.fixture
def mock_components():
    """Create mock atmosphere and ocean components."""
    return {
        "atm": MockComponent("atm"),
        "ocn": MockComponent("ocn"),
    }


@pytest.fixture
def mock_states(mock_components):
    """Create initial states for mock components."""
    states = {}
    for name, comp in mock_components.items():
        state = comp.initialize()
        # Set some non-zero values
        if name == "atm":
            state = state.copy(
                prog_kwargs=dict(temperature=jnp.ones((4, 8)) * 280.0),
                phydata_kwargs=dict(heat_flux=jnp.ones((4, 8)) * 50.0),
            )
        elif name == "ocn":
            state = state.copy(
                prog_kwargs=dict(temperature=jnp.ones((4, 8)) * 290.0),
                phydata_kwargs=dict(heat_flux=jnp.ones((4, 8)) * -50.0),
            )
        states[name] = state
    return states


# =============================================================================
# Tests for strget and strset utility functions
# =============================================================================


class TestStrget:
    """Tests for strget utility function."""

    def test_single_level_access(self):
        """Test accessing single-level attribute."""

        @dataclass
        class Simple:
            value: float

        obj = Simple(value=42.0)
        result = strget(obj, "value")
        assert result == 42.0

    def test_nested_access(self):
        """Test accessing nested attribute with dot notation."""

        @dataclass
        class Inner:
            data: float

        @dataclass
        class Outer:
            inner: Inner

        obj = Outer(inner=Inner(data=3.14))
        result = strget(obj, "inner.data")
        assert result == 3.14

    def test_deeply_nested_access(self):
        """Test accessing deeply nested attribute."""

        @dataclass
        class Level3:
            value: float

        @dataclass
        class Level2:
            level3: Level3

        @dataclass
        class Level1:
            level2: Level2

        obj = Level1(level2=Level2(level3=Level3(value=99.0)))
        result = strget(obj, "level2.level3.value")
        assert result == 99.0

    def test_access_array_attribute(self):
        """Test accessing JAX array attribute."""

        @dataclass
        class Container:
            array: jnp.ndarray

        arr = jnp.array([1.0, 2.0, 3.0])
        obj = Container(array=arr)
        result = strget(obj, "array")
        assert jnp.allclose(result, arr)

    def test_with_component_state(self):
        """Test strget with actual component state structure."""
        prog_cls = create_field_group_class(
            "Prog", [("temperature", float, (2, 2))]
        )
        phydata_cls = create_field_group_class(
            "Phydata", [("flux", float, (2, 2))]
        )
        state_cls = create_component_state_class(prog_cls, phydata_cls)

        state = state_cls.ones()
        result = strget(state, "prog.temperature")
        assert jnp.allclose(result, 1.0)


class TestStrset:
    """Tests for strset utility function."""

    def test_single_level_set(self):
        """Test setting single-level attribute."""

        @dataclass
        class Simple:
            value: float

        obj = Simple(value=0.0)
        strset(obj, "value", 42.0)
        assert obj.value == 42.0

    def test_nested_set(self):
        """Test setting nested attribute with dot notation."""

        @dataclass
        class Inner:
            data: float

        @dataclass
        class Outer:
            inner: Inner

        obj = Outer(inner=Inner(data=0.0))
        strset(obj, "inner.data", 3.14)
        assert obj.inner.data == 3.14

    def test_set_array_attribute(self):
        """Test setting JAX array attribute."""

        @dataclass
        class Container:
            array: jnp.ndarray

        obj = Container(array=jnp.zeros(3))
        new_arr = jnp.array([1.0, 2.0, 3.0])
        strset(obj, "array", new_arr)
        assert jnp.allclose(obj.array, new_arr)


# =============================================================================
# Tests for ForcingMapper class
# =============================================================================


class TestForcingMapperInit:
    """Tests for ForcingMapper initialization."""

    def test_init_with_components_only(self, mock_components):
        """Test initialization with only components."""
        mapper = ForcingMapper(mock_components)

        assert mapper.component_names == ["atm", "ocn"]
        assert mapper.forcing_mappings == {}
        assert mapper.transformations == {}

    def test_init_with_mappings(self, mock_components):
        """Test initialization with forcing mappings."""
        mappings = {
            ("atm", "ocn"): {"prog.temperature": "scalar.external_temp"},
        }
        mapper = ForcingMapper(mock_components, forcing_mappings=mappings)

        assert ("atm", "ocn") in mapper.forcing_mappings
        assert mapper.forcing_mappings[("atm", "ocn")] == {
            "prog.temperature": "scalar.external_temp"
        }

    def test_init_builds_connections(self, mock_components):
        """Test that initialization builds connection graph."""
        mappings = {
            ("atm", "ocn"): {"prog.temperature": "scalar.external_temp"},
            ("ocn", "atm"): {"prog.temperature": "scalar.external_temp"},
        }
        mapper = ForcingMapper(mock_components, forcing_mappings=mappings)

        assert "ocn" in mapper.connections["atm"]
        assert "atm" in mapper.connections["ocn"]

    def test_stores_component_forcing_classes(self, mock_components):
        """Test that forcing classes are stored from components."""
        mapper = ForcingMapper(mock_components)

        assert "atm" in mapper.component_forcing_classes
        assert "ocn" in mapper.component_forcing_classes


class TestForcingMapperAddMapping:
    """Tests for add_forcing_mapping method."""

    def test_add_new_mapping(self, mock_components):
        """Test adding a new forcing mapping."""
        mapper = ForcingMapper(mock_components)

        mapper.add_forcing_mapping(
            "atm", "ocn", {"prog.temperature": "scalar.external_temp"}
        )

        assert ("atm", "ocn") in mapper.forcing_mappings
        assert mapper.connections["atm"] == ["ocn"]

    def test_add_multiple_mappings(self, mock_components):
        """Test adding multiple mappings."""
        mapper = ForcingMapper(mock_components)

        mapper.add_forcing_mapping(
            "atm", "ocn", {"prog.temperature": "scalar.external_temp"}
        )
        mapper.add_forcing_mapping(
            "ocn", "atm", {"prog.temperature": "scalar.external_temp"}
        )

        assert len(mapper.forcing_mappings) == 2


class TestForcingMapperAddTransformation:
    """Tests for add_transformation method."""

    def test_add_transformation(self, mock_components):
        """Test adding a transformation function."""
        mapper = ForcingMapper(mock_components)

        def scale_by_two(x):
            return x * 2.0

        mapper.add_transformation(
            "atm", "ocn", "prog.temperature", "scalar.external_temp", scale_by_two
        )

        key = ("atm", "ocn", "prog.temperature", "scalar.external_temp")
        assert key in mapper.transformations
        # Test the transformation works
        result = mapper.transformations[key](jnp.array([1.0, 2.0]))
        assert jnp.allclose(result, jnp.array([2.0, 4.0]))


class TestForcingMapperMapForcings:
    """Tests for map_forcings method."""

    def test_map_forcings_creates_forcing_for_each_component(
        self, mock_components, mock_states
    ):
        """Test that map_forcings creates forcing dict for each component."""
        mapper = ForcingMapper(mock_components)
        forcings = mapper.map_forcings(mock_states)

        assert "atm" in forcings
        assert "ocn" in forcings

    def test_map_forcings_with_mapping(self, mock_components, mock_states):
        """Test that map_forcings applies the mapping correctly."""
        mapper = ForcingMapper(mock_components)
        mapper.add_forcing_mapping(
            "atm", "ocn", {"prog.temperature": "scalar.external_temp"}
        )

        forcings = mapper.map_forcings(mock_states)

        # Ocean should receive atmosphere temperature as external_temp
        assert jnp.allclose(
            forcings["ocn"].scalar.external_temp,
            mock_states["atm"].prog.temperature,
        )

    def test_map_forcings_with_transformation(self, mock_components, mock_states):
        """Test that map_forcings applies transformations."""
        mapper = ForcingMapper(mock_components)
        mapper.add_forcing_mapping(
            "atm", "ocn", {"prog.temperature": "scalar.external_temp"}
        )

        # Add transformation that converts temperature
        def kelvin_to_celsius(t):
            return t - 273.15

        mapper.add_transformation(
            "atm", "ocn", "prog.temperature", "scalar.external_temp", kelvin_to_celsius
        )

        forcings = mapper.map_forcings(mock_states)

        expected = mock_states["atm"].prog.temperature - 273.15
        assert jnp.allclose(forcings["ocn"].scalar.external_temp, expected)

    def test_map_forcings_accumulates_from_multiple_sources(self, mock_components):
        """Test that forcings accumulate when multiple sources map to same target."""
        # Create a third component
        components = {
            "atm": MockComponent("atm"),
            "ocn": MockComponent("ocn"),
            "lnd": MockComponent("lnd"),
        }

        mapper = ForcingMapper(components)

        # Both atm and lnd contribute to ocean forcing
        mapper.add_forcing_mapping(
            "atm", "ocn", {"phydata.heat_flux": "flux.incoming_heat"}
        )
        mapper.add_forcing_mapping(
            "lnd", "ocn", {"phydata.heat_flux": "flux.incoming_heat"}
        )

        # Create states with known heat fluxes
        states = {
            "atm": components["atm"].initialize().copy(
                phydata_kwargs=dict(heat_flux=jnp.ones((4, 8)) * 30.0)
            ),
            "ocn": components["ocn"].initialize(),
            "lnd": components["lnd"].initialize().copy(
                phydata_kwargs=dict(heat_flux=jnp.ones((4, 8)) * 20.0)
            ),
        }

        forcings = mapper.map_forcings(states)

        # Ocean should receive sum of atm and lnd heat fluxes
        assert jnp.allclose(forcings["ocn"].flux.incoming_heat, 50.0)


class TestForcingMapperCoupleComponents:
    """Tests for couple_components method."""

    def test_couple_components_calls_map_forcings(self, mock_components, mock_states):
        """Test that couple_components returns same as map_forcings."""
        mapper = ForcingMapper(mock_components)
        mapper.add_forcing_mapping(
            "atm", "ocn", {"prog.temperature": "scalar.external_temp"}
        )

        forcings1 = mapper.map_forcings(mock_states)
        forcings2 = mapper.couple_components(mock_states)

        # Should produce identical results
        assert jnp.allclose(
            forcings1["ocn"].scalar.external_temp,
            forcings2["ocn"].scalar.external_temp,
        )
