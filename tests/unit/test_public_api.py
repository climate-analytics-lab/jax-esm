"""What ``import jem`` and ``import jem.components`` must give a user.

An export list is the one part of an API that no other test exercises: every
test here imports from the module a name actually lives in, so a name dropped
from an ``__all__`` -- or an ``__all__`` entry that no longer resolves --
breaks nothing until a user follows the documentation.
"""

import importlib
import subprocess
import sys

import pytest

import jem
import jem.components

# What a coupled run is written against: the coupler, the contract a component
# implements, and the state types that travel between them.
CORE_NAMES = (
    "Carry",
    "Component",
    "CoupledCarry",
    "Coupler",
    "CouplingTime",
    "Diagnostics",
    "Exchanger",
    "SupportsBind",
    "SupportsCheckpoint",
    "SupportsXarray",
    "TimeAxis",
    # The helpers an exchanger reaches into a nested coupler's carry with.
    "nested_carry",
    "with_nested_carry",
    # The declarative form of an exchanger, and the standard wiring.
    "Exchange",
    "ExchangeSpec",
    "default_exchangers",
    "default_exchanges",
    "default_workflow",
    # What of a component's carry the table says somebody else supplies.
    "exchanged_fields",
    # Reading and writing one field of a coupled carry by the same address.
    "read_field",
    "replace_field",
    # Writing a chunk of a run out: the labelling and the reduction, each on
    # its own (the health gate needs the unreduced chunk) and composed.
    "chunk_datasets",
    "datasets_for_chunk",
    "postprocess",
    "postprocess_datasets",
    "write_chunk",
    # The run loop, and the gate it gives a chunk.
    "RunResult",
    "default_health_check",
    "run_chunked",
)

# What a coupled run is built out of: the atmosphere wrapper, the slab models
# with their parameter structs, the grid they share and the boundary-data
# loader they read climatologies with.
COMPONENT_NAMES = (
    "JCMComponent",
    "SlabAtmosphereModel",
    "SlabAtmosphereParameters",
    "SlabGrid",
    "SlabLandModel",
    "SlabLandParameters",
    "SlabOceanModel",
    "SlabOceanParameters",
    "SlabSeaiceModel",
    "SlabSeaiceParameters",
    "load_monthly_climatology",
)


@pytest.mark.parametrize("module_name", ["jem", "jem.components", "jem.base"])
def test_every_exported_name_resolves(module_name):
    """No ``__all__`` entry may name something the module cannot produce."""
    module = importlib.import_module(module_name)
    for name in module.__all__:
        assert getattr(module, name) is not None, name


@pytest.mark.parametrize("name", CORE_NAMES)
def test_coupling_core_is_exported_from_jem(name):
    assert name in jem.__all__
    assert hasattr(jem, name)


@pytest.mark.parametrize("name", COMPONENT_NAMES)
def test_components_are_exported(name):
    assert name in jem.components.__all__
    assert hasattr(jem.components, name)


def test_veros_is_exported_lazily():
    """Veros is an optional dependency, so it may not be imported eagerly.

    ``import jem.components`` must work without it installed, which means
    ``VerosComponent`` is resolved by ``__getattr__`` on first use -- and it
    still has to be reachable, or the export is a lie.
    """
    assert "VerosComponent" in jem.components.__all__
    veros = pytest.importorskip("veros")
    del veros
    assert jem.components.VerosComponent.__name__ == "VerosComponent"


def test_importing_jem_does_not_import_the_components():
    """``import jem`` must not drag in the atmosphere.

    The JCM wrapper imports ``jcm.model``, and with it dinosaur and the whole
    physics tree. That belongs to a run that uses an atmosphere, not to
    anyone who imports the coupler.
    """
    source = (
        "import sys, jem;"
        " assert 'jem.components.jcm.component' not in sys.modules,"
        " sorted(m for m in sys.modules if m.startswith('jem'))"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_importing_jem_does_not_import_configurations():
    """``import jem`` must not drag in the recipe door either (issue #131).

    ``jem.configurations`` composes Hydra configs and builds real components
    through ``jem.runners``, exactly the weight ``jem.runners`` itself is kept
    out of ``import jem`` for (see the comment in ``jem/__init__.py``). It is
    reached with ``import jem.configurations`` when a caller actually wants
    the door, never as a side effect of ``import jem``.
    """
    source = (
        "import sys, jem;"
        " assert 'jem.configurations' not in sys.modules,"
        " sorted(m for m in sys.modules if m.startswith('jem'))"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_configurations_is_the_recipe_door():
    """``jem.configurations`` is reachable and carries the door's whole API.

    Not part of ``jem.__all__`` (see the test above) -- reached instead as a
    submodule, ``import jem.configurations`` or ``from jem import
    configurations``, the same way jax-gcm's own ``jcm.configurations`` is.
    """
    import jem.configurations as configurations

    assert callable(configurations.available)
    assert callable(configurations.load)
    assert configurations.LoadedConfiguration is not None
