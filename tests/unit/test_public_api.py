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
