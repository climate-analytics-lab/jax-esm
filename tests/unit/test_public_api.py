"""What ``import jem`` and ``import jem.components`` must give a user.

An export list is the one part of an API that no other test exercises: every
test here imports from the module a name actually lives in, so a name dropped
from an ``__all__`` -- or an ``__all__`` entry that no longer resolves --
breaks nothing until a user follows the documentation.
"""

import importlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

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
    "SupportsClockReset",
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
    "BucketLandModel",
    "BucketLandParameters",
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


def test_veros_is_reachable_but_not_star_exported():
    """``VerosComponent`` is resolved lazily and is not in ``__all__``.

    A name in ``__all__`` is looked up by ``from jem.components import *``,
    which would then fail on an install without Veros, so the optional
    component stays out of the list and reachable by attribute.
    """
    assert "VerosComponent" not in jem.components.__all__
    pytest.importorskip("veros")
    assert jem.components.VerosComponent.__name__ == "VerosComponent"


# Runs `source` in a fresh interpreter in which `import veros` fails, as it does
# on an install without the jittable fork.
_WITHOUT_VEROS = "import sys; sys.modules['veros'] = None;"


def test_star_import_works_without_veros():
    source = _WITHOUT_VEROS + " from jem.components import *; JCMComponent"
    subprocess.run([sys.executable, "-c", source], check=True)


def test_missing_veros_says_which_veros_to_install():
    """The error names the jittable fork, not the PyPI package that cannot work."""
    source = _WITHOUT_VEROS + (
        " import jem.components\n"
        "try:\n"
        "    jem.components.VerosComponent\n"
        "except ImportError as error:\n"
        "    assert 'veros-jittable' in str(error), error\n"
        "    assert 'getting_started.rst' in str(error), error\n"
        "    assert 'PyPI' in str(error), error\n"
        "    assert isinstance(error.__cause__, ImportError), error.__cause__\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_a_broken_veros_install_is_not_relabelled_as_a_missing_one():
    """Only a missing ``veros`` gets the install hint; any other import error passes.

    A ``veros`` that imports but lacks ``runtime_settings`` is a different
    problem, and telling its user to install the fork would hide it.
    """
    source = (
        "import sys, types; sys.modules['veros'] = types.ModuleType('veros')\n"
        "import jem.components\n"
        "try:\n"
        "    jem.components.VerosComponent\n"
        "except ImportError as error:\n"
        "    assert 'veros-jittable' not in str(error), error\n"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_removed_deprecated_aliases_are_gone():
    assert not hasattr(jem.components, "JCM")
    assert not hasattr(jem.components, "Veros")
    with pytest.raises(ImportError):
        importlib.import_module("jem.components.jcm_component")


def test_importing_jem_does_not_import_jcm():
    """``import jem`` must not import the atmosphere package, nor Veros.

    ``jcm/__init__`` imports ``jcm.model``, dinosaur and ``flax.nnx``; the
    places jem needs ``jcm.date`` import it where they use it.
    """
    source = (
        "import sys, jem;"
        " heavy = sorted(m for m in sys.modules"
        " if m.split('.')[0] in ('jcm', 'dinosaur', 'veros'));"
        " assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


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


def test_the_declared_status_follows_the_version():
    """The classifier, the README badge and the docs agree with ``__version__``.

    The version is the one place the project's maturity is decided -- an
    ``a``/``b``/``rc`` segment, or none -- and three other places repeat it in
    words. A release that bumps the version and leaves one of them behind
    publishes a 1.0 that calls itself alpha, which is what this guards.
    """
    stage = re.fullmatch(r"\d+(?:\.\d+)*(?:(a|b|rc)\d+)?", jem.__version__)
    assert stage is not None, f"unrecognised version {jem.__version__!r}"
    word, classifier = {
        "a": ("alpha", "3 - Alpha"),
        "b": ("beta", "4 - Beta"),
        "rc": ("beta", "4 - Beta"),
        None: ("stable", "5 - Production/Stable"),
    }[stage.group(1)]

    root = Path(jem.__file__).resolve().parents[1]
    classifiers = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "classifiers"
    ]
    statuses = [c for c in classifiers if c.startswith("Development Status ::")]
    assert statuses == [f"Development Status :: {classifier}"]

    badge = re.search(r"!\[Status: (\w+)\]", (root / "README.md").read_text())
    assert badge is not None and badge.group(1).lower() == word

    front_page = (root / "docs" / "source" / "index.rst").read_text()
    others = {"alpha", "beta", "stable"} - {word}
    assert f"JEM is {word} software" in front_page
    assert not any(f"JEM is {other} software" in front_page for other in others)
