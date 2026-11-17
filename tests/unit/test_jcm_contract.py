"""Tests for the jax-gcm dependency contract (:mod:`jem.components.jcm.contract`).

These are the tests that turn a jax-gcm rename into a named failure here
instead of an ``AttributeError`` or a ``KeyError`` in the middle of somebody's
coupled run. Every one of them is cheap: the only model built is the smallest
SPEEDY configuration there is, shared across the module, and it is built only
to resolve the diagnostics keys and the attributes ``Model`` sets on instances
rather than on the class.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.resources
import re
from pathlib import Path

import jcm
import pytest
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.surface.surface_exchange import SurfaceExchange as JcmSurfaceExchange
from jcm.predictions import ModelPredictions
from jcm.terrain import TerrainData

from jem.components.jcm.contract import (
    JCM_INTEGRATION_POINTS,
    JCM_SUPPORTED_REV,
    JCM_SUPPORTED_VERSION,
    IntegrationPoint,
)

# The smallest configuration SPEEDY physics supports: T21 on jcm's matching
# (64, 32) nodal grid, and 5 levels (its convective cloud-top search needs
# kx >= 5). Mirrors tests/unit/test_jcm_component.py.
LAYERS = 5
TRUNCATION = 21

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "tests.yml"


@pytest.fixture(scope="module")
def model() -> Model:
    """Build the cheapest real jcm model, once for the whole module."""
    coords = get_speedy_coords(layers=LAYERS, spectral_truncation=TRUNCATION)
    return Model(coords=coords, terrain=TerrainData.aquaplanet(coords))


def _resolve(target: str):
    """Import ``target`` as a module, or as an attribute of its parent module.

    Entries name either a module (``"jcm.runners"``) or something inside one
    (``"jcm.model.Model"``), and the contract list should not have to say
    which -- the distinction is jax-gcm's to change.
    """
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError:
        module_name, _, name = target.rpartition(".")
        return getattr(importlib.import_module(module_name), name)


#: Classes whose contract attributes are set on instances, not on the class,
#: so ``hasattr(cls, name)`` is not the right question to ask. Each entry maps
#: a contract ``target`` to a callable returning a representative instance;
#: nothing is read off these instances beyond whether the attribute exists.
_INSTANCE_FACTORIES = {
    "jcm.model.Model": lambda model: model,
    # ``ModelPredictions.__init__`` only stores what it is handed, and the
    # contract cares about the names it stores under -- so an empty one is a
    # faithful probe and costs nothing.
    "jcm.predictions.ModelPredictions": lambda model: ModelPredictions(
        None, None, None
    ),
}


def _has_attribute(point: IntegrationPoint, model: Model) -> bool:
    """Report whether the name in ``point`` still exists on the installed jcm."""
    owner = _resolve(point.target)
    if hasattr(owner, point.attribute):
        return True
    # A dataclass field without a default (``RunState.dynamics``,
    # ``ForcingData.sea_surface_temperature``) is not an attribute of the
    # class, but it is what the adapter reads off an instance.
    if dataclasses.is_dataclass(owner) and point.attribute in {
        f.name for f in dataclasses.fields(owner)
    }:
        return True
    factory = _INSTANCE_FACTORIES.get(point.target)
    return factory is not None and hasattr(factory(model), point.attribute)


def _missing(point: IntegrationPoint, detail: str = "") -> str:
    """Build the failure message: what moved, what needed it, and since when."""
    if point.access.startswith("diagnostics"):
        what = (
            f"{point.target.upper()} physics no longer publishes the"
            f" diagnostics entry {point.attribute!r}"
        )
    else:
        what = (
            f"jax-gcm renamed or removed {point.target}.{point.attribute}"
            f" ({point.access})"
        )
    return (
        f"{what}{detail}. JAX-ESM supports jax-gcm at {JCM_SUPPORTED_REV},"
        f" where it existed and was used for: {point.used_for} Fix the adapter"
        " and update the entry in jem/components/jcm/contract.py, or bump"
        " JCM_SUPPORTED_REV if the installed jcm is deliberately older or"
        " newer."
    )


def _points(access: str) -> list[IntegrationPoint]:
    """Select the contract entries whose access kind is ``access``."""
    return [p for p in JCM_INTEGRATION_POINTS if p.access.split(",")[0] == access]


def _ids(points: list[IntegrationPoint]) -> list[str]:
    """Name each parametrised case after the jax-gcm name it covers."""
    return [f"{p.target}.{p.attribute}" for p in points]


_ATTRIBUTE_POINTS = _points("public") + _points("private")
_DIAGNOSTICS_POINTS = _points("diagnostics")
_PACKAGE_DATA_POINTS = _points("package data")


def test_every_integration_point_is_classified():
    """Check no entry opts out of being tested by mistyping its access kind."""
    checked = set(_ATTRIBUTE_POINTS + _DIAGNOSTICS_POINTS + _PACKAGE_DATA_POINTS)
    unchecked = [p for p in JCM_INTEGRATION_POINTS if p not in checked]
    assert not unchecked, (
        "These contract entries have an unrecognised `access` and are"
        f" therefore checked by nothing: {_ids(unchecked)}. Use 'public',"
        " 'private', 'diagnostics' or 'package data'."
    )


@pytest.mark.parametrize("point", _ATTRIBUTE_POINTS, ids=_ids(_ATTRIBUTE_POINTS))
def test_jcm_attribute_still_exists(point: IntegrationPoint, model: Model):
    """Check every jax-gcm name JAX-ESM calls is still there, public or private."""
    assert _has_attribute(point, model), _missing(point)


@pytest.mark.parametrize("point", _DIAGNOSTICS_POINTS, ids=_ids(_DIAGNOSTICS_POINTS))
def test_diagnostics_field_still_exists(point: IntegrationPoint):
    """Check every surface-exchange field JAX-ESM reads still exists.

    Every physics package that resolves a surface (SPEEDY, ECHAM) fills the
    SAME package-independent ``SurfaceExchange`` struct (jax-gcm#754,
    #911/#914), so this needs no model build and no per-package branch:
    every ``"diagnostics"`` entry names one of its field names, checked
    directly (:func:`dataclasses.fields`).
    """
    assert point.target == "surface_exchange", (
        f"{point.target!r} diagnostics are not covered by this test; only"
        " 'surface_exchange' (the package-independent contract) is."
    )
    names = {f.name for f in dataclasses.fields(JcmSurfaceExchange)}
    assert point.attribute in names, _missing(
        point, f"; SurfaceExchange's fields are {sorted(names)}"
    )


@pytest.mark.parametrize(
    "point", _PACKAGE_DATA_POINTS, ids=_ids(_PACKAGE_DATA_POINTS)
)
def test_jcm_package_data_still_shipped(point: IntegrationPoint):
    """Check the YAML and climatology JAX-ESM reads out of the jcm wheel ship."""
    resource = importlib.resources.files(point.target)
    for part in point.attribute.split("/"):
        resource = resource / part
    assert resource.is_file() or resource.is_dir(), _missing(point)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _jcm_imports() -> dict[tuple[str, bool], list[str]]:
    """Return every jax-gcm name ``jem/`` imports, as ``(name, is_module) -> locations``.

    Parses the source of every ``.py`` file under ``jem/`` and records

    * ``import jcm.x.y`` / ``import jcm.x.y as z`` as the module ``jcm.x.y``;
    * ``from jcm.x import a, b`` as the names ``jcm.x.a`` and ``jcm.x.b``;
    * ``alias.attr`` where ``alias`` is a name bound by an
      ``import jcm.x as alias`` statement, as the name ``jcm.x.attr`` (dunder
      attributes such as ``jcm.__file__`` are skipped).

    Nothing is imported, so the result cannot depend on import side effects or
    on which optional dependency is installed. What it does not see: a
    ``jcm`` name reached through a *value* rather than an import -- an
    attribute of an object a jax-gcm function returned (``model.dt_si``, the
    constants read off ``jcm.constants.physical_constants``), a name built as
    a string for ``importlib``, or one named in a YAML config. Those are listed
    in the contract by hand, so this check is a floor under the contract, not
    the whole of it.
    """
    found: dict[tuple[str, bool], list[str]] = {}
    for path in sorted((REPO_ROOT / "jem").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        where = path.relative_to(REPO_ROOT)
        module_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "jcm" or alias.name.startswith("jcm."):
                        found.setdefault((alias.name, True), []).append(f"{where}:{node.lineno}")
                        if alias.asname:
                            module_aliases[alias.asname] = alias.name
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
                and (node.module == "jcm" or node.module.startswith("jcm."))
            ):
                for alias in node.names:
                    found.setdefault((f"{node.module}.{alias.name}", False), []).append(
                        f"{where}:{node.lineno}"
                    )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in module_aliases
                and not node.attr.startswith("__")
            ):
                dotted = f"{module_aliases[node.value.id]}.{node.attr}"
                found.setdefault((dotted, False), []).append(f"{where}:{node.lineno}")
    return found


def _covered_by_an_integration_point(dotted: str, is_module: bool) -> bool:
    """Report whether some entry of the contract accounts for ``dotted``.

    An imported *name* (``from jcm.model import Model``, ``c.p0``) must be
    exactly the ``target.attribute`` of an entry. An imported *module*
    (``import jcm.constants as c``) names no attribute itself, so it is
    covered when it is the ``target`` of an entry or a dotted prefix of one:
    the entries for the attributes read off it are what accounts for it.
    """
    for point in JCM_INTEGRATION_POINTS:
        if dotted == f"{point.target}.{point.attribute}":
            return True
        if is_module and (
            dotted == point.target or point.target.startswith(dotted + ".")
        ):
            return True
    return False


def test_every_jcm_import_under_jem_is_an_integration_point():
    """Check no jax-gcm name ``jem/`` imports is missing from the contract.

    The contract is only worth its name if it cannot rot: a name imported at
    module scope (the Winton ice's ``jcm.physics.speedy.params``) breaks
    ``import jem.components`` when jax-gcm renames it, and the per-entry tests
    above can only check the names somebody remembered to list. This reads the
    imports from the source (see :func:`_jcm_imports` for what that does not
    see) and fails naming each one no entry covers and where it is imported.
    """
    uncovered = {
        dotted: places
        for (dotted, is_module), places in _jcm_imports().items()
        if not _covered_by_an_integration_point(dotted, is_module)
    }
    listing = "; ".join(
        f"{name} (imported at {', '.join(places[:3])})"
        for name, places in sorted(uncovered.items())
    )
    assert not uncovered, (
        "jem/ imports these jax-gcm names, which no entry of"
        " JCM_INTEGRATION_POINTS (jem/components/jcm/contract.py) covers, so a"
        f" rename on the jax-gcm side would not be reported by name: {listing}."
        " Add an entry saying what each is used for."
    )


def test_the_jcm_import_scan_is_not_vacuous():
    """Check the import scan finds jax-gcm names at all.

    An empty scan (a moved source tree, a changed import style) would make the
    coverage test above pass for nothing.
    """
    found = {dotted for dotted, _ in _jcm_imports()}
    for expected in (
        "jcm.model.Model",
        "jcm.physics.speedy.params.SurfaceFluxParameters",
        "jcm.constants.p0",
        "jcm.constants",
    ):
        assert expected in found, f"The import scan no longer finds {expected}."


def test_workflow_pins_the_supported_revision():
    """Check CI checks out exactly the revision the contract says is supported.

    The pin is worth nothing if the workflow and the contract can drift, so
    the two are tied together here rather than by a comment asking a reviewer
    to remember.
    """
    text = WORKFLOW.read_text()
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml arrives with hydra-core
        match = re.search(r"^\s*JCM_REV:\s*(\S+)\s*$", text, re.MULTILINE)
        assert match is not None, f"No JCM_REV in {WORKFLOW}"
        workflow_rev = match.group(1)
    else:
        workflow_rev = yaml.safe_load(text)["env"]["JCM_REV"]

    assert workflow_rev == JCM_SUPPORTED_REV, (
        f"{WORKFLOW.name} checks out jax-gcm at {workflow_rev} but"
        " jem/components/jcm/contract.py says JAX-ESM is supported against"
        f" {JCM_SUPPORTED_REV}. Bump both, or CI is not testing what the"
        " contract claims."
    )


def test_the_dockerfile_installs_the_supported_revision():
    """Check the image is built against the revision the contract supports.

    The Dockerfile clones jax-gcm at a sha of its own, and an image built at
    another one installs cleanly -- the version floor is satisfied -- and then
    fails on `import jem.components`, which is the image's entry point.
    """
    dockerfile = WORKFLOW.parents[2] / "Dockerfile"
    match = re.search(r"^ARG JCM_REV=(\S+)\s*$", dockerfile.read_text(), re.MULTILINE)
    assert match is not None, f"No `ARG JCM_REV=` in {dockerfile}"
    assert match.group(1) == JCM_SUPPORTED_REV, (
        f"{dockerfile.name} builds against jax-gcm {match.group(1)} but"
        " jem/components/jcm/contract.py says JAX-ESM is supported against"
        f" {JCM_SUPPORTED_REV}. Bump it with the pin."
    )


def test_the_readthedocs_build_installs_the_supported_revision():
    """Check the hosted docs are built against the revision the contract supports.

    Read the Docs cannot `pip install` the project (the `jcm` floor is not on
    PyPI), so `.readthedocs.yaml` installs jax-gcm from git at a sha of its
    own. A docs build at another one succeeds on install and then renders the
    API pages of an adapter that does not match the atmosphere it documents.
    """
    config = WORKFLOW.parents[2] / ".readthedocs.yaml"
    shas = re.findall(
        r"git\+https://github\.com/climate-analytics-lab/jax-gcm@([0-9a-f]+)",
        config.read_text(),
    )
    assert shas, f"No `git+https://github.com/climate-analytics-lab/jax-gcm@<sha>` in {config}"
    assert set(shas) == {JCM_SUPPORTED_REV}, (
        f"{config.name} installs jax-gcm {sorted(set(shas))} but"
        " jem/components/jcm/contract.py says JAX-ESM is supported against"
        f" {JCM_SUPPORTED_REV}. Bump it with the pin."
    )


def test_the_jcm_floor_is_the_supported_version():
    """Check `pip` cannot resolve `jcm` to a release older than the contract's.

    The requirement's floor is what `pip install jax-esm` resolves against.
    One lower than the version jax-gcm reports at the supported revision
    admits older releases from PyPI, and those install without complaint and
    then fail on import: jcm 3.0.0 lacks names the adapter imports at module
    scope.
    """
    import tomllib

    pyproject = tomllib.loads((WORKFLOW.parents[2] / "pyproject.toml").read_text())
    requirements = [
        requirement
        for requirement in pyproject["project"]["dependencies"]
        if re.match(r"jcm\b", requirement)
    ]
    assert requirements == [f"jcm>={JCM_SUPPORTED_VERSION}"], (
        f"pyproject.toml requires {requirements}, but the supported jax-gcm"
        f" revision reports {JCM_SUPPORTED_VERSION}: the floor must be that"
        " version, so that no older jcm release satisfies it."
    )


@pytest.mark.parametrize(
    "doc",
    ["README.md", "docs/source/getting_started.rst", "docs/source/developers.rst"],
)
def test_install_instructions_check_out_the_supported_revision(doc):
    """Check the documented jax-gcm install checks out the pinned revision.

    A user following the install steps should get the jax-gcm CI tested,
    not whatever ``dev`` has moved to since.
    """
    text = (WORKFLOW.parents[2] / doc).read_text()
    assert f"checkout {JCM_SUPPORTED_REV}" in text, (
        f"{doc} does not install jax-gcm at {JCM_SUPPORTED_REV}, the revision"
        " jem/components/jcm/contract.py supports. Bump it with the pin."
    )


def test_required_jobs_use_the_pin_and_the_canary_tracks_dev():
    """Check every required job uses the pin and only the canary tracks ``dev``.

    Pinning is only reproducible if *all* the required jobs are pinned -- one
    leftover ``ref: dev`` reintroduces the failure mode for whichever job kept
    it -- and the canary is only an early warning if it is allowed to fail.
    """
    yaml = pytest.importorskip("yaml", reason="pyyaml is needed to parse the workflow")
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]

    refs = {
        name: step["with"]["ref"]
        for name, job in jobs.items()
        for step in job["steps"]
        if step.get("with", {}).get("repository") == "climate-analytics-lab/jax-gcm"
    }
    assert refs, "No job checks out jax-gcm at all"

    tracking_dev = {name for name, ref in refs.items() if ref == "dev"}
    assert tracking_dev == {"canary-jcm-dev"}, (
        f"Jobs checking out jax-gcm `dev`: {sorted(tracking_dev)}. Only the"
        " canary may; every required job must use the JCM_REV pin so a"
        " JAX-ESM commit that passes today still passes tomorrow."
    )
    assert jobs["canary-jcm-dev"]["continue-on-error"] is True, (
        "The canary must be continue-on-error: jax-gcm `dev` breaking is not a"
        " reason to block a JAX-ESM pull request."
    )


def test_installed_jcm_matches_contract():
    """Check the jcm in this environment is the version the contract names.

    The version is read from ``jcm.__version__``, the source of the checkout
    that is actually imported, and not from the distribution metadata: an
    editable install records its version once, when it is installed, so a
    checkout that has since been moved to another revision keeps advertising
    the old one to ``importlib.metadata``. That is precisely the situation a
    pin bump creates, and a check that passes because the metadata is stale
    would be worse than no check at all.
    """
    installed = getattr(jcm, "__version__", None)
    assert installed == JCM_SUPPORTED_VERSION, (
        f"The imported jcm ({getattr(jcm, '__file__', 'unknown location')})"
        f" reports version {installed}, but JAX-ESM needs jax-gcm"
        f" {JCM_SUPPORTED_VERSION} at revision {JCM_SUPPORTED_REV}. (None"
        " means this jcm defines no `__version__` at all.) The `jcm` on PyPI"
        " is 3.0.0 and cannot run JAX-ESM: install jax-gcm from source at the"
        " pinned revision instead:\n"
        "  git clone https://github.com/climate-analytics-lab/jax-gcm\n"
        f"  cd jax-gcm && git checkout {JCM_SUPPORTED_REV} && pip install -e ."
        "\nIf the move to another jax-gcm is deliberate, update"
        " jem/components/jcm/contract.py (see 'How to bump the pin')."
    )
