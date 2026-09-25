"""Tests for the jax-gcm dependency contract (:mod:`jem.components.jcm.contract`).

These are the tests that turn a jax-gcm rename into a named failure here
instead of an ``AttributeError`` or a ``KeyError`` in the middle of somebody's
coupled run. Every one of them is cheap: the only model built is the smallest
SPEEDY configuration there is, shared across the module, and it is built only
to resolve the diagnostics keys and the attributes ``Model`` sets on instances
rather than on the class.
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def speedy_diagnostics(model: Model) -> dict:
    """Return jax-gcm's own template of one step's SPEEDY diagnostics dict.

    ``Physics.get_empty_data`` is what jcm itself uses to build the zero
    accumulator an averaged run adds into, so its keys and struct fields are
    exactly the ones a real step produces -- without paying for a step. The
    keys themselves are written as dict literals in
    ``jcm/physics/speedy/speedy_terms.py`` (``_diagnostics_from_data``), so
    this template is the closest thing jax-gcm has to an importable
    declaration of them.
    """
    return model.physics.get_empty_data(model.coords)


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
    # ``ComposablePhysics.terms`` is set in ``__init__``, not on the class,
    # so the model's own (real SPEEDY) physics package is the instance to
    # check it against -- no extra construction needed.
    "jcm.physics.composable_physics.ComposablePhysics": lambda model: model.physics,
}


def _has_attribute(point: IntegrationPoint, model: Model) -> bool:
    """Report whether the name in ``point`` still exists on the installed jcm."""
    owner = _resolve(point.target)
    if hasattr(owner, point.attribute):
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
def test_diagnostics_field_still_exists(
    point: IntegrationPoint, speedy_diagnostics: dict
):
    """Check every diagnostics field the surface exchange reads still exists.

    Two kinds of entry share this parametrization since jax-gcm#754 (PR 877):

    - ``target == "speedy"`` is SPEEDY's own private, non-contract wind-
      vector key (``_surface_flux.u0``/``.v0``) --
      ``jem.components.jcm.exchange_fields``'s one remaining package-specific
      read (see its module docstring) -- checked against a real SPEEDY
      diagnostics template, exactly as before #754.
    - ``target == "surface_exchange"`` is jax-gcm's package-independent
      contract struct, checked directly against its own field names
      (:func:`dataclasses.fields`). This needs no model build and no
      per-package branch: every physics package that resolves a surface
      fills the SAME struct, which is the entire point of #754.
    """
    if point.target == "surface_exchange":
        names = {f.name for f in dataclasses.fields(JcmSurfaceExchange)}
        assert point.attribute in names, _missing(
            point, f"; SurfaceExchange's fields are {sorted(names)}"
        )
        return
    assert point.target == "speedy", (
        f"{point.target!r} diagnostics are not covered by this test; only"
        " 'speedy' (a private key) and 'surface_exchange' (the #754"
        " contract) are."
    )
    key, _, field = point.attribute.partition(".")
    assert key in speedy_diagnostics, _missing(
        point, f"; the diagnostics dict holds {sorted(speedy_diagnostics)}"
    )
    assert hasattr(speedy_diagnostics[key], field), _missing(
        point, f"; the {key!r} struct is a {type(speedy_diagnostics[key]).__name__}"
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
        f"The imported jcm reports {installed} but JAX-ESM is supported"
        f" against {JCM_SUPPORTED_VERSION} ({JCM_SUPPORTED_REV}); None means"
        " this jcm defines no `__version__` at all. Check out that revision,"
        " or update jem/components/jcm/contract.py if the move is deliberate."
    )
