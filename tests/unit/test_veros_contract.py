"""Tests that hold every place naming the Veros fork to ``VEROS_SUPPORTED_REV``.

Mirrors the jax-gcm half of ``test_jcm_contract.py``. None of this imports
Veros, which is an optional dependency: the constant lives in
:mod:`jem.components.veros`, whose ``__init__`` imports nothing, and the
other side of each comparison is read as text.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from jem.components.veros import VEROS_SUPPORTED_REV

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
VEROS_REPOSITORY = "meteorologytoday/veros-jittable"


def test_the_pin_is_a_full_commit_sha():
    """A branch or an abbreviated sha does not name one tree."""
    assert re.fullmatch(r"[0-9a-f]{40}", VEROS_SUPPORTED_REV)


def test_the_constant_imports_without_veros():
    """The package that holds the pin must stay importable without Veros.

    Run in a fresh interpreter with ``veros`` made un-importable, so it holds
    wherever the suite happens to run.
    """
    code = (
        "import sys; sys.modules['veros'] = None;"
        " from jem.components.veros import VEROS_SUPPORTED_REV;"
        " print(VEROS_SUPPORTED_REV)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == VEROS_SUPPORTED_REV


def test_workflow_pins_the_supported_revision():
    """CI's env pin is the constant, and every Veros checkout uses the pin."""
    yaml = pytest.importorskip("yaml", reason="pyyaml is needed to parse the workflow")
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert workflow["env"]["VEROS_REV"] == VEROS_SUPPORTED_REV, (
        f"{WORKFLOW.name} checks out the Veros fork at"
        f" {workflow['env']['VEROS_REV']} but"
        " jem/components/veros/__init__.py says JAX-ESM is supported against"
        f" {VEROS_SUPPORTED_REV}. Bump both."
    )

    refs = {
        name: step["with"]["ref"]
        for name, job in workflow["jobs"].items()
        for step in job["steps"]
        if step.get("with", {}).get("repository") == VEROS_REPOSITORY
    }
    assert refs, "No job checks out the Veros fork at all"
    unpinned = {name: ref for name, ref in refs.items() if ref != "${{ env.VEROS_REV }}"}
    assert not unpinned, (
        f"Jobs checking out the Veros fork at something other than the"
        f" VEROS_REV pin: {unpinned}. Every job, the canary included, must"
        " test the supported revision."
    )


def test_the_dockerfile_installs_the_supported_revision():
    """The image is built against the supported revision, by sha."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    match = re.search(r"^ARG VEROS_REV=(\S+)\s*$", dockerfile, re.MULTILINE)
    assert match is not None, "No `ARG VEROS_REV=` in the Dockerfile"
    assert match.group(1) == VEROS_SUPPORTED_REV, (
        f"Dockerfile builds against Veros {match.group(1)} but the supported"
        f" revision is {VEROS_SUPPORTED_REV}. Bump it with the pin."
    )
    # `git clone --branch` accepts a branch or a tag, not a sha, so the
    # revision has to be checked out after the clone.
    assert 'checkout "${VEROS_REV}"' in dockerfile


@pytest.mark.parametrize(
    "doc",
    [
        "README.md",
        "docs/source/getting_started.rst",
        "docs/source/developers.rst",
    ],
)
def test_install_instructions_check_out_the_supported_revision(doc):
    """A user following the install steps gets the Veros CI tested."""
    text = (ROOT / doc).read_text()
    assert f"checkout {VEROS_SUPPORTED_REV}" in text, (
        f"{doc} does not install the Veros fork at {VEROS_SUPPORTED_REV}, the"
        " revision jem/components/veros/__init__.py supports. Bump it with"
        " the pin."
    )


@pytest.mark.parametrize(
    "doc",
    ["README.md", "docs/source/getting_started.rst", "docs/source/developers.rst"],
)
def test_install_instructions_name_no_floating_veros_ref(doc):
    """No documented Veros install clones the fork without pinning it."""
    text = (ROOT / doc).read_text()
    if "veros-jittable.git" in text:
        assert VEROS_SUPPORTED_REV in text
    assert "dev/naive-fix-AD-needs-physical-justification" not in text
