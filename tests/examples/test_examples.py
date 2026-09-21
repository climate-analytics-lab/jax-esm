"""Executes every example notebook, end to end, on a real (T31) atmosphere.

This is the slow suite CI runs on pull requests only (`.github/workflows/
tests.yml`, the `examples` job): each notebook is a several-minute build and
integrate of a real coupled model, which is what makes it worth running as
its own test rather than trusting that the API it calls is covered elsewhere.
A failure names the notebook that caused it (`pytest -k <stem>` reruns just
that one), and the notebook's own traceback -- not a summary string -- is
what pytest reports, so the failure is diagnosable from the CI log alone.
"""

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# Every example plots; skip the whole suite (rather than fail) where the
# `plot` extra is not installed.
pytest.importorskip("matplotlib")

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"


def _notebook_paths() -> list[Path]:
    return [
        path
        for path in EXAMPLES.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    ]


@pytest.mark.slow
@pytest.mark.parametrize("notebook", sorted(_notebook_paths()), ids=lambda p: p.stem)
def test_notebook_runs(notebook: Path) -> None:
    """Execute ``notebook`` in place, in its own directory, and let it raise."""
    nb = nbformat.read(notebook, as_version=4)
    processor = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    # Not written back: a run's own `output/` directory (gitignored) is where
    # anything worth keeping goes; the executed notebook itself is a CI
    # artifact of this one call, not something to leave dirty in the tree.
    processor.preprocess(nb, {"metadata": {"path": str(notebook.parent)}})
