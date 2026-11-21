"""Executes every example notebook, end to end, on a real (T31) atmosphere.

This is the slow suite CI runs on pull requests only (`.github/workflows/
tests.yml`, the `examples` job): each notebook is a several-minute build and
integrate of a real coupled model, which is what makes it worth running as
its own test rather than trusting that the API it calls is covered elsewhere.
A failure names the notebook that caused it (`pytest -k <stem>` reruns just
that one), and the notebook's own traceback -- not a summary string -- is
what pytest reports, so the failure is diagnosable from the CI log alone.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import nbformat
import pytest
from jupyter_client.kernelspec import NoSuchKernel
from nbconvert.preprocessors import ExecutePreprocessor

# Every example plots; skip the whole suite (rather than fail) where the
# `plot` extra is not installed.
pytest.importorskip("matplotlib")

#: The kernelspec every notebook is executed with: the `python3` kernel
#: `ipykernel` registers in whichever environment runs the tests.
KERNEL_NAME = "python3"

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"


def _notebook_paths() -> list[Path]:
    return [
        path
        for path in EXAMPLES.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    ]


@pytest.fixture(autouse=True)
def _clean_notebook_output(notebook: Path) -> Iterator[None]:
    """Remove the notebook's gitignored ``output/`` directory after it runs.

    Each notebook that runs a coupled model writes its files into
    ``Path("output") / <name>`` next to itself (its ``output_dir = ...``
    cell); the spring notebook couples no climate components and writes
    nothing. Left alone across reruns of this suite those directories
    accumulate -- one model notebook's atmosphere file alone is tens of MB
    -- and a second run into the same files logs "already exists and is
    being overwritten" for each one. ``output/`` is this test's own scratch
    space and nothing else reads it, so the whole directory is removed here,
    whether the notebook passed or failed, rather than the test guessing at
    which per-notebook subdirectory the notebook happened to name it.
    ``notebook.parent`` may be shared by several notebooks (``01_basic``
    holds four); each only ever has its own subdirectory present when its
    own test's cleanup runs, since the suite runs one notebook at a time.
    """
    yield
    shutil.rmtree(notebook.parent / "output", ignore_errors=True)


@pytest.mark.slow
@pytest.mark.parametrize("notebook", sorted(_notebook_paths()), ids=lambda p: p.stem)
def test_notebook_runs(notebook: Path) -> None:
    """Execute ``notebook`` in place, in its own directory, and let it raise."""
    nb = nbformat.read(notebook, as_version=4)
    processor = ExecutePreprocessor(timeout=1800, kernel_name=KERNEL_NAME)
    # Not written back: a run's own `output/` directory (gitignored, and
    # removed by `_clean_notebook_output` above) is where anything worth
    # keeping goes; the executed notebook itself is a CI artifact of this
    # one call, not something to leave dirty in the tree.
    try:
        processor.preprocess(nb, {"metadata": {"path": str(notebook.parent)}})
    except NoSuchKernel:
        # `jupyter`/`ipykernel` are in no `jem` extra, so a fresh environment
        # can have the notebook machinery (nbconvert) but no kernelspec to
        # run a notebook in; say what to install rather than surface the
        # bare kernel name.
        pytest.fail(
            f"no Jupyter kernel named {KERNEL_NAME!r} is installed in this "
            "environment: run `pip install ipykernel jupyter` (the "
            f"{KERNEL_NAME!r} kernel is then this environment's own "
            "interpreter), or see what is registered with "
            "`jupyter kernelspec list`.",
            pytrace=False,
        )
