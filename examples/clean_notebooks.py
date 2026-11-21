#!/usr/bin/env python3
"""Clear cell outputs, execution counts and run-specific metadata from notebooks.

Notebooks are committed unexecuted, so a diff shows only what an author wrote.
"Clean" is defined once, in :func:`clean_nb`, and both modes use it: cleaning
rewrites a notebook to that form, and ``--check`` (run by the ``lint`` job in
``.github/workflows/tests.yml``) reports any notebook that cleaning would
change. Keeping them one function is what stops the check from silently
covering less than the clean strips.
"""

import argparse
import copy
import sys
from pathlib import Path

import nbformat

DEFAULT_ROOT = Path(__file__).resolve().parent
SKIP_DIR_NAMES = {".ipynb_checkpoints", "output"}

#: Per-cell metadata keys that record how a notebook was edited or run rather
#: than what it says: Jupyter's timing record, and the jupytext cell-spacing
#: hints a notebook exported from a ``.py`` script carries.
RUN_CELL_METADATA = ("execution", "lines_to_next_cell", "lines_to_end_of_cell_marker")


def find_notebooks(paths: list[Path]) -> list[Path]:
    notebooks: list[Path] = []
    for path in paths:
        if path.is_file():
            if path.suffix == ".ipynb":
                notebooks.append(path)
            continue
        for nb_path in sorted(path.rglob("*.ipynb")):
            if SKIP_DIR_NAMES.isdisjoint(nb_path.relative_to(path).parts):
                notebooks.append(nb_path)
    return notebooks


def clean_nb(nb: nbformat.NotebookNode) -> bool:
    """Strip everything that records a run from ``nb`` in place; True if it changed.

    Removed: code-cell outputs and execution counts, the per-cell
    :data:`RUN_CELL_METADATA`, the notebook-level ``widgets`` state, and
    ``language_info.version``. The last is the Python the author happened to
    run (``3.11.15`` in one notebook, ``3.13.12`` in another, absent from the
    rest), which is a property of a machine, not of the notebook, and would
    otherwise change on every re-execution under a different interpreter.
    """
    changed = False

    for cell in nb.cells:
        metadata = cell.get("metadata", {})
        for key in RUN_CELL_METADATA:
            if metadata.pop(key, None) is not None:
                changed = True
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if nb.get("metadata", {}).pop("widgets", None) is not None:
        changed = True
    if nb.get("metadata", {}).get("language_info", {}).pop("version", None) is not None:
        changed = True

    return changed


def clean_notebook(nb_path: Path) -> bool:
    """Clean ``nb_path`` in place. Returns True if the notebook changed."""
    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    changed = clean_nb(nb)
    if changed:
        nbformat.write(nb, nb_path)
    return changed


def is_clean(nb_path: Path) -> bool:
    """Whether cleaning ``nb_path`` would leave it unchanged (it is not written)."""
    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    return not clean_nb(copy.deepcopy(nb))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear cell outputs, execution counts and run-specific "
        "metadata from Jupyter notebooks."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_ROOT],
        help="Notebook files or directories to clean (default: this examples/ directory).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report notebooks that would be changed, without modifying them. "
             "Exits with status 1 if any notebook is not already clean.",
    )
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            parser.error(f"path does not exist: {path}")

    notebooks = find_notebooks(args.paths)
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 0

    dirty = []
    for nb_path in notebooks:
        if args.check:
            if not is_clean(nb_path):
                dirty.append(nb_path)
                print(f"would clean: {nb_path}")
        elif clean_notebook(nb_path):
            print(f"cleaned: {nb_path}")

    if args.check and dirty:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
