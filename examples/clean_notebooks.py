#!/usr/bin/env python3
"""Clear cell outputs and execution counts from Jupyter notebooks."""

import argparse
import sys
from pathlib import Path

import nbformat

DEFAULT_ROOT = Path(__file__).resolve().parent
SKIP_DIR_NAMES = {".ipynb_checkpoints", "output"}


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


def clean_notebook(nb_path: Path) -> bool:
    """Clear outputs/execution counts in-place. Returns True if the notebook changed."""
    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    changed = False

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("metadata", {}).pop("execution", None) is not None:
            changed = True

    if nb.get("metadata", {}).pop("widgets", None) is not None:
        changed = True

    if changed:
        nbformat.write(nb, nb_path)

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear cell outputs and execution counts from Jupyter notebooks."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_ROOT],
        help="Notebook files or directories to clean (default: this notebooks/ directory).",
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
            nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
            is_dirty = any(
                cell.get("cell_type") == "code"
                and (cell.get("outputs") or cell.get("execution_count") is not None)
                for cell in nb.cells
            )
            if is_dirty:
                dirty.append(nb_path)
                print(f"would clean: {nb_path}")
        else:
            if clean_notebook(nb_path):
                print(f"cleaned: {nb_path}")

    if args.check and dirty:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
