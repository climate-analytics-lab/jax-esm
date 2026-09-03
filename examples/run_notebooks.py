#!/usr/bin/env python3
"""Execute Jupyter notebooks in place from the command line, no web UI required."""

import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

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


def execute_notebook(nb_path: Path, timeout: int, kernel_name: str) -> None:
    """Run every cell in nb_path and write the outputs back into the same file."""
    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
    nbformat.write(nb, nb_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute Jupyter notebooks in place from the command line, "
                     "no web UI required."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_ROOT],
        help="Notebook files or directories to execute (default: this examples/ "
             "directory, recursively, skipping .ipynb_checkpoints and output/).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell execution timeout in seconds, or -1 for no timeout (default: 600).",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name to execute with (default: python3).",
    )
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            parser.error(f"path does not exist: {path}")

    notebooks = find_notebooks(args.paths)
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 0

    failed: list[Path] = []
    for nb_path in notebooks:
        print(f"executing: {nb_path}")
        try:
            execute_notebook(nb_path, timeout=args.timeout, kernel_name=args.kernel)
        except CellExecutionError as e:
            failed.append(nb_path)
            print(f"FAILED: {nb_path}\n{e}", file=sys.stderr)
        else:
            print(f"done: {nb_path}")

    if failed:
        print(
            f"\n{len(failed)}/{len(notebooks)} notebook(s) failed:",
            *[f"  {p}" for p in failed],
            sep="\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
