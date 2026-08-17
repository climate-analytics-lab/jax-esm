#!/usr/bin/env bash
#
# clean_notebooks.sh - Clear cell outputs and execution counts from the
# project's Jupyter notebooks.
#
# Wraps clean_notebooks.py (argparse-based) so it can be run consistently
# from anywhere and fails loudly on errors.
#
# Usage:
#   ./clean_notebooks.sh [-c] [PATH ...]
#
# Options:
#   -c    Check only: report notebooks that are not already clean and exit
#         non-zero if any are found, without modifying files. Useful as a
#         pre-commit/CI guard.
#   -h    Show this help message and exit.
#
# PATH ...  Notebook files or directories to clean. Defaults to this
#           notebooks/ directory (recursively), skipping
#           .ipynb_checkpoints and output/ subdirectories.
#
# Examples:
#   ./clean_notebooks.sh                       # clean every notebook here
#   ./clean_notebooks.sh -c                     # CI check, no changes made
#   ./clean_notebooks.sh 01_basic/01_aquaplanet.ipynb

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

usage() {
    sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

CHECK_ARGS=()

while getopts ":ch" opt; do
    case "${opt}" in
        c) CHECK_ARGS=(--check) ;;
        h) usage; exit 0 ;;
        \?) echo "Error: invalid option -${OPTARG}" >&2; usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found on PATH." >&2
    exit 1
fi

if ! python3 -c "import nbformat" >/dev/null 2>&1; then
    echo "Error: the 'nbformat' package is required (pip install nbformat)." >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/clean_notebooks.py" "${CHECK_ARGS[@]}" "$@"
