#!/usr/bin/env bash
#
# run_notebooks.sh - Execute Jupyter notebooks from the shell, in place,
# without opening the Jupyter web UI.
#
# Wraps run_notebooks.py (argparse-based) so it can be run consistently
# from anywhere and fails loudly on errors.
#
# Usage:
#   ./run_notebooks.sh [-t TIMEOUT] [-k KERNEL] [PATH ...]
#
# Options:
#   -t TIMEOUT  Per-cell execution timeout in seconds, or -1 for no timeout
#               (default: 600).
#   -k KERNEL   Jupyter kernel name to execute with (default: python3).
#   -h          Show this help message and exit.
#
# PATH ...  Notebook files or directories to execute. Defaults to this
#           examples/ directory (recursively), skipping .ipynb_checkpoints
#           and output/ subdirectories.
#
# Examples:
#   ./run_notebooks.sh                                  # run every notebook here
#   ./run_notebooks.sh -t 1200 01_basic/                 # longer timeout, one subdir
#   ./run_notebooks.sh 01_basic/01_aquaplanet.ipynb       # single notebook

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

RUN_ARGS=()

while getopts ":t:k:h" opt; do
    case "${opt}" in
        t) RUN_ARGS+=(--timeout "${OPTARG}") ;;
        k) RUN_ARGS+=(--kernel "${OPTARG}") ;;
        h) usage; exit 0 ;;
        \?) echo "Error: invalid option -${OPTARG}" >&2; usage; exit 1 ;;
        :) echo "Error: option -${OPTARG} requires an argument." >&2; usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found on PATH." >&2
    exit 1
fi

if ! python3 -c "import nbformat, nbconvert, ipykernel" >/dev/null 2>&1; then
    echo "Error: the 'nbformat', 'nbconvert', and 'ipykernel' packages are required" >&2
    echo "       (pip install nbformat nbconvert ipykernel)." >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/run_notebooks.py" "${RUN_ARGS[@]}" "$@"
