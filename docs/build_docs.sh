#!/usr/bin/env bash
#
# build_docs.sh - Build the Sphinx documentation for Jax-esm.
#
# This wraps the docs/Makefile workflow (notebook copy + sphinx-build)
# with dependency installation, a clean option, and optional
# auto-open of the built HTML in a browser.
#
# Usage:
#   ./build_docs.sh [-c] [-i] [-o] [-b BUILDER]
#
# Options:
#   -c            Clean the build directory before building.
#   -i            Install/upgrade docs requirements (docs/requirements.txt)
#                 before building.
#   -o            Open the built HTML docs in the default browser
#                 after a successful build (only meaningful with the
#                 default "html" builder).
#   -b BUILDER    Sphinx builder to use (default: html). Passed through
#                 to `make <BUILDER>` in docs/, e.g. "html", "latexpdf".
#   -h            Show this help message and exit.
#
# Examples:
#   ./build_docs.sh                # build html docs
#   ./build_docs.sh -c -o          # clean, build, then open in browser
#   ./build_docs.sh -i -b html     # install deps, then build html docs

set -euo pipefail

# Resolve paths relative to this script so it can be run from anywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DOCS_DIR="${SCRIPT_DIR}"
BUILD_DIR="${DOCS_DIR}/build"

CLEAN=0
INSTALL_DEPS=0
OPEN_AFTER=0
BUILDER="html"

usage() {
    # Print the header comment block (lines 2-27) as help text.
    sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while getopts ":cioB:b:h" opt; do
    case "${opt}" in
        c) CLEAN=1 ;;
        i) INSTALL_DEPS=1 ;;
        o) OPEN_AFTER=1 ;;
        b) BUILDER="${OPTARG}" ;;
        h) usage; exit 0 ;;
        \?) echo "Error: invalid option -${OPTARG}" >&2; usage; exit 1 ;;
        :) echo "Error: option -${OPTARG} requires an argument" >&2; usage; exit 1 ;;
    esac
done

if ! command -v sphinx-build >/dev/null 2>&1 && [[ "${INSTALL_DEPS}" -eq 0 ]]; then
    echo "Error: sphinx-build not found on PATH. Re-run with -i to install" >&2
    echo "docs/requirements.txt, or install it manually." >&2
    exit 1
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
    echo "==> Installing docs requirements from ${DOCS_DIR}/requirements.txt"
    python3 -m pip install -r "${DOCS_DIR}/requirements.txt"
fi

if [[ "${CLEAN}" -eq 1 ]]; then
    echo "==> Cleaning ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

echo "==> Building docs (target: ${BUILDER}) in ${DOCS_DIR}"
make -C "${DOCS_DIR}" "${BUILDER}"

INDEX_HTML="${BUILD_DIR}/html/index.html"

if [[ "${OPEN_AFTER}" -eq 1 ]]; then
    if [[ "${BUILDER}" != "html" ]]; then
        echo "Warning: -o only supports the html builder; skipping open." >&2
    elif [[ ! -f "${INDEX_HTML}" ]]; then
        echo "Warning: ${INDEX_HTML} not found; skipping open." >&2
    else
        echo "==> Opening ${INDEX_HTML}"
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "${INDEX_HTML}"
        elif command -v open >/dev/null 2>&1; then
            open "${INDEX_HTML}"
        else
            echo "Warning: no known browser-opener found; open manually:" >&2
            echo "  ${INDEX_HTML}" >&2
        fi
    fi
fi

echo "==> Done. Output in ${BUILD_DIR}/${BUILDER}"
