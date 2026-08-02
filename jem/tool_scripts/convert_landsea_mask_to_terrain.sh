#!/usr/bin/env bash
# Convert a landsea_mask netCDF file (lon, lat dims) into a JCM-canonical
# terrain file (orog, lsm; flat orography). Wraps
# convert_landsea_mask_to_terrain.py -- see that script for format details.
#
# Usage:
#   ./convert_landsea_mask_to_terrain.sh <input_file> <output_file>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_file> <output_file>" >&2
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: input file not found: $INPUT_FILE" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/convert_landsea_mask_to_terrain.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE"
