#!/bin/bash
# Run the coupled JCM + Veros Earth example.
#
# Fail on the first error, on an undefined variable, and on a failure anywhere
# in a pipeline, so a broken step cannot be masked by a later one.
set -euo pipefail

truncation_number=31
simulation_name=example_02-03_jcm_veros_earth

# main.py resumes from the newest checkpoint under its output directory, and
# exits immediately when that checkpoint says the run is already complete. A
# directory left behind by an earlier run therefore turns this script into a
# no-op that tests nothing -- which is what happens under tests/examples,
# where the same working tree is reused. This example is a short
# demonstration, not a production run to be continued, so it always starts
# from a clean slate.
output_dir="output_T${truncation_number}/${simulation_name}"
echo "Removing any previous output in ${output_dir}"
rm -rf "${output_dir}"

echo "PYTHONPATH=${PYTHONPATH:-}"
python3 -c "import jcm ; print('jcm.__file__ = ', jcm.__file__);"
python3 -c "import veros ; print('veros.__file__ = ', veros.__file__);"
time python3 main.py \
    --total-simulation-days 60                      \
    --simulation-interval-days 30                   \
    --truncation-number "$truncation_number"        \
    --simulation-name "$simulation_name"            \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --max-rerun-attempts 0                          \
    --grid-folder data                              \
    --do-not-average-time                           \
    --explode-log explode.log
