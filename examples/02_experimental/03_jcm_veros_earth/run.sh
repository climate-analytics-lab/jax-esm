#!/bin/bash
echo "PYTHONPATH=$PYTHONPATH"
time python3 main_forward.py \
    --total-simulation-days 60                      \
    --simulation-interval-days 30                   \
    --truncation-number 31                          \
    --simulation-name earth                         \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --max-rerun-attempts 0                          \
    --grid-folder data                              \
    --do-not-average-time                           \
    --explode-log explode.log

