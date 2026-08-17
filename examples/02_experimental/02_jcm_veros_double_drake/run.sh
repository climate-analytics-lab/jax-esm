#!/bin/bash

echo "PYTHONPATH=$PYTHONPATH"
terrain_planet_type=double_drake
time python3 main_forward.py \
    --total-simulation-days 60                      \
    --simulation-interval-days 30                   \
    --truncation-number 31                          \
    --simulation-name $terrain_planet_type          \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --terrain-planet-type $terrain_planet_type      \
    --max-rerun-attempts 0                          \
    --do-not-average-time                           \
    --explode-log explode.log

