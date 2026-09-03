#!/bin/bash

echo "PYTHONPATH=$PYTHONPATH"
python3 -c "import jcm ; print('jcm.__file__ = ', jcm.__file__);"
python3 -c "import veros ; print('veros.__file__ = ', veros.__file__);"
terrain_planet_type=double_drake
time python3 main.py \
    --total-simulation-days 60                      \
    --simulation-interval-days 30                   \
    --truncation-number 31                          \
    --simulation-name example_02-02_jcm_veros_double_drake \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --terrain-planet-type $terrain_planet_type      \
    --max-rerun-attempts 0                          \
    --do-not-average-time                           \
    --explode-log explode.log

