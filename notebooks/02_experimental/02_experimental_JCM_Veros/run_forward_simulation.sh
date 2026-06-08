#!/bin/bash

source $HOME/.bashrc_jcm_v2

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:$PYTHONPATH
export PYTHONHASHSEED=0 

echo "PYTOHNPATH=$PYTHONPATH"

python3 main_forward.py \
    --total-simulation-days $(( 365 * 1000 ))       \
    --simulation-interval-days 365                 \
    --simulation-name test
