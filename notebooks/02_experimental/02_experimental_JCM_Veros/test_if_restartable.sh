#!/bin/bash

source $HOME/.bashrc_jcm_v2

echo "# One long run"
JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 python3 main.py \
    --total-simulation-days 10        \
    --simulation-interval-days 10      \
    --simulation-name one_long_run

echo "# One long run break into two batches"
JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 python3 main.py \
    --total-simulation-days 10        \
    --simulation-interval-days 5      \
    --simulation-name one_long_run_two_batches

for i in $( seq 1 2 ) ; do
    echo "# Short run: $i"
    JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 python3 main.py \
        --total-simulation-days 5     \
        --simulation-interval-days 5  \
        --simulation-name two_short_run
done





