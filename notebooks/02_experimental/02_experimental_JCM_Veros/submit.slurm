#!/bin/bash
#
#SBATCH --job-name=JEM-coupled-long-run
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:2
#SBATCH --output=sbatch.64.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=23:00:00

nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax

source /home/tienyiao/.bashrc_jcm_v2
export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:$PYTHONPATH
export PYTHONHASHSEED=0 

echo "PYTHONPATH=$PYTHONPATH"

time python3 main.py \
    --total-simulation-days $(( 365 * 1000 ))       \
    --simulation-interval-days 365                  \
    --simulation-name test

