# CLEWS modeling


## Setting up a Python environment with Miniconda

If you don't have a Python environment set up, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a lightweight way to manage one.

```
conda create -n clews python=3.11
conda activate clews
```

All `pip install` commands below should be run inside this activated environment.

## Prerequisites

- Python 3.10+
- JAX (CPU or GPU build — see [JAX installation](https://github.com/google/jax#installation))
- `jax_datetime`, `numpy`, `xarray`, `netCDF4`

Install extra dependencies:

```
pip install jax-datetime numpy xarray netCDF4
```


## Getting JAX-GCM and JAX-ESM

```
mkdir working_directory
cd working_directory

git clone -b jem_CLEWS https://github.com/climate-analytics-lab/jax-esm.git
git clone -b dev https://github.com/climate-analytics-lab/jax-gcm.git

```

For each package, install them and their dependencies.

```
cd jax-gcm
pip install -e .

cd ../jax-esm
pip install -e .

```

## Test if you have it correctly

In python, inspect the package.

```
import jcm
print(jcm.__file__)

import jem
print(jem.__file__)

# You should see their paths consistent with where you download them.

```

## Run the model

```
cd notebooks/04_CLEWS
python main.py
```

## Configuration

Key parameters at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `spectral_truncation` | `31` | Atmospheric resolution (T31 ≈ 3.75°) |
| `total_simulation_months` | `12` | Total number of months to simulate |
| `simulation_name` | `"default_simulation"` | Sub-directory name under `output_T{N}/` |
| `average_output_of_each_batch` | `True` | Save monthly mean instead of daily snapshots |
| emission function | `emission_zero` | Swap in `emission_stepwise` or `emission_linear` in `interactions()` |

## Output

Each batch (one month) writes one NetCDF file per component into:

```
output_T{spectral_truncation}/{simulation_name}/
  atm-00000.nc
  ocn-00000.nc
  lnd-00000.nc
  co2_atm_boxmodel-00000.nc
  ...
```

## Checkpoint and resume

After each batch the model state is saved to:

```
output_T{spectral_truncation}/{simulation_name}/checkpoint/batch_NNNNN/
```

If this directory exists when `main.py` starts, the run resumes automatically from the latest saved batch. Delete the checkpoint directory to restart from scratch.
