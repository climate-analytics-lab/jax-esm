# Couple JCM and Veros using JAX-ESM (JEM).

import sys
from pathlib import Path

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)  # only cache if compile took >1s

#jax.config.update("jax_enable_x64", False) 
import jax.numpy as jnp # for interaction
import numpy as np # to take average of output
import jcm
from jcm.forcing import ForcingData
from importlib import resources

import jax_datetime as jdt
import xarray as xr

import jem
import jem.utils.tree_tools as tree_tools
from jem.utils.checkpoints import (
    save_coupled_carry, load_coupled_carry,
    save_veros_carry, load_veros_carry,
)

from model_setup import build_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--total-simulation-days", type=int, help="Total time of simulation in days", default=10)
parser.add_argument("--simulation-interval-days", type=int, help="Simulation interval in days", default=5)
parser.add_argument("--simulation-name", type=str, help="Simulation name for output", default="default")
args = parser.parse_args()

print(f"jcm library is located at: {jcm.__file__}")
print(f"jem library is located at: {jem.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")

# Configurations
calendar = "365_day"
truncation_number = 31
total_simulation_time = jdt.to_timedelta(args.total_simulation_days, "day")
simulation_interval = jdt.to_timedelta(args.simulation_interval_days, "day")
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(24, "hour")

output_dir = (Path(f"output_T{truncation_number}") / args.simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
one_second = jdt.to_timedelta(1, "second")

# Build the coupled JCM + Veros + SlabOceanModel system. Packaged as a
# function in `model_setup.py` so that other scripts (e.g. a jax.grad
# sensitivity experiment) can build exactly the same model.
model, config = build_model(
    truncation_number=truncation_number,
    start_datetime=start_datetime,
    coupling_timestep=coupling_timestep,
    calendar=calendar,
)
ocn_model = model.components["ocn"].raw_component

print("Model info: ")
tree_tools.print_tree(model.get_info(), root="Model")

# Run Coupled Model
initial_carry = model.initialize()
batches = int(total_simulation_time / simulation_interval)
checkpoint_dir = output_dir / "checkpoint"
resume_batch = 0
if checkpoint_dir.exists():
    saved = sorted(checkpoint_dir.glob("batch_*"))
    if saved:
        resume_batch = int(saved[-1].name.split("_")[1]) + 1
        print(f"Resuming from batch {resume_batch}")
        initial_carry = load_coupled_carry(
            saved[-1], ["atm", "ocn", "fakelnd"],
            component_loaders={"ocn": lambda path: load_veros_carry(path, ocn_model)},
        )

if resume_batch == batches:
    print(f"Target batches: {batches:d} is all done. Exit the program.")
    sys.exit()

for b in range(resume_batch, batches):
    
    print(f"[batch={b:d}/{batches:d}] Simulation...")
    
    _, final_carry, predictions = model.run(
        initial_carry = initial_carry,
        workflow=["mapper", "ocn", "atm", "fakelnd"],
        iterations = int(simulation_interval / coupling_timestep),
        jitted=True,
        reuse_last_available_trajectory=True,
    )
    
    output_dict = model.predictions_to_xarray(predictions)

    for component_name, ds in output_dict.items():
        output_dict[component_name] = ds.reduce(np.mean, dim="time", keepdims=True)
 
    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}-{b:05d}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, unlimited_dims="time", engine="netcdf4")
        ds.close()
  
    if jnp.any( jnp.isnan(output_dict["atm"]["specific_humidity"].to_numpy()) ):
        print("Error: Model exploded. End program")
        break

    initial_carry = final_carry
    save_coupled_carry(
        final_carry, checkpoint_dir / f"batch_{b:05d}",
        component_savers={"ocn": save_veros_carry},
    )

print(f"Program ends.")

