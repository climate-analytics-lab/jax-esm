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
parser.add_argument("--truncation-number", type=int, help="Truncation number", default=31)
parser.add_argument("--jcm-timestep-min", type=int, help="JCM timestep in minutes", default=30)
parser.add_argument("--veros-timestep-min", type=int, help="Veros timestep in minutes", default=30)
parser.add_argument("--do-not-average-time", action="store_true", help="Do not average time dimension for each interval.")
parser.add_argument("--max-rerun-attempts", type=int, help="If model exploded, then the model would rerun because stochasticitiy might bypass the instability next time. This value is by default 0, but if you set any positive integer number, model will rerun N times before it gave up.", default=0)
parser.add_argument("--explode-log", type=str, help="Path to log file for recording model explosion events.", default="explode.log")
parser.add_argument("--debug-mode", action="store_true", help="Turn on debug mode. Detect NaN and enter breakpoint.")
parser.add_argument("--grid-folder", type=str, help="Grid folder containing grid, land-sea mask, and regrid weightings.", required=True)
args = parser.parse_args()

print(f"jcm library is located at: {jcm.__file__}")
print(f"jem library is located at: {jem.__file__}")
import dinosaur
print(f"dinosaur library is located at: {dinosaur.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")

# Configurations
calendar = "365_day"
truncation_number = args.truncation_number
total_simulation_time = jdt.to_timedelta(args.total_simulation_days, "day")
simulation_interval = jdt.to_timedelta(args.simulation_interval_days, "day")
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")

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
    debug_mode=args.debug_mode,
    jcm_dt = args.jcm_timestep_min * 60.0,
    veros_dt_mom=args.veros_timestep_min * 60,
    veros_dt_tracer=args.veros_timestep_min * 60,
    grid_folder = args.grid_folder,
)
ocn_model = model.components["ocn"].raw_component

print("Model info: ")
tree_tools.print_tree(model.get_info(), root="Model")

atm_model = model.components["atm"].raw_component
transport_scheme = type(atm_model.dycore._primitive).__name__
print(f"jcm dycore transport scheme: {transport_scheme}")

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

    # The model might explode due to instability. However, since GPU simulation is in general
    # non-deterministic, the re-run might by pass the instability. So, I provide the option
    # --max-rerun-attempts to allow such rerun
    total_attempts = 1 + args.max_rerun_attempts
    model_failed = False
    for run_attempt in range(total_attempts): 
        _, final_carry, predictions = model.run(
            initial_carry = initial_carry,
            workflow=config["workflow"],
            iterations = int(simulation_interval / coupling_timestep),
            jitted=True,
            reuse_last_available_trajectory=True,
        )
        
        output_dict = model.predictions_to_xarray(predictions)

    
        model_is_stable = jnp.all( jnp.isfinite(output_dict["atm"]["specific_humidity"].to_numpy()) )

        if model_is_stable:
            print("All values of humidity are finite. Model does not explode.")
            if not args.do_not_average_time:
                for component_name, ds in output_dict.items():
                    output_dict[component_name] = ds.reduce(np.mean, dim="time", keepdims=True)

            break
        else:
            msg = f"batch={b:d}, attempt={run_attempt+1:d}/{total_attempts:d}: model exploded (non-finite humidity)"
            print(f"Error: {msg}")
            with open(output_dir / args.explode_log, "a") as f:
                f.write(msg + "\n")
            if run_attempt == total_attempts - 1:
                print(f"Error: Model exploded on all {total_attempts:d} attempt(s).")
                print("Output un-averaged results for debugging.")
                for component_name, ds in output_dict.items():
                    output_file = output_dir / f"exploded_{component_name:s}-{b:05d}.nc"
                    print("Output file: ", str(output_file))
                    ds.to_netcdf(output_file, unlimited_dims="time", engine="netcdf4")
                    ds.close()
                 
                print("Exit program.")
                sys.exit(1) 



    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}-{b:05d}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, unlimited_dims="time", engine="netcdf4")
        ds.close()
  
    initial_carry = final_carry
    save_coupled_carry(
        final_carry, checkpoint_dir / f"batch_{b:05d}",
        component_savers={"ocn": save_veros_carry},
    )

print(f"Program ends.")

