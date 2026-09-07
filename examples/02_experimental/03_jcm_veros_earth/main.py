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

import jax_datetime as jdt

import jem
from jem.utils.checkpoints import (
    latest_complete_checkpoint, remaining_batches,
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

print("Coupled model: ")
print(repr(model))

atm_model = model.components["atm"].model
transport_scheme = type(atm_model.dycore._primitive).__name__
print(f"jcm dycore transport scheme: {transport_scheme}")

# Run Coupled Model
#
# The run is chunked into batches of `steps_per_batch` coupled steps so that
# output and a checkpoint are written between them. Both the batch length and
# the total have to be whole numbers of coupling steps, or the run would stop
# somewhere other than where it was asked to.
def _whole_steps(duration, what):
    """Return `duration` as a whole number of coupling steps, or fail loudly."""
    n_steps = float(duration / coupling_timestep)
    if n_steps != int(n_steps) or n_steps < 1:
        raise ValueError(
            f"{what} ({duration!r}) must be a whole number of coupling "
            f"timesteps ({coupling_timestep!r}); it is {n_steps:g} of them."
        )
    return int(n_steps)


steps_per_batch = _whole_steps(simulation_interval, "--simulation-interval-days")
total_steps = _whole_steps(total_simulation_time, "--total-simulation-days")

# One trajectory function, compiled once and reused for every full batch: the
# coupled step counter lives in the carry, not in the scan index, so calling it
# again on the carry it returned continues the run rather than restarting it.
run = model.generate_trajectory_function(steps_per_batch)

carry = model.initialize()
checkpoint_dir = output_dir / "checkpoint"
# The newest `step_*` directory is not necessarily a loadable checkpoint: a
# save interrupted after the directory was created but before its completion
# marker was written leaves one behind, and `model.load_state` refuses it.
# `latest_complete_checkpoint` skips those (logging each) and returns the newest
# checkpoint the run can actually resume from.
saved = latest_complete_checkpoint(checkpoint_dir)
if saved is not None:
    print(f"Resuming from checkpoint {saved.name}")
    # The coupler knows which of its components read themselves back by hand
    # -- the Veros ocean, through its HDF5 restart file -- so the driver does
    # not have to name them.
    carry = model.load_state(saved)
    print(f"Resuming at coupled step {int(carry.step):d}")

# What is left to run comes from the restored carry's own clock -- the only
# record of how far the run got -- rather than from the checkpoint's name. The
# checkpoint carries the coupled step counter, so the resumed run also picks up
# the seasonal cycle where it left off; and because the remaining work is
# counted in coupling steps, a run resumed with a different
# --simulation-interval-days (or --total-simulation-days) still stops at the
# total simulated time that was asked for.
steps_done = int(carry.step)
batch_lengths = remaining_batches(steps_done, total_steps, steps_per_batch)

if not batch_lengths:
    print(
        f"Target of {total_steps:d} coupled steps is already simulated "
        f"({steps_done:d} done). Exit the program."
    )
    sys.exit()

for batch_length in batch_lengths:
    # `first_step` is the coupled step this batch starts from; without it every
    # batch would be labelled with the first batch's dates. It comes from the
    # carry's own clock, so a run resumed with a different
    # --simulation-interval-days still labels its output with the dates it
    # actually simulated.
    first_step = int(carry.step)
    last_step = first_step + batch_length
    if batch_length == steps_per_batch:
        run_batch = run
    else:
        # The final batch is short when the total is not a whole number of
        # batches. It gets its own trajectory function: one extra compile,
        # once, in exchange for stopping exactly at the requested total.
        print(f"Compiling a {batch_length:d}-step trajectory for the final batch.")
        run_batch = model.generate_trajectory_function(batch_length)

    print(f"[steps {first_step:d}-{last_step:d} of {total_steps:d}] Simulation...")

    # The model might explode due to instability. However, since GPU simulation is in general
    # non-deterministic, the re-run might by pass the instability. So, I provide the option
    # --max-rerun-attempts to allow such rerun
    total_attempts = 1 + args.max_rerun_attempts
    for run_attempt in range(total_attempts):
        final_carry, diagnostics = run_batch(carry)

        output_dict = model.to_xarray(diagnostics, first_step=first_step)

    
        model_is_stable = jnp.all( jnp.isfinite(output_dict["atm"]["specific_humidity"].to_numpy()) )

        if model_is_stable:
            print("All values of humidity are finite. Model does not explode.")
            if not args.do_not_average_time:
                for component_name, ds in output_dict.items():
                    output_dict[component_name] = ds.reduce(np.mean, dim="time", keepdims=True)

            break
        else:
            msg = (
                f"steps {first_step:d}-{last_step:d}, "
                f"attempt={run_attempt+1:d}/{total_attempts:d}: "
                "model exploded (non-finite humidity)"
            )
            print(f"Error: {msg}")
            with open(output_dir / args.explode_log, "a") as f:
                f.write(msg + "\n")
            if run_attempt == total_attempts - 1:
                print(f"Error: Model exploded on all {total_attempts:d} attempt(s).")
                print("Output un-averaged results for debugging.")
                for component_name, ds in output_dict.items():
                    output_file = output_dir / f"exploded_{component_name:s}-{last_step:08d}.nc"
                    print("Output file: ", str(output_file))
                    ds.to_netcdf(output_file, unlimited_dims="time", engine="netcdf4")
                    ds.close()
                 
                print("Exit program.")
                sys.exit(1) 

    # Output and checkpoint are named after the coupled step they end at, which
    # is what the resume reads back, so the names stay meaningful across runs
    # with different batch lengths.
    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}-{last_step:08d}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, unlimited_dims="time", engine="netcdf4")
        ds.close()
  
    carry = final_carry
    model.save_state(final_carry, checkpoint_dir / f"step_{int(final_carry.step):08d}")

print("Program ends.")

