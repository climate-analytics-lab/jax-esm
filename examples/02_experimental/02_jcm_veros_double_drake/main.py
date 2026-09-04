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
parser.add_argument("--terrain-planet-type", type=str, help="Simulation name for output", required=True)
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
    terrain_planet_type=args.terrain_planet_type,
    jcm_dt = args.jcm_timestep_min * 60.0,
    veros_dt_mom=args.veros_timestep_min * 60,
    veros_dt_tracer=args.veros_timestep_min * 60,
)
# `coupler.components[name]` is the component object itself -- for the two
# wrapped models, `.model` is the underlying VerosSetup / jcm.model.Model.
ocn_component = model.components["ocn"]
ocn_model = ocn_component.model

print("Coupled model: ")
print(repr(model))

atm_model = model.components["atm"].model
transport_scheme = type(atm_model.dycore._primitive).__name__
print(f"jcm dycore transport scheme: {transport_scheme}")

# Run Coupled Model
#
# One trajectory function, compiled once and reused for every batch: the
# coupled step counter lives in the carry, not in the scan index, so calling it
# again on the carry it returned continues the run rather than restarting it.
steps_per_batch = int(simulation_interval / coupling_timestep)
run = model.generate_trajectory_function(steps_per_batch)

carry = model.initialize()
batches = int(total_simulation_time / simulation_interval)
checkpoint_dir = output_dir / "checkpoint"
resume_batch = 0
if checkpoint_dir.exists():
    saved = sorted(checkpoint_dir.glob("batch_*"))
    if saved:
        resume_batch = int(saved[-1].name.split("_")[1]) + 1
        print(f"Resuming from batch {resume_batch}")
        # `save_coupled_carry` stores the per-component carries only, so the
        # coupled step counter is restored from the batch index. (The Hydra
        # driver of Phase 2 will checkpoint it directly.)
        carry = carry.replace(
            components=load_coupled_carry(
                saved[-1], ["atm", "ocn", "fakelnd"],
                component_loaders={
                    "ocn": lambda path: load_veros_carry(path, ocn_model),
                },
            ),
            step=jnp.int32(resume_batch * steps_per_batch),
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
    for run_attempt in range(total_attempts):
        final_carry, diagnostics = run(carry)

        # `first_step` is the coupled step this batch started from; without it
        # every batch would be labelled with the first batch's dates.
        output_dict = model.to_xarray(
            diagnostics, first_step=b * steps_per_batch
        )

    
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
  
    carry = final_carry
    save_coupled_carry(
        final_carry.components, checkpoint_dir / f"batch_{b:05d}",
        component_savers={"ocn": save_veros_carry},
    )

print("Program ends.")

