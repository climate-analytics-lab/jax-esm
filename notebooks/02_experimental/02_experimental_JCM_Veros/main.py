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
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from importlib import resources

import jax_datetime as jdt
import xarray as xr

import jem
from jem.components import JCM, Veros, SlabOceanModel
from jem.mapping import IdentityRegridder
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools
from jem.utils.checkpoints import (
    save_coupled_carry, load_coupled_carry,
    save_veros_carry, load_veros_carry,
)

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

# Create terrain
from modify_jcm_terrain import modify_jcm_terrain
from jem.tool_scripts.generate_jcm_forcing_and_topography_files import generate_jcm_forcing_and_topography_files


calendar = "365_day"

truncation_number = 31
total_simulation_time = jdt.to_timedelta(args.total_simulation_days, "day")
simulation_interval = jdt.to_timedelta(args.simulation_interval_days, "day")
jcm_files = generate_jcm_forcing_and_topography_files(
    resolution=truncation_number,
)

coords = get_speedy_coords(spectral_truncation=truncation_number)
modified_jcm_terrain_file = modify_jcm_terrain(jcm_files["terrain"], "double_drake", "./data")
terrain = TerrainData.from_file(
    modified_jcm_terrain_file,
    coords=coords
)

# Configurations
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(24, "hour")

output_dir = (Path(f"output_T{truncation_number}") / args.simulation_name).resolve()

output_dir.mkdir(exist_ok=True, parents=True)
one_second = jdt.to_timedelta(1, "second")

# Create Component

# Create JCM
atm_model = jcm.model.Model(
    coords = coords,
    start_date=start_datetime,
    terrain = terrain,
    time_step = 30,
    calendar = calendar,
)

JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)
    
atm_D2_nodal_shape = atm_model.coords.nodal_shape[1:]

# Create Veros
# First need to remove `output_veros.*.nc` files, otherwise veros complains.

import glob
import os
files = glob.glob("output_veros.*.nc")
for f in files:
    print(f"Deleting: {f}")
    os.remove(f)

from veros_case_setup import generateVerosSetup
ocn_model = generateVerosSetup(
    nx = atm_D2_nodal_shape[0],
    ny = atm_D2_nodal_shape[1],
    land_sea_mask_file = modified_jcm_terrain_file,
    dt_mom = 3600.0,
    dt_tracer = 3600.0,
)()
ocn_model.setup()
Veros.make_jem_compatible(
    ocn_model,
    coupling_timestep=coupling_timestep,
)

# Create Slab Ocean model
fakelnd_model=SlabOceanModel(
    grid_specification=f"JCM::T{truncation_number:d}",
    start_datetime=start_datetime,
    timestep=coupling_timestep/one_second,
    mask_file=modified_jcm_terrain_file,
    forcing_method=None,
    mask_value=1.0,
    calendar = calendar,
)

# Creating Flux and Scalar Exchange between Components
#
# Here we demonstrote the flexibility of JEM: You do not have to use the `BasicMapper` that JEM provides. You can define your own mapping function. In this example, we simply define a function `interaction` as below.

# Creating regridders and mapping
def veros_to_jcm_regridder(arr):
    return arr #jnp.pad(arr, ((0, 0), (4, 4)), constant_values=150)
def jcm_to_veros_regridder(arr):
    return arr#[:, 4:-4]

# Note: Remember to return the `coupled_carry` at the end.
def interaction(coupled_carry):
    atm = coupled_carry["atm"]
    ocn = coupled_carry["ocn"]
    fakelnd = coupled_carry["fakelnd"]

    # ===== compute wind stress begin =====
    # Tien-Yiao's ad-hoc way to compute wind stress
    # This conveniently demonstrates how the flux computation can be its own
    # function or module
    drag_coefficient = 1e-3 # dimensionless
    air_density = 1.22 # kg / m^3
    wind_x = jcm_to_veros_regridder(atm["derived"]["physics"]["_surface_flux"].u0)
    wind_y = jcm_to_veros_regridder(atm["derived"]["physics"]["_surface_flux"].v0)
    wind_velocity = jnp.sqrt(wind_x**2 + wind_y**2)    
    vs = ocn["state"].variables
    surface_taux = drag_coefficient * air_density * wind_velocity * wind_x
    surface_tauy = drag_coefficient * air_density * wind_velocity * wind_y
    # ===== compute wind stress end =====

    # Cap total heat flux for now. There seems to be instability coming from JCM. Need investigation
    #total_heat_flux = jnp.clip(atm["derived"]["total_heat_flux"], min=-1372.0, max=1372.0)
    total_heat_flux = atm["derived"]["total_heat_flux"]

    # Mapping
    ocn["forcing"].surface_taux = surface_taux
    ocn["forcing"].surface_tauy = surface_tauy     
    ocn["forcing"].heat_flux = jcm_to_veros_regridder(total_heat_flux)
    ocn["forcing"].freshwater_flux = jcm_to_veros_regridder(atm["derived"]["total_freshwater_flux"])
    ocn["forcing"].wind_x = jcm_to_veros_regridder(wind_x)
    ocn["forcing"].wind_y = jcm_to_veros_regridder(wind_y)
    fakelnd["forcing"].total_heat_flux = jcm_to_veros_regridder(total_heat_flux)
    fakelnd["state"].sea_surface_temperature = jnp.clip(
        fakelnd["state"].sea_surface_temperature,
        200.0,
        273.15 + 30.0,
    )
    atm["forcing"].sea_surface_temperature = veros_to_jcm_regridder(ocn["derived"]["sea_surface_temperature"])
    atm["forcing"].stl_am = fakelnd["state"]["sea_surface_temperature"]
    
    return coupled_carry

# Create Coupled Model
model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=ocn_model,
        fakelnd=fakelnd_model,
    ),
    mappers=dict(mapper=interaction),
)

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

