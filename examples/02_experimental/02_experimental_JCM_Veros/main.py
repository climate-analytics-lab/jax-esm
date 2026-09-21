# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Coupling JCM and Veros
#
# Couple JCM and Veros using JAX-ESM (JEM).
# %%
from pathlib import Path
import jax
#jax.config.update("jax_enable_x64", False) 
import jax.numpy as jnp # for interaction
import numpy as np # to take average of output
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

import jax_datetime as jdt
import xarray as xr

from jem.components import JCMComponent, SlabOceanModel, VerosComponent
from jem.components.slab import SlabGrid, SlabOceanParameters
from jem.base.coupler import Coupler

use_ipython = 'get_ipython' in globals()

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")

# %% [markdown]
# ## Choose terrain
# In this example, you can specify one of the three configurations, "aquaplanet", "toy_earth", and "capped_earth", when calling the function `modify_jcm_terrain`. Because Veros cannot simulate poles, this example uses a slab model to cap the poles. We choose slab ocean model to be "fake land". Using slab land model currently will yield unrealistic temperature at poles because the albedo of ice and snow is not implemented in idealize setup.

# %%
from modify_jcm_terrain import modify_jcm_terrain
from jem.tool_scripts.generate_jcm_forcing_and_topography_files import generate_jcm_forcing_and_topography_files

truncation_number = 31
total_simulation_time = jdt.to_timedelta(20, "day")
simulation_interval = jdt.to_timedelta(10, "day")
jcm_files = generate_jcm_forcing_and_topography_files(
    resolution=truncation_number,
)
# There are three choices: "aquaplanet", "toy_earth", and "capped_earth". The outcome will be saved in the folder "data"
coords = get_speedy_coords(spectral_truncation=truncation_number)
modified_jcm_terrain_file = modify_jcm_terrain(jcm_files["terrain"], "toy_earth", "./data")
terrain = TerrainData.from_file(
    modified_jcm_terrain_file,
    coords=coords
)

# %% [markdown]
# ## Configurations
# %%
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(24, "hour")
output_dir = (Path(f"output_T{truncation_number}") /"02-02_experimental_JCM_Veros").resolve()

output_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Create Components
# %% [markdown]
# ### Create JCM
# %%
atm_model = jcm.model.Model(
    coords = coords,
    start_date=start_datetime,
    terrain = terrain,
    time_step = 10,
)

atm_D2_nodal_shape = atm_model.coords.nodal_shape[1:]
# %% [markdown]
# ### Create Veros
#
# First need to remove `output_veros.*.nc` files, otherwise veros complains.
# %%
import glob
import os
files = glob.glob("output_veros.*.nc")
for f in files:
    print(f"Deleting: {f}")
    os.remove(f)

# %%
from veros_case_setup import generateVerosSetup
ocn_model = generateVerosSetup(
    nx = atm_D2_nodal_shape[0],
    ny = atm_D2_nodal_shape[1],
    land_sea_mask_file = modified_jcm_terrain_file,
    dt_mom = 3600.0,
    dt_tracer = 3600.0,
)()
ocn_model.setup()
# %% [markdown]
# ### Create Slab Ocean model
# %%
# The slab grid comes from the atmosphere's own horizontal grid, with the land
# fraction the atmosphere was built with -- one source of truth for both.
fakelnd_grid = SlabGrid.from_coords(
    coords.horizontal,
    fractional_mask=terrain.fmask,
)
# `ocean_mask_value=1.0` makes this slab ocean integrate the cells the grid
# marks as land: it is standing in for a land model, capping the poles Veros
# cannot simulate.
fakelnd_model = SlabOceanModel(
    fakelnd_grid,
    SlabOceanParameters(forcing_method="none", ocean_mask_value=1.0),
    name="fakelnd",
)
# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components
#
# An *exchanger* is the one place a component's carry is read by another. It receives the mapping of every component's carry together with the coupler's clock and returns the mapping to continue with; it is traced with the rest of the coupled step, so it builds new carries with `.replace(...)` rather than writing into the ones it is handed.
# %%
# Creating regridders and mapping
def veros_to_jcm_regridder(arr):
    return arr #jnp.pad(arr, ((0, 0), (4, 4)), constant_values=150)
def jcm_to_veros_regridder(arr):
    return arr#[:, 4:-4]

def interaction(components, time):
    del time  # this exchange does not depend on the date

    atm = components["atm"]
    ocn = components["ocn"]
    fakelnd = components["fakelnd"]

    # ===== compute wind stress begin =====
    # Tien-Yiao's ad-hoc way to compute wind stress
    # This conveniently demonstrates how the flux computation can be its own
    # function or module
    drag_coefficient = 1e-3 # dimensionless
    air_density = 1.22 # kg / m^3
    wind_x = jcm_to_veros_regridder(atm["derived"].physics["_surface_flux"].u0)
    wind_y = jcm_to_veros_regridder(atm["derived"].physics["_surface_flux"].v0)
    wind_velocity = jnp.sqrt(wind_x**2 + wind_y**2)    
    surface_taux = drag_coefficient * air_density * wind_velocity * wind_x
    surface_tauy = drag_coefficient * air_density * wind_velocity * wind_y
    # ===== compute wind stress end =====

    # Cap total heat flux for now. There seems to be instability coming from JCM. Need investigation
    #total_heat_flux = jnp.clip(atm["derived"].total_heat_flux, min=-1372.0, max=1372.0)
    total_heat_flux = atm["derived"].total_heat_flux

    # Mapping
    ocn = dict(ocn, forcing=ocn["forcing"].replace(
        surface_taux=surface_taux,
        surface_tauy=surface_tauy,
        heat_flux=jcm_to_veros_regridder(total_heat_flux),
        freshwater_flux=jcm_to_veros_regridder(atm["derived"].total_freshwater_flux),
    ))
    fakelnd = dict(
        fakelnd,
        forcing=fakelnd["forcing"].replace(
            total_heat_flux=jcm_to_veros_regridder(total_heat_flux),
        ),
        state=fakelnd["state"].replace(
            sea_surface_temperature=jnp.clip(
                fakelnd["state"].sea_surface_temperature,
                200.0,
                273.15 + 30.0,
            ),
        ),
    )
    atm = dict(atm, forcing=atm["forcing"].replace(
        sea_surface_temperature=veros_to_jcm_regridder(
            ocn["derived"].sea_surface_temperature
        ),
        stl_am=fakelnd["state"].sea_surface_temperature,
    ))

    return dict(components, atm=atm, ocn=ocn, fakelnd=fakelnd)

# %% [markdown]
# ## Create Coupled Model
# %%
# The coupler owns the clock and binds it to every component: it checks that
# the coupling timestep is a whole multiple of both JCM's and Veros' own
# timesteps, and that JCM was built on this start date and calendar.
model = Coupler(
    dict(
        atm=JCMComponent(atm_model),
        ocn=VerosComponent(ocn_model),
        fakelnd=fakelnd_model,
    ),
    dict(exchange=interaction),
    coupling_timestep=coupling_timestep,
    start_date=start_datetime,
    workflow=["exchange", "ocn", "atm", "fakelnd"],
)

print(repr(model))
# %% [markdown]
# ## Run Coupled Model

# %%
carry = model.initialize()

steps_per_batch = int(simulation_interval / coupling_timestep)
batches = int(total_simulation_time / simulation_interval)

# One trajectory function, compiled once and reused for every batch: the
# coupled step counter lives in the carry, not in the scan index, so calling it
# again on the carry it returned continues the run rather than restarting it.
run = model.generate_trajectory_function(steps_per_batch)

for b in range(batches):
    
    print(f"[batch={b:d}/{batches:d}] Simulation...")
    
    carry, diagnostics = run(carry)

    # `first_step` is the coupled step this batch started from; without it
    # every batch would be labelled with the first batch's dates.
    output_dict = model.to_xarray(diagnostics, first_step=b * steps_per_batch)

    ds_atm = output_dict["atm"]
    output_dict["atm_mean"] = ds_atm.reduce(np.mean, dim="time", keepdims=True)
    output_dict["atm_daily"] = xr.merge([
        ds_atm["specific_humidity"].isel(level=0),
        ds_atm["surface_flux.tsfc"],
        ds_atm["surface_flux.tskin"],
        ds_atm["shortwave_rad.rsns"],
        ds_atm["convection.precnv"],
        ds_atm["normalized_surface_pressure"],
        ds_atm["surface_flux.u0"],
        ds_atm["surface_flux.v0"],
        ((ds_atm["surface_flux.u0"]**2 + ds_atm["surface_flux.v0"]**2)**0.5).rename("wind_mag"),
    ])
    del output_dict["atm"]
    
    ds_ocn = output_dict["ocn"]
    output_dict["ocn_daily"] = xr.merge([
        ds_ocn["sea_surface_temperature"],
        ds_ocn["sea_surface_salinity"],
        ds_ocn["sea_surface_u"],
        ds_ocn["sea_surface_v"],
        ds_ocn["forcing_heat_flux"],
    ])
    output_dict["ocn_mean"] = ds_ocn.reduce(np.mean, dim="time", keepdims=True)
    del output_dict["ocn"]
    
    ds_fakelnd = output_dict["fakelnd"]
    output_dict["fakelnd_mean"] = ds_fakelnd.reduce(np.mean, dim="time", keepdims=True)
    del output_dict["fakelnd"]
 
    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}-{b:05d}.nc"
        print("Output file: ", str(output_file))
        ds.to_netcdf(output_file, engine="netcdf4")
        ds.close()
   
    if jnp.any( jnp.isnan(output_dict["atm_mean"]["specific_humidity"].to_numpy()) ):
        print("Error: Model exploded. End program")
        break


