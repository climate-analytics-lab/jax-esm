# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # JEM - Coupling between jcm and veros
#
# This notebook demonstrates a JAX-ESM (JEM) example using JAX-GCM (JCM), Slab Ocean Model, and Slab Land Model.
# Please also see the adapter functions in `jem/components/Veros.py` and `jem/components/JCM.py`.
#
# In this example, you can specify one of the three configurations, "aquaplanet", "toy_earth", and "capped_earth", when calling the function `modify_jcm_terrain`. Because Veros cannot simulate poles, this example uses a slab model to cap the poles. We choose slab ocean model to be "fake land". Using slab land model currently will yield unrealistic temperature at poles because the albedo of ice and snow is not implemented in idealize setup.
# %% [markdown]
# ## Import Packages

# %%
import os, sys
from pathlib import Path

# %%
import jax
jax.config.update("jax_enable_x64", False) 
import jax.numpy as jnp
import jcm
from jcm.geometry import Geometry
import jax_datetime as jdt

from jem.tool_scripts.generate_jcm_forcing_and_topography_files import (
    generate_jcm_forcing_and_topography_files,
)
from jem.components import JCM, Veros, SlabOceanModel
from jem.mapping import IdentityRegridder
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

# %%
from modify_jcm_terrain import modify_jcm_terrain
external_files = generate_jcm_forcing_and_topography_files()

# There are three choices: "aquaplanet", "toy_earth", and "capped_earth" 
terrain_file = modify_jcm_terrain(external_files["terrain"], "toy_earth", "./data")

# %% [markdown]
# ## Configurations
# %%
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
output_dir = Path("output/JCM_VEROS_SLM").resolve()

output_dir.mkdir(exist_ok=True, parents=True)
one_second = jdt.to_timedelta(1, "second")

# %% [markdown]
# ## Create Components
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
import veros_case_setup
veros_case_setup.land_sea_mask_file = terrain_file
ocn_model = veros_case_setup.VerosCaseSetup()
ocn_model.setup()
Veros.make_jem_compatible(
    ocn_model,
    coupling_timestep=coupling_timestep,
)
# %% [markdown]
# ### Create JCM
# %%
atm_model = jcm.model.Model(
    start_date=start_datetime,
    geometry=Geometry.from_file(terrain_file),
)

JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
    land_model_active=True,
)
# %% [markdown]
# ### Create Slab Ocean model
# %%
slab_ocean_model=SlabOceanModel(
    start_datetime=start_datetime,
    timestep=coupling_timestep/one_second,
    mask_file=terrain_file ,
    forcing_method=None,
    mask_value=1.0,
)
# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components
# %%
# Creating regridders and mapping
def veros_to_jcm_regridder(arr):
    return jnp.pad(arr, ((0, 0), (4, 4)), constant_values=150)
def jcm_to_veros_regridder(arr):
    return arr[:, 4:-4]

# Here we can see the flexibility of JEM: You do not have to use the BasicMapper
# that we provide. You can define your own mapping function.
# Note: Remember to return the `coupled_carry` at the end.
def interaction(coupled_carry):
    atm = coupled_carry["atm"]
    ocn = coupled_carry["ocn"]
    slab_ocean = coupled_carry["slab_ocean"]

    # ===== compute wind stress begin =====
    # Tien-Yiao's ad-hoc way to compute wind stress
    # This conveniently demonstrates how the flux computation can be its own
    # function or module
    drag_coefficient = 1e-3 # dimensionless
    air_density = 1.22 # kg / m^3
    wind_x = jcm_to_veros_regridder(atm["derived"]["physics"].surface_flux.u0)
    wind_y = jcm_to_veros_regridder(atm["derived"]["physics"].surface_flux.v0)
    wind_velocity = jnp.sqrt(wind_x**2 + wind_y**2)    
    vs = ocn["state"].variables
    surface_taux = drag_coefficient * air_density * wind_velocity * wind_x
    surface_tauy = drag_coefficient * air_density * wind_velocity * wind_y
    # ===== compute wind stress end =====

    # Mapping
    ocn["forcing"].surface_taux = surface_taux
    ocn["forcing"].surface_tauy = surface_tauy     
    ocn["forcing"].heat_flux = jcm_to_veros_regridder(atm["derived"]["total_heat_flux"])
    ocn["forcing"].freshwater_flux = jcm_to_veros_regridder(atm["derived"]["total_freshwater_flux"])
    ocn["forcing"].wind_x = jcm_to_veros_regridder(atm["derived"]["physics"].surface_flux.u0)
    ocn["forcing"].wind_y = jcm_to_veros_regridder(atm["derived"]["physics"].surface_flux.v0)
    slab_ocean["forcing"].total_heat_flux = atm["derived"]["total_heat_flux"]
    atm["forcing"].sea_surface_temperature = veros_to_jcm_regridder(ocn["derived"]["sea_surface_temperature"])
    atm["forcing"].stl_am = slab_ocean["state"]["sea_surface_temperature"]

    return coupled_carry
    
# %% [markdown]
# ## Create Coupled Model
# %%
model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=ocn_model,
        slab_ocean=slab_ocean_model,
    ),
    mappers=dict(mapper=interaction),
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")
# %% [markdown]
# ## Run Coupled Model

# %%
simulation_interval = jdt.to_timedelta(60, "day")
initial_state, final_state, predictions = model.run(
    workflow=["mapper", "ocn", "atm", "slab_ocean"],
    iterations = int(simulation_interval / coupling_timestep),
    jitted=True,
)
# %% [markdown]
# ## Output into NetCDF

# %%
output_dict = model.predictions_to_xarray(predictions)
for component_name, ds in output_dict.items():
    output_file = output_dir / f"{component_name:s}.nc"
    print("Output file: ", str(output_file))
    ds.to_netcdf(output_file, engine="netcdf4")


# %% [markdown]
# # Visualization

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ## Atmosphere

# %%
ds = output_dict["atm"]
ds['condensation.precls'].plot(x='lon', y='lat', col='time', col_wrap=2, aspect=2)
ds['convection.precnv'].plot(x='lon', y='lat', col='time', col_wrap=2, aspect=2)

# %%
ds['specific_humidity'].mean('lon').plot(x='lat', y='level', col='time', col_wrap=3, aspect=6, yincrease=False)
ds['specific_humidity'].isel(level=3).plot(x='lon', y='lat', col='time', col_wrap=3, aspect=2)

# %% [markdown]
# ## Ocean

# %%
ds = output_dict["ocn"]
print(str(ds))

# %%
g = (ds['sea_surface_temperature'] - 273.15).plot(x='longitude', y='latitude', col='time', col_wrap=3, aspect=2, cmap="gnuplot")
g.fig.suptitle("Sea Surface Temperature [${}^\\circ \\mathrm{C}$]", fontsize=16)

# %% [markdown]
# ## Slab Ocean Model (fake land)

# %%
ds = output_dict["slab_ocean"]
print(str(ds))

# %%
g = (ds['sea_surface_temperature'] - 273.15).plot(x='longitude', y='latitude', col='time', col_wrap=3, aspect=2, cmap="bwr", center=0)
g.fig.suptitle("Land Surface Temperature [${}^\\circ \\mathrm{C}$]", fontsize=16, y=1.02)

# %%
