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
# This notebook demonstrates a JAX-ESM (JEM) example using JAX-GCM (JCM), Slab Ocean Model, and Slab Land Model.
# Please also see the adapter functions in `jem/components/Veros.py` and `jem/components/JCM.py`.
# %% [markdown]
# ## Import Packages

# %%
import os, sys
from pathlib import Path

# or `export PYTHONPATH=/path/to/jax-esm/root/directory`
sys.path.append( (Path(os.getcwd()) / ".." ).resolve())

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
from jem.components import JCM, Veros, SlabLandModel
from jem.mapping import IdentityRegridder
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

# %% [markdown]
# ## Configurations
# %%
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_interval = jdt.to_timedelta(45, "day")
output_dir = Path("output/JCM_VEROS_SLM").resolve()

output_dir.mkdir(exist_ok=True, parents=True)
one_second = jdt.to_timedelta(1, "second")
terrain_file = Path("funky_earth_add_cap.nc")

# %% [markdown]
# ## Create Components
# %% [markdown]
# ### Create Veros
#
# First need to remove `acc.*.nc` files, otherwise veros complains.
# %%
import glob
import os
files = glob.glob("acc.*.nc")
for f in files:
    print(f"Deleting: {f}")
    os.remove(f)

# %%
from acc import ACCSetup
ocn_model = ACCSetup()
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
# ### Create Land model
# %%
lnd_model = SlabLandModel(
    start_datetime=start_datetime,
    topography_file=terrain_file,
    mask_file=terrain_file,
)
# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components
# %%
# Creating regridders and mapping
def veros_to_jcm_regridder(arr):
    return jnp.pad(arr, ((0, 0), (4, 4)), constant_values=150)
def jcm_to_veros_regridder(arr):
    return arr[:, 4:-4]
mapper = BasicMapper()
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("ocn", "forcing.total_heat_flux"),
    regridder = jcm_to_veros_regridder,
)
mapper.add_mapping(
    source = ("atm", "derived.physics.surface_flux.u0"),
    target = ("ocn", "forcing.wind_x"),
    regridder = jcm_to_veros_regridder,
)
mapper.add_mapping(
    source = ("atm", "derived.physics.surface_flux.v0"),
    target = ("ocn", "forcing.wind_y"),
    regridder = jcm_to_veros_regridder,
)
mapper.add_mapping(
    source = ("ocn", "derived.sea_surface_temperature"),
    target = ("atm", "forcing.sea_surface_temperature"),
    regridder = veros_to_jcm_regridder,
)
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("lnd", "forcing.total_heat_flux"),
)
mapper.add_mapping(
    source = ("lnd", "state.land_surface_temperature"),
    target = ("atm", "forcing.stl_am"),
)
# %% [markdown]
# ## Create Coupled Model
# %%
model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=ocn_model,
        lnd=lnd_model,
    ),
    mappers=dict(mapper=mapper),
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")
# %% [markdown]
# ## Run Coupled Model

# %%
initial_state, final_state, predictions = model.run(
    workflow=["mapper", "atm", "ocn", "lnd"],
    iterations = int(simulation_interval / coupling_timestep),
    jitted=False,
)
# %% [markdown]
# ## Output into NetCDF

# %%
output_dict = model.predictions_to_xarray(predictions)
for component_name, ds in output_dict.items():
    output_file = output_dir / f"{component_name:s}.nc"
    print("Output file: ", str(output_file))
    ds.to_netcdf(output_file, engine="netcdf4")


# %%
