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
# # Slab Ocean Model and inverse Q-flux
#
# This notebook attempts to inverse Q-flux of slab ocean model.

# %% [markdown]
# ## Import Packages

# %%
import os, sys
from pathlib import Path

# or `export PYTHONPATH=/path/to/jax-esm/root/directory`
sys.path.append( (Path(os.getcwd()) / ".." ).resolve())
print(f"sys.path = {str(sys.path)}")

# %%
import jcm
from jcm.geometry import Geometry
import jax_datetime as jdt

from jem.tool_scripts.generate_jcm_forcing_and_topography_files import (
    generate_jcm_forcing_and_topography_files,
)
from jem.components import JCM, SlabLandModel, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

# %% [markdown]
# ## Configurations

# %%
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_interval = jdt.to_timedelta(3, "day")
output_dir = Path("output/Qflux").resolve()

external_files = generate_jcm_forcing_and_topography_files()
output_dir.mkdir(exist_ok=True, parents=True)
geometry = Geometry.from_file(external_files["terrain"])
one_second = jdt.to_timedelta(1, "second")
# %% [markdown]
# ## Create Components

# %%
# Creating components

atm_model = jcm.model.Model(
    start_date=start_datetime,
    geometry=geometry
)

JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)

components = dict(
    atm=atm_model,
    ocn=SlabOceanModel(
        start_datetime=start_datetime,
        mask_file=external_files["terrain"],
        forcing_method="Qflux",
    ),
    lnd=SlabLandModel(
        start_datetime=start_datetime,
        topography_file=external_files["terrain"],
        mask_file=external_files["terrain"],
        land_clim_file=external_files["forcing"],
    ),
)

# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components

# %%
# Creating regridders and mapping
identity = lambda x: x
mapper = BasicMapper(components=components)
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("ocn", "forcing.total_heat_flux"),
    regridder = identity,
)
mapper.add_mapping(
    source = ("ocn", "state.sea_surface_temperature"),
    target = ("atm", "forcing.sea_surface_temperature"),
    regridder = identity,
)
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("lnd", "forcing.total_heat_flux"),
    regridder = identity,
)
mapper.add_mapping(
    source = ("lnd", "state.land_surface_temperature"),
    target = ("atm", "forcing.stl_am"),
    regridder = identity,
)

# %% [markdown]
# ## Create Coupled Model

# %%
model = Coupler(
    components=components,
    mappers=dict(mapper=mapper),
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")

# %% [markdown]
# ## Run Coupled Model

# %%
# Obtain initial condition
initial_coupled_state_forcing = model.initialize()

print(initial_coupled_state_forcing["ocn"]["state"]["mixed_layer_depth"])

print("Model state:")
tree_tools.print_tree(initial_coupled_state_forcing, root="ModelState")

print("Create model trajectory function...")
trajectory_function = model.generate_trajectory_function(
    workflow=["mapper", "atm", "ocn", "lnd"],
    iterations = int(simulation_interval / coupling_timestep),
)

# Run coupled model
print("Running model...")
state_holder, predictions = trajectory_function(initial_coupled_state_forcing)
print("Simulation finished.")
# %% [markdown]
# ## Output into NetCDF

# %%
output_dict = model.predictions_to_xarray(predictions)
for component_name, ds in output_dict.items():
    output_file = output_dir / f"{component_name:s}.nc"
    print("Output file: ", str(output_file))
    ds.to_netcdf(output_file, engine="netcdf4")

