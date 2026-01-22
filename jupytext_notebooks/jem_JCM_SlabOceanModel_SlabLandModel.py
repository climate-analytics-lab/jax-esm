# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # JEM - Coupling Quick Start
#
# This notebook demonstrates a JAX-ESM (JEM) example using JAX-GCM (JCM), Slab Ocean Model, and Slab Land Model.

# %% [markdown]
# ## Import Packages

# %%
import os, sys
from pathlib import Path

# or `export PYTHONPATH=/path/to/jax-esm/root/directory`
sys.path.append( (Path(os.getcwd()) / ".." ).resolve())

# %%
import jcm
from jcm.geometry import Geometry
import jax_datetime as jdt


from jem.tool_scripts.generate_jcm_forcing_and_topography_files import (
    generate_jcm_forcing_and_topography_files,
)
from jem.components import JCM, SlabLandModel, SlabOceanModel
from jem.coupling.transformer import IdentityTransformer
from jem.coupling.forcing_mapper import ForcingMapper
from jem.coupling.coupler import Coupler
import jem.utils.tree_tools as tree_tools

# %% [markdown]
# ## Configurations

# %%
resolution = 31
grid_specification = f"JCM::T{resolution:d}"

coupling_timestep = 86400.0
start_datetime = jdt.to_datetime("2000-01-01")
simulation_interval = jdt.to_timedelta(50, "day")
output_dir = Path("output/JCM_SOM_SLM").resolve()

external_files = generate_jcm_forcing_and_topography_files(resolution=resolution)
print("Simulation output dir: ", str(output_dir))
output_dir.mkdir(exist_ok=True, parents=True)

geometry = Geometry.from_file(external_files["terrain"])



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
    save_interval=3600.0 * 12,
)



components = dict(
    atm=atm_model,
    ocn=SlabOceanModel(
        grid_specification=grid_specification,
        timestep=coupling_timestep,
        start_datetime=start_datetime,
        save_interval=coupling_timestep,
        relaxation_time=60 * 86400.0,
        mask_file=external_files["terrain"],
        SST_clim_file=external_files["forcing"],
    ),
    lnd=SlabLandModel(
        grid_specification=grid_specification,
        timestep=coupling_timestep,
        start_datetime=start_datetime,
        save_interval=coupling_timestep,
        relaxation_time=60 * 86400.0,
        topography_file=external_files["terrain"],
        mask_file=external_files["terrain"],
        land_clim_file=external_files["forcing"],
    ),
)

# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components

# %%
# Creating transformations
transformers = dict(
    a2o = dict(
        identity_transformer = IdentityTransformer(
            source_grid = components["atm"].domain.horizontal_grids["T"],
            target_grid = components["ocn"].domain.horizontal_grids["T"],
        ),
    ),
    o2a = dict(
        identity_transformer = IdentityTransformer(
            source_grid = components["ocn"].domain.horizontal_grids["T"],
            target_grid = components["atm"].domain.horizontal_grids["T"],
        ),
    ),
    a2l = dict(
        identity_transformer = IdentityTransformer(
            source_grid = components["atm"].domain.horizontal_grids["T"],
            target_grid = components["lnd"].domain.horizontal_grids["T"],
        ),
    ),
    l2a = dict(
        identity_transformer = IdentityTransformer(
            source_grid = components["lnd"].domain.horizontal_grids["T"],
            target_grid = components["atm"].domain.horizontal_grids["T"],
        ),
    ),
)

forcing_mapper = ForcingMapper(components=components)
forcing_mapper.add_forcing_mapping(
    source = ("atm", "extra.total_heat_flux"),
    target = ("ocn", "flux.total_heat_flux"),
    transformer = transformers["a2o"]["identity_transformer"],
)
forcing_mapper.add_forcing_mapping(
    source = ("ocn", "prog.sea_surface_temperature"),
    target = ("atm", "sea_surface_temperature"),
    transformer = transformers["o2a"]["identity_transformer"],
)
forcing_mapper.add_forcing_mapping(
    source = ("atm", "extra.total_heat_flux"),
    target = ("lnd", "flux.total_heat_flux"),
    transformer = transformers["a2l"]["identity_transformer"],
)
forcing_mapper.add_forcing_mapping(
    source = ("lnd", "prog.land_surface_temperature"),
    target = ("atm", "stl_am"),
    transformer = transformers["l2a"]["identity_transformer"],
)

# %% [markdown]
# ## Create Coupled Model

# %%
model = Coupler(
    components=components,
    forcing_mapper=forcing_mapper,
    coupling_timestep=coupling_timestep,
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")


# %% [markdown]
# ## Run Coupled Model

# %%
# Obtain initial condition
initial_coupled_state_forcing = model.initialize()

print("Model state:")
tree_tools.print_tree(initial_coupled_state_forcing, root="ModelState")

print("Create model trajectory function...")
trajectory_function = model.generate_trajectory_function(
    start_time=0,
    end_time=simulation_interval / jdt.to_timedelta(1, "second"),
    jitted=True,
    show_progress=True,
    tqdm_kwargs=dict(desc="Simulation"),
)

# Run coupled model
print("Running model...")
state_holder, predictions = trajectory_function(initial_coupled_state_forcing)

# %% [markdown]
# ## Output into NetCDF

# %%
output_dict = model.predictions_to_xarray(predictions)
for component_name, ds in output_dict.items():
    output_file = output_dir / f"{component_name:s}.nc"
    print("Output file: ", str(output_file))
    ds.to_netcdf(output_file, engine="netcdf4")


exit()
# %% [markdown]
# ## Visualization

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ### Atmosphere

# %%
ds = output_dict["atm"]
print(str(ds))

# %% [markdown]
# #### Precipitation

# %%
ds['condensation.precls'].plot(x='lon', y='lat', col='time', col_wrap=2, aspect=2)
ds['convection.precnv'].plot(x='lon', y='lat', col='time', col_wrap=2, aspect=2)

# %% [markdown]
# #### Moisture

# %%
ds['specific_humidity'].mean('lon').plot(x='lat', y='level', col='time', col_wrap=3, aspect=6, yincrease=False)
ds['specific_humidity'].isel(level=3).plot(x='lon', y='lat', col='time', col_wrap=3, aspect=2)

# %% [markdown]
# #### Clouds

# %% jupyter={"outputs_hidden": true}
ds['shortwave_rad.cloudc'].plot(x='lon', y='lat', col='time', col_wrap=3, aspect=2)
ds['shortwave_rad.qcloud'].plot(x='lon', y='lat', col='time', col_wrap=3, aspect=2)

# %% [markdown]
# ### Ocean

# %%
ds = output_dict["ocn"]
print(str(ds))

# %% [markdown]
# #### Sea Surface Temperature
# We plot both the SST and its difference with respect to the initial condition.

# %%
g = (ds['sea_surface_temperature'] - 273.15).plot(x='longitude', y='latitude', col='time', col_wrap=3, aspect=2, cmap="gnuplot")
g.fig.suptitle("Sea Surface Temperature [${}^\\circ \\mathrm{C}$]", fontsize=16)

# %%
g = (ds['sea_surface_temperature'] - ds['sea_surface_temperature'].isel(time=0)).plot(x='longitude', y='latitude', col='time', col_wrap=3, aspect=2, center=0)
g.fig.suptitle("Difference of Sea Surface Temperature between labeled time and initial condition [${}^\\circ \\mathrm{C}$]", fontsize=16)

# %% [markdown]
# #### Total heat flux

# %%
g = ds['total_heat_flux'].plot(x='longitude', y='latitude', col='time', col_wrap=3, aspect=2, cmap="bwr_r", center=0.0)
g.fig.suptitle("Total heat flux (upward positive) [$\\mathrm{W} / \\mathrm{m}^2$]", fontsize=16)

# %% [markdown]
# ### Land

# %%
ds = output_dict["lnd"]
print(str(ds))

# %% [markdown]
# #### Land Surface Temperature

# %%
g = (ds['land_surface_temperature'] - 273.15).plot(x='longitude', y='latitude', col='time', col_wrap=3, aspect=2, cmap="bwr", center=0)
g.fig.suptitle("Land Surface Temperature [${}^\\circ \\mathrm{C}$]", fontsize=16, y=1.02)

# %%
plt.show()
