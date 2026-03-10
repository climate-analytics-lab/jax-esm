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
# # Aquaplanet
#
# This notebook demonstrates a JAX-ESM (JEM) example using JAX-GCM (JCM) and Slab Ocean Model as an aquaplanet.

# %%
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
import jax_datetime as jdt

from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools
use_ipython = 'get_ipython' in globals()

def positive_cosine_cubic_latitude_squared(
    lat: jnp.ndarray,
    amplitude: float = 1.0,
) -> jnp.ndarray:
    return jnp.where(
        jnp.abs(lat) < jnp.pi / 3, amplitude * jnp.cos(3 * lat / 2) ** 2, 0
    )



# %% [markdown]
# ## Configurations

# %%
total_simulation_time = jdt.to_timedelta(100, "day")
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_name = "02-03_long_aquaplanet"
output_dir = (Path("output") / simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
one_second = jdt.to_timedelta(1, "second")
# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components

# %%
mapper = BasicMapper()
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("ocn", "forcing.total_heat_flux"),
    regridder = lambda x: x * 0,  # identity is default
)
mapper.add_mapping(
    source = ("ocn", "state.sea_surface_temperature"),
    target = ("atm", "forcing.sea_surface_temperature"),
)

# %% [markdown]
# ## Create Components

# %%
atm_model = jcm.model.Model(
    coords=get_speedy_coords(),  # T31 spectral resolution with 8 vertical levels
    start_date=start_datetime,
    time_step=10.0,
)

atm_model = JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)


hgrid = atm_model.coords.horizontal
lat = hgrid.latitudes
lon = hgrid.longitudes

llat = jnp.repeat(
    lat[None, :],
    repeats=len(lon),
    axis=0,
)
llon = jnp.repeat(
    lon[:, None],
    repeats=len(lat),
    axis=1,
)


model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
        ),
    ),
    mappers=dict(mapper=mapper),
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")

# %% [markdown]
# ## Run Coupled Model

# %%

initial_carry = model.initialize()
simulation_interval = jdt.to_timedelta(1, "day")
batches = int(total_simulation_time / simulation_interval)

initial_carry["ocn"]["state"].sea_surface_temperature = (
    273.15 + positive_cosine_cubic_latitude_squared(llat) * 50.0
) 

for b in range(batches):
    print(f"[batch={b:d}/{batches:d}] Simulation...")
    _, final_carry, predictions = model.run(
        initial_carry = initial_carry,
        workflow=["mapper", "atm", "ocn"],
        iterations = int(simulation_interval / coupling_timestep),
        jitted=True,
        reuse_last_available_trajectory=True,
    )
    
    output_dict = model.predictions_to_xarray(predictions)


    
    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}-{b:03d}.nc"
        print("Output file: ", str(output_file))
        ds = ds.reduce(np.mean, dim="time", keepdims=True)
        ds.to_netcdf(output_file, engine="netcdf4")
        ds.close()
   
    if jnp.any( jnp.isnan(output_dict["atm"]["specific_humidity"].to_numpy()) ):
        print("Error: Model exploded. End program")
        break

    initial_carry = final_carry


