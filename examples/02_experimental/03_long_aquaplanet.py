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

from jem.base.coupler import Coupler
from jem.components import JCMComponent, SlabOceanModel
from jem.components.slab import SlabGrid
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
spectral_truncation = 106
total_simulation_time = jdt.to_timedelta(360 * 100, "day")
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_name = "02-03_long_aquaplanet"
output_dir = (Path(f"output_T{spectral_truncation:d}") / simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components

# %%
def exchange(components, time):
    """Send the surface heat flux down and the SST back up.

    An exchanger is traced with the rest of the coupled step, so it builds new
    carries with `.replace(...)` rather than writing into the ones it is handed.
    """
    del time  # this exchange does not depend on the date

    atm = components["atm"]
    ocn = components["ocn"]

    ocn = dict(ocn, forcing=ocn["forcing"].replace(
        total_heat_flux=atm["derived"].total_heat_flux,
    ))
    atm = dict(atm, forcing=atm["forcing"].replace(
        sea_surface_temperature=ocn["state"].sea_surface_temperature,
    ))

    return dict(components, atm=atm, ocn=ocn)

# %% [markdown]
# ## Create Components

# %%
atm_model = jcm.model.Model(
    coords=get_speedy_coords(spectral_truncation=spectral_truncation),
    start_date=start_datetime,
    time_step=10.0,
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


# Aquaplanet: no fractional mask, so every cell of the slab grid is ocean.
ocn_grid = SlabGrid.from_coords(hgrid)

model = Coupler(
    dict(
        atm=JCMComponent(atm_model),
        ocn=SlabOceanModel(ocn_grid),
    ),
    dict(exchange=exchange),
    coupling_timestep=coupling_timestep,
    start_date=start_datetime,
)

print(repr(model))

# %% [markdown]
# ## Run Coupled Model

# %%

simulation_interval = jdt.to_timedelta(30, "day")
steps_per_batch = int(simulation_interval / coupling_timestep)
batches = int(total_simulation_time / simulation_interval)

# One trajectory function, compiled once and reused for every batch. The
# coupled step counter lives in the carry, not in the scan index, so calling it
# again on the carry it returned continues the run instead of restarting it.
run = model.generate_trajectory_function(steps_per_batch)

initial_carry = model.initialize()
ocn_carry = initial_carry.components["ocn"]
ocn_carry = dict(ocn_carry, state=ocn_carry["state"].replace(
    sea_surface_temperature=(
        273.15 + positive_cosine_cubic_latitude_squared(llat) * 50.0
    ),
))
carry = initial_carry.replace(
    components=dict(initial_carry.components, ocn=ocn_carry),
)

for b in range(batches):
    print(f"[batch={b:d}/{batches:d}] Simulation...")
    carry, diagnostics = run(carry)

    # `first_step` is the coupled step this batch started from. Without it
    # every batch would be labelled with the first batch's dates.
    output_dict = model.to_xarray(diagnostics, first_step=b * steps_per_batch)

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}-{b:03d}.nc"
        print("Output file: ", str(output_file))
        ds = ds.reduce(np.mean, dim="time", keepdims=True)
        ds.to_netcdf(output_file, engine="netcdf4")
        ds.close()

    if jnp.any( jnp.isnan(output_dict["atm"]["specific_humidity"].to_numpy()) ):
        print("Error: Model exploded. End program")
        break
