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

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
import jax_datetime as jdt

from jem.components import JCM, SlabOceanModel, SlabSeaiceModel
from jem.components.slab.grid import generate_slab_grid
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools
use_ipython = 'get_ipython' in globals()

# %% [markdown]
# ## Configurations

# %%
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_name = "01-01_aquaplanet"
output_dir = (Path("output") / simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
output_figure = output_dir / "animation_humidity_sst.gif"
one_second = jdt.to_timedelta(1, "second")
# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components

# %%
def mapper(coupled_carry):
    atm = coupled_carry["atm"]
    ocn = coupled_carry["ocn"]
    seaice = coupled_carry["seaice"]

    ocn["forcing"].total_heat_flux = atm["derived"].total_heat_flux
    seaice["forcing"].ice_frazil_melt_energy = ocn["derived"].ice_frazil_melt_energy
    atm["forcing"].sea_surface_temperature = ocn["state"].sea_surface_temperature
    atm["forcing"].sice_am = seaice["derived"].ice_fraction

    return coupled_carry


# %% [markdown]
# ## Create Components

# %%
atm_model = jcm.model.Model(
    coords=get_speedy_coords(),  # T31 spectral resolution with 8 vertical levels
    start_date=start_datetime,
)

atm_model = JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)

# Aquaplanet: no mask_file, so fractional_mask defaults to all-zero (no land).
aquaplanet_grid = generate_slab_grid("JCM::T31")

model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            grid=aquaplanet_grid,
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
        ),
        seaice=SlabSeaiceModel(
            grid=aquaplanet_grid,
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
simulation_interval = jdt.to_timedelta(180, "day")
initial_state, final_state, predictions = model.run(
    workflow=["mapper", "atm", "ocn", "seaice"],
    iterations = int(simulation_interval / coupling_timestep),
)
# %% [markdown]
# ## Output into NetCDF

# %%
output_dict = model.predictions_to_xarray(predictions)
output_dict_subsample = {}
subsample_skip = 5
for component_name, ds in output_dict.items():
    output_file = output_dir / f"{component_name:s}.nc"
    print(f"Output file: {str(output_file)}, with subsample_skip = {subsample_skip:d}")
    ds = ds.isel(time=slice(None, None, subsample_skip))
    ds.to_netcdf(output_file, engine="netcdf4")
    output_dict_subsample[component_name] = ds


# %% [markdown]
# ## Visualization: animation of specific humidity of the surface grid

# %%

import matplotlib as mplt
if not use_ipython:
    mplt.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import numpy as np

output_dict_animation = {
    component_name: _ds.isel(time=slice(None, None, 1))
    for component_name, _ds in output_dict_subsample.items()
}

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.gridlines(draw_labels=True)
cb = None
cf = None
cs = None
ch = None

def update(frame):
    print(f"Plotting frame={frame:d}")
    global cf, cb, cs, ch
    _data_q = output_dict_animation["atm"]["specific_humidity"].isel(time=frame, level=0)
    _data_sst = output_dict_animation["ocn"]["sea_surface_temperature"].isel(time=frame) - 273.15
    _data_sit = output_dict_animation["seaice"]["ice_thickness"].isel(time=frame)
    coords = _data_q.coords
    time_str = _data_q['time'].dt.strftime('%Y-%m-%d').to_numpy().item()
    lat = coords["lat"]
    lon = coords["lon"]

    # Remove previous frame's artists before drawing the new ones
    cf and cf.remove()
    cs and cs.remove()
    ch and ch.remove()
    
    # Plot the humidity field for the current time step
    cyclic_data_q, cyclic_lon = add_cyclic_point(_data_q.to_numpy().transpose(), coord=lon)
    mappable = ax.contourf(
        cyclic_lon, lat,
        cyclic_data_q,
        levels=1 + np.linspace(0, 1, 21) * 10,
        transform=ccrs.PlateCarree(), 
        cmap='GnBu',
        extend="both",
    )
    
    cyclic_data_sst, cyclic_lon = add_cyclic_point(_data_sst.to_numpy().transpose(), coord=lon)
    cs = ax.contour(
        cyclic_lon, lat,
        cyclic_data_sst,
        levels=np.arange(-2, 31, 4),
        transform=ccrs.PlateCarree(),
        colors="black",
    )
    ax.clabel(cs, fontsize=12)

    # Dot-hatch grid cells that carry any sea ice (thickness above zero)
    cyclic_data_sit, cyclic_lon = add_cyclic_point(_data_sit.to_numpy().transpose(), coord=lon)
    ch = ax.contourf(
        cyclic_lon, lat,
        cyclic_data_sit,
        levels=[1e-6, np.inf],
        colors="none",
        hatches=["."],
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(f"[{time_str:s}]\nSurface specific humidity (shading) and sea surface temperature (contours, ${{}}^\\circ \\mathrm{{C}}$),\nwith sea ice (dotted hatching)")
    if cb is None:
        cb = plt.colorbar(ax=ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.07)
        cb.set_label("[g/kg]", fontsize=12)
    
    return [cf,]
    
# Generate and save
ani = FuncAnimation(fig, update, frames=len(output_dict_animation["atm"].coords["time"]), interval=120, blit=False)
print("Saving animation: ", output_figure)
ani.save(output_figure, writer='pillow', dpi=200)

if use_ipython:
    from IPython.display import Image
    display(Image(output_figure))
