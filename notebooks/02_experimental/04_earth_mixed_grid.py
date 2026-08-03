# Earth-like configuration
# This notebook demonstrates a JAX-ESM (JEM) example using JAX-GCM (JCM), Slab Ocean Model, and Slab Land Model.

from pathlib import Path

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from importlib import resources

import jax_datetime as jdt

from jem.components import JCM, SlabLandModel, SlabOceanModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_name = "02-01_earth"
output_dir = (Path("output") / simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
output_figures = {
    "animation": output_dir / "animation_humidity_sst.gif",
}
one_second = jdt.to_timedelta(1, "second")

## Topography and forcing
coords = get_speedy_coords()  # T31 spectral resolution with 8 vertical levels

# Load realistic orography and land-sea mask, interpolated to T31 grid
data_dir = resources.files("jem.data.grid")
rotated_grid_landsea_mask_file = data_dir / "landsea_mask_RG_4.00deg.nc"

#terrain_file = data_dir / "terrain.nc"
#terrain = TerrainData.from_file(terrain_file, coords=coords)

# Load realistic forcing data (SST, sea ice, soil moisture, etc.) interpolated to T31 grid
#forcing_file = data_dir / "forcing.nc"
#forcing = ForcingData.from_file(forcing_file, coords=coords)

# %% [markdown]
# ## Creating Flux and Scalar Exchange between Components

# %%
import jax.numpy as jnp
mapper = BasicMapper()
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("ocn", "forcing.total_heat_flux"),
)
mapper.add_mapping(
    source = ("ocn", "state.sea_surface_temperature"),
    target = ("atm", "forcing.sea_surface_temperature"),
)
mapper.add_mapping(
    source = ("atm", "derived.total_heat_flux"),
    target = ("lnd", "forcing.total_heat_flux"),
)
mapper.add_mapping(
    source = ("lnd", "state.land_surface_temperature"),
    target = ("atm", "forcing.stl_am"),
)
mapper.add_mapping(
    source = ("lnd", "state.snowc"),
    target = ("atm", "forcing.snowc_am"),
)
mapper.add_mapping(
    source = ("lnd", "state.soilw"),
    target = ("atm", "forcing.soilw_am"),
)

# %% [markdown]
# ## Create Components

# %%
atm_model = jcm.model.Model(
    coords = coords, 
    start_date=start_datetime,
    terrain = terrain,
)

atm_model = JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)

model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=SlabOceanModel(
            start_datetime=start_datetime,
            mask_file=terrain_file,
        ),
        lnd=SlabLandModel(
            start_datetime=start_datetime,
            topography_file=terrain_file,
            tdland = 86400.0, 
        ),
    ),
    mappers=dict(mapper=mapper),
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")

# %% [markdown]
# ## Run Coupled Model

# %%
simulation_interval = jdt.to_timedelta(5, "day")
initial_state, final_state, predictions = model.run(
    workflow=["mapper", "atm", "ocn", "lnd"],
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
# ## Visualization

# %%
import matplotlib as mplt
if not use_ipython:
    mplt.use("Agg")
import matplotlib.pyplot as plt


# %%
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import numpy as np
output_dict_animation = {
    component_name: _ds.isel(time=slice(None, None, 1))
    for component_name, _ds in output_dict.items()
}

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)
cb = None
cf = None
cs = None

def update(frame):
    print(f"Plotting frame={frame:d}")
    global cf, cb, cs 
    _data_q = output_dict_animation["atm"]["specific_humidity"].isel(time=frame, level=0)
    _data_sst = output_dict_animation["ocn"]["sea_surface_temperature"].isel(time=frame) - 273.15
    coords = _data_q.coords
    time_str = _data_q['time'].dt.strftime('%Y-%m-%d').to_numpy().item()
    lat = coords["lat"]
    lon = coords["lon"]

    # Remove contour and contourf if not empty
    cf and cf.remove()
    cs and cs.remove()
    
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
    ax.set_title(f"[{time_str:s}]\nSurface specific humidity (shading) and sea surface temperature (contours, ${{}}^\\circ \\mathrm{{C}}$)")
    if cb is None:
        cb = plt.colorbar(ax=ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.07)
        cb.set_label("[g/kg]", fontsize=12)
    
    return [cf,]
    
# Generate and save
ani = FuncAnimation(fig, update, frames=len(output_dict_animation["atm"].coords["time"]), interval=120, blit=False)
print("Saving animation: ", output_figures['animation'])
ani.save(output_figures['animation'], writer='pillow', dpi=200)

if use_ipython:
    from IPython.display import Image
    display(Image(output_figures['animation']))


# %%
