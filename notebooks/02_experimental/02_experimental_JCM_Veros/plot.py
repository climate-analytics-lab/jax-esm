import xarray as xr
from pathlib import Path
use_ipython = 'get_ipython' in globals()
print(f"use_ipython = {str(use_ipython)}")


import matplotlib as mplt
if not use_ipython:
    mplt.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import numpy as np


def plot(frame):
    fig, ax = plt.subplots(
        3, 1,
        figsize=(10, 9),
        subplot_kw = dict(
            projection=ccrs.PlateCarree(),
        ),
    )

    for _ax in ax.flatten():
        _ax.gridlines(draw_labels=True)

    _data_q = output_dict_animation["atm"]["specific_humidity"].isel(time=frame)
    _data_tskin = output_dict_animation["atm"]["surface_flux.tsfc"].isel(time=frame)
    _data_ocean_kinetic_energy = 0.5 * (
          output_dict_animation["ocn"]["sea_surface_u"].isel(time=int(frame // atm_ocn_speed_ratio))**2
        + output_dict_animation["ocn"]["sea_surface_v"].isel(time=int(frame // atm_ocn_speed_ratio))**2
    )

    coords = _data_q.coords
    time_str = _data_q['time'].dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy().item()
    lat = coords["lat"]
    lon = coords["lon"]

    # Plot the humidity field for the current time step
    cyclic_data_q, cyclic_lon = add_cyclic_point(_data_q.to_numpy().transpose(), coord=lon)
    cyclic_data_tskin, cyclic_lon = add_cyclic_point(_data_tskin.to_numpy().transpose(), coord=lon)
    cyclic_data_ocean_kinetic_energy, cyclic_lon = add_cyclic_point(_data_ocean_kinetic_energy.to_numpy().transpose(), coord=lon)

    _ax = ax[0]
    mappable = _ax.contourf(
        cyclic_lon, lat,
        cyclic_data_q,
        levels=1.0 + np.linspace(0, 1, 21) * 7,
        transform=ccrs.PlateCarree(), 
        cmap='GnBu',
        extend="both",
    )
    cb = plt.colorbar(ax=_ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.07)
    cb.set_label("[g/kg]", fontsize=12)

    _ax = ax[1]
    mappable = _ax.contourf(
        cyclic_lon, lat,
        cyclic_data_tskin - 273.15,
        levels=np.linspace(-10, 30, 100),
        transform=ccrs.PlateCarree(), 
        cmap='gnuplot',
        extend="both",
    )
    cb = plt.colorbar(ax=_ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.07)
    cb.set_label("[degC]", fontsize=12)

    _ax = ax[2]
    mappable = _ax.contourf(
        cyclic_lon, lat,
        cyclic_data_ocean_kinetic_energy,
        levels=np.linspace(0, 1, 21) * (0.01**2),
        transform=ccrs.PlateCarree(), 
        cmap='hot_r',
        extend="max",
    )
    cb = plt.colorbar(ax=_ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.07)
    cb.set_label("[$\\mathrm{m}^2 / \\mathrm{s}^2$]", fontsize=12)

    fig.suptitle(f"[{time_str:s}]")
    
    ax[0].set_title("(a) Surface specific humidity")
    ax[1].set_title("(b) Surface temperature")
    ax[2].set_title("(c) Surface ocean kinectic energy")
 
    return fig

if __name__ == "__main__": 
    
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dict_animation = {
        component_name : xr.open_mfdataset([
            f"output_T85_old/02-02_experimental_JCM_Veros/{component_name:s}-{i:03d}.nc"
            for i in range(80)
        ], concat_dim='time', combine='nested')
        for component_name in ["atm", "ocn"]
    }

    atm_ocn_speed_ratio = len(output_dict_animation["atm"].coords["time"]) / len(output_dict_animation["ocn"].coords["time"])

    print("Ratio: ", atm_ocn_speed_ratio)

    if atm_ocn_speed_ratio % 1 != 0 :
        raise Exception("atm_ocn_speed_ratio is not an integer")

    for frame in range(len(output_dict_animation["atm"].coords["time"])):

        #frame += 300
        print(f"Plotting frame = {frame:d}")
        fig = plot(frame)
        output_figure = output_dir / f"frame-{frame:04d}.png"
        print("Saving figure: ", output_figure)
        fig.savefig(output_figure, dpi=200)
        plt.close(fig)
