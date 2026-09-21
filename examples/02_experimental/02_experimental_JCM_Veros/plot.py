import traceback
import multiprocessing
from pathlib import Path

import pandas as pd

import xarray as xr

use_ipython = 'get_ipython' in globals()
print(f"use_ipython = {str(use_ipython)}")


import matplotlib as mplt
if not use_ipython:
    mplt.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean as cmo
import cartopy.crs as ccrs
import numpy as np
    
ocean_kinetic_energy_scale = 0.1


def plot_simulation(detail):

    output_dir = detail["output_dir"]
    segement_index = detail["segement_index"]
    records_per_file = detail["records_per_file"]
    phase = detail["phase"]
    result = dict(detail=detail, status="UNKNOWN")

    try:

        output_figures = [
            Path(output_dir) / f"frame-{segement_index*records_per_file+i:05d}.png"
            for i in range(records_per_file)
        ]

        if phase == "detect":
            result["needs_work"] = not all([output_figure.exists() for output_figure in output_figures])
            result["status"] = "detect_ok"
            return result

        # Load files
        plot_data = {
            component_name : xr.open_dataset(f"output_T106/02-02_experimental_JCM_Veros/{component_name:s}-{segement_index:05d}.nc")
            for component_name in ["atm_daily", "ocn_daily"]
        }

        segement_time = pd.to_datetime(plot_data["atm_daily"]["time"].isel(time=0).values)

        levels_q = 1.0 + np.linspace(0, 30, 50*3+1)
        levels_tskin = np.linspace(10, 35, 35+1)
        levels_sea_surface_u = np.linspace(-1, 1, 51) * 1.0
        levels_sea_surface_salinity = np.linspace(34.5, 38, 51)

        if segement_time > pd.Timestamp("2001-02-01"):
            levels_q = 1.0 + np.linspace(0, 50, 50*3+1)
            levels_tskin = np.linspace(10, 45, 35+1)
            levels_sea_surface_salinity = np.linspace(34.5, 42, 51)




        for i, output_figure in enumerate(output_figures):

            try:
                frame = segement_index * records_per_file + i
                print(f"Plotting frame = {frame:d}")
                
                fig, axes = plt.subplots(
                    2, 2,
                    figsize=(16, 12),
                    subplot_kw = dict(
                        projection=ccrs.PlateCarree(),
                    ),
                )

                ax = axes.flatten()

                for _ax in ax.flatten():
                    _ax.gridlines(draw_labels=True)

                _data_q = plot_data["atm_daily"]["specific_humidity"].isel(time=i)
                _data_tskin = plot_data["atm_daily"]["surface_flux.tsfc"].isel(time=i) - 273.15
                _data_ocean_kinetic_energy = 0.5 * (
                      plot_data["ocn_daily"]["sea_surface_u"].isel(time=i)**2
                    + plot_data["ocn_daily"]["sea_surface_v"].isel(time=i)**2
                )
                _data_sea_surface_u = plot_data["ocn_daily"]["sea_surface_u"].isel(time=i)
                _data_sea_surface_salinity = plot_data["ocn_daily"]["sea_surface_salinity"].isel(time=i)


                    
                coords = _data_q.coords
                time_str = _data_q['time'].dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy().item()
                lat = coords["lat"].to_numpy()
                lon = coords["lon"].to_numpy()
                
                dlat = lat[1] - lat[0]
                dlon = lon[1] - lon[0]
                lat_bnd = np.concatenate((lat - 0.5*dlat, [lat[-1]+0.5*dlat]))
                lon_bnd = np.concatenate((lon - 0.5*dlon, [lon[-1]+0.5*dlon]))

                def pcolormesh(ax, data, levels, cmap, extend):
                    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')
                    mappable = ax.pcolormesh(
                        lon_bnd, lat_bnd,
                        data.transpose(),
                        norm=norm,
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap,
                    )
                    cb = plt.colorbar(ax=_ax, mappable=mappable, orientation='vertical', shrink=0.7, pad=0.1)
                    return cb



                # Plot the humidity field for the current time step
                ax_i = 0
                _ax = ax[ax_i]
                ax_i += 1
                cb = pcolormesh(_ax, _data_q, levels=levels_q, cmap=cmo.cm.rain, extend="both")
                cb.set_label("[g/kg]", fontsize=12)
                _ax.set_title("(a) Surface specific humidity")

                _ax = ax[ax_i]
                ax_i += 1
                cb = pcolormesh(_ax, _data_tskin, levels=levels_tskin, cmap=cmo.cm.thermal, extend="both")
                cb.set_label("[degC]", fontsize=12)
                _ax.set_title("(b) Surface temperature")

                _ax = ax[ax_i]
                ax_i += 1
                cb = pcolormesh(_ax, _data_sea_surface_u, levels=levels_sea_surface_u, cmap=cmo.cm.balance, extend="both")
                cb.set_label("[$\\mathrm{m} / \\mathrm{s}$]", fontsize=12)
                _ax.set_title("(c) Surface ocean zonal velocity")


                _ax = ax[ax_i]
                ax_i += 1
                cb = pcolormesh(_ax, _data_sea_surface_salinity, levels=levels_sea_surface_salinity, cmap=plt.get_cmap('hot_r'), extend="both")
                cb.set_label("[PSU]", fontsize=12)
                _ax.set_title("(d) Sea surface salinity")

                fig.suptitle(f"[{time_str:s}]")
                
                print("Saving figure: ", output_figure)
                fig.savefig(output_figure, dpi=200)
                plt.close(fig)

            except Exception:

                traceback.print_exc()
                result["status"] = "ERROR_individual_frame"

    except Exception:

        traceback.print_exc()
        result["status"] = "ERROR"

    return result

if __name__ == "__main__": 

    num_cores = multiprocessing.cpu_count()
    print(f"Parallelizing across {num_cores} cores...")
   

    input_args = [] 

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    records_per_file = 10
    plot_segement_indices = np.arange(74)

    phase = 'detect'
    for segement_index in plot_segement_indices:
        detail = dict(
            output_dir = str(output_dir),
            records_per_file = records_per_file,
            segement_index = segement_index,
            phase = phase,
        )
        
        result = plot_simulation(detail)
        if result["needs_work"]:
            detail["phase"] = "work"
            input_args.append(detail)
        else:
            print(f"Segement_index={segement_index} does not need work.")

    with multiprocessing.Pool(processes=num_cores) as pool:
        # pool.map distributes the frame indices to the plotting function
        pool.map(plot_simulation, input_args)
        
    print("All frames generated in 'figures/' folder.")



