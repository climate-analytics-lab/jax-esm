import xarray as xr
from pathlib import Path
import numpy as np

label = "add_cap"

reference_file = Path("~/.cache/jcm/terrain_t31.nc")
output_file = Path(f"./funky_earth_{label:s}.nc").resolve()
print(f"Modifying reference jcm terrain file: {str(reference_file)}")
print(f"Target output file: {str(output_file)}")


ds = xr.open_dataset(reference_file)
print("Loaded xarray metadata")
print(ds)

llon, llat = np.meshgrid(
    ds.coords["lon"].to_numpy() % 360,
    ds.coords["lat"].to_numpy(),
    indexing = "ij",
)

cap_edge_latitude = 75.0

divider_width = 3.75
basin1_lon_beg = 0.0
basin1_lat_edge = -10.0

basin2_lon_beg = 120.0
basin2_lat_edge = -50.0

if label == "add_cap":
    mask = np.where(
        ( ds["lsm"].to_numpy() == 1 )
        | (np.abs(llat) >= 75.0)
        , 1
        , 0
    ).astype(int)
elif label == "simple":
     mask = np.where(
        (np.abs(llat) >= 75.0)
        | (
            (llat >= basin1_lat_edge)
            & (llon >= basin1_lon_beg)
            & (llon < basin1_lon_beg + divider_width)
        )
        | (
            (llat >= basin2_lat_edge)
            & (llon >= basin2_lon_beg)
            & (llon < basin2_lon_beg + divider_width)
        )
        , 1
        , 0
    ).astype(int)
else:
    raise Exception(f"Unknown label: {label}") 
ds["lsm"].data[:] = mask
ds["orog"].data[:] = np.where(
    (np.abs(llat) >= 75.0),
    5.0,
    ds["orog"].data
)


print(f"Writing result to: {str(output_file)}")
ds.to_netcdf(output_file)









