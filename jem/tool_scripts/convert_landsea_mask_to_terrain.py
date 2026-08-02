#!/usr/bin/env python3
"""Convert a landsea_mask-style netCDF file into a JCM-canonical terrain file.

Run once per landsea_mask file (e.g. ``landsea_mask_JCM_T31.nc``). The output
file plugs into ``jcm.terrain.TerrainData.from_file``, which requires
variables named ``orog`` and ``lsm`` on ``(lon, lat)`` dimensions.

Only plain, non-rotated (lon, lat) grids are supported -- rotated/curvilinear
grids (dims i, j with a separate true_lat/true_lon) cannot be converted by
simple renaming, since JCM's terrain loader interpolates directly on lon/lat
as real geographic coordinates.

Orography is not available in landsea_mask files, so it is always set to
zero (flat terrain).

Usage::

    python jem/tool_scripts/convert_landsea_mask_to_terrain.py \\
        --input jem/data/grid/landsea_mask_JCM_T31.nc \\
        --output jem/data/grid/terrain_JCM_T31.nc
"""

import argparse
from pathlib import Path

import xarray as xr


def convert_landsea_mask_to_terrain(input_file: Path, output_file: Path) -> None:
    ds = xr.open_dataset(input_file, engine="netcdf4")

    if not ("lon" in ds.dims and "lat" in ds.dims):
        raise ValueError(
            f"'{input_file}' does not have (lon, lat) dimensions. Only a "
            "plain, non-rotated grid can be converted by this script -- "
            "rotated/curvilinear grids (i, j) are not supported."
        )

    if "land_fraction" not in ds:
        raise ValueError(f"'{input_file}' is missing 'land_fraction'.")

    lsm = ds["land_fraction"].transpose("lon", "lat").astype("float32")
    orog = xr.zeros_like(lsm)  # flat terrain -- landsea_mask files carry no elevation data

    out = xr.Dataset(dict(orog=orog, lsm=lsm))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_file, engine="netcdf4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the landsea_mask netCDF file (lon, lat dims).",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the terrain-conformant netCDF file.",
    )
    args = parser.parse_args()

    convert_landsea_mask_to_terrain(args.input, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
