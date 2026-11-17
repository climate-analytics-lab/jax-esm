"""Regrid a regular lat-lon bathymetry onto the rotated Gaussian ocean grid.

``+configuration=veros-earth`` runs its ocean on the packaged
``RotatedGaussianLatLon`` SCRIP grid, whose wet columns are fixed by the
ERA5-derived land fraction (:func:`read_wet_mask`). The bottom depth of each
wet column comes from here, in two steps:

1. :func:`etopo_to_coarse_bathymetry` block-averages NOAA NCEI's ETOPO 2022
   60-arc-second relief (21600 x 10800 points) onto a regular 0.25-degree
   grid of ocean-mean depth and ocean fraction.
2. :func:`ocean_bathymetry` averages that grid (or any regular lat-lon
   bathymetry, e.g. Veros' ``global_1deg`` asset) onto the ocean grid's cells
   and writes a small netCDF that
   :func:`jem.components.veros.setups.earth.earth_setup` turns into ``kbot``.

The set of wet columns is not an output of this tool: it is the mask file's,
and the bathymetry only says how deep each of them is.

Source and terms
----------------
The packaged ``jem/data/bathymetry_RotatedGaussianLatLon.nc`` is derived from
NOAA National Centers for Environmental Information, 2022: *ETOPO 2022 15
Arc-Second Global Relief Model*, https://doi.org/10.25921/fd45-gt74 (the
60-arc-second ``surface`` netCDF, ``ETOPO_2022_v1_60s_N90W180_surface.nc``).
The dataset's metadata states "These data were produced by NOAA and are not
subject to copyright protection in the United States. NOAA waives any
potential copyright and related rights in these data worldwide through the
Creative Commons Zero 1.0 Universal Public Domain Dedication (CC0-1.0)", so a
derivative may be redistributed under this package's MIT licence. The
citation the dataset asks for is carried in the output's ``source_citation``
attribute and in the documentation.

Regenerating the packaged file
------------------------------
The 478 MB source is neither shipped nor needed by the tests. The
0.25-degree intermediate of step 1 is kept in ``tests/unit/data/`` (under
1 MB), and a test checks that step 2 applied to it reproduces the packaged
file. To rebuild both from the public source::

    curl -O https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc

    from importlib import resources
    from jem.tools.ocean_bathymetry import etopo_to_coarse_bathymetry, ocean_bathymetry

    data = resources.files("jem.data")
    etopo_to_coarse_bathymetry(
        "ETOPO_2022_v1_60s_N90W180_surface.nc", "tests/unit/data/etopo2022_0p25deg_ocean_depth.nc"
    )
    ocean_bathymetry(
        "tests/unit/data/etopo2022_0p25deg_ocean_depth.nc",
        data / "RotatedGaussianLatLon.SCRIP.nc",
        data / "landsea_mask_fraction_RotatedGaussianLatLon.nc",
        "jem/data/bathymetry_RotatedGaussianLatLon.nc",
    )

The regridding method, and why:

* Step 1 reads the source in latitude bands (15 degrees, about 80 MB, at a
  time) and, for each 15 x 15 block of 1/60-degree points, records the mean
  depth of its ocean points (elevation below zero; depth is minus elevation)
  and the fraction of the block that is ocean. This is the same arithmetic as
  assigning every point to a target cell directly, with the nearest-centre
  choice made once per block instead of once per point: a block is 0.25
  degrees across against target cells several degrees across, so a block
  straddles two target cells' regions only along their boundary. Measured on
  the packaged grid against a direct nearest-centre assignment of every third
  source point in each direction, the block route differs by 6 m on average
  (at most 148 m, no cell by more than 200 m) over the 3018 wet cells; a
  0.5-degree block differs by up to 817 m, which is why the block is not
  coarser. Assigning every point directly needs a nearest-centre search
  over about 160 million ocean points. Both steps are numpy-only.
* Step 2 assigns each source cell to the target cell whose centre is nearest
  on the sphere (3-D unit vectors, so the rotated pole and the periodic seam
  need no special handling), and a target cell's depth is the weighted mean of
  the ocean-only depths of its source cells, weighted by ``cos(lat)`` (the
  cell's area) times the source cell's ocean fraction when the source has one,
  so that the result is the mean over the underlying ocean points.
* Only ocean source cells enter the mean: a coastal target cell that is
  mostly source land still gets the depth of the sea it does contain, rather
  than a depth diluted by zeros. Land below sea level (the Caspian Sea, the
  Dead Sea, inland depressions) is below zero in the source and so counts as
  ocean; it matters only where it shares a wet target cell with real sea.
* A wet target cell that receives no ocean source cell (a small inlet the
  mask keeps wet may fall between source ocean cells, and an ice-shelf coast
  whose surface is above sea level has none) takes the mean of its
  already-filled wet neighbours (8-neighbourhood in the grid's own index space,
  periodic in longitude), repeated until every wet cell has a depth. A wet
  region with no source ocean cell anywhere in it is set to the minimum depth.
  Both counts are recorded in the output.
* Depths are clipped to ``[minimum_depth, maximum_depth]``. The maximum is
  the model's bottom (the sum of the layer thicknesses), since a deeper
  source cell cannot be represented. The minimum is the depth of the two
  shallowest layers: Veros' own coarse global setup drops any column with
  fewer than two wet layers (``global_4deg``'s ``kbot == nz`` test), a column
  of one layer has no vertical structure for the mixing schemes to act on,
  and this setup must not drop columns, so it raises them instead.

The output is ``depth`` (metres, positive down, NaN on dry cells) with dims
``("latitude", "longitude")`` in the same row-major ``(ny, nx)`` cell order
as the land-fraction file and the SCRIP file's ``grid_center_*`` arrays,
with the cells' true (post-rotation) ``lat``/``lon`` as 2-D coordinates.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

#: Depth of the two shallowest default layers (50 m + 70 m): the least a wet
#: column may have; see the module docstring.
MINIMUM_DEPTH = 120.0

#: The sum of the default layer thicknesses in
#: :data:`jem.components.veros.setups._layers.LAYER_THICKNESSES` (restated
#: here because importing that package pulls in Veros, which switches
#: ``jax_enable_x64`` on for the process); a test pins the two together.
MAXIMUM_DEPTH = 5200.0

#: Where ETOPO 2022 comes from, what it is called, and the terms it states;
#: written into every file derived from it (as ``source_*`` attributes).
ETOPO_2022_ATTRS = {
    "source_dataset": "ETOPO 2022 15 Arc-Second Global Relief Model, 60 arc-second surface elevation",
    "source_doi": "10.25921/fd45-gt74",
    "source_citation": (
        "NOAA National Centers for Environmental Information. 2022: ETOPO 2022 15 Arc-Second"
        " Global Relief Model. NOAA National Centers for Environmental Information."
        " https://doi.org/10.25921/fd45-gt74"
    ),
    "source_url": (
        "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/"
        "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
    ),
    "source_terms": (
        "These data were produced by NOAA and are not subject to copyright protection in the"
        " United States. NOAA waives any potential copyright and related rights in these data"
        " worldwide through the Creative Commons Zero 1.0 Universal Public Domain Dedication"
        " (CC0-1.0). SPDX License: CC0-1.0 (dataset metadata, https://doi.org/10.25921/fd45-gt74)."
        " Use liability: NOAA and NCEI cannot provide any warranty as to the accuracy,"
        " reliability, or completeness of furnished data."
    ),
}

#: Source attributes (prefix) that :func:`ocean_bathymetry` copies from its
#: source file into its output, so provenance survives the second step.
_PROVENANCE_PREFIX = "source_"


def read_wet_mask(
    landsea_mask_file: str | Path, landsea_mask_threshold: float, ny: int, nx: int
) -> np.ndarray:
    """Return the ocean grid's wet mask, ``(ny, nx)``, ``True`` where wet.

    The one definition of which columns are ocean, shared by the setup and
    this tool: the land fraction ``lsm`` (1 = land) in the file's row-major
    ``(ny, nx)`` cell order, thresholded at ``landsea_mask_threshold`` -- a
    cell is land at or above it.
    """
    with xr.open_dataset(landsea_mask_file) as mask_ds:
        land_fraction = mask_ds["lsm"].to_numpy().reshape(ny, nx)
    return ~(land_fraction >= landsea_mask_threshold)


def etopo_to_coarse_bathymetry(
    etopo_file: str | Path,
    output_file: str | Path,
    *,
    block: int = 15,
    band_rows: int = 900,
    variable: str = "z",
) -> Path:
    """Block-average an ETOPO-style elevation grid to ocean-mean depth and ocean fraction.

    Parameters
    ----------
    etopo_file : str or pathlib.Path
        Global elevation, ``variable`` with dims ``(lat, lon)`` in metres,
        positive up, land positive, on a regular grid covering the whole
        sphere with ``lat`` increasing: ETOPO 2022's 60-arc-second netCDF
        (10800 x 21600). A point is ocean where its elevation is below zero.
    output_file : str or pathlib.Path
        Where to write; the parent directory is created. The result has
        ``bathymetry`` (dims ``(yt, xt)``, metres, negative down, the mean
        over the block's ocean points, zero where it has none) and
        ``ocean_fraction``, in the layout :func:`ocean_bathymetry` reads.
    block : int
        Source points per output cell along each axis; the default 15 turns
        1/60 degree into 0.25 degree. Must divide both source dimensions.
    band_rows : int
        Source rows read at a time (a multiple of ``block``); it bounds the
        memory use, not the result.
    variable : str
        The elevation variable's name.

    Returns
    -------
    pathlib.Path
        ``output_file``.

    """
    if band_rows % block:
        raise ValueError(f"band_rows ({band_rows}) must be a multiple of block ({block})")
    with xr.open_dataset(etopo_file) as ds:
        elevation = ds[variable]
        if elevation.dims != ("lat", "lon"):
            raise ValueError(f"{etopo_file}: {variable} has dims {elevation.dims}, expected ('lat', 'lon')")
        n_lat, n_lon = elevation.shape
        if n_lat % block or n_lon % block:
            raise ValueError(f"{etopo_file}: grid {elevation.shape} is not a multiple of block {block}")
        source_lat = ds["lat"].to_numpy().astype(np.float64)
        source_lon = ds["lon"].to_numpy().astype(np.float64)
        ocean_depth_sum = np.zeros((n_lat // block, n_lon // block))
        ocean_count = np.zeros((n_lat // block, n_lon // block))
        for first in range(0, n_lat, band_rows):
            rows = slice(first, min(first + band_rows, n_lat))
            z = elevation.isel(lat=rows).to_numpy()
            n_rows = z.shape[0]
            # NaN (a fill value read as missing) compares False: never ocean.
            is_ocean = z < 0
            depth = np.where(is_ocean, -z.astype(np.float64), 0.0)
            shape = (n_rows // block, block, n_lon // block, block)
            out_rows = slice(first // block, first // block + n_rows // block)
            ocean_depth_sum[out_rows] = depth.reshape(shape).sum(axis=(1, 3))
            ocean_count[out_rows] = is_ocean.reshape(shape).sum(axis=(1, 3))
    mean_depth = np.divide(
        ocean_depth_sum, ocean_count, out=np.zeros_like(ocean_depth_sum), where=ocean_count > 0
    )
    # Block centres: the mean of the block's point coordinates (the source's
    # own, so a half-point registration offset is inherited, not assumed).
    lat = source_lat.reshape(-1, block).mean(axis=1)
    lon = source_lon.reshape(-1, block).mean(axis=1)

    output = xr.Dataset(
        data_vars=dict(
            bathymetry=(
                ("yt", "xt"),
                -mean_depth,
                {"long_name": "mean depth of the ocean points in the block, negative down; 0 with no ocean", "units": "m"},
            ),
            ocean_fraction=(
                ("yt", "xt"),
                ocean_count / block**2,
                {"long_name": "fraction of the block's source points that are ocean", "units": "1"},
            ),
        ),
        coords=dict(
            yt=("yt", lat, {"units": "degrees_north"}),
            xt=("xt", lon, {"units": "degrees_east"}),
        ),
        attrs={
            "title": f"ETOPO 2022 ocean depth block-averaged to {block}x{block} source points",
            **ETOPO_2022_ATTRS,
            "method": (
                f"mean of -elevation over the ocean (elevation < 0) points of each {block}x{block}"
                " block of the source, and the fraction of ocean points; depths stored to 1 m"
            ),
            "block_points": block,
            "generator": "jem.tools.ocean_bathymetry.etopo_to_coarse_bathymetry",
        },
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Depth to the nearest metre in int16 (the ocean is at most ~11 000 m deep)
    # and the ocean fraction in uint16 steps of 1/block**2, which is exact: the
    # file is a reproducibility intermediate and must stay small.
    # Both arrays are finite by construction (zero where there is no ocean), so
    # xarray's "integer dtype without a _FillValue" warning is about a hazard
    # that cannot occur here.
    packed = {"_FillValue": None, "zlib": True, "complevel": 9, "shuffle": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xr.SerializationWarning)
        output.to_netcdf(
            output_file,
            encoding={
                "bathymetry": {"dtype": "int16", "scale_factor": 1.0, **packed},
                "ocean_fraction": {"dtype": "uint16", "scale_factor": 1.0 / block**2, **packed},
            },
        )
    return output_file


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _nearest_target(targets: np.ndarray, sources: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Index of the target unit vector nearest each source one (largest dot product).

    Brute force in chunks (a chunk of 2048 sources against 4608 targets is
    75 MB): a million sources is some ten billion multiplies, once, and needs
    no dependency beyond numpy.
    """
    nearest = np.empty(len(sources), dtype=np.int64)
    for start in range(0, len(sources), chunk):
        nearest[start:start + chunk] = np.argmax(sources[start:start + chunk] @ targets.T, axis=1)
    return nearest


def _fill_from_wet_neighbours(depth: np.ndarray, wet: np.ndarray) -> int:
    """Fill NaN wet cells of ``depth`` (``(ny, nx)``) in place; return how many.

    A wet region with no source cell anywhere in it (an inland sea or lake
    the source's ocean mask does not reach) cannot be filled from anything and
    is left NaN; the caller decides what those get.
    """
    missing = wet & np.isnan(depth)
    n_requested = int(missing.sum())
    while missing.any():
        known = wet & ~np.isnan(depth)
        total = np.zeros_like(depth)
        count = np.zeros(depth.shape)
        values = np.where(known, depth, 0.0)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == 0 and di == 0:
                    continue
                shifted_values = np.roll(values, (dj, di), axis=(0, 1))
                shifted_known = np.roll(known, (dj, di), axis=(0, 1)).astype(float)
                # np.roll wraps the poles' rows too; undo that so row 0 and
                # row ny-1 are not neighbours of one another.
                if dj == 1:
                    shifted_known[0, :] = 0.0
                elif dj == -1:
                    shifted_known[-1, :] = 0.0
                total += shifted_values * shifted_known
                count += shifted_known
        reachable = missing & (count > 0)
        if not reachable.any():
            break
        depth[reachable] = total[reachable] / count[reachable]
        missing = wet & np.isnan(depth)
    return n_requested - int(missing.sum())


def ocean_bathymetry(
    source_file: str | Path,
    scrip_grid_file: str | Path,
    landsea_mask_file: str | Path,
    output_file: str | Path,
    *,
    landsea_mask_threshold: float = 0.5,
    variable: str = "bathymetry",
    minimum_depth: float = MINIMUM_DEPTH,
    maximum_depth: float = MAXIMUM_DEPTH,
    source_description: str = "",
) -> Path:
    """Write the ocean depth on the rotated grid's cells to ``output_file``.

    Parameters
    ----------
    source_file : str or pathlib.Path
        A regular lat-lon bathymetry: ``variable`` with dims ``(yt, xt)`` in
        metres (negative down or positive down; the magnitude is used), zero
        on land, with ``xt``/``yt`` the cell centres in degrees. Veros'
        ``global_1deg`` asset ``forcing_1deg_global.nc`` is one, and so is
        the output of :func:`etopo_to_coarse_bathymetry`. An optional
        ``ocean_fraction`` variable on the same grid (the share of each source
        cell that is ocean) weights the average. Global attributes starting
        with ``source_`` are copied to the output.
    scrip_grid_file : str or pathlib.Path
        The ocean grid's SCRIP file (``grid_center_lat``/``grid_center_lon``
        are the true cell centres, ``grid_dims`` is ``(nx, ny)``).
    landsea_mask_file : str or pathlib.Path
        The land-fraction file that decides which columns are wet.
    output_file : str or pathlib.Path
        Where to write; the parent directory is created.
    landsea_mask_threshold : float
        Land fraction at or above which a cell is land. Default ``0.5``, the
        setup's default.
    variable : str
        The bathymetry variable's name in ``source_file``.
    minimum_depth, maximum_depth : float
        Wet columns are clipped to this range, metres; see the module
        docstring for why each.
    source_description : str
        Recorded in the output's ``source`` attribute, so the file says where
        its bathymetry came from; defaults to the source file's name.

    Returns
    -------
    pathlib.Path
        ``output_file``.

    """
    with xr.open_dataset(scrip_grid_file) as grid_ds:
        nx, ny = (int(n) for n in grid_ds["grid_dims"].to_numpy())
        target_lat = grid_ds["grid_center_lat"].to_numpy().reshape(ny, nx)
        target_lon = grid_ds["grid_center_lon"].to_numpy().reshape(ny, nx)
    wet = read_wet_mask(landsea_mask_file, landsea_mask_threshold, ny, nx)

    with xr.open_dataset(source_file) as source_ds:
        if source_ds[variable].dims != ("yt", "xt"):
            raise ValueError(
                f"{source_file}: {variable} has dims {source_ds[variable].dims}, expected ('yt', 'xt')"
            )
        source_depth = np.abs(source_ds[variable].to_numpy().astype(np.float64))
        source_lat = source_ds["yt"].to_numpy().astype(np.float64)
        source_lon = source_ds["xt"].to_numpy().astype(np.float64)
        if "ocean_fraction" in source_ds:
            source_fraction = source_ds["ocean_fraction"].to_numpy().astype(np.float64)
        else:
            source_fraction = np.ones_like(source_depth)
        provenance = {k: v for k, v in source_ds.attrs.items() if k.startswith(_PROVENANCE_PREFIX)}

    source_lat2, source_lon2 = np.meshgrid(source_lat, source_lon, indexing="ij")
    is_ocean = (source_depth > 0) & (source_fraction > 0)
    nearest = _nearest_target(
        _unit_vectors(target_lat, target_lon).reshape(-1, 3),
        _unit_vectors(source_lat2, source_lon2)[is_ocean],
    )

    weight = np.cos(np.deg2rad(source_lat2[is_ocean])) * source_fraction[is_ocean]
    weighted_depth = np.bincount(nearest, weights=weight * source_depth[is_ocean], minlength=nx * ny)
    total_weight = np.bincount(nearest, weights=weight, minlength=nx * ny)
    flat_depth = np.full(nx * ny, np.nan)
    has_source = total_weight > 0
    flat_depth[has_source] = weighted_depth[has_source] / total_weight[has_source]
    depth = flat_depth.reshape(ny, nx)
    depth[~wet] = np.nan

    n_filled = _fill_from_wet_neighbours(depth, wet)
    # An isolated wet region has no ocean depth to borrow from: Veros' own
    # coarse setups would make such water land, which this setup must not, so
    # it is given the minimum depth (shallow, the safe side for an inland
    # water body) and counted.
    isolated = wet & np.isnan(depth)
    n_isolated = int(isolated.sum())
    depth[isolated] = minimum_depth
    n_below = int((depth[wet] < minimum_depth).sum())
    n_above = int((depth[wet] > maximum_depth).sum())
    depth[wet] = np.clip(depth[wet], minimum_depth, maximum_depth)
    logger.info(
        "bathymetry: %d wet cells, %d filled from neighbours, %d isolated, %d raised to %g m, %d clipped to %g m",
        int(wet.sum()), n_filled, n_isolated, n_below, minimum_depth, n_above, maximum_depth,
    )

    output = xr.Dataset(
        data_vars=dict(
            depth=(
                ("latitude", "longitude"),
                depth.astype(np.float32),
                {
                    "long_name": "ocean depth at the cell centre, positive down",
                    "units": "m",
                    "coordinates": "lat lon",
                },
            )
        ),
        coords=dict(
            lat=(("latitude", "longitude"), target_lat, {"units": "degrees_north"}),
            lon=(("latitude", "longitude"), target_lon, {"units": "degrees_east"}),
        ),
        attrs={
            "title": "Ocean depth on the rotated Gaussian lat-lon grid",
            "source": source_description or f"{Path(source_file).name}:{variable}",
            **provenance,
            "method": (
                "weighted mean of the ocean-only source depths (weights cos(latitude) times the source"
                " ocean fraction when it has one) whose nearest target cell centre (3-D unit vectors)"
                " is this cell; wet cells with no ocean source take the mean of their filled wet"
                " neighbours, or the minimum depth when their whole wet region has none; wet cells"
                f" clipped to [{minimum_depth:g}, {maximum_depth:g}] m; NaN on dry cells"
            ),
            "wet_cells": int(wet.sum()),
            "cells_filled_from_neighbours": n_filled,
            "cells_isolated_set_to_minimum_depth": n_isolated,
            "cells_raised_to_minimum_depth": n_below,
            "cells_clipped_to_maximum_depth": n_above,
            "minimum_depth_m": float(minimum_depth),
            "maximum_depth_m": float(maximum_depth),
            "landsea_mask_threshold": float(landsea_mask_threshold),
            "wet_mask": (
                f"{Path(landsea_mask_file).name} lsm < {landsea_mask_threshold:g};"
                " the bathymetry does not change which cells are wet"
            ),
            "generator": "jem.tools.ocean_bathymetry.ocean_bathymetry",
        },
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    compression = {"zlib": True, "complevel": 9, "shuffle": True}
    output.to_netcdf(
        output_file,
        encoding={"depth": dict(compression), "lat": dict(compression), "lon": dict(compression)},
    )
    return output_file


def read_depth(bathymetry_file: str | Path, ny: int, nx: int) -> np.ndarray:
    """Read a file written by :func:`ocean_bathymetry` as a ``(nx, ny)`` array.

    ``(nx, ny)`` is the layout the setup's grid arrays use (the transpose of
    the file's ``(ny, nx)`` cell order, as for the land mask).
    """
    with xr.open_dataset(bathymetry_file) as ds:
        depth = ds["depth"].to_numpy()
    if depth.shape != (ny, nx):
        raise ValueError(f"{bathymetry_file}: depth has shape {depth.shape}, the grid is {(ny, nx)}")
    return depth.astype(np.float64).T
