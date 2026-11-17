"""Regrid a 1-degree bathymetry onto the rotated Gaussian ocean grid.

``+configuration=veros-earth`` runs its ocean on the packaged
``RotatedGaussianLatLon`` SCRIP grid, whose wet columns are fixed by the
ERA5-derived land fraction (:func:`read_wet_mask`). The bottom depth of each
wet column comes from here: :func:`ocean_bathymetry` averages a regular
lat-lon bathymetry (Veros' ``global_1deg`` asset, ``bathymetry``, metres
negative down) onto the grid's cells, and writes a small netCDF that
:func:`jem.components.veros.setups.earth.earth_setup` turns into ``kbot``.

The set of wet columns is not an output of this tool: it is the mask file's,
and the bathymetry only says how deep each of them is.

The regridding method, and why:

* Each source cell is assigned to the target cell whose centre is nearest on
  the sphere (3-D unit vectors, so the rotated pole and the periodic seam
  need no special handling), and a target cell's depth is the ``cos(lat)``
  weighted mean of the ocean-only depths of its source cells. At about 14
  source cells per target cell the nearest-centre (Voronoi) cell is a good
  stand-in for the target's true quadrilateral, and a conservative polygon
  intersection would add a dependency for no visible change in a field that
  is then quantised to 15 layers.
* Only ocean source cells enter the mean: a coastal target cell that is
  mostly source land still gets the depth of the sea it does contain, rather
  than a depth diluted by zeros.
* A wet target cell that receives no ocean source cell (the source stops at
  79.5 degrees, so the polar caps and ice-shelf coasts have none, and a
  small inlet the mask keeps wet may fall between source ocean cells) takes
  the mean of its already-filled wet neighbours (8-neighbourhood in the
  grid's own index space, periodic in longitude), repeated until every wet
  cell has a depth. A wet region with no source cell at all (an inland sea
  or lake the source's ocean mask does not reach) is set to the minimum depth.
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


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _nearest_target(targets: np.ndarray, sources: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Index of the target unit vector nearest each source one (largest dot product).

    Brute force in chunks: 57600 sources against 4608 targets is a few
    hundred million multiplies, once, and needs no dependency beyond numpy.
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
        ``global_1deg`` asset ``forcing_1deg_global.nc`` is one.
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
        its bathymetry came from.

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

    source_lat2, source_lon2 = np.meshgrid(source_lat, source_lon, indexing="ij")
    is_ocean = source_depth > 0
    nearest = _nearest_target(
        _unit_vectors(target_lat, target_lon).reshape(-1, 3),
        _unit_vectors(source_lat2, source_lon2)[is_ocean],
    )

    weight = np.cos(np.deg2rad(source_lat2[is_ocean]))
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
            "method": (
                "cos(latitude)-weighted mean of the ocean-only source depths whose nearest"
                " target cell centre (3-D unit vectors) is this cell; wet cells with no"
                " ocean source take the mean of their filled wet neighbours, or the minimum depth"
                f" when their whole wet region has none; wet cells clipped to [{minimum_depth:g}, {maximum_depth:g}] m; NaN on dry cells"
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
    output.to_netcdf(output_file)
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
