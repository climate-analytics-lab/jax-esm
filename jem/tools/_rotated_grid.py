"""Regridding helpers shared by the tools that put an observed field on the ocean grid.

:mod:`jem.tools.ocean_bathymetry` and :mod:`jem.tools.ocean_initial_state` map
a regular lat-lon source onto the rotated Gaussian grid's cells the same way,
and a coastline convention that differed between them would make the initial
state and the bottom depth disagree about where the ocean is. The method is:

* a source point belongs to the target cell whose centre is nearest on the
  sphere (:func:`nearest_target`), found with 3-D unit vectors so the rotated
  pole and the periodic seam need no special handling;
* a target cell's value is the weighted mean of its ocean source points'
  values (:func:`weighted_target_means`), so that land in the source never
  dilutes a coastal cell;
* a wet target cell left with no source value takes the mean of its filled
  wet neighbours (:func:`fill_from_wet_neighbours`).

Everything is numpy, and nothing here imports Veros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def read_scrip_cell_centres(scrip_grid_file: str | Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return ``(nx, ny, lat, lon)`` of a SCRIP grid, ``lat`` and ``lon`` as ``(ny, nx)`` degrees.

    ``grid_center_lat``/``grid_center_lon`` are the cells' true (post-rotation)
    centres, in the row-major ``(ny, nx)`` cell order the land-fraction file
    shares.
    """
    with xr.open_dataset(scrip_grid_file) as grid_ds:
        nx, ny = (int(n) for n in grid_ds["grid_dims"].to_numpy())
        lat = grid_ds["grid_center_lat"].to_numpy().reshape(ny, nx)
        lon = grid_ds["grid_center_lon"].to_numpy().reshape(ny, nx)
    return nx, ny, lat, lon


def read_wet_mask(
    landsea_mask_file: str | Path, landsea_mask_threshold: float, ny: int, nx: int
) -> np.ndarray:
    """Return the ocean grid's wet mask, ``(ny, nx)``, ``True`` where wet.

    The one definition of which columns are ocean, shared by the setup and the
    tools: the land fraction ``lsm`` (1 = land) in the file's row-major
    ``(ny, nx)`` cell order, thresholded at ``landsea_mask_threshold`` -- a
    cell is land at or above it.
    """
    with xr.open_dataset(landsea_mask_file) as mask_ds:
        land_fraction = mask_ds["lsm"].to_numpy().reshape(ny, nx)
    return ~(land_fraction >= landsea_mask_threshold)


def unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Cartesian unit vectors of points given in degrees, shape ``lat_deg.shape + (3,)``."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def nearest_target(targets: np.ndarray, sources: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Index of the target unit vector nearest each source one (largest dot product).

    Brute force in chunks (a chunk of 2048 sources against 4608 targets is
    75 MB): a million sources is some ten billion multiplies, once, and needs
    no dependency beyond numpy.
    """
    nearest = np.empty(len(sources), dtype=np.int64)
    for start in range(0, len(sources), chunk):
        nearest[start:start + chunk] = np.argmax(sources[start:start + chunk] @ targets.T, axis=1)
    return nearest


def weighted_target_means(
    nearest: np.ndarray, weight: np.ndarray, value: np.ndarray, n_targets: int
) -> np.ndarray:
    """Weighted mean of ``value`` over the sources assigned to each target; NaN with none.

    Parameters
    ----------
    nearest : numpy.ndarray
        Target index of each source (:func:`nearest_target`).
    weight, value : numpy.ndarray
        Per-source weight and value, the same shape as ``nearest``. Sources
        that must not count (land, missing) are left out by the caller, or
        given zero weight.
    n_targets : int
        Number of target cells.

    Returns
    -------
    numpy.ndarray
        ``(n_targets,)``; NaN where the total weight is zero.

    """
    weighted = np.bincount(nearest, weights=weight * value, minlength=n_targets)
    total = np.bincount(nearest, weights=weight, minlength=n_targets)
    mean = np.full(n_targets, np.nan)
    has_source = total > 0
    mean[has_source] = weighted[has_source] / total[has_source]
    return mean


def fill_from_wet_neighbours(field: np.ndarray, wet: np.ndarray) -> int:
    """Fill NaN wet cells of ``field`` (``(ny, nx)``) in place; return how many.

    Each pass gives every unfilled wet cell the mean of its already-filled wet
    neighbours (8-neighbourhood in the grid's own index space, periodic in
    longitude, the two polar rows not neighbours of one another), until none
    is reachable. A wet region with no source cell anywhere in it (an inland
    sea or lake the source does not reach) cannot be filled from anything and
    is left NaN; the caller decides what those get.
    """
    missing = wet & np.isnan(field)
    n_requested = int(missing.sum())
    while missing.any():
        known = wet & ~np.isnan(field)
        total = np.zeros_like(field)
        count = np.zeros(field.shape)
        values = np.where(known, field, 0.0)
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
        field[reachable] = total[reachable] / count[reachable]
        missing = wet & np.isnan(field)
    return n_requested - int(missing.sum())
