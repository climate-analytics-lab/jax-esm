"""Generate the land fraction a sea ice on the ``veros-earth`` ocean grid is built on.

A sea ice composed into a Veros configuration lives on the ocean's grid and is
integrated per unit area of the sea part of each cell, ``1 - fractional_mask``.
That sea has to be Veros' own wet mask -- a binary mask, thresholded from an ERA5
land fraction by ``jem.components.veros.setups.earth.GridInfo`` -- or the ice's
fluxes reach the wrong columns (:meth:`jem.fluxes.VerosExchange.check_ice_mask`
refuses an ice whose sea differs). The packaged
``landsea_mask_fraction_RotatedGaussianLatLon.nc`` is the *un-thresholded*
fraction, and the grid's SCRIP file declares every cell ocean, so neither is
that mask: this writes the thresholded one, in the layout
``seaice.land_fraction_file`` reads (variable ``lsm``, ``1`` = land).

The wet mask is read by :func:`jem.tools.veros_earth_terrain.wet_mask`, the same
function the atmosphere's terrain is derived from, so the atmosphere, the ocean
and the ice share one coastline.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import numpy as np
import xarray as xr

from jem.tools.veros_earth_terrain import LANDSEA_MASK_THRESHOLD, wet_mask

logger = logging.getLogger(__name__)

#: Name of the file this writes, in ``jem.data``.
PACKAGED_NAME = "landsea_mask_binary_RotatedGaussianLatLon.nc"


def veros_earth_ice_mask(
    output_file: str | Path,
    *,
    scrip_grid_file: str | Path | None = None,
    landsea_mask_file: str | Path | None = None,
    threshold: float = LANDSEA_MASK_THRESHOLD,
) -> Path:
    """Write the binary land fraction of Veros' wet mask, on the ocean's grid.

    Parameters
    ----------
    output_file : str or pathlib.Path
        Path to write; its parent directory is created if needed.
    scrip_grid_file : str or pathlib.Path, optional
        The ocean's SCRIP grid, for its shape. Default the packaged
        ``RotatedGaussianLatLon.SCRIP.nc``.
    landsea_mask_file : str or pathlib.Path, optional
        The ocean's land-fraction file. Default the packaged
        ``landsea_mask_fraction_RotatedGaussianLatLon.nc``.
    threshold : float
        Land fraction at and above which a column is dry; the ocean's own
        ``landsea_mask_threshold``.

    Returns
    -------
    pathlib.Path
        ``output_file``, holding ``lsm`` of shape ``(1, n_lat, n_lon)`` --
        ``1.0`` where Veros' column is dry, ``0.0`` where it is wet.

    """
    data = resources.files("jem.data")
    scrip_grid_file = scrip_grid_file or str(data / "RotatedGaussianLatLon.SCRIP.nc")
    landsea_mask_file = landsea_mask_file or str(
        data / "landsea_mask_fraction_RotatedGaussianLatLon.nc")
    with xr.open_dataset(scrip_grid_file) as grid:
        n_lon, n_lat = (int(n) for n in grid["grid_dims"].to_numpy())
    wet = wet_mask(landsea_mask_file, n_lon, n_lat, threshold=threshold)
    # `wet` is (n_lon, n_lat); the file's layout, which the runner transposes
    # back, is (time, lat, lon).
    land = (1.0 - wet).transpose()[None]
    output = xr.Dataset(
        data_vars=dict(lsm=(("time", "lat", "lon"), land.astype(np.float64), dict(
            long_name="land fraction of Veros' wet mask (1 = dry column)",
            units="1"))),
        attrs=dict(
            title="Binary land-sea mask of the Veros ocean on the rotated Gaussian grid",
            comment=(
                "1 - Veros' wet mask: the packaged ERA5 land fraction on this "
                "grid thresholded as GridInfo thresholds it. It is the "
                "seaice.land_fraction_file of a sea ice on the veros-earth "
                "ocean grid, so that the ice's sea is the ocean's wet area."),
            source_ocean_mask=Path(landsea_mask_file).name,
            source_grid=Path(scrip_grid_file).name,
            ocean_land_fraction_threshold=float(threshold),
            generated_by="jem.tools.veros_earth_ice_mask.veros_earth_ice_mask",
        ),
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(output_file)
    logger.info("Wrote %s: %d of %d columns dry.", output_file,
                int(land.sum()), land.size)
    return output_file
