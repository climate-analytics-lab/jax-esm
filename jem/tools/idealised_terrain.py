"""Generate the boundary conditions of an idealised planet for JCM.

A real terrain file is ERA5-derived (see the packaged
``terrain_JCM_T31.nc``): its land-sea mask is the actual coastline, at
whatever resolution and grid it was regridded to. An idealised experiment
(an aquaplanet, a two-continent "double-drake" ocean, ...) wants a
land-sea mask that is a simple, reproducible *geometric shape* on the same
grid instead, at whatever resolution the atmosphere happens to run at --
which is not a file worth shipping once per shape per resolution, but is
cheap to *generate* whenever it is needed.

Two generators, one per file the atmosphere reads:

* :func:`idealised_terrain` writes the **terrain** file (land-sea mask and
  orography). Its ``reference_file`` must be JCM-canonical
  (:func:`jcm.terrain.TerrainData.from_file`'s layout: dims ``(lon, lat)``,
  data variables ``lsm`` and ``orog``, e.g. the packaged
  ``jem/data/terrain_JCM_T31.nc``); only its *shape* (which grid the output is
  on) and, for ``"capped_earth"``, its own land mask are used. The ``lon`` and
  ``lat`` axes written are the grid's own (:func:`grid_axes`), never the
  reference file's labels: JCM interpolates a terrain file onto its grid *by
  coordinate*, so a label that is off by a fraction of a degree moves the mask.
  The resulting ``orog`` is always zero: an idealised planet has no real
  orography, only a land-sea mask.
* :func:`idealised_forcing` writes the matching **forcing** file, the land
  boundary conditions SPEEDY's land reads (bare-land albedo, snow cover, soil
  water, the land skin-temperature climatology). Without one the atmosphere
  keeps ``jcm.forcing.ForcingData.zeros``: a bare-land albedo of 0, so the land
  absorbs all the sunlight it receives. Earth's climatology cannot stand in,
  because it describes Earth's continents, not an idealised geography's.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr

from jem.components.slab.grid import to_degrees

#: Latitude poleward of which every idealised geography is land -- a polar
#: ice cap, whatever the ocean basins below it look like.
CAP_LATITUDE = 75.0


@dataclasses.dataclass(frozen=True)
class _Basin:
    """One meridional land divider: land from `lat_edge` to the pole, in a
    longitude band `width` degrees wide starting at `lon_beg`.
    """

    lon_beg: float
    lat_edge: float
    width: float


@dataclasses.dataclass(frozen=True)
class _Geography:
    """The land-mask geometry for one ``planet_type``.

    ``include_reference_land`` additionally ORs in ``reference_file``'s own
    land mask (``"capped_earth"``: real continents, plus a capped ice edge).
    ``basins`` are meridional land dividers on top of the polar cap.
    """

    include_reference_land: bool = False
    basins: tuple[_Basin, ...] = ()


#: One entry per supported `planet_type`, with every geometry constant
#: `modify_jcm_terrain.py` (the driver this replaces) hard-coded, unchanged.
_GEOGRAPHIES: dict[str, _Geography] = {
    "aquaplanet": _Geography(),
    "capped_earth": _Geography(include_reference_land=True),
    "toy_earth": _Geography(basins=(
        _Basin(lon_beg=0.0, lat_edge=-10.0, width=3.75),
        _Basin(lon_beg=120.0, lat_edge=-50.0, width=3.75),
    )),
    "double_drake": _Geography(basins=(
        _Basin(lon_beg=0.0, lat_edge=-45.0, width=3.85),
        _Basin(lon_beg=67.5, lat_edge=-45.0, width=3.85),
    )),
    "double_drake_equal_basin_width": _Geography(basins=(
        _Basin(lon_beg=0.0, lat_edge=-45.0, width=3.85),
        _Basin(lon_beg=180.0, lat_edge=-45.0, width=3.85),
    )),
}


def grid_axes(nlon: int, nlat: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``lon`` and ``lat`` axes, in degrees, of JCM's grid with this shape.

    JCM picks its spectral truncation from the nodal shape, so the shape alone
    names the grid. The axes are the grid's own cell centres, converted the way
    ``jcm.utils.data_to_xarray`` writes its coordinates
    (:func:`jem.components.slab.grid.to_degrees`), so a dataset on them merges
    with the atmosphere's on an exact axis. The latitudes are the Gaussian
    latitudes of that grid (for T31, ``-87.159 ... 87.159``), not a rounded or
    padded copy of them.

    Parameters
    ----------
    nlon, nlat : int
        The nodal shape ``(lon, lat)``, e.g. ``(96, 48)`` for T31.

    Returns
    -------
    lon, lat : numpy.ndarray
        float64 axes, ``lon`` ascending from 0 and ``lat`` south to north.

    Raises
    ------
    ValueError
        If JCM has no grid with this shape.

    """
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    try:
        horizontal = get_speedy_coords(nodal_shape=(nlon, nlat)).horizontal
    except Exception as error:  # jcm raises its own types for an unknown shape
        raise ValueError(
            f"JCM has no grid with nodal shape (lon, lat) = {(nlon, nlat)}."
        ) from error
    return to_degrees(horizontal.longitudes), to_degrees(horizontal.latitudes)


def idealised_terrain(
    reference_file: str | Path, planet_type: str, output_file: str | Path
) -> Path:
    """Write a JCM terrain file whose land-sea mask is an idealised geography.

    Parameters
    ----------
    reference_file : str or pathlib.Path
        A JCM-canonical terrain file (dims ``(lon, lat)``, vars ``lsm``/
        ``orog``, e.g. ``jem/data/terrain_JCM_T31.nc``). Its shape picks the
        grid the output is written on (:func:`grid_axes`); its own coordinate
        labels are not copied. For ``planet_type="capped_earth"`` only, its
        land mask is part of the output, taken by index: it must therefore sit
        on the grid's rows.
    planet_type : str
        One of ``"aquaplanet"``, ``"capped_earth"``, ``"toy_earth"``,
        ``"double_drake"``, ``"double_drake_equal_basin_width"``.
    output_file : str or pathlib.Path
        Full path to write to; its parent directory is created if needed.

    Returns
    -------
    pathlib.Path
        ``output_file``.

    Raises
    ------
    ValueError
        If ``planet_type`` is not one of the geographies above; the message
        names the value given and the valid ones.

    """
    geography = _GEOGRAPHIES.get(planet_type)
    if geography is None:
        raise ValueError(
            f"Unknown planet_type {planet_type!r}; expected one of "
            f"{sorted(_GEOGRAPHIES)!r}."
        )

    reference = xr.open_dataset(reference_file)
    lon, lat = grid_axes(reference.sizes["lon"], reference.sizes["lat"])
    # Longitudes are wrapped to [0, 360) for the basin geometry below only --
    # a basin's bounds are given in that convention -- the output keeps the
    # grid's own (unwrapped) coordinate values.
    mesh_lon, mesh_lat = np.meshgrid(lon % 360, lat, indexing="ij")

    is_land = np.abs(mesh_lat) >= CAP_LATITUDE
    if geography.include_reference_land:
        is_land = is_land | (reference["lsm"].to_numpy() == 1)
    for basin in geography.basins:
        in_basin = (
            (mesh_lat >= basin.lat_edge)
            & (mesh_lon >= basin.lon_beg)
            & (mesh_lon < basin.lon_beg + basin.width)
        )
        is_land = is_land | in_basin
    # Cast to the reference file's own `lsm` dtype (float32 for the packaged
    # ERA5-derived terrain) rather than a hardcoded one, so an idealised
    # geography is stored at the same precision as the real one it stands in
    # for -- the deleted `modify_jcm_terrain.py` got this for free by
    # mutating the reference dataset's own array in place; building a fresh
    # one here means matching its dtype explicitly instead.
    mask = is_land.astype(reference["lsm"].dtype)
    orog = np.zeros_like(reference["orog"].to_numpy())

    output = xr.Dataset(
        data_vars=dict(
            lsm=(("lon", "lat"), mask),
            orog=(("lon", "lat"), orog),
        ),
        coords=dict(lon=("lon", lon), lat=("lat", lat)),
        attrs=dict(
            title=f"Idealised {planet_type} terrain in JCM's terrain layout",
            grid_type=reference.attrs.get("grid_type", "gaussian"),
            source="jem.tools.idealised_terrain.idealised_terrain",
            planet_type=planet_type,
            comment=(
                "lsm is 1 on land and 0 on ocean; orog is zero. lon and lat are "
                "the grid's own axes, in degrees, as jcm writes them."
            ),
        ),
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(output_file)
    return output_file


def earth_ice_free_land_means() -> dict[str, float]:
    """Area-weighted means of Earth's land boundary conditions over ice-free land.

    These are the numbers :func:`idealised_forcing`'s defaults are rounded
    from. They are computed from the climatology and land mask jax-gcm
    packages for T30 (``jcm.data.bc``, ``t30/clim/forcing.nc`` and
    ``terrain.nc``), the files ``veros-earth`` and ``earth-slab`` run on.

    *Land* is a cell whose land fraction exceeds 0.5, weighted by that
    fraction and by the cell's area (the Gaussian weight of its latitude row;
    the longitude spacing is uniform). *Ice-free* excludes the cells whose snow
    cover (``snowc``, SWE / ``sd2sc``, whose ``min(1, .)`` is the cover
    fraction) averages at least 1 over the year: the ice sheets, which a
    uniform idealised land has no counterpart for. Seasonally snow-covered
    land stays in, so the means are over land without a permanent ice cover,
    not over snow-free land.

    Returns
    -------
    dict[str, float]
        ``bare_land_albedo`` (fraction), ``soil_water_availability``
        (``ws / wsmx``, fraction) and ``land_skin_temperature`` (K, the annual
        mean of the monthly area-weighted mean); also ``land_fraction`` of the
        sphere that is ice-free land, and ``land_cells``.

    """
    from importlib import resources

    bc = resources.files("jcm.data.bc") / "t30" / "clim"
    with (
        xr.open_dataset(str(bc / "forcing.nc")) as forcing,
        xr.open_dataset(str(bc / "terrain.nc")) as terrain,
    ):
        land_fraction = terrain["lsm"].to_numpy()  # (lon, lat)
        _, gauss_weight = np.polynomial.legendre.leggauss(forcing.sizes["lat"])
        ice_free = (land_fraction > 0.5) & (forcing["snowc"].mean("time").to_numpy() < 1.0)
        weight = land_fraction * ice_free * gauss_weight[None, :]
        total = weight.sum()

        def mean(field: str) -> float:
            values = forcing[field].to_numpy()
            if values.ndim == 3:  # monthly: the annual mean of the monthly means
                return float((weight[..., None] * values).sum() / total / values.shape[-1])
            return float((weight * values).sum() / total)

        return dict(
            bare_land_albedo=mean("alb"),
            soil_water_availability=mean("soilw_am"),
            land_skin_temperature=mean("stl"),
            land_fraction=float(weight.sum() / (gauss_weight.sum() * land_fraction.shape[0])),
            land_cells=int(ice_free.sum()),
        )


def idealised_forcing(
    terrain_file: str | Path,
    output_file: str | Path,
    *,
    bare_land_albedo: float = 0.2,
    soil_water_availability: float = 0.65,
    land_skin_temperature: float = 290.0,
    sea_surface_temperature: float = 288.15,
) -> Path:
    """Write the JCM-canonical forcing file of an idealised planet.

    The file is what ``forcing@atmosphere.forcing=from_file`` reads
    (:func:`jcm.forcing.ForcingData.from_file`): monthly fields on the
    terrain file's grid, dims ``(lon, lat, time)`` (``alb`` has no time axis).
    SPEEDY's land reads the bare-land albedo (``alb``), the snow cover
    (``snowc``), the soil-water availability (``soilw_am``) and the land
    skin-temperature climatology (``stl``) that its slab land departs from;
    the ocean side reads ``icec`` and ``sst``.

    **Why a uniform land.** An idealised geography has no soils, vegetation or
    snowfields to map, and any pattern copied from Earth would place Earth's
    geography on a different one (on the double-drake mask, 442 of the 896 land
    cells are Earth *ocean* cells). The land is therefore one spatially
    uniform surface, and the file needs nothing from the terrain file but its
    grid: land fields are written in every cell and only the cells the terrain
    file calls land are read. Nothing varies with the month either, so the
    annual cycle of the land's boundary conditions is absent by construction
    (the land still has the seasonal cycle of the insolation it receives). The
    file nevertheless has twelve records, because JCM treats a twelve-record
    file as a monthly climatology to replay every year
    (``atmosphere.forcing.align=wrap_year``).

    **Where the numbers come from.** The defaults are the area-weighted means
    over Earth's ice-free land in jax-gcm's T30 climatology
    (:func:`earth_ice_free_land_means`), rounded to a round value:

    ============================  ==========  =========================
    field                         default     Earth's ice-free land mean
    ============================  ==========  =========================
    ``bare_land_albedo``          0.2         0.185
    ``soil_water_availability``   0.65        0.646
    ``land_skin_temperature``     290 K       289.9 K (annual mean)
    ============================  ==========  =========================

    The land skin temperature is the climatology the land's own heat budget is
    measured from (SPEEDY's slab land adds its anomaly to it), not a target the
    land relaxes to.

    **No snow, no sea ice, a harmless SST.** ``snowc`` and ``icec`` are zero:
    the planet has no permanent ice and a coupled ocean forms no ice the
    atmosphere sees. ``sst`` is a field the coupled run overwrites from the
    ocean at every coupling step (``coupling.exchanged_forcing``); the value
    written only has to be a valid sea surface temperature, and is the same
    288.15 K ``ForcingData.zeros`` uses.

    Parameters
    ----------
    terrain_file : str or pathlib.Path
        The idealised terrain file this forcing goes with (its shape picks the
        grid, see :func:`grid_axes`).
    output_file : str or pathlib.Path
        Full path to write to; its parent directory is created if needed.
    bare_land_albedo : float
        Snow-free land albedo, fraction in [0, 1].
    soil_water_availability : float
        Root-zone soil-water availability ``ws / wsmx``, fraction in [0, 1].
    land_skin_temperature : float
        Land skin-temperature climatology, K.
    sea_surface_temperature : float
        Placeholder sea surface temperature, K.

    Returns
    -------
    pathlib.Path
        ``output_file``.

    """
    terrain = xr.open_dataset(terrain_file)
    lon, lat = grid_axes(terrain.sizes["lon"], terrain.sizes["lat"])
    # Twelve month starts of one non-leap year: the climatology's time labels
    # are nominal (it is replayed every year), the file only has to be a valid
    # twelve-record monthly axis.
    time = np.arange("2001-01", "2002-01", dtype="datetime64[M]").astype("datetime64[ns]")
    shape = (lon.size, lat.size, time.size)
    monthly = ("lon", "lat", "time")

    def field(value: float, long_name: str, units: str) -> xr.Variable:
        return xr.Variable(
            monthly, np.full(shape, value, dtype=np.float32),
            attrs=dict(long_name=long_name, units=units),
        )

    output = xr.Dataset(
        data_vars=dict(
            alb=xr.Variable(
                ("lon", "lat"), np.full(shape[:2], bare_land_albedo, dtype=np.float32),
                attrs=dict(long_name="bare-land albedo [0-1]", units="1"),
            ),
            snowc=field(0.0, "snow cover, SWE / sd2sc (zero: no snow)", "1"),
            soilw_am=field(soil_water_availability, "soil-water availability ws/wsmx [0-1]", "1"),
            stl=field(land_skin_temperature, "land skin-temperature climatology", "K"),
            icec=field(0.0, "sea-ice concentration [0-1] (zero: no sea ice)", "1"),
            sst=field(sea_surface_temperature, "sea-surface temperature, overwritten by the ocean", "K"),
        ),
        coords=dict(
            lon=("lon", lon, dict(units="degrees_east")),
            lat=("lat", lat, dict(units="degrees_north")),
            time=("time", time),
        ),
        attrs=dict(
            title="Uniform land boundary conditions of an idealised planet, in JCM's forcing layout",
            grid_type=terrain.attrs.get("grid_type", "gaussian"),
            source="jem.tools.idealised_terrain.idealised_forcing",
            comment=(
                "Spatially uniform and constant in time: the same value in every "
                "cell and month. Values are the rounded area-weighted means over "
                "Earth's ice-free land in jax-gcm's T30 climatology. Replay as a "
                "climatology (forcing.align=wrap_year)."
            ),
        ),
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # A constant field compresses to almost nothing; without this the file is
    # over a megabyte, which is too much to package.
    output.to_netcdf(
        output_file,
        encoding={name: dict(zlib=True, complevel=9) for name in output.data_vars},
    )
    return output_file
