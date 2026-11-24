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


def _earth_land_climatology() -> dict[str, np.ndarray]:
    """Read the T30 climatology and land mask jax-gcm packages, as plain arrays.

    The files are ``jcm.data.bc``'s ``t30/clim/forcing.nc`` and ``terrain.nc``,
    the ones ``veros-earth`` and ``earth-slab`` run on. Fields are ``(lon, lat)``
    or, for the monthly ones, ``(lon, lat, month)``.
    """
    from importlib import resources

    bc = resources.files("jcm.data.bc") / "t30" / "clim"
    with (
        xr.open_dataset(str(bc / "forcing.nc")) as forcing,
        xr.open_dataset(str(bc / "terrain.nc")) as terrain,
    ):
        _, gauss_weight = np.polynomial.legendre.leggauss(forcing.sizes["lat"])
        return dict(
            lat=forcing["lat"].to_numpy(),
            gauss_weight=gauss_weight,
            land_fraction=terrain["lsm"].to_numpy(),
            snow_cover=forcing["snowc"].to_numpy(),
            albedo=forcing["alb"].to_numpy(),
            soil_water=forcing["soilw_am"].to_numpy(),
            skin_temperature=forcing["stl"].to_numpy(),
        )


def earth_ice_free_land_means() -> dict[str, float]:
    """Area-weighted means of Earth's land boundary conditions over ice-free land.

    These are the numbers :func:`idealised_forcing`'s defaults are rounded
    from. They are computed from the climatology and land mask jax-gcm
    packages for T30 (:func:`_earth_land_climatology`).

    *Land* is a cell whose land fraction exceeds 0.5, weighted by that
    fraction and by the cell's area (the Gaussian weight of its latitude row;
    the longitude spacing is uniform). *Ice-free* excludes the cells whose snow
    cover (``snowc``, SWE / ``sd2sc``, whose ``min(1, .)`` is the cover
    fraction) averages at least 1 over the year: the ice sheets, which
    :func:`earth_ice_sheet_means` describes. Seasonally snow-covered land
    stays in, so the means are over land without a permanent ice cover, not
    over snow-free land.

    Returns
    -------
    dict[str, float]
        ``bare_land_albedo`` (fraction), ``soil_water_availability``
        (``ws / wsmx``, fraction) and ``land_skin_temperature`` (K, the annual
        mean of the monthly area-weighted mean); the fit of the annual-mean
        zonal-mean skin temperature to ``T = T_equator - drop * sin(lat)**2``
        (``land_skin_temperature_equator``, ``land_skin_temperature_pole_drop``
        in K, weighted by each latitude row's land area) and its root-mean-square
        misfit ``land_skin_temperature_fit_rms`` (K); also ``land_fraction``
        of the sphere that is ice-free land, and ``land_cells``.

    """
    earth = _earth_land_climatology()
    land_fraction = earth["land_fraction"]  # (lon, lat)
    gauss_weight = earth["gauss_weight"]
    ice_free = (land_fraction > 0.5) & (earth["snow_cover"].mean(-1) < 1.0)
    weight = land_fraction * ice_free * gauss_weight[None, :]
    total = weight.sum()

    def mean(field: str) -> float:
        values = earth[field]
        if values.ndim == 3:  # monthly: the annual mean of the monthly means
            return float((weight[..., None] * values).sum() / total / values.shape[-1])
        return float((weight * values).sum() / total)

    # The annual-mean zonal mean over each latitude row's ice-free land, and
    # the row's land area as its weight, for the sin^2(latitude) fit.
    annual = earth["skin_temperature"].mean(-1)
    row_weight = weight.sum(axis=0)
    has_land = row_weight > 0
    zonal = (weight * annual).sum(axis=0)[has_land] / row_weight[has_land]
    sin2 = np.sin(np.deg2rad(earth["lat"][has_land])) ** 2
    design = np.stack([np.ones_like(sin2), -sin2], axis=1)
    root = np.sqrt(row_weight[has_land])
    equator, drop = np.linalg.lstsq(design * root[:, None], zonal * root, rcond=None)[0]
    misfit = design @ np.array([equator, drop]) - zonal
    return dict(
        bare_land_albedo=mean("albedo"),
        soil_water_availability=mean("soil_water"),
        land_skin_temperature=mean("skin_temperature"),
        land_skin_temperature_equator=float(equator),
        land_skin_temperature_pole_drop=float(drop),
        land_skin_temperature_fit_rms=float(
            np.sqrt((row_weight[has_land] * misfit**2).sum() / row_weight[has_land].sum())
        ),
        land_fraction=float(weight.sum() / (gauss_weight.sum() * land_fraction.shape[0])),
        land_cells=int(ice_free.sum()),
    )


def earth_ice_sheet_means() -> dict[str, float]:
    """Area-weighted means of Earth's ice sheets poleward of :data:`CAP_LATITUDE`.

    These are the numbers :func:`idealised_forcing`'s polar-cap defaults are
    rounded from: the land of Earth's T30 climatology that is covered by snow
    or ice all year (annual-mean ``snowc`` of at least 1) at or poleward of
    ``CAP_LATITUDE``, which is Antarctica and the Greenland sheet's northern
    end. Cells are weighted as in :func:`earth_ice_free_land_means`.

    Returns
    -------
    dict[str, float]
        ``ice_cap_albedo`` (the background albedo ``alb``, fraction),
        ``ice_cap_soil_water_availability`` (``ws / wsmx``, fraction),
        ``ice_cap_skin_temperature`` (K, the annual mean of the monthly
        area-weighted mean) and ``land_cells``.

    """
    earth = _earth_land_climatology()
    poleward = np.abs(earth["lat"]) >= CAP_LATITUDE
    ice_sheet = (
        (earth["land_fraction"] > 0.5)
        & (earth["snow_cover"].mean(-1) >= 1.0)
        & poleward[None, :]
    )
    weight = earth["land_fraction"] * ice_sheet * earth["gauss_weight"][None, :]
    total = weight.sum()
    months = earth["skin_temperature"].shape[-1]
    return dict(
        ice_cap_albedo=float((weight * earth["albedo"]).sum() / total),
        ice_cap_soil_water_availability=float(
            (weight[..., None] * earth["soil_water"]).sum() / total / months
        ),
        ice_cap_skin_temperature=float(
            (weight[..., None] * earth["skin_temperature"]).sum() / total / months
        ),
        land_cells=int(ice_sheet.sum()),
    )


def idealised_forcing(
    terrain_file: str | Path,
    output_file: str | Path,
    *,
    bare_land_albedo: float = 0.2,
    soil_water_availability: float = 0.65,
    land_skin_temperature_equator: float = 302.0,
    land_skin_temperature_pole_drop: float = 40.0,
    ice_cap_albedo: float = 0.47,
    ice_cap_soil_water_availability: float = 0.4,
    ice_cap_skin_temperature: float = 237.0,
    ice_cap_snow_cover: float = 1.0,
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

    **Why a zonally uniform land.** An idealised geography has no soils,
    vegetation or snowfields to map, and any pattern copied from Earth would
    place Earth's geography on a different one (on the double-drake mask, 442
    of the 896 land cells are Earth *ocean* cells). The land is therefore
    two surfaces that depend on latitude alone: the ice-free land and the
    polar cap. The file needs nothing from the terrain file but its grid: land
    fields are written in every cell and only the cells the terrain file calls
    land are read. Nothing varies with the month either, so the annual cycle of
    the land's boundary conditions is absent by construction (the land still
    has the seasonal cycle of the insolation it receives). The file
    nevertheless has twelve records, because JCM treats a twelve-record file as
    a monthly climatology to replay every year
    (``atmosphere.forcing.align=wrap_year``).

    **The ice-free land** equatorward of :data:`CAP_LATITUDE` has a uniform
    albedo and soil water and a skin-temperature climatology that falls with
    latitude as ``T = T_equator - drop * sin(latitude)**2``. The defaults are
    Earth's ice-free land (:func:`earth_ice_free_land_means`), rounded:

    ===================================  ==========  =========================
    field                                default     Earth's ice-free land
    ===================================  ==========  =========================
    ``bare_land_albedo``                 0.2         0.185
    ``soil_water_availability``          0.65        0.646
    ``land_skin_temperature_equator``    302 K       301.6 K
    ``land_skin_temperature_pole_drop``  40 K        40.1 K
    ===================================  ==========  =========================

    The temperature profile is the fit of Earth's annual-mean zonal-mean land
    skin temperature to ``sin(latitude)**2`` (root-mean-square misfit 2.3 K),
    which has a 290 K area-weighted mean over Earth's land, as a uniform
    climatology would. It is the climatology the land's own heat budget is
    measured from (SPEEDY's slab land adds its anomaly to it and relaxes the
    anomaly on 40 days), not a target the land relaxes to; it matters because the
    relaxation is weak, so a uniform value leaves the high latitudes' winters
    far too warm and gives the temperature at which the heat budget starts.

    **The polar cap** is land poleward of :data:`CAP_LATITUDE` in every
    geography this module makes (:func:`idealised_terrain`), so it is an ice
    sheet: snow-covered all year (``snowc`` is ``ice_cap_snow_cover``, at or
    above 1 meaning full cover; SPEEDY then takes the albedo of snow,
    ``albsn``, over the background one), with Earth's ice-sheet background
    albedo, soil-water availability and skin temperature
    (:func:`earth_ice_sheet_means`: 0.47, 0.41 and 237 K, rounded). A background
    albedo of at least 0.4 also marks the cell as land ice for SPEEDY's slab
    land, whose slab is then the thick, heavy ice one. Without this the cap is
    a bare, evaporating land at 290 K whose summer sun, at 24 hours a day,
    warms it to 315 K and above.

    **No sea ice, a harmless SST.** ``icec`` is zero: a coupled ocean forms no
    ice the atmosphere sees. ``sst`` is a field the coupled run overwrites from
    the ocean at every coupling step (``coupling.exchanged_forcing``); the
    value written only has to be a valid sea surface temperature, and is the
    same 288.15 K ``ForcingData.zeros`` uses.

    Parameters
    ----------
    terrain_file : str or pathlib.Path
        The idealised terrain file this forcing goes with (its shape picks the
        grid, see :func:`grid_axes`).
    output_file : str or pathlib.Path
        Full path to write to; its parent directory is created if needed.
    bare_land_albedo : float
        Snow-free albedo of the ice-free land, fraction in [0, 1].
    soil_water_availability : float
        Root-zone soil-water availability ``ws / wsmx`` of the ice-free land,
        fraction in [0, 1].
    land_skin_temperature_equator : float
        Skin-temperature climatology of the ice-free land at the equator, K.
    land_skin_temperature_pole_drop : float
        How far that climatology falls from the equator to the pole, K
        (``sin(latitude)**2`` shaped).
    ice_cap_albedo : float
        Background albedo of the polar cap, fraction in [0, 1].
    ice_cap_soil_water_availability : float
        Soil-water availability of the polar cap, fraction in [0, 1].
    ice_cap_skin_temperature : float
        Skin-temperature climatology of the polar cap, K.
    ice_cap_snow_cover : float
        Snow cover ``snowc`` (SWE / ``sd2sc``) of the polar cap; 1 is full cover.
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

    # Everything depends on latitude alone: a (lat,) profile broadcast over
    # longitude, and over the twelve months for the monthly fields.
    on_cap = np.abs(lat) >= CAP_LATITUDE
    ice_free_temperature = (
        land_skin_temperature_equator
        - land_skin_temperature_pole_drop * np.sin(np.deg2rad(lat)) ** 2
    )

    def profile(ice_free, cap) -> np.ndarray:
        row = np.where(on_cap, cap, ice_free)
        return np.broadcast_to(row[None, :], shape[:2])

    def field(values: np.ndarray, long_name: str, units: str) -> xr.Variable:
        monthly_values = np.broadcast_to(values[..., None], shape).astype(np.float32)
        return xr.Variable(monthly, monthly_values, attrs=dict(long_name=long_name, units=units))

    output = xr.Dataset(
        data_vars=dict(
            alb=xr.Variable(
                ("lon", "lat"),
                profile(bare_land_albedo, ice_cap_albedo).astype(np.float32),
                attrs=dict(long_name="background land albedo [0-1]", units="1"),
            ),
            snowc=field(
                profile(0.0, ice_cap_snow_cover),
                "snow cover, SWE / sd2sc (full on the polar cap, none elsewhere)", "1",
            ),
            soilw_am=field(
                profile(soil_water_availability, ice_cap_soil_water_availability),
                "soil-water availability ws/wsmx [0-1]", "1",
            ),
            stl=field(
                profile(ice_free_temperature, ice_cap_skin_temperature),
                "land skin-temperature climatology", "K",
            ),
            icec=field(profile(0.0, 0.0), "sea-ice concentration [0-1] (zero: no sea ice)", "1"),
            sst=field(
                profile(sea_surface_temperature, sea_surface_temperature),
                "sea-surface temperature, overwritten by the ocean", "K",
            ),
        ),
        coords=dict(
            lon=("lon", lon, dict(units="degrees_east")),
            lat=("lat", lat, dict(units="degrees_north")),
            time=("time", time),
        ),
        attrs=dict(
            title="Latitude-dependent land boundary conditions of an idealised planet, in JCM's forcing layout",
            grid_type=terrain.attrs.get("grid_type", "gaussian"),
            source="jem.tools.idealised_terrain.idealised_forcing",
            comment=(
                "Zonally uniform and constant in time. Ice-free land: uniform albedo and "
                "soil water, skin temperature T_equator - drop * sin(lat)**2; polar cap "
                f"(|lat| >= {CAP_LATITUDE}): full snow cover, ice-sheet albedo, soil water "
                "and skin temperature. Values are rounded fits to Earth's land in "
                "jax-gcm's T30 climatology. Replay as a climatology (forcing.align=wrap_year)."
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
