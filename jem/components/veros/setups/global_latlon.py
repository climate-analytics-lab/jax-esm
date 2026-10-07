"""A realistic global Veros ocean on a regular lat-lon grid, started from observations.

The other two cases are idealised in the ways that matter for a forecast:
:mod:`.double_drake` is a two-continent toy, and :mod:`.earth` has a flat
bottom and starts from a horizontally uniform, linearly stratified cold start.
This case is the one to use when the *initial ocean state itself* is the
object of study -- a sensitivity of an atmospheric forecast to the ocean
state, say -- because the ocean it starts from is the observed one:

- the grid, the bathymetry and the land-sea mask are those of Veros' own
  ``global_1deg`` asset (Levitus World Ocean Atlas, 1 degree, 79.5S-79.5N);
- the temperature and salinity are the Levitus annual-mean climatology from
  the same file, interpolated onto this case's (much coarser) layers;
- an optional ``sea_surface_temperature`` -- an observed analysis for the
  start date, e.g. ERA5's -- replaces the climatological temperature of the
  mixed layer (see :func:`blend_observed_sst`), so the surface the atmosphere
  sees on the first step is that day's ocean, not an annual mean.

The climatology file is the asset Veros downloads for its own
``global_1deg`` setup; :func:`climatology_file` fetches it (checksummed)
through Veros' asset cache, so nothing is packaged here.
"""

# Importing this before `veros.core.operators` is what points Veros at the
# JAX backend before its operators bind to a default one -- see
# `jem.components.veros_component.configure_veros_runtime`'s docstring and
# the same import in `.earth`.
from jem.components import veros_component  # noqa: F401

import os
from collections.abc import Sequence

import jax.numpy as npx
import numpy as np
import xarray as xr
from veros import VerosSetup, veros_routine

def stretched_layer_centres(
    top: float = 5.0,
    spacing: float = 10.0,
    n_uniform: int = 5,
    growth: float = 1.25,
    n: int = 25,
) -> tuple[float, ...]:
    """Depths of ``n`` layer centres: ``n_uniform`` evenly spaced, then stretched.

    The first ``n_uniform`` centres are ``spacing`` apart starting at ``top``;
    after that each gap is ``growth`` times the one above it. With the
    defaults the upper 50 m are resolved by five 10 m layers -- the mixed
    layer that sets how fast the SST responds to the surface fluxes (the
    shared 15-layer column of `.earth` puts the whole upper 50 m in one
    cell) -- and 25 layers reach 4.8 km.
    """
    gaps = [spacing] * (n_uniform - 1)
    while len(gaps) < n - 1:
        gaps.append(gaps[-1] * growth)
    return tuple(float(top + d) for d in np.concatenate([[0.0], np.cumsum(gaps)]))


#: Layer-centre depths in metres, surface first; see
#: :func:`stretched_layer_centres` and :func:`layer_thicknesses_from_centres`.
GLOBAL_LATLON_LAYER_CENTRES: tuple[float, ...] = stretched_layer_centres()


def layer_thicknesses_from_centres(centres: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the layer thicknesses Veros needs to put its T points at ``centres``.

    Veros is given layer *thicknesses* (``dzt``) and derives its T-point
    depths (``zt``) from them by a recursion anchored at the sea floor that
    places every interior interface exactly midway between the T points
    either side of it (``veros.core.numerics.u_centered_grid``). Only a
    uniform column has T points at its cell centres too, so for any other
    thicknesses the derived ``zt`` zigzags about the cell centres -- with a
    geometric stretching, badly enough to be non-monotonic. Specifying the
    centres and deriving the thicknesses from the same midpoint rule (and a
    bottom T point at the centre of the bottom layer, where the recursion
    starts) makes Veros' ``zt`` reproduce ``centres`` exactly.

    Parameters
    ----------
    centres : Sequence[float] or numpy.ndarray
        Strictly increasing T-point depths, metres, positive down, surface
        first.

    Returns
    -------
    numpy.ndarray
        Layer thicknesses, metres, surface first.

    """
    z = np.asarray(centres, dtype=float)
    interfaces = np.empty(z.size + 1)
    interfaces[0] = 0.0
    interfaces[1:-1] = (z[:-1] + z[1:]) / 2
    interfaces[-1] = 2 * z[-1] - interfaces[-2]
    thickness = np.diff(interfaces)
    if np.any(thickness <= 0):
        raise ValueError(f"layer centres {centres!r} do not give positive layer thicknesses")
    return thickness


#: Veros' own ``global_1deg`` asset: grid, bathymetry and the Levitus
#: climatology. The URL and checksum are copied from
#: ``veros/setups/global_1deg/assets.json``.
CLIMATOLOGY_ASSET = {
    "forcing": {
        "url": "https://sid.erda.dk/share_redirect/gsdZADr8to/global_1deg/forcing_1deg_global.nc",
        "md5": "1fc86f88acd820da078c8da5873cfa01",
    }
}


def climatology_file() -> str:
    """Return the local path of the ``global_1deg`` climatology asset.

    Downloads it on first use into Veros' asset cache (``VEROS_ASSET_DIR``,
    default ``~/.veros/assets``), verifying its checksum.
    """
    import json
    import tempfile

    import veros.tools

    with tempfile.TemporaryDirectory() as tmp:
        asset_json = os.path.join(tmp, "assets.json")
        with open(asset_json, "w") as f:
            json.dump(CLIMATOLOGY_ASSET, f)
        return str(veros.tools.get_assets("global_1deg", asset_json)["forcing"])


def interpolate_columns(
    field: np.ndarray, source_depth: np.ndarray, target_depth: np.ndarray
) -> np.ndarray:
    """Interpolate ``(nx, ny, nz_source)`` columns linearly in depth.

    Levitus marks a dry cell with an exact ``0``. Below a column's deepest wet
    source level the deepest wet value is carried down, so a target layer that
    this case's coarser bathymetry makes wet but Levitus' does not still gets
    a temperature from its own column rather than a zero. Columns with no wet
    source level at all stay zero (they are land in both).
    """
    nx, ny, _ = field.shape
    wet = field != 0.0
    out = np.zeros((nx, ny, target_depth.size))
    n_wet = wet.sum(axis=-1)
    for i, j in zip(*np.nonzero(n_wet)):
        column = field[i, j, wet[i, j]]
        depth = source_depth[wet[i, j]]
        # `np.interp` holds the end values outside the range: the shallowest
        # value above the first level, the deepest one below the last.
        out[i, j] = np.interp(target_depth, depth, column)
    return out


def blend_observed_sst(
    temperature: np.ndarray,
    sea_surface_temperature: np.ndarray,
    depth: np.ndarray,
    mixed_layer_depth: float,
) -> np.ndarray:
    """Replace the climatological mixed layer with an observed SST.

    The SST anomaly ``sst - temperature[..., 0]`` is added in full down to
    ``mixed_layer_depth`` and tapered linearly to zero at twice that depth, so
    the observed SST is exactly the top layer's temperature and the interior
    below stays climatological. A mixed layer is well mixed by definition, so
    an anomaly measured at the surface is the anomaly of the whole layer; the
    taper avoids a temperature step at its base. A single depth is a
    simplification -- the observed mixed layer deepens poleward and in winter
    -- and is a parameter of the case for that reason.

    Parameters
    ----------
    temperature : numpy.ndarray
        ``(nx, ny, nz)`` climatological temperature, surface layer first.
    sea_surface_temperature : numpy.ndarray
        ``(nx, ny)`` observed SST in the same units; NaN where there is no
        observation (the climatology is kept there).
    depth : numpy.ndarray
        ``(nz,)`` layer-centre depths, metres, positive down.
    mixed_layer_depth : float
        Depth (m) down to which the anomaly is applied in full.

    Returns
    -------
    numpy.ndarray
        ``(nx, ny, nz)``.

    """
    anomaly = np.nan_to_num(sea_surface_temperature - temperature[..., 0])
    weight = np.clip(2.0 - depth / mixed_layer_depth, 0.0, 1.0)
    blended: np.ndarray = temperature + anomaly[..., None] * weight
    return blended


def global_latlon_setup(
    *,
    climatology: str | None = None,
    sea_surface_temperature: np.ndarray | None = None,
    mixed_layer_depth: float = 50.0,
    layer_centres: Sequence[float] = GLOBAL_LATLON_LAYER_CENTRES,
    dt_mom: float = 1800.0,
    dt_tracer: float = 1800.0,
) -> type[VerosSetup]:
    """Return the Veros setup for a global lat-lon ocean started from observations.

    The grid is the climatology file's own: one degree, 79.5S-79.5N, for the
    default asset.

    Parameters
    ----------
    climatology : str, optional
        Path of Veros' ``global_1deg`` asset (``forcing_1deg_global.nc``).
        Default: :func:`climatology_file`, which downloads it once.
    sea_surface_temperature : numpy.ndarray, optional
        An observed SST in degrees Celsius on this case's grid -- the
        climatology file's own ``(xt, yt)`` cell centres, longitude first:
        ``(360, 160)`` from 90.5E and 79.5S for Veros' asset -- typically an
        analysis for the start date.
        NaN where unobserved. Blended into the mixed layer by
        :func:`blend_observed_sst`. Default ``None``: the Levitus annual mean.
    mixed_layer_depth : float
        See :func:`blend_observed_sst`. Default ``50.0`` m.
    layer_centres : Sequence[float]
        T-point depths in metres, surface first; the thicknesses follow from
        :func:`layer_thicknesses_from_centres`. Default
        :data:`GLOBAL_LATLON_LAYER_CENTRES`.
    dt_mom, dt_tracer : float
        Momentum and tracer timesteps, seconds. Default ``1800.0``, Veros'
        own ``global_1deg`` value for this grid.

    Returns
    -------
    type[veros.VerosSetup]

    """
    climatology = climatology or climatology_file()
    with xr.open_dataset(climatology) as ds:
        # The asset stores (z, y, x); this module works in (x, y, z) with the
        # surface first, and the setup flips to Veros' deepest-first order
        # at assignment.
        source_depth = -ds["zt"].to_numpy().astype(float)
        source_temperature = ds["temperature"].to_numpy().transpose(2, 1, 0)
        source_salinity = ds["salinity"].to_numpy().transpose(2, 1, 0)
        bathymetry = -ds["bathymetry"].to_numpy().T.astype(float)  # m, positive down
        longitude = ds["xt"].to_numpy().astype(float)
        latitude = ds["yt"].to_numpy().astype(float)

    nx, ny = longitude.size, latitude.size
    # Veros' grid is uniform in each direction (`dxt`/`dyt` here are single
    # values), and the zonal axis has to close on itself for the cyclic
    # boundary.
    lon_steps, lat_steps = np.diff(longitude), np.diff(latitude)
    if not (np.allclose(lon_steps, lon_steps[0]) and np.allclose(lat_steps, lat_steps[0])):
        raise ValueError(f"{climatology} is not on a uniform lat-lon grid")
    dlon, dlat = float(lon_steps[0]), float(lat_steps[0])
    if not np.isclose(nx * dlon, 360.0):
        raise ValueError(f"{climatology}'s {nx} longitudes do not span the globe")
    depth = np.asarray(layer_centres, dtype=float)
    thickness = layer_thicknesses_from_centres(depth)
    nz = depth.size

    # A column is ocean where both the bathymetry and the Levitus surface
    # salinity say so (the same test as Veros' own global_1deg setup), and it
    # holds every layer whose centre lies above the sea floor -- at least one.
    is_ocean = (bathymetry > 0) & (source_salinity[..., 0] != 0)
    n_wet = np.where(
        is_ocean,
        np.maximum((depth[None, None, :] < bathymetry[..., None]).sum(-1), 1),
        0,
    )
    # Veros' `kbot` counts from the bottom (1 = the deepest layer is wet,
    # 0 = land), the opposite way round to `n_wet`.
    kbot = np.where(n_wet > 0, nz - n_wet + 1, 0)

    temperature = interpolate_columns(source_temperature, source_depth, depth)
    salinity = interpolate_columns(source_salinity, source_depth, depth)
    if sea_surface_temperature is not None:
        sst = np.asarray(sea_surface_temperature, dtype=float)
        if sst.shape != (nx, ny):
            raise ValueError(
                f"sea_surface_temperature has shape {sst.shape}; this grid is"
                f" {(nx, ny)} (longitude first, from {longitude[0]}E; latitude"
                f" from {latitude[0]}N)."
            )
        temperature = blend_observed_sst(temperature, sst, depth, mixed_layer_depth)

    # Imported here, after every input has been read and checked, rather than
    # at module scope: importing `veros.core.operators` initialises Veros' JAX
    # backend, which switches `jax_enable_x64` on for the whole process. The
    # column helpers above need none of Veros, and neither does refusing a
    # bad input, so neither must do that.
    from veros.core.operators import at, update

    class GlobalLatLonSetup(VerosSetup):
        """A global lat-lon ocean with Levitus bathymetry and initial state."""

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "output_veros"
            settings.description = "Global lat-lon ocean started from observations"

            settings.nx, settings.ny, settings.nz = nx, ny, nz
            settings.dt_mom = dt_mom
            settings.dt_tracer = dt_tracer
            settings.runlen = 86400 * 365

            # Veros puts its first interior T point half a cell *before* the
            # origin (`u_centered_grid`), so the origin is the asset's first
            # centre plus half its spacing -- the same 91.0/-79.0 Veros' own
            # global_1deg setup uses for its 1-degree file.
            settings.x_origin = float(longitude[0] + dlon / 2)
            settings.y_origin = float(latitude[0] + dlat / 2)
            settings.coord_degree = True
            settings.enable_cyclic_x = True

            # The physics is `.earth`'s, which is the configuration the
            # coupled Veros runs (and their gradients) are validated with;
            # only the grid, the bathymetry and the initial state differ.
            settings.enable_streamfunction = False
            # The fork's NaN scanner is a debugging aid: slow, and it prints a
            # line per integer variable every time a step is traced.
            settings.enable_nan_checks = False

            settings.enable_neutral_diffusion = True
            settings.K_iso_0 = 1000.0
            settings.K_iso_steep = 500.0
            settings.iso_dslope = 0.005
            settings.iso_slopec = 0.01
            settings.enable_skew_diffusion = True

            settings.enable_hor_friction = True
            # `.earth`'s biharmonic-scaled Laplacian viscosity rule, applied
            # to this grid's spacing (2.7e4 m2 s-1 at one degree).
            settings.A_h = ((dlon + dlat) / 2 * settings.degtom) ** 3 * 2e-11
            settings.enable_hor_friction_cos_scaling = True
            settings.hor_friction_cosPower = 1

            settings.enable_bottom_friction = True
            settings.r_bot = 1e-5
            settings.enable_implicit_vert_friction = True

            settings.enable_tke = True
            settings.c_k = 0.1
            settings.c_eps = 0.7
            settings.alpha_tke = 30.0
            settings.mxl_min = 1e-8
            settings.tke_mxl_choice = 2
            settings.kappaM_min = 2e-4
            settings.kappaH_min = 2e-5
            settings.enable_kappaH_profile = True

            settings.K_gm_0 = 1000.0
            settings.enable_eke = True
            settings.eke_k_max = 1e4
            settings.eke_c_k = 0.4
            settings.eke_c_eps = 0.5
            settings.eke_cross = 2.0
            settings.eke_crhin = 1.0
            settings.eke_lmin = 100.0
            settings.enable_eke_superbee_advection = True
            settings.enable_eke_isopycnal_diffusion = True

            settings.enable_idemix = False
            settings.eq_of_state_type = 1

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            vs.dxt = update(vs.dxt, at[...], dlon)
            vs.dyt = update(vs.dyt, at[...], dlat)
            vs.dzt = update(vs.dzt, at[...], npx.asarray(thickness[::-1]))

        @veros_routine
        def set_coriolis(self, state):
            vs = state.variables
            settings = state.settings
            vs.coriolis_t = update(
                vs.coriolis_t, at[...],
                2 * settings.omega * npx.sin(vs.yt[None, :] / 180.0 * settings.pi),
            )

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            vs.kbot = update(vs.kbot, at[2:-2, 2:-2], npx.asarray(kbot))

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            mask = vs.maskT[2:-2, 2:-2, :]
            initial_temperature = npx.asarray(temperature[..., ::-1]) * mask
            initial_salinity = npx.asarray(salinity[..., ::-1]) * mask
            for level in range(3):
                vs.temp = update(vs.temp, at[2:-2, 2:-2, :, level], initial_temperature)
                vs.salt = update(vs.salt, at[2:-2, 2:-2, :, level], initial_salinity)

        @veros_routine
        def set_forcing(self, state):
            # A coupled run's surface forcing comes from the coupler's
            # exchanger; VerosComponent replaces this with a no-op.
            pass

        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

        @veros_routine
        def after_timestep(self, state):
            pass

    return GlobalLatLonSetup
