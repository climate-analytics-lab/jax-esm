"""The double-drake Veros case: an idealised two-continent ocean.

A partially-closed spherical domain on **the atmosphere's own grid**: two
meridional land dividers running from the tropics to a polar ice cap carve
the world ocean into two basins. The ocean's cell centres are the
atmosphere's -- a uniform longitude axis and its Gaussian latitude axis, read
from the atmosphere's SCRIP grid file -- because the two are exchanged
index-to-index with no regridding, so a flux must land on the ocean row at the
same latitude, and the ocean's metric terms (``cos(lat)``, areas) must be those
of that latitude. Adapted from pyOM2's ``ACC 2`` setup
(https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/ACC%202).

The land-sea mask comes from a JCM-canonical terrain file (dims ``(lon,
lat)``, vars ``lsm``/``orog`` -- see :func:`jem.tools.idealised_terrain`),
which must have the atmosphere grid's shape.
"""

# Importing this before `veros.core.operators` is what points Veros at the
# JAX backend before its operators bind to a default one -- see
# `jem.components.veros_component.configure_veros_runtime`'s docstring.
# `VerosComponent.from_setup` always imports this module first (it is one of
# `veros_component`'s own methods), so a config-driven run never depends on
# this import; a standalone import of this module (a test, `sphinx-build`'s
# autosummary) does, which is why it is repeated here rather than left as an
# assumption about who imports this module.
from jem.components import veros_component  # noqa: F401

from collections.abc import Sequence

import jax.numpy as npx
import numpy as np
import xarray as xr
from veros import VerosSetup, veros_routine
from veros.core.operators import at, update

from jem.components.veros.setups._grid import axis_spacing_and_origin, read_scrip_axes
from jem.components.veros.setups._layers import LAYER_THICKNESSES


def double_drake_setup(
    *,
    atmosphere_grid_file: str,
    land_sea_mask_file: str,
    layer_thicknesses: Sequence[float] = LAYER_THICKNESSES,
    dt_mom: float = 3600.0,
    dt_tracer: float = 3600.0,
    cold_start_temperature_celsius: float = 15.0,
) -> type[VerosSetup]:
    """Return the Veros setup for an idealised two-continent (double-drake) ocean.

    Parameters
    ----------
    atmosphere_grid_file : str
        The SCRIP grid file of the atmosphere this ocean is coupled to (for
        the T31 atmosphere, the packaged ``JCM_T31.SCRIP.nc``). Its
        longitude and latitude cell centres become the ocean's, to round-off,
        and its shape is the ocean's shape; it must be a separable lon/lat
        grid. Pass the same grid the atmosphere runs on: the coupling is
        index-to-index, so no other grid is coupled correctly.
    land_sea_mask_file : str
        A JCM-canonical terrain file (``jem.tools.idealised_terrain``'s
        output) whose ``lsm`` -- a **binary** 0/1 mask, 1 = land, 0 = ocean,
        not a fractional one (unlike :func:`~jem.components.veros.setups.
        earth.earth_setup`, this factory does not threshold) -- masks the
        ocean. It must have the shape of ``atmosphere_grid_file``'s grid.
    layer_thicknesses : Sequence[float]
        Vertical layer thicknesses in metres, surface first. Default
        :data:`jem.components.veros.setups._layers.LAYER_THICKNESSES` (15
        layers, 50 m to 690 m); a shallower ocean is
        ``layer_thicknesses=LAYER_THICKNESSES[:n]``.
    dt_mom, dt_tracer : float
        Momentum and tracer timesteps, seconds. Default ``3600.0``, the value
        this configuration has been run with; it is not otherwise tuned.
    cold_start_temperature_celsius : float
        Initial ocean temperature just below the surface, in degrees Celsius
        (``vs.temp``'s units). The initial state is horizontally uniform --
        every wet column has the same profile, poles included -- and falls
        linearly with depth, ``T0 * (1 - z / zw[0])``, reaching zero at
        ``|zw[0]|``, the depth of the top face of the deepest layer (4510 m
        for the default layers). The deepest layer lies below that depth, so
        it starts slightly below zero (-1.15 degC for the default layers),
        not at zero. Default ``15.0``.

    Returns
    -------
    type[veros.VerosSetup]

    Raises
    ------
    ValueError
        If ``land_sea_mask_file``'s ``lsm`` holds any value other than 0 or
        1 -- a fractional mask would give a fractional ``kbot``, which Veros
        would silently truncate rather than refuse -- or if its shape is not
        the atmosphere grid's.

    """
    lsm = xr.open_dataset(land_sea_mask_file)["lsm"].to_numpy()
    if not np.isin(lsm, (0, 1)).all():
        raise ValueError(
            f"{land_sea_mask_file!r}'s lsm must be a binary 0/1 land-sea "
            "mask (1 = land, 0 = ocean); double_drake_setup does not "
            "threshold a fractional mask the way earth_setup does. Got "
            f"values ranging {float(lsm.min())!r} to {float(lsm.max())!r}."
        )
    # 1 = land, 0 = ocean in the terrain file; Veros' `kbot` wants the
    # opposite (0 = land, so the column is inactive from the bottom up).
    land_sea_mask = (1 - lsm).astype(int)
    longitude, latitude = read_scrip_axes(atmosphere_grid_file)
    nx, ny = longitude.size, latitude.size
    if land_sea_mask.shape != (nx, ny):
        raise ValueError(
            f"{land_sea_mask_file!r}'s lsm has shape {land_sea_mask.shape}, "
            f"but the atmosphere grid {atmosphere_grid_file!r} is {(nx, ny)} "
            "(lon, lat); the ocean is exchanged index-to-index with the "
            "atmosphere, so the two must be the same grid."
        )

    ddz = npx.array(layer_thicknesses)
    nz = len(ddz)

    # The ocean's cell centres are the atmosphere's: the exchange is
    # index-to-index, so ocean row j must be at atmosphere row j's latitude.
    # Veros reconstructs its centres from spacing and origin, which
    # `axis_spacing_and_origin` inverts so that `vs.xt`/`vs.yt` come out
    # identical to these axes (the latitude axis is Gaussian, not uniform).
    dxt_axis, x_origin = axis_spacing_and_origin(longitude, cyclic=True)
    dyt, y_origin = axis_spacing_and_origin(latitude, cyclic=False)
    dxt = float(dxt_axis[0])  # the longitude axis is uniform

    class DoubleDrakeSetup(VerosSetup):
        """A spherical-coordinate ocean with a partially closed domain.

        The land-sea mask carves the world ocean into two basins; the
        surface forcing comes from the coupler. Its horizontal grid is the
        atmosphere's own (see the module docstring), and its initial state is
        horizontally uniform (see ``cold_start_temperature_celsius``).
        """

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "output_veros"
            settings.description = "Double-drake idealised two-continent ocean"

            settings.enable_streamfunction = False  # then it solves linear free surface
            settings.enable_nan_checks = False

            settings.nx, settings.ny, settings.nz = nx, ny, nz
            settings.dt_mom = dt_mom
            settings.dt_tracer = dt_tracer
            settings.runlen = 86400 * 365

            settings.x_origin = x_origin
            settings.y_origin = y_origin

            settings.coord_degree = True
            settings.enable_cyclic_x = True

            settings.enable_neutral_diffusion = True
            settings.K_iso_0 = 1000.0
            settings.K_iso_steep = 500.0
            settings.iso_dslope = 0.005
            settings.iso_slopec = 0.01
            settings.enable_skew_diffusion = True

            settings.enable_hor_friction = True
            settings.A_h = ((dxt + dyt.mean()) / 2 * settings.degtom) ** 3 * 2e-11
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
            # Ghost cells are filled from the interior by Veros'
            # `calc_grid_spacings_kernel`; only the interior is set here.
            vs.dxt = update(vs.dxt, at[2:-2], dxt)
            vs.dyt = update(vs.dyt, at[2:-2], dyt)  # per row: the Gaussian axis is not uniform
            vs.dzt = update(vs.dzt, at[...], ddz[::-1])  # ocean grid starts from below

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
            x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
            vs.kbot = npx.zeros_like(x)
            vs.kbot = update(vs.kbot, at[2:-2, 2:-2], land_sea_mask)

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            # Horizontally uniform, linear in depth; see the
            # `cold_start_temperature_celsius` docstring for where it
            # reaches zero. Veros' `zt`/`zw` are negative below the surface.
            vs.temp = update(
                vs.temp, at[...],
                ((1 - vs.zt[None, None, :] / vs.zw[0])
                 * cold_start_temperature_celsius * vs.maskT)[..., None],
            )
            vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

            # No surface forcing is initialised here: a coupled run's wind
            # stress, heat and freshwater fluxes -- and the TKE surface forcing
            # derived from the stress -- are written by `VerosComponent` before
            # every step, and `set_forcing` below is a no-op, so nothing in
            # this setup reads a wind stress or a restoring temperature.

            if settings.enable_idemix:
                vs.forc_iw_bottom = 1e-6 * vs.maskW[:, :, -1]
                vs.forc_iw_surface = 1e-7 * vs.maskW[:, :, -1]

        @veros_routine
        def set_forcing(self, state):
            # A coupled run's surface forcing comes from the coupler's
            # exchanger; VerosComponent replaces this with a no-op, but the
            # setup keeps its own definition (a no-op already) for a
            # standalone `veros run` of this file.
            pass

        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

        @veros_routine
        def after_timestep(self, state):
            pass

    return DoubleDrakeSetup
