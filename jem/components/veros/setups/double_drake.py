"""The double-drake Veros case: an idealised two-continent ocean.

A partially-closed spherical domain on a **uniform** lat-lon grid: two
meridional land dividers running from the tropics to a polar ice cap carve
the world ocean into two basins, wind and buoyancy forcing drive a
large-scale overturning circulation in each. Adapted from pyOM2's ``ACC 2``
setup (https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/ACC%202).

The land-sea mask comes from a JCM-canonical terrain file (dims ``(lon,
lat)``, vars ``lsm``/``orog`` -- see :func:`jem.tools.idealised_terrain`),
whose shape fixes ``nx``/``ny``: reading it is unavoidable to build the mask,
so two separate numbers that have to agree with it are not asked for as well.
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
from veros.distributed import global_max, global_min
from veros.variables import Variable, allocate

from jem.components.veros.setups._layers import LAYER_THICKNESSES


def double_drake_setup(
    *,
    land_sea_mask_file: str,
    layer_thicknesses: Sequence[float] = LAYER_THICKNESSES,
    dt_mom: float = 3600.0,
    dt_tracer: float = 3600.0,
    cold_start_temperature_celsius: float = 15.0,
) -> type[VerosSetup]:
    """Return the Veros setup for an idealised two-continent (double-drake) ocean.

    Parameters
    ----------
    land_sea_mask_file : str
        A JCM-canonical terrain file (``jem.tools.idealised_terrain``'s
        output) whose ``lsm`` -- a **binary** 0/1 mask, 1 = land, 0 = ocean,
        not a fractional one (unlike :func:`~jem.components.veros.setups.
        earth.earth_setup`, this factory does not threshold) -- both masks
        the ocean and fixes ``nx``/``ny`` from its shape.
    layer_thicknesses : Sequence[float]
        Vertical layer thicknesses in metres, surface first. Default
        :data:`jem.components.veros.setups._layers.LAYER_THICKNESSES` (15
        layers, 50 m to 690 m); a shallower ocean is
        ``layer_thicknesses=LAYER_THICKNESSES[:n]``.
    dt_mom, dt_tracer : float
        Momentum and tracer timesteps, seconds. Default ``3600.0``, the value
        validated by this case's original standalone driver.
    cold_start_temperature_celsius : float
        Initial ocean temperature at the surface, decaying linearly to zero
        at the sea floor (``vs.temp``'s units are already degrees Celsius).
        Default ``15.0``.

    Returns
    -------
    type[veros.VerosSetup]

    Raises
    ------
    ValueError
        If ``land_sea_mask_file``'s ``lsm`` holds any value other than 0 or
        1 -- a fractional mask would give a fractional ``kbot``, which Veros
        would silently truncate rather than refuse.

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
    nx, ny = land_sea_mask.shape

    ddz = npx.array(layer_thicknesses)
    nz = len(ddz)

    # A uniform lat-lon grid: `dyt = 180/ny` gives every row the same
    # meridional spacing, which deliberately approximates JCM's own Gaussian
    # latitudes (denser near the equator, sparser at the poles) with a
    # uniform one -- the double-drake case has always done this (it predates
    # this move into the package), and it is why no atmosphere<->ocean
    # regridding is configured for this case even though the two latitude
    # axes are not, in fact, the same grid. See jax-esm#121 (reported, not
    # fixed here: it needs its own validation campaign against a Gaussian
    # ocean grid, which `earth_setup`'s `GridInfo` already shows how to
    # build).
    dxt = 360.0 / nx
    dyt = 180.0 / ny
    x_origin: float = 0.0
    y_origin: float = -90.0

    class DoubleDrakeSetup(VerosSetup):
        """A model using spherical coordinates with a partially closed domain.

        Wind forcing over the channel part and buoyancy relaxation drive a
        large-scale meridional overturning circulation in each of the two
        basins the land-sea mask carves out.

        This setup demonstrates:
         - setting up an idealized geometry
         - updating surface forcings
         - basic usage of diagnostics
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
            settings.A_h = ((dxt + dyt) / 2 * settings.degtom) ** 3 * 2e-11
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

            var_meta = state.var_meta
            var_meta.update(
                t_star=Variable("t_star", ("yt",), "deg C", "Reference surface temperature"),
                t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
            )

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            vs.dxt = update(vs.dxt, at[...], dxt)
            vs.dyt = update(vs.dyt, at[...], dyt)
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

            vs.temp = update(
                vs.temp, at[...],
                ((1 - vs.zt[None, None, :] / vs.zw[0])
                 * cold_start_temperature_celsius * vs.maskT)[..., None],
            )
            vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

            # wind stress forcing
            yt_min = global_min(vs.yt.min())
            yu_min = global_min(vs.yu.min())
            yt_max = global_max(vs.yt.max())
            yu_max = global_max(vs.yu.max())

            taux = allocate(state.dimensions, ("yt",))
            taux = npx.where(vs.yt < -20, 0.1 * npx.sin(settings.pi * (vs.yu - yu_min) / (-20.0 - yt_min)), taux)
            taux = npx.where(vs.yt > 10, 0.1 * (1 - npx.cos(2 * settings.pi * (vs.yu - 10.0) / (yu_max - 10.0))), taux)
            vs.surface_taux = taux * vs.maskU[:, :, -1]

            # surface heatflux forcing
            vs.t_star = allocate(state.dimensions, ("yt",), fill=15)
            vs.t_star = npx.where(vs.yt < -20, 15 * (vs.yt - yt_min) / (-20 - yt_min), vs.t_star)
            vs.t_star = npx.where(vs.yt > 20, 15 * (1 - (vs.yt - 20) / (yt_max - 20)), vs.t_star)
            vs.t_rest = vs.dzt[npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

            if settings.enable_tke:
                vs.forc_tke_surface = update(
                    vs.forc_tke_surface,
                    at[2:-2, 2:-2],
                    npx.sqrt(
                        (0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0) ** 2
                        + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0) ** 2
                    )
                    ** (1.5),
                )

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
