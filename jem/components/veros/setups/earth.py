"""The earth Veros case: a rotated-pole global ocean read from a SCRIP grid.

Unlike :mod:`.double_drake`'s uniform lat-lon grid, this case's ocean grid is
the packaged ``RotatedGaussianLatLon`` SCRIP grid -- a Gaussian lat-lon grid
whose pole has been rigidly rotated onto land, read here through
:class:`GridInfo`, which reconstructs Veros' own grid-spacing arrays from it
exactly (see its docstring for why that reconstruction needs care) and
carries the true (post-rotation) latitude Coriolis needs.
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


class GridInfo:
    """Read a SCRIP grid's geometry into what Veros' own grid needs.

    Attributes
    ----------
    nx, ny : int
        Grid shape, from the SCRIP file's own ``grid_dims``.
    dxt : float
        Uniform zonal spacing, degrees.
    dyt : numpy.ndarray
        Per-row meridional spacing, degrees, shape ``(ny,)`` -- non-uniform,
        since the native axis is Gaussian.
    x_origin, y_origin : float
        What Veros' own grid-spacing reconstruction needs to reproduce the
        native axis exactly; see :meth:`get_grid_info`.
    true_lat_xy : numpy.ndarray
        Each cell's true (post-rotation) geographic latitude, ``(nx, ny)``,
        for the Coriolis parameter.
    landsea_mask : numpy.ndarray
        ``(nx, ny)``, ``1`` = ocean / ``0`` = land (Veros' ``kbot``
        convention), from the matching land-sea mask file thresholded at
        ``landsea_mask_threshold``.

    """

    scrip_grid_file: str
    landsea_mask_file: str
    landsea_mask_threshold: float

    def __init__(self, scrip_grid_file: str, landsea_mask_file: str, landsea_mask_threshold: float):
        """Record the grid/mask files and read the grid metadata immediately.

        `get_grid_info` is called here rather than lazily so a bad path fails
        at setup time, not part-way into a run.
        """
        self.scrip_grid_file = scrip_grid_file
        self.landsea_mask_threshold = landsea_mask_threshold
        self.landsea_mask_file = landsea_mask_file
        self.get_grid_info()

    def get_grid_info(self):
        """Read the grid and mask files and populate this object's attributes."""
        # `scrip_grid_file` is a SCRIP grid file (e.g. RotatedGaussianLatLon.SCRIP.nc)
        # whose pole may be rigidly rotated away from Earth's true pole. It carries
        # both the grid's native (pre-rotation) Gaussian lat-lon axis --
        # `native_lat`/`native_lon` centres and `native_lat_bounds`/`native_lon_bounds`
        # cell faces -- and each cell's true (post-rotation) geographic location
        # (`grid_center_lat`/`grid_center_lon`). The native axis sizes and shapes
        # the Veros grid itself (a rigid rotation preserves the sphere exactly, so
        # the grid's own metric terms -- dxt/dyt, cost, cosu, tantr, area -- are
        # correct when computed in the native frame). The Coriolis parameter is
        # different: it depends on position relative to Earth's actual spin axis,
        # which is fixed in space regardless of the coordinate mesh chosen to
        # discretize the domain, so it must use the true (post-rotation) latitude
        # instead -- see `set_coriolis` below.
        grid_ds = xr.open_dataset(self.scrip_grid_file)
        nlon, nlat = grid_ds["grid_dims"].to_numpy()
        nx, ny = int(nlon), int(nlat)

        native_lat = grid_ds["native_lat"].to_numpy()  # (ny,) centres, non-uniform (Gaussian)
        native_lon = grid_ds["native_lon"].to_numpy()  # (nx,) centres, uniform

        # Veros reconstructs cell-centre vs.yt/vs.xt from vs.dyt/vs.dxt via a
        # leapfrog-style recursion (veros.core.numerics.u_centered_grid) that is
        # only exact when spacing[j] equals the *central* difference of the
        # target centres, (centre[j+1]-centre[j-1])/2 -- not the cell width
        # implied by native_lat_bounds/native_lon_bounds. For the (uniform)
        # longitude axis those coincide, but for the (non-uniform, Gaussian)
        # latitude axis the pole-clamped outermost bounds break that relation,
        # which -- left uncorrected -- introduces up to ~1.5 degrees of error at
        # the two polar-cap rows (verified numerically), enough to meaningfully
        # bias cos/tan there. Build spacing from centre differences instead, and
        # solve for x_origin/y_origin by exactly replicating Veros's own
        # reconstruction (host-side, degrees) so vs.yt/vs.xt come out identical
        # to native_lat/native_lon.
        def _spacing_from_centers(centers):
            d = np.empty_like(centers)
            d[1:-1] = (centers[2:] - centers[:-2]) / 2.0
            d[0] = centers[1] - centers[0]
            d[-1] = centers[-1] - centers[-2]
            return d

        def _calibrate_origin(spacing, first_center, cyclic):
            """Solve for the origin Veros needs so that its own u_centered_grid
            reconstruction from `spacing` places the first interior centre
            exactly at `first_center`. Mirrors
            veros.core.numerics.calc_grid_spacings_kernel's ghost-cell fill and
            u_centered_grid exactly.
            """
            n = spacing.size
            padded = np.zeros(n + 4)
            padded[2:-2] = spacing
            if cyclic:
                padded[-2:] = padded[2:4]
                padded[:2] = padded[-4:-2]
            else:
                padded[-2:] = padded[-3]
                padded[:2] = padded[2]
            yu = np.zeros(n + 4)
            yu[1:] = np.cumsum(padded[1:])
            yt = np.zeros(n + 4)
            yt[0] = yu[0] - padded[0] * 0.5
            yt[1:] = 2 * yu[:-1]
            alt = np.ones(n + 4)
            alt[::2] = -1
            yt = alt * np.cumsum(alt * yt)
            return first_center - yt[2] + yu[2]

        dyt = _spacing_from_centers(native_lat)                              # (ny,)
        dxt = float(native_lon[1] - native_lon[0])                           # scalar, uniform
        y_origin: float = float(_calibrate_origin(dyt, native_lat[0], cyclic=False))
        x_origin: float = float(_calibrate_origin(np.full(nx, dxt), native_lon[0], cyclic=True))

        # True (post-rotation) geographic latitude of each (j, i) cell -- used for
        # the Coriolis parameter only, not for grid geometry.
        true_lat = grid_ds["grid_center_lat"].to_numpy().reshape(ny, nx)
        true_lat_xy = true_lat.transpose()  # (j, i) -> (xt, yt)

        # TODO: real bathymetry is still needed here -- this only gives a
        # binary land/sea mask, so every ocean column gets the same flat
        # bottom depth (kbot) rather than actual varying ocean depth.
        # ERA5-derived fractional land-sea mask on the same SCRIP grid; convention: 1 = land.
        mask_ds = xr.open_dataset(self.landsea_mask_file)
        lsm = mask_ds["lsm"].to_numpy().reshape(ny, nx)
        is_land = lsm >= self.landsea_mask_threshold
        landsea_mask = (1 - is_land.astype(int)).transpose()  # -> ocean=1/land=0, (xt, yt); Veros kbot wants 0 = land

        self.nx = nx
        self.ny = ny
        self.dyt = dyt
        self.dxt = dxt
        self.y_origin = y_origin
        self.x_origin = x_origin
        self.true_lat_xy = true_lat_xy
        self.landsea_mask = landsea_mask


def earth_setup(
    *,
    scrip_grid_file: str,
    landsea_mask_file: str,
    landsea_mask_threshold: float = 0.5,
    layer_thicknesses: Sequence[float] = LAYER_THICKNESSES,
    dt_mom: float = 3600.0,
    dt_tracer: float = 3600.0,
    cold_start_temperature_celsius: float = 15.0,
) -> type[VerosSetup]:
    """Return the Veros setup for a rotated-pole global ocean read from a SCRIP grid.

    Parameters
    ----------
    scrip_grid_file : str
        A SCRIP grid file carrying the native (pre-rotation) Gaussian
        lat-lon axis and each cell's true (post-rotation) latitude -- the
        packaged ``RotatedGaussianLatLon.SCRIP.nc``. See :class:`GridInfo`.
    landsea_mask_file : str
        An ERA5-derived fractional land-sea mask on the same SCRIP grid.
    landsea_mask_threshold : float
        Fraction at or above which a cell counts as land. Default ``0.5``.
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

    """
    grid_info = GridInfo(
        scrip_grid_file=scrip_grid_file,
        landsea_mask_file=landsea_mask_file,
        landsea_mask_threshold=landsea_mask_threshold,
    )

    ddz = npx.array(layer_thicknesses)
    nz = len(ddz)

    class EarthSetup(VerosSetup):
        """A rotated-pole global ocean on a Gaussian lat-lon grid.

        This setup demonstrates:
         - reconstructing Veros' own grid-spacing arrays from a SCRIP grid
           whose native axis is non-uniform (Gaussian)
         - the true (post-rotation) latitude a rotated grid's Coriolis
           parameter needs
        """

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "output_veros"
            settings.description = "Rotated-pole global ocean"

            settings.enable_streamfunction = False  # then it solves linear free surface
            settings.enable_nan_checks = True

            settings.nx, settings.ny, settings.nz = grid_info.nx, grid_info.ny, nz
            settings.dt_mom = dt_mom
            settings.dt_tracer = dt_tracer
            settings.runlen = 86400 * 365

            settings.x_origin = grid_info.x_origin
            settings.y_origin = grid_info.y_origin

            settings.coord_degree = True
            settings.enable_cyclic_x = True

            settings.enable_neutral_diffusion = True
            settings.K_iso_0 = 1000.0
            settings.K_iso_steep = 500.0
            settings.iso_dslope = 0.005
            settings.iso_slopec = 0.01
            settings.enable_skew_diffusion = True

            settings.enable_hor_friction = True
            settings.A_h = ((grid_info.dxt + grid_info.dyt.mean()) / 2 * settings.degtom) ** 3 * 2e-11
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
            # Ghost cells ([:2]/[-2:]) are filled automatically from the
            # interior by calc_grid_spacings_kernel -- only the interior
            # needs to be set here (see e.g. global_flexible's set_grid).
            vs.dxt = update(vs.dxt, at[2:-2], grid_info.dxt)
            vs.dyt = update(vs.dyt, at[2:-2], grid_info.dyt)  # per-row array: native Gaussian latitude spacing is non-uniform
            vs.dzt = update(vs.dzt, at[...], ddz[::-1])  # ocean grid starts from below

        @veros_routine
        def set_coriolis(self, state):
            vs = state.variables
            settings = state.settings
            # Coriolis depends on position relative to Earth's true spin axis,
            # not the grid's own (rotated) latitude -- see note in GridInfo above.
            vs.coriolis_t = update(
                vs.coriolis_t, at[2:-2, 2:-2],
                2 * settings.omega * npx.sin(grid_info.true_lat_xy / 180.0 * settings.pi),
            )

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
            vs.kbot = npx.zeros_like(x)
            vs.kbot = update(vs.kbot, at[2:-2, 2:-2], grid_info.landsea_mask)

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

    return EarthSetup
