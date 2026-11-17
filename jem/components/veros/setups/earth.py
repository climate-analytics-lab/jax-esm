"""The earth Veros case: a rotated-pole global ocean read from a SCRIP grid.

Unlike :mod:`.double_drake`, whose ocean sits on the atmosphere's own grid,
this case's ocean grid is the packaged ``RotatedGaussianLatLon`` SCRIP grid --
a Gaussian lat-lon grid whose pole has been rigidly rotated onto land, read
here through :class:`GridInfo`, which reconstructs Veros' own grid-spacing
arrays from it exactly (see its docstring for why that reconstruction needs
care) and carries the true (post-rotation) latitude Coriolis needs.

The ocean is idealised: there is no sea ice. Its wet columns are the
land-sea mask's, and their depth is either a bathymetry file's
(``bathymetry_file``, made by :mod:`jem.tools.ocean_bathymetry`) or, with none
given, the same flat bottom everywhere. It starts either from an observed
climatology (``initial_state_file``, made by
:mod:`jem.tools.ocean_initial_state`) or, with none given, from a horizontally
uniform state (``cold_start_temperature_celsius``).
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

import numpy as np
import jax.numpy as npx
import xarray as xr
from veros import VerosSetup, veros_routine
from veros.core.operators import at, update
from veros.core.utilities import enforce_boundaries

from jem.components.veros.setups._grid import axis_spacing_and_origin
from jem.components.veros.setups._layers import LAYER_THICKNESSES
from jem.tools.ocean_bathymetry import read_depth, read_wet_mask
from jem.tools.ocean_initial_state import layer_centres, read_initial_state

#: The fewest wet layers a column may have. A one-layer column has no vertical
#: structure for the mixing schemes to act on (Veros' own ``global_4deg``
#: setup makes such a column land); this setup does not change which columns
#: are wet, so it gives them two layers instead.
MINIMUM_WET_LAYERS = 2


def kbot_from_depth(
    depth: np.ndarray, wet: np.ndarray, layer_thicknesses: Sequence[float]
) -> np.ndarray:
    """Veros' ``kbot`` for each column: how many layers from the bottom are wet.

    ``kbot`` counts from the bottom (``1`` = every layer is wet, ``nz`` = only
    the surface layer, ``0`` = land). A layer is wet if its centre lies above
    the sea floor, as in :mod:`.global_latlon`, and a wet column has at least
    :data:`MINIMUM_WET_LAYERS` of them.

    Parameters
    ----------
    depth : numpy.ndarray
        Sea-floor depth in metres, positive down, ``(nx, ny)``. Only wet
        columns are read.
    wet : numpy.ndarray
        ``(nx, ny)``, true where the column is ocean.
    layer_thicknesses : Sequence[float]
        Layer thicknesses in metres, surface first.

    Returns
    -------
    numpy.ndarray
        Integer ``kbot``, ``(nx, ny)``.

    """
    thickness = np.asarray(layer_thicknesses, dtype=float)
    nz = thickness.size
    centres = np.cumsum(thickness) - thickness / 2
    if np.isnan(depth[wet]).any():
        raise ValueError("the bathymetry has no depth for some wet columns")
    n_wet = np.maximum((centres[None, None, :] < np.nan_to_num(depth)[..., None]).sum(-1),
                       min(MINIMUM_WET_LAYERS, nz))
    return np.where(wet, nz - n_wet + 1, 0).astype(int)


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
    true_lat_xy, true_lon_xy : numpy.ndarray
        Each cell's true (post-rotation) geographic latitude and longitude in
        degrees, ``(nx, ny)``: the latitude for the Coriolis parameter, and
        both for labelling the output, whose own axes are the rotated ones.
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

        # Veros reconstructs vs.yt/vs.xt from dyt/dxt and the origins, exactly
        # only when the spacing is the central difference of the target
        # centres -- not the cell width the SCRIP bounds imply, which is wrong
        # by up to ~1.5 degrees at the Gaussian axis' two polar rows. See
        # `jem.components.veros.setups._grid`, which builds the spacing and
        # origin that make `vs.xt`/`vs.yt` come out identical to the native
        # centres.
        dyt, y_origin = axis_spacing_and_origin(native_lat, cyclic=False)  # (ny,)
        dxt_axis, x_origin = axis_spacing_and_origin(native_lon, cyclic=True)
        dxt = float(dxt_axis[0])  # uniform axis, so one scalar

        # True (post-rotation) geographic latitude of each (j, i) cell -- used for
        # the Coriolis parameter only, not for grid geometry.
        true_lat = grid_ds["grid_center_lat"].to_numpy().reshape(ny, nx)
        true_lat_xy = true_lat.transpose()  # (j, i) -> (xt, yt)
        true_lon_xy = grid_ds["grid_center_lon"].to_numpy().reshape(ny, nx).transpose()

        # The land-sea mask alone decides which columns are wet, never the
        # bathymetry: it is the ERA5-derived fractional land fraction on the
        # same SCRIP grid (convention: 1 = land), thresholded at
        # `landsea_mask_threshold` into a binary mask. The atmosphere's
        # terrain is derived from this same wet set, so a bathymetry that
        # changed it would leave the two components disagreeing about where
        # the coast is.
        wet = read_wet_mask(self.landsea_mask_file, self.landsea_mask_threshold, ny, nx)
        landsea_mask = wet.astype(int).transpose()  # ocean=1/land=0, (xt, yt); Veros kbot wants 0 = land

        self.nx = nx
        self.ny = ny
        self.dyt = dyt
        self.dxt = dxt
        self.y_origin = y_origin
        self.x_origin = x_origin
        self.true_lat_xy = true_lat_xy
        self.true_lon_xy = true_lon_xy
        self.landsea_mask = landsea_mask


def _read_matching_initial_state(
    initial_state_file: str,
    grid_info: GridInfo,
    kbot: np.ndarray,
    layer_thicknesses: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Read an initial-state file and check it is for this grid, these layers and these wet cells.

    The file's NaN cells must be exactly the cells that ``kbot`` leaves dry
    (dry columns and the layers below a column's floor): a file made for
    another bathymetry would leave wet cells at zero temperature and
    salinity, or put values under the sea floor, and neither is a state this
    model can start from.

    Returns
    -------
    tuple of numpy.ndarray
        Temperature and salinity, ``(nx, ny, nz)``, surface layer first, with
        zero (not NaN) on dry cells.

    """
    nz = len(layer_thicknesses)
    temperature, salinity, depth, centres = read_initial_state(
        initial_state_file, nz, grid_info.ny, grid_info.nx
    )
    regenerate = (
        f"regenerate it for this setup with jem.tools.ocean_initial_state.ocean_initial_state, passing the"
        f" same bathymetry file as `bathymetry_file` ({initial_state_file!r} cannot be used with"
        " a different bathymetry, land-sea mask or layers)"
    )
    if not np.allclose(depth, layer_centres(layer_thicknesses)):
        raise ValueError(f"{initial_state_file}: its layer centres {depth} are not this setup's; {regenerate}")
    if not (
        np.allclose(centres[0], grid_info.true_lat_xy.T) and np.allclose(centres[1], grid_info.true_lon_xy.T)
    ):
        raise ValueError(f"{initial_state_file}: its cell centres are not this setup's grid; {regenerate}")
    n_wet = nz - kbot + 1  # (nx, ny) layers per column, from the surface
    expected_wet = (kbot[..., None] > 0) & (np.arange(nz) < n_wet[..., None])
    has_value = np.isfinite(temperature) & np.isfinite(salinity)
    if not np.array_equal(has_value, expected_wet):
        raise ValueError(
            f"{initial_state_file}: {int((has_value & ~expected_wet).sum())} cells have values where this"
            f" setup is dry and {int((expected_wet & ~has_value).sum())} wet cells have none; {regenerate}"
        )
    return np.where(expected_wet, temperature, 0.0), np.where(expected_wet, salinity, 0.0)


def earth_setup(
    *,
    scrip_grid_file: str,
    landsea_mask_file: str,
    landsea_mask_threshold: float = 0.5,
    layer_thicknesses: Sequence[float] = LAYER_THICKNESSES,
    dt_mom: float = 3600.0,
    dt_tracer: float = 3600.0,
    cold_start_temperature_celsius: float = 15.0,
    bathymetry_file: str | None = None,
    initial_state_file: str | None = None,
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
        this configuration has been run with; it is not otherwise tuned.
    cold_start_temperature_celsius : float
        The uniform start, used only when ``initial_state_file`` is ``None``:
        the initial ocean temperature just below the surface, in degrees
        Celsius (``vs.temp``'s units). Every wet column has the same profile,
        poles included, and it falls linearly with depth, ``T0 * (1 - z /
        zw[0])``, reaching zero at ``|zw[0]|``, the depth of the top face of
        the deepest layer (4510 m for the default layers). The deepest layer
        lies below that depth, so it starts slightly below zero (-1.15 degC
        for the default layers), not at zero. Salinity is 35 everywhere.
        Default ``15.0``.
    bathymetry_file : str, optional
        A depth file on this grid, as written by
        :func:`jem.tools.ocean_bathymetry.ocean_bathymetry`. Each wet
        column's number of layers is those whose centre lies above its depth
        (at least two; see :func:`kbot_from_depth`); the wet columns
        themselves are still the land-sea mask's. Default ``None``: every wet
        column has the flat bottom of all the layers (5200 m for the default
        layers). Everything else in the setup (the initial profile, the
        masks) follows from ``kbot`` and holds for either.
    initial_state_file : str, optional
        A temperature and salinity file on this grid, as written by
        :func:`jem.tools.ocean_initial_state.ocean_initial_state`:
        Conservative Temperature (degrees C, ``vs.temp``'s variable) and
        Absolute Salinity (g/kg, ``vs.salt``'s) for every wet cell, surface
        layer first, which become ``temp`` and ``salt`` on every time level
        Veros reads at its first step. The file is valid only with the
        bathymetry (and so the wet cells) it was made for; one that does not
        match this setup's grid, layers or wet cells is refused here, before
        Veros is built, with a message saying how to regenerate it. Default
        ``None``: the uniform start of ``cold_start_temperature_celsius``,
        exactly.

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

    # Host-side numpy, once: `kbot` is a static input to the setup, not a
    # traced value.
    if bathymetry_file is None:
        kbot = grid_info.landsea_mask
    else:
        depth = read_depth(bathymetry_file, grid_info.ny, grid_info.nx)
        kbot = kbot_from_depth(depth, grid_info.landsea_mask > 0, layer_thicknesses)

    initial_temperature = initial_salinity = None
    if initial_state_file is not None:
        initial_temperature, initial_salinity = _read_matching_initial_state(
            initial_state_file, grid_info, kbot, layer_thicknesses
        )

    class EarthSetup(VerosSetup):
        """A rotated-pole global ocean on a Gaussian lat-lon grid.

        This setup demonstrates:
         - reconstructing Veros' own grid-spacing arrays from a SCRIP grid
           whose native axis is non-uniform (Gaussian)
         - the true (post-rotation) latitude a rotated grid's Coriolis
           parameter needs
        """

        # Veros' own `xt`/`yt` are this grid's native, pre-rotation axes: up
        # to ~12 degrees from where a cell really is. These two say where it
        # is, and `VerosComponent` labels the output with them.
        geographic_longitude = grid_info.true_lon_xy
        geographic_latitude = grid_info.true_lat_xy

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "output_veros"
            settings.description = "Rotated-pole global ocean"

            settings.enable_streamfunction = False  # then it solves linear free surface
            # Off, as in the other setups: the coupled run has its own health gate
            # (`jem.driver.run_chunked`), and Veros' scanner adds reductions at
            # every instrumented point of the step and prints to stdout.
            settings.enable_nan_checks = False

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
            # The true latitude is known for the interior only, so the zonal
            # halo is filled from the other side of the periodic seam: Veros
            # never halo-exchanges `coriolis_t` itself, and its Coriolis
            # tendency for `u` averages `coriolis_t` with its eastern
            # neighbour, which for the last interior column is a ghost cell.
            vs.coriolis_t = enforce_boundaries(
                update(
                    vs.coriolis_t, at[2:-2, 2:-2],
                    2 * settings.omega * npx.sin(grid_info.true_lat_xy / 180.0 * settings.pi),
                ),
                settings.enable_cyclic_x,
            )

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
            vs.kbot = npx.zeros_like(x)
            vs.kbot = update(vs.kbot, at[2:-2, 2:-2], kbot)

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            if initial_temperature is None:
                # Horizontally uniform, linear in depth; see the
                # `cold_start_temperature_celsius` docstring for where it
                # reaches zero. Veros' `zt`/`zw` are negative below the surface.
                vs.temp = update(
                    vs.temp, at[...],
                    ((1 - vs.zt[None, None, :] / vs.zw[0])
                     * cold_start_temperature_celsius * vs.maskT)[..., None],
                )
                vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])
            else:
                # The file is surface-first and Veros' layers are bottom-first.
                # Veros reads all three time levels at its first step
                # (`taum1`, `tau`, `taup1`), so each gets the same field; the
                # halo is left zero and is filled from the interior (the
                # periodic seam) by `calc_initial_conditions`.
                mask = vs.maskT[2:-2, 2:-2, :]
                for level in range(3):
                    vs.temp = update(
                        vs.temp, at[2:-2, 2:-2, :, level], npx.asarray(initial_temperature[..., ::-1]) * mask
                    )
                    vs.salt = update(
                        vs.salt, at[2:-2, 2:-2, :, level], npx.asarray(initial_salinity[..., ::-1]) * mask
                    )

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

    return EarthSetup
