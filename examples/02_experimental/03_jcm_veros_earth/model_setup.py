# Construction of the coupled JCM + Veros + SlabOceanModel system used by
# the double-drake experiment, packaged as a function so that the forward
# run (main.py) and other scripts (e.g. a jax.grad sensitivity experiment)
# build exactly the same model.

import glob
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jem.tool_scripts.generate_jcm_forcing_and_topography_files import generate_jcm_forcing_and_topography_files

from jem.components import JCM, Veros, SlabOceanModel
from jem.components.slab.grid import generate_slab_grid
from jem.base.coupler import Coupler
from jem.utils.esmf_regrid import ESMFRegridder

from modify_jcm_terrain import modify_jcm_terrain
from veros_case_setup import generateVerosSetup

from veros.core.operators import update, at

one_second = jdt.to_timedelta(1, "second")

def is_pytree_all_finite(tree):
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc & jnp.all(jnp.isfinite(x)),
        tree,
        jnp.array(True),
    )

def report_first_nonfinite(name, x, lon=None, lat=None):
    """If `x` has any non-finite value, debug-print the index of its
    first occurrence (and lon/lat, if `x` is a 2D (lon, lat) field).

    `jax.debug.print` cannot format a tuple-of-tracers (e.g. the result
    of `jnp.unravel_index`) directly, so each index component is passed
    as its own scalar kwarg. The format string is built in Python from
    `x.ndim`, which is static under tracing.
    """
    x = jnp.asarray(x)
    finite = jnp.isfinite(x)
    any_bad = jnp.any(~finite)
    flat_idx = jnp.argmin(finite.ravel().astype(jnp.int32))
    idx = jnp.unravel_index(flat_idx, x.shape)
    val = x.ravel()[flat_idx]

    idx_fmt = ", ".join(f"{{i{d}}}" for d in range(x.ndim))
    idx_kwargs = {f"i{d}": idx[d] for d in range(x.ndim)}

    def report(_):
        fmt = name + f": first non-finite at idx=({idx_fmt}), value={{val:.6e}}"
        kwargs = dict(idx_kwargs, val=val)
        if lon is not None and lat is not None and x.ndim == 2:
            fmt += ", lon={lo:.2f}, lat={la:.2f}"
            # `lon`/`lat` are plain numpy arrays (since `jnp.pi` is a
            # Python float, `numpy_array * 180 / jnp.pi` stays numpy).
            # Indexing a numpy array with a JAX tracer raises
            # TracerArrayConversionError, so convert to jnp first.
            kwargs["lo"] = jnp.asarray(lon)[idx[0]]
            kwargs["la"] = jnp.asarray(lat)[idx[1]]
        jax.debug.print(fmt, **kwargs)

    jax.lax.cond(any_bad, report, lambda _: None, None)



def _freeze_season(atm_model, freeze_season_at_day):
    """Pin `atm_model`'s seasonal (day-of-year) forcing to a fixed day, while
    leaving the diurnal (time-of-day) cycle running.

    `jcm.model.Model._date_from_sim_time` turns elapsed `sim_time` into a
    `DateData`; its "days" component sets the season (`tyear`/`orbital_phase`
    in `jcm.forcing._solar_from_date`, and any climatological `TimeSeries`
    forcing sliced by `ForcingData.select`), while its "seconds-within-day"
    component sets the diurnal phase (`synodic_phase`). Replacing only the
    days component -- with the real `sim_time % 86400` still feeding the
    seconds component -- freezes the season (the sun stops advancing past
    `freeze_season_at_day`) without freezing day/night.

    This monkey-patches the *instance*, not the `jcm.model.Model` class:
    Python attribute lookup checks the instance's `__dict__` before the
    class's, so `atm_model._date_from_sim_time` (called internally as
    `self._date_from_sim_time(...)`) resolves to this wrapper without
    touching the `jcm` package itself.
    """
    original_date_from_sim_time = atm_model._date_from_sim_time
    frozen_days_seconds = freeze_season_at_day * 86400.0

    def frozen_date_from_sim_time(sim_time):
        seconds_of_day = sim_time % 86400.0
        return original_date_from_sim_time(frozen_days_seconds + seconds_of_day)

    atm_model._date_from_sim_time = frozen_date_from_sim_time


def _build_jcm_terrain_file(mask_file, output_file):
    """Build a JCM-canonical terrain file (dims (lon, lat), vars lsm/orog --
    see jcm.terrain.TerrainData.from_file) from an ERA5-derived fractional
    land-sea mask + topography file (dims (latitude, longitude), vars
    lsm/topography, extra singleton valid_time axis) -- e.g.
    landsea_mask_fraction_JCM_T{n}.nc. Deliberately *not* using
    modify_jcm_terrain here: that only supports idealized synthetic
    geographies (aquaplanet, double_drake, ...), not real ERA5 topography.
    """
    ds = xr.open_dataset(mask_file)
    lsm = ds["lsm"]
    orog = ds["topography"]
    if "valid_time" in lsm.dims:
        lsm = lsm.isel(valid_time=0, drop=True)
        orog = orog.isel(valid_time=0, drop=True)

    out_ds = xr.Dataset(
        data_vars=dict(
            lsm=(("lon", "lat"), lsm.transpose("longitude", "latitude").to_numpy()),
            orog=(("lon", "lat"), orog.transpose("longitude", "latitude").to_numpy()),
        ),
        coords=dict(
            lon=("lon", ds["lon"].to_numpy()),
            lat=("lat", ds["lat"].to_numpy()),
        ),
        attrs=dict(grid_type="gaussian"),
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(output_file)
    return output_file


def build_model(
    *,
    truncation_number,
    start_datetime,
    coupling_timestep,
    calendar,
    veros_dt_mom=3600.0,
    veros_dt_tracer=3600.0,
    jcm_dt=1800.0,
    number_of_ocean_layers=None,
    debug_mode=False,
    freeze_season_at_day=None,
    grid_folder: str,
):
    """Build the coupled JCM + Veros + SlabOceanModel system.

    Returns the assembled `Coupler` together with the underlying component
    models and supporting objects (e.g. `ocn_model`, needed by Veros'
    checkpoint loader), so that callers can build exactly the same model.

    `freeze_season_at_day`: if not `None`, the number of days after
    `start_datetime` at which to freeze the atmosphere's seasonal (solar and
    climatological-forcing) cycle -- e.g. `0` pins it at `start_datetime`
    itself. The diurnal cycle keeps advancing normally. See `_freeze_season`.
    """

    landsea_mask_threshold = 0.5
    grid_folder = Path(grid_folder)
    data_files = {
        "grid" : {
            # No need for atm. It is built into jcm
            "ocn" : grid_folder / "RotatedGaussianLatLon.SCRIP.nc",
        },
        "landsea_mask" : {
            "atm" : grid_folder / f"landsea_mask_fraction_JCM_T{truncation_number:d}.nc",
            "ocn" : grid_folder / f"landsea_mask_fraction_RotatedGaussianLatLon.nc",
        },
        "regrid" : {
            "a2o" : {
                "bilinear" : ESMFRegridder(str(grid_folder / f"weight_algo-bilinear_JCM_T{truncation_number:d}_to_RotatedGaussianLatLon.nc")),
                "conserve" : ESMFRegridder(str(grid_folder / f"weight_algo-conserve_JCM_T{truncation_number:d}_to_RotatedGaussianLatLon.nc")),
            },
            "o2a" : {
                "bilinear" : ESMFRegridder(str(grid_folder / f"weight_algo-bilinear_RotatedGaussianLatLon_to_JCM_T{truncation_number:d}.nc")),
                "conserve" : ESMFRegridder(str(grid_folder / f"weight_algo-conserve_RotatedGaussianLatLon_to_JCM_T{truncation_number:d}.nc")),
            },
        },
    }
            
    grid_ds = xr.load_dataset(str(data_files["grid"]["ocn"]))
    # grid_cos_angle/grid_sin_angle are flat (grid_size,), SCRIP lon-fastest;
    # reshape with order="F" to (n_lon, n_lat), matching ESMFRegridder's own
    # convention so these line up index-for-index with regridded u0/v0.
    nlon, nlat = (int(n) for n in grid_ds["grid_dims"].to_numpy())
    cos_angle = jnp.array(grid_ds["grid_cos_angle"].to_numpy().reshape((nlon, nlat), order="F"))
    sin_angle = jnp.array(grid_ds["grid_sin_angle"].to_numpy().reshape((nlon, nlat), order="F"))


    coords = get_speedy_coords(spectral_truncation=truncation_number)

    # Real (ERA5-derived) topography, on JCM's own native (unrotated) grid --
    # not the rotated ocean grid's mask file. JCM's Coriolis parameter is
    # `2*omega*sin(lat)` computed directly from its own native latitude
    # (dinosaur.primitive_equations.PrimitiveEquations.coriolis_parameter,
    # hard-coded, no override hook), so terrain must be geographically
    # consistent with *that* latitude, not the ocean's rotated one.
    jcm_terrain_file = _build_jcm_terrain_file(
        data_files["landsea_mask"]["atm"],
        grid_folder / f"terrain_JCM_T{truncation_number:d}.nc",
    )
    terrain = TerrainData.from_file(jcm_terrain_file, coords=coords)

    # Create JCM
    atm_model = jcm.model.Model(
        coords=coords,
        start_date=start_datetime,
        terrain=terrain,
        time_step=jcm_dt/60.0,
        calendar=calendar,
    )
    JCM.make_jem_compatible(atm_model, coupling_timestep=coupling_timestep)
    if freeze_season_at_day is not None:
        _freeze_season(atm_model, freeze_season_at_day)
    atm_D2_nodal_shape = atm_model.coords.nodal_shape[1:]

    # Longitude/latitude (in degrees) of the atmosphere's nodal grid, used by
    # `report_first_nonfinite` in `interaction` to locate non-finite values.
    lon_deg = atm_model.coords.horizontal.longitudes * 180 / jnp.pi
    lat_deg = atm_model.coords.horizontal.latitudes * 180 / jnp.pi

    # Create Veros
    # First need to remove `output_veros.*.nc` files, otherwise veros complains.
    for f in glob.glob("output_veros.*.nc"):
        print(f"Deleting: {f}")
        os.remove(f)

    ocn_model = generateVerosSetup(
        scrip_grid_file = data_files["grid"]["ocn"],
        landsea_mask_file = data_files["landsea_mask"]["ocn"],
        landsea_mask_threshold = landsea_mask_threshold,
        dt_mom=veros_dt_mom,
        dt_tracer=veros_dt_tracer,
        ddz=[50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0][:number_of_ocean_layers],
    )()
    ocn_model.setup()
    Veros.make_jem_compatible(ocn_model, coupling_timestep=coupling_timestep)

    # Create Slab Ocean model
    #
    # `jem.components.slab.grid.load_jcm_fractional_mask` does a bare
    # `jnp.asarray(ds["lsm"])`, but this file's "lsm" carries an extra
    # singleton `valid_time` axis and (lat, lon) order rather than JEM's
    # (lon, lat) -- squeeze and transpose before handing it to SlabGrid.
    atm_mask_ds = xr.open_dataset(data_files["landsea_mask"]["atm"])
    atm_lsm = atm_mask_ds["lsm"].to_numpy()
    if atm_lsm.ndim == 3:
        atm_lsm = atm_lsm[0]
    atm_fractional_mask = jnp.asarray(atm_lsm.T)  # (lat, lon) -> (lon, lat)

    fakelnd_grid = generate_slab_grid(
        f"JCM::T{truncation_number:d}",
        fractional_mask=atm_fractional_mask,
    )
    fakelnd_model = SlabOceanModel(
        grid=fakelnd_grid,
        start_datetime=start_datetime,
        timestep=coupling_timestep / one_second,
        forcing_method=None,
        calendar=calendar,
    )

    # Creating Flux and Scalar Exchange between Components
    #
    # Here we demonstrate the flexibility of JEM: you do not have to use
    # the `BasicMapper` that JEM provides. You can define your own mapping
    # function. In this example, we simply define a function `interaction`
    # as below.

    a2o_regridder = data_files["regrid"]["a2o"]["conserve"]
    o2a_regridder = data_files["regrid"]["o2a"]["bilinear"]

    def veros_to_jcm_regridder(arr):
        return o2a_regridder(arr)

    def jcm_to_veros_regridder(arr):
        return a2o_regridder(arr)


    # Note: Remember to return the `coupled_carry` at the end.
    def interaction(coupled_carry):
        atm = coupled_carry["atm"]
        ocn = coupled_carry["ocn"]
        fakelnd = coupled_carry["fakelnd"]

        # ===== compute wind stress begin =====
        # Tien-Yiao's ad-hoc way to compute wind stress.
        # This conveniently demonstrates how the flux computation can be
        # its own function or module.
        drag_coefficient = 1e-3  # dimensionless
        air_density = 1.22  # kg / m^3
        # `jcm_to_veros_regridder` interpolates u0/v0 component-wise, which
        # is fine since JCM's own grid is unrotated (u0/v0 are genuinely
        # true-east/true-north everywhere on it). The result is then
        # rotated into the ocean grid's local (rotated) frame below using
        # its per-cell `grid_cos_angle`/`grid_sin_angle`.
        u0 = jcm_to_veros_regridder(atm["derived"].physics["_surface_flux"].u0)
        v0 = jcm_to_veros_regridder(atm["derived"].physics["_surface_flux"].v0)

        wind_x =   cos_angle * u0 + sin_angle * v0 
        wind_y = - sin_angle * u0 + cos_angle * v0
        # `sqrt` has an infinite derivative at 0, so AD through `sqrt(x)`
        # blows up to NaN as x -> 0, even though the primal value (0) stays
        # finite. Floor the *squared* speed -- sqrt's argument -- at
        # min_speed_magnitude**2 so sqrt and its derivative stay bounded;
        # this naturally caps the resulting wind speed from below at
        # min_speed_magnitude. Mirrors the analogous fix in
        # `jem.components.Veros.make_jem_compatible` for `forc_tke_surface`.
        min_speed_magnitude = 1e-3
        wind_speed_squared = wind_x**2 + wind_y**2
        wind_velocity = jnp.sqrt(jnp.maximum(wind_speed_squared, min_speed_magnitude ** 2))
        surface_taux = drag_coefficient * air_density * wind_velocity * wind_x
        surface_tauy = drag_coefficient * air_density * wind_velocity * wind_y
        # ===== compute wind stress end =====

        # Cap total heat flux for now. There seems to be instability coming
        # from JCM. Need investigation.
        # total_heat_flux = jnp.clip(atm["derived"]["total_heat_flux"], min=-1372.0, max=1372.0)
        total_heat_flux = atm["derived"].total_heat_flux

        # ===== simple "swamp" sea ice mask begin =====
        # Ported from `veros/setups/global_1deg/global_1deg.py`: once the
        # ocean surface reaches the freezing point, sea ice insulates it
        # from further heat/freshwater exchange with the atmosphere. Flux
        # that would warm the ocean (melt) is always let through. `heat_flux`
        # is upward-positive (see `jem.components.Veros`), so a flux that
        # warms the ocean is negative.
        freezing_point_K = 273.15 - 1.8
        heat_flux = jcm_to_veros_regridder(total_heat_flux)
        freshwater_flux = jcm_to_veros_regridder(atm["derived"].total_freshwater_flux)
        not_frozen = ocn["derived"].sea_surface_temperature > freezing_point_K
        would_warm = heat_flux < 0
        ice_free = jnp.logical_or(not_frozen, would_warm)
        heat_flux = heat_flux * ice_free
        freshwater_flux = freshwater_flux * ice_free
        # ===== simple "swamp" sea ice mask end =====

        # Mapping
        ocn["forcing"].surface_taux = surface_taux
        ocn["forcing"].surface_tauy = surface_tauy
        ocn["forcing"].heat_flux = heat_flux
        ocn["forcing"].freshwater_flux = freshwater_flux
        ocn["forcing"].wind_x = wind_x
        ocn["forcing"].wind_y = wind_y
        fakelnd["forcing"].total_heat_flux = total_heat_flux
        fakelnd["state"].sea_surface_temperature = jnp.clip(
            fakelnd["state"].sea_surface_temperature,
            200.0, #273.15 - 50.0,
            273.15 + 30.0,
        )
        atm["forcing"].sea_surface_temperature = veros_to_jcm_regridder(ocn["derived"].sea_surface_temperature)
        atm["forcing"].stl_am = fakelnd["state"].sea_surface_temperature


        if debug_mode:

            print("Notice: debug_mode is on.")

            # Debug: report wind speed (atm), ocean velocity, and stratification (ocn) extremes
            vs_ocn = ocn["state"].variables
            jax.debug.print(
                "max|wind_u| (atm) = {wu:.6e}, max|wind_v| (atm) = {wv:.6e}, "
                "max|u| (ocn) = {ou:.6e}, max|v| (ocn) = {ov:.6e}, min(Nsqr) (ocn) = {nsqr:.6e}",
                wu=jnp.max(jnp.abs(wind_x)),
                wv=jnp.max(jnp.abs(wind_y)),
                ou=jnp.max(jnp.abs(vs_ocn.u[..., vs_ocn.tau])),
                ov=jnp.max(jnp.abs(vs_ocn.v[..., vs_ocn.tau])),
                nsqr=jnp.min(vs_ocn.Nsqr[..., vs_ocn.tau]),
            )

            # Debug model explosion
            def breakpoint_if_nonfinite(x):
                all_finite = is_pytree_all_finite(x)
                def true_fn(x):
                    pass
                def false_fn(x):
                    jax.debug.print("Model exploded. Activate breakpoint to see what happened.")
                    report_first_nonfinite("wind_x", wind_x, lon_deg, lat_deg)
                    report_first_nonfinite("wind_y", wind_y, lon_deg, lat_deg)
                    report_first_nonfinite("ocn.forcing.heat_flux", ocn["forcing"].heat_flux, lon_deg, lat_deg)
                    report_first_nonfinite("atm.forcing.stl_am", atm["forcing"].stl_am, lon_deg, lat_deg)
                    # The crash was first seen in the ocean state (u, v, Nsqr
                    # all NaN while the atmosphere was still finite), so check
                    # the ocean's prognostic fields too (3D: x, y, depth).
                    report_first_nonfinite("ocn.state.temp", vs_ocn.temp[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.salt", vs_ocn.salt[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.u", vs_ocn.u[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.v", vs_ocn.v[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.Nsqr", vs_ocn.Nsqr[..., vs_ocn.tau])
                    # jax.debug.breakpoint()  # disabled: no TTY in SLURM batch jobs -> JaxRuntimeError
                jax.lax.cond(all_finite, true_fn, false_fn, x)

            breakpoint_if_nonfinite(coupled_carry) 


        return coupled_carry

    model = Coupler(
        components=dict(
            atm=atm_model,
            ocn=ocn_model,
            fakelnd=fakelnd_model,
        ),
        mappers=dict(mapper=interaction),
    )

    config = dict(
        #terrain=terrain,
        #modified_jcm_terrain_file=modified_jcm_terrain_file,
        coords=coords,
        workflow=["mapper", "atm", "ocn", "fakelnd"],
    )

    return model, config
