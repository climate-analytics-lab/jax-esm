# Construction of the coupled JCM + Veros + SlabOceanModel system used by
# the double-drake experiment, packaged as a function so that the forward
# run (main.py) and other scripts (e.g. a jax.grad sensitivity experiment)
# build exactly the same model.

import glob
import os

import jax
import jax.numpy as jnp
import jax_datetime as jdt

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jem.tool_scripts.generate_jcm_forcing_and_topography_files import generate_jcm_forcing_and_topography_files

from jem.components import JCM, Veros, SlabOceanModel
from jem.base.coupler import Coupler

from modify_jcm_terrain import modify_jcm_terrain
from veros_case_setup import generateVerosSetup

from veros.core.operators import update, at

one_second = jdt.to_timedelta(1, "second")

# Veros hard-coded ghost cell number; see jem/components/Veros.py, where
# `derived["sea_surface_temperature"]` is diagnosed from this same slice
# of the prognostic temperature field.
_OCEAN_GHOST_CELL = 2


def get_ocean_surface_temperature(ocn_state):
    """Read the ocean's sea surface temperature [K] from its prognostic state.

    Mirrors the diagnostic computed in `jem.components.Veros.make_jem_compatible`,
    so callers can read/perturb exactly the field the coupled dynamics use.
    """
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    return vs.temp[g:-g, g:-g, -1, vs.tau] + 273.15


def set_ocean_surface_temperature(ocn_state, sst):
    """Functionally overwrite the ocean's prognostic surface temperature [K].

    Unlike `coupled_carry["ocn"]["derived"]["sea_surface_temperature"]`, which
    is recomputed from the prognostic field every sub-step, this writes into
    `state.variables.temp` itself, so a perturbation here actually propagates
    through the ocean dynamics.
    """
    vs = ocn_state.variables
    g = _OCEAN_GHOST_CELL
    with vs.unlock():
        vs.temp = update(vs.temp, at[g:-g, g:-g, -1, vs.tau], sst - 273.15)
    return ocn_state


def build_model(
    *,
    truncation_number,
    start_datetime,
    coupling_timestep,
    calendar,
    terrain_planet_type="double_drake",
    terrain_output_directory="./data",
    veros_dt_mom=3600.0,
    veros_dt_tracer=3600.0,
    debug_mode=False,
):
    """Build the coupled JCM + Veros + SlabOceanModel system.

    Returns the assembled `Coupler` together with the underlying component
    models and supporting objects (e.g. `ocn_model`, needed by Veros'
    checkpoint loader), so that callers can build exactly the same model.
    """

    coords = get_speedy_coords(spectral_truncation=truncation_number)

    jcm_files = generate_jcm_forcing_and_topography_files(resolution=truncation_number)
    modified_jcm_terrain_file = modify_jcm_terrain(
        jcm_files["terrain"], terrain_planet_type, terrain_output_directory,
    )
    terrain = TerrainData.from_file(modified_jcm_terrain_file, coords=coords)

    # Create JCM
    atm_model = jcm.model.Model(
        coords=coords,
        start_date=start_datetime,
        terrain=terrain,
        time_step=30,
        calendar=calendar,
    )
    JCM.make_jem_compatible(atm_model, coupling_timestep=coupling_timestep)
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
        nx=atm_D2_nodal_shape[0],
        ny=atm_D2_nodal_shape[1],
        land_sea_mask_file=modified_jcm_terrain_file,
        dt_mom=veros_dt_mom,
        dt_tracer=veros_dt_tracer,
    )()
    ocn_model.setup()
    Veros.make_jem_compatible(ocn_model, coupling_timestep=coupling_timestep)

    # Create Slab Ocean model
    fakelnd_model = SlabOceanModel(
        grid_specification=f"JCM::T{truncation_number:d}",
        start_datetime=start_datetime,
        timestep=coupling_timestep / one_second,
        mask_file=modified_jcm_terrain_file,
        forcing_method=None,
        mask_value=1.0,
        calendar=calendar,
    )

    # Creating Flux and Scalar Exchange between Components
    #
    # Here we demonstrate the flexibility of JEM: you do not have to use
    # the `BasicMapper` that JEM provides. You can define your own mapping
    # function. In this example, we simply define a function `interaction`
    # as below.

    def veros_to_jcm_regridder(arr):
        return arr  # jnp.pad(arr, ((0, 0), (4, 4)), constant_values=150)

    def jcm_to_veros_regridder(arr):
        return arr  # arr[:, 4:-4]

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
        wind_x = jcm_to_veros_regridder(atm["derived"]["physics"]["_surface_flux"].u0)
        wind_y = jcm_to_veros_regridder(atm["derived"]["physics"]["_surface_flux"].v0)
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
        total_heat_flux = atm["derived"]["total_heat_flux"]

        # Mapping
        ocn["forcing"].surface_taux = surface_taux
        ocn["forcing"].surface_tauy = surface_tauy
        ocn["forcing"].heat_flux = jcm_to_veros_regridder(total_heat_flux)
        ocn["forcing"].freshwater_flux = jcm_to_veros_regridder(atm["derived"]["total_freshwater_flux"])
        ocn["forcing"].wind_x = jcm_to_veros_regridder(wind_x)
        ocn["forcing"].wind_y = jcm_to_veros_regridder(wind_y)
        fakelnd["forcing"].total_heat_flux = jcm_to_veros_regridder(total_heat_flux)
        fakelnd["state"].sea_surface_temperature = jnp.clip(
            fakelnd["state"].sea_surface_temperature,
            200.0, #273.15 - 50.0,
            273.15 + 30.0,
        )
        atm["forcing"].sea_surface_temperature = veros_to_jcm_regridder(ocn["derived"]["sea_surface_temperature"])
        atm["forcing"].stl_am = fakelnd["state"]["sea_surface_temperature"]

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

        if debug_mode:
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
                    jax.debug.breakpoint()
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
        terrain=terrain,
        modified_jcm_terrain_file=modified_jcm_terrain_file,
        coords=coords,
        workflow=["mapper", "atm", "ocn", "fakelnd"],
    )

    return model, config
