# Construction of the coupled JCM + Veros + SlabOceanModel system used by
# the double-drake experiment, packaged as a function so that the forward
# run (main.py) and other scripts (e.g. a jax.grad sensitivity experiment)
# build exactly the same model.

import glob
import os

import jax.numpy as jnp
import jax_datetime as jdt

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jem.tool_scripts.generate_jcm_forcing_and_topography_files import generate_jcm_forcing_and_topography_files

from jem.components import JCM, SlabOceanModel
from jem.base.coupler import Coupler

from modify_jcm_terrain import modify_jcm_terrain

from veros.core.operators import update, at

one_second = jdt.to_timedelta(1, "second")

def build_model(
    *,
    truncation_number,
    start_datetime,
    coupling_timestep,
    calendar,
    terrain_output_directory="./data",
):
    """Build the coupled JCM + Veros + SlabOceanModel system.

    Returns the assembled `Coupler` together with the underlying component
    models and supporting objects (e.g. `ocn_model`, needed by Veros'
    checkpoint loader), so that callers can build exactly the same model.
    """

    coords = get_speedy_coords(spectral_truncation=truncation_number)

    jcm_files = generate_jcm_forcing_and_topography_files(resolution=truncation_number)
    modified_jcm_terrain_file = modify_jcm_terrain(
        jcm_files["terrain"], terrain_output_directory,
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

    def interaction(coupled_carry):
        """
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
        wind_velocity = jnp.sqrt(wind_x**2 + wind_y**2)
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
            200.0,
            273.15 + 30.0,
        )
        atm["forcing"].sea_surface_temperature = veros_to_jcm_regridder(ocn["derived"]["sea_surface_temperature"])
        atm["forcing"].stl_am = fakelnd["state"]["sea_surface_temperature"]
        """
        return coupled_carry

    model = Coupler(
        components=dict(
            atm=atm_model,
        ),
        mappers=dict(mapper=interaction),
    )

    config = dict(
        terrain=terrain,
        modified_jcm_terrain_file=modified_jcm_terrain_file,
        coords=coords,
        workflow=["atm"],
    )

    return model, config
