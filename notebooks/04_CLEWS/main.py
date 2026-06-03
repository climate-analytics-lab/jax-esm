# Realistic Earth - JCM + chemistry slab ocean model

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from checkpoint_aquaplanet_co2boxmodel import save_coupled_carry, load_coupled_carry

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from importlib import resources

import jax_datetime as jdt

from jem.components import JCM, SlabOceanModelBGC, SlabLandModel, CO2AtmosphereBoxModel
from jem.mapping import BasicMapper
from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools
use_ipython = 'get_ipython' in globals()

jax.config.update('jax_platform_name', 'cpu')
print("Print devices: ", jax.devices())


def positive_cosine_cubic_latitude_squared(
    lat: jnp.ndarray,
    amplitude: float = 1.0,
) -> jnp.ndarray:
    return jnp.where(
        jnp.abs(lat) < jnp.pi / 3, amplitude * jnp.cos(3 * lat / 2) ** 2, 0
    )

# Configurations

average_output_of_each_batch = True
calendar = "365_day"
spectral_truncation = 31
total_simulation_months = 12 * 1
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")
simulation_name = "default"
output_dir = (Path(f"output_T{spectral_truncation:d}") / simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
one_second = jdt.to_timedelta(1, "second")


# Load realistic orography and land-sea mask, interpolated to target grid
coords=get_speedy_coords(spectral_truncation=spectral_truncation)

data_dir = resources.files(f"jcm.data.bc.t{spectral_truncation-1:d}.clim")
terrain_file = data_dir / "terrain.nc"
terrain = TerrainData.from_file(terrain_file, coords=coords)

# Load realistic forcing data (SST, sea ice, soil moisture, etc.) interpolated to target grid
forcing_file = data_dir / "forcing.nc"
forcing = ForcingData.from_file(forcing_file, coords=coords)


# Define emission shapes

def emission_stepwise(days_after_start):
    emission_rate = 10e12 / (86400 * 365)
    return jnp.select(
        condlist=[
            days_after_start < 41.0,
            days_after_start < 80.0,
        ],
        choicelist=[
            0.0,
            emission_rate / 3,
        ],
        default=emission_rate,
    )

def emission_linear(days_after_start):
    emission_rate_base = 10e12 / (86400 * 365)
    emission_rate_change_per_decade = 10e12 / (86400 * 365)
    return emission_rate_base + emission_rate_change_per_decade * days_after_start / (365*10) 

def emission_zero(days_after_start):
    return jnp.array(0.0)


# Creating Flux and Scalar Exchange between Components
# Note: Remember to return the `coupled_carry` at the end.
def interactions(coupled_carry):
    print(list(coupled_carry.keys()))
    atm = coupled_carry["atm"]
    lnd = coupled_carry["lnd"]
    ocn = coupled_carry["ocn"]
    co2_atm_boxmodel = coupled_carry["co2_atm_boxmodel"]

    # Mapping
    ocn["forcing"].total_heat_flux = atm["derived"]["ocean_heat_flux"]
    lnd["forcing"].total_heat_flux = atm["derived"]["land_heat_flux"]
    atm["forcing"].sea_surface_temperature = ocn["state"]["sea_surface_temperature"]
    atm["forcing"].stl_am = lnd["state"]["land_surface_temperature"]
    atm["forcing"].snowc_am = lnd["state"]["snowc"]
    atm["forcing"].soilw_am = lnd["state"]["soilw"]

    # Carbon coupling
    u0 = atm["derived"]["physics"]["_surface_flux"].u0
    v0 = atm["derived"]["physics"]["_surface_flux"].v0
    ocn["forcing"].U10 = jnp.sqrt(u0**2 + v0**2)
    ocn["forcing"].air_co2_volume_mixing_ratio = co2_atm_boxmodel.co2_mixing_ratio
    atm["forcing"].co2_vmr = co2_atm_boxmodel.co2_mixing_ratio
    co2_atm_boxmodel.emission = emission_zero(days_after_start) #emission_linear(co2_atm_boxmodel.t / 86400.0)
    co2_atm_boxmodel.forcing_source_and_sink = (
        ocn["derived"]["total_carbon_flux"]
    )
    
    return coupled_carry

# Create Components

co2_atm_boxmodel = CO2AtmosphereBoxModel(
    timestep=coupling_timestep / one_second,
    initial_co2_mixing_ratio = 400.0,
)

atm_model = jcm.model.Model(
    coords=get_speedy_coords(spectral_truncation=spectral_truncation),
    start_date=start_datetime,
    time_step=10.0,
    calendar=calendar,
    terrain=terrain,
)

atm_model = JCM.make_jem_compatible(
    atm_model,
    coupling_timestep=coupling_timestep,
)


hgrid = atm_model.coords.horizontal
lat = hgrid.latitudes
lon = hgrid.longitudes

llat = jnp.repeat(
    lat[None, :],
    repeats=len(lon),
    axis=0,
)
llon = jnp.repeat(
    lon[:, None],
    repeats=len(lat),
    axis=1,
)

model = Coupler(
    components=dict(
        co2_atm_boxmodel=co2_atm_boxmodel,
        atm=atm_model,
        ocn=SlabOceanModelBGC(
            grid_specification=f"JCM::T{spectral_truncation:d}",
            start_datetime=start_datetime,
            timestep=coupling_timestep / one_second,
            mask_file=terrain_file,
            SST_clim_file=forcing_file,
            forcing_method="relaxation",
            relaxation_time= 360 * 86400.0 * 1000,
            calendar=calendar,
        ),
        lnd=SlabLandModel(
            start_datetime=start_datetime,
            topography_file=terrain_file,
            mask_file=terrain_file,
            land_clim_file=forcing_file,
            tdland =  360 * 86400.0 * 1000, 
            calendar=calendar,
        ),
    ),
    mappers=dict(interactions=interactions),
)

print("Model info: ") 
tree_tools.print_tree(model.get_info(), root="Model")

# Run Coupled Model
initial_carry = model.initialize()
days_of_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
simulation_intervals = [ jdt.to_timedelta(_d, "day") for _d in days_of_month ]
batches = total_simulation_months
checkpoint_dir = output_dir / "checkpoint"
resume_batch = 0
trajectory_functions = dict()

if checkpoint_dir.exists():
    saved = sorted(checkpoint_dir.glob("batch_*"))
    if saved:
        resume_batch = int(saved[-1].name.split("_")[1]) + 1
        print(f"Resuming from batch {resume_batch}")
        initial_carry = load_coupled_carry(saved[-1], list(model.components.keys()))
    else:
        initial_carry["ocn"]["state"].sea_surface_temperature = (
            273.15 + positive_cosine_cubic_latitude_squared(llat) * 50.0
        )
else:
    initial_carry["ocn"]["state"].sea_surface_temperature = (
        273.15 + positive_cosine_cubic_latitude_squared(llat) * 50.0
    )

for b in range(resume_batch, resume_batch + batches):
    print(f"[batch={b:d}/{batches:d}] Simulation...")
    simulation_interval = simulation_intervals[b % len(simulation_intervals)]
    iterations = int(simulation_interval / coupling_timestep)

    if iterations not in trajectory_functions:
        print(f"Trajectory for iterations {iterations:d} does not exist. Create one.")
        trajectory_functions[iterations] = model.generate_trajectory_function(
            workflow=["interactions", "co2_atm_boxmodel", "atm", "ocn", "lnd"],
            iterations = int(simulation_interval / coupling_timestep),
            jitted=True,
        )
    
    
    final_carry, predictions = trajectory_functions[iterations](initial_carry)
    output_dict = model.predictions_to_xarray(predictions)

    for component_name, ds in output_dict.items():
        output_file = output_dir / f"{component_name:s}-{b:05d}.nc"
        print("Output file: ", str(output_file))
        if average_output_of_each_batch:
            first_time = ds.coords["time"][:1]
            ds = ds.reduce(np.mean, dim="time", keepdims=True)
            ds = ds.assign_coords(time=first_time)
        ds.to_netcdf(output_file, engine="netcdf4", unlimited_dims=["time"])
        ds.close()

    if jnp.any( jnp.isnan(output_dict["atm"]["specific_humidity"].to_numpy()) ):
        print("Error: Model exploded. End program")
        break

    initial_carry = final_carry
    save_coupled_carry(final_carry, checkpoint_dir / f"batch_{b:05d}")


