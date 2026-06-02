import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.typing import ArrayLike
from dataclasses import dataclass
import tree_math

from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

earth_radius = 6.371e6          # Unit: m
total_air_mass_per_unit_area = 1e4   # Unit: kg / m^2
default_total_atmosphere_mass = 4 * jnp.pi * earth_radius**2 * total_air_mass_per_unit_area 
dry_air_molecular_weight = 28.9e-3 # Unit: kg / mole
carbon_molecular_weight =  12e-3   # Unit: kg / mole

@tree_math.struct
@dataclass
class CO2AtmosphereBoxModelCarry:
    t: ArrayLike                       # Unit: seconds
    co2_mixing_ratio: ArrayLike        # Unit: ppm
    forcing_source_and_sink: ArrayLike # Unit: kg / second
    emission: ArrayLike                # Unit: kg / second

class CO2AtmosphereBoxModel:

    def __init__(self,
        timestep: float,
        initial_co2_mixing_ratio: float = 280.0,
        initial_forcing_source_and_sink: float = 0.0,
        total_atmosphere_mass: float = default_total_atmosphere_mass, 
    ):

        self.timestep = timestep
        self.initial_co2_mixing_ratio = initial_co2_mixing_ratio
        self.initial_forcing_source_and_sink = initial_forcing_source_and_sink
        self.total_atmosphere_mass = total_atmosphere_mass

    def initialize(self):
        return CO2AtmosphereBoxModelCarry(
            t = jnp.array(0),
            co2_mixing_ratio = jnp.array(self.initial_co2_mixing_ratio),
            forcing_source_and_sink = jnp.array(self.initial_forcing_source_and_sink),
            emission = jnp.array(0),
        )

    def generate_step_function(self):
        dt = self.timestep
        total_atmosphere_mass = self.total_atmosphere_mass
        def step_function(carry, step):
            """Integrates one time step of the CO2 atmosphere box model."""

            
            total_source_and_sink = carry.forcing_source_and_sink + carry.emission

            # Physics: dco2_mixing_ratio/dt = (total_source_and_sink / carbon_molecular_weight) / (total_atmosphere_mass / dry_air_molecular_weight) * 1e6 (ppm)
            dco2_mixing_ratio = (total_source_and_sink / carbon_molecular_weight) / (total_atmosphere_mass / dry_air_molecular_weight) * 1e6 * dt
                
            new_co2_mixing_ratio = carry.co2_mixing_ratio + dco2_mixing_ratio
            total_carbon = (new_co2_mixing_ratio * 1e-6) * (total_atmosphere_mass / dry_air_molecular_weight) * carbon_molecular_weight
            new_carry = CO2AtmosphereBoxModelCarry(
                t = carry.t + dt,
                co2_mixing_ratio = new_co2_mixing_ratio,
                forcing_source_and_sink = carry.forcing_source_and_sink,
                emission = carry.emission,
            )

            return new_carry, dict(
                t=new_carry.t,
                total_carbon=total_carbon,
                co2_mixing_ratio=new_carry.co2_mixing_ratio,
                forcing_source_and_sink=new_carry.forcing_source_and_sink,
                emission = carry.emission,
            )

        return step_function

    def predictions_to_xarray(self, predictions) -> xr.Dataset:
        return xr.Dataset(
            data_vars=dict(
                co2_mixing_ratio=(
                    ["time"],
                    np.array(predictions["co2_mixing_ratio"]),
                    {"long_name": "Atmospheric CO2 volume mixing ratio", "units": "ppm"},
                ),

                emission=(
                    ["time"],
                    np.array(predictions["emission"]),
                    {"long_name": "CO2 emission rate", "units": "kg / s"},
                ),

                forcing_source_and_sink=(
                    ["time"],
                    np.array(predictions["forcing_source_and_sink"]),
                    {"long_name": "CO2 source and sink forcing", "units": "kg / s"},
                ),
                total_carbon=(
                    ["time"],
                    np.array(predictions["total_carbon"]),
                    {"long_name": "Total carbon (of CO2) in the atmosphere", "units": "kg"},
                ),
            ),
            coords=dict(
                time=(
                    ["time"],
                    np.array(predictions["t"]),
                    {"long_name": "Simulation time", "units": "s"},
                ),
            ),
        )
