import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from dataclasses import dataclass
import tree_math

from jem.base.coupler import Coupler
import jem.utils.tree_tools as tree_tools

earth_radius = 6.371e6          # Unit: m
total_air_mass_per_unit_area = 1e4   # Unit: kg / m^2
default_total_atmosphere_mass = 4 * jnp.pi * earth_radius**2 * total_air_mass_per_unit_area 
dry_air_molecular_weight = 28.9e-3 # Unit: kg / mole
co2_gas_molecular_weight = 44e-3   # Unit: kg / mole

@tree_math.struct
@dataclass
class CO2AtmosphereBoxModelCarry:
    t: ArrayLike                       # Unit: seconds
    co2_mixing_ratio: ArrayLike        # Unit: ppm
    forcing_source_and_sink: ArrayLike # Unit: kg / second

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
        )

    def generate_step_function(self):
        dt = self.timestep
        total_atmosphere_mass = self.total_atmosphere_mass
        def step_function(carry, step):
            """Integrates one time step of the CO2 atmosphere box model."""

            # Physics: dco2_mixing_ratio/dt = (forcing_source_and_sink / co2_gas_molecular_weight) / (total_atmosphere_mass / dry_air_molecular_weight) * 1e6 (ppm)
            dco2_mixing_ratio = (carry.forcing_source_and_sink / co2_gas_molecular_weight) / (total_atmosphere_mass / dry_air_molecular_weight) * 1e6 * dt

            new_carry = CO2AtmosphereBoxModelCarry(
                t = carry.t + dt,
                co2_mixing_ratio = carry.co2_mixing_ratio + dco2_mixing_ratio,
                forcing_source_and_sink = carry.forcing_source_and_sink,
            )

            return new_carry, dict(t=new_carry.t, co2_mixing_ratio=new_carry.co2_mixing_ratio, forcing_source_and_sink=new_carry.forcing_source_and_sink)

        return step_function
