"""Slab ocean model biogeochem component."""

from typing import Optional, Dict, Any, Annotated
from pathlib import Path

import jax_datetime as jdt
import jax.numpy as jnp
import xarray as xr

from jem import constants
from jem.utils.bulk_op import stack_objects
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared
import jem.utils.data_structure as data_structure
from jem.components.slab.base import SlabModelBase
from jem.components.slab.slab_ocean_model.biogeochem_coefficients import compute_co2_flux

default_land_surface_temperature = 288.15
carbon_molecular_weight = 12e-3  # kg / mol


@data_structure.typed_and_dimensioned
class OceanState:
    sim_time: Annotated[float, (), "zero_dimensional"]
    sea_surface_temperature: Annotated[
        float, ("longitude", "latitude"), "two_dimensional"
    ]
    mixed_layer_depth: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    mixed_layer_dissolved_inorganic_carbon: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    deep_layer_dissolved_inorganic_carbon: Annotated[float, (), "zero_dimensional"]


@data_structure.typed_and_dimensioned
class OceanForcing:
    total_heat_flux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    q_flux: Annotated[float, ("longitude", "latitude", "month"), "two_dimensional_with_month"]
    U10: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    co2_flux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    air_co2_volume_mixing_ratio: Annotated[float, (), "zero_dimensional"]

class SlabOceanModelBGC(SlabModelBase):
    """Slab ocean model with prescribed mixed layer depth and climatology.

    This model simulates sea surface temperature evolution using a simple
    thermodynamic equation with optional relaxation to climatology.
        
    dT/dt = F_net/(rho * cp * h) + forcing


    where:
        T: sea surface temperature
        F_net: total heat flux (positive upward)
        rho: ocean density
        cp: ocean specific heat capacity
        h: mixed layer depth
        forcing: the forcing of temperature. See below for explaination
    
    (1) If `forcing_method` == "None" (or just None), then forcing = 0.

    (2) If `forcing_method` == "Qflux", then traditional Q-flux adjust, i.e., periodic forcing
        over a year, is used:
 
            forcing = Q / (rho * cp * h)

        where variable `Q` will be read from a file given in `Q_flux_file`. If `Q_flux_file`
        is not provided, then Q will be all zeros, which is possible when doing training.
    
    (3) If `forcing_method` == "relaxation", then linear relaxation will be used

            forcing = - (T - T_clim) / tau

        where tau is the relaxation timescale to climatology (can be jnp.inf), and T_clim
        is the climatology read from `SST_clim_file`. If `SST_clim_file` is not provided, 
        then T_clim will be all zeros, which is possible when doing training.
    """

    def __init__(
        self,
        grid_specification: str = "JCM::T31",
        start_datetime: jdt.Datetime = jdt.to_datetime("2001-01-01"),
        timestep: float = 86400.0,
        relaxation_time: float = 60 * 86400.0,
        mixed_layer_depth_min: float = 40.0,
        mixed_layer_depth_max: float = 60.0,
        deep_layer_depth: float = 500.0,
        topography_file: Optional[str] = None,
        mask_file: Optional[str] = None,
        SST_clim_file: Optional[str] = None,
        Q_flux_file: Optional[str] = None,
        forcing_method: Optional[str] = None,
        initialization_sea_surface_temperature: float = 288.15,
        mask_value: float = 0.0,
        init_mixed_layer_dissolved_inorganic_carbon: float = 2.0,  # mol / m^3
        init_deep_layer_dissolved_inorganic_carbon: float = 2.0,   # mol / m^3
        mixed_deep_layer_exchange_time_scale: float = 86400.0 * 365 * 1000, # 1000 years
        total_alkalinity: float = 2.3,
        calendar: str = "365_day",
    ):
        """Initialize slab ocean model.

        Args:
            grid_specification: Grid spec string (e.g., "JCM::T31")
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds
            relaxation_time: Relaxation timescale to climatology in seconds
            mixed_layer_depth_min: Minimum mixed layer depth in meters
            mixed_layer_depth_max: Maximum mixed layer depth in meters
            topography_file: Optional path to topography NetCDF file
            mask_file: Optional path to land/ocean mask NetCDF file
            SST_clim_file: Optional path to SST climatology NetCDF file
        """
        self.relaxation_time = relaxation_time
        self.mixed_layer_depth_min = mixed_layer_depth_min
        self.mixed_layer_depth_max = mixed_layer_depth_max
        self.deep_layer_depth = deep_layer_depth
        self.SST_clim_file = SST_clim_file
        self.Q_flux_file = Q_flux_file
        self.init_mixed_layer_dissolved_inorganic_carbon = init_mixed_layer_dissolved_inorganic_carbon
        self.init_deep_layer_dissolved_inorganic_carbon = init_deep_layer_dissolved_inorganic_carbon
        self.mixed_deep_layer_exchange_time_scale = mixed_deep_layer_exchange_time_scale
        self.total_alkalinity = total_alkalinity

        super().__init__(
            name="SlabOceanModel",
            grid_specification=grid_specification,
            start_datetime=start_datetime,
            timestep=timestep,
            topography_file=topography_file,
            mask_file=mask_file,
            calendar=calendar,
        )

        # Climatology data (loaded during initialize)
        self.SST_clim = None
        self.time_factor = None
        self.cd_factor = None
        self.forcing_method = forcing_method or "None"
        self.mask_value = mask_value

        self.validate()

    def validate(self):
        super().validate()
        if self.forcing_method == "None":
            # Do nothing
            pass
        elif self.forcing_method == "Qflux":
            if self.Q_flux_file is None:
                print("Notice: `Q_flux_file` is not given. Default values (zeros) will be used.")
            elif not Path(self.Q_flux_file).exists():
                raise FileNotFoundError(f"Q-flux file \"{str(self.Q_flux_file):s}\" is specified but it does not exist.")
        elif self.forcing_method == "relaxation":
            if self.SST_clim_file is None:
                print("Notice: `SST_clim_file` is not given. Default values (zeros) will be used.")
            elif not Path(self.SST_clim_file).exists():
                raise FileNotFoundError(f"SST climatology file \"{str(self.SST_clim_file):s}\" is specified but does not exist.")
            elif (self.relaxation_time < 0) or jnp.isnan(self.relaxation_time):
                raise ValueError("`relaxation_time` must be a positive number or infinity.")
        else: 
            raise ValueError(f"Unknown `forcing_method` is given: \"{str(self.forcing_method):s}\" ")
 

    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for ocean model."""
        decorator = data_structure.build_dataclass_from_typed_and_dimensioned({"two_dimensional": self.grid_shape})
        decorator = data_structure.build_dataclass_from_typed_and_dimensioned(
            {
                "two_dimensional": self.grid_shape,
                "two_dimensional_with_month": self.grid_shape + (12,),
            }
        )
        self.component_state_class = decorator(OceanState)
        self.component_forcing_class = decorator(OceanForcing)

    def _create_variable_registries(self) -> None:
        self.state_variable_registry = {}
        self.forcing_variable_registry = {}

        for target_registry, target_class in [
            (self.state_variable_registry, self.component_state_class),
            (self.forcing_variable_registry, self.component_forcing_class),
        ]:
            for name, _, dimensions, shape in target_class.typed_and_dimensioned_info():
                target_registry[name] = (shape, dimensions)

    def initialize(self):
        """Initialize ocean model fields."""
        ocn_idx = self.horizontal_grids["T"].bmask == self.mask_value
        nonocn_idx = self.horizontal_grids["T"].bmask != self.mask_value

        # Initialize mixed layer depth with latitudinal variation
        init_mixed_layer_depth = (
            self.mixed_layer_depth_max
            + (self.mixed_layer_depth_min - self.mixed_layer_depth_max)
            * jnp.cos(self.latitude_radian) ** 3
        )

        # Load or create initial SST
        if self.SST_clim_file is not None:
            print("SST climatology file. The given initial SST will be used.")
            print("SST climatology file: ", self.SST_clim_file)
            self.SST_clim = jnp.array(xr.open_dataset(self.SST_clim_file)["sst"])
            init_sea_surface_temperature = self.SST_clim[:, :, 0].copy()
        else:
            print("Boundary does not exist. Idealized initial SST will be used.")
            init_sea_surface_temperature = (
                positive_cosine_cubic_latitude_squared(self.latitude_radian) * 27.0
                + constants.freezing_point_K
            )

        # Apply mask
        init_sea_surface_temperature = init_sea_surface_temperature.at[nonocn_idx].set(
            default_land_surface_temperature
        )

        init_mixed_layer_dissolved_inorganic_carbon = jnp.zeros_like(init_sea_surface_temperature).at[ocn_idx].set(self.init_mixed_layer_dissolved_inorganic_carbon)
        init_deep_layer_dissolved_inorganic_carbon = jnp.array(self.init_deep_layer_dissolved_inorganic_carbon)

        # Validate mask consistency
        if jnp.sum(jnp.isnan(init_sea_surface_temperature)) == 0:
            print("grid.bmask and SST_clim do share the same mask.")
        else:
            raise Exception(
                "Warning: fmask_ocn and sea_surface_temperature_init do not share the same mask."
            )

        # Set relaxation time to infinity if no climatology
        if self.SST_clim_file is None:
            print("Notice: Climaology SST does not exist. Set relaxation time to inifinity.")
            self.relaxation_time = jnp.inf

        # Compute heat capacity and time factors for Euler backward scheme
        cd = (
            constants.ocean_density
            * constants.ocean_specific_heat_capacity
            * init_mixed_layer_depth
        )

        if self.forcing_method == "relaxation":
            tau = jnp.ones_like(cd) * self.relaxation_time
        else:
            tau = jnp.inf
        
        self.time_factor = (1.0 + self.timestep / tau) ** (-1)
        self.cd_factor = self.timestep / cd
        
        forcing = self.component_forcing_class.zeros()
        forcing = forcing.copy({
            "q_flux": forcing.q_flux   
        })

        return dict(
            state=self.component_state_class.zeros().copy({
                "mixed_layer_depth": init_mixed_layer_depth,
                "sea_surface_temperature": init_sea_surface_temperature,
                "mixed_layer_dissolved_inorganic_carbon": init_mixed_layer_dissolved_inorganic_carbon,
                "deep_layer_dissolved_inorganic_carbon": init_deep_layer_dissolved_inorganic_carbon,
            }),
            forcing=forcing,
            derived={
                "total_carbon_flux": jnp.array(0.0),
            },
        )

    def _create_step_function_body(self):
        """Create the step function for ocean model."""
        start_day_offset = self._compute_start_day_offset()
        ocn_idx = self.horizontal_grids["T"].bmask == self.mask_value
        nonocn_idx = self.horizontal_grids["T"].bmask != self.mask_value
        total_alkalinity = self.total_alkalinity
        area = self.horizontal_grids["T"].area
        exchange_timescale = self.mixed_deep_layer_exchange_time_scale
        V_deep = jnp.sum(area * ocn_idx) * self.deep_layer_depth

        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]
            new_sea_surface_temperature_anom = state.sea_surface_temperature
            total_heat_flux = forcing.total_heat_flux
            predictions = {}
            print(f"Using method: {self.forcing_method}")
            if self.forcing_method == "relaxation":
                sst_clim_beg = jnp.where(
                    ocn_idx,
                    self._interpolate_cyclic(state.sim_time, start_day_offset, self.SST_clim),
                    default_land_surface_temperature,
                )
                sst_clim_end = jnp.where(
                    ocn_idx,
                    self._interpolate_cyclic(state.sim_time + self.timestep, start_day_offset, self.SST_clim),
                    default_land_surface_temperature,
                )
                new_sea_surface_temperature_anom = state.sea_surface_temperature - sst_clim_beg
            elif self.forcing_method == "Qflux":
                snapshot_Qflux = jnp.where(
                    ocn_idx,
                    self._interpolate_cyclic(state.sim_time, start_day_offset, forcing.q_flux),
                    0.0,
                )
                total_heat_flux = total_heat_flux + snapshot_Qflux


            # Euler backward step
            new_sim_time = state.sim_time + self.timestep
            new_sea_surface_temperature_anom = self.time_factor * (
                new_sea_surface_temperature_anom
                + self.cd_factor * (- total_heat_flux)
            )

            co2_flux, pco2_seawater = compute_co2_flux(
                co2_volume_mixing_ratio = forcing.air_co2_volume_mixing_ratio,              # ppm
                surface_dry_air_pressure = 1.0,                                             # atm
                wind_10m = forcing.U10,                                                     # m/s
                dissolved_inorganic_carbon = state.mixed_layer_dissolved_inorganic_carbon*1e-3,  # mol/m^3 => M (mol/dm^3)
                total_alkalinity = total_alkalinity*1e-3,                                        # mol/m^3 => M (mol/dm^3)
                seawater_temperature = state.sea_surface_temperature,                        # K
                salinity = 35.0,                                                             # psu
            )
 
            co2_flux = co2_flux.at[nonocn_idx].set(0.0)

            # Exchange flux [mol/m²/s]: positive = carbon flows from deep box into mixed layer
            exchange_flux = (
                (state.deep_layer_dissolved_inorganic_carbon - state.mixed_layer_dissolved_inorganic_carbon)
                * state.mixed_layer_depth / exchange_timescale
            )
            exchange_flux = exchange_flux.at[nonocn_idx].set(0.0)

            # Mixed layer DIC: Euler forward, air-sea flux + deep-mixed exchange
            new_mixed_layer_dissolved_inorganic_carbon = state.mixed_layer_dissolved_inorganic_carbon + (
                self.timestep * (-co2_flux + exchange_flux) / state.mixed_layer_depth
            )

            # Deep box DIC: loses exactly what all mixed layer columns gain (mass conserving)
            new_deep_layer_dissolved_inorganic_carbon = state.deep_layer_dissolved_inorganic_carbon - (
                jnp.sum(exchange_flux * area) * self.timestep / V_deep
            )
           
            # Add climatology back
            new_sea_surface_temperature = new_sea_surface_temperature_anom
            if self.forcing_method == "relaxation":
                new_sea_surface_temperature += sst_clim_end
            
            # Apply land mask
            new_sea_surface_temperature = new_sea_surface_temperature.at[
                nonocn_idx
            ].set(default_land_surface_temperature)

            new_state = state.copy(
                {
                    "sea_surface_temperature": new_sea_surface_temperature,
                    "mixed_layer_dissolved_inorganic_carbon": new_mixed_layer_dissolved_inorganic_carbon,
                    "deep_layer_dissolved_inorganic_carbon": new_deep_layer_dissolved_inorganic_carbon,
                    "sim_time": new_sim_time,
                }
            )

            total_carbon_mixed_layer = jnp.sum(new_mixed_layer_dissolved_inorganic_carbon * area * state.mixed_layer_depth) * carbon_molecular_weight
            total_carbon_deep_layer = new_deep_layer_dissolved_inorganic_carbon * V_deep * carbon_molecular_weight
            total_carbon = total_carbon_mixed_layer + total_carbon_deep_layer
            total_carbon_flux = jnp.sum(co2_flux * area) * carbon_molecular_weight


            derived = dict(total_carbon_flux=total_carbon_flux)

            predictions = dict(
                state=new_state,
                forcing=dict(
                    total_heat_flux=total_heat_flux,
                ),
                bgc=dict(
                    air_co2_volume_mixing_ratio = forcing.air_co2_volume_mixing_ratio,
                    co2_flux = co2_flux,
                    total_carbon_mixed_layer = total_carbon_mixed_layer,
                    total_carbon_deep_layer = total_carbon_deep_layer,
                    total_carbon = total_carbon,
                    total_carbon_flux = total_carbon_flux,
                    pco2_seawater = pco2_seawater,
                ),
            )
            if self.forcing_method == "Qflux":
                predictions["forcing"]["q_flux"] = snapshot_Qflux

            return dict(
                state=new_state,
                forcing=forcing,
                derived=derived,
            ), stack_objects([predictions])

        return step_function

    def _create_xarray_data_vars(self, predictions) -> Dict[str, Any]:
        """Create xarray data variables for ocean output."""
        state = predictions["state"]
        forcing = predictions["forcing"]
        bgc = predictions["bgc"]
        T_grid_dims = ("time",) + self.horizontal_grids["T"].coordinate.dims
        data_vars = dict(
            sea_surface_temperature=(
                T_grid_dims,
                state["sea_surface_temperature"],
                {
                    "long_name": "Sea surface temperature",
                    "units": "K",
                }
            ),
            mixed_layer_depth=(
                T_grid_dims,
                state["mixed_layer_depth"],
                {
                    "long_name": "Mixed layer depth",
                    "units": "m",
                }
            ),
            mixed_layer_dissolved_inorganic_carbon=(
                T_grid_dims,
                state["mixed_layer_dissolved_inorganic_carbon"],
                {
                    "long_name": "Mixed layer dissolved inorganic carbon",
                    "units": "mol / m^3",
                }
            ),
            deep_layer_dissolved_inorganic_carbon=(
                ("time",),
                state["deep_layer_dissolved_inorganic_carbon"],
                {
                    "long_name": "Deep layer dissolved inorganic carbon",
                    "units": "mol / m^3",
                }
            ),
            total_heat_flux=(
                T_grid_dims,
                forcing["total_heat_flux"],
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m-2",
                    "positive": "upward",
                }
            ),
            co2_flux=(
                T_grid_dims,
                bgc["co2_flux"],
                {
                    "long_name": "Flux of co2",
                    "units": "mol / m^2 / s",
                    "positive": "upward",
                }
            ),
            pco2_seawater=(
                T_grid_dims,
                bgc["pco2_seawater"],
                {
                    "long_name": "pCO2 of seawater",
                    "units": "atm",
                }
            ),
            total_carbon_mixed_layer=(
                ("time",),
                bgc["total_carbon_mixed_layer"],
                {
                    "long_name": "Total carbon mass in the mixed layer",
                    "units": "kg",
                }
            ),
            total_carbon_deep_layer=(
                ("time",),
                bgc["total_carbon_deep_layer"],
                {
                    "long_name": "Total carbon mass in the deep ocean box",
                    "units": "kg",
                }
            ),
            total_carbon=(
                ("time",),
                bgc["total_carbon"],
                {
                    "long_name": "Total carbon mass in the ocean (mixed + deep)",
                    "units": "kg",
                }
            ),
            total_carbon_flux=(
                ("time",),
                bgc["total_carbon_flux"],
                {
                    "long_name": "Total carbon mass flux",
                    "units": "kg / s",
                    "positive": "upward",
                }
            ),
            air_co2_volume_mixing_ratio=(
                ("time",),
                bgc["air_co2_volume_mixing_ratio"],
                {
                    "long_name": "Atmosphere co2 mixing ratio",
                    "units": "ppm",
                }
            ),

        )

        if self.forcing_method == "Qflux":
            data_vars["q_flux"] = (
                T_grid_dims,
                forcing["q_flux"],
                {
                    "long_name": "Q-flux",
                    "units": "W m-2",
                    "positive": "Heating the ocean",
                }
            )

        return data_vars

    def get_info(self):
        return {
            'relaxation_time' : self.relaxation_time,
        }
