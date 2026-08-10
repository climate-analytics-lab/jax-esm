"""Land surface model component - translated from Speedy Fortran.

This module implements a slab land-surface model with:
- Temperature evolution with heat capacity and dissipation
- Snow depth climatology
- Soil water availability climatology
- Land/ice-sheet discrimination based on albedo

Physics based on SPEEDY (Simplified Parameterizations, primiTivE-Equation DYnamics):
Molteni, F. (2003). Atmospheric simulations using a GCM with simplified 
physical parametrizations. I: model climatology and variability in 
multi-decadal experiments. Climate Dynamics, 20(2-3), 175-191.

Programmer: Aya Lalou

Translation from: https://github.com/samhatfield/speedy.f90/blob/master/source/land_model.f90
"""
from typing import Annotated, Any

import jax.numpy as jnp
import jax_datetime as jdt
import xarray as xr

from jem.components.slab.base import SlabModelBase
from jem.utils import data_structure
from jem.utils.bulk_op import stack_objects


@data_structure.typed_and_dimensioned
class LandState:
    sim_time: Annotated[float, (), "zero_dimensional"]
    land_surface_temperature: Annotated[
        float, ("longitude", "latitude"), "two_dimensional"
    ]
    snowc: Annotated[float, ("longitude", "latitude"), "two_dimensional"]
    soilw: Annotated[float, ("longitude", "latitude"), "two_dimensional"]


@data_structure.typed_and_dimensioned
class LandForcing:
    total_heat_flux: Annotated[float, ("longitude", "latitude"), "two_dimensional"]

class SlabLandModel(SlabModelBase):
    """
    Slab land-surface model with:
    - Heat capacity-based temperature evolution
    - Snow depth and soil moisture from climatology
    - Separate treatment of soil and ice sheets
    
    Based on SPEEDY land model with prescribed climatological boundary conditions.
    """
    
    def __init__(
        self,
        grid_specification: str = "JCM::T31",
        start_datetime: jdt.Datetime = jdt.to_datetime("2001-01-01"),
        timestep: float = 86400.0,
        topography_file: str | None = None,
        mask_file: str | None = None,
        land_clim_file: str | None = None,
        depth_soil: float = 1.0,
        depth_lice: float = 5.0,
        tdland: float = 40.0 * 86400.0,
        flandmin: float = 1.0/3.0,
        land_threshold: float = 0.1,
        calendar: str = "365_day",
    ):
        """Initialize land surface model.
        
        Args:
            grid_specification: Grid specification string (e.g., "JCM::T31")
            start_datetime: Start datetime for simulation
            timestep: Model timestep in seconds
            topography_file: Optional path to topography NetCDF file
            mask_file: Optional path to mask NetCDF file
            land_clim_file: Optional path to land climatology NetCDF file
            depth_soil: Soil layer depth in meters (default: 1.0)
            depth_lice: Land-ice depth in meters (default: 5.0)
            tdland: Dissipation timescale for anomalies in seconds (default: 40 days)
            flandmin: Minimum land fraction for anomaly computation (default: 1/3)
            land_threshold: Land mask threshold (default: 0.1)
        """
        
        super().__init__(
            name="LandModel",
            grid_specification=grid_specification,
            start_datetime=start_datetime,
            timestep=timestep,
            topography_file=topography_file,
            mask_file=mask_file,
            calendar=calendar,
        )

        self.land_clim_file = land_clim_file
        
        # Physical parameters from Fortran defaults
        self.depth_soil = depth_soil  # m
        self.depth_lice = depth_lice  # m
        self.tdland = tdland  # seconds
        self.flandmin = flandmin
        self.land_threshold = land_threshold
        
        # Heat capacities per m^2 (depth * volumetric_heat_capacity)
        # Fortran values: hcapl = depth_soil*2.50e+6, hcapli = depth_lice*1.93e+6
        self.hcapl = self.depth_soil * 2.50e6  # J/(m^2 K) for soil
        self.hcapli = self.depth_lice * 1.93e6  # J/(m^2 K) for land ice
        
        # Soil moisture parameters (not evolved, just for completeness)
        self.swcap = 0.30  # Field capacity
        self.swwil = 0.17  # Wilting point
        
        # Snow depth to snow cover conversion parameter
        self.sd2sc = 60.0  # mm water equivalent

        self.validate()
 
    def validate(self):
        super().validate()

    def _create_state_and_forcing_classes(self) -> None:
        """Create state and forcing classes for land model."""

        decorator = data_structure.build_dataclass_from_typed_and_dimensioned({"two_dimensional": self.grid_shape})
        self.component_state_class = decorator(LandState)
        self.component_forcing_class = decorator(LandForcing)

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
        """Initialize land surface model state and climatology.
        
        Returns:
            Initial component state with land temperature, snow, and soil moisture
        """
        
        # =========================================================================
        # Initialize land masks from domain
        # =========================================================================
        
        print("Initializing land masks from domain...")
        
        # Use domain masks
        thrsh = self.land_threshold
        fmask_raw = self.horizontal_grids["T"].fmask
        D2_nodal_shape = self.horizontal_grids["T"].shape
        
        # Create binary and fractional land masks (Fortran: land_model_init lines 72-82)
        self.fmask_l = jnp.where(
            fmask_raw >= thrsh,
            jnp.where(fmask_raw > (1.0 - thrsh), 1.0, fmask_raw),
            0.0
        )
        
        self.bmask_l = jnp.where(fmask_raw >= thrsh, 1.0, 0.0)
        
        # Domain mask for anomaly computation (Fortran: lines 149-154)
        self.dmask = jnp.where(self.fmask_l >= self.flandmin, 1.0, 0.0)
        
        print(f"Total land grid count: {self.bmask_l.sum()}")
        
        # =========================================================================
        # Load climatology fields
        # =========================================================================
        
        if self.land_clim_file is not None:
            print(f"Loading land climatology from: {self.land_clim_file}")
            # Load monthly climatology from NetCDF
            ds = xr.open_dataset(self.land_clim_file)
            
            # Land surface temperature climatology
            # Note: Data format is (lon, lat, time) to match JCM nodal ordering
            if "stl" in ds:
                stl_data = jnp.array(ds["stl"].values)
                print(f"Loaded stl climatology with shape: {stl_data.shape}")
                # Store as (lon, lat, time) to match JCM nodal ordering
                self.stl_clim = stl_data
                self.n_clim_steps = stl_data.shape[2]  # Number of time steps
            else:
                print("Warning: 'stl' not in boundary file, using idealized temperature")
                ideal_clim = self._idealized_land_temperature(D2_nodal_shape)
                # Transpose from (12, lat, lon) to (lat, lon, 12)
                self.stl_clim = jnp.transpose(ideal_clim, (1, 2, 0))
                self.n_clim_steps = 12

            # Snow depth climatology (mm water equivalent)
            if "snowd" in ds:
                print("Notice: found 'snowd' in boundary file.")
                self.snowd_clim = jnp.array(ds["snowd"].values)
            elif "snowc" in ds:
                # Tien-Yiao [2016/03/01]: jax-gcm forcing file seems to use snowc instead of snowd
                self.snowd_clim = jnp.array(ds["snowc"].values)
            else:
                print("Warning: cannot find 'snowd' or 'snowc' not in boundary file, using zero snow")
                self.snowd_clim = jnp.zeros(D2_nodal_shape + (self.n_clim_steps,))
            
            # Soil water availability climatology (0-1)
            if "soilw" in ds:
                print("Notice: found 'soilw' in boundary file.")
                self.soilw_clim = jnp.array(ds["soilw"].values)
            elif "soilw_am" in ds:
                # Tien-Yiao [2016/03/01]: jax-gcm forcing file use soilw_am instead of soilw
                print("Notice: found 'snowd_am' in boundary file.")
                self.soilw_clim = jnp.array(ds["soilw_am"].values)
            else:
                print("Warning: 'soilw' not in boundary file, using uniform soil moisture")
                self.soilw_clim = jnp.ones(D2_nodal_shape + (self.n_clim_steps,)) * 0.5
                
        else:
            # Create idealized climatology
            print("No boundary file specified. Using idealized land climatology.")
            ideal_clim = self._idealized_land_temperature(D2_nodal_shape)
            # Transpose from (12, lat, lon) to (lat, lon, 12)
            self.stl_clim = jnp.transpose(ideal_clim, (1, 2, 0))
            self.n_clim_steps = 12
            self.snowd_clim = jnp.zeros(D2_nodal_shape + (12,))
            self.soilw_clim = jnp.ones(D2_nodal_shape + (12,)) * 0.5
        
        # =========================================================================
        # Compute heat capacity and dissipation time fields
        # =========================================================================
        
        # Get albedo to discriminate soil vs ice (Fortran: lines 157-163)
        # Default: assume all soil (low albedo) for now
        # TODO: Could load from topography file if available
        alb0 = jnp.ones(D2_nodal_shape) * 0.2
        
        # rhcapl = timestep / heat_capacity (Fortran uses delt which is timestep in seconds)
        # Use ice heat capacity where albedo >= 0.4, else soil
        self.rhcapl = jnp.where(
            alb0 < 0.4,
            self.timestep / self.hcapl,   # Soil
            self.timestep / self.hcapli   # Ice
        )
        
        # cdland = dissipation coefficient (Fortran: line 165)
        # cdland = dmask * tdland / (1 + dmask * tdland)
        tdland_timesteps = self.tdland / self.timestep
        self.cdland = (self.dmask * tdland_timesteps / (1.0 + self.dmask * tdland_timesteps))
        
        print(f"Heat capacity range: {self.rhcapl.min():.2e} - {self.rhcapl.max():.2e}")
        print(f"Dissipation coefficient range: {self.cdland.min():.3f} - {self.cdland.max():.3f}")
        
        # =========================================================================
        # Initialize land surface temperature from climatology
        # =========================================================================
        
        # Get initial time index (0-based)
        # For daily data, use day of year; for monthly, use month
        if self.n_clim_steps > 100:  # Assume daily data
            init_time_idx = self.start_datetime.to_pydatetime().timetuple().tm_yday - 1
        else:  # Assume monthly data
            init_time_idx = self.start_datetime.to_pydatetime().month - 1
        
        # Initial land surface temperature from climatology (lon, lat, time)
        init_T = self.stl_clim[:, :, init_time_idx]
        
        # Apply land mask (set ocean points to reasonable value)
        init_T = jnp.where(self.bmask_l > 0, init_T, 273.15 + 15.0)
        
        print(f"Initial land temperature range: {init_T.min():.2f} - {init_T.max():.2f} K")
        
        return {
            "state": self.component_state_class.zeros().copy({
                "land_surface_temperature" : init_T,
                "sim_time" : 0,
                "heatflx" : jnp.zeros(D2_nodal_shape),
                "snowc" : jnp.minimum(1.0, self.snowd_clim[:, :, init_time_idx] / self.sd2sc),
                "soilw" : self.soilw_clim[:, :, init_time_idx],
            }),
            "forcing": self.component_forcing_class.zeros()
        }
    
    def _idealized_land_temperature(self, shape: tuple[int, int]) -> jnp.ndarray:
        """Create idealized land temperature climatology.
        
        Args:
            shape: (nlat, nlon) shape for temperature field
            
        Returns:
            Monthly land temperature climatology (12, nlat, nlon) in Kelvin
        """
        nlat, nlon = shape
        
        # Create latitude array
        lat = jnp.linspace(-90, 90, nlat)
        lat_rad = jnp.deg2rad(lat)
        
        # Base temperature with latitude dependence
        base_T = 273.15 + 25.0 * jnp.cos(lat_rad)
        
        # Add seasonal cycle (12 months)
        months = jnp.arange(12)
        seasonal_phase = 2 * jnp.pi * (months - 2) / 12.0  # Peak in March
        
        # Seasonal amplitude stronger at mid-latitudes
        seasonal_amp = 15.0 * jnp.sin(jnp.abs(lat_rad))**2
        
        # Combine: (12, nlat)
        T_lat = base_T[None, :] + seasonal_amp[None, :] * jnp.cos(seasonal_phase[:, None])
        
        # Broadcast to (12, nlat, nlon)
        T_clim = jnp.broadcast_to(T_lat[:, :, None], (12, nlat, nlon))
        
        return T_clim
    
    def _create_step_function_body(self):
        """Generate step function for land model.
        
        Args:
            jitted: Whether to JIT-compile the step function
            
        Returns:
            Step function with signature: (state, forcing, t) -> (new_state, predictions)
        """
        
        start_day_offset = self._compute_start_day_offset()
        
        def step_function(carry, t):
            """Land model time step.
            
            Implements the slab land model from Fortran run_land_model subroutine:
            1. Interpolate climatology to current day
            2. Compute temperature anomaly w.r.t. climatology
            3. Evolve anomaly with heat flux forcing and dissipation
            4. Add climatology to get final temperature
            
            Args:
                state: Land component state
                forcing: Land component forcing
                t: Current simulation time in seconds
                
            Returns:
                Tuple of (new_state, predictions_dict)
            """
            
            state = carry["state"]
            forcing = carry["forcing"]
            
            # =====================================================================
            # Time and climatology management
            # =====================================================================
            
            # Linearly interpolate climatology at current and end-of-step time
            stl_clim_beg  = self._interpolate_cyclic(state.sim_time,               start_day_offset, self.stl_clim)
            stl_clim_end  = self._interpolate_cyclic(state.sim_time + self.timestep, start_day_offset, self.stl_clim)
            snowd_clim    = self._interpolate_cyclic(state.sim_time,               start_day_offset, self.snowd_clim)
            soilw_clim    = self._interpolate_cyclic(state.sim_time,               start_day_offset, self.soilw_clim)
            
            # =====================================================================
            # Land surface temperature evolution (Fortran: run_land_model)
            # =====================================================================
            
            # Get heat flux from forcing (positive downward into land)
            # Negate because atmosphere uses positive upward convention
            heatflx = - forcing.total_heat_flux
            
            # Temperature anomaly w.r.t. climatology (Fortran: line 204)
            T_anom = (state.land_surface_temperature - stl_clim_beg)
            
            # Time evolution of temperature anomaly (Fortran: line 207)
            # tanom = cdland * (tanom + rhcapl * hfluxn)
            # This is an implicit scheme: (1 - cdland) * T_anom + cdland * rhcapl * hfluxn
            new_T_anom = (self.cdland * (T_anom + self.rhcapl * heatflx))
            
            # Full surface temperature (Fortran: line 210)
            new_T = (new_T_anom + stl_clim_end)
            
            # Apply land mask
            new_T = jnp.where(self.bmask_l > 0, new_T, 273.15 + 15.0)
            
            # Update simulation time - keep as float32
            new_sim_time = state.sim_time + self.timestep
            
            # =====================================================================
            # Create new state
            # =====================================================================
            
            new_state = state.copy({
                "sim_time" : new_sim_time,
                "land_surface_temperature" : new_T,
                "snowc" : jnp.minimum(1.0, snowd_clim / self.sd2sc),
                "soilw" : soilw_clim,
            })
            
            # Return new state and predictions for output
            return (
                {
                    "state": new_state, 
                    "forcing": forcing,
                }, stack_objects([{
                    "state": new_state,
                    "forcing": forcing,
                }])
            )
        
        return step_function
    
    def _create_xarray_data_vars(
        self,
        predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert predictions to xarray Dataset.
        
        Args:
            predictions: Dictionary with 'state' and 'forcing' fields from model run
            
        Returns:
            xarray Dataset with land surface variables and coordinates
        """
        
        state = predictions["state"]
        forcing = predictions["forcing"]
        T_grid_dims = ("time",) + self.horizontal_grids["T"].coordinate.dims
        
        return { 
            "land_surface_temperature": (
                T_grid_dims,
                state.land_surface_temperature, 
                {
                    "long_name": "Land surface temperature",
                    "units": "K",
                }
            ),
            "snowc": (
                T_grid_dims, state.snowc,
                {
                    "long_name": "Snow cover fraction",
                    "units": "1",
                }
            ),
            "soilw": (
                T_grid_dims,
                state.soilw,
                {
                    "long_name": "Soil water availability",
                    "units": "1",
                }
            ),
            "total_heat_flux": (
                T_grid_dims,
                forcing.total_heat_flux, 
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m-2",
                    "positive": "upward",
                }
            ),
        }

    def _create_xarray_global_attributes(self) -> dict[str, Any]:
        return {
            "description": "SPEEDY-based slab land surface model output",
            "depth_soil": f"{self.depth_soil} m",
            "depth_lice": f"{self.depth_lice} m",
            "tdland": f"{self.tdland} days",
        }
