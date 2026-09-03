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
from typing import Any, Sequence

import jax.numpy as jnp
import jax_datetime as jdt
import tree_math
import xarray as xr

from jem.components.slab.base import _DEFAULT_START_DATETIME, SlabModelBase
from jem.components.slab.grid import SlabGrid
from jem.utils.bulk_op import stack_objects


@tree_math.struct
class LandState:
    sim_time: jnp.ndarray
    land_surface_temperature: jnp.ndarray
    snowc: jnp.ndarray
    soilw: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        sim_time=None,
        land_surface_temperature=None,
        snowc=None,
        soilw=None,
        n_soil_layers=2,
    ):
        return cls(
            sim_time if sim_time is not None else jnp.zeros(()),
            land_surface_temperature if land_surface_temperature is not None else jnp.zeros(shape),
            snowc if snowc is not None else jnp.zeros(shape),
            soilw if soilw is not None else jnp.zeros(shape + (n_soil_layers,)),
        )


@tree_math.struct
class LandForcing:
    total_heat_flux: jnp.ndarray
    precipitation: jnp.ndarray     # kg/m^2/s => equivalent to mm/s if density = 1000 kg/m^3

    @classmethod
    def zeros(cls, shape, total_heat_flux=None, precipitation=None):
        return cls(
            total_heat_flux if total_heat_flux is not None else jnp.zeros(shape),
            precipitation if precipitation is not None else jnp.zeros(shape),
        )


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
        grid: SlabGrid,
        start_datetime: jdt.Datetime = _DEFAULT_START_DATETIME,
        timestep: float = 86400.0,
        land_clim_file: str | None = None,
        depth_soil: Sequence[float] = (0.1, 0.9),
        depth_lice: float = 5.0,
        tdland: float = 40.0 * 86400.0,
        flandmin: float = 1.0/3.0,
        tau_drain: Sequence[float] = (5.0 * 86400.0, 60.0 * 86400.0),
        swcap: Sequence[float] = (0.30, 0.30),
        swwil: Sequence[float] = (0.17, 0.17),
        land_threshold: float = 0.1,
        calendar: str = "365_day",
    ):
        """Initialize land surface model.

        Args:
            grid: The model's grid. See jem.components.slab.grid.SlabGrid.
            start_datetime: Start datetime for simulation
            timestep: Model timestep in seconds
            land_clim_file: Optional path to land climatology NetCDF file
            depth_soil: Per-layer soil bucket depth in meters, one entry per
                soil moisture layer (default: (0.1, 0.9), a thin surface
                layer and a thick deep layer summing to 1.0 m). Land surface
                temperature uses the combined depth of all layers.
            depth_lice: Land-ice depth in meters (default: 5.0)
            tdland: Dissipation timescale for anomalies in seconds (default: 40 days)
            flandmin: Minimum land fraction for anomaly computation (default: 1/3)
            tau_drain: Per-layer soil moisture drainage timescale in seconds
                (default: (5, 60) days). Layer 0 drains into layer 1; layer 1
                drains out as deep drainage.
            swcap: Per-layer field capacity, unitless 0-1 (default: (0.30, 0.30)).
                Currently unused by the bucket dynamics.
            swwil: Per-layer wilting point, unitless 0-1 (default: (0.17, 0.17)).
                Currently unused by the bucket dynamics.
            land_threshold: Land mask threshold (default: 0.1)
        """
        super().__init__(
            name="LandModel",
            grid=grid,
            start_datetime=start_datetime,
            timestep=timestep,
            calendar=calendar,
        )

        self.land_clim_file = land_clim_file

        # Physical parameters from Fortran defaults
        self.depth_soil = jnp.asarray(depth_soil)  # m, per soil moisture layer
        self.n_soil_layers = self.depth_soil.shape[0]
        self.depth_lice = depth_lice  # m
        self.tdland = tdland  # seconds
        self.flandmin = flandmin
        self.land_threshold = land_threshold
        self.tau_drain = jnp.asarray(tau_drain)  # seconds, per soil moisture layer
        self.rho_water = 1000.0  # kg/m^3

        # Heat capacities per m^2 (depth * volumetric_heat_capacity)
        # Fortran values: hcapl = depth_soil*2.50e+6, hcapli = depth_lice*1.93e+6
        # Temperature is single-layer, spanning the combined depth of all soil moisture layers.
        self.hcapl = jnp.sum(self.depth_soil) * 2.50e6  # J/(m^2 K) for soil
        self.hcapli = self.depth_lice * 1.93e6  # J/(m^2 K) for land ice

        # Soil moisture parameters (not evolved, just for completeness)
        self.swcap = jnp.asarray(swcap)  # Field capacity, per soil moisture layer
        self.swwil = jnp.asarray(swwil)  # Wilting point, per soil moisture layer

        # Snow depth to snow cover conversion parameter
        self.sd2sc = 60.0  # mm water equivalent

        self.validate()
 
    def validate(self):
        super().validate()

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
        fmask_raw = self.grid.fractional_mask
        D2_nodal_shape = self.grid.shape
        
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
            "state": LandState.zeros(
                D2_nodal_shape,
                land_surface_temperature=init_T,
                snowc=jnp.minimum(1.0, self.snowd_clim[:, :, init_time_idx] / self.sd2sc),
                soilw=jnp.zeros(D2_nodal_shape + (self.n_soil_layers,)),
            ),
            "forcing": LandForcing.zeros(D2_nodal_shape),
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

            # =====================================================================
            # Land surface soil moisture evolution (two-layer leaky bucket)
            # =====================================================================
            # Layer 0 (surface): receives precipitation, drains into layer 1.
            # Layer 1 (deep): receives layer 0's drainage, drains out as deep drainage.
            W1 = state.soilw[..., 0]
            W2 = state.soilw[..., 1]

            drain1 = W1 / self.tau_drain[0]
            inflow2 = (self.depth_soil[0] / self.depth_soil[1]) * drain1

            new_W1 = jnp.maximum(
                0.0,
                W1 + (forcing.precipitation / (self.rho_water * self.depth_soil[0]) - drain1) * self.timestep,
            )
            new_W2 = jnp.maximum(
                0.0,
                W2 + (inflow2 - W2 / self.tau_drain[1]) * self.timestep,
            )

            new_soilw = jnp.stack([new_W1, new_W2], axis=-1)
            new_soilw = jnp.where(self.bmask_l[..., None] > 0, new_soilw, 0.0)
            
            # Update simulation time - keep as float32
            new_sim_time = state.sim_time + self.timestep
            
            # =====================================================================
            # Create new state
            # =====================================================================
            
            new_state = state.replace(
                sim_time=new_sim_time,
                land_surface_temperature=new_T,
                snowc=jnp.minimum(1.0, snowd_clim / self.sd2sc),
                soilw=new_soilw, #soilw_clim,
            )
            
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
        T_grid_dims = ("time",) + self.grid.dims
        
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
                T_grid_dims + ("soil_layer",),
                state.soilw,
                {
                    "long_name": "Soil water content by layer (0=surface, 1=deep)",
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
            "precipitation": (
                T_grid_dims,
                forcing.precipitation,
                {
                    "long_name": "Precipitation rate forcing",
                    "units": "mm s-1",
                    "positive": "downward",
                }
            ),
        }

    def _create_xarray_global_attributes(self) -> dict[str, Any]:
        return {
            "description": "SPEEDY-based slab land surface model output",
            "depth_soil": f"{list(self.depth_soil)} m",
            "depth_lice": f"{self.depth_lice} m",
            "tdland": f"{self.tdland} days",
            "tau_drain": f"{[t / 86400.0 for t in self.tau_drain]} days",
        }
