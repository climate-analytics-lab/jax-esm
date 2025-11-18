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

Translation from: https://github.com/samhatfield/speedy.f90/blob/master/source/land_model.f90
"""

from typing import Dict, Tuple, Any, List, Optional

import jax
import jax.numpy as jnp

from jax_esm import constants as constants
from jax_esm.utils.bulk_op import stack_objects

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

from jax_esm.components.base import (
    Component,
    ComponentConfig,
    create_component_state_class,
    create_field_group_class,
)


class LandModel(Component):
    """
    Slab land-surface model with:
    - Heat capacity-based temperature evolution
    - Snow depth and soil moisture from climatology
    - Separate treatment of soil and ice sheets
    
    Based on SPEEDY land model with prescribed climatological boundary conditions.
    """
    
    def __init__(
        self,
        config: ComponentConfig,
    ):
        """Initialize land surface model.
        
        Args:
            config: Component configuration with required params:
                - coords: Coordinate system with nodal_shape
                - geometry: Grid geometry 
                - boundaries: Optional boundary data with land masks
                - boundary_file: Optional path to NetCDF with land climatology
                - depth_soil: Soil layer depth in meters (default: 1.0)
                - depth_lice: Land-ice depth in meters (default: 5.0)
                - tdland: Dissipation timescale for anomalies in days (default: 40.0)
                - flandmin: Minimum land fraction for anomaly computation (default: 1/3)
        """
        
        super().__init__(config)

        # Extract configuration
        self.coords = config.params["coords"]
        self.geometry = config.params.get("geometry", None)
        
        self.start_dt = config.start_dt
        self.timestep = config.timestep
        self.substeps = config.substeps
        self.subtimestep = self.timestep / self.substeps
        
        # Physical parameters from Fortran defaults
        self.depth_soil = config.params.get("depth_soil", 1.0)  # m
        self.depth_lice = config.params.get("depth_lice", 5.0)  # m
        self.tdland = config.params.get("tdland", 40.0)  # days
        self.flandmin = config.params.get("flandmin", 1.0/3.0)
        
        # Heat capacities per m^2 (depth * volumetric_heat_capacity)
        # Fortran values: hcapl = depth_soil*2.50e+6, hcapli = depth_lice*1.93e+6
        self.hcapl = self.depth_soil * 2.50e6  # J/(m^2 K) for soil
        self.hcapli = self.depth_lice * 1.93e6  # J/(m^2 K) for land ice
        
        # Soil moisture parameters (not evolved, just for completeness)
        self.swcap = config.params.get("swcap", 0.30)  # Field capacity
        self.swwil = config.params.get("swwil", 0.17)  # Wilting point
        
        # Snow depth to snow cover conversion parameter
        self.sd2sc = config.params.get("sd2sc", 60.0)  # mm water equivalent
        
        # Land mask threshold
        self.land_threshold = config.params.get("land_threshold", 0.1)
        
        # Define state structure
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        
        self.component_state_class = create_component_state_class(
            prog_cls = create_field_group_class(
                cls_name = "state",
                fields = [
                    ("sim_time", float, ()),
                    ("T", float, D2_nodal_shape),  # Land surface temperature (K)
                ],
            ),
            phydata_cls = create_field_group_class(
                cls_name = "phydata",
                fields = [
                    ("heatflx", float, D2_nodal_shape),  # Heat flux into land (W/m^2)
                    ("snowd", float, D2_nodal_shape),    # Snow depth (mm water equiv)
                    ("soilw", float, D2_nodal_shape),    # Soil water availability (0-1)
                ],
            ),
        )
        
    def initialize(self):
        """Initialize land surface model state and climatology.
        
        Returns:
            Initial component state with land temperature, snow, and soil moisture
        """
        
        D3_nodal_shape = self.coords.nodal_shape
        D2_nodal_shape = D3_nodal_shape[1:]
        config = self.config
        
        # =========================================================================
        # Initialize land masks
        # =========================================================================
        
        if "boundaries" in config.params and config.params["boundaries"] is not None:
            print("Boundary exists. Using boundary file for land initialization.")
            print("Boundary file: ", config.params.get("boundary_file", "N/A"))
            
            boundaries = config.params["boundaries"]
            thrsh = self.land_threshold
            
            # Fractional land mask from boundaries
            fmask_raw = boundaries.fmask
            
            # Create binary and fractional land masks (Fortran: land_model_init lines 72-82)
            self.fmask_l = jnp.where(
                fmask_raw >= thrsh,
                jnp.where(fmask_raw > (1.0 - thrsh), 1.0, fmask_raw),
                0.0
            )
            
            self.bmask_l = jnp.where(fmask_raw >= thrsh, 1.0, 0.0)
            
            # Domain mask for anomaly computation (Fortran: lines 149-154)
            self.dmask = jnp.where(self.fmask_l >= self.flandmin, 1.0, 0.0)
            
        else:
            print("No boundary data. Using uniform land mask.")
            self.fmask_l = jnp.ones(D2_nodal_shape, dtype=jnp.float32)
            self.bmask_l = jnp.ones(D2_nodal_shape, dtype=jnp.float32)
            self.dmask = jnp.ones(D2_nodal_shape, dtype=jnp.float32)
        
        # =========================================================================
        # Load climatology fields
        # =========================================================================
        
        if "boundary_file" in config.params and config.params["boundary_file"] is not None:
            # Load monthly climatology from NetCDF
            ds = xr.open_dataset(config.params["boundary_file"])
            
            # Land surface temperature climatology (12 months)
            if "stl" in ds:
                self.stl_clim = jnp.array(ds["stl"].values, dtype=jnp.float32)
            else:
                print("Warning: 'stl' not in boundary file, using idealized temperature")
                self.stl_clim = self._idealized_land_temperature(D2_nodal_shape)
            
            # Snow depth climatology (12 months, mm water equivalent)
            if "snowd" in ds:
                self.snowd_clim = jnp.array(ds["snowd"].values, dtype=jnp.float32)
            else:
                print("Warning: 'snowd' not in boundary file, using zero snow")
                self.snowd_clim = jnp.zeros((12,) + D2_nodal_shape, dtype=jnp.float32)
            
            # Soil water availability climatology (12 months, 0-1)
            if "soilw" in ds:
                self.soilw_clim = jnp.array(ds["soilw"].values, dtype=jnp.float32)
            else:
                print("Warning: 'soilw' not in boundary file, using uniform soil moisture")
                self.soilw_clim = jnp.ones((12,) + D2_nodal_shape, dtype=jnp.float32) * 0.5
                
        else:
            # Create idealized climatology
            print("No boundary file specified. Using idealized land climatology.")
            self.stl_clim = self._idealized_land_temperature(D2_nodal_shape)
            self.snowd_clim = jnp.zeros((12,) + D2_nodal_shape, dtype=jnp.float32)
            self.soilw_clim = jnp.ones((12,) + D2_nodal_shape, dtype=jnp.float32) * 0.5
        
        # =========================================================================
        # Compute heat capacity and dissipation time fields
        # =========================================================================
        
        # Get albedo to discriminate soil vs ice (Fortran: lines 157-163)
        if "boundaries" in config.params and hasattr(config.params["boundaries"], "alb0"):
            alb0 = config.params["boundaries"].alb0
        else:
            # Default: assume all soil (low albedo)
            alb0 = jnp.ones(D2_nodal_shape, dtype=jnp.float32) * 0.2
        
        # rhcapl = timestep / heat_capacity (Fortran uses delt which is timestep in seconds)
        # Use ice heat capacity where albedo >= 0.4, else soil
        self.rhcapl = jnp.where(
            alb0 < 0.4,
            self.timestep / self.hcapl,   # Soil
            self.timestep / self.hcapli   # Ice
        )
        
        # cdland = dissipation coefficient (Fortran: line 165)
        # cdland = dmask * tdland / (1 + dmask * tdland)
        # Convert tdland from days to timesteps
        tdland_timesteps = self.tdland * 86400.0 / self.timestep
        self.cdland = self.dmask * tdland_timesteps / (1.0 + self.dmask * tdland_timesteps)
        
        # =========================================================================
        # Initialize land surface temperature from climatology
        # =========================================================================
        
        # Get initial month (0-11 for indexing)
        init_month = self.start_dt.month - 1
        
        # Initial land surface temperature from climatology
        init_T = self.stl_clim[init_month, :, :]
        
        # Apply land mask (set ocean points to reasonable value)
        init_T = jnp.where(self.bmask_l > 0, init_T, 273.15 + 15.0)
        
        return self.component_state_class.zeros().copy(
            prog_kwargs = dict(
                T = init_T,
                sim_time = 0.0,
            ),
            phydata_kwargs = dict(
                heatflx = jnp.zeros(D2_nodal_shape, dtype=jnp.float32),
                snowd = self.snowd_clim[init_month, :, :],
                soilw = self.soilw_clim[init_month, :, :],
            ),
        )
    
    def _idealized_land_temperature(self, shape: Tuple[int, int]) -> jnp.ndarray:
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
        
        return T_clim.astype(jnp.float32)
    
    def gen_step_fn(self):
        """Generate JIT-compiled step function for land model.
        
        Returns:
            JIT-compiled function: (coupled_state, time_index) -> (new_state, predictions)
        """
        
        # Compute reference for climatology indexing
        ref_dt = pd.Timestamp(year=self.start_dt.year, month=1, day=1)
        start_dt_offset = jnp.int32(jnp.floor((self.start_dt - ref_dt) / pd.Timedelta(days=1)))
        
        @jax.jit
        def step_fn(cpl, t):
            """Land model time step.
            
            Implements the slab land model from Fortran run_land_model subroutine:
            1. Interpolate climatology to current day
            2. Compute temperature anomaly w.r.t. climatology
            3. Evolve anomaly with heat flux forcing and dissipation
            4. Add climatology to get final temperature
            
            Args:
                cpl: Coupled state with land component
                t: Time index
                
            Returns:
                Tuple of (new_land_state, predictions_dict)
            """
            
            # =====================================================================
            # Time and climatology management
            # =====================================================================
            
            # Days since start
            days_after_start = jnp.floor(cpl.lnd.prog.sim_time / 86400.0).astype(jnp.int32)
            
            # Current day of year for climatology lookup
            current_day = start_dt_offset + days_after_start
            
            # Get current month (0-11) for climatology - use simple division
            # This matches Fortran's forin5/forint interpolation approach
            current_month_idx = jnp.mod(current_day // 30, 12).astype(jnp.int32)
            next_month_idx = jnp.mod(current_month_idx + 1, 12).astype(jnp.int32)
            
            # Linear interpolation weight within month
            day_in_month = jnp.mod(current_day, 30)
            month_weight = day_in_month / 30.0
            
            # Interpolate land surface temperature climatology
            stl_clim_current = (
                (1.0 - month_weight) * self.stl_clim[current_month_idx, :, :] +
                month_weight * self.stl_clim[next_month_idx, :, :]
            )
            
            # Interpolate snow depth climatology
            snowd_clim_current = (
                (1.0 - month_weight) * self.snowd_clim[current_month_idx, :, :] +
                month_weight * self.snowd_clim[next_month_idx, :, :]
            )
            
            # Interpolate soil water climatology
            soilw_clim_current = (
                (1.0 - month_weight) * self.soilw_clim[current_month_idx, :, :] +
                month_weight * self.soilw_clim[next_month_idx, :, :]
            )
            
            # =====================================================================
            # Land surface temperature evolution (Fortran: run_land_model)
            # =====================================================================
            
            # Get heat flux from flux component (positive downward into land)
            # Note: flux component convention is positive upward, so we negate
            # If no flux component, assume zero flux
            if hasattr(cpl, 'flx') and cpl.flx is not None:
                heatflx = -cpl.flx.phydata.heatflx
            else:
                heatflx = jnp.zeros_like(cpl.lnd.prog.T)
            
            # Temperature anomaly w.r.t. climatology (Fortran: line 204)
            T_anom = cpl.lnd.prog.T - stl_clim_current
            
            # Time evolution of temperature anomaly (Fortran: line 207)
            # tanom = cdland * (tanom + rhcapl * hfluxn)
            # This is an implicit scheme: (1 - cdland) * T_anom + cdland * rhcapl * hfluxn
            new_T_anom = self.cdland * (T_anom + self.rhcapl * heatflx)
            
            # Full surface temperature (Fortran: line 210)
            new_T = new_T_anom + stl_clim_current
            
            # Apply land mask
            new_T = jnp.where(self.bmask_l > 0, new_T, 273.15 + 15.0)
            
            # Update simulation time
            new_sim_time = cpl.lnd.prog.sim_time + self.timestep
            
            # =====================================================================
            # Create new state
            # =====================================================================
            
            new_state = cpl.lnd.copy(
                prog_kwargs = dict(
                    T = new_T,
                    sim_time = new_sim_time,
                ),
                phydata_kwargs = dict(
                    heatflx = heatflx,
                    snowd = snowd_clim_current,
                    soilw = soilw_clim_current,
                ),
            )
            
            # Return new state and predictions for output
            return new_state, stack_objects([dict(
                prog=new_state.prog,
                phydata=new_state.phydata,
            )])
        
        return step_fn
    
    def predictions_to_xarray(
        self,
        predictions: Dict[str, Any],
    ) -> xr.Dataset:
        """Convert predictions to xarray Dataset.
        
        Args:
            predictions: Dictionary with 'prog' and 'phydata' fields from model run
            
        Returns:
            xarray Dataset with land surface variables and coordinates
        """
        
        prog = predictions["prog"]
        phydata = predictions["phydata"]
        
        ds = xr.Dataset(
            data_vars = dict(
                T = (["time", "lon", "lat"], prog.T, {
                    "long_name": "Land surface temperature",
                    "units": "K",
                }),
                heatflx = (["time", "lon", "lat"], phydata.heatflx, {
                    "long_name": "Surface heat flux into land",
                    "units": "W m-2",
                    "positive": "downward",
                }),
                snowd = (["time", "lon", "lat"], phydata.snowd, {
                    "long_name": "Snow depth (water equivalent)",
                    "units": "mm",
                }),
                soilw = (["time", "lon", "lat"], phydata.soilw, {
                    "long_name": "Soil water availability",
                    "units": "1",
                }),
            ),
            coords = dict(
                time = (["time"], jnp.atleast_1d(prog.sim_time).flatten(), {
                    "long_name": "Simulation time",
                    "units": "seconds since initialization",
                }),
            ),
            attrs = dict(
                description = "SPEEDY-based slab land surface model output",
                depth_soil = f"{self.depth_soil} m",
                depth_lice = f"{self.depth_lice} m",
                tdland = f"{self.tdland} days",
            ),
        )
        
        return ds
