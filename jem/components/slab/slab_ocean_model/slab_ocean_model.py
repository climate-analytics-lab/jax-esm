"""Slab ocean model component."""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import tree_math
import xarray as xr

from jem import constants
from jem.components.slab.base import _DEFAULT_START_DATETIME, SlabModelBase
from jem.components.slab.grid import SlabGrid
from jem.utils.bulk_op import stack_objects
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared

default_land_surface_temperature = 288.15

# Dimension names a monthly climatology file may use, per axis, in the order
# the loader normalises them to: (longitude, latitude, time).
_CLIMATOLOGY_DIM_ALIASES = (
    ("lon", "longitude"),
    ("lat", "latitude"),
    ("time",),
)

_MONTHS_PER_YEAR = 12

# Tolerance for matching a climatology file's coordinates against the grid.
# The jcm T30 climatology that ships with jax-gcm stores its Gaussian
# latitudes rounded to three decimal places -- up to ~5e-4 degrees away from
# the exact Gaussian abscissae the grid computes -- so the tolerance has to be
# looser than that rounding. It is still two orders of magnitude tighter than
# the ~0.4 degree spacing of the finest grid JEM supports, so a file written
# on a genuinely different grid is rejected rather than silently regridded.
_COORDINATE_TOLERANCE_DEGREES = 1e-3


def _resolve_dim_name(
    dims: tuple[str, ...],
    aliases: tuple[str, ...],
    path,
    var: str,
) -> str:
    """Return which of `aliases` names one of `dims`."""
    for alias in aliases:
        if alias in dims:
            return alias
    raise ValueError(
        f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has dimensions "
        f"{dims!r}, none of which is one of {aliases!r}."
    )


def _grid_axes_degrees(grid: SlabGrid, path) -> tuple[np.ndarray, np.ndarray]:
    """Return the grid's 1-D (longitude, latitude) axes in degrees.

    A SlabGrid only stores 2-D radian fields, so the 1-D axes a climatology
    file is written on have to be recovered from them. That is only
    meaningful for a separable lat-lon grid, which is checked here rather
    than left to surface later as a confusing coordinate mismatch.
    """
    longitude = np.rad2deg(np.asarray(grid.longitude_radian, dtype=np.float64))
    latitude = np.rad2deg(np.asarray(grid.latitude_radian, dtype=np.float64))
    longitude_axis = longitude[:, 0]
    latitude_axis = latitude[0, :]

    separable = np.allclose(
        longitude, longitude_axis[:, None], atol=_COORDINATE_TOLERANCE_DEGREES
    ) and np.allclose(
        latitude, latitude_axis[None, :], atol=_COORDINATE_TOLERANCE_DEGREES
    )
    if not separable:
        raise ValueError(
            f"Climatology file \"{path!s:s}\" cannot be matched against this grid: "
            "the grid is curvilinear (longitude/latitude are not separable), so it "
            "has no 1-D longitude/latitude axes to compare the file's coordinates to."
        )

    return longitude_axis, latitude_axis


def _check_axis_matches(
    file_values: np.ndarray,
    grid_values: np.ndarray,
    axis_name: str,
    path,
    var: str,
    periodic: bool = False,
) -> None:
    """Raise if a file coordinate axis does not match the grid's."""
    if file_values.shape != grid_values.shape:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has "
            f"{file_values.size:d} {axis_name:s} points but the grid has "
            f"{grid_values.size:d}."
        )

    difference = file_values - grid_values
    if periodic:
        # Longitudes may be written on 0-360 or -180-180; compare modulo 360.
        difference = (difference + 180.0) % 360.0 - 180.0

    largest = float(np.max(np.abs(difference)))
    if largest > _COORDINATE_TOLERANCE_DEGREES:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" is on a "
            f"different {axis_name:s} axis than the grid (largest difference "
            f"{largest:.6g} degrees, tolerance {_COORDINATE_TOLERANCE_DEGREES:g}). "
            "The file must already be on the model grid; this loader does not regrid."
        )


def _load_monthly_climatology(path, var: str, grid: SlabGrid) -> jnp.ndarray:
    """Load a monthly climatology field onto the model grid.

    The array is transposed BY NAME rather than by position, so a file written
    in any axis order loads correctly and a file written on the wrong grid is
    rejected instead of being reinterpreted -- the previous loader did a bare
    ``jnp.array(ds[var])`` and silently accepted any array whose shape happened
    to fit.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a netCDF file.
    var : str
        Name of the variable to read.
    grid : SlabGrid
        The model grid the file must already be on.

    Returns
    -------
    jnp.ndarray
        The field with shape ``(n_lon, n_lat, 12)``.

    Raises
    ------
    ValueError
        If the variable is missing, its dimensions are not recognisable as
        longitude/latitude/time, it does not have exactly 12 time records, or
        its coordinates do not match the grid's. The message names the file
        and the check that failed.
    """
    dataset = xr.open_dataset(path)

    if var not in dataset:
        raise ValueError(
            f"Climatology file \"{path!s:s}\" has no variable \"{var:s}\" "
            f"(it has {sorted(dataset.data_vars)!r})."
        )
    field = dataset[var]

    longitude_name, latitude_name, time_name = (
        _resolve_dim_name(field.dims, aliases, path, var)
        for aliases in _CLIMATOLOGY_DIM_ALIASES
    )

    unexpected = set(field.dims) - {longitude_name, latitude_name, time_name}
    if unexpected:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has extra "
            f"dimensions {sorted(unexpected)!r}; only longitude, latitude and time "
            "are supported."
        )

    n_records = field.sizes[time_name]
    if n_records != _MONTHS_PER_YEAR:
        raise ValueError(
            f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has "
            f"{n_records:d} \"{time_name:s}\" records; exactly "
            f"{_MONTHS_PER_YEAR:d} monthly records are required."
        )

    for name in (longitude_name, latitude_name):
        if name not in field.coords:
            raise ValueError(
                f"Climatology file \"{path!s:s}\": variable \"{var:s}\" has no "
                f"\"{name:s}\" coordinate variable, so its grid cannot be verified."
            )

    grid_longitude, grid_latitude = _grid_axes_degrees(grid, path)
    _check_axis_matches(
        np.asarray(field.coords[longitude_name].values, dtype=np.float64),
        grid_longitude,
        "longitude",
        path,
        var,
        periodic=True,
    )
    _check_axis_matches(
        np.asarray(field.coords[latitude_name].values, dtype=np.float64),
        grid_latitude,
        "latitude",
        path,
        var,
    )

    return jnp.asarray(
        field.transpose(longitude_name, latitude_name, time_name).values
    )


@tree_math.struct
class OceanState:
    sim_time: jnp.ndarray
    sea_surface_temperature: jnp.ndarray
    mixed_layer_depth: jnp.ndarray

    @classmethod
    def zeros(cls, shape, sim_time=None, sea_surface_temperature=None, mixed_layer_depth=None):
        return cls(
            sim_time if sim_time is not None else jnp.zeros(()),
            sea_surface_temperature if sea_surface_temperature is not None else jnp.zeros(shape),
            mixed_layer_depth if mixed_layer_depth is not None else jnp.zeros(shape),
        )


@tree_math.struct
class OceanForcing:
    total_heat_flux: jnp.ndarray
    q_flux: jnp.ndarray

    @classmethod
    def zeros(cls, shape, total_heat_flux=None, q_flux=None):
        return cls(
            total_heat_flux if total_heat_flux is not None else jnp.zeros(shape),
            q_flux if q_flux is not None else jnp.zeros(shape + (12,)),
        )


@tree_math.struct
class OceanDerived:
    ice_frazil_melt_energy: jnp.ndarray
    effective_total_heat_flux: jnp.ndarray
    q_flux_snapshot: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        ice_frazil_melt_energy=None,
        effective_total_heat_flux=None,
        q_flux_snapshot=None,
    ):
        return cls(
            ice_frazil_melt_energy if ice_frazil_melt_energy is not None else jnp.zeros(shape),
            effective_total_heat_flux if effective_total_heat_flux is not None else jnp.zeros(shape),
            q_flux_snapshot if q_flux_snapshot is not None else jnp.zeros(shape),
        )


class SlabOceanModel(SlabModelBase):
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

    Freeze/melt potential
    ----------------------
    Following CESM's slab-ocean/CICE coupling convention: after the update above,
    `sea_surface_temperature` is clamped so it never drops below `T_freezing` (the seawater
    freezing point), and the heat that clamp removes (or, symmetrically, the heat available
    above freezing) is reported as a single signed diagnostic, `ice_frazil_melt_energy`
    (J/m^2, energy released over this coupling step -- not a flux):

        ice_frazil_melt_energy = (T_freezing - T_unclamped)
            * mixed_layer_depth * ocean_density * ocean_specific_heat_capacity

    Positive values mean the mixed layer would have gone sub-freezing -- that deficit forms
    new (frazil) ice. Negative values mean the mixed layer sits above freezing -- that surplus
    is available to melt existing ice from below. This is exactly CESM's `frzmlt`: one signed
    quantity, computed once per coupling step with no separate relaxation timescale (the
    coupling step itself is the timescale). This ocean model has no ice physics of its own, so
    `ice_frazil_melt_energy` is meant to be consumed by a sea-ice component (e.g.
    `SlabSeaiceModel`) via the coupler.
    """

    def __init__(
        self,
        grid: SlabGrid,
        start_datetime: jdt.Datetime = _DEFAULT_START_DATETIME,
        timestep: float = 86400.0,
        relaxation_time: float = 60 * 86400.0,
        mixed_layer_depth_min: float = 40.0,
        mixed_layer_depth_max: float = 60.0,
        SST_clim_file: str | None = None,
        Q_flux_file: str | None = None,
        forcing_method: str | None = None,
        initialization_sea_surface_temperature: float = 288.15,
        mask_value: float = 0.0,
        calendar: str = "365_day",
    ):
        """Initialize slab ocean model.

        Args:
            grid: The model's grid. See jem.components.slab.grid.SlabGrid.
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds
            relaxation_time: Relaxation timescale to climatology in seconds
            mixed_layer_depth_min: Minimum mixed layer depth in meters
            mixed_layer_depth_max: Maximum mixed layer depth in meters
            SST_clim_file: Optional path to SST climatology NetCDF file

        """
        self.relaxation_time = relaxation_time
        self.mixed_layer_depth_min = mixed_layer_depth_min
        self.mixed_layer_depth_max = mixed_layer_depth_max
        self.SST_clim_file = SST_clim_file
        self.Q_flux_file = Q_flux_file

        super().__init__(
            name="SlabOceanModel",
            grid=grid,
            start_datetime=start_datetime,
            timestep=timestep,
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
                raise FileNotFoundError(f"Q-flux file \"{self.Q_flux_file!s:s}\" is specified but it does not exist.")
        elif self.forcing_method == "relaxation":
            if self.SST_clim_file is None:
                print("Notice: `SST_clim_file` is not given. Default values (zeros) will be used.")
            elif not Path(self.SST_clim_file).exists():
                raise FileNotFoundError(f"SST climatology file \"{self.SST_clim_file!s:s}\" is specified but does not exist.")
            elif (self.relaxation_time < 0) or jnp.isnan(self.relaxation_time):
                raise ValueError("`relaxation_time` must be a positive number or infinity.")
        else:
            raise ValueError(f"Unknown `forcing_method` is given: \"{self.forcing_method!s:s}\" ")

    def initialize(self):
        """Initialize ocean model fields."""
        nonocn_idx = self.grid.binary_mask != self.mask_value

        # Initialize mixed layer depth with latitudinal variation
        init_mixed_layer_depth = (
            self.mixed_layer_depth_max
            + (self.mixed_layer_depth_min - self.mixed_layer_depth_max)
            * jnp.cos(self.grid.latitude_radian) ** 3
        )

        # Load or create initial SST
        if self.SST_clim_file is not None:
            print("SST climatology file. The given initial SST will be used.")
            print("SST climatology file: ", self.SST_clim_file)
            self.SST_clim = _load_monthly_climatology(
                self.SST_clim_file, "sst", self.grid
            )
            init_sea_surface_temperature = self.SST_clim[:, :, 0].copy()
        else:
            print("Boundary does not exist. Idealized initial SST will be used.")
            init_sea_surface_temperature = (
                positive_cosine_cubic_latitude_squared(self.grid.latitude_radian) * 10.0
                + constants.freezing_point_K
            )

        # Apply mask
        init_sea_surface_temperature = init_sea_surface_temperature.at[nonocn_idx].set(
            default_land_surface_temperature
        )

        # Validate mask consistency
        if jnp.sum(jnp.isnan(init_sea_surface_temperature)) == 0:
            print("grid.bmask and SST_clim do share the same mask.")
        else:
            raise ValueError(
                "fmask_ocn and sea_surface_temperature_init do not share the same mask."
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
        
        # The Q-flux climatology lives in the forcing carry, so it has to be
        # loaded here: `validate()` only checks that the file exists, and
        # before this it was never read at all -- `forcing_method="Qflux"`
        # silently ran with Q = 0 everywhere.
        if self.forcing_method == "Qflux" and self.Q_flux_file is not None:
            q_flux = _load_monthly_climatology(self.Q_flux_file, "qflux", self.grid)
        else:
            q_flux = None

        return {
            "state": OceanState.zeros(
                self.grid.shape,
                mixed_layer_depth=init_mixed_layer_depth,
                sea_surface_temperature=init_sea_surface_temperature,
            ),
            "forcing": OceanForcing.zeros(self.grid.shape, q_flux=q_flux),
            "derived": OceanDerived.zeros(self.grid.shape),
        }

    def _create_step_function_body(self):
        """Create the step function for ocean model."""
        start_day_offset = self._compute_start_day_offset()
        ocn_idx = self.grid.binary_mask == self.mask_value
        nonocn_idx = self.grid.binary_mask != self.mask_value

        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]
            new_sea_surface_temperature_anom = state.sea_surface_temperature
            total_heat_flux = forcing.total_heat_flux
            snapshot_Qflux = jnp.zeros(self.grid.shape)
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

            # Add climatology back
            new_sea_surface_temperature = new_sea_surface_temperature_anom
            if self.forcing_method == "relaxation":
                new_sea_surface_temperature += sst_clim_end
            
            # Apply land mask
            new_sea_surface_temperature = new_sea_surface_temperature.at[
                nonocn_idx
            ].set(default_land_surface_temperature)

            # Freeze/melt potential (CESM's `frzmlt`): heat surplus/deficit of the mixed
            # layer relative to freezing, for this coupling step. Positive -> forms new ice;
            # negative -> available to melt existing ice from below.
            ice_frazil_melt_energy = jnp.where(
                ocn_idx,
                (constants.seawater_freezing_point_K - new_sea_surface_temperature)
                * state.mixed_layer_depth
                * constants.ocean_density
                * constants.ocean_specific_heat_capacity,
                0.0,
            )

            # The ocean itself never carries a sub-freezing SST -- that deficit was just
            # diverted into ice_frazil_melt_energy above.
            new_sea_surface_temperature = jnp.where(
                ocn_idx,
                jnp.maximum(new_sea_surface_temperature, constants.seawater_freezing_point_K),
                new_sea_surface_temperature,
            )

            new_state = state.replace(
                sea_surface_temperature=new_sea_surface_temperature,
                sim_time=new_sim_time,
            )

            new_derived = OceanDerived.zeros(
                self.grid.shape,
                ice_frazil_melt_energy=ice_frazil_melt_energy,
                effective_total_heat_flux=total_heat_flux,
                q_flux_snapshot=snapshot_Qflux,
            )

            result = {
                "state": new_state,
                "forcing": forcing,
                "derived": new_derived,
            }
            return result, stack_objects([result])

        return step_function

    def _create_xarray_data_vars(self, predictions) -> dict[str, Any]:
        """Create xarray data variables for ocean output."""
        state = predictions["state"]
        derived = predictions["derived"]
        T_grid_dims = ("time",) + self.grid.dims

        data_vars = {
            "sea_surface_temperature": (
                T_grid_dims,
                state.sea_surface_temperature,
                {
                    "long_name": "Sea surface temperature",
                    "units": "K",
                }
            ),
            "mixed_layer_depth": (
                T_grid_dims,
                state.mixed_layer_depth,
                {
                    "long_name": "Mixed layer depth",
                    "units": "m",
                }
            ),
            "total_heat_flux": (
                T_grid_dims,
                derived.effective_total_heat_flux,
                {
                    "long_name": "Total heat flux forcing",
                    "units": "W m-2",
                    "positive": "upward",
                }
            ),
            "ice_frazil_melt_energy": (
                T_grid_dims,
                derived.ice_frazil_melt_energy,
                {
                    "long_name": "Freeze/melt potential (frzmlt): positive forms ice, negative melts ice",
                    "units": "J m-2",
                }
            ),
        }

        if self.forcing_method == "Qflux":
            data_vars["q_flux"] = (
                T_grid_dims,
                derived.q_flux_snapshot,
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
