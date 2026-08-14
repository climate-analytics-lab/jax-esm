"""Slab sea-ice model component."""

from typing import Any

import jax.numpy as jnp
import jax_datetime as jdt
import tree_math

from jem import constants
from jem.components.slab.base import _DEFAULT_START_DATETIME, SlabModelBase
from jem.components.slab.grid import SlabGrid
from jem.utils.bulk_op import stack_objects

default_land_surface_temperature = 288.15


@tree_math.struct
class SeaiceState:
    sim_time: jnp.ndarray
    ice_thickness: jnp.ndarray
    ice_surface_temperature: jnp.ndarray

    @classmethod
    def zeros(cls, shape, sim_time=None, ice_thickness=None, ice_surface_temperature=None):
        return cls(
            sim_time if sim_time is not None else jnp.zeros(()),
            ice_thickness if ice_thickness is not None else jnp.zeros(shape),
            ice_surface_temperature if ice_surface_temperature is not None else jnp.zeros(shape),
        )


@tree_math.struct
class SeaiceForcing:
    ice_frazil_melt_energy: jnp.ndarray

    @classmethod
    def zeros(cls, shape, ice_frazil_melt_energy=None):
        return cls(
            ice_frazil_melt_energy if ice_frazil_melt_energy is not None else jnp.zeros(shape),
        )


@tree_math.struct
class SeaiceDerived:
    ice_fraction: jnp.ndarray
    ice_frazil_melt_energy: jnp.ndarray

    @classmethod
    def zeros(cls, shape, ice_fraction=None, ice_frazil_melt_energy=None):
        return cls(
            ice_fraction if ice_fraction is not None else jnp.zeros(shape),
            ice_frazil_melt_energy if ice_frazil_melt_energy is not None else jnp.zeros(shape),
        )


class SlabSeaiceModel(SlabModelBase):
    """Sea-ice model driven purely by the ocean's freeze/melt potential.

    Ice cover is binary per grid cell (a cell is either open ocean or
    fully ice-covered). There is no heat capacity, no conductive
    temperature profile, no snow layer, and no dynamics/advection: ice
    only grows or melts at the base, and only in response to
    `ice_frazil_melt_energy` -- the freeze/melt potential diagnosed each
    coupling step by an ocean model such as `SlabOceanModel` (CESM calls
    this quantity `frzmlt`).

    Governing equation for ice thickness `h`:

        h_new = max(0, h + ice_frazil_melt_energy / (rho_ice * L_ice))

    where:
        - ``ice_frazil_melt_energy``: forcing input (J/m^2, this coupling step's energy,
          not a flux). Positive means the ocean mixed layer had a heat deficit relative
          to freezing -- that deficit freezes new ice at the base. Negative means the
          ocean had surplus heat above freezing -- that surplus melts ice from below.
        - ``rho_ice``: ice_density
        - ``L_ice``: ice_latent_heat_fusion

    Because `ice_frazil_melt_energy` is already a per-step energy (its host ocean model
    folds the coupling step directly into the diagnostic, matching CESM's convention of a
    single `frzmlt` term with no separate relaxation timescale), it is added to `h`
    directly -- no further multiplication by a timestep is needed here.

    `h` is clipped at zero: melt cannot drive thickness negative.

    `min_ice_thickness` plays no role in the tendency itself -- it is used only to
    diagnose whether a cell counts as ice-covered or open water, for
    `ice_surface_temperature` below. Since there is no conductive temperature profile to
    solve for, `ice_surface_temperature` is not physically diagnosed -- it is simply
    reported as `T_melt` wherever a cell carries ice above `min_ice_thickness`, and
    `T_freeze` over ice-free ocean, for output/diagnostic purposes only.

    This is a standalone component in the sense that it has no direct knowledge of SST or
    mixed layer depth -- all of that is folded into `ice_frazil_melt_energy` by the ocean
    model. Wiring an ocean model's `derived.ice_frazil_melt_energy` to this component's
    `forcing.ice_frazil_melt_energy` via the coupler's mapper is required for this model
    to do anything.

    Ice fraction
    ------------
    Since ice cover here is binary (a cell is fully ice-covered or fully open water), an
    ice *fraction* -- needed as a boundary condition by an atmosphere model (e.g. jcm's
    `sice_am`/`icec` forcing) -- has to come from a closure rather than a genuine sub-grid
    concentration. This model uses a smooth, monotonic saturating closure so the field
    stays differentiable in `h` everywhere (no kink, unlike the classic linear-ramp
    closure `min(1, h / h0)`):

        ice_fraction = 1 - exp(-h / ice_fraction_thickness_scale)

    `ice_fraction_thickness_scale` sets how quickly a cell "fills in" as it thickens:
    `ice_fraction -> 0` as `h -> 0` and `ice_fraction -> 1` as `h` grows well past that
    scale. This is reported as `derived.ice_fraction`, for a mapper to route to an
    atmosphere model's ice-fraction forcing.
    """

    def __init__(
        self,
        grid: SlabGrid,
        start_datetime: jdt.Datetime = _DEFAULT_START_DATETIME,
        timestep: float = 86400.0,
        initialization_ice_thickness: float = 0.0,
        min_ice_thickness: float = 1e-3,
        ice_fraction_thickness_scale: float = 0.5,
        mask_value: float = 0.0,
        calendar: str = "365_day",
    ):
        """Initialize slab sea-ice model.

        Args:
            grid: The model's grid. See jem.components.slab.grid.SlabGrid.
            start_datetime: Simulation start datetime
            timestep: Model timestep in seconds
            initialization_ice_thickness: Uniform initial ice thickness over ocean points (m)
            min_ice_thickness: Thickness above which a cell is diagnosed as ice-covered (m)
            ice_fraction_thickness_scale: Thickness scale (m) of the smooth
                `1 - exp(-h / scale)` closure used to derive `ice_fraction` from thickness
            mask_value: `bmask` value that identifies ocean grid points
            calendar: Calendar used for the simulation clock
        """
        self.initialization_ice_thickness = initialization_ice_thickness
        self.min_ice_thickness = min_ice_thickness
        self.ice_fraction_thickness_scale = ice_fraction_thickness_scale
        self.mask_value = mask_value

        super().__init__(
            name="SlabSeaiceModel",
            grid=grid,
            start_datetime=start_datetime,
            timestep=timestep,
            calendar=calendar,
        )

        self.validate()

    def validate(self):
        super().validate()
        if self.min_ice_thickness <= 0:
            raise ValueError("`min_ice_thickness` must be a positive number.")
        if self.ice_fraction_thickness_scale <= 0:
            raise ValueError("`ice_fraction_thickness_scale` must be a positive number.")

    def initialize(self):
        """Initialize sea-ice model fields."""
        ocn_idx = self.grid.binary_mask == self.mask_value

        init_ice_thickness = jnp.where(
            ocn_idx, self.initialization_ice_thickness, 0.0
        )
        init_ice_surface_temperature = jnp.where(
            ocn_idx & (init_ice_thickness > self.min_ice_thickness),
            constants.ice_melting_point_K,
            jnp.where(ocn_idx, constants.seawater_freezing_point_K, default_land_surface_temperature),
        )

        init_ice_fraction = jnp.where(
            ocn_idx,
            1.0 - jnp.exp(-init_ice_thickness / self.ice_fraction_thickness_scale),
            0.0,
        )

        return {
            "state": SeaiceState.zeros(
                self.grid.shape,
                ice_thickness=init_ice_thickness,
                ice_surface_temperature=init_ice_surface_temperature,
            ),
            "forcing": SeaiceForcing.zeros(self.grid.shape),
            "derived": SeaiceDerived.zeros(self.grid.shape, ice_fraction=init_ice_fraction),
        }

    def _create_step_function_body(self):
        """Create the step function for the sea-ice model."""
        ocn_idx = self.grid.binary_mask == self.mask_value
        min_ice_thickness = self.min_ice_thickness
        ice_fraction_thickness_scale = self.ice_fraction_thickness_scale

        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]

            h = state.ice_thickness
            ice_frazil_melt_energy = forcing.ice_frazil_melt_energy

            new_ice_thickness = h + ice_frazil_melt_energy / (
                constants.ice_density * constants.ice_latent_heat_fusion
            )
            new_ice_thickness = jnp.clip(new_ice_thickness, 0.0, None)
            new_ice_thickness = jnp.where(ocn_idx, new_ice_thickness, 0.0)

            new_surface_temperature = jnp.where(
                ocn_idx & (new_ice_thickness > min_ice_thickness),
                constants.ice_melting_point_K,
                jnp.where(ocn_idx, constants.seawater_freezing_point_K, default_land_surface_temperature),
            )

            new_ice_fraction = jnp.where(
                ocn_idx,
                1.0 - jnp.exp(-new_ice_thickness / ice_fraction_thickness_scale),
                0.0,
            )

            new_sim_time = state.sim_time + self.timestep

            new_state = state.replace(
                ice_thickness=new_ice_thickness,
                ice_surface_temperature=new_surface_temperature,
                sim_time=new_sim_time,
            )

            new_derived = SeaiceDerived.zeros(
                self.grid.shape,
                ice_fraction=new_ice_fraction,
                ice_frazil_melt_energy=ice_frazil_melt_energy,
            )

            result = {
                "state": new_state,
                "forcing": forcing,
                "derived": new_derived,
            }
            return result, stack_objects([result])

        return step_function

    def _create_xarray_data_vars(self, predictions) -> dict[str, Any]:
        """Create xarray data variables for sea-ice output."""
        state = predictions["state"]
        derived = predictions["derived"]
        T_grid_dims = ("time",) + self.grid.dims

        return {
            "ice_thickness": (
                T_grid_dims,
                state.ice_thickness,
                {
                    "long_name": "Sea ice thickness",
                    "units": "m",
                },
            ),
            "ice_surface_temperature": (
                T_grid_dims,
                state.ice_surface_temperature,
                {
                    "long_name": "Sea ice surface temperature",
                    "units": "K",
                },
            ),
            "ice_frazil_melt_energy": (
                T_grid_dims,
                derived.ice_frazil_melt_energy,
                {
                    "long_name": "Freeze/melt potential (frzmlt): positive forms ice, negative melts ice",
                    "units": "J m-2",
                },
            ),
            "ice_fraction": (
                T_grid_dims,
                derived.ice_fraction,
                {
                    "long_name": "Sea ice areal fraction (smooth closure from thickness)",
                    "units": "1",
                },
            ),
        }

    def get_info(self):
        return {
            "min_ice_thickness": self.min_ice_thickness,
            "ice_fraction_thickness_scale": self.ice_fraction_thickness_scale,
        }
