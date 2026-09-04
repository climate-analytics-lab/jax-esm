"""Slab sea-ice model component."""

from typing import Any

import jax.numpy as jnp
import jcm.constants as jcm_constants
import tree_math

from jem import constants
from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import MASKED_SURFACE_TEMPERATURE, SlabModelBase
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_seaice_model.params import SlabSeaiceParameters


@tree_math.struct
class SeaiceState:
    ice_thickness: jnp.ndarray
    ice_surface_temperature: jnp.ndarray

    @classmethod
    def zeros(cls, shape, ice_thickness=None, ice_surface_temperature=None):
        return cls(
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
        - ``rho_ice``: ``jcm.constants.rhoi``
        - ``L_ice``: ``jcm.constants.alhf``

    Because `ice_frazil_melt_energy` is already a per-step energy (its host ocean model
    folds the coupling step directly into the diagnostic, matching CESM's convention of a
    single `frzmlt` term with no separate relaxation timescale), it is added to `h`
    directly -- no further multiplication by a timestep is needed here.

    `h` is clipped at zero: melt cannot drive thickness negative.

    `min_ice_thickness` plays no role in the tendency itself -- it is used only to
    diagnose whether a cell counts as ice-covered or open water, for
    `ice_surface_temperature` below. Since there is no conductive temperature profile to
    solve for, `ice_surface_temperature` is not physically diagnosed -- it is simply
    reported as the fresh-ice melting point ``jcm.constants.tmelt`` wherever a cell
    carries ice above `min_ice_thickness`, and the seawater freezing point over ice-free
    ocean, for output/diagnostic purposes only.

    This is a standalone component in the sense that it has no direct knowledge of SST or
    mixed layer depth -- all of that is folded into `ice_frazil_melt_energy` by the ocean
    model. Wiring an ocean model's `derived.ice_frazil_melt_energy` to this component's
    `forcing.ice_frazil_melt_energy` through the coupler's exchanger is required for this
    model to do anything.

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
    scale. This is reported as `derived.ice_fraction`, for an exchanger to route to an
    atmosphere model's ice-fraction forcing.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabSeaiceParameters | None = None,
        *,
        name: str = "ice",
    ):
        """Initialize the slab sea-ice model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid.
        params : SlabSeaiceParameters, optional
            Tunable parameters; defaults to
            :meth:`SlabSeaiceParameters.default`.
        name : str
            Component name in the coupler's workflow and carry.

        Raises
        ------
        ValueError
            If a thickness scale is not positive, which would make the ice
            fraction closure or the ice-cover diagnosis undefined.

        """
        super().__init__(name=name, grid=grid)
        self.params = SlabSeaiceParameters.default() if params is None else params

        if not float(self.params.min_ice_thickness) > 0.0:
            raise ValueError("min_ice_thickness must be a positive number of metres.")
        if not float(self.params.ice_fraction_thickness_scale) > 0.0:
            raise ValueError(
                "ice_fraction_thickness_scale must be a positive number of metres."
            )
        if float(self.params.initial_ice_thickness) < 0.0:
            raise ValueError("initial_ice_thickness cannot be negative.")

    def _ocean_cells(self, params: SlabSeaiceParameters) -> jnp.ndarray:
        """Boolean mask of the cells this model integrates."""
        return self.grid.binary_mask == params.ocean_mask_value

    def initialize(self) -> Carry:
        """Build the initial sea-ice carry."""
        params = self.params
        ocean = self._ocean_cells(params)

        ice_thickness = jnp.where(ocean, params.initial_ice_thickness, 0.0)

        return {
            "params": params,
            "state": SeaiceState.zeros(
                self.grid.shape,
                ice_thickness=ice_thickness,
                ice_surface_temperature=_surface_temperature(
                    ice_thickness, ocean, params
                ),
            ),
            "forcing": SeaiceForcing.zeros(self.grid.shape),
            "derived": SeaiceDerived.zeros(
                self.grid.shape,
                ice_fraction=_ice_fraction(ice_thickness, ocean, params),
            ),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Grow or melt ice at the base by one coupling step.

        The step is independent of ``time``: the forcing it integrates is
        already an energy per coupling step, not a flux, so there is no ``dt``
        to apply and no seasonal cycle to look up.
        """
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]
        ocean = self._ocean_cells(params)

        ice_frazil_melt_energy = forcing.ice_frazil_melt_energy
        ice_thickness = state.ice_thickness + ice_frazil_melt_energy / (
            jcm_constants.rhoi * jcm_constants.alhf
        )
        ice_thickness = jnp.clip(ice_thickness, 0.0, None)
        ice_thickness = jnp.where(ocean, ice_thickness, 0.0)

        new_state = state.replace(
            ice_thickness=ice_thickness,
            ice_surface_temperature=_surface_temperature(ice_thickness, ocean, params),
        )
        new_derived = SeaiceDerived.zeros(
            self.grid.shape,
            ice_fraction=_ice_fraction(ice_thickness, ocean, params),
            ice_frazil_melt_energy=ice_frazil_melt_energy,
        )

        diagnostics = {
            "state": new_state,
            "forcing": forcing,
            "derived": new_derived,
        }
        return {"params": params, **diagnostics}, diagnostics

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for sea-ice output."""
        state = diagnostics["state"]
        derived = diagnostics["derived"]
        dims = ("time",) + self.grid.dims

        return {
            "ice_thickness": (
                dims,
                state.ice_thickness,
                {
                    "long_name": "Sea ice thickness",
                    "units": "m",
                },
            ),
            "ice_surface_temperature": (
                dims,
                state.ice_surface_temperature,
                {
                    "long_name": "Sea ice surface temperature",
                    "units": "K",
                },
            ),
            "ice_frazil_melt_energy": (
                dims,
                derived.ice_frazil_melt_energy,
                {
                    "long_name": (
                        "Freeze/melt potential (frzmlt): positive forms ice, "
                        "negative melts ice"
                    ),
                    "units": "J m-2",
                },
            ),
            "ice_fraction": (
                dims,
                derived.ice_fraction,
                {
                    "long_name": "Sea ice areal fraction (smooth closure from thickness)",
                    "units": "1",
                },
            ),
        }


def _surface_temperature(
    ice_thickness: jnp.ndarray,
    ocean: jnp.ndarray,
    params: SlabSeaiceParameters,
) -> jnp.ndarray:
    """Diagnose the surface temperature reported for each cell.

    Shared by ``initialize`` and ``step`` so the initial carry and the stepped
    carry cannot diverge in how a cell is classified.
    """
    return jnp.where(
        ocean & (ice_thickness > params.min_ice_thickness),
        jcm_constants.tmelt,
        jnp.where(ocean, constants.seawater_freezing_point_K, MASKED_SURFACE_TEMPERATURE),
    )


def _ice_fraction(
    ice_thickness: jnp.ndarray,
    ocean: jnp.ndarray,
    params: SlabSeaiceParameters,
) -> jnp.ndarray:
    """Areal ice fraction from thickness, by the smooth closure."""
    return jnp.where(
        ocean,
        1.0 - jnp.exp(-ice_thickness / params.ice_fraction_thickness_scale),
        0.0,
    )
