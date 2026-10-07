"""Slab sea-ice model component."""

import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import tree_math

from jem import constants
from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import (
    SlabModelBase,
    forcing_variable,
    load_monthly_climatology,
    role_attrs,
)
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_seaice_model.params import SlabSeaiceParameters
from jem.utils.cycles import evaluate_cyclic_linear

logger = logging.getLogger(__name__)

#: How far above the seawater freezing point (K) the sea surface has to be
#: for a cell to be ice-free; the ice fraction ramps linearly to full cover
#: across this range. See :class:`SlabSeaiceModel`.
ICE_FREE_SST_EXCESS = 1.8


@tree_math.struct
class SeaiceState:
    ice_fraction: jnp.ndarray

    @classmethod
    def zeros(cls, shape, ice_fraction=None):
        return cls(
            ice_fraction if ice_fraction is not None else jnp.zeros(shape),
        )


@tree_math.struct
class SeaiceForcing:
    sea_surface_temperature: jnp.ndarray

    @classmethod
    def zeros(cls, shape, sea_surface_temperature=None):
        return cls(
            sea_surface_temperature if sea_surface_temperature is not None else jnp.zeros(shape),
        )


class SlabSeaiceModel(SlabModelBase):
    """Sea-ice cover diagnosed from the sea surface temperature.

    The model has no thickness, no heat capacity and no memory: each coupling
    step it reads the SST the ocean handed it and reports an areal ice
    fraction that ramps linearly from open water to full cover as the surface
    cools to the seawater freezing point ``T_f``
    (``jem.constants.seawater_freezing_point_K``)::

        ice_fraction = clip((T_f + ICE_FREE_SST_EXCESS - SST) / ICE_FREE_SST_EXCESS, 0, 1)

    so a cell at or below ``T_f`` is fully covered, one
    :data:`ICE_FREE_SST_EXCESS` kelvin or more above it is ice-free, and land
    cells carry no ice. The fraction is the model's only state field,
    ``state.ice_fraction``; ``forcing.sea_surface_temperature`` is its only
    input, and wiring an ocean's SST to it through an exchanger is required
    for the model to do anything
    (``ocn.state.sea_surface_temperature -> seaice.forcing.sea_surface_temperature``
    in :data:`jem.exchangers.STANDARD_EXCHANGES`).

    The ice does not feed back on the ocean: it takes no energy to form and
    releases none when it goes, so the ocean's own freeze/melt diagnostic
    (``derived.ice_frazil_melt_energy``) is not consumed by this model. What
    the ice changes is the atmosphere's surface, through the fraction an
    exchanger routes to the atmosphere's ice-cover forcing (jcm's
    ``sice_am``).

    Initial condition
    -----------------
    With an ice-concentration climatology (``ice_clim_file``) the run starts
    from the observed cover: the climatology is sampled at the month the run
    starts in (:attr:`SlabModelBase.start_year_fraction`, which the coupler
    sets through ``bind``). Without one the run starts ice-free.

    The choice matters for exactly one coupling step. Under the standard
    workflow the exchange runs before the components, so the fraction
    :meth:`initialize` returns is what the atmosphere is handed on its first
    step, before this model has seen any SST; from the first :meth:`step` on
    the fraction is the one diagnosed from the ocean. An Earth-like run
    started without a climatology spends that first step with open water at
    both poles.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabSeaiceParameters | None = None,
        *,
        name: str = "seaice",
        ice_clim_file: str | None = None,
    ):
        """Initialize the slab sea-ice model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid.
        params : SlabSeaiceParameters, optional
            Tunable parameters; defaults to
            :meth:`SlabSeaiceParameters.default`. They are what
            :meth:`initialize` puts in the carry unless it is handed
            parameters of its own.
        name : str
            Component name in the coupler's workflow and carry. The default is
            the name the standard coupling wires the sea ice under
            (:func:`jem.exchangers.default_exchanges`), so a model registered
            as ``{"seaice": SlabSeaiceModel(grid)}`` is connected without
            being renamed.
        ice_clim_file : str, optional
            netCDF file holding a 12-month ``icec`` sea-ice *concentration*
            climatology (a fraction in ``[0, 1]``) on the model grid. Used
            for the initial condition only -- this model has no relaxation
            and never reads it again. Without it the run starts ice-free.

        Raises
        ------
        ValueError
            If the climatology has NaNs over this grid's ocean points.
        FileNotFoundError
            If ``ice_clim_file`` does not exist.

        """
        super().__init__(name=name, grid=grid)
        self.params = SlabSeaiceParameters.default() if params is None else params

        self.ice_clim_file = ice_clim_file
        self.ice_climatology = self._load(ice_clim_file, "icec")

        if self.ice_climatology is not None:
            ocean = self._ocean_cells(self.params)
            if bool(jnp.any(jnp.isnan(self.ice_climatology) & ocean[..., None])):
                raise ValueError(
                    f"Sea-ice climatology file \"{ice_clim_file!s:s}\" has NaNs over "
                    "ocean points of this grid: the file's land mask and the grid's "
                    "disagree."
                )

    def _load(self, path: str | None, var: str) -> jnp.ndarray | None:
        """Load a monthly climatology, or return None when no file was given."""
        if path is None:
            return None
        if not Path(path).exists():
            raise FileNotFoundError(f"Climatology file \"{path!s:s}\" does not exist.")
        logger.info("%s: loading %r climatology from %s", self.name, var, path)
        return load_monthly_climatology(path, var, self.grid)

    def _ocean_cells(self, params: SlabSeaiceParameters) -> jnp.ndarray:
        """Boolean mask of the cells this model integrates."""
        return self.grid.binary_mask == params.ocean_mask_value

    def initialize(self, params: SlabSeaiceParameters | None = None) -> Carry:
        """Build the initial sea-ice carry.

        Parameters
        ----------
        params : SlabSeaiceParameters, optional
            Parameters to start from; defaults to the ones the model was
            constructed with. The same object goes into ``carry["params"]``,
            so the parameters ``step`` reads are the ones the initial state
            was built from.

        """
        params = self._initial_params(params)

        ocean = self._ocean_cells(params)
        if self.ice_climatology is not None:
            # The observed concentration at the month the run starts in. It
            # is what the atmosphere is handed on its first step, before this
            # model has seen any SST. Land cells are selected out *before*
            # the clip, not after: a file's land fill value is routinely NaN.
            ice_fraction = jnp.clip(
                jnp.where(
                    ocean,
                    evaluate_cyclic_linear(
                        self.start_year_fraction, self.ice_climatology),
                    0.0,
                ),
                0.0, 1.0,
            )
        else:
            ice_fraction = jnp.zeros(self.grid.shape)

        return {
            "params": params,
            "state": SeaiceState.zeros(
                self.grid.shape, ice_fraction=ice_fraction,
            ),
            "forcing": SeaiceForcing.zeros(self.grid.shape),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Diagnose the ice fraction from the SST this step was handed.

        The step is independent of ``time`` and of the previous state: the
        fraction is a function of the forcing alone, so there is no ``dt`` to
        apply and no seasonal cycle to look up.
        """
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]
        ocean = self._ocean_cells(params)

        sea_surface_temperature = forcing.sea_surface_temperature

        all_closed_temperature = constants.seawater_freezing_point_K
        all_opened_temperature = all_closed_temperature + ICE_FREE_SST_EXCESS

        ice_fraction = (
            (all_opened_temperature - sea_surface_temperature)
            / ICE_FREE_SST_EXCESS
        )
        ice_fraction = jnp.clip(ice_fraction, 0.0, 1.0)
        ice_fraction = jnp.where(ocean, ice_fraction, 0.0)

        new_state = state.replace(
            ice_fraction=ice_fraction,
        )

        diagnostics = {
            "state": new_state,
            "forcing": forcing,
        }
        return {"params": params, **diagnostics}, diagnostics

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for sea-ice output."""
        state = diagnostics["state"]
        forcing = diagnostics["forcing"]
        dims = ("time",) + self.grid.dims

        return {
            # The SST the ocean handed this model, prefixed so it does not
            # collide with the ocean's own `sea_surface_temperature` when the
            # two datasets are merged.
            forcing_variable("sea_surface_temperature"): (
                dims,
                forcing.sea_surface_temperature,
                {
                    "long_name": (
                        "Sea surface temperature the ice fraction was "
                        "diagnosed from"
                    ),
                    "units": "K",
                    **role_attrs("forcing"),
                },
            ),
            "ice_fraction": (
                dims,
                state.ice_fraction,
                {
                    "long_name": "Sea ice areal fraction",
                    "units": "1",
                    **role_attrs("state"),
                },
            ),
        }

