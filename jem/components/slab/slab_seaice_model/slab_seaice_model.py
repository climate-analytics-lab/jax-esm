"""Slab sea-ice model.

Sea-ice cover diagnosed from the sea surface temperature. The ice has no
thickness, no heat capacity and no memory, and it does not exchange energy
with the ocean. Its only effect on the coupled system is the ice fraction
handed to the atmosphere.
"""

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

#: Width (K) of the SST ramp from full ice cover to open water. A cell is
#: ice-free when its SST is this much above the seawater freezing point.
ICE_FREE_SST_EXCESS = 1.8


@tree_math.struct
class SeaiceState:
    """Sea-ice state.

    Attributes
    ----------
    ice_fraction : (n_lon, n_lat) array
        Areal ice fraction in [0, 1]. Zero on all-land cells.

    """

    ice_fraction: jnp.ndarray

    @classmethod
    def zeros(cls, shape, ice_fraction=None):
        return cls(
            ice_fraction if ice_fraction is not None else jnp.zeros(shape),
        )


@tree_math.struct
class SeaiceForcing:
    """Sea-ice forcing, filled by an exchanger.

    Attributes
    ----------
    sea_surface_temperature : (n_lon, n_lat) array
        SST (K) from the ocean.

    """

    sea_surface_temperature: jnp.ndarray

    @classmethod
    def zeros(cls, shape, sea_surface_temperature=None):
        return cls(
            sea_surface_temperature if sea_surface_temperature is not None else jnp.zeros(shape),
        )


class SlabSeaiceModel(SlabModelBase):
    """Sea-ice cover diagnosed from the sea surface temperature.

    Each step the ice fraction is computed from the SST alone::

        ice_fraction = clip((T_f + ICE_FREE_SST_EXCESS - SST) / ICE_FREE_SST_EXCESS, 0, 1)

    where ``T_f`` is ``jem.constants.seawater_freezing_point_K``. A cell at or
    below ``T_f`` is fully covered, a cell :data:`ICE_FREE_SST_EXCESS` K or
    more above it is ice-free, and the fraction is linear in between. Cells
    with no ocean (``grid.binary_ocean_mask == 0``) have no ice.

    The SST must be supplied by an exchanger. The standard coupling does this
    with ``ocn.state.sea_surface_temperature -> seaice.forcing.sea_surface_temperature``
    (see :data:`jem.exchangers.STANDARD_EXCHANGES`), and passes the ice
    fraction on to the atmosphere's ice-cover forcing (jcm's ``sice_am``).

    The ice does not feed back on the ocean. Forming or melting it costs no
    energy, so the ocean's ``derived.ice_frazil_melt_energy`` is not used here.

    Initial condition
    -----------------
    If ``ice_clim_file`` is given, the initial ice fraction is the
    climatology at the month the run starts in
    (:attr:`SlabModelBase.start_year_fraction`, set by the coupler through
    ``bind``). Otherwise the run starts ice-free.

    This only matters for the first coupling step. In the standard workflow
    the exchange runs before the components, so the atmosphere's first step
    sees the initial fraction. From the first :meth:`step` on, the fraction
    comes from the ocean's SST.
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
            Model parameters, defaulting to
            :meth:`SlabSeaiceParameters.default`. :meth:`initialize` puts them
            in the carry unless it is given other parameters.
        name : str
            Component name in the coupler's workflow and carry. The default,
            ``"seaice"``, is the name the standard coupling
            (:func:`jem.exchangers.default_exchanges`) expects.
        ice_clim_file : str, optional
            netCDF file with a 12-month ``icec`` sea-ice concentration
            climatology (fraction in [0, 1]) on the model grid. It is only
            used for the initial condition. Without it the run starts
            ice-free.

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
            ocean = self.grid.binary_ocean_mask
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

    def initialize(self, params: SlabSeaiceParameters | None = None) -> Carry:
        """Build the initial sea-ice carry.

        Parameters
        ----------
        params : SlabSeaiceParameters, optional
            Parameters to put in ``carry["params"]``. Defaults to the ones
            the model was constructed with.

        Returns
        -------
        Carry
            ``{"params", "state", "forcing"}``. The forcing starts at zero
            and is filled by the exchanger before the first step.

        """
        params = self._initial_params(params)

        ocean = self.grid.binary_ocean_mask
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
        """Diagnose the ice fraction from the current SST forcing.

        The result depends only on ``carry["forcing"]``. The previous state
        and ``time`` are not used.

        Parameters
        ----------
        carry : Carry
            ``{"params", "state", "forcing"}`` as built by :meth:`initialize`.
        time : CouplingTime
            Not used.

        Returns
        -------
        tuple[Carry, Diagnostics]
            The new carry, and ``{"state", "forcing"}`` for output.

        """
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]
        ocean = self.grid.binary_ocean_mask

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

