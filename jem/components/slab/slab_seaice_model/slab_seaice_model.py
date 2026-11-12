"""Slab sea-ice model component."""

import logging
import math
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


@tree_math.struct
class SeaiceDerived:
    """What the ice diagnoses and the other components read.

    The ice fraction is diagnosed from this step's forcing and carries no
    memory, so it is a derived field, not state: nothing in :meth:`step`
    reads its previous value. It is the ice fraction of the sea part of the
    cell, ``sea_fraction`` (``1 - fractional_mask``), which is what a
    regridded row onto another grid is weighted by.
    """

    ice_fraction: jnp.ndarray
    sea_fraction: jnp.ndarray

    @classmethod
    def zeros(cls, shape, ice_fraction=None, sea_fraction=None):
        return cls(
            ice_fraction if ice_fraction is not None else jnp.zeros(shape),
            sea_fraction if sea_fraction is not None else jnp.zeros(shape),
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

        ice_fraction = clip((T_f + w - SST) / w, 0, 1)

    with ``w = params.ice_free_sst_excess``, so a cell at or below ``T_f`` is
    fully covered, one ``w`` kelvin or more above it is ice-free, and land
    cells carry no ice. The fraction is published as
    ``derived.ice_fraction``; ``forcing.sea_surface_temperature`` is its only
    input, and wiring an ocean's SST to it through an exchanger is required
    for the model to do anything
    (``ocn.state.sea_surface_temperature -> seaice.forcing.sea_surface_temperature``
    in :data:`jem.exchangers.STANDARD_EXCHANGES`).

    The ice has no energy budget: it takes no energy to form and releases
    none when it goes, so it does not take the ocean's freeze/melt potential
    (``derived.ice_frazil_melt_energy``). The ocean it is coupled to must
    therefore not clamp its mixed layer at the freezing point
    (``SlabOceanParameters.clamp_at_freezing=False``, the default): the
    mixed layer then keeps its own deficit below freezing, which is where
    this model reads full ice cover from, and the energy stays in the ocean
    rather than being created by a clamp with no ice to account for it. What
    the ice changes is the atmosphere's surface, through the fraction an
    exchanger routes to the atmosphere's ice-cover forcing (jcm's
    ``sice_am``, the ice fraction of the sea part of a cell).

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
            If ``params.ice_free_sst_excess`` is not finite and strictly
            positive, or the climatology has NaNs over this grid's ocean
            points.
        FileNotFoundError
            If ``ice_clim_file`` does not exist.

        """
        super().__init__(name=name, grid=grid)
        self.params = SlabSeaiceParameters.default() if params is None else params
        # The ramp divides by its width; checked here, where the value is
        # concrete, because inside `step` it is a traced leaf.
        width = float(self.params.ice_free_sst_excess)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError(
                "params.ice_free_sst_excess must be a finite positive "
                f"temperature difference in kelvin; got {width!r}."
            )

        self.ice_clim_file = ice_clim_file
        self.ice_climatology = self._load(ice_clim_file, "icec")

        if self.ice_climatology is not None:
            ocean = self._ocean_cells()
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

    def _ocean_cells(self) -> jnp.ndarray:
        """Boolean mask of the cells this model integrates: every cell with
        any ocean, coast cells included (see :class:`SlabGrid`).
        """
        return self.grid.ocean_mask

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

        ocean = self._ocean_cells()
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
            "forcing": SeaiceForcing.zeros(self.grid.shape),
            "derived": SeaiceDerived.zeros(
                self.grid.shape, ice_fraction=ice_fraction,
                sea_fraction=1.0 - self.grid.fractional_mask,
            ),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Diagnose the ice fraction from the SST this step was handed.

        The step is independent of ``time`` and of the previous state: the
        fraction is a function of the forcing alone, so there is no ``dt`` to
        apply and no seasonal cycle to look up.
        """
        params = carry["params"]
        forcing = carry["forcing"]
        ocean = self._ocean_cells()

        width = params.ice_free_sst_excess
        ice_free_temperature = constants.seawater_freezing_point_K + width
        ice_fraction = jnp.clip(
            (ice_free_temperature - forcing.sea_surface_temperature) / width,
            0.0, 1.0,
        )
        derived = SeaiceDerived.zeros(
            self.grid.shape, ice_fraction=jnp.where(ocean, ice_fraction, 0.0),
            sea_fraction=1.0 - self.grid.fractional_mask,
        )

        diagnostics = {"forcing": forcing, "derived": derived}
        return {"params": params, **diagnostics}, diagnostics

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for sea-ice output."""
        forcing = diagnostics["forcing"]
        derived = diagnostics["derived"]
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
                derived.ice_fraction,
                {
                    "long_name": "Sea ice fraction of the sea part of the cell",
                    "units": "1",
                    **role_attrs("derived"),
                },
            ),
        }

