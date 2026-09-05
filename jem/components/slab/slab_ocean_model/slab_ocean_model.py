"""Slab ocean model component."""

import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import tree_math

from jem import constants
from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import (
    MASKED_SURFACE_TEMPERATURE,
    SlabModelBase,
    end_of_step,
    forcing_variable,
    load_monthly_climatology,
)
from jem.components.slab.grid import SlabGrid
from jem.components.slab.slab_ocean_model.params import (
    FORCING_METHODS,
    SlabOceanParameters,
)
from jem.utils.cycles import evaluate_cyclic_linear
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared

logger = logging.getLogger(__name__)

#: Equator-to-pole range (K) of the idealized initial SST profile used when no
#: SST climatology is given.
IDEALIZED_SST_RANGE = 10.0


@tree_math.struct
class OceanState:
    """The evolving state: the SST only.

    The mixed-layer depth is not state. It is a prescribed profile of the
    carried parameters (``mixed_layer_depth_min``/``_max``), recomputed in
    every ``step`` so that replacing or differentiating those parameters
    takes effect; it is written to the output as a derived field.
    """

    sea_surface_temperature: jnp.ndarray

    @classmethod
    def zeros(cls, shape, sea_surface_temperature=None):
        return cls(
            sea_surface_temperature if sea_surface_temperature is not None else jnp.zeros(shape),
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
    mixed_layer_depth: jnp.ndarray
    ice_frazil_melt_energy: jnp.ndarray
    effective_total_heat_flux: jnp.ndarray
    q_flux_snapshot: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        mixed_layer_depth=None,
        ice_frazil_melt_energy=None,
        effective_total_heat_flux=None,
        q_flux_snapshot=None,
    ):
        return cls(
            mixed_layer_depth if mixed_layer_depth is not None else jnp.zeros(shape),
            ice_frazil_melt_energy if ice_frazil_melt_energy is not None else jnp.zeros(shape),
            effective_total_heat_flux if effective_total_heat_flux is not None else jnp.zeros(shape),
            q_flux_snapshot if q_flux_snapshot is not None else jnp.zeros(shape),
        )


class SlabOceanModel(SlabModelBase):
    """Slab ocean model with prescribed mixed layer depth and climatology.

    This model simulates sea surface temperature evolution using a simple
    thermodynamic equation with optional relaxation to climatology::

        dT/dt = -F_net/(rho * cp * h) + forcing

    where ``T`` is the sea surface temperature, ``F_net`` the total heat flux
    (positive upward, so it cools the mixed layer), ``rho`` the ocean density,
    ``cp`` the ocean specific heat capacity, ``h`` the mixed layer depth, and
    ``forcing`` the extra temperature forcing selected by
    ``params.forcing_method``:

    (1) ``forcing_method == "none"``: forcing = 0.

    (2) ``forcing_method == "qflux"``: the traditional Q-flux adjustment, a
        prescribed periodic heat source over the year::

            forcing = Q / (rho * cp * h)

        where ``Q`` is read from ``q_flux_file`` (variable ``qflux``). With no
        file, Q is zero everywhere -- a valid setup when the Q-flux itself is
        the thing being trained.

    (3) ``forcing_method == "relaxation"``: linear relaxation to climatology::

            forcing = - (T - T_clim) / tau

        with ``tau = params.relaxation_time`` and ``T_clim`` read from
        ``sst_clim_file`` (variable ``sst``). A relaxation run REQUIRES that
        file: without a target there is nothing to relax to, and the previous
        behaviour -- silently setting tau to infinity and then dereferencing a
        climatology that was never loaded -- could not work.

    Initial condition
    -----------------
    With an SST climatology, the initial SST is that climatology sampled at
    the month the run starts in (:attr:`SlabModelBase.start_year_fraction`,
    which the coupler sets through ``bind``). Without one it is an idealized
    profile,
    ``params.initial_sst`` at the poles rising by
    :data:`IDEALIZED_SST_RANGE` towards the equator. ``initial_sst`` is what
    the constructor has always accepted (as
    ``initialization_sea_surface_temperature``) but never used: the base of the
    idealized profile was hard-wired to the freezing point, making the
    argument dead. Wiring it up moves the default idealized ocean from
    273-283 K to 288-298 K, which is also the more sensible aquaplanet start.

    Freeze/melt potential
    ---------------------
    Following CESM's slab-ocean/CICE coupling convention: after the update
    above, ``sea_surface_temperature`` is clamped so it never drops below the
    seawater freezing point, and the heat that clamp removes (or, symmetrically,
    the heat available above freezing) is reported as a single signed
    diagnostic, ``ice_frazil_melt_energy`` (J/m^2, energy released over this
    coupling step -- not a flux)::

        ice_frazil_melt_energy = (T_freezing - T_unclamped)
            * mixed_layer_depth * ocean_density * ocean_specific_heat_capacity

    Positive values mean the mixed layer would have gone sub-freezing -- that
    deficit forms new (frazil) ice. Negative values mean the mixed layer sits
    above freezing -- that surplus is available to melt existing ice from below.
    This is exactly CESM's ``frzmlt``: one signed quantity, computed once per
    coupling step with no separate relaxation timescale (the coupling step
    itself is the timescale). This ocean model has no ice physics of its own,
    so ``ice_frazil_melt_energy`` is meant to be consumed by a sea-ice
    component (e.g. ``SlabSeaiceModel``) through the coupler.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: SlabOceanParameters | None = None,
        *,
        name: str = "ocn",
        sst_clim_file: str | None = None,
        q_flux_file: str | None = None,
    ):
        """Initialize the slab ocean model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid.
        params : SlabOceanParameters, optional
            Tunable parameters; defaults to
            :meth:`SlabOceanParameters.default`.
        name : str
            Component name in the coupler's workflow and carry.
        sst_clim_file : str, optional
            netCDF file holding a 12-month ``sst`` climatology on the model
            grid. Used for the initial condition, and required for
            ``forcing_method == "relaxation"``.
        q_flux_file : str, optional
            netCDF file holding a 12-month ``qflux`` climatology on the model
            grid. Only meaningful for ``forcing_method == "qflux"``.

        Raises
        ------
        ValueError
            If the configuration cannot run: an unknown forcing method,
            relaxation without a climatology or with a non-positive timescale,
            a Q-flux file a non-Q-flux run would ignore, or a climatology whose
            ocean points are not finite.
        FileNotFoundError
            If a named file does not exist.

        """
        super().__init__(name=name, grid=grid)
        self.params = SlabOceanParameters.default() if params is None else params
        self.sst_clim_file = sst_clim_file
        self.q_flux_file = q_flux_file

        forcing_method = self.params.forcing_method
        if forcing_method not in FORCING_METHODS:
            raise ValueError(
                f"Unknown forcing_method {forcing_method!r}; expected one of "
                f"{list(FORCING_METHODS)!r}."
            )
        if q_flux_file is not None and forcing_method != "qflux":
            raise ValueError(
                f"q_flux_file was given but forcing_method is {forcing_method!r}, "
                "which never reads it. Set forcing_method='qflux' or drop the file."
            )

        if forcing_method == "relaxation":
            if sst_clim_file is None:
                raise ValueError(
                    "forcing_method='relaxation' needs sst_clim_file: there is no "
                    "climatology to relax towards without it."
                )
            relaxation_time = float(self.params.relaxation_time)
            if not relaxation_time > 0.0:
                raise ValueError(
                    "relaxation_time must be a positive number of seconds; got "
                    f"{relaxation_time!r}."
                )

        # Boundary data is *configuration*, so it is read here rather than in
        # ``initialize()``: that keeps initialize() pure with respect to self
        # (it only assembles arrays) and surfaces a bad file at construction,
        # where the traceback still points at the caller's own line.
        self.sst_climatology = self._load(sst_clim_file, "sst")
        self.q_flux_climatology = self._load(q_flux_file, "qflux")

        if self.sst_climatology is not None:
            ocean = self._ocean_cells(self.params)
            if bool(jnp.any(jnp.isnan(self.sst_climatology) & ocean[..., None])):
                raise ValueError(
                    f"SST climatology file \"{sst_clim_file!s:s}\" has NaNs over ocean "
                    "points of this grid: the file's land mask and the grid's disagree."
                )

    def _load(self, path: str | None, var: str) -> jnp.ndarray | None:
        """Load a monthly climatology, or return None when no file was given."""
        if path is None:
            return None
        if not Path(path).exists():
            raise FileNotFoundError(f"Climatology file \"{path!s:s}\" does not exist.")
        logger.info("%s: loading %r climatology from %s", self.name, var, path)
        return load_monthly_climatology(path, var, self.grid)

    def _ocean_cells(self, params: SlabOceanParameters) -> jnp.ndarray:
        """Boolean mask of the cells this model integrates."""
        return self.grid.binary_mask == params.ocean_mask_value

    def initialize(self) -> Carry:
        """Build the initial ocean carry."""
        params = self.params
        ocean = self._ocean_cells(params)

        if self.sst_climatology is not None:
            sea_surface_temperature = evaluate_cyclic_linear(
                self.start_year_fraction, self.sst_climatology
            )
        else:
            sea_surface_temperature = params.initial_sst + IDEALIZED_SST_RANGE * (
                positive_cosine_cubic_latitude_squared(self.grid.latitude_radian)
            )
        sea_surface_temperature = jnp.where(
            ocean, sea_surface_temperature, MASKED_SURFACE_TEMPERATURE
        )

        return {
            "params": params,
            "state": OceanState.zeros(
                self.grid.shape,
                sea_surface_temperature=sea_surface_temperature,
            ),
            "forcing": OceanForcing.zeros(
                self.grid.shape, q_flux=self.q_flux_climatology
            ),
            "derived": OceanDerived.zeros(
                self.grid.shape,
                mixed_layer_depth=self._mixed_layer_depth(params),
            ),
        }

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Advance the mixed layer by one coupling step.

        The temperature update is Euler backward in the relaxation term, which
        is why the timescale appears as ``1 / (1 + dt/tau)`` rather than as an
        explicit tendency: the relaxation is the stiff term here, and an
        explicit step of it is unstable once ``dt`` approaches ``tau``.
        """
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]
        ocean = self._ocean_cells(params)

        # From the CARRIED parameters, every step: a depth cached at
        # initialization would make `mixed_layer_depth_min/max` dead
        # parameters with zero gradient.
        mixed_layer_depth = self._mixed_layer_depth(params)
        heat_capacity = (
            constants.ocean_density
            * constants.ocean_specific_heat_capacity
            * mixed_layer_depth
        )

        total_heat_flux = forcing.total_heat_flux
        q_flux_snapshot = jnp.zeros(self.grid.shape)
        anomaly = state.sea_surface_temperature
        climatology_end = None
        time_factor = 1.0

        # ``forcing_method`` is static configuration, so this branches at trace
        # time and only the selected term is ever compiled.
        if params.forcing_method == "relaxation":
            climatology_begin = self._climatology_at(time, ocean)
            climatology_end = self._climatology_at(end_of_step(time), ocean)
            anomaly = state.sea_surface_temperature - climatology_begin
            time_factor = 1.0 / (1.0 + time.dt / params.relaxation_time)
        elif params.forcing_method == "qflux":
            q_flux_snapshot = jnp.where(
                ocean, evaluate_cyclic_linear(time.year_fraction, forcing.q_flux), 0.0
            )
            # Q is a heat SOURCE for the mixed layer (positive Q warms it, see
            # the class docstring and the output attribute), while
            # ``total_heat_flux`` is UPWARD positive (it cools the mixed layer)
            # and is negated in the update below. Folding Q into the upward flux
            # therefore needs a minus sign; adding it (as an earlier version
            # did) silently reversed every prescribed Q-flux experiment.
            total_heat_flux = total_heat_flux - q_flux_snapshot

        new_anomaly = time_factor * (
            anomaly + time.dt / heat_capacity * (-total_heat_flux)
        )
        sea_surface_temperature = new_anomaly
        if climatology_end is not None:
            sea_surface_temperature = sea_surface_temperature + climatology_end
        sea_surface_temperature = jnp.where(
            ocean, sea_surface_temperature, MASKED_SURFACE_TEMPERATURE
        )

        # Freeze/melt potential (CESM's ``frzmlt``): the heat surplus or deficit
        # of the mixed layer relative to freezing, for this coupling step.
        # Positive -> forms new ice; negative -> available to melt existing ice.
        ice_frazil_melt_energy = jnp.where(
            ocean,
            (constants.seawater_freezing_point_K - sea_surface_temperature)
            * mixed_layer_depth
            * constants.ocean_density
            * constants.ocean_specific_heat_capacity,
            0.0,
        )

        # The ocean itself never carries a sub-freezing SST -- that deficit was
        # just diverted into ice_frazil_melt_energy above.
        sea_surface_temperature = jnp.where(
            ocean,
            jnp.maximum(
                sea_surface_temperature, constants.seawater_freezing_point_K
            ),
            sea_surface_temperature,
        )

        new_state = state.replace(sea_surface_temperature=sea_surface_temperature)
        new_derived = OceanDerived.zeros(
            self.grid.shape,
            mixed_layer_depth=mixed_layer_depth,
            ice_frazil_melt_energy=ice_frazil_melt_energy,
            effective_total_heat_flux=total_heat_flux,
            q_flux_snapshot=q_flux_snapshot,
        )

        diagnostics = {
            "state": new_state,
            "forcing": forcing,
            "derived": new_derived,
        }
        return {"params": params, **diagnostics}, diagnostics

    def _mixed_layer_depth(self, params: SlabOceanParameters) -> jnp.ndarray:
        """Prescribed mixed-layer depth: ``max`` at the poles, ``min`` at the equator."""
        return (
            params.mixed_layer_depth_max
            + (params.mixed_layer_depth_min - params.mixed_layer_depth_max)
            * jnp.cos(self.grid.latitude_radian) ** 3
        )

    def _climatology_at(self, time: CouplingTime, ocean: jnp.ndarray) -> jnp.ndarray:
        """SST climatology interpolated to ``time``, masked to ocean cells."""
        # Only reachable with forcing_method="relaxation", which the constructor
        # refuses to build without a climatology file.
        assert self.sst_climatology is not None
        return jnp.where(
            ocean,
            evaluate_cyclic_linear(time.year_fraction, self.sst_climatology),
            MASKED_SURFACE_TEMPERATURE,
        )

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for ocean output."""
        state = diagnostics["state"]
        derived = diagnostics["derived"]
        dims = ("time",) + self.grid.dims

        data_vars = {
            "sea_surface_temperature": (
                dims,
                state.sea_surface_temperature,
                {
                    "long_name": "Sea surface temperature",
                    "units": "K",
                },
            ),
            "mixed_layer_depth": (
                dims,
                derived.mixed_layer_depth,
                {
                    "long_name": "Mixed layer depth",
                    "units": "m",
                },
            ),
            # Not `forcing_total_heat_flux`: this is the flux the mixed
            # layer was actually cooled by, which is the received flux with
            # any Q-flux adjustment already folded in -- a quantity this
            # model computed.
            "total_heat_flux": (
                dims,
                derived.effective_total_heat_flux,
                {
                    "long_name": "Effective heat flux applied to the mixed layer",
                    "units": "W m-2",
                    "positive": "upward",
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
        }

        if self.params.forcing_method == "qflux":
            # The prescribed Q-flux climatology evaluated at this step: a
            # boundary condition the ocean was given, not one it produced,
            # even though the snapshot is carried in `derived`.
            data_vars[forcing_variable("q_flux")] = (
                dims,
                derived.q_flux_snapshot,
                {
                    "long_name": "Prescribed Q-flux forcing",
                    "units": "W m-2",
                    "positive": "Heating the ocean",
                },
            )

        return data_vars
