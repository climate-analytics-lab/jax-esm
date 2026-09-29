"""Winton (2000) three-layer thermodynamic sea-ice component: snow + two ice layers.

The thermodynamics follow MITgcm ``pkg/thsice`` (``thsice_solve4temp.F`` and
``thsice_calc_thickn.F``), which is the implementation the MIT aquaplanet papers
used, so that the JAX code can be compared to that Fortran at machine precision.
Where thsice departs from the Winton (2000) paper we follow thsice:

* layer enthalpies ``q1, q2`` (J/kg, positive = energy needed to melt the layer to
  liquid water at 0 degC): ``q1 = -c_w Tm + c_i (Tm - T1) + L (1 - Tm/T1)``,
  ``q2 = -c_i T2 + L`` with ``Tm = -mu S`` the brine melting point;
* the surface is clamped at 0 degC when melting (snow or bare ice);
* surface melt is applied before basal growth; then basal melt; then snowfall,
  flooding (snow below the waterline becomes ice), and re-equalisation of the two
  ice layers with the ``q2 > L`` guard.

Sub-grid ice fraction uses a Hibler (1979) lead closure (this part is not from
thsice): frazil ice from the ocean closes leads over ``lead_closing_thickness``;
melt reduces the fraction as ``-(f/2h) dh``. Thermodynamics act on the ice-covered
part of the cell (``h`` is the thickness of that part).

Surface fluxes over ice are computed here with SPEEDY-style bulk formulae (same
coefficients as JCM's ``speedy_surface_flux``) from the atmosphere's near-surface
state, so ``F(Ts)`` and ``dF/dTs`` are available inside the implicit solve. JCM
computes its sea-fraction fluxes at a single surface temperature, so the mapper
hands JCM an ice-fraction-weighted surface temperature, and the ocean receives
JCM's sea flux minus what the ice absorbed (exact conservation of the
atmosphere-side energy), plus what the ice passes down (transmitted shortwave,
melt surplus) minus the basal heat the ice draws.

Sign conventions: fluxes at the ice surface are positive downward (into ice);
``F_cb = 4 K (T2 - Tf)/h`` is the conductive flux toward the base (positive
downward); ``F_b`` is the ocean-to-ice basal flux (positive = ocean warms ice);
``ocean_heat_flux_up`` is positive upward (JEM slab-ocean convention).
Temperatures inside the ice model are in degC; the JEM interface uses K.

Provenance and licence: the thermodynamic core is a port of MITgcm pkg/thsice (MIT License, Copyright (c)
2018 MITgcm Developers and Contributors); see NOTICE in this package for the licence text and the references
(Winton 2000; Hibler 1979; the transport follows Ferreira, Marshall and Rose 2011, Edwards and Marsh 2005 and
Flato and Hibler 1992). `tests/reference/thsice` rebuilds the Fortran oracle the port is checked against.
"""

import dataclasses
import logging
import math
from typing import Any

import jax
import jax.numpy as jnp
import jcm.constants as jcm_constants
import tree_math
from jcm.physics.speedy.params import ModRadConParameters, SurfaceFluxParameters

from jem.base.component import Carry, CouplingTime, Diagnostics
from jem.components.slab.base import (
    MASKED_SURFACE_TEMPERATURE,
    SlabModelBase,
    forcing_variable,
    role_attrs,
)
from jem.components.slab.grid import SlabGrid
from jem.components.slab.winton_seaice_model.ice_transport import (
    IceTransportGrid,
    transport_fields,
)
from jem.components.slab.winton_seaice_model.params import WintonSeaiceParameters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants (Winton 2000 Table 1)
# These are the constants of the thsice scheme itself, and they differ from
# jcm.constants' where the two model families made different choices (ice
# density 905 here against jcm's 917, heat of fusion 3.34e5 against 3.33e5).
# They are kept module-level and fixed because the machine-precision comparison
# against the Fortran oracle (tests/reference/thsice) is a comparison of *this*
# parameter set, and because the ice and snow columns are integrated as a
# closed enthalpy budget in these units: the ocean is charged in joules, so the
# budget closes whichever value the atmosphere uses. Quantities the ice shares
# with the atmosphere's surface-flux scheme (cpd, rd, p0, sbc, alhc) are read
# from jcm.constants when the step is traced, so a set_constants(...) override
# reaches the bulk fluxes over ice as it reaches the atmosphere.
RHO_ICE = 905.0
RHO_SNOW = 330.0
RHO_SW = 1026.0          # seawater, for flooding
C_ICE = 2100.0
C_W = 3990.0             # thsice cpWater (enthalpy reference to liquid water at 0 degC)
L_ICE = 3.34e5
K_ICE = 2.03
K_SNOW = 0.31
MU = 0.054
S_ICE = 1.0
T_MELT = -MU * S_ICE     # brine melting point used in the enthalpy (thsice Tmlt1)
T_SURF_MELT = 0.0        # thsice clamps the melting surface at 0 degC (snow or bare ice)
# Freezing point of the seawater under the ice (degC). It is the value of
# jem.constants.seawater_freezing_point_K (271.35 K) expressed in the ice
# model's degC; it is fixed here rather than read live because the layer
# enthalpies and the oracle comparison are built on it.
T_FREEZE = -1.8
Q_SNOW = L_ICE           # thsice qsnow
FLOOD_FAC = (RHO_SW - RHO_ICE) / RHO_SNOW
B_MELT = 0.006           # thsice bMeltCoef
USTAR_SLAB = 5.0e-3      # thsice ustar for zero ocean velocity: sqrt(25e-6)

# The Celsius-to-Kelvin offset of the ice model's degC state, a unit
# conversion rather than a physical constant to override.
KELVIN = 273.15


# ---------------------------------------------------------------- enthalpy (thsice convention)
def q_from_T1(T1):
    return -C_W * T_MELT + C_ICE * (T_MELT - T1) + L_ICE * (1.0 - T_MELT / T1)


def q_from_T2(T2):
    return -C_ICE * T2 + L_ICE


def T1_from_q(q1):
    """thsice: a1 T^2 + b1 T + c1 = 0 with a1 = c_i, b1 = q1 + (c_w - c_i) Tm - L, c1 = L Tm."""
    a1 = C_ICE
    b1 = q1 + (C_W - C_ICE) * T_MELT - L_ICE
    c1 = L_ICE * T_MELT
    return 0.5 * (-b1 - jnp.sqrt(b1 * b1 - 4.0 * a1 * c1)) / a1


def T2_from_q(q2):
    return (L_ICE - q2) / C_ICE


def column_enthalpy_to_melt(h, hs, q1, q2):
    """Energy (J/m2) needed to melt the whole column to water at 0 degC."""
    return RHO_ICE * 0.5 * h * (q1 + q2) + RHO_SNOW * Q_SNOW * hs


# ---------------------------------------------------------------- surface fluxes over ice
def qsat_ice(T_kelvin, p):
    """Saturation specific humidity (kg/kg) over ice (Magnus) and its temperature derivative."""
    Tc = T_kelvin - KELVIN
    es = 611.2 * jnp.exp(22.46 * Tc / (Tc + 272.62))
    des = es * 22.46 * 272.62 / (Tc + 272.62) ** 2
    q = 0.622 * es / (p - 0.378 * es)
    dq = 0.622 * p / (p - 0.378 * es) ** 2 * des
    return q, dq


def ice_surface_flux(Ts_c, rlds, t_air, q_air, wind, p_air, sfp=None, emis=None):
    """SPEEDY-style non-solar surface flux over ice (positive downward) and its exact derivative dF/dTs.

    sfp: JCM SurfaceFluxParameters (exchange coefficient chs, gust speed vgust, stability dtheta/fstab/lscasym);
    emis: longwave emissivity (JCM ModRadConParameters.emisfc). Defaults are JCM's defaults; physical constants
    (cpd, rd, p0, sbc, alhc) are read from jcm.constants.physical_constants when the step is traced. The
    derivative is taken with forward-mode AD so it includes the stability dependence of the exchange velocity.
    """
    sfp = SurfaceFluxParameters.default() if sfp is None else sfp
    emis = ModRadConParameters.default().emisfc if emis is None else emis
    c = jcm_constants.physical_constants
    astab = jnp.where(sfp.lscasym, 0.5, 1.0)            # SPEEDY: asymmetric stability coefficient
    rho = c.p0 * p_air / (c.rd * t_air)

    def flux(Ts_c_):
        Ts = Ts_c_ + KELVIN
        dth = jnp.where(Ts > t_air, jnp.minimum(sfp.dtheta, Ts - t_air), jnp.maximum(-sfp.dtheta, astab * (Ts - t_air)))
        denv = rho * jnp.sqrt(wind ** 2 + sfp.vgust ** 2) * (1.0 + dth * sfp.fstab / sfp.dtheta)
        q_s, _ = qsat_ice(Ts, c.p0 * p_air)
        return rlds - emis * c.sbc * Ts ** 4 - sfp.chs * c.cpd * denv * (Ts - t_air) - sfp.chs * denv * c.alhc * (q_s - q_air)

    F, dF_exact = jax.jvp(flux, (Ts_c,), (jnp.ones_like(Ts_c),))
    # The exact derivative includes d(denv)/dTs from the stability correction; on the stable side that term can
    # make dF positive, and the implicit surface solve needs dF < 0 (it divides by k12 - dF). So the linearisation
    # uses the exact derivative where it is at least as steep as the derivative with the exchange velocity frozen,
    # and the frozen-velocity derivative (always negative) otherwise: Newton where Newton is safe.
    Ts = Ts_c + KELVIN
    dth = jnp.where(Ts > t_air, jnp.minimum(sfp.dtheta, Ts - t_air), jnp.maximum(-sfp.dtheta, astab * (Ts - t_air)))
    denv = rho * jnp.sqrt(wind ** 2 + sfp.vgust ** 2) * (1.0 + dth * sfp.fstab / sfp.dtheta)
    _, dq_s = qsat_ice(Ts, c.p0 * p_air)
    dF_frozen = -4.0 * emis * c.sbc * Ts ** 3 - sfp.chs * c.cpd * denv - sfp.chs * denv * c.alhc * dq_s
    return F, jnp.minimum(dF_exact, dF_frozen)


def winton_temperature_step(h, hs, T1, T2, Ts, flux_fn, sw_abs, dt, i0=0.3, ksolar=1.5, n_iter=1):
    """Implicit ice temperature update.

    flux_fn(Ts) -> (F_nonsw, dF/dTs), positive downward. sw_abs is the absorbed
    shortwave at the surface (after albedo). With ``n_iter=1`` and a linear flux this is
    exactly thsice with external fluxes; larger ``n_iter`` re-linearises (Newton).
    Returns (T1, T2, Ts, M_s, F_cb, sw_ocn): surface melt energy flux M_s (>= 0),
    conductive flux toward the base F_cb = 4K(T2 - Tf)/h, shortwave passed to the ocean.
    """
    snow = hs > 0.0
    sw_pen = jnp.where(snow, 0.0, sw_abs * i0)      # thsice: no penetration through snow
    sw_ocn = sw_pen * jnp.exp(-ksolar * h)
    sw_int = sw_pen - sw_ocn
    sw_sfc = sw_abs - sw_pen
    k12 = 4.0 * K_ICE * K_SNOW / (K_SNOW * h + 4.0 * K_ICE * hs)
    k32 = 2.0 * K_ICE / h
    rhc = RHO_ICE * C_ICE * h
    a10 = rhc / (2.0 * dt) + k32 * (4.0 * dt * k32 + rhc) / (6.0 * dt * k32 + rhc)
    b10 = -h * (RHO_ICE * C_ICE * T1 + RHO_ICE * L_ICE * T_MELT / T1) / (2.0 * dt) \
          - k32 * (4.0 * dt * k32 * T_FREEZE + rhc * T2) / (6.0 * dt * k32 + rhc) - sw_int
    c10 = RHO_ICE * L_ICE * h * T_MELT / (2.0 * dt)

    Ts_it = Ts
    for _ in range(n_iter):
        F, dF = flux_fn(Ts_it)
        Fnet = F + sw_sfc
        a1 = a10 - k12 * dF / (k12 - dF)
        b1 = b10 - k12 * (Fnet - dF * Ts_it) / (k12 - dF)
        T1_free = -(b1 + jnp.sqrt(jnp.maximum(b1 * b1 - 4.0 * a1 * c10, 0.0))) / (2.0 * a1)
        dTs = (Fnet + k12 * (T1_free - Ts_it)) / (k12 - dF)
        Ts_free = Ts_it + dTs
        Ts_it = jnp.minimum(Ts_free, T_SURF_MELT)
    melting = Ts_free > T_SURF_MELT
    a1m = a10 + k12
    b1m = b10 - k12 * T_SURF_MELT
    T1_melt = -(b1m + jnp.sqrt(jnp.maximum(b1m * b1m - 4.0 * a1m * c10, 0.0))) / (2.0 * a1m)
    T1n = jnp.where(melting, T1_melt, T1_free)
    Tsn = jnp.where(melting, T_SURF_MELT, Ts_free)
    T2n = (2.0 * dt * k32 * (T1n + 2.0 * T_FREEZE) + rhc * T2) / (6.0 * dt * k32 + rhc)
    F_final, _ = flux_fn(Tsn)
    M_s = jnp.where(melting, F_final + sw_sfc - k12 * (Tsn - T1n), 0.0)
    F_cb = 4.0 * K_ICE * (T2n - T_FREEZE) / h
    return T1n, T2n, Tsn, M_s, F_cb, sw_ocn


# ---------------------------------------------------------------- mass step (thsice_calc_thickn)
def winton_mass_step(h, hs, q1, q2, M_s, F_b, F_cb, snowfall, dt, h_min=0.01):
    """Thickness changes for the ice-covered column, in thsice order.

    q1, q2: layer enthalpies (thsice convention). M_s: surface melt energy flux (>= 0).
    F_b: ocean-to-ice basal flux (>= 0 warms/melts the base). F_cb: conductive flux toward
    the base. snowfall: kg/m2/s. Returns (h, hs, q1, q2, energy_to_ocean [J/m2], vanished),
    where energy_to_ocean is the melt surplus (positive) or, if the column vanished, minus
    the energy the ocean must supply to melt what was left (negative).

    Every division whose denominator can be zero on a branch that a ``where`` later
    discards uses the double-``where`` form (a safe stand-in denominator, then the
    select). A floor such as ``maximum(x, 1e-300)`` underflows to exactly zero in
    float32, the dtype of a JEM run, and reverse-mode AD then turns the discarded
    branch's ``0 * inf`` into NaN.
    """
    h1 = h2 = 0.5 * h
    etop = jnp.maximum(M_s, 0.0) * dt
    ebot = (F_cb + F_b) * dt          # > 0 melts the base, < 0 freezes
    # --- top melt: snow, layer 1, layer 2
    rq = RHO_SNOW * Q_SNOW
    d = jnp.minimum(etop / rq, hs)
    hs = hs - d
    etop = etop - d * rq
    rq = RHO_ICE * q1
    d = jnp.minimum(etop / rq, h1)
    h1 = h1 - d
    etop = etop - d * rq
    rq = RHO_ICE * q2
    d = jnp.minimum(etop / rq, h2)
    h2 = h2 - d
    etop = etop - d * rq
    # --- basal growth
    qbot = -C_ICE * T_FREEZE + L_ICE
    dhi = jnp.where(ebot < 0.0, -ebot / (qbot * RHO_ICE), 0.0)
    grown = h2 + dhi > 0.0
    grown_thickness = jnp.where(grown, h2 + dhi, 1.0)
    q2 = jnp.where(grown, (h2 * q2 + dhi * qbot) / grown_thickness, q2)
    h2 = h2 + dhi
    ebot = jnp.maximum(ebot, 0.0)
    # --- basal melt: layer 2, layer 1, snow
    rq = RHO_ICE * q2
    d = jnp.minimum(ebot / rq, h2)
    h2 = h2 - d
    ebot = ebot - d * rq
    rq = RHO_ICE * q1
    d = jnp.minimum(ebot / rq, h1)
    h1 = h1 - d
    ebot = ebot - d * rq
    rq = RHO_SNOW * Q_SNOW
    d = jnp.minimum(ebot / rq, hs)
    hs = hs - d
    ebot = ebot - d * rq
    surplus = etop + ebot
    # --- vanishing column: ocean supplies what is left
    h_tot = h1 + h2
    vanished = h_tot < h_min
    left = RHO_ICE * (h1 * q1 + h2 * q2) + RHO_SNOW * Q_SNOW * hs
    surplus = jnp.where(vanished, surplus - left, surplus)
    h1 = jnp.where(vanished, 0.0, h1)
    h2 = jnp.where(vanished, 0.0, h2)
    hs = jnp.where(vanished, 0.0, hs)
    # --- snowfall (only on surviving ice)
    hs = hs + jnp.where(vanished, 0.0, snowfall * dt / RHO_SNOW)
    # --- flooding: snow below the waterline becomes ice in layer 1
    h_tot = h1 + h2
    flood = (hs > h_tot * FLOOD_FAC) & (h_tot > 0.0)
    dhs = jnp.where(flood, (hs - h_tot * FLOOD_FAC) * RHO_ICE / RHO_SW, 0.0)
    dhi = dhs * RHO_SNOW / RHO_ICE
    flooded_mass = jnp.where(flood, RHO_ICE * (h1 + dhi), 1.0)
    q1 = jnp.where(
        flood, (RHO_ICE * q1 * h1 + RHO_SNOW * Q_SNOW * dhs) / flooded_mass, q1
    )
    h1 = h1 + dhi
    hs = hs - dhs
    # --- re-equalise layers (thsice inlined THSICE_RESHAPE_LAYERS)
    h_tot = h1 + h2
    hlyr = 0.5 * h_tot
    safe_hlyr = jnp.where(hlyr > 0.0, hlyr, 1.0)
    f1a = (h1 - hlyr) / safe_hlyr
    q2tmp = f1a * q1 + (1.0 - f1a) * q2
    q1_alt = (h1 * q1 + h2 * q2 - hlyr * q2) / safe_hlyr      # keep q2 if q2tmp <= L (T2 > 0 guard)
    q1_a = jnp.where(q2tmp > L_ICE, q1, q1_alt)
    q2_a = jnp.where(q2tmp > L_ICE, q2tmp, q2)
    f1b = h1 / safe_hlyr
    q1_b = f1b * q1 + (1.0 - f1b) * q2
    upper_thicker = h1 > h2
    q1n = jnp.where(upper_thicker, q1_a, q1_b)
    q2n = jnp.where(upper_thicker, q2_a, q2)
    q1n = jnp.where(vanished, q_from_T1(T_FREEZE), q1n)
    q2n = jnp.where(vanished, q_from_T2(T_FREEZE), q2n)
    return h_tot, hs, q1n, q2n, surplus, vanished


# ---------------------------------------------------------------- JEM component
@tree_math.struct
class WintonState:
    """What the ice integrates. There is no clock: the coupler owns it."""

    ice_thickness: jnp.ndarray          # m, over the ice-covered fraction
    snow_thickness: jnp.ndarray         # m
    ice_fraction: jnp.ndarray
    upper_ice_temperature: jnp.ndarray  # degC
    lower_ice_temperature: jnp.ndarray  # degC
    ice_surface_temperature: jnp.ndarray  # degC


@tree_math.struct
class WintonForcing:
    """What the other components supply, one coupling step late.

    ``snowfall`` is a liquid-phase mass flux whose latent heat of fusion the
    supplier has **not** released: the ice charges the ocean the fusion enthalpy
    of the snow that lands on it (thsice's convention), so a supplier that
    already released it would have it counted twice.
    """

    rsds: jnp.ndarray
    rlds: jnp.ndarray
    air_temperature: jnp.ndarray        # K
    air_specific_humidity: jnp.ndarray  # kg/kg
    wind_speed: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray
    snowfall: jnp.ndarray               # kg/m2/s
    sea_surface_temperature: jnp.ndarray   # K
    ice_frazil_melt_energy: jnp.ndarray    # J/m2 per coupling step, + = freezing potential (CESM frzmlt)
    atm_sea_heat_flux: jnp.ndarray         # W/m2 downward, atmosphere's flux over the sea fraction
    ice_velocity_u: jnp.ndarray            # m/s along the grid's x (used only with transport)
    ice_velocity_v: jnp.ndarray            # m/s along the grid's y

    @classmethod
    def initial(cls, shape):
        """Return the forcing under which ice at the freezing point stays put.

        Coupling is lagged, so the first step integrates whatever
        :meth:`WintonSeaiceModel.initialize` leaves here. Air and sea at the
        freezing point, a blackbody downward longwave flux at that temperature
        and the saturation humidity over ice there leave no surface flux
        of consequence, so the first step neither melts nor grows the ice for
        want of a supplier. Wind and pressure are the exceptions that are not
        zero: a zero surface pressure makes the air density infinite, and the
        turbulent exchange velocity is built from the wind.
        """
        zeros = jnp.zeros(shape)
        freezing = KELVIN + T_FREEZE
        pressure = 1.0
        humidity, _ = qsat_ice(freezing, jcm_constants.p0 * pressure)
        return cls(
            rsds=zeros,
            rlds=zeros + jcm_constants.sbc * freezing**4,
            air_temperature=zeros + freezing,
            air_specific_humidity=zeros + humidity,
            wind_speed=zeros + 5.0,
            normalized_surface_pressure=zeros + pressure,
            snowfall=zeros,
            sea_surface_temperature=zeros + freezing,
            ice_frazil_melt_energy=zeros,
            atm_sea_heat_flux=zeros,
            ice_velocity_u=zeros,
            ice_velocity_v=zeros,
        )


@tree_math.struct
class WintonDerived:
    """What the ice diagnoses, for the other components and for output."""

    ice_fraction: jnp.ndarray
    effective_sea_surface_temperature: jnp.ndarray
    ocean_heat_flux_up: jnp.ndarray
    ice_surface_temperature_K: jnp.ndarray
    ice_volume: jnp.ndarray
    snow_volume: jnp.ndarray
    surface_melt_flux: jnp.ndarray
    basal_growth_flux: jnp.ndarray
    ice_atm_heat_flux: jnp.ndarray
    ocean_freshwater_flux_up: jnp.ndarray   # kg/m2/s per cell area; + = ice growth removes water from the ocean
    ice_albedo: jnp.ndarray                 # effective albedo of the ice-covered part (snow-aware)
    ice_energy_tendency: jnp.ndarray        # W/m2 per cell area; THERMODYNAMIC rate of change of the ice+snow enthalpy (relative to water at 0 C), i.e. the exchange with ocean and atmosphere; transport is excluded
    ice_energy_transport: jnp.ndarray       # W/m2 per cell area; change of ice+snow enthalpy by transport convergence (zero without transport)
    ocean_frazil_heating: jnp.ndarray       # W/m2; latent heat of the frazil ice the ocean clamped at freezing, returned to the ocean top layer

    @classmethod
    def zeros(cls, shape, **overrides):
        """Zero-filled derived fields on a ``shape`` grid, with ``overrides`` in place."""
        zeros = jnp.zeros(shape)
        return cls(**{
            field.name: overrides.get(field.name, zeros)
            for field in dataclasses.fields(cls)
        })


class WintonSeaiceModel(SlabModelBase):
    """Three-layer thermodynamic sea ice: snow, two ice layers and a Hibler lead closure.

    Each ocean cell holds an ice-covered fraction ``f`` with ice of thickness
    ``h`` and snow of thickness ``hs`` over that fraction, two ice-layer
    temperatures and the surface temperature. Every coupling step integrates
    ``n_substeps`` thermodynamic substeps (an implicit surface-temperature
    solve, surface and basal melt or growth, snowfall, flooding and layer
    re-equalisation, then the lead closure for the fraction) and, if the model
    was built with transport, one conservative advection-diffusion step of the
    ice. Water and enthalpy diagnostics are exact for the thermodynamic state,
    so a coupler can close the ice-ocean-atmosphere budget.

    The numerics, provenance and sign conventions are in the module docstring;
    the coupling contract -- which fields the ice needs, which it publishes and
    what the default exchange fills -- is in ``docs/source/design/winton_seaice.md``.

    The model advances by the coupler's ``time.dt``; it has no timestep or
    clock of its own. Its tunables travel in ``carry["params"]`` as a
    :class:`~jem.components.slab.winton_seaice_model.WintonSeaiceParameters`.
    """

    def __init__(
        self,
        grid: SlabGrid,
        params: WintonSeaiceParameters | None = None,
        *,
        name: str = "seaice",
        transport: IceTransportGrid | bool = False,
    ):
        """Initialize the Winton sea-ice model.

        Parameters
        ----------
        grid : SlabGrid
            The model's grid.
        params : WintonSeaiceParameters, optional
            Tunable parameters; defaults to
            :meth:`WintonSeaiceParameters.default`. They are what
            :meth:`initialize` builds the initial state from unless it is
            handed parameters of its own, and what the checks below are made
            against: validation applies to these concrete, construction-time
            values. ``initialize(params)`` is the differentiable entry point
            for the initial condition and takes traced values, so it is
            deliberately not re-validated there.
        name : str
            Component name in the coupler's workflow and carry. The default is
            the name the standard coupling wires the sea ice under
            (:func:`jem.exchangers.default_exchanges`).
        transport : IceTransportGrid or bool
            ``False`` (the default) integrates thermodynamics only. ``True``
            also advects and diffuses the ice, with grid metrics derived from a
            separable lon/lat ``grid`` (:meth:`IceTransportGrid.from_grid`);
            pass an :class:`IceTransportGrid` for any other grid. The
            diffusivity and substep count are in ``params``.

        Raises
        ------
        ValueError
            If a parameter is outside the range the scheme is defined on (a
            non-finite value, a non-positive lead-closing thickness, a substep
            or iteration count below one), or if the transport metrics do not
            have the grid's shape.

        """
        super().__init__(name=name, grid=grid)
        self.params = WintonSeaiceParameters.default() if params is None else params
        _validate_parameters(self.params)

        self.transport: IceTransportGrid | None
        if isinstance(transport, IceTransportGrid):
            self.transport = transport
        elif transport:
            self.transport = IceTransportGrid.from_grid(grid)
        else:
            self.transport = None
        if self.transport is not None:
            for metric_name in ("dx", "dy"):
                metric = getattr(self.transport, metric_name)
                if tuple(jnp.shape(metric)) != tuple(grid.shape):
                    raise ValueError(
                        f"Transport metric {metric_name} has shape "
                        f"{tuple(jnp.shape(metric))} but the grid has shape "
                        f"{tuple(grid.shape)}."
                    )
            logger.info("%s: ice transport enabled (cyclic_x=%s)", name, self.transport.cyclic_x)

    def _ocean_cells(self, params: WintonSeaiceParameters) -> jnp.ndarray:
        """Boolean mask of the cells this model integrates."""
        return self.grid.binary_mask == params.ocean_mask_value

    def initialize(self, params: WintonSeaiceParameters | None = None) -> Carry:
        """Build the initial sea-ice carry.

        Parameters
        ----------
        params : WintonSeaiceParameters, optional
            Parameters to start from; defaults to the ones the model was
            constructed with. ``initial_ice_thickness`` is read here and
            nowhere else, so this is the entry point that makes it
            differentiable. The same object goes into ``carry["params"]``, so
            the process parameters ``step`` reads are the ones the initial
            state was built from.

        """
        params = self._initial_params(params)
        shape = self.grid.shape
        ocean = self._ocean_cells(params)
        thickness = jnp.where(ocean, jnp.asarray(params.initial_ice_thickness), 0.0)
        fraction = jnp.where(thickness > 0, 1.0, 0.0)
        zeros = jnp.zeros(shape)
        # mypy cannot see the fields `tree_math.struct` gives the class.
        state = WintonState(  # type: ignore[call-arg]
            thickness, zeros, fraction,
            zeros + T_FREEZE, zeros + T_FREEZE, zeros + T_FREEZE,
        )
        forcing = WintonForcing.initial(shape)
        derived = _derived_from_state(
            WintonDerived.zeros(shape), state, forcing, ocean, params
        )
        return {"params": params, "state": state, "forcing": forcing, "derived": derived}

    def step(self, carry: Carry, time: CouplingTime) -> tuple[Carry, Diagnostics]:
        """Integrate the ice over one coupling step of ``time.dt`` seconds."""
        params = carry["params"]
        forcing = carry["forcing"]
        new_state, derived = self._advance(params, carry["state"], forcing, time.dt)
        diagnostics = {"state": new_state, "forcing": forcing, "derived": derived}
        return {"params": params, **diagnostics}, diagnostics

    def _advance(
        self,
        params: WintonSeaiceParameters,
        state: WintonState,
        forcing: WintonForcing,
        coupling_dt: float,
    ) -> tuple[WintonState, WintonDerived]:
        """Advance ``state`` by one coupling step and diagnose the step's budgets."""
        ocean = self._ocean_cells(params)
        n = params.n_substeps
        dt = coupling_dt / n
        h_min, f_min, h0 = params.min_ice_thickness, params.min_ice_fraction, params.lead_closing_thickness
        ice_albedo = params.ice_albedo
        snow_albedo = ice_albedo if params.snow_albedo is None else params.snow_albedo
        snow_melt_albedo = snow_albedo if params.snow_melt_albedo is None else params.snow_melt_albedo
        qbot = -C_ICE * T_FREEZE + L_ICE
        fc = forcing

        def flux_fn(Ts_c):
            return ice_surface_flux(Ts_c, fc.rlds, fc.air_temperature, fc.air_specific_humidity,
                                    fc.wind_speed, fc.normalized_surface_pressure,
                                    sfp=params.surface_flux, emis=params.emissivity)

        def substep(s, _):
            h, hs, f, T1, T2, Ts = (s.ice_thickness, s.snow_thickness, s.ice_fraction,
                                    s.upper_ice_temperature, s.lower_ice_temperature, s.ice_surface_temperature)
            icy = (h > h_min) & (f > f_min)
            h_safe = jnp.where(icy, h, 1.0)

            albedo = jnp.where(hs > 1e-3, jnp.where(Ts > -0.1, snow_melt_albedo, snow_albedo), ice_albedo)
            sw_abs = fc.rsds * (1.0 - albedo)
            T1n, T2n, Tsn, M_s, F_cb, sw_ocn = winton_temperature_step(
                h_safe, hs, T1, T2, Ts, flux_fn, sw_abs, dt, params.i0_fraction, params.ksolar,
                params.n_flux_iterations)
            F_nonsw, _ = flux_fn(Tsn)
            F_ice_atm = F_nonsw + sw_abs
            sst_c = fc.sea_surface_temperature - KELVIN
            F_b_raw = jnp.maximum(RHO_SW * C_W * B_MELT * USTAR_SLAB * (sst_c - T_FREEZE), 0.0)
            surplus_flux = jnp.maximum(-fc.ice_frazil_melt_energy, 0.0) / coupling_dt
            F_b = jnp.minimum(F_b_raw, surplus_flux)
            hn, hsn, q1n, q2n, e_ocn, vanished = winton_mass_step(
                h_safe, hs, q_from_T1(T1n), q_from_T2(T2n), M_s, F_b, F_cb, fc.snowfall, dt, h_min)
            # Hibler lateral melt: melting shrinks the ice-covered area by -(f/2h) dh; the volume the thermodynamics
            # paid for is f*hn, so the thickness of the remaining area is raised to keep fn*hn = f*hn exactly.
            dh_melt = jnp.maximum(h_safe - hn, 0.0)
            fn_t = f * (1.0 - 0.5 * dh_melt / h_safe)
            hn = jnp.where(fn_t > 0.0, f * hn / jnp.maximum(fn_t, 1e-12), hn)
            # frazil (per cell area, per substep): the ocean's freezing potential makes ice at enthalpy qbot
            frazil = jnp.maximum(fc.ice_frazil_melt_energy, 0.0) / n
            v_frazil = frazil / (RHO_ICE * qbot)
            alive_before = icy & ~vanished
            # ice below the thermodynamic thresholds (a sliver, e.g. diffused in by transport or freshly frozen):
            # kept and grown while the ocean keeps freezing, otherwise melted back into the ocean with its
            # enthalpy charged to the ocean, so nothing is ever discarded without its energy
            sliver = ~icy & (f * h > 0.0)
            keep_sliver = sliver & (v_frazil > 0.0)
            dispose = sliver & ~(v_frazil > 0.0)
            e_dispose = jnp.where(dispose, f * column_enthalpy_to_melt(h, hs, q_from_T1(T1), q_from_T2(T2)), 0.0)
            vol_old = jnp.where(alive_before, fn_t * hn, jnp.where(keep_sliver, f * h, 0.0))
            f_old = jnp.where(alive_before, fn_t, jnp.where(keep_sliver, f, 0.0))
            snow_old = jnp.where(alive_before, f * hsn, jnp.where(keep_sliver, f * hs, 0.0))   # snow volume per area
            E1_old = 0.5 * jnp.where(alive_before, fn_t * hn * q1n, jnp.where(keep_sliver, f * h * q_from_T1(T1), 0.0))
            E2_old = 0.5 * jnp.where(alive_before, fn_t * hn * q2n, jnp.where(keep_sliver, f * h * q_from_T2(T2), 0.0))
            vol = vol_old + v_frazil
            fn = jnp.clip(f_old + v_frazil / h0, 0.0, 1.0)
            alive = vol > 0.0
            fs = jnp.where(alive, jnp.maximum(fn, 1e-12), 1.0)
            hn = jnp.where(alive, vol / fs, 0.0)
            hsn = jnp.where(alive, snow_old / fs, 0.0)
            # frazil enters both layers at qbot, so the column enthalpy grows by exactly v_frazil*qbot
            half_v = jnp.where(alive, 0.5 * vol, 1.0)
            q1n = (E1_old + 0.5 * v_frazil * qbot) / half_v
            q2n = (E2_old + 0.5 * v_frazil * qbot) / half_v
            T1n = jnp.where(alive, jnp.clip(T1_from_q(q1n), -80.0, T_MELT), T_FREEZE)
            T2n = jnp.where(alive, jnp.clip(T2_from_q(q2n), -80.0, 0.0), T_FREEZE)
            Tsn = jnp.where(alive, jnp.where(alive_before, Tsn, jnp.where(keep_sliver, Ts, T_FREEZE)), T_FREEZE)
            fn = jnp.where(alive, fn, 0.0)
            # energy to the ocean (downward positive), per cell area, over this substep: the atmosphere's flux over
            # the sea cell minus what the ice absorbed, plus what the ice passes down, minus the basal heat it draws;
            # the fusion enthalpy of snow that landed on ice (the atmosphere condensed it as liquid) and the
            # enthalpy of disposed slivers are charged here so the coupled budget closes
            down = (fc.atm_sea_heat_flux - jnp.where(icy, f * F_ice_atm, 0.0)
                    + jnp.where(icy, f * (sw_ocn + e_ocn / dt - F_b), 0.0)
                    + jnp.where(icy & ~vanished, f * fc.snowfall * L_ICE, 0.0)
                    - e_dispose / dt)
            new_s = WintonState(hn * ocean, hsn * ocean, fn * ocean, T1n, T2n, Tsn)
            intercepted = jnp.where(icy & ~vanished, f * fc.snowfall * dt, 0.0)   # snow mass that landed on ice, kg/m2
            diag = (jnp.where(icy, M_s, 0.0), jnp.where(icy, -(F_b + F_cb), 0.0), jnp.where(icy, F_ice_atm, 0.0), -down,
                    intercepted, jnp.where(icy, albedo, ice_albedo))
            return new_s, diag

        new_state, diags = jax.lax.scan(substep, state, None, length=n)
        m_s, growth, f_ice_atm, q_up, intercepted, albedo = (d.mean(axis=0) for d in diags)

        # water budget: ice+snow mass change minus snowfall intercepted by ice = water taken from the ocean
        def mass(st):
            return st.ice_fraction * (RHO_ICE * st.ice_thickness + RHO_SNOW * st.snow_thickness)
        fw_up = (mass(new_state) - mass(state) - intercepted * n) / coupling_dt

        def energy(st):   # J/m2 per cell area, relative to liquid water at 0 C (ice/snow are negative)
            return -st.ice_fraction * column_enthalpy_to_melt(
                st.ice_thickness, st.snow_thickness,
                q_from_T1(st.upper_ice_temperature), q_from_T2(st.lower_ice_temperature))
        de_dt = (energy(new_state) - energy(state)) / coupling_dt     # thermodynamic tendency: exchange with ocean/atm
        # transport after the local budgets above: it moves ice between cells and exchanges nothing with the ocean
        thermo_state = new_state
        if self.transport is not None:
            new_state = self._apply_transport(params, new_state, forcing, coupling_dt, ocean)
        de_transport = (energy(new_state) - energy(thermo_state)) / coupling_dt   # convergence of ice enthalpy

        # every derived field from the final state
        derived = _derived_from_state(
            WintonDerived.zeros(self.grid.shape), new_state, forcing, ocean, params
        ).replace(
            ocean_heat_flux_up=q_up,
            surface_melt_flux=m_s,
            basal_growth_flux=growth,
            ice_atm_heat_flux=f_ice_atm,
            ocean_freshwater_flux_up=fw_up,
            ice_albedo=albedo,
            ice_energy_tendency=de_dt,
            ice_energy_transport=de_transport,
            ocean_frazil_heating=jnp.maximum(forcing.ice_frazil_melt_energy, 0.0) / coupling_dt,
        )
        return new_state, derived

    def _apply_transport(
        self,
        params: WintonSeaiceParameters,
        s: WintonState,
        fc: WintonForcing,
        coupling_dt: float,
        ocean: jnp.ndarray,
    ) -> WintonState:
        """Move the conserved quantities (area, ice and snow volume, layer enthalpies, area-weighted Ts),
        then rebuild the state; convergence beyond full cover ridges (thickness up, fraction capped).
        """
        assert self.transport is not None
        tr = self.transport
        h_min_frac, h0 = params.min_ice_fraction, params.lead_closing_thickness
        f, h, hs = s.ice_fraction, s.ice_thickness, s.snow_thickness
        V = f * h
        # all transported quantities must be non-negative (the scheme clips at zero): Ts is in degC <= 0, so carry -Ts
        fields = (f, V, f * hs, 0.5 * V * q_from_T1(s.upper_ice_temperature), 0.5 * V * q_from_T2(s.lower_ice_temperature),
                  -f * s.ice_surface_temperature)
        fn, Vn, Vsn, Q1n, Q2n, FTn = transport_fields(
            fields, fc.ice_velocity_u, fc.ice_velocity_v, jnp.asarray(tr.dx), jnp.asarray(tr.dy), ocean, coupling_dt,
            params.transport_diffusivity, params.transport_n_substeps, tr.cyclic_x, compact_threshold=0.98)
        # Ice arriving in (nearly) empty cells comes with a diluted fraction; give it at least the lead-closing
        # thickness h0 (fraction from volume) and a fraction above the thermodynamics' threshold. Nothing is
        # discarded here: sub-threshold slivers are kept by the next thermodynamic step while the ocean freezes,
        # or melted back into it with their enthalpy charged to the ocean, so nothing leaks.
        diluted = fn < 2.0 * h_min_frac
        f_use = jnp.where(diluted, jnp.clip(jnp.minimum(1.0, Vn / h0), 2.0 * h_min_frac, 1.0), jnp.minimum(fn, 1.0))
        alive = Vn > 0.0
        fs = jnp.where(alive, f_use, 1.0)
        half_v = jnp.where(alive, 0.5 * Vn, 1.0)
        Ts_avg = jnp.clip(-FTn / jnp.maximum(fn, 1e-12), -80.0, 0.0)    # transported area-weighted mean
        return WintonState(  # type: ignore[call-arg]
            jnp.where(alive, Vn / fs, 0.0), jnp.where(alive, Vsn / fs, 0.0), jnp.where(alive, f_use, 0.0),
            jnp.where(alive, jnp.clip(T1_from_q(Q1n / half_v), -80.0, T_MELT), T_FREEZE),
            jnp.where(alive, jnp.clip(T2_from_q(Q2n / half_v), -80.0, 0.0), T_FREEZE),
            jnp.where(alive, Ts_avg, T_FREEZE))

    def _create_xarray_data_vars(self, diagnostics: Diagnostics) -> dict[str, Any]:
        """Create xarray data variables for sea-ice output."""
        dims = ("time",) + tuple(self.grid.dims)
        s, d, fc = diagnostics["state"], diagnostics["derived"], diagnostics["forcing"]
        state, derived, forcing = role_attrs("state"), role_attrs("derived"), role_attrs("forcing")
        return {
            "ice_thickness": (dims, s.ice_thickness, {"units": "m", "long_name": "thickness of ice-covered part", **state}),
            "snow_thickness": (dims, s.snow_thickness, {"units": "m", "long_name": "thickness of snow on the ice-covered part", **state}),
            "ice_fraction": (dims, s.ice_fraction, {"units": "1", "long_name": "ice-covered fraction of the cell", **state}),
            "upper_ice_temperature": (dims, s.upper_ice_temperature + KELVIN, {"units": "K", "long_name": "upper ice layer temperature", **state}),
            "lower_ice_temperature": (dims, s.lower_ice_temperature + KELVIN, {"units": "K", "long_name": "lower ice layer temperature", **state}),
            "ice_surface_temperature": (dims, d.ice_surface_temperature_K, {"units": "K", "long_name": "ice surface temperature", **derived}),
            "ice_volume": (dims, d.ice_volume, {"units": "m", "long_name": "cell-mean ice thickness", **derived}),
            "snow_volume": (dims, d.snow_volume, {"units": "m", "long_name": "cell-mean snow thickness", **derived}),
            "effective_sea_surface_temperature": (dims, d.effective_sea_surface_temperature, {"units": "K", "long_name": "ice-fraction-weighted sea surface temperature", **derived}),
            "ocean_heat_flux_up": (dims, d.ocean_heat_flux_up, {"units": "W m-2", "long_name": "heat flux from the ocean into the ice-covered cell (positive upward)", **derived}),
            "surface_melt_flux": (dims, d.surface_melt_flux, {"units": "W m-2", "long_name": "surface melt energy flux of the ice-covered part", **derived}),
            "ocean_freshwater_flux_up": (dims, d.ocean_freshwater_flux_up, {"units": "kg m-2 s-1", "long_name": "freshwater removed from the ocean by ice growth (positive upward)", **derived}),
            "ice_albedo": (dims, d.ice_albedo, {"units": "1", "long_name": "effective albedo of the ice-covered part", **derived}),
            "ice_energy_tendency": (dims, d.ice_energy_tendency, {"units": "W m-2", "long_name": "thermodynamic d/dt of ice+snow enthalpy per cell area (exchange with ocean and atmosphere; transport excluded)", **derived}),
            "ice_energy_transport": (dims, d.ice_energy_transport, {"units": "W m-2", "long_name": "d/dt of ice+snow enthalpy per cell area by transport convergence", **derived}),
            "ocean_frazil_heating": (dims, d.ocean_frazil_heating, {"units": "W m-2", "long_name": "heat added to the ocean top layer by clamping it at the freezing point (frazil latent heat)", **derived}),
            "basal_growth_flux": (dims, d.basal_growth_flux, {"units": "W m-2", "long_name": "energy flux out of the ice base of the ice-covered part (positive grows ice)", **derived}),
            "ice_atm_heat_flux": (dims, d.ice_atm_heat_flux, {"units": "W m-2", "long_name": "heat flux from the atmosphere into the ice-covered part (positive downward)", **derived}),
            forcing_variable("ice_frazil_melt_energy"): (dims, fc.ice_frazil_melt_energy, {"units": "J m-2", "long_name": "Freeze/melt potential (frzmlt) this component was forced with: positive forms ice, negative melts ice", **forcing}),
            forcing_variable("ice_velocity_u"): (dims, fc.ice_velocity_u, {"units": "m s-1", "long_name": "ice drift along grid x", **forcing}),
            forcing_variable("ice_velocity_v"): (dims, fc.ice_velocity_v, {"units": "m s-1", "long_name": "ice drift along grid y", **forcing}),
        }


def _derived_from_state(
    derived: Any,
    state: Any,
    forcing: Any,
    ocean: jnp.ndarray,
    params: WintonSeaiceParameters,
) -> Any:
    """Fill the diagnostics that are pure functions of the final state.

    Shared by ``initialize`` and ``step`` so the initial carry and the stepped
    carry cannot diverge in how they are derived. Cells the model does not
    integrate report ``MASKED_SURFACE_TEMPERATURE`` for the ice surface, as
    every slab model does for a cell it does not integrate.
    """
    f = state.ice_fraction
    surface_K = jnp.where(
        ocean, state.ice_surface_temperature + KELVIN, MASKED_SURFACE_TEMPERATURE
    )
    return derived.replace(
        ice_fraction=f,
        effective_sea_surface_temperature=(1.0 - f) * forcing.sea_surface_temperature + f * surface_K,
        ice_surface_temperature_K=surface_K,
        ice_volume=f * state.ice_thickness,
        snow_volume=f * state.snow_thickness,
        ice_albedo=jnp.zeros_like(f) + params.ice_albedo,
    )


def _validate_parameters(params: WintonSeaiceParameters) -> None:
    """Refuse parameters the scheme is undefined for, naming the offender.

    Runs on the concrete construction-time values, which is why it can read
    them as Python floats. Finiteness is checked as well as sign, because a NaN
    compares False against every threshold and so fails quietly at run time.
    """
    def finite(field: str, value: Any) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field} must be finite; got {number!r}.")
        return number

    def at_least(field: str, value: Any, floor: float, *, strict: bool) -> None:
        number = finite(field, value)
        if number < floor or (strict and number == floor):
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{field} must be {relation} {floor:g}; got {number!r}.")

    for field in ("ice_albedo", "i0_fraction"):
        number = finite(field, getattr(params, field))
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{field} must lie in [0, 1]; got {number!r}.")
    for field in ("snow_albedo", "snow_melt_albedo"):
        value = getattr(params, field)
        if value is not None and not 0.0 <= finite(field, value) <= 1.0:
            raise ValueError(f"{field} must lie in [0, 1] or be None; got {value!r}.")
    at_least("ksolar", params.ksolar, 0.0, strict=False)
    at_least("lead_closing_thickness", params.lead_closing_thickness, 0.0, strict=True)
    at_least("min_ice_thickness", params.min_ice_thickness, 0.0, strict=False)
    at_least("min_ice_fraction", params.min_ice_fraction, 0.0, strict=False)
    at_least("initial_ice_thickness", params.initial_ice_thickness, 0.0, strict=False)
    at_least("emissivity", params.emissivity, 0.0, strict=False)
    at_least("transport_diffusivity", params.transport_diffusivity, 0.0, strict=False)
    for field in ("n_substeps", "n_flux_iterations", "transport_n_substeps"):
        count = getattr(params, field)
        if int(count) != count or int(count) < 1:
            raise ValueError(f"{field} must be an integer of at least 1; got {count!r}.")
