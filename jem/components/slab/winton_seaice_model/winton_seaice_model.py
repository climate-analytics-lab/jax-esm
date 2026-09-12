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

from typing import Any

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import tree_math

from jem.components.slab.base import _DEFAULT_START_DATETIME, SlabModelBase
from jem.components.slab.grid import SlabGrid
from jem.components.slab.winton_seaice_model.ice_transport import transport_fields
from jem.utils.bulk_op import stack_objects

# ---------------------------------------------------------------- constants (Winton 2000 Table 1)
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
T_FREEZE = -1.8
Q_SNOW = L_ICE           # thsice qsnow
FLOOD_FAC = (RHO_SW - RHO_ICE) / RHO_SNOW
B_MELT = 0.006           # thsice bMeltCoef
USTAR_SLAB = 5.0e-3      # thsice ustar for zero ocean velocity: sqrt(25e-6)
# Bulk-flux constants and parameters over ice come from JCM at call time (jcm.constants.physical_constants and the
# SPEEDY SurfaceFluxParameters / ModRadConParameters the coupled atmosphere uses), so an override on the JCM side
# is seen here too; the model constructor takes the parameter objects explicitly.
from jcm import constants as jcm_constants
from jcm.physics.speedy.params import ModRadConParameters, SurfaceFluxParameters

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
    """SPEEDY-style non-solar surface flux over ice (positive downward) and dF/dTs.

    sfp: JCM SurfaceFluxParameters (exchange coefficient chs, gust speed vgust, stability dtheta/fstab/lscasym);
    emis: longwave emissivity (JCM ModRadConParameters.emisfc). Defaults are JCM's defaults; physical constants
    (cpd, rd, p0, sbc, alhc) are read from jcm.constants.physical_constants at call time.
    """
    sfp = SurfaceFluxParameters.default() if sfp is None else sfp
    emis = ModRadConParameters.default().emisfc if emis is None else emis
    c = jcm_constants.physical_constants
    astab = jnp.where(sfp.lscasym, 0.5, 1.0)            # SPEEDY: asymmetric stability coefficient
    Ts = Ts_c + KELVIN
    rho = c.p0 * p_air / (c.rd * t_air)
    dth = jnp.where(Ts > t_air, jnp.minimum(sfp.dtheta, Ts - t_air), jnp.maximum(-sfp.dtheta, astab * (Ts - t_air)))
    denv = rho * jnp.sqrt(wind ** 2 + sfp.vgust ** 2) * (1.0 + dth * sfp.fstab / sfp.dtheta)
    q_s, dq_s = qsat_ice(Ts, c.p0 * p_air)
    F = rlds - emis * c.sbc * Ts ** 4 - sfp.chs * c.cpd * denv * (Ts - t_air) - sfp.chs * denv * c.alhc * (q_s - q_air)
    dF = -4.0 * emis * c.sbc * Ts ** 3 - sfp.chs * c.cpd * denv - sfp.chs * denv * c.alhc * dq_s
    return F, dF


# ---------------------------------------------------------------- temperature step (thsice_solve4temp)
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
    """
    h1 = h2 = 0.5 * h
    etop = jnp.maximum(M_s, 0.0) * dt
    ebot = (F_cb + F_b) * dt          # > 0 melts the base, < 0 freezes
    # --- top melt: snow, layer 1, layer 2
    rq = RHO_SNOW * Q_SNOW
    d = jnp.minimum(etop / rq, hs); hs = hs - d; etop = etop - d * rq
    rq = RHO_ICE * q1
    d = jnp.minimum(etop / rq, h1); h1 = h1 - d; etop = etop - d * rq
    rq = RHO_ICE * q2
    d = jnp.minimum(etop / rq, h2); h2 = h2 - d; etop = etop - d * rq
    # --- basal growth
    qbot = -C_ICE * T_FREEZE + L_ICE
    dhi = jnp.where(ebot < 0.0, -ebot / (qbot * RHO_ICE), 0.0)
    q2 = jnp.where(h2 + dhi > 0.0, (h2 * q2 + dhi * qbot) / jnp.maximum(h2 + dhi, 1e-300), q2)
    h2 = h2 + dhi
    ebot = jnp.maximum(ebot, 0.0)
    # --- basal melt: layer 2, layer 1, snow
    rq = RHO_ICE * q2
    d = jnp.minimum(ebot / rq, h2); h2 = h2 - d; ebot = ebot - d * rq
    rq = RHO_ICE * q1
    d = jnp.minimum(ebot / rq, h1); h1 = h1 - d; ebot = ebot - d * rq
    rq = RHO_SNOW * Q_SNOW
    d = jnp.minimum(ebot / rq, hs); hs = hs - d; ebot = ebot - d * rq
    surplus = etop + ebot
    # --- vanishing column: ocean supplies what is left
    h_tot = h1 + h2
    vanished = h_tot < h_min
    left = RHO_ICE * (h1 * q1 + h2 * q2) + RHO_SNOW * Q_SNOW * hs
    surplus = jnp.where(vanished, surplus - left, surplus)
    h1 = jnp.where(vanished, 0.0, h1); h2 = jnp.where(vanished, 0.0, h2); hs = jnp.where(vanished, 0.0, hs)
    # --- snowfall (only on surviving ice)
    hs = hs + jnp.where(vanished, 0.0, snowfall * dt / RHO_SNOW)
    # --- flooding: snow below the waterline becomes ice in layer 1
    h_tot = h1 + h2
    flood = (hs > h_tot * FLOOD_FAC) & (h_tot > 0.0)
    dhs = jnp.where(flood, (hs - h_tot * FLOOD_FAC) * RHO_ICE / RHO_SW, 0.0)
    dhi = dhs * RHO_SNOW / RHO_ICE
    q1 = jnp.where(flood, (RHO_ICE * q1 * h1 + RHO_SNOW * Q_SNOW * dhs) / jnp.maximum(RHO_ICE * (h1 + dhi), 1e-300), q1)
    h1 = h1 + dhi; hs = hs - dhs
    # --- re-equalise layers (thsice inlined THSICE_RESHAPE_LAYERS)
    h_tot = h1 + h2
    hlyr = 0.5 * h_tot
    safe_hlyr = jnp.maximum(hlyr, 1e-300)
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
    sim_time: jnp.ndarray
    ice_thickness: jnp.ndarray          # m, over the ice-covered fraction
    snow_thickness: jnp.ndarray         # m
    ice_fraction: jnp.ndarray
    upper_ice_temperature: jnp.ndarray  # degC
    lower_ice_temperature: jnp.ndarray  # degC
    ice_surface_temperature: jnp.ndarray  # degC


@tree_math.struct
class WintonForcing:
    rsds: jnp.ndarray
    rlds: jnp.ndarray
    air_temperature: jnp.ndarray        # K
    air_specific_humidity: jnp.ndarray  # kg/kg
    wind_speed: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray
    snowfall: jnp.ndarray               # kg/m2/s
    sea_surface_temperature: jnp.ndarray   # K
    ocean_frazil_melt_energy: jnp.ndarray  # J/m2 per coupling step, + = freezing potential
    atm_sea_heat_flux: jnp.ndarray         # W/m2 downward, atmosphere's flux over the sea fraction
    ice_velocity_u: jnp.ndarray            # m/s along the grid's x (used only with transport enabled)
    ice_velocity_v: jnp.ndarray            # m/s along the grid's y


@tree_math.struct
class WintonDerived:
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
    ice_energy_tendency: jnp.ndarray        # W/m2 per cell area; rate of change of the ice+snow enthalpy (relative to water at 0 C)


class WintonSeaiceModel(SlabModelBase):
    """JEM component wrapping the Winton/thsice thermodynamics with a Hibler lead closure."""

    def __init__(
        self,
        grid: SlabGrid,
        start_datetime: jdt.Datetime = _DEFAULT_START_DATETIME,
        timestep: float = 86400.0,
        n_substeps: int = 4,
        n_flux_iterations: int = 3,
        ice_albedo: float = 0.60,            # bare ice
        snow_albedo: float | None = None,    # dry snow on ice (None: snow takes ice_albedo)
        snow_melt_albedo: float | None = None,  # melting snow (surface at 0 C)
        i0_fraction: float = 0.3,
        ksolar: float = 1.5,
        lead_closing_thickness: float = 0.5,
        min_ice_thickness: float = 0.01,
        min_ice_fraction: float = 0.01,
        initial_ice_thickness: float = 0.0,
        mask_value: float = 0.0,
        calendar: str = "365_day",
        transport: dict | None = None,
        surface_flux_parameters: SurfaceFluxParameters | None = None,
        emissivity=None,
    ):
        """transport: None (thermodynamics only) or dict(dx=, dy= (m, cell centres, grid shape), cyclic_x=True,
        diffusivity=2e4 m2/s, n_substeps=12): advect the ice with forcing.ice_velocity_{u,v} and diffuse it.
        surface_flux_parameters / emissivity: the JCM SurfaceFluxParameters and longwave emissivity used for the
        bulk fluxes over ice; pass the same objects the coupled atmosphere runs with (defaults: JCM's defaults)."""
        self.surface_flux_parameters = SurfaceFluxParameters.default() if surface_flux_parameters is None else surface_flux_parameters
        self.emissivity = ModRadConParameters.default().emisfc if emissivity is None else emissivity
        self.n_substeps = n_substeps
        self.n_flux_iterations = n_flux_iterations
        self.ice_albedo = ice_albedo
        self.snow_albedo = ice_albedo if snow_albedo is None else snow_albedo
        self.snow_melt_albedo = self.snow_albedo if snow_melt_albedo is None else snow_melt_albedo
        self.i0_fraction = i0_fraction
        self.ksolar = ksolar
        self.lead_closing_thickness = lead_closing_thickness
        self.min_ice_thickness = min_ice_thickness
        self.min_ice_fraction = min_ice_fraction
        self.initial_ice_thickness = initial_ice_thickness
        self.mask_value = mask_value
        super().__init__(name="WintonSeaiceModel", grid=grid, start_datetime=start_datetime,
                         timestep=timestep, calendar=calendar)
        self.transport = None
        if transport is not None:
            self.transport = {**dict(cyclic_x=True, diffusivity=2e4, n_substeps=12), **transport}
            self.transport["dx"] = jnp.asarray(self.transport["dx"], dtype=float)
            self.transport["dy"] = jnp.asarray(self.transport["dy"], dtype=float)

    def initialize(self):
        shape = self.grid.shape
        ocn = self.grid.binary_mask == self.mask_value
        h = jnp.where(ocn, self.initial_ice_thickness, 0.0)
        f = jnp.where(h > 0, 1.0, 0.0)
        z = jnp.zeros(shape)
        state = WintonState(jnp.zeros(()), h, z, f, z + T_FREEZE, z + T_FREEZE, z + T_FREEZE)
        forcing = WintonForcing(z, z, z + 288.15, z + 1e-3, z + 5.0, z + 1.0, z, z + 288.15, z, z, z, z)
        derived = WintonDerived(f, z + 288.15, z, z + KELVIN + T_FREEZE, h * f, z, z, z, z, z, z + self.ice_albedo, z)
        return {"state": state, "forcing": forcing, "derived": derived}

    def _create_step_function_body(self):
        ocn = self.grid.binary_mask == self.mask_value
        n = self.n_substeps
        dt = self.timestep / n
        h_min, f_min, h0 = self.min_ice_thickness, self.min_ice_fraction, self.lead_closing_thickness
        qbot = -C_ICE * T_FREEZE + L_ICE

        def substep(carry, _):
            s, fc = carry
            h, hs, f, T1, T2, Ts = (s.ice_thickness, s.snow_thickness, s.ice_fraction,
                                    s.upper_ice_temperature, s.lower_ice_temperature, s.ice_surface_temperature)
            icy = (h > h_min) & (f > f_min)
            h_safe = jnp.where(icy, h, 1.0)

            def flux_fn(Ts_c):
                return ice_surface_flux(Ts_c, fc.rlds, fc.air_temperature, fc.air_specific_humidity,
                                        fc.wind_speed, fc.normalized_surface_pressure,
                                        sfp=self.surface_flux_parameters, emis=self.emissivity)

            albedo = jnp.where(hs > 1e-3, jnp.where(Ts > -0.1, self.snow_melt_albedo, self.snow_albedo), self.ice_albedo)
            sw_abs = fc.rsds * (1.0 - albedo)
            T1n, T2n, Tsn, M_s, F_cb, sw_ocn = winton_temperature_step(
                h_safe, hs, T1, T2, Ts, flux_fn, sw_abs, dt, self.i0_fraction, self.ksolar, self.n_flux_iterations)
            F_nonsw, _ = flux_fn(Tsn)
            F_ice_atm = F_nonsw + sw_abs
            sst_c = fc.sea_surface_temperature - KELVIN
            F_b_raw = jnp.maximum(RHO_SW * C_W * B_MELT * USTAR_SLAB * (sst_c - T_FREEZE), 0.0)
            surplus_flux = jnp.maximum(-fc.ocean_frazil_melt_energy, 0.0) / self.timestep
            F_b = jnp.minimum(F_b_raw, surplus_flux)
            hn, hsn, q1n, q2n, e_ocn, vanished = winton_mass_step(
                h_safe, hs, q_from_T1(T1n), q_from_T2(T2n), M_s, F_b, F_cb, fc.snowfall, dt, h_min)
            # Hibler lateral melt on the thermodynamic thickness loss
            dh_melt = jnp.maximum(h_safe - hn, 0.0)
            fn = f * (1.0 - 0.5 * dh_melt / h_safe)
            # frazil (per cell area, per substep) makes new ice at Tf in leads / under the ice
            frazil = jnp.maximum(fc.ocean_frazil_melt_energy, 0.0) / n
            v_frazil = frazil / (RHO_ICE * qbot)
            alive_before = icy & ~vanished
            vol = jnp.where(alive_before, fn * hn, 0.0) + v_frazil
            fn = jnp.where(alive_before, fn, 0.0) + v_frazil / h0
            fn = jnp.clip(fn, 0.0, 1.0)
            alive = (vol > h_min * f_min) & (fn > f_min)
            hn = jnp.where(alive, vol / jnp.maximum(fn, f_min), 0.0)
            hsn = jnp.where(alive & alive_before, hsn * jnp.where(alive_before, f, 1.0) / jnp.maximum(fn, f_min), 0.0)  # snow volume conserved
            fn = jnp.where(alive, fn, 0.0)
            # frazil added at Tf: layer-2 enthalpy mixes toward qbot (weight by volume)
            w_new = jnp.where(alive, v_frazil / jnp.maximum(vol, 1e-300), 0.0)
            q2n = jnp.where(alive_before, (1.0 - w_new) * q2n + w_new * qbot, qbot)
            q1n = jnp.where(alive_before, q1n, q_from_T1(T_FREEZE))
            T1n = jnp.where(alive, T1_from_q(q1n), T_FREEZE)
            T2n = jnp.where(alive, T2_from_q(q2n), T_FREEZE)
            Tsn = jnp.where(alive, jnp.where(alive_before, Tsn, T_FREEZE), T_FREEZE)
            # energy to the ocean (downward positive), per cell area, over this substep
            down = (fc.atm_sea_heat_flux - jnp.where(icy, f * F_ice_atm, 0.0)
                    + jnp.where(icy, f * (sw_ocn + e_ocn / dt - F_b), 0.0))
            new_s = WintonState(s.sim_time + dt, hn * ocn, hsn * ocn, fn * ocn, T1n, T2n, Tsn)
            intercepted = jnp.where(icy & ~vanished, f * fc.snowfall * dt, 0.0)   # snow mass that landed on ice, kg/m2
            diag = (jnp.where(icy, M_s, 0.0), jnp.where(icy, -(F_b + F_cb), 0.0), jnp.where(icy, F_ice_atm, 0.0), -down,
                    intercepted, jnp.where(icy, albedo, self.ice_albedo))
            return (new_s, fc), diag

        def apply_transport(s, fc):
            """Move the conserved quantities (area, ice and snow volume, layer enthalpies, area-weighted Ts),
            then rebuild the state; convergence beyond full cover ridges (thickness up, fraction capped)."""
            tr = self.transport
            f, h, hs = s.ice_fraction, s.ice_thickness, s.snow_thickness
            V = f * h
            # all transported quantities must be non-negative (the scheme clips at zero): Ts is in degC <= 0, so carry -Ts
            fields = (f, V, f * hs, 0.5 * V * q_from_T1(s.upper_ice_temperature), 0.5 * V * q_from_T2(s.lower_ice_temperature),
                      -f * s.ice_surface_temperature)
            fn, Vn, Vsn, Q1n, Q2n, FTn = transport_fields(
                fields, fc.ice_velocity_u, fc.ice_velocity_v, tr["dx"], tr["dy"], ocn, self.timestep,
                tr["diffusivity"], tr["n_substeps"], tr["cyclic_x"], compact=f >= 0.98)
            # Ice arriving in (nearly) empty cells comes with a diluted fraction; give it at least the lead-closing
            # thickness h0 (fraction from volume) and a fraction above the thermodynamics' threshold. Nothing is
            # discarded here: sub-threshold slivers are dropped by the next thermodynamic step, whose water budget
            # hands them to the ocean, so transport itself is exactly conservative.
            diluted = fn < 2.0 * f_min
            f_use = jnp.where(diluted, jnp.clip(jnp.minimum(1.0, Vn / h0), 2.0 * f_min, 1.0), jnp.minimum(fn, 1.0))
            alive = Vn > 0.0
            fs = jnp.where(alive, f_use, 1.0)
            half_v = jnp.where(alive, 0.5 * Vn, 1.0)
            Ts_avg = jnp.clip(-FTn / jnp.maximum(fn, 1e-12), -80.0, 0.0)    # transported area-weighted mean
            return WintonState(
                s.sim_time, jnp.where(alive, Vn / fs, 0.0), jnp.where(alive, Vsn / fs, 0.0), jnp.where(alive, f_use, 0.0),
                jnp.where(alive, T1_from_q(Q1n / half_v), T_FREEZE), jnp.where(alive, T2_from_q(Q2n / half_v), T_FREEZE),
                jnp.where(alive, Ts_avg, T_FREEZE))

        def step_function(carry, step):
            state, forcing = carry["state"], carry["forcing"]
            (new_state, _), diags = jax.lax.scan(substep, (state, forcing), None, length=n)
            m_s, growth, f_ice_atm, q_up, intercepted, albedo = (d.mean(axis=0) for d in diags)
            f = new_state.ice_fraction
            # water budget: ice+snow mass change minus snowfall intercepted by ice = water taken from the ocean
            def mass(st):
                return st.ice_fraction * (RHO_ICE * st.ice_thickness + RHO_SNOW * st.snow_thickness)
            fw_up = (mass(new_state) - mass(state) - intercepted * n) / self.timestep
            def energy(st):   # J/m2 per cell area, relative to liquid water at 0 C (ice/snow are negative)
                return -st.ice_fraction * column_enthalpy_to_melt(st.ice_thickness, st.snow_thickness,
                                                                   q_from_T1(st.upper_ice_temperature), q_from_T2(st.lower_ice_temperature))
            de_dt = (energy(new_state) - energy(state)) / self.timestep
            # transport after the local budgets above: it moves ice between cells and exchanges nothing with the ocean
            if self.transport is not None:
                new_state = apply_transport(new_state, forcing)
            Ts_K = new_state.ice_surface_temperature + KELVIN
            derived = WintonDerived(
                ice_fraction=f,
                effective_sea_surface_temperature=(1.0 - f) * forcing.sea_surface_temperature + f * Ts_K,
                ocean_heat_flux_up=q_up,
                ice_surface_temperature_K=Ts_K,
                ice_volume=f * new_state.ice_thickness,
                snow_volume=f * new_state.snow_thickness,
                surface_melt_flux=m_s,
                basal_growth_flux=growth,
                ice_atm_heat_flux=f_ice_atm,
                ocean_freshwater_flux_up=fw_up,
                ice_albedo=albedo,
                ice_energy_tendency=de_dt,
            )
            result = {"state": new_state, "forcing": forcing, "derived": derived}
            return result, stack_objects([result])

        return step_function

    def _create_xarray_data_vars(self, predictions) -> dict[str, Any]:
        dims = ("time",) + tuple(self.grid.dims)
        s, d = predictions["state"], predictions["derived"]
        return {
            "ice_thickness": (dims, s.ice_thickness, {"units": "m", "long_name": "thickness of ice-covered part"}),
            "snow_thickness": (dims, s.snow_thickness, {"units": "m"}),
            "ice_fraction": (dims, s.ice_fraction, {"units": "1"}),
            "ice_volume": (dims, d.ice_volume, {"units": "m", "long_name": "cell-mean ice thickness"}),
            "snow_volume": (dims, d.snow_volume, {"units": "m"}),
            "ice_surface_temperature": (dims, d.ice_surface_temperature_K, {"units": "K"}),
            "upper_ice_temperature": (dims, s.upper_ice_temperature + KELVIN, {"units": "K"}),
            "lower_ice_temperature": (dims, s.lower_ice_temperature + KELVIN, {"units": "K"}),
            "effective_sea_surface_temperature": (dims, d.effective_sea_surface_temperature, {"units": "K"}),
            "ocean_heat_flux_up": (dims, d.ocean_heat_flux_up, {"units": "W m-2"}),
            "surface_melt_flux": (dims, d.surface_melt_flux, {"units": "W m-2"}),
            "ocean_freshwater_flux_up": (dims, d.ocean_freshwater_flux_up, {"units": "kg m-2 s-1"}),
            "ice_albedo": (dims, d.ice_albedo, {"units": "1"}),
            "ice_energy_tendency": (dims, d.ice_energy_tendency, {"units": "W m-2", "long_name": "d/dt of ice+snow enthalpy per cell area"}),
            "ocean_frazil_heating": (dims, jnp.maximum(predictions["forcing"].ocean_frazil_melt_energy, 0.0) / self.timestep,
                                     {"units": "W m-2", "long_name": "heat added to the ocean top layer by clamping it at the freezing point (frazil latent heat)"}),
            "basal_growth_flux": (dims, d.basal_growth_flux, {"units": "W m-2"}),
            "ice_atm_heat_flux": (dims, d.ice_atm_heat_flux, {"units": "W m-2"}),
            "ice_velocity_u": (dims, predictions["forcing"].ice_velocity_u, {"units": "m s-1", "long_name": "ice drift along grid x"}),
            "ice_velocity_v": (dims, predictions["forcing"].ice_velocity_v, {"units": "m s-1", "long_name": "ice drift along grid y"}),
        }
