"""Helper functions for computing biogeochemical coefficients."""

import jax.numpy as jnp
from jax.experimental import checkify

def compute_solubility_K0(temp, salinity):
    """
    Computes CO2 solubility (K0) using the Weiss (1974) parameterization.

    Parameters:
    -----------
    temp : float or ndarray
        Sea surface temperature matrix in Kelvin.
    salinity : float or ndarray
        Salinity matrix in practical salinity units (psu).

    Returns:
    --------
    K0 : float or ndarray
        Solubility constant in mol/ (dm^-3 * atm) = M / atm.
    """
    T_100 = temp / 100.0

    ln_K0 = (-58.0931 + (90.5069 / T_100) + 22.2940 * jnp.log(T_100) +
             salinity * (0.027766 - 0.025888 * T_100 + 0.0050578 * T_100**2))

    return jnp.exp(ln_K0)


def compute_gas_transfer_velocity(temp, wind_speed):
    """
    Computes the air-sea gas transfer velocity (k) for CO2 in seawater
    using the Wanninkhof (2014) parameterization.

    Parameters:
    -----------
    temp : float or ndarray
        Sea surface temperature matrix in Kelvin
    wind_speed : float or ndarray
        Wind speed matrix 10 meters above the sea surface (U_10) in m/s.

    Returns:
    --------
    k_m_s : float or ndarray
        Gas transfer velocity parameterized in units of meters per second (m/s),
        ready to be multiplied directly by your model's grid cell thickness/flux steps.
    """
    t = temp - 273.15

    # Polynomial coefficients from Wanninkhof (2014), Table 1.
    Sc = 2116.8 - 136.25 * t + 4.7353 * t**2 - 0.092307 * t**3 + 0.0007555 * t**4

    k_cm_hr = 0.251 * (wind_speed**2) * (Sc / 660.0)**(-0.5)

    # Unit conversion: cm/hr -> m/s
    return k_cm_hr * 0.01 / 3600


def compute_carbonate_constants_K(temp_kelvin, salinity):
    """
    Computes K1 and K2 using the Lueker, Dickson, & Keeling (2000) formulations.

    Parameters:
    -----------
    temp_kelvin : float or ndarray
        Sea surface temperature matrix in Kelvin (K).
    salinity : float or ndarray
        Salinity matrix in practical salinity units (psu).

    Returns:
    --------
    K1, K2 : float or ndarray
        Equilibrium constants in units of mol/L.
    """
    T = temp_kelvin
    S = salinity

    pK1 = (3633.86 / T) - 61.2172 + (9.67770 * jnp.log(T)) - (0.011555 * S) + (0.0001152 * S**2)
    K1 = 10**(-pK1)

    pK2 = (471.78 / T) + 25.9290 - (3.16967 * jnp.log(T)) - (0.01781 * S) + (0.0001122 * S**2)
    K2 = 10**(-pK2)

    return K1, K2

@checkify.checkify
def check_positive(x):
  checkify.check(x > 0, "must be positive!")
  return x

def solve_for_co2_aq(
    dissolved_inorganic_carbon,
    total_alkalinity,
    temperature,
    salinity,
):

    K1, K2 = compute_carbonate_constants_K(temperature, salinity) 
    DIC_over_TA = dissolved_inorganic_carbon / total_alkalinity
    
    # Solving second order polynomial problem
    b = K1 * (1.0 - DIC_over_TA)
    c = K1 * K2 * (1.0 - 2 * DIC_over_TA)
    errors, determinant = check_positive(b**2 - 4*c)
    checkify.check_error(errors)
    
    H = 0.5 * ( - b + jnp.sqrt(determinant) ) # solve for [H+]
    #co2_aq = dissolved_inorganic_carbon / ( 1.0 + K1 / H +  K1*K2 / H**2 )
    co2_aq = total_alkalinity / ( K1 / H +  2 * K1*K2 / H**2 )
   
    return co2_aq
 

def compute_co2_flux(
    co2_volume_mixing_ratio,            # ppm
    surface_dry_air_pressure,    # atm
    wind_10m,                    # m/s
    dissolved_inorganic_carbon,  # M
    total_alkalinity,            # M
    seawater_temperature,        # K
    salinity,           # psu
):
    """
    Computes the air-sea CO2 flux using the bulk formula.

    The flux is computed as:
        F = k * K0 * (pCO2_sea - pCO2_air)

    where pCO2_sea is derived from DIC and TA via carbonate equilibrium,
    and k is the Wanninkhof (2014) gas transfer velocity.

    Parameters:
    -----------
    co2_volume_mixing_ratio : float or ndarray
        Atmospheric CO2 volume mixing ratio in ppm.
    surface_dry_air_pressure : float or ndarray
        Surface dry air pressure in atm.
    wind_10m : float or ndarray
        Wind speed at 10 m above the surface in m/s.
    dissolved_inorganic_carbon : float or ndarray
        Mixed layer DIC concentration in mol/L (M).
    total_alkalinity : float or ndarray
        Mixed layer total alkalinity in mol/L (M).
    seawater_temperature : float or ndarray
        Sea surface temperature in Kelvin.
    salinity : float or ndarray
        Sea surface salinity in practical salinity units (psu).

    Returns:
    --------
    co2_flux_in_volume_mixing_ratio : float or ndarray
        CO2 flux expressed as k * (pCO2_sea - pCO2_air) / P * 1e6 in m/s * ppm.
        Positive upward (ocean to atmosphere).
    co2_flux_in_concentration : float or ndarray
        CO2 flux expressed as k * K0 * (pCO2_sea - pCO2_air) in m/s * mol/L.
        Positive upward (ocean to atmosphere).
    """
    

    pco2_air = (co2_volume_mixing_ratio * 1e-6) * surface_dry_air_pressure

    gas_transfer_velocity = compute_gas_transfer_velocity(seawater_temperature, wind_10m)
    co2_aq = solve_for_co2_aq(
        dissolved_inorganic_carbon = dissolved_inorganic_carbon,
        total_alkalinity = total_alkalinity,
        temperature = seawater_temperature,
        salinity = salinity,
    )
    K0 = compute_solubility_K0(seawater_temperature, salinity)
    pco2_seawater = co2_aq / K0 # in atm

    co2_flux_in_volume_mixing_ratio = gas_transfer_velocity * (pco2_seawater - pco2_air) / surface_dry_air_pressure * 1e6 
    co2_flux_in_concentration = gas_transfer_velocity * K0 * (pco2_seawater - pco2_air)
    
    return co2_flux_in_volume_mixing_ratio, co2_flux_in_concentration

