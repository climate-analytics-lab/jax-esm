"""Helper functions for computing biogeochemical coefficients."""

import jax.numpy as jnp


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

    return jnp.exp(ln_K0) / 101300.0


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
