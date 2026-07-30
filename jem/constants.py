# Physical constants used by the model regardless of Physics module

# =============================================================================
# Fundamental constants
# =============================================================================
g0 = 9.81  # Gravitational acceleration on Earth (m/s²)
stephan_boltzmann_const = 5.67e-8  # Stefan-Boltzmann constant (W/m²/K⁴)
solar_const = 1367.0  # Solar constant at top of atmosphere (W/m²)

# =============================================================================
# Atmosphere properties
# =============================================================================
atmosphere_column_mass = 1e4  # Total mass of atmospheric column (kg/m²)
atmosphere_specific_heat_capacity_under_constant_pressure = (
    1004.0  # Cp for dry air (J/K/kg)
)

# Surface layer properties (for bulk formula calculations)
surface_air_density = 1.22  # Surface air density at sea level (kg/m³)
bulk_drag_coefficient = 1e-3  # Bulk aerodynamic drag coefficient (dimensionless)

# =============================================================================
# Ocean properties
# =============================================================================
ocean_density = 1025.0  # Seawater density (kg/m³)
ocean_specific_heat_capacity = 3985.0  # Specific heat capacity of seawater (J/kg/K)

# Default mixed layer depth range (m)
default_mld_min = 40.0  # Minimum mixed layer depth (m)
default_mld_max = 60.0  # Maximum mixed layer depth (m)

# =============================================================================
# Land properties
# =============================================================================
land_density = 3000.0  # Effective land surface density (kg/m³)
land_specific_heat_capacity = 830.0  # Land surface heat capacity (J/kg/K)

# Default land depth range (m)
default_land_depth_min = 40.0  # Minimum land depth (m)
default_land_depth_max = 60.0  # Maximum land depth (m)

# =============================================================================
# Default temperatures
# =============================================================================
freezing_point_K = 273.15  # Freezing point of water (K) = 0°C

# =============================================================================
# Sea ice properties
# =============================================================================
ice_density = 917.0  # Density of sea ice (kg/m³)
ice_thermal_conductivity = 2.03  # Thermal conductivity of sea ice (W/m/K)
ice_latent_heat_fusion = 3.34e5  # Latent heat of fusion of ice (J/kg)
seawater_freezing_point_K = 271.35  # Freezing point of seawater (K) = -1.8°C
ice_melting_point_K = 273.15  # Melting point of (fresh) ice/snow surface (K) = 0°C
