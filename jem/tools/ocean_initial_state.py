"""Temperature and salinity for the ``veros-earth`` ocean's initial state, from the World Ocean Atlas.

``+configuration=veros-earth`` runs its ocean on the packaged
``RotatedGaussianLatLon`` grid. :func:`ocean_initial_state` turns NOAA NCEI's
World Ocean Atlas 2018 annual climatology (1-degree, 102 standard depths) into
the ocean's initial temperature and salinity on that grid's cells and layers,
and writes the small netCDF that
:func:`jem.components.veros.setups.earth.earth_setup` starts the ocean from
(``initial_state_file``).

Source and terms
----------------
The packaged ``jem/data/ocean_initial_state_RotatedGaussianLatLon.nc`` is
derived from the annual (all-years, 1955-2017) objectively analysed
temperature ``t_an`` and salinity ``s_an`` of World Ocean Atlas 2018
(``woa18_decav_t00_01.nc`` and ``woa18_decav_s00_01.nc``), NOAA National
Centers for Environmental Information, https://www.ncei.noaa.gov/products/world-ocean-atlas.
Its terms, as the data state them:

* the files' ``license`` attribute: "These data are openly available to the
  public. Please acknowledge the use of these data with the text given in the
  acknowledgment attribute." (the files carry no ``acknowledgment`` attribute;
  the requested text is the citation below);
* the dataset's catalogue record (NCEI accession ``NCEI-WOA18``) has
  ``accessLevel: public`` and asks "Cite as: Boyer, Tim P.; Garcia, Hernan E.;
  Locarnini, Ricardo A.; Zweng, Melissa M.; Mishonov, Alexey V.; Reagan, James
  R.; Weathers, Katharine A.; Baranova, Olga K.; Paver, Christopher R.;
  Seidov, Dan; Smolyar, Igor V. (2018). World Ocean Atlas 2018. [indicate
  subset used]. NOAA National Centers for Environmental Information. Dataset.
  https://www.ncei.noaa.gov/archive/accession/NCEI-WOA18", and "When using
  temperature, please also cite: Locarnini, R. A., ... 2019. World Ocean Atlas
  2018, Volume 1: Temperature. NOAA Atlas NESDIS 81, https://doi.org/10.25923/e5rn-9711"
  and likewise for salinity (Zweng et al., 2019, NOAA Atlas NESDIS 82,
  https://doi.org/10.25923/9pgv-1224);
* it also states "Use liability: NOAA and NCEI cannot provide any warranty as
  to the accuracy, reliability, or completeness of furnished data."

The data are a US federal government product offered to the public with an
acknowledgement request and no restriction on redistribution, and the only
conditions stated are the attribution and the disclaimer, which the output
file carries (``source_citation``, ``source_terms``). Unlike ETOPO 2022 (which
states CC0 outright), WOA states no named licence, so this is the reading of
those sentences, not a licence identifier. The coarsened derivative is
therefore redistributed under this package's MIT licence with that attribution.

What the file holds, and why
----------------------------
* **Annual mean, not a month.** Monthly WOA fields stop at 1500 m and a run
  can start in any month; the annual mean covers the whole column (5500 m),
  matches no particular month's upper-ocean seasonal cycle and so is the
  start a spin-up of unknown season should have. It is stated in the file.
* **Conservative Temperature and Absolute Salinity.** Veros' ``temp`` is
  Conservative Temperature and ``salt`` Absolute Salinity in g/kg (the
  variables' own descriptions, and the TEOS-10 equation-of-state kernels
  take them), whatever ``eq_of_state_type`` is selected. WOA gives *in-situ*
  temperature and *practical* salinity. In-situ temperature differs from
  potential temperature by 0.07 K at 1000 m and 0.45 K at 5000 m in a cold
  column, so it cannot be used as is: it is converted at every WOA point, at
  the point's own pressure, before any averaging (the conversion is
  non-linear and Conservative Temperature and Absolute Salinity are the
  quantities that mix linearly). See :func:`conservative_temperature_from_in_situ`
  and :func:`absolute_salinity_from_practical`.
* **Layer means, not point values.** A model layer is a volume average, and
  the layers are up to 690 m thick, so each layer takes the mean of the
  profile (linear between WOA depths) over the layer's full thickness, not its
  value at the centre; at a thermocline the two differ by degrees.
* **The same regridding as the bathymetry** (``jem.tools._rotated_grid``):
  each WOA point belongs to the target cell whose centre is nearest on the
  sphere, and a cell's value at a WOA depth is the ``cos(latitude)``-weighted
  mean over its source points that have ocean at that depth.
* **Which cells have values.** The wet cells are the model's own: the
  land-sea mask's wet columns, each with the layers whose centre lies above
  the packaged bathymetry's depth (:func:`wet_layer_mask`). A wet cell that
  has no source value (WOA's own sea floor is shallower there, or the cell
  holds only sea WOA does not resolve) is filled, first horizontally from the
  mean of its already-filled wet neighbours at the same layer
  (``jem.tools._rotated_grid.fill_from_wet_neighbours``), then, if no
  wet neighbour has one, from the layer above (or, at the surface, from the
  nearest filled wet cell on the sphere); the counts are recorded in the
  file (``cells_*``). Because the wet pattern depends on the bathymetry, the
  file is valid only with the bathymetry file it was made with, and the setup
  refuses any other.

Regenerating the packaged file
------------------------------
The two WOA files (185 MB and 114 MB) are neither shipped nor needed by the
tests, and no reduction of them to under 1 MB exists that is not the output
itself, so the tests check the packaged file's self-consistency, its geography
and its attributes, and the generator on synthetic input. To rebuild it::

    curl -O https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav/1.00/woa18_decav_t00_01.nc
    curl -O https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/salinity/decav/1.00/woa18_decav_s00_01.nc

    from importlib import resources
    from jem.tools.ocean_initial_state import ocean_initial_state

    data = resources.files("jem.data")
    ocean_initial_state(
        "woa18_decav_t00_01.nc",
        "woa18_decav_s00_01.nc",
        data / "RotatedGaussianLatLon.SCRIP.nc",
        data / "landsea_mask_fraction_RotatedGaussianLatLon.nc",
        data / "bathymetry_RotatedGaussianLatLon.nc",
        "jem/data/ocean_initial_state_RotatedGaussianLatLon.nc",
    )

Layout
------
``temperature`` (degrees C, Conservative Temperature) and ``salinity``
(g/kg, Absolute Salinity), float32, dims ``("depth", "latitude", "longitude")``:
``depth`` is the layer centres in metres, surface first; ``(latitude,
longitude)`` is the same row-major ``(ny, nx)`` cell order as the SCRIP and
land-fraction files, with the cells' true (post-rotation) ``lat``/``lon`` as
2-D coordinates. ``NaN`` marks a cell that is not wet in the model (dry
column, or a layer below the column's floor), and it is exactly those cells.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from jem.components.veros.setups._layers import LAYER_THICKNESSES
from jem.tools._rotated_grid import (
    fill_from_wet_neighbours,
    nearest_target,
    read_scrip_cell_centres,
    read_wet_mask,
    unit_vectors,
    weighted_target_means,
)
from jem.tools.ocean_bathymetry import MINIMUM_DEPTH, read_depth

logger = logging.getLogger(__name__)

#: The fewest wet layers a wet column has, which are the layers of
#: ``MINIMUM_DEPTH``
#: (:func:`jem.components.veros.setups.earth.kbot_from_depth`'s own minimum).
MINIMUM_WET_LAYERS = 2

# Conversion constants (TEOS-10; IOC, SCOR and IAPSO, 2010, "The international
# thermodynamic equation of seawater - 2010: calculation and use of
# thermodynamic properties", UNESCO).
#: Reference-Composition Salinity per unit practical salinity, so that
#: ``SR = SP * 35.16504 / 35``.
REFERENCE_SALINITY_FACTOR = 35.16504 / 35.0
#: The specific heat that relates Conservative Temperature to potential enthalpy, J kg-1 K-1.
CP0 = 3991.86795711963

#: Where World Ocean Atlas 2018 comes from, what it is called, and the terms it
#: states; written into every file derived from it (as ``source_*`` attributes).
WOA18_ATTRS = {
    "source_dataset": (
        "World Ocean Atlas 2018, annual (1955-2017) objectively analysed mean temperature (t_an)"
        " and salinity (s_an), 1 degree, 102 standard depths"
    ),
    "source_url": (
        "https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav/1.00/"
        "woa18_decav_t00_01.nc and .../salinity/decav/1.00/woa18_decav_s00_01.nc"
    ),
    "source_citation": (
        "Boyer, Tim P.; Garcia, Hernan E.; Locarnini, Ricardo A.; Zweng, Melissa M.; Mishonov,"
        " Alexey V.; Reagan, James R.; Weathers, Katharine A.; Baranova, Olga K.; Paver,"
        " Christopher R.; Seidov, Dan; Smolyar, Igor V. (2018). World Ocean Atlas 2018. Annual"
        " temperature and salinity, 1 degree. NOAA National Centers for Environmental"
        " Information. Dataset. https://www.ncei.noaa.gov/archive/accession/NCEI-WOA18."
        " Temperature: Locarnini, R. A., et al., 2019. World Ocean Atlas 2018, Volume 1:"
        " Temperature. A. Mishonov Technical Ed.; NOAA Atlas NESDIS 81, 52 pp."
        " https://doi.org/10.25923/e5rn-9711. Salinity: Zweng, M. M., et al., 2019. World Ocean"
        " Atlas 2018, Volume 2: Salinity. A. Mishonov Technical Ed.; NOAA Atlas NESDIS 82, 50 pp."
        " https://doi.org/10.25923/9pgv-1224"
    ),
    "source_terms": (
        "These data are openly available to the public. Please acknowledge the use of these data"
        " with the text given in the acknowledgment attribute (WOA18 file attribute `license`;"
        " the acknowledgement requested by the NCEI catalogue record is the citation in"
        " source_citation). NOAA and NCEI cannot provide any warranty as to the accuracy,"
        " reliability, or completeness of furnished data. Users assume responsibility to determine"
        " the usability of these data (NCEI accession NCEI-WOA18 'Use liability')."
    ),
}

# The linear equation of state of the setup (`eq_of_state_type = 1`), as
# `veros/core/density/linear_eq.py` defines it: restated because importing
# Veros switches `jax_enable_x64` on for the process, and a test pins these to
# Veros' own values.
LINEAR_EOS_RHO0 = 1024.0
LINEAR_EOS_THETA0 = 283.0 - 273.15
LINEAR_EOS_S0 = 35.0
LINEAR_EOS_BETA_T = 1.67e-4
LINEAR_EOS_BETA_S = 0.78e-3


# ---------------------------------------------------------------------------
# Conversion from what WOA holds to what Veros holds.
# ---------------------------------------------------------------------------


def depth_from_pressure(pressure: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Depth in metres of a pressure in dbar, latitude in degrees (Saunders, 1981).

    Fofonoff, N. P. and R. C. Millard Jr., 1983: Algorithms for computation
    of fundamental properties of seawater. UNESCO Technical Papers in Marine
    Science 44, eq. 25 (the standard-ocean depth with the latitude and
    pressure dependence of gravity). Check value in that report: 9712.653 m
    at 10000 dbar, 30 degrees.
    """
    x = np.sin(np.deg2rad(latitude)) ** 2
    gravity = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * pressure
    return np.asarray(
        (((-1.82e-15 * pressure + 2.279e-10) * pressure - 2.2512e-5) * pressure + 9.72659) * pressure / gravity
    )


def pressure_from_depth(depth: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Pressure in dbar of a depth in metres: the inverse of :func:`depth_from_pressure`.

    Found by fixed-point iteration on ``p += depth - depth_from_pressure(p)``,
    which converges because ``d(depth)/dp`` is within 3 % of one.
    """
    depth, latitude = np.broadcast_arrays(np.asarray(depth, dtype=float), np.asarray(latitude, dtype=float))
    pressure = depth.copy()
    for _ in range(12):
        pressure = pressure + (depth - depth_from_pressure(pressure, latitude))
    return pressure


def adiabatic_lapse_rate(practical_salinity: np.ndarray, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Adiabatic temperature gradient, K per dbar (Bryden, 1973).

    Fofonoff and Millard (1983), UNESCO Technical Papers in Marine Science
    44, ``ATG``; ``temperature`` in degrees C, ``pressure`` in dbar. Check
    value: 3.255976e-4 K/dbar at salinity 40, temperature 40, 10000 dbar.
    """
    ds = practical_salinity - 35.0
    t, p = temperature, pressure
    gradient = (
        ((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p
        + (
            (2.7759e-12 * t - 1.1351e-10) * ds
            + ((-5.4481e-14 * t + 8.7330e-12) * t - 6.7795e-10) * t
            + 1.8741e-8
        )
    ) * p + (-4.2393e-8 * t + 1.8932e-6) * ds + ((6.6228e-10 * t - 6.8360e-8) * t + 8.5258e-6) * t + 3.5803e-5
    return np.asarray(gradient)


def potential_temperature(
    practical_salinity: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    reference_pressure: float = 0.0,
) -> np.ndarray:
    """Potential temperature (degrees C) of in-situ ``temperature`` moved adiabatically to ``reference_pressure``.

    The fourth-order Runge-Kutta integral of :func:`adiabatic_lapse_rate`
    of Fofonoff and Millard (1983), UNESCO Technical Papers in Marine Science
    44 (``PTMP``). Check value: 36.89073 degrees C for salinity 40,
    temperature 40, 10000 dbar to 0 dbar. It uses practical salinity and the
    EOS-80 formulation; it agrees with the TEOS-10 potential temperature to a
    few millikelvin over the ocean's range, far below WOA's own uncertainty.
    """
    h = reference_pressure - pressure
    xk = h * adiabatic_lapse_rate(practical_salinity, temperature, pressure)
    t = temperature + 0.5 * xk
    q = xk
    p = pressure + 0.5 * h
    xk = h * adiabatic_lapse_rate(practical_salinity, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * adiabatic_lapse_rate(practical_salinity, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * adiabatic_lapse_rate(practical_salinity, t, p)
    return np.asarray(t + (xk - 2.0 * q) / 6.0)


def absolute_salinity_from_practical(practical_salinity: np.ndarray) -> np.ndarray:
    """Reference-Composition Salinity in g/kg: ``SP * 35.16504 / 35`` (TEOS-10).

    This is Absolute Salinity with the salinity anomaly ``delta SA`` taken as
    zero: it is up to about 0.03 g/kg (0.1 % of the mean salinity contrast)
    in the Pacific and Indian oceans, and its atlas needs a dependency and a
    data file for an effect an order of magnitude below WOA's own analysis
    error. The 0.47 % scale factor is applied everywhere, which is what
    makes the density Veros computes from ``salt`` consistent with the
    TEOS-10 equations the setup's variables are documented in.
    """
    return np.asarray(practical_salinity) * REFERENCE_SALINITY_FACTOR


def conservative_temperature_from_potential(absolute_salinity: np.ndarray, potential_temperature: np.ndarray) -> np.ndarray:
    """Conservative Temperature (degrees C) from potential temperature and Absolute Salinity.

    McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003:
    Accurate and computationally efficient algorithms for potential
    temperature and density of seawater. J. Atmos. Oceanic Technol., 20,
    730-741, in the polynomial form of the TEOS-10 Gibbs SeaWater Oceanographic
    Toolbox (``gsw_ct_from_pt``): the potential enthalpy at the surface
    divided by :data:`CP0`. Valid for ``0 <= SA <= 42``, ``-2 <= pt <= 40``.
    """
    x2 = 0.0248826675584615 * absolute_salinity
    x = np.sqrt(x2)
    y = potential_temperature * 0.025
    potential_enthalpy = (
        61.01362420681071
        + y * (168776.46138048015 + y * (-2735.2785605119625 + y * (2574.2164453821433
        + y * (-1536.6644434977543 + y * (545.7340497931629 + (-50.91091728474331 - 18.30489878927802 * y) * y)))))
        + x2 * (268.5520265845071 + y * (-12019.028203559312 + y * (3734.858026725145
        + y * (-2046.7671145057618 + y * (465.28655623826234 + (-0.6370820302376359 - 10.650848542359153 * y) * y))))
        + x * (937.2099110620707 + y * (588.1802812170108 + y * (248.39476522971285
        + (-3.871557904936333 - 2.6268019854268356 * y) * y))
        + x * (-1687.914374187449 + x * (246.9598888781377 + x * (123.59576582457964 - 48.5891069025409 * x))
        + y * (936.3206544460336 + y * (-942.7827304544439 + y * (369.4389437509002
        + (-33.83664947895248 - 9.987880382780322 * y) * y))))))
    )
    return potential_enthalpy / CP0


def conservative_temperature_from_in_situ(
    practical_salinity: np.ndarray, temperature: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    """Conservative Temperature from in-situ temperature, practical salinity and pressure.

    In-situ temperature to potential temperature
    (:func:`potential_temperature`), then to Conservative Temperature
    (:func:`conservative_temperature_from_potential`) with the Absolute
    Salinity of :func:`absolute_salinity_from_practical`.
    """
    theta = potential_temperature(practical_salinity, temperature, pressure)
    return conservative_temperature_from_potential(absolute_salinity_from_practical(practical_salinity), theta)


# ---------------------------------------------------------------------------
# The model's wet cells, and a static-stability check on the result.
# ---------------------------------------------------------------------------


def layer_centres(layer_thicknesses: Sequence[float] | np.ndarray) -> np.ndarray:
    """Depth of each layer's centre, metres, surface layer first."""
    thickness = np.asarray(layer_thicknesses, dtype=float)
    return np.cumsum(thickness) - thickness / 2


def wet_layer_mask(
    depth: np.ndarray, wet: np.ndarray, layer_thicknesses: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Return the model's 3-D wet mask, ``depth.shape + (nz,)``, surface layer first.

    A layer of a wet column is wet if its centre lies above the sea floor,
    and a wet column has at least :data:`MINIMUM_WET_LAYERS` of them: the rule
    :func:`jem.components.veros.setups.earth.kbot_from_depth` uses to set
    ``kbot`` (and so Veros' ``maskT``). The setup compares an initial-state
    file with the mask it actually built, so a drift between the two cannot go
    unnoticed.
    """
    thickness = np.asarray(layer_thicknesses, dtype=float)
    centres = layer_centres(thickness)
    n_wet = np.maximum(
        (centres[None, None, :] < np.nan_to_num(depth)[..., None]).sum(-1), min(MINIMUM_WET_LAYERS, thickness.size)
    )
    return np.asarray(wet[..., None] & (np.arange(thickness.size) < n_wet[..., None]))


def linear_density_anomaly(conservative_temperature: np.ndarray, absolute_salinity: np.ndarray) -> np.ndarray:
    """Density anomaly of the setup's linear equation of state, kg m-3."""
    return (
        -(
            LINEAR_EOS_BETA_T * (conservative_temperature - LINEAR_EOS_THETA0)
            - LINEAR_EOS_BETA_S * (absolute_salinity - LINEAR_EOS_S0)
        )
        * LINEAR_EOS_RHO0
    )


def static_stability(
    temperature: np.ndarray, salinity: np.ndarray
) -> dict[str, float | int | np.ndarray]:
    """Count the vertically adjacent wet pairs whose upper layer is denser than the lower one.

    Parameters
    ----------
    temperature, salinity : numpy.ndarray
        Conservative Temperature and Absolute Salinity, ``(nz, ny, nx)`` with
        the surface layer first and NaN where the cell is not wet (a pair
        counts only when both its cells are wet).

    Returns
    -------
    dict
        ``pairs``: the number of wet adjacent pairs; ``inverted``: how many
        are inverted in the setup's linear equation of state
        (:func:`linear_density_anomaly`); ``largest``: the largest inversion
        (the upper layer's excess density, kg m-3, 0 if none);
        ``inversion`` : the ``(nz - 1, ny, nx)`` excess densities, positive
        where inverted and NaN where the pair is not wet.

    """
    rho = linear_density_anomaly(temperature, salinity)
    excess = rho[:-1] - rho[1:]
    pairs = int(np.isfinite(excess).sum())
    inverted = np.where(np.isfinite(excess) & (excess > 0), excess, 0.0)
    return {
        "pairs": pairs,
        "inverted": int((inverted > 0).sum()),
        "largest": float(inverted.max()) if pairs else 0.0,
        "inversion": np.where(np.isfinite(excess), np.maximum(excess, 0.0), np.nan),
    }


# ---------------------------------------------------------------------------
# The generator.
# ---------------------------------------------------------------------------


def _read_woa_field(woa_file: str | Path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a WOA climatology as ``(field, depth, lat, lon)``; ``field`` is ``(nz, nlat, nlon)`` with NaN missing."""
    with xr.open_dataset(woa_file, decode_times=False) as ds:
        da = ds[variable]
        if da.dims != ("time", "depth", "lat", "lon") or da.sizes["time"] != 1:
            raise ValueError(f"{woa_file}: {variable} has dims {da.dims}, expected ('time', 'depth', 'lat', 'lon')")
        field = da.isel(time=0).to_numpy().astype(np.float64)
        depth = ds["depth"].to_numpy().astype(np.float64)
        lat = ds["lat"].to_numpy().astype(np.float64)
        lon = ds["lon"].to_numpy().astype(np.float64)
    return field, depth, lat, lon


def _layer_means(
    profile: np.ndarray, source_depth: np.ndarray, layer_thicknesses: np.ndarray, n_sub: int = 100
) -> np.ndarray:
    """Mean of one column's profile over each layer; NaN where the profile does not cover the layer's centre.

    ``profile`` holds the values at ``source_depth`` (NaN below the column's
    last value, and wherever the source has no ocean); the profile is linear
    between its valid depths and absent outside them. A layer's mean is taken
    over the part of it the profile covers (midpoint rule with ``n_sub``
    sub-layers), so a layer that is half covered is represented by its covered
    half; a layer whose centre is not covered is NaN and is filled by the
    caller.
    """
    valid = np.isfinite(profile)
    out = np.full(layer_thicknesses.size, np.nan)
    if not valid.any():
        return out
    depths, values = source_depth[valid], profile[valid]
    bottoms = np.cumsum(layer_thicknesses)
    tops = bottoms - layer_thicknesses
    for j in range(layer_thicknesses.size):
        centre = 0.5 * (tops[j] + bottoms[j])
        if centre < depths[0]:
            continue
        if centre > depths[-1]:
            break
        sample = tops[j] + (np.arange(n_sub) + 0.5) * (layer_thicknesses[j] / n_sub)
        covered = (sample >= depths[0]) & (sample <= depths[-1])
        out[j] = np.interp(sample[covered], depths, values).mean()
    return out


def ocean_initial_state(
    temperature_file: str | Path,
    salinity_file: str | Path,
    scrip_grid_file: str | Path,
    landsea_mask_file: str | Path,
    bathymetry_file: str | Path,
    output_file: str | Path,
    *,
    landsea_mask_threshold: float = 0.5,
    layer_thicknesses: Sequence[float] | np.ndarray = LAYER_THICKNESSES,
    temperature_variable: str = "t_an",
    salinity_variable: str = "s_an",
    source_attrs: dict[str, str] | None = None,
) -> Path:
    """Write the ocean's initial Conservative Temperature and Absolute Salinity on the model's cells.

    Parameters
    ----------
    temperature_file, salinity_file : str or pathlib.Path
        World Ocean Atlas 2018 annual 1-degree climatologies
        (``woa18_decav_t00_01.nc``, ``woa18_decav_s00_01.nc``): in-situ
        temperature in degrees C and practical salinity, ``(time, depth, lat,
        lon)``, on the same grid.
    scrip_grid_file : str or pathlib.Path
        The ocean grid's SCRIP file.
    landsea_mask_file : str or pathlib.Path
        The land-fraction file that decides which columns are wet.
    bathymetry_file : str or pathlib.Path
        The depth file (:func:`jem.tools.ocean_bathymetry.ocean_bathymetry`)
        that decides how many layers each wet column has. The result is valid
        only with this file.
    output_file : str or pathlib.Path
        Where to write; the parent directory is created.
    landsea_mask_threshold : float
        Land fraction at or above which a cell is land (the setup's default).
    layer_thicknesses : sequence of float
        The model's layer thicknesses in metres, surface first.
    temperature_variable, salinity_variable : str
        The variables to read from the two files.
    source_attrs : dict, optional
        Provenance attributes to write; default :data:`WOA18_ATTRS`. Pass
        your own when the inputs are not WOA18.

    Returns
    -------
    pathlib.Path
        ``output_file``.

    Notes
    -----
    The module docstring describes the method and its choices. The counts of
    cells each fill rule set, and the static-stability summary, are logged and
    written as attributes.

    """
    thickness = np.asarray(layer_thicknesses, dtype=float)
    nz = thickness.size
    nx, ny, target_lat, target_lon = read_scrip_cell_centres(scrip_grid_file)
    wet2d = read_wet_mask(landsea_mask_file, landsea_mask_threshold, ny, nx)
    depth_file = read_depth(bathymetry_file, ny, nx)  # (nx, ny)
    wet3d = np.moveaxis(wet_layer_mask(depth_file.T, wet2d, thickness), -1, 0)  # (nz, ny, nx)

    t_in_situ, woa_depth, woa_lat, woa_lon = _read_woa_field(temperature_file, temperature_variable)
    practical, depth_s, lat_s, lon_s = _read_woa_field(salinity_file, salinity_variable)
    if not (np.array_equal(woa_depth, depth_s) and np.array_equal(woa_lat, lat_s) and np.array_equal(woa_lon, lon_s)):
        raise ValueError("the temperature and salinity files are not on the same grid")

    # Conservative Temperature and Absolute Salinity at every source point, level by level.
    lat2, lon2 = np.meshgrid(woa_lat, woa_lon, indexing="ij")
    valid = np.isfinite(t_in_situ) & np.isfinite(practical)
    conservative = np.full(t_in_situ.shape, np.nan)
    absolute = np.full(t_in_situ.shape, np.nan)
    for k, z in enumerate(woa_depth):
        ok = valid[k]
        pressure = pressure_from_depth(np.full(int(ok.sum()), z), lat2[ok])
        conservative[k][ok] = conservative_temperature_from_in_situ(practical[k][ok], t_in_situ[k][ok], pressure)
        absolute[k][ok] = absolute_salinity_from_practical(practical[k][ok])

    # Horizontal: each source point to the nearest target centre, a cell's mean over its ocean points.
    n_cells = nx * ny
    nearest_all = nearest_target(unit_vectors(target_lat, target_lon).reshape(-1, 3), unit_vectors(lat2, lon2).reshape(-1, 3))
    area_weight = np.cos(np.deg2rad(lat2)).ravel()
    on_target = np.full((2, woa_depth.size, ny, nx), np.nan)
    for k in range(woa_depth.size):
        ok = valid[k].ravel()
        for v, field in enumerate((conservative, absolute)):
            on_target[v, k] = weighted_target_means(
                nearest_all[ok], area_weight[ok], field[k].ravel()[ok], n_cells
            ).reshape(ny, nx)

    # Vertical: the mean of each column's profile over each model layer.
    layered = np.full((2, nz, ny, nx), np.nan)
    for j in range(ny):
        for i in range(nx):
            if not wet2d[j, i]:
                continue
            for v in range(2):
                layered[v, :, j, i] = _layer_means(on_target[v, :, j, i], woa_depth, thickness)
    layered[:, ~wet3d] = np.nan
    from_source = int(np.isfinite(layered[0]).sum())
    if not np.array_equal(np.isfinite(layered[0]), np.isfinite(layered[1])):
        raise AssertionError("temperature and salinity have values in different cells")

    # Fill every wet cell without a value: horizontally at its layer, then from the layer above.
    n_horizontal = n_downward = n_surface = 0
    for j in range(nz):
        missing = wet3d[j] & np.isnan(layered[0, j])
        if not missing.any():
            continue
        for v in range(2):
            fill_from_wet_neighbours(layered[v, j], wet3d[j])
        still = wet3d[j] & np.isnan(layered[0, j])
        n_horizontal += int(missing.sum() - still.sum())
        if j > 0:
            for v in range(2):
                layered[v, j][still] = layered[v, j - 1][still]
            n_downward += int(still.sum())
        elif still.any():
            # A wet region the source does not reach at the surface and that no
            # filled wet cell touches (the model keeps Lake Huron wet, and WOA
            # has no lake): the filled wet surface cell nearest on the sphere
            # stands in, and is counted.
            donors = wet3d[0] & np.isfinite(layered[0, 0])
            if not donors.any():
                raise ValueError("the source has no value at the surface anywhere on the model's wet cells")
            donor_vectors = unit_vectors(target_lat[donors], target_lon[donors])
            for j_cell, i_cell in zip(*np.nonzero(still)):
                nearest_donor = np.argmax(donor_vectors @ unit_vectors(target_lat[j_cell, i_cell], target_lon[j_cell, i_cell]))
                for v in range(2):
                    layered[v, 0, j_cell, i_cell] = layered[v, 0][donors][nearest_donor]
            n_surface = int(still.sum())
    if (wet3d & ~np.isfinite(layered[0])).any() or (~wet3d & np.isfinite(layered[0])).any():
        raise AssertionError("the filled field's wet cells are not the model's")
    stability = static_stability(layered[0], layered[1])
    logger.info(
        "initial state: %d wet cells, %d from the source, %d filled horizontally, %d from the layer above, %d at the surface from the nearest cell;"
        " %d of %d adjacent pairs inverted (largest %.4f kg m-3)",
        int(wet3d.sum()), from_source, n_horizontal, n_downward, n_surface,
        stability["inverted"], stability["pairs"], stability["largest"],
    )

    attrs = dict(WOA18_ATTRS if source_attrs is None else source_attrs)
    output = xr.Dataset(
        data_vars=dict(
            temperature=(
                ("depth", "latitude", "longitude"),
                layered[0].astype(np.float32),
                {
                    "long_name": "Conservative Temperature, the layer mean of the annual climatology",
                    "standard_name": "sea_water_conservative_temperature",
                    "units": "degC",
                    "coordinates": "lat lon",
                },
            ),
            salinity=(
                ("depth", "latitude", "longitude"),
                layered[1].astype(np.float32),
                {
                    "long_name": "Absolute Salinity (Reference-Composition Salinity), the layer mean of the annual climatology",
                    "standard_name": "sea_water_absolute_salinity",
                    "units": "g kg-1",
                    "coordinates": "lat lon",
                },
            ),
        ),
        coords=dict(
            depth=("depth", layer_centres(thickness), {"units": "m", "positive": "down", "long_name": "layer centre"}),
            lat=(("latitude", "longitude"), target_lat, {"units": "degrees_north"}),
            lon=(("latitude", "longitude"), target_lon, {"units": "degrees_east"}),
        ),
        attrs={
            "title": "Initial ocean temperature and salinity on the rotated Gaussian lat-lon grid",
            **attrs,
            "method": (
                "annual-mean climatology; in-situ temperature and practical salinity converted at each"
                " source point to Conservative Temperature (UNESCO 1983 adiabatic lapse rate and"
                " potential temperature, TEOS-10 conservative-temperature polynomial) and Reference-"
                "Composition Salinity (SP * 35.16504 / 35); each source point assigned to the target cell"
                " with the nearest centre (3-D unit vectors) and averaged with weight cos(latitude)"
                " over the points that have ocean at that depth; each model layer takes the mean of the"
                " cell's profile (linear between source depths) over the layer thickness; wet cells with"
                " no value are filled from the mean of their filled wet neighbours at the same layer, then"
                " from the layer above (at the surface, the nearest filled wet cell)"
            ),
            "wet_cells": int(wet3d.sum()),
            "cells_from_source": from_source,
            "cells_filled_horizontally": n_horizontal,
            "cells_filled_from_layer_above": n_downward,
            "cells_filled_at_surface_from_nearest_cell": n_surface,
            "adjacent_wet_pairs": stability["pairs"],
            "adjacent_pairs_statically_inverted": stability["inverted"],
            "largest_inversion_kg_m3": stability["largest"],
            "static_stability_equation_of_state": (
                "the setup's linear equation of state, rho0=1024 beta_T=1.67e-4 beta_S=0.78e-3 theta0=9.85 S0=35"
            ),
            "layer_thicknesses_m": ", ".join(f"{t:g}" for t in thickness),
            "layout": "(depth, latitude, longitude), depth = layer centres surface first, (latitude, longitude) = SCRIP (ny, nx) order",
            "wet_mask": (
                f"{Path(landsea_mask_file).name} lsm < {landsea_mask_threshold:g}, with the layers of"
                f" {Path(bathymetry_file).name} whose centre lies above the depth (at least {MINIMUM_WET_LAYERS}):"
                " NaN exactly where the model is not wet"
            ),
            "bathymetry_file": Path(bathymetry_file).name,
            "minimum_depth_m": float(MINIMUM_DEPTH),
            "generator": "jem.tools.ocean_initial_state.ocean_initial_state",
        },
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    compression = {"zlib": True, "complevel": 9, "shuffle": True}
    output.to_netcdf(
        output_file,
        encoding={name: dict(compression) for name in ("temperature", "salinity", "lat", "lon")},
    )
    return output_file


def read_initial_state(
    initial_state_file: str | Path, nz: int, ny: int, nx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a file written by :func:`ocean_initial_state`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(temperature, salinity, depth, lat_lon_check)``: temperature and
        salinity as ``(nx, ny, nz)`` float64 with the surface layer first and
        NaN on cells that are not wet (the setup's grid-array layout, the
        transpose of the file's ``(nz, ny, nx)``), the layer-centre ``depth``
        ``(nz,)``, and the file's ``(lat, lon)`` stacked as ``(2, ny, nx)``
        for the caller to compare with its grid.

    """
    with xr.open_dataset(initial_state_file) as ds:
        temperature = ds["temperature"].to_numpy()
        salinity = ds["salinity"].to_numpy()
        depth = ds["depth"].to_numpy().astype(np.float64)
        centres = np.stack([ds["lat"].to_numpy(), ds["lon"].to_numpy()]).astype(np.float64)
    if temperature.shape != (nz, ny, nx) or salinity.shape != (nz, ny, nx):
        raise ValueError(
            f"{initial_state_file}: temperature has shape {temperature.shape} and salinity {salinity.shape},"
            f" the model is (layers, ny, nx) = {(nz, ny, nx)}; regenerate it with"
            " jem.tools.ocean_initial_state.ocean_initial_state"
        )
    return (
        np.transpose(temperature, (2, 1, 0)).astype(np.float64),
        np.transpose(salinity, (2, 1, 0)).astype(np.float64),
        depth,
        centres,
    )
