"""Build Veros' own grid spacings so that its cell centres land on a target axis.

Veros does not take cell centres as input. It takes the spacing ``dxt``/``dyt``
and one origin per axis, and reconstructs ``xt``/``yt`` from them with a
leapfrog-style recursion (``veros.core.numerics.u_centered_grid``) that places
the first T point at ``origin - spacing / 2`` and then alternates. That
recursion is only exact when ``spacing[j]`` is the *central* difference of the
target centres, ``(centre[j+1] - centre[j-1]) / 2``, which is not the cell
width a Gaussian axis' bounds imply (the pole-clamped outermost bounds break
the relation and put the two polar rows up to ~1.5 degrees off).

:func:`axis_spacing_and_origin` returns the spacing and the origin that make
Veros' reconstruction reproduce the target centres to round-off, so that a
setup can put its ocean cells exactly where another component's cells are --
the premise of an index-to-index exchange with the atmosphere, and of the
ocean's metric terms (``cost``, areas) being those of the right latitudes.
Both Veros setups in this package use it.

This module imports no Veros code: it replicates the recursion host-side in
NumPy, in degrees, so it can be used (and tested) before Veros is set up.
"""

import numpy as np
import xarray as xr


def spacing_from_centers(centers: np.ndarray) -> np.ndarray:
    """Per-cell spacing whose Veros reconstruction hits ``centers``.

    The central difference of the target centres in the interior, and the
    one-sided difference at the two ends.

    Parameters
    ----------
    centers : numpy.ndarray
        Target cell-centre coordinates, shape ``(n,)``, increasing.

    Returns
    -------
    numpy.ndarray
        Spacing, shape ``(n,)``, in the units of ``centers``.

    """
    spacing = np.empty_like(centers, dtype=float)
    spacing[1:-1] = (centers[2:] - centers[:-2]) / 2.0
    spacing[0] = centers[1] - centers[0]
    spacing[-1] = centers[-1] - centers[-2]
    return spacing


def calibrate_origin(spacing: np.ndarray, first_center: float, cyclic: bool) -> float:
    """Return the origin that puts Veros' first interior centre at ``first_center``.

    Mirrors the ghost-cell fill of ``veros.core.numerics.
    calc_grid_spacings_kernel`` (periodic if ``cyclic``, edge value otherwise)
    and the recursion of ``u_centered_grid`` exactly.

    Parameters
    ----------
    spacing : numpy.ndarray
        Interior spacing, shape ``(n,)``.
    first_center : float
        Where the first interior T point must be.
    cyclic : bool
        Whether the axis is periodic (``settings.enable_cyclic_x``).

    Returns
    -------
    float
        The value for ``settings.x_origin`` / ``settings.y_origin``.

    """
    n = spacing.size
    padded = np.zeros(n + 4)
    padded[2:-2] = spacing
    if cyclic:
        padded[-2:] = padded[2:4]
        padded[:2] = padded[-4:-2]
    else:
        padded[-2:] = padded[-3]
        padded[:2] = padded[2]
    yu = np.zeros(n + 4)
    yu[1:] = np.cumsum(padded[1:])
    yt = np.zeros(n + 4)
    yt[0] = yu[0] - padded[0] * 0.5
    yt[1:] = 2 * yu[:-1]
    alt = np.ones(n + 4)
    alt[::2] = -1
    yt = alt * np.cumsum(alt * yt)
    return float(first_center - yt[2] + yu[2])


def axis_spacing_and_origin(
    centers: np.ndarray, cyclic: bool
) -> tuple[np.ndarray, float]:
    """Spacing and origin that make Veros reconstruct ``centers`` exactly.

    Parameters
    ----------
    centers : numpy.ndarray
        Target cell centres, degrees, shape ``(n,)``, increasing.
    cyclic : bool
        Whether the axis is periodic.

    Returns
    -------
    tuple[numpy.ndarray, float]
        ``(spacing, origin)`` for ``vs.dxt``/``vs.dyt`` (interior) and
        ``settings.x_origin``/``settings.y_origin``.

    """
    spacing = spacing_from_centers(np.asarray(centers, dtype=float))
    return spacing, calibrate_origin(spacing, float(centers[0]), cyclic)


def read_scrip_axes(scrip_grid_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the longitude and latitude cell-centre axes of a separable SCRIP grid.

    SCRIP stores cell centres flat in longitude-fastest order, so the axes are
    the first row and the first column of the ``(ny, nx)`` reshape.

    Parameters
    ----------
    scrip_grid_file : str
        A SCRIP file of a regular lon/lat or Gaussian grid whose
        ``grid_center_lon``/``grid_center_lat`` are separable.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(longitude, latitude)`` in degrees, shapes ``(nx,)`` and ``(ny,)``.

    Raises
    ------
    ValueError
        If the centres are not separable (the grid is curvilinear), because a
        single axis would then not describe it.

    """
    grid = xr.open_dataset(scrip_grid_file)
    nx, ny = (int(n) for n in grid["grid_dims"].to_numpy())
    lon = grid["grid_center_lon"].to_numpy().reshape(ny, nx)
    lat = grid["grid_center_lat"].to_numpy().reshape(ny, nx)
    if not (np.allclose(lon, lon[0][None, :]) and np.allclose(lat, lat[:, :1])):
        raise ValueError(
            f"{scrip_grid_file!r}'s cell centres are not a separable lon/lat "
            "grid, so it has no single longitude and latitude axis."
        )
    return lon[0].copy(), lat[:, 0].copy()
