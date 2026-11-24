"""The horizontal coordinates every component writes into its dataset.

A coupled run writes one dataset per component, to be combined with
``xr.merge([...], join="exact")``. Two layouts exist, and this module is the one
place that defines them so the slab models and the Veros component cannot
drift apart:

``separable``
    A grid whose axes are meridians and parallels. Dimensions ``("lon",
    "lat")``, with 1-D coordinates of the same names -- the atmosphere's own,
    so the datasets share an axis and merge on it.

``curvilinear``
    A grid whose axes are not (a displaced pole, a rotated pole). Index
    dimensions ``("x", "y")`` and 2-D auxiliary coordinates named
    :data:`~jem.components.coordinates.CURVILINEAR_LONGITUDE` /
    :data:`~jem.components.coordinates.CURVILINEAR_LATITUDE`
    (``longitude``/``latitude``), listed in each variable's CF ``coordinates``
    attribute.

The 2-D coordinates are **not** called ``lon``/``lat`` because those are the
atmosphere's 1-D *dimension* coordinates, and a dataset holds one variable per
name: ``xr.merge`` of an atmosphere and a curvilinear component keeps the
atmosphere's ``lon``/``lat`` and silently drops the other component's 2-D
positions, whatever ``compat`` is given (``equals``, ``no_conflicts`` and
``override`` all drop them), leaving its variables with no geolocation at all.
CF does not care what an auxiliary coordinate is called -- ``standard_name`` and
``units`` identify it and ``coordinates`` lists it -- so a name that cannot
collide costs nothing.

The index dimensions ``x``/``y`` carry no coordinate values, so two components
share them only if they share a grid *shape*: two components on the same
curvilinear grid merge exactly (and share the 2-D coordinates, which are equal),
two components on curvilinear grids of different shapes cannot be merged by
xarray, which raises on the conflicting size of ``x``/``y``.
"""

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import xarray as xr

#: Dimension names of a curvilinear grid, which are positions in index space.
CURVILINEAR_DIMS = ("x", "y")

#: Dimension and coordinate names of a separable grid -- JCM's own.
SEPARABLE_DIMS = ("lon", "lat")

#: Names of the 2-D auxiliary coordinates of a curvilinear grid. They differ
#: from :data:`SEPARABLE_DIMS` so that they can never collide in a merge with
#: the atmosphere's 1-D ``lon``/``lat`` (see the module docstring).
CURVILINEAR_LONGITUDE = "longitude"
#: The latitude counterpart of :data:`CURVILINEAR_LONGITUDE`.
CURVILINEAR_LATITUDE = "latitude"

#: The CF ``coordinates`` attribute a variable on a curvilinear grid carries.
CURVILINEAR_COORDINATES_ATTRIBUTE = f"{CURVILINEAR_LONGITUDE} {CURVILINEAR_LATITUDE}"

#: CF attributes for the horizontal coordinates, copied from
#: ``jcm.cf_metadata._COORD_ATTRS`` so every component's dataset describes
#: the axes it shares with the atmosphere identically.
_LONGITUDE_ATTRS = {
    "standard_name": "longitude",
    "units": "degrees_east",
    "long_name": "longitude",
}
_LATITUDE_ATTRS = {
    "standard_name": "latitude",
    "units": "degrees_north",
    "long_name": "latitude",
}


def separable_coordinates(
    longitude: np.ndarray, latitude: np.ndarray
) -> dict[str, tuple[Any, ...]]:
    """Return the 1-D ``lon``/``lat`` coordinates of a separable grid.

    Parameters
    ----------
    longitude, latitude : numpy.ndarray
        The axes in degrees, ``(nx,)`` and ``(ny,)``. They are written as
        given, so a caller that must match the atmosphere bit for bit passes
        the values :func:`jem.components.slab.grid.to_degrees` produces.

    Returns
    -------
    dict
        ``{"lon": (("lon",), values, attrs), "lat": ...}``, with the ``axis``
        attribute CF reserves for a true coordinate variable.

    """
    return {
        "lon": (("lon",), np.asarray(longitude), {**_LONGITUDE_ATTRS, "axis": "X"}),
        "lat": (("lat",), np.asarray(latitude), {**_LATITUDE_ATTRS, "axis": "Y"}),
    }


def curvilinear_coordinates(
    longitude: np.ndarray, latitude: np.ndarray
) -> dict[str, tuple[Any, ...]]:
    """Return the 2-D ``longitude``/``latitude`` coordinates of a curvilinear grid.

    Parameters
    ----------
    longitude, latitude : numpy.ndarray
        Each cell's position in degrees, both shaped ``(nx, ny)``.

    Returns
    -------
    dict
        ``{"longitude": (("x", "y"), values, attrs), "latitude": ...}``.
        There is no ``axis`` attribute: CF reserves it for true coordinate
        variables, and these are 2-D fields over index dimensions.

    """
    return {
        CURVILINEAR_LONGITUDE: (
            CURVILINEAR_DIMS, np.asarray(longitude), dict(_LONGITUDE_ATTRS)),
        CURVILINEAR_LATITUDE: (
            CURVILINEAR_DIMS, np.asarray(latitude), dict(_LATITUDE_ATTRS)),
    }


def geolocated_variable_attrs(
    dims: tuple[Hashable, ...], attrs: Mapping[str, Any]
) -> dict[str, Any]:
    """Return ``attrs`` plus the CF ``coordinates`` attribute, if ``dims`` need it.

    CF: a variable on index dimensions names its auxiliary coordinates, or a
    reader has no way to geolocate it. Only a variable spanning both curvilinear
    dimensions gets the attribute -- a column mean over ``x``/``y`` or a depth
    profile has no 2-D position to name.
    """
    if all(dim in dims for dim in CURVILINEAR_DIMS):
        return {**attrs, "coordinates": CURVILINEAR_COORDINATES_ATTRIBUTE}
    return dict(attrs)


def geolocate(dataset: xr.Dataset) -> xr.Dataset:
    """Add the CF ``coordinates`` attribute to every curvilinear data variable, in place.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset already carrying the 2-D coordinates of
        :func:`curvilinear_coordinates`.

    """
    for variable in dataset.data_vars.values():
        variable.attrs = geolocated_variable_attrs(variable.dims, variable.attrs)
    return dataset
