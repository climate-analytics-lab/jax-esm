"""A component's view of the output time coordinate.

A coupled run produces one dataset per component. They are only useful
together if ``xr.merge`` aligns them, and that requires the ``time``
coordinate to carry not just the same *instants* but the same
*representation* of them, down to the last bit.

That representation is defined once, on
:class:`jem.base.component.TimeAxis` -- the object the coupler hands to every
component's ``to_xarray`` -- and this module is the call site every component
shares (the slab models and the Veros adapter), not a second implementation:
:func:`time_coordinate` is the two-line convenience that unpacks a
``TimeAxis`` into the ``(values, attrs)`` pair :class:`xarray.Dataset` wants.
See ``TimeAxis.datetimes`` for the convention itself (JCM's float64-days
arithmetic, end-of-interval labelling, and the sub-day start date JCM drops
and JAX-ESM keeps).
"""

import numpy as np

from jem.base.component import TimeAxis


def time_coordinate(time: TimeAxis) -> tuple[np.ndarray, dict[str, str]]:
    """Return the ``time`` coordinate values and attributes for an output file.

    Parameters
    ----------
    time : jem.base.component.TimeAxis
        The coupler's description of the records being written: the run's
        start date, the coupling timestep and the coupled-step index of each
        record.

    Returns
    -------
    values : numpy.ndarray
        ``datetime64[ns]``, one per record, stamped at the end of the coupling
        step the record came from.
    attrs : dict of str
        CF attributes for the coordinate. A fresh dict per call, because
        xarray stores what it is given and a shared one would let a caller's
        edit reach every other dataset.

    """
    return time.datetimes(), dict(time.attrs)
