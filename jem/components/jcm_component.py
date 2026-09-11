"""Deprecated entry point for the JCM adapter.

The adapter now lives in :mod:`jem.components.jcm` as
:class:`~jem.components.jcm.component.JCMComponent`, a wrapper object that
satisfies the :class:`~jem.base.component.Component` protocol. This module
keeps the old name importable for one release.
"""

import warnings

import jax_datetime as jdt
from jcm.model import Model

from jem.components.jcm.component import JCMComponent, JCMDerived

__all__ = [
    "JCMComponent",
    "JCMDerived",
    "make_jem_compatible",
]


def make_jem_compatible(
    model: Model,
    coupling_timestep: jdt.Timedelta,
) -> JCMComponent:
    """Return a :class:`~jem.components.jcm.component.JCMComponent` for ``model``.

    .. deprecated::
        Construct ``JCMComponent(model)`` directly and register it with a
        ``Coupler``, which binds the coupling timestep.

    Parameters
    ----------
    model : jcm.model.Model
        The atmosphere to wrap.
    coupling_timestep : jax_datetime.Timedelta
        Ignored. The coupler now supplies the coupling timestep, together
        with the start date and calendar, through
        :meth:`~jem.components.jcm.component.JCMComponent.bind`.

    Returns
    -------
    JCMComponent
        A new wrapper. ``model`` itself is left untouched.

    """
    warnings.warn(
        "make_jem_compatible() is deprecated: use"
        " jem.components.jcm.JCMComponent(model) and register it with a"
        " Coupler. The returned component is a wrapper object -- unlike the"
        " old function it does not inject initialize()/step() methods onto"
        " the jcm Model, so anything calling model.initialize() must call"
        " component.initialize() instead. The coupling_timestep argument is"
        " ignored; the Coupler passes it to JCMComponent.bind().",
        DeprecationWarning,
        stacklevel=2,
    )
    return JCMComponent(model)
