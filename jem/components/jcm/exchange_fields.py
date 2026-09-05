"""Surface exchange fields read out of a JCM physics-diagnostics dict.

JCM does not (yet) publish a package-independent surface-exchange contract:
each physics package writes its own struct into the threaded diagnostics
dict under its own key, with its own names, units and sign conventions.
This module is the single place where those package-specific layouts are
translated into the one struct the coupler exchanges,
:class:`SurfaceExchange`, in JEM's conventions:

- heat flux **positive upward** (out of the surface, into the atmosphere),
- water fluxes in ``kg m-2 s-1``, evaporation upward and precipitation
  downward,
- near-surface wind in ``m s-1``.

Adding a physics package means adding one reader here and one entry in
:func:`detect`; nothing else in JEM needs to know which physics a
:class:`~jem.components.jcm.component.JCMComponent` was built with.

The long-term fix is jax-gcm#754 — a ``SurfaceExchange``-like struct
published by every JCM physics package, the way units tables already are.
Once that lands this module collapses to a single reader.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp

# JCM's SPEEDY physics reports the surface water fluxes as mass density
# fluxes in g m-2 s-1 (``jcm/physics/speedy/units_table.csv``: ``evap``,
# ``precnv``, ``precls``); JEM works in SI, so every one of them is divided
# by this.
GRAMS_PER_KILOGRAM = 1000.0

# Diagnostics keys that identify a physics package. SPEEDY's surface-flux
# term writes ``_surface_flux`` (leading underscore: a struct, not a plain
# output array; ``jcm/physics/surface/speedy_surface_flux.py``); the ECHAM
# surface term declares ``provides = ("surface",)`` and writes a grid-mean
# ``SurfaceData`` there (``jcm/physics/surface/echam/surface_types.py``).
# The two are mutually exclusive in practice, which is what makes detection
# by key safe.
SPEEDY_SURFACE_KEY = "_surface_flux"
ECHAM_SURFACE_KEY = "surface"


class SurfaceExchange(NamedTuple):
    """Surface fluxes and near-surface wind, in JEM's conventions.

    Every field is a ``(ix, il)`` horizontal map on the atmosphere's nodal
    grid (the grid-cell mean over land and sea; JCM no longer publishes the
    per-tile breakdown).

    Attributes
    ----------
    total_heat_flux : jax.Array
        Net heat flux out of the surface, ``W m-2``, **positive upward**.
    evaporation : jax.Array
        Evaporation, ``kg m-2 s-1``, positive upward.
    precipitation : jax.Array
        Total precipitation (convective plus large-scale) reaching the
        surface, ``kg m-2 s-1``, positive downward.
    u0 : jax.Array
        Near-surface zonal wind, ``m s-1``.
    v0 : jax.Array
        Near-surface meridional wind, ``m s-1``.

    """

    total_heat_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    u0: jnp.ndarray
    v0: jnp.ndarray


def speedy(diagnostics: dict[str, Any]) -> SurfaceExchange:
    """Read the surface exchange out of SPEEDY physics diagnostics.

    Parameters
    ----------
    diagnostics : dict
        One coupling step's physics diagnostics dict, with the length-1
        save axis already stripped.

    Returns
    -------
    SurfaceExchange
        The fluxes converted to JEM's sign and unit conventions.

    Notes
    -----
    Source fields and their JCM conventions (``jcm/physics/surface/
    speedy_surface_flux.py`` and ``jcm/physics/speedy/units_table.csv``):

    ``_surface_flux.hfluxn``
        ``W m-2``, net heat flux **downward** into the surface, weighted by
        land fraction over the land/sea tiles. JEM takes heat flux positive
        upward, so it is negated exactly here, at the component boundary.
    ``_surface_flux.evap``
        ``g m-2 s-1``, evaporation, already upward-positive.
    ``_convection.precnv``, ``_condensation.precls``
        ``g m-2 s-1``, convective and large-scale precipitation, downward.
    ``_surface_flux.u0``, ``_surface_flux.v0``
        ``m s-1``, wind extrapolated to the surface layer (sigma = 0.99).

    """
    surface_flux = diagnostics[SPEEDY_SURFACE_KEY]
    precipitation = (
        diagnostics["_convection"].precnv + diagnostics["_condensation"].precls
    )
    return SurfaceExchange(
        total_heat_flux=-surface_flux.hfluxn,
        evaporation=surface_flux.evap / GRAMS_PER_KILOGRAM,
        precipitation=precipitation / GRAMS_PER_KILOGRAM,
        u0=surface_flux.u0,
        v0=surface_flux.v0,
    )


def echam(diagnostics: dict[str, Any]) -> SurfaceExchange:
    """Read the surface exchange out of ECHAM physics diagnostics.

    Raises
    ------
    NotImplementedError
        Always, until JCM publishes a package-independent surface-exchange
        struct (jax-gcm#754).

    Notes
    -----
    The pieces exist in JCM but not in one place and not in one convention.
    The ECHAM surface term publishes a grid-mean ``SurfaceData`` under the
    ``"surface"`` key (``jcm/physics/surface/echam/surface_types.py``) that
    carries **no** net heat flux and **no** precipitation — the component
    sensible / latent / longwave / shortwave / ground-heat fluxes are held
    per surface type (water, ice, land) — while precipitation lives in the
    cloud microphysics under ``diagnostics["clouds"].precip_rain`` and
    ``.precip_snow``, in ``kg m-2 s-1``.

    Summing and area-weighting those here would reimplement ECHAM's surface
    energy balance in the coupler, in a form that silently rots the first
    time the tiling or the term composition changes. So this fails loudly
    instead, and waits for jax-gcm#754 to publish the same struct from every
    physics package.

    """
    raise NotImplementedError(
        "The ECHAM surface exchange is not implemented: the 'surface'"
        " diagnostics key carries per-surface-type sensible/latent/radiative"
        " fluxes with no grid-mean net heat flux, and ECHAM's precipitation"
        " lives in diagnostics['clouds'].precip_rain/precip_snow, so the"
        " quantities the coupler exchanges would have to be reassembled --"
        " an ECHAM surface energy balance reimplemented inside JEM. This"
        " waits on jax-gcm#754 (a surface-exchange struct published by every"
        " JCM physics package). Use SPEEDY physics for coupled runs until"
        " then."
    )


def detect(diagnostics: dict[str, Any]) -> Callable[[dict[str, Any]], SurfaceExchange]:
    """Return the reader matching the physics package that wrote ``diagnostics``.

    Parameters
    ----------
    diagnostics : dict
        One coupling step's physics diagnostics dict.

    Returns
    -------
    callable
        :func:`speedy` or :func:`echam`.

    Raises
    ------
    KeyError
        If no known surface key is present. The message lists the keys that
        *are* present, because that is what identifies the physics package
        the caller actually built.

    """
    if SPEEDY_SURFACE_KEY in diagnostics:
        return speedy
    if ECHAM_SURFACE_KEY in diagnostics:
        return echam
    raise KeyError(
        "No known surface-flux diagnostics in the JCM physics output: expected"
        f" {SPEEDY_SURFACE_KEY!r} (SPEEDY) or {ECHAM_SURFACE_KEY!r} (ECHAM),"
        f" found {sorted(diagnostics)!r}. A physics package without a surface"
        " term cannot be coupled to a surface component."
    )
