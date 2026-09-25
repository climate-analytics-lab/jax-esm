"""Surface exchange fields, read off jax-gcm's package-independent contract.

Until jax-gcm#754, each JCM physics package wrote its own struct into the
threaded diagnostics dict, under its own key, with its own names, units and
sign conventions, so this module had to carry a reader per package plus a
``detect()`` dispatch to tell them apart. jax-gcm#754 (jax-gcm PR 877) closes
that gap with :class:`jcm.physics.surface.surface_exchange.SurfaceExchange`,
published identically under ``diagnostics["surface_exchange"]`` by every
physics package that resolves a surface (SPEEDY and ECHAM; Held-Suarez opts
out because it has no surface fluxes at all). This module therefore collapses
to :func:`from_diagnostics`, the single reader :func:`jcm.physics.surface.
surface_exchange.surface_exchange_from` reads that struct through.

Sign and unit reconciliation
-----------------------------
The two conventions differ, and getting this wrong silently inverts the
coupling rather than failing loudly, so the mapping is spelled out field by
field, against ``docs/source/design/surface_exchange.md`` (jax-gcm) and this
repository's own ``CLAUDE.md`` ("Sign and mask conventions"):

============================  =================================  ========================================  ========================
Contract field (jax-gcm 877)  jax-gcm sign / units                JEM field                                  JEM sign / units
============================  =================================  ========================================  ========================
``net_heat_flux``             W m-2, **positive down** (into the  ``total_heat_flux``                        W m-2, **positive up**
                               surface medium)
``evaporation``                kg m-2 s-1, positive up            ``evaporation``                            kg m-2 s-1, positive up
``precipitation``              kg m-2 s-1, positive down, >= 0     ``precipitation``                          kg m-2 s-1, positive down
``sensible_heat_flux``        (not exchanged -- folded into        --                                         --
``latent_heat_flux``           jem's own ``total_heat_flux``)
``stress_u``/``stress_v``     (not exchanged -- JEM's own          --                                         --
                               Veros coupling computes its own
                               stress independently, see below)
``wind_speed``                m s-1, scalar                        (not exchanged -- see the wind-vector
                                                                     note below)
============================  =================================  ========================================  ========================

- **``total_heat_flux = -net_heat_flux``.** The only sign flip needed: jax-gcm
  publishes the net downward flux into the surface medium (SPEEDY's
  ``hfluxn`` convention, SW_net + LW_net - SHF - LHF); JEM's convention is the
  net flux leaving the surface into the atmosphere, i.e. exactly the
  negative. This is the same transform the pre-#754 SPEEDY-only reader
  applied to ``_surface_flux.hfluxn`` (see the numeric-equivalence test in
  ``tests/unit/test_jcm_component.py``), so a SPEEDY run's translated value
  is unchanged by this collapse -- only where it comes from changed (the
  published contract rather than a private diagnostics key), and the same
  transform now also applies, for the first time, to ECHAM (whose old
  reader always raised ``NotImplementedError``, precisely for lack of this
  contract -- see "What this replaces" below).
- **``evaporation`` and ``precipitation`` need no unit conversion.** The
  pre-#754 SPEEDY reader divided SPEEDY's *private* ``g m-2 s-1`` diagnostics
  by 1000 to reach JEM's ``kg m-2 s-1``; the *published* contract fields are
  already ``kg m-2 s-1`` (jax-gcm normalises them once, in the publisher, to
  the contract's units -- see ``SpeedySurfaceFlux._publish_surface_exchange``
  and ``EchamSurfaceExchange.__call__``), so no conversion happens here any
  more. **Verified empirically, not just read off the source**: stepping a
  real SPEEDY model, ``diagnostics["surface_exchange"].evaporation`` is
  exactly ``1/1000`` of the same step's private ``_surface_flux.evap`` (g
  m-2 s-1) -- keeping the old division would have made evaporation 1000x too
  small, silently, and the numeric equivalence test below (which compares
  the OLD adapter's g/1000 arithmetic against this module's un-converted
  read of the SAME diagnostics dict, on precipitation too, once it is
  non-zero a few steps into a real run) is what actually catches either
  direction of that mistake rather than trusting either the source or a
  one-line reading of it. Similarly, ``precipitation`` was previously
  assembled by hand from two SPEEDY-only diagnostics entries
  (``_convection.precnv + _condensation.precls``); the contract's
  ``precipitation`` is already that same total (convective + large-scale for
  SPEEDY, convective + stratiform for ECHAM), computed once, in the
  publisher, from whichever precipitation-producing terms are actually
  composed -- so this module no longer needs to know which terms those are,
  or that they differ between the two packages.
- **``sensible_heat_flux``/``latent_heat_flux``/``stress_u``/``stress_v`` are
  on the contract but JEM does not exchange them separately**: the slab
  ocean/land/sea-ice models and ``jem.exchangers`` couple on the *net*
  ``total_heat_flux`` (matching the pre-#754 behaviour, which also only ever
  exchanged the net), and momentum is not exchanged through this struct at
  all -- see the wind-vector note below.

Why the near-surface wind is still a narrow exception
-------------------------------------------------------
:attr:`~jem.components.jcm.exchange_fields.SurfaceExchange.u0`/``v0`` are the
near-surface wind's true-east/true-north *components*, needed by
:func:`jem.fluxes.bulk_wind_stress` (the independent bulk-drag law
:class:`jem.fluxes.VerosExchange` applies for a Veros ocean, deliberately
**not** reusing jax-gcm's own delivered stress -- see that module's
docstring). jax-gcm's #754 contract does **not** publish a wind vector, only
the scalar ``wind_speed`` -- a direction cannot be recovered from a
magnitude, so ``wind_speed`` cannot stand in for ``u0``/``v0`` here, and
JAX-ESM does **not** synthesise one from it, or from the contract's
``stress_u``/``stress_v`` (a separate physics decision -- see
:mod:`jem.fluxes`'s docstring on why ``bulk_wind_stress`` is an independent
drag law rather than a reuse of jax-gcm's own delivered stress).

This is not a gap #754 could have closed and didn't: SPEEDY happens to still
carry a true wind vector internally, as an artefact of its bulk-formula
extrapolation to the surface layer, in its own *private*, non-contract
diagnostics key (``_surface_flux.u0``/``.v0``,
``jcm/physics/surface/speedy_surface_flux.py`` -- the same source
``SpeedySurfaceFlux._publish_surface_exchange`` reads to build the
contract's ``wind_speed = sqrt(u0**2 + v0**2)``). ECHAM has no vector wind
anywhere in its diagnostics to publish, contract or no contract: its
boundary-layer scheme diagnoses only a wind *speed*
(``vertical_diffusion.wind_10m``, ``jcm/physics/vertical_diffusion/tte_tke/
vertical_diffusion_types.py``: ``|U(10 m)|``, a scalar). So this asymmetry
between the two packages predates #754 and is not introduced by this
collapse.

**jax-esm#129 made that absence explicit and static, rather than a raise.**
:func:`from_diagnostics` still keeps exactly one package-specific read after
the collapse -- for ``u0``/``v0`` only, off SPEEDY's private key -- but where
it used to raise ``NotImplementedError`` for any other package, it now
returns ``u0=None``/``v0=None``: whether a composed physics package publishes
a wind vector is a fact fixed at composition (it never changes step to step
for a given model), so the absence is a **static** property of the carry --
``None`` is an empty JAX pytree node, so it survives ``jit``, the coupled
``lax.scan``, a checkpoint round trip and output serialization the same way
an entirely-absent field of a delegated component's carry already does (see
``jem/checkpoint.py``'s own docstring on that point) -- rather than a value
guessed at (zero, NaN, or something derived from ``wind_speed``/the
contract's stress). :meth:`jem.components.jcm.component.JCMComponent.step`
fills :class:`~jem.components.jcm.component.JCMDerived`'s ``u0``/``v0`` from
exactly this, and
:meth:`~jem.components.jcm.component.JCMDerived.zeros` (the initial carry)
decides the *same* way from a structural template of the diagnostics dict, so
the field is ``None`` in the carry from step 0 onward for a windless package
-- never an array on one step and ``None`` on another, which is what a
``lax.scan`` carry structure forbids.

So an ECHAM-composed coupled model now completes a step, whatever the
exchanger, exactly as a SPEEDY one does, as long as nothing it runs actually
*needs* the wind vector. Today the one thing that does is
:class:`jem.fluxes.VerosExchange` (via :func:`jem.fluxes.bulk_wind_stress`),
and it is checked at **composition time**: :meth:`jem.fluxes.VerosExchange.
validate` (invoked by :func:`jem.runners._validate_exchangers`, inside
:func:`jem.runners.build_coupler`, in the same slot
``ComposablePhysics.require_surface_exchange`` fills for the surface struct
itself -- see :func:`jem.runners.build_atmosphere`) raises, naming the
composed atmosphere's physics and jax-esm#132, if the atmosphere it is
coupled to publishes no wind vector -- so an ECHAM/Veros combination is
**refused at build time**, before a coupled run is even compiled, rather
than mid-run or with a value that quietly means nothing. This is reachable
through a shipped command, not only a hand-built coupler: ``python -m
jem.main +configuration=veros-earth physics@atmosphere.physics=echam``
reaches ``build_coupler`` too, since JAX-ESM reuses jax-gcm's own ``physics``
config group and jax-gcm ships ``echam.yaml``. Choosing this exchanger's
wind-stress source for a windless atmosphere is the decision jax-esm#132
tracks; jax-esm#129 (this collapse) closes only the eager-read failure
above.

What this replaces
-------------------
Before jax-gcm#754, this module read SPEEDY's private ``_surface_flux``/
``_convection``/``_condensation`` diagnostics keys by hand (negating
``hfluxn``, dividing g m-2 s-1 fields by 1000, summing two precipitation
terms) and could not build a grid-mean struct for ECHAM at all -- its
``echam()`` reader always raised ``NotImplementedError``, naming jax-gcm#754
as the fix, with "Use SPEEDY physics for coupled runs until then." A
``detect()`` function picked between the two readers by which package-marker
key was present in the diagnostics dict. All three -- ``speedy()``,
``echam()``, ``detect()`` -- are gone: the heat and water fluxes both
packages deliver are now read identically, off the one contract. ECHAM's
grid-mean heat and water fluxes are therefore published and translatable for
the first time, and -- since jax-esm#129 made the absent wind vector a static
``None`` rather than a raise, as described above -- an ECHAM coupled step now
completes on them like any other package's. The old ``speedy()`` reader's source
(commit ``756cc2c``, the last commit before this collapse) is vendored,
frozen, as
``tests/unit/_pre754_exchange_reader.py`` and used directly, as the
historical baseline, by the numeric old-vs-new equivalence test in
``tests/unit/test_jcm_component.py`` -- vendored rather than loaded from git
history at test time (as an earlier version of that test did) because CI's
shallow ``actions/checkout`` does not have commit ``756cc2c`` in its object
store.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jcm.physics.surface.surface_exchange import surface_exchange_from

#: SPEEDY's private, non-contract diagnostics key that still carries the
#: near-surface wind *vector* (see the module docstring's wind-vector note).
#: Not a "physics package marker" in the old ``detect()`` sense -- the
#: heat/water fluxes above never look at this key, and a package that does
#: not write it (ECHAM, or any future package) still gets a valid
#: :class:`SurfaceExchange` from :func:`from_diagnostics`, just without
#: ``u0``/``v0``.
_SPEEDY_WIND_VECTOR_KEY = "_surface_flux"


class SurfaceExchange(NamedTuple):
    """Surface fluxes and near-surface wind, in JEM's conventions.

    Every field is a ``(ix, il)`` horizontal map on the atmosphere's nodal
    grid (the grid-cell mean over land and sea -- jax-gcm's contract
    guarantees only the grid mean; see
    ``docs/source/design/surface_exchange.md``).

    Attributes
    ----------
    total_heat_flux : jax.Array
        Net heat flux out of the surface, ``W m-2``, **positive upward**.
    evaporation : jax.Array
        Evaporation, ``kg m-2 s-1``, positive upward.
    precipitation : jax.Array
        Total precipitation (convective plus large-scale/stratiform)
        reaching the surface, ``kg m-2 s-1``, positive downward.
    u0 : jax.Array or None
        Near-surface zonal wind, ``m s-1``, or ``None`` where the composed
        physics package publishes no wind *vector* -- today, everything but
        SPEEDY (see the module docstring). Whether this is ``None`` is a
        **static** fact of the composed physics, fixed once at model
        construction and identical on every step, never a per-step decision
        -- see :func:`has_wind_vector`.
    v0 : jax.Array or None
        Near-surface meridional wind, ``m s-1``. Same caveat as ``u0``.

    """

    total_heat_flux: jnp.ndarray
    evaporation: jnp.ndarray
    precipitation: jnp.ndarray
    u0: jnp.ndarray | None
    v0: jnp.ndarray | None


def has_wind_vector(diagnostics: dict[str, Any]) -> bool:
    """Whether ``diagnostics`` carries a near-surface wind *vector*.

    True only when the composed physics package writes SPEEDY's private
    :data:`_SPEEDY_WIND_VECTOR_KEY` entry (see the module docstring's
    "Why the near-surface wind is still a narrow exception" section) --
    today, SPEEDY only. This is a **structural** question -- it can be
    answered from a diagnostics *template* (all-zero leaves, the right keys)
    just as well as from a real step's output. It is the single tested
    predicate the absence decision is made from: :func:`_near_surface_wind_vector`
    calls it directly (rather than duplicating the ``dict.get`` check), so
    both :meth:`~jem.components.jcm.component.JCMComponent.step` (a real
    diagnostics dict) and :meth:`~jem.components.jcm.component.JCMDerived.zeros`
    (a structural template, through the same :func:`from_diagnostics` call)
    decide ``u0``/``v0``'s presence the exact same way, once each, rather
    than through two predicates that could in principle disagree.

    Parameters
    ----------
    diagnostics : dict
        A physics diagnostics dict, or a structural zero-filled template of
        one (:meth:`jcm.physics_interface.Physics.get_empty_data`).

    Returns
    -------
    bool

    """
    return diagnostics.get(_SPEEDY_WIND_VECTOR_KEY) is not None


def _near_surface_wind_vector(
    diagnostics: dict[str, Any]
) -> tuple[Any, Any] | tuple[None, None]:
    """Return ``(u0, v0)``, or ``(None, None)`` if this physics package has none.

    See the module docstring's "Why the near-surface wind is still a narrow
    exception" section: jax-gcm's #754 contract has no wind vector, only the
    scalar ``wind_speed``, so this is the one read left that is not off the
    published struct.

    Before jax-esm#129 this raised ``NotImplementedError`` for any package
    other than SPEEDY, which made ``JCMComponent.step`` fail on *every*
    ECHAM-composed coupled step (it calls :func:`from_diagnostics`, which
    calls this, unconditionally). #129 made the absence a value instead: a
    package either always writes :data:`_SPEEDY_WIND_VECTOR_KEY` or never
    does (see :func:`has_wind_vector`), so returning ``(None, None)`` here is
    exactly as safe as raising was, and lets a caller that does not need the
    wind (the heat/water fluxes above, or any exchanger that never reads
    ``u0``/``v0``) keep going. A caller that does need it --
    :class:`jem.fluxes.VerosExchange` -- is instead responsible for checking
    at composition time (:meth:`~jem.fluxes.VerosExchange.validate`) and
    again, defensively, wherever it is actually used.
    """
    if not has_wind_vector(diagnostics):
        return None, None
    speedy_flux = diagnostics[_SPEEDY_WIND_VECTOR_KEY]
    return speedy_flux.u0, speedy_flux.v0


def from_diagnostics(diagnostics: dict[str, Any]) -> SurfaceExchange:
    """Read the surface exchange out of one step's physics diagnostics.

    The single reader jax-gcm#754 (jax-gcm PR 877) makes possible: every
    guaranteed field of :class:`jcm.physics.surface.surface_exchange.
    SurfaceExchange` is filled identically by every physics package that
    resolves a surface, so this function no longer needs to know which
    package produced ``diagnostics``. See the module docstring for the
    field-by-field sign/unit mapping.

    Parameters
    ----------
    diagnostics : dict
        One coupling step's physics diagnostics dict, with the length-1
        save axis already stripped.

    Returns
    -------
    SurfaceExchange
        The fluxes translated to JEM's sign and unit conventions.
        ``u0``/``v0`` are ``None`` when the composed physics package
        publishes no near-surface wind vector (see :func:`has_wind_vector`
        and the module docstring's wind-vector note) -- never raised, since
        jax-esm#129: whether the wind is available is a static fact of the
        composed physics, not a per-call failure, so a caller that does not
        need it (the heat/water fluxes above) is unaffected, and a caller
        that does (:class:`jem.fluxes.VerosExchange`) is checked at
        composition time instead of here.

    Raises
    ------
    KeyError
        If the composed physics package publishes no ``surface_exchange``
        struct at all (Held-Suarez opts out -- it resolves no surface
        fluxes). Raised by
        :func:`jcm.physics.surface.surface_exchange.surface_exchange_from`,
        which names the packages that do publish and points at the design
        doc. Prefer
        :meth:`~jcm.physics.composable_physics.ComposablePhysics.require_surface_exchange`
        at composition time (see ``jem.runners.build_atmosphere``) so this
        is caught before the first coupled step rather than during it.

    """
    exchange = surface_exchange_from(diagnostics)
    u0, v0 = _near_surface_wind_vector(diagnostics)
    return SurfaceExchange(
        total_heat_flux=-exchange.net_heat_flux,
        evaporation=exchange.evaporation,
        precipitation=exchange.precipitation,
        u0=u0,
        v0=v0,
    )
