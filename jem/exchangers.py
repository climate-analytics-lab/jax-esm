"""Declarative exchangers: the standard coupling written as a table of fields.

An *exchanger* is the one place a component's carry is read by another
(:data:`jem.base.component.Exchanger`). Written by hand it is a small
function that pulls fields out of one carry and puts them into another with
``.replace()``; written here it is a **table**, because almost every exchange
in a coupled Earth-system model is exactly that: field X of component A
becomes field Y of component B, optionally regridded on the way.

:class:`ExchangeSpec` is one row of that table, :class:`Exchange` executes
it, and :func:`default_exchanges` is the standard atmosphere/ocean/land/
sea-ice wiring that every example in this repository writes out by hand
today. A declarative exchange is not more capable than a hand-written one --
an exchanger that computes a flux, converts units or blends two fields still
has to be a function -- but it is checkable: :meth:`Exchange.validate` names
a mistyped field *before* a run starts, and :meth:`Exchange.__repr__` prints
the whole coupling as a table.

Carry layout
------------
A spec addresses a field as ``"component.section.field"``, where ``section``
is one of :data:`SECTIONS` -- ``state``, ``derived`` or ``forcing``. That is
the layout every packaged component uses: the carry is a mapping whose
``"state"``, ``"derived"`` and ``"forcing"`` entries are structs
(``tree_math.struct``, ``flax.struct``) holding the fields, with anything
else a component needs (the slab models' ``"params"``, the JCM wrapper's
``"physics"``) alongside them. The three sections mean, respectively: what
the component integrates, what it diagnosed for others to read, and what it
was given.

Component wrappers do not all name the same physical field the same way, and
they do not all keep it in the same section: the Veros ocean takes its surface
heat flux as ``forcing.heat_flux`` and publishes its sea surface temperature
from ``derived``, where a slab ocean has ``forcing.total_heat_flux`` and
``state.sea_surface_temperature``. So there is one standard table per carry
layout -- :data:`STANDARD_EXCHANGES` and :data:`VEROS_OCEAN_EXCHANGES` --
and :func:`default_exchanges` picks by the type of the component registered as
``"ocn"``. A carry that neither describes is coupled with a hand-written
exchanger.

Lagged coupling
---------------
This module changes nothing about *when* fields move, and the default
workflow couples with a lag of one coupling step. With the default workflow
``["exchange", "atm", "ocn"]``:

- ``exchange`` runs **first**, on the carries as they were left at the end of
  step *n-1*. So at step *n* the ocean is driven by the atmosphere's fluxes
  from step *n-1*, and the atmosphere sees the sea surface temperature the
  ocean reached at the end of step *n-1*.
- On the **first** step there is no previous step, so each component receives
  whatever its ``initialize()`` put in its forcing section -- zeros, for
  every packaged component. A run therefore begins with one step of
  uncoupled spin-up: the ocean's first step sees no heat flux at all.
- The lag is a property of the *workflow*, not of the exchanger: an
  ``["atm", "exchange", "ocn"]`` workflow hands the ocean the atmosphere's
  fluxes from the same step, at the cost of giving the atmosphere a
  two-step-old sea surface temperature. Neither order gives every component
  same-step information; that needs either an iterated (implicit) exchange or
  a partitioned workflow, which is a follow-up.

Writing the default coupling down in one place is what makes that lag
reviewable at all: before, every example spelled the same exchange out by
hand and none of them said what step the fields came from.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from jem.base.component import Carry, Component, CouplingTime, Exchanger

logger = logging.getLogger(__name__)

#: The carry sections a spec may address. See the module docstring.
SECTIONS = ("state", "derived", "forcing")

#: The component names :func:`default_exchanges` knows the standard wiring
#: for. A coupler may register its components under any names it likes; the
#: default wiring is only defined for these, and is filtered to the ones
#: actually present. Every packaged component is constructed with the name it
#: is wired under here, so the standard wiring needs no renaming.
STANDARD_COMPONENT_NAMES = ("atm", "ocn", "lnd", "seaice")

# Names a coupled model plausibly uses for a component the standard wiring
# knows under another name. A component registered under one of these is not
# wired -- the wiring is by name and guessing would be worse -- but it is worth
# a warning, because the failure it would otherwise cause is silent: a sea-ice
# model registered as "ice" would simply never receive or publish anything.
_NEAR_MISS_NAMES = {
    "ice": "seaice",
    "sea_ice": "seaice",
    "sic": "seaice",
    "ocean": "ocn",
    "land": "lnd",
    "atmosphere": "atm",
}

#: Components that, in a mixed-grid configuration, live on the *ocean* grid
#: rather than the atmosphere's. Which specs cross a grid boundary -- and so
#: may need a regridder -- is decided from this: the land surface shares the
#: atmosphere's grid in every configuration JAX-ESM ships, while the ocean
#: and the sea ice share the ocean model's.
OCEAN_GRID_COMPONENTS = ("ocn", "seaice")

#: The name :func:`default_exchangers` registers its exchanger under, and so
#: the name the default workflow runs first.
DEFAULT_EXCHANGER_NAME = "exchange"

#: What a row's third element says: whether the quantity is **extensive**
#: (a flux, an energy, an areal fraction -- regridded conservatively so its
#: budget survives the interface) or **intensive** (a state variable such as a
#: temperature -- interpolated bilinearly, which does not leave a conservative
#: map's staircase in a smooth field). It is written on each row rather than
#: inferred from the carry section the field is read from, because the two do
#: not agree: a slab ocean publishes its sea surface temperature from
#: ``state`` and a Veros ocean the same temperature from ``derived``, and it
#: is the same intensive field either way.
KINDS = ("flux", "state")

#: The standard coupling, as ``(src, dst, kind)`` rows. Kept as strings rather
#: than built :class:`ExchangeSpec` objects so that this table reads as the
#: documentation it is; :func:`default_exchanges` turns the rows that apply
#: into specs. This is the table for JAX-ESM's own slab components; an ocean
#: with a carry of its own has its own (:data:`VEROS_OCEAN_EXCHANGES`).
STANDARD_EXCHANGES: tuple[tuple[str, str, str], ...] = (
    # Surface energy: the atmosphere's net surface heat flux (upward
    # positive, already converted from JCM's downward-positive `hfluxn` by
    # the JCM wrapper) drives both surfaces.
    ("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux", "flux"),
    ("atm.derived.total_heat_flux", "lnd.forcing.total_heat_flux", "flux"),
    # The mixed layer's freeze/melt potential (CESM's `frzmlt`) is what the
    # sea ice grows and melts on.
    ("ocn.derived.ice_frazil_melt_energy",
     "seaice.forcing.ice_frazil_melt_energy", "flux"),
    # The surface state the atmosphere's surface-flux scheme reads. In an
    # uncoupled JCM run these are prescribed boundary conditions; here the
    # surface components provide them, under JCM's own field names.
    ("ocn.state.sea_surface_temperature",
     "atm.forcing.sea_surface_temperature", "state"),
    ("seaice.derived.ice_fraction", "atm.forcing.sice_am", "flux"),
    ("lnd.state.land_surface_temperature", "atm.forcing.stl_am", "state"),
    ("lnd.state.snowc", "atm.forcing.snowc_am", "state"),
    ("lnd.state.soilw", "atm.forcing.soilw_am", "state"),
)

#: The standard coupling when the ocean is the **Veros** GCM
#: (:class:`jem.components.veros_component.VerosComponent`), whose carry is
#: laid out differently from a slab's: the surface heat flux arrives as
#: ``forcing.heat_flux`` rather than ``forcing.total_heat_flux``, the ocean
#: takes a freshwater flux as well, and the sea surface temperature is
#: published from ``derived`` (Veros' ``state`` is Veros' own ``VerosState``
#: object, which is not a struct of exchangeable fields). The rows that do not
#: involve the ocean are the same as :data:`STANDARD_EXCHANGES`'.
#:
#: What this table deliberately does **not** carry, because no copy of a field
#: can express it -- each needs a hand-written exchanger
#: (``coupling.exchanger``):
#:
#: - **wind stress.** Veros integrates ``forcing.surface_taux/tauy``; the
#:   atmosphere publishes a near-surface *wind* (``derived.u0``/``v0``). Going
#:   from one to the other is a bulk drag law, and on a rotated ocean grid it
#:   is followed by a rotation into the grid's local frame.
#: - **the "swamp" sea-ice insulation** the example drivers apply, which
#:   masks the heat and freshwater fluxes wherever the surface has reached the
#:   freezing point.
#: - **the freeze/melt potential a slab sea ice runs on.** Veros publishes no
#:   ``ice_frazil_melt_energy``, so there is no ``ocn`` to ``seaice`` row and
#:   a sea-ice component coupled to a Veros ocean is not driven by it;
#:   :func:`default_exchanges` warns when it sees that combination.
#:
#: ``forcing.surface_air_temperature`` has no row either: Veros carries it
#: only to write it back out as a diagnostic, and nothing in the integration
#: reads it.
VEROS_OCEAN_EXCHANGES: tuple[tuple[str, str, str], ...] = (
    ("atm.derived.total_heat_flux", "ocn.forcing.heat_flux", "flux"),
    ("atm.derived.total_freshwater_flux", "ocn.forcing.freshwater_flux", "flux"),
    ("atm.derived.total_heat_flux", "lnd.forcing.total_heat_flux", "flux"),
    ("ocn.derived.sea_surface_temperature",
     "atm.forcing.sea_surface_temperature", "state"),
    ("seaice.derived.ice_fraction", "atm.forcing.sice_am", "flux"),
    ("lnd.state.land_surface_temperature", "atm.forcing.stl_am", "state"),
    ("lnd.state.snowc", "atm.forcing.snowc_am", "state"),
    ("lnd.state.soilw", "atm.forcing.soilw_am", "state"),
)

#: Module holding the ocean wrapper whose carry :data:`VEROS_OCEAN_EXCHANGES`
#: describes. It is looked up in ``sys.modules`` rather than imported: Veros is
#: an optional dependency and :mod:`jem.components.veros_component` imports it
#: at module scope, so importing that module here would make every coupled run
#: -- slab ones included -- depend on Veros being installed. If it has never
#: been imported then nothing in the process can be a ``VerosComponent``, which
#: is the answer without importing anything at all.
VEROS_COMPONENT_MODULE = "jem.components.veros_component"


def _parse_path(path: str, end: str, spec: Any) -> tuple[str, str, str]:
    """Split a ``"component.section.field"`` path, or raise naming the spec."""
    parts = path.split(".")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Exchange spec {spec}: the {end} path {path!r} is not of the form "
            "\"component.section.field\"."
        )
    component, section, field = parts
    if section not in SECTIONS:
        raise ValueError(
            f"Exchange spec {spec}: the {end} path {path!r} names the carry "
            f"section {section!r}, which is not one of {list(SECTIONS)!r}."
        )
    return component, section, field


@dataclasses.dataclass(frozen=True)
class ExchangeSpec:
    """One field moved from one component to another.

    Attributes
    ----------
    src : str
        Where the value is read: ``"component.section.field"``, with
        ``section`` one of :data:`SECTIONS`.
    dst : str
        Where it is written, in the same form.
    regrid : str or None
        Name of the regridder to apply on the way, looked up in the mapping
        :class:`Exchange` was built with. ``None`` (the default) copies the
        field unchanged, which is right whenever the two components share a
        grid.

    """

    src: str
    dst: str
    regrid: str | None = None

    def __post_init__(self) -> None:
        """Reject a malformed path at construction rather than at trace time."""
        _parse_path(self.src, "source", self)
        _parse_path(self.dst, "destination", self)

    def __str__(self) -> str:
        """Return the spec as one line of the coupling table."""
        via = f"  [{self.regrid}]" if self.regrid else ""
        return f"{self.src} -> {self.dst}{via}"

    @property
    def src_parts(self) -> tuple[str, str, str]:
        """Return the source as ``(component, section, field)``."""
        return _parse_path(self.src, "source", self)

    @property
    def dst_parts(self) -> tuple[str, str, str]:
        """Return the destination as ``(component, section, field)``."""
        return _parse_path(self.dst, "destination", self)


def _as_spec(spec: ExchangeSpec | Mapping[str, Any]) -> ExchangeSpec:
    """Return ``spec`` as an :class:`ExchangeSpec`, accepting a mapping.

    A mapping is accepted so that a coupling table can come straight out of
    YAML (``{src: ..., dst: ..., regrid: ...}``) without the configuration
    layer having to import this module to build the dataclass.
    """
    if isinstance(spec, ExchangeSpec):
        return spec
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"src", "dst", "regrid"}
        if unknown:
            raise ValueError(
                f"Exchange spec {dict(spec)!r} has unknown key(s) "
                f"{sorted(unknown)!r}; an exchange spec has `src`, `dst` and "
                "an optional `regrid`."
            )
        try:
            return ExchangeSpec(**spec)
        except TypeError as error:
            raise ValueError(
                f"Exchange spec {dict(spec)!r} is incomplete: {error}"
            ) from error
    raise TypeError(
        f"An exchange spec must be an ExchangeSpec or a mapping with `src`/"
        f"`dst`; got {type(spec).__name__}."
    )


def _field_names(section: Any) -> list[str]:
    """Return the field names of a carry section, for an error message."""
    if dataclasses.is_dataclass(section) and not isinstance(section, type):
        return [field.name for field in dataclasses.fields(section)]
    return sorted(name for name in dir(section) if not name.startswith("_"))


class Exchange:
    """An :data:`~jem.base.component.Exchanger` built from a table of specs.

    Calling it reads every source field from the carries it is handed,
    regrids the ones whose spec names a regridder, and returns a new mapping
    in which every destination field has been replaced. Nothing is written in
    place: new section structs are built with ``.replace(...)``, new carries
    with ``dict(carry, ...)``, and the mapping itself is rebuilt -- which is
    what the exchanger contract requires and what makes an exchange safe to
    re-run and to differentiate through.

    **Every source is read from the mapping as it arrives**, before any
    destination is written, so the result does not depend on the order of the
    specs: an exchange is simultaneous, not sequential. (The hand-written
    exchangers this replaces happen to agree, because none of them writes a
    field another row reads; making it a rule means a reordered table cannot
    quietly change a run.)

    Parameters
    ----------
    specs : Sequence
        The coupling table: :class:`ExchangeSpec` objects, or mappings with
        ``src``/``dst``/``regrid`` keys.
    regridders : Mapping[str, Callable], optional
        The regridders a spec's ``regrid`` may name. Each is called as
        ``regridder(value)`` and returns the field on the destination grid.
        A spec naming a regridder that is not here is a ``KeyError`` at
        construction -- the cheapest place to catch it.

    Raises
    ------
    KeyError
        If a spec names a regridder that ``regridders`` does not have.
    ValueError
        If a spec's path is malformed, or two specs write the same
        destination field (which would make the result depend on the table's
        order).

    """

    def __init__(
        self,
        specs: Sequence[ExchangeSpec | Mapping[str, Any]],
        regridders: Mapping[str, Callable[[Any], Any]] | None = None,
    ):
        """Build an exchanger from a coupling table; see the class docstring."""
        self.specs: tuple[ExchangeSpec, ...] = tuple(_as_spec(spec) for spec in specs)
        self.regridders: dict[str, Callable[[Any], Any]] = dict(regridders or {})

        written: dict[str, ExchangeSpec] = {}
        for spec in self.specs:
            self._regridder(spec)
            if spec.dst in written:
                raise ValueError(
                    f"Exchange spec {spec} writes {spec.dst!r}, which "
                    f"{written[spec.dst]} already writes; one destination field "
                    "can only have one source."
                )
            written[spec.dst] = spec

    def __repr__(self) -> str:
        """Return the coupling table, one spec per line."""
        if not self.specs:
            return f"{type(self).__name__}([])"
        rows = "\n".join(f"  {spec}" for spec in self.specs)
        return f"{type(self).__name__}(\n{rows}\n)"

    def __call__(
        self, components: dict[str, Carry], time: CouplingTime
    ) -> dict[str, Carry]:
        """Apply the coupling table and return the mapping to continue with.

        The clock is not used: a declarative exchange is a copy, and a
        coupling that depends on the date (a ramp, a seasonal blend) is a
        hand-written exchanger. It is accepted because that is the exchanger
        signature.

        This runs inside the traced coupled step, so its lookups fail at
        *trace* time -- before any compilation -- with the same messages
        :meth:`validate` gives. :meth:`validate` is still worth calling from
        a runner, because it fails before a model is built rather than after.

        Parameters
        ----------
        components : dict[str, Carry]
            Every component's carry, as the coupler hands it over.
        time : jem.base.component.CouplingTime
            The coupler's clock. Unused; see above.

        Returns
        -------
        dict[str, Carry]

        """
        del time
        # component -> section -> {field: value}, collected before anything
        # is written so that every source is read from the incoming carries.
        updates: dict[str, dict[str, dict[str, Any]]] = {}
        for spec in self.specs:
            component, section, field = spec.src_parts
            source = self._section(components, component, section, spec)
            self._require_field(source, field, spec, spec.src)
            value = getattr(source, field)
            regridder = self._regridder(spec)
            if regridder is not None:
                value = regridder(value)
            component, section, field = spec.dst_parts
            # Resolve the destination section too, so a typo there is caught
            # here rather than inside `.replace()`, which would raise a
            # TypeError naming neither the spec nor the component.
            destination = self._section(components, component, section, spec)
            self._require_field(destination, field, spec, spec.dst)
            updates.setdefault(component, {}).setdefault(section, {})[field] = value

        exchanged = dict(components)
        for name, sections in updates.items():
            carry = dict(exchanged[name])
            for section, fields in sections.items():
                carry[section] = carry[section].replace(**fields)
            exchanged[name] = carry
        return exchanged

    def validate(self, components: Mapping[str, Carry]) -> None:
        """Check every spec against real carries, and raise naming the bad one.

        The eager pre-flight: called with ``coupler.initialize()`` before a
        run is compiled, it turns a mistyped component, section, field or
        regridder into an error that names the spec, instead of a trace-time
        failure inside the coupled step (or, worse, a silently unused field).

        Parameters
        ----------
        components : Mapping[str, Carry]
            The initial carries, as ``Coupler.initialize()`` builds them
            (``CoupledCarry.components``).

        Raises
        ------
        KeyError
            If a spec names a component, a carry section or a regridder that
            does not exist.
        ValueError
            If a spec names a field its section does not have.
        TypeError
            If a component's carry is not a mapping, so it has no sections to
            address.

        """
        for spec in self.specs:
            self._regridder(spec)
            for path, (component, section, field) in (
                (spec.src, spec.src_parts),
                (spec.dst, spec.dst_parts),
            ):
                resolved = self._section(components, component, section, spec)
                self._require_field(resolved, field, spec, path)

    # -- lookups, shared by __call__ and validate --------------------------

    def _regridder(self, spec: ExchangeSpec) -> Callable[[Any], Any] | None:
        """Return the regridder ``spec`` names, or None if it names none."""
        if spec.regrid is None:
            return None
        try:
            return self.regridders[spec.regrid]
        except KeyError:
            raise KeyError(
                f"Exchange spec {spec} names the regridder {spec.regrid!r}, which "
                f"was not given to this Exchange (it has "
                f"{sorted(self.regridders)!r})."
            ) from None

    @staticmethod
    def _section(
        components: Mapping[str, Carry], name: str, section: str, spec: ExchangeSpec
    ) -> Any:
        """Return one section of one component's carry, or raise naming the spec."""
        if name not in components:
            raise KeyError(
                f"Exchange spec {spec} names the component {name!r}, which this "
                f"coupled model does not have (it has {sorted(components)!r})."
            )
        carry = components[name]
        if not isinstance(carry, Mapping):
            raise TypeError(
                f"Exchange spec {spec}: the carry of {name!r} is a "
                f"{type(carry).__name__}, not a mapping, so it has no {section!r} "
                "section to address. Couple this component with a hand-written "
                "exchanger."
            )
        if section not in carry:
            raise KeyError(
                f"Exchange spec {spec}: {name!r}'s carry has no {section!r} section "
                f"(it has {sorted(carry)!r})."
            )
        return carry[section]

    @staticmethod
    def _require_field(
        section: Any, field: str, spec: ExchangeSpec, path: str
    ) -> None:
        """Raise, naming the spec, if ``section`` does not have ``field``."""
        if not hasattr(section, field):
            raise ValueError(
                f"Exchange spec {spec}: {path!r} names the field {field!r}, which "
                f"{type(section).__name__} does not have (it has "
                f"{_field_names(section)!r})."
            )


def default_exchanges(
    components: Mapping[str, Component] | Iterable[str],
    regrid: Mapping[str, str] | None = None,
) -> list[ExchangeSpec]:
    """Return the standard coupling table, filtered to the components present.

    This is exactly the exchange every example in this repository writes out
    by hand, in one place (:data:`STANDARD_EXCHANGES`):

    ==========================================  =======================================
    source                                      destination
    ==========================================  =======================================
    ``atm.derived.total_heat_flux``             ``ocn.forcing.total_heat_flux``
    ``atm.derived.total_heat_flux``             ``lnd.forcing.total_heat_flux``
    ``ocn.derived.ice_frazil_melt_energy``      ``seaice.forcing.ice_frazil_melt_energy``
    ``ocn.state.sea_surface_temperature``       ``atm.forcing.sea_surface_temperature``
    ``seaice.derived.ice_fraction``             ``atm.forcing.sice_am``
    ``lnd.state.land_surface_temperature``      ``atm.forcing.stl_am``
    ``lnd.state.snowc``                         ``atm.forcing.snowc_am``
    ``lnd.state.soilw``                         ``atm.forcing.soilw_am``
    ==========================================  =======================================

    A row survives only if **both** its components are present, so an
    aquaplanet without a land model gets the four rows that do not mention
    ``lnd``, and an atmosphere/ocean pair gets two. The names are
    :data:`STANDARD_COMPONENT_NAMES`; a component registered under any other
    name is not wired by this function.

    Which table
    -----------
    The table above describes JAX-ESM's own slab components. A component with
    a carry of its own needs its own rows, and there is one such table so far:
    a :class:`~jem.components.veros_component.VerosComponent` registered as
    ``"ocn"`` selects :data:`VEROS_OCEAN_EXCHANGES` instead, which is the same
    wiring spelled in Veros' field names and with the freshwater flux Veros
    also takes. That constant documents what it cannot express -- above all
    the wind stress, which is a drag law rather than a copy. The choice is
    made by type, so it needs real components: called with just a list of
    *names*, this cannot tell one ocean from another and gives the slab table.

    Regridding
    ----------
    ``regrid`` names a regridder for the specs that cross the
    atmosphere/ocean-grid boundary -- the ones between ``atm`` and a
    component of :data:`OCEAN_GRID_COMPONENTS`. Its keys are
    ``"<direction>_<kind>"``, or a bare ``"<direction>"`` covering both
    kinds, where

    - *direction* is ``"a2o"`` (atmosphere to ocean grid) or ``"o2a"``, and
    - *kind* is the row's own ``"flux"``/``"state"`` (:data:`KINDS`).

    That split is the one the mixed-grid example makes by hand: extensive
    quantities -- heat fluxes, the freeze/melt energy, an areal ice fraction
    -- are regridded conservatively to keep their budgets, while an intensive
    state variable such as the sea surface temperature is interpolated
    bilinearly. So the mixed-grid configuration is::

        default_exchanges(components, regrid={
            "a2o_flux": "a2o_conserve",   # surface heat flux onto the ocean grid
            "o2a_flux": "o2a_conserve",   # ice fraction onto the atmosphere grid
            "o2a_state": "o2a_bilinear",  # SST onto the atmosphere grid
        })

    and a configuration with one regridder per direction is
    ``regrid={"a2o": "a2o", "o2a": "o2a"}``. Rows that stay on one grid
    (``ocn`` to ``seaice``, ``atm`` to ``lnd``) never get a regridder.

    Parameters
    ----------
    components : Mapping[str, Component] or iterable of str
        The coupled model's components, or just their names.
    regrid : Mapping[str, str], optional
        Regridder *names* per direction and kind, as above. The callables
        themselves are given to :class:`Exchange`; see
        :func:`default_exchangers` for the one-call form.

    Returns
    -------
    list[ExchangeSpec]

    """
    present = set(components)
    regrid = dict(regrid or {})
    unknown = set(regrid) - {
        f"{direction}{suffix}"
        for direction in ("a2o", "o2a")
        for suffix in ("", *(f"_{kind}" for kind in KINDS))
    }
    if unknown:
        raise ValueError(
            f"Unknown regrid key(s) {sorted(unknown)!r}: a key is a direction "
            "(\"a2o\", \"o2a\") optionally followed by \"_flux\" or \"_state\"."
        )

    table = _exchange_table(components)
    specs: list[ExchangeSpec] = []
    for src, dst, kind in table:
        src_component, _, _ = _parse_path(src, "source", src)
        dst_component, _, _ = _parse_path(dst, "destination", dst)
        if not {src_component, dst_component} <= present:
            continue
        direction = _grid_direction(src_component, dst_component)
        name = None
        if direction is not None:
            name = regrid.get(f"{direction}_{kind}", regrid.get(direction))
        specs.append(ExchangeSpec(src, dst, name))

    skipped = [name for name in STANDARD_COMPONENT_NAMES if name not in present]
    if skipped:
        logger.debug(
            "Default exchange: no component named %s, so the specs that mention "
            "them are left out.", ", ".join(repr(name) for name in skipped),
        )
    for name in sorted(present):
        standard = _NEAR_MISS_NAMES.get(name)
        if standard is not None and standard not in present:
            logger.warning(
                "The default coupling wires the %s component under the name %r, "
                "not %r, so %r is left unconnected. Register it as %r (for "
                "example SlabSeaiceModel(grid, name='seaice')) or write the "
                "exchange out by hand.",
                standard, standard, name, name, standard,
            )
    if table is VEROS_OCEAN_EXCHANGES and "seaice" in present:
        logger.warning(
            "A sea-ice component is coupled to a Veros ocean, which publishes "
            "no freeze/melt potential (`ice_frazil_melt_energy`), so the "
            "default coupling has no row that drives it: the ice will only "
            "respond to what it computes itself. Write the exchange out by "
            "hand (coupling.exchanger) if the ice is meant to grow on the "
            "ocean's heat budget."
        )
    return specs


def _exchange_table(
    components: Mapping[str, Component] | Iterable[str],
) -> tuple[tuple[str, str, str], ...]:
    """Return the standard table matching the ocean ``components`` holds.

    :data:`VEROS_OCEAN_EXCHANGES` when ``"ocn"`` is a
    :class:`~jem.components.veros_component.VerosComponent`, and
    :data:`STANDARD_EXCHANGES` otherwise -- including when the caller passed
    names rather than components, since a name says nothing about a carry.
    """
    if not isinstance(components, Mapping):
        return STANDARD_EXCHANGES
    ocean = components.get("ocn")
    # `sys.modules`, not an import: see VEROS_COMPONENT_MODULE.
    module = sys.modules.get(VEROS_COMPONENT_MODULE)
    if ocean is None or module is None:
        return STANDARD_EXCHANGES
    if not isinstance(ocean, module.VerosComponent):
        return STANDARD_EXCHANGES
    logger.debug(
        "The ocean is a Veros GCM, so the default coupling is the Veros table."
    )
    return VEROS_OCEAN_EXCHANGES


def _grid_direction(src_component: str, dst_component: str) -> str | None:
    """Return ``"a2o"``/``"o2a"`` if a spec crosses the grid boundary, else None."""
    if src_component == "atm" and dst_component in OCEAN_GRID_COMPONENTS:
        return "a2o"
    if src_component in OCEAN_GRID_COMPONENTS and dst_component == "atm":
        return "o2a"
    return None


def default_exchangers(
    components: Mapping[str, Component] | Iterable[str],
    regrid: Mapping[str, Callable[[Any], Any]] | None = None,
) -> dict[str, Exchanger]:
    """Return the standard coupling as a coupler's ``exchangers=`` argument.

    One exchanger, named :data:`DEFAULT_EXCHANGER_NAME`, running the table
    :func:`default_exchanges` builds::

        Coupler(components, default_exchangers(components), ...)

    Parameters
    ----------
    components : Mapping[str, Component] or iterable of str
        The coupled model's components, or just their names.
    regrid : Mapping[str, Callable], optional
        The regridders, keyed by direction and kind exactly as
        :func:`default_exchanges` documents -- ``{"a2o_flux": conservative,
        "o2a_state": bilinear, ...}``. Here the values are the callables
        themselves, and their keys double as the names the specs carry, so a
        mixed-grid model needs one mapping rather than a mapping and a naming
        convention::

            default_exchangers(components, regrid={
                "a2o_flux": ESMFRegridder(a2o_conservative_weights),
                "o2a_flux": ESMFRegridder(o2a_conservative_weights),
                "o2a_state": ESMFRegridder(o2a_bilinear_weights),
            })

    Returns
    -------
    dict[str, Exchanger]

    """
    regridders = dict(regrid or {})
    names = {key: key for key in regridders}
    return {DEFAULT_EXCHANGER_NAME: Exchange(
        default_exchanges(components, names), regridders
    )}


def default_workflow(
    components: Mapping[str, Component] | Iterable[str],
    exchangers: Mapping[str, Exchanger] | Iterable[str],
) -> list[str]:
    """Return the workflow a :class:`~jem.base.coupler.Coupler` runs by default.

    Every exchanger, then every component, in registration order -- so the
    fields are exchanged first and the components then step on the same
    state. This is the same order ``Coupler.workflow`` produces when no
    ``workflow=`` is given; it exists so that a configuration layer can write
    the default down, extend it, or diff it against a hand-written one
    without having to build a coupler first.

    The one-step coupling lag this order implies is described in the module
    docstring.

    Parameters
    ----------
    components : Mapping[str, Component] or iterable of str
        The components, or their names, in the order they are registered.
    exchangers : Mapping[str, Exchanger] or iterable of str
        The exchangers, or their names, in the order they are registered.

    Returns
    -------
    list[str]

    """
    return [*exchangers, *components]
