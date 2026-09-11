"""Tests for ``jem.exchangers`` -- the declarative form of an exchanger.

The components here are the three real slab surface models (ocean, land, sea
ice) on a 4x3 grid, coupled to a **stub atmosphere**. The stub exists because
the standard wiring is defined against the atmosphere JAX-ESM couples in
practice, which is JCM: the fields it receives are JCM's own boundary
conditions (``sice_am``, ``stl_am``, ``snowc_am``, ``soilw_am``,
``sea_surface_temperature``) and the flux it publishes is
``derived.total_heat_flux``. ``SlabAtmosphereModel`` is a toy with its own,
different names -- it has no ``sice_am`` and calls its flux
``internal_total_heat_flux`` -- so it cannot stand in for JCM here, and
building a real JCM model would make these tests a minute long apiece for no
extra coverage of the code under test. The stub is 20 lines and carries
exactly JCM's names, so a spec in :func:`jem.exchangers.default_exchanges`
that named a field JCM does not have would still fail here.
"""

import dataclasses
import logging
import sys
import types

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import tree_math

from jem import exchangers
from jem.base.coupler import Coupler
from jem.components.slab import (
    SlabGrid,
    SlabLandModel,
    SlabOceanModel,
    SlabSeaiceModel,
)
from jem.exchangers import (
    VEROS_OCEAN_EXCHANGES,
    Exchange,
    ExchangeSpec,
    default_exchangers,
    default_exchanges,
    default_workflow,
)
from tests.unit.slab_test_utils import make_grid

START_DATE = jdt.to_datetime("2001-01-01")
CALENDAR = "365_day"
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")

#: The standard wiring, written out here independently of the module under
#: test: if a spec is added, removed or re-pointed, this table has to be
#: edited too, deliberately.
EXPECTED_SPECS = (
    ("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux"),
    ("atm.derived.total_heat_flux", "lnd.forcing.total_heat_flux"),
    ("ocn.derived.ice_frazil_melt_energy", "seaice.forcing.ice_frazil_melt_energy"),
    ("ocn.state.sea_surface_temperature", "atm.forcing.sea_surface_temperature"),
    ("seaice.derived.ice_fraction", "atm.forcing.sice_am"),
    ("lnd.state.land_surface_temperature", "atm.forcing.stl_am"),
    ("lnd.state.snowc", "atm.forcing.snowc_am"),
    ("lnd.state.soilw", "atm.forcing.soilw_am"),
)


# ---------------------------------------------------------------------------
# The stub atmosphere
# ---------------------------------------------------------------------------


@tree_math.struct
class _AtmosphereState:
    air_temperature: jnp.ndarray


@tree_math.struct
class _AtmosphereDerived:
    total_heat_flux: jnp.ndarray
    # Published by `JCMDerived` too, and taken by a Veros ocean (a slab does
    # not read it), so the stub has to carry it or the Veros table could not
    # be checked against a JCM-shaped atmosphere at all.
    total_freshwater_flux: jnp.ndarray


@tree_math.struct
class _AtmosphereForcing:
    sea_surface_temperature: jnp.ndarray
    sice_am: jnp.ndarray
    stl_am: jnp.ndarray
    snowc_am: jnp.ndarray
    soilw_am: jnp.ndarray


class StubAtmosphere:
    """An atmosphere with JCM's exchange field names and no physics.

    The heat flux it publishes depends on every field it was given, so a spec
    that failed to deliver one would change the run rather than pass silently.
    """

    def __init__(self, shape, name="atm"):
        """Name the component and remember the grid shape it works on."""
        self.name = name
        self.shape = shape

    def initialize(self):
        zeros = jnp.zeros(self.shape)
        return {
            "state": _AtmosphereState(zeros + 280.0),
            "derived": _AtmosphereDerived(zeros, zeros),
            "forcing": _AtmosphereForcing(zeros, zeros, zeros, zeros, zeros),
        }

    def step(self, carry, time):
        del time
        forcing = carry["forcing"]
        state = carry["state"]
        surface_temperature = (
            forcing.sea_surface_temperature * (1.0 - forcing.sice_am)
            + forcing.stl_am * forcing.sice_am
        )
        flux = 10.0 * (surface_temperature - state.air_temperature)
        # The snow and soil fields enter weakly, so they too are load-bearing.
        flux = flux * (1.0 + 0.01 * forcing.snowc_am + 0.01 * forcing.soilw_am)
        new_state = _AtmosphereState(state.air_temperature + 1e-3 * flux)
        new_carry = {
            "state": new_state,
            "derived": _AtmosphereDerived(flux, 1e-5 * flux),
            "forcing": forcing,
        }
        return new_carry, {"air_temperature": new_state.air_temperature}


# ---------------------------------------------------------------------------
# The stub Veros ocean
# ---------------------------------------------------------------------------

#: The Veros rows of the standard wiring, written out independently of the
#: module under test, exactly as EXPECTED_SPECS is for the slab table.
EXPECTED_VEROS_SPECS = (
    ("atm.derived.total_heat_flux", "ocn.forcing.heat_flux"),
    ("atm.derived.total_freshwater_flux", "ocn.forcing.freshwater_flux"),
    ("ocn.derived.sea_surface_temperature", "atm.forcing.sea_surface_temperature"),
)


@tree_math.struct
class _VerosLikeState:
    """Stands in for Veros' own ``VerosState``: nothing here is exchangeable.

    The real object is Veros' state container, not a struct of surface fields,
    which is exactly why the Veros table reads the sea surface temperature
    from ``derived`` where the slab table reads it from ``state``.
    """

    temperature: jnp.ndarray


@tree_math.struct
class _VerosLikeDerived:
    sea_surface_temperature: jnp.ndarray
    sea_surface_u: jnp.ndarray
    sea_surface_v: jnp.ndarray


@tree_math.struct
class _VerosLikeForcing:
    heat_flux: jnp.ndarray
    freshwater_flux: jnp.ndarray
    surface_taux: jnp.ndarray
    surface_tauy: jnp.ndarray
    surface_air_temperature: jnp.ndarray


class StubVerosOcean:
    """An ocean with ``VerosComponent``'s carry layout and no ocean physics.

    Field for field and section for section as
    :class:`jem.components.veros_component.VerosComponent`
    (``VerosForcing``, ``VerosDerived``, and a ``state`` that holds no
    exchangeable surface field), so a row of
    :data:`~jem.exchangers.VEROS_OCEAN_EXCHANGES` that named something Veros
    does not have would fail here. Veros itself is an optional dependency and
    is never imported by these tests.
    """

    name = "ocn"

    def __init__(self, shape):
        """Remember the horizontal shape the carry is built on."""
        self.shape = shape

    def initialize(self):
        zeros = jnp.zeros(self.shape)
        return {
            "state": _VerosLikeState(zeros + 285.0),
            "derived": _VerosLikeDerived(zeros + 285.0, zeros, zeros),
            "forcing": _VerosLikeForcing(zeros, zeros, zeros, zeros, zeros),
        }

    def step(self, carry, time):
        del time
        forcing = carry["forcing"]
        # Upward-positive heat flux cools the ocean, as it does in Veros'
        # `forc_temp_surface`; the freshwater flux enters weakly so that a
        # row that failed to deliver it would change the trajectory.
        temperature = (
            carry["state"].temperature
            - 1e-3 * forcing.heat_flux
            - 1.0 * forcing.freshwater_flux
        )
        new_carry = {
            "state": _VerosLikeState(temperature),
            "derived": _VerosLikeDerived(
                temperature, carry["derived"].sea_surface_u,
                carry["derived"].sea_surface_v,
            ),
            "forcing": forcing,
        }
        return new_carry, {"sea_surface_temperature": temperature}


@pytest.fixture
def veros_module(monkeypatch):
    """Register :class:`StubVerosOcean` as the Veros wrapper for one test.

    ``jem.exchangers`` recognises a Veros ocean by looking its module up in
    ``sys.modules`` and testing ``isinstance`` against the class there --
    deliberately, so that a JAX-ESM without the optional Veros dependency
    never imports it. Veros is not installed in the unit-test job, so the
    module is stood in for here; the code path taken is the real one.
    """
    module = types.ModuleType(exchangers.VEROS_COMPONENT_MODULE)
    module.VerosComponent = StubVerosOcean
    monkeypatch.setitem(sys.modules, exchangers.VEROS_COMPONENT_MODULE, module)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def half_land_grid() -> SlabGrid:
    """Return the 4x3 test grid with its two eastern columns land."""
    fraction = np.zeros((4, 3))
    fraction[2:, :] = 1.0
    return make_grid(fractional_mask=fraction)


def build_components() -> dict:
    """Return the four components of the standard wiring, on the test grid."""
    grid = half_land_grid()
    return {
        "atm": StubAtmosphere(grid.fractional_mask.shape),
        "ocn": SlabOceanModel(grid),
        "lnd": SlabLandModel(grid),
        "seaice": SlabSeaiceModel(grid, name="seaice"),
    }


def build_coupler(components, exchangers) -> Coupler:
    """Return a coupler on the test clock for ``components``/``exchangers``."""
    return Coupler(
        components,
        exchangers,
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )


@pytest.fixture
def components():
    return build_components()


@pytest.fixture
def veros_components(veros_module):
    """Return a JCM-shaped atmosphere over a Veros-shaped ocean."""
    del veros_module
    shape = half_land_grid().fractional_mask.shape
    return {"atm": StubAtmosphere(shape), "ocn": StubVerosOcean(shape)}


# ---------------------------------------------------------------------------
# The default wiring
# ---------------------------------------------------------------------------


def test_default_exchanges_is_the_documented_table(components):
    """The default wiring is exactly the table the documentation prints."""
    specs = default_exchanges(components)
    assert [(spec.src, spec.dst) for spec in specs] == list(EXPECTED_SPECS)
    assert all(spec.regrid is None for spec in specs)


@pytest.mark.parametrize(
    "present, expected",
    [
        (("atm", "ocn"), 2),
        (("atm", "ocn", "seaice"), 4),
        (("atm", "ocn", "lnd"), 6),
        (("atm", "ocn", "lnd", "seaice"), 8),
        (("ocn", "lnd"), 0),
    ],
)
def test_default_exchanges_filters_to_present_components(present, expected):
    """A spec survives only if both of its components are in the model."""
    specs = default_exchanges(present)
    assert len(specs) == expected
    for spec in specs:
        assert spec.src_parts[0] in present
        assert spec.dst_parts[0] in present


def test_default_exchanges_ignores_components_it_has_no_wiring_for():
    """An unrecognised component name is simply not wired, not an error."""
    specs = default_exchanges(("atm", "ocn", "spring", "chemistry"))
    assert [(spec.src, spec.dst) for spec in specs] == [
        ("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux"),
        ("ocn.state.sea_surface_temperature", "atm.forcing.sea_surface_temperature"),
    ]


def test_default_workflow_matches_coupler_default(components):
    """The written-down default workflow must be the one ``Coupler`` runs."""
    exchangers = default_exchangers(components)
    coupler = build_coupler(components, exchangers)
    assert coupler.workflow == tuple(default_workflow(components, exchangers))
    assert list(default_workflow(components, exchangers)) == [
        "exchange", "atm", "ocn", "lnd", "seaice",
    ]


def test_default_exchangers_validates_against_the_real_carries(components):
    """Every default spec resolves against the carries the components build."""
    exchangers = default_exchangers(components)
    coupler = build_coupler(components, exchangers)
    exchangers["exchange"].validate(coupler.initialize().components)


# ---------------------------------------------------------------------------
# Exchange semantics
# ---------------------------------------------------------------------------


def test_exchange_roundtrip(components):
    """Every field moves, and nothing the exchanger was handed is touched."""
    exchange = default_exchangers(components)["exchange"]
    coupler = build_coupler(components, {"exchange": exchange})

    # One coupled step first, so the fields being moved are not all zero.
    carry, _ = coupler.generate_step_function()(coupler.initialize())
    incoming = dict(carry.components)
    leaves_before = jax.tree_util.tree_leaves(incoming)

    exchanged = exchange(incoming, coupler.coupling_time(carry.step))

    # Values moved: each destination now holds the very array the source did.
    assert (
        exchanged["ocn"]["forcing"].total_heat_flux
        is incoming["atm"]["derived"].total_heat_flux
    )
    assert (
        exchanged["lnd"]["forcing"].total_heat_flux
        is incoming["atm"]["derived"].total_heat_flux
    )
    assert (
        exchanged["seaice"]["forcing"].ice_frazil_melt_energy
        is incoming["ocn"]["derived"].ice_frazil_melt_energy
    )
    atmosphere_forcing = exchanged["atm"]["forcing"]
    assert (
        atmosphere_forcing.sea_surface_temperature
        is incoming["ocn"]["state"].sea_surface_temperature
    )
    assert atmosphere_forcing.sice_am is incoming["seaice"]["derived"].ice_fraction
    assert (
        atmosphere_forcing.stl_am
        is incoming["lnd"]["state"].land_surface_temperature
    )
    assert atmosphere_forcing.snowc_am is incoming["lnd"]["state"].snowc
    assert atmosphere_forcing.soilw_am is incoming["lnd"]["state"].soilw

    # Nothing was written in place: the mapping, the carries and every leaf
    # the exchanger was handed are exactly as they were.
    assert exchanged is not incoming
    assert set(exchanged) == set(incoming)
    after = jax.tree_util.tree_leaves(incoming)
    assert len(after) == len(leaves_before)
    assert all(new is old for new, old in zip(after, leaves_before))
    # And the state sections, which no spec writes, are the same objects.
    for name in incoming:
        assert exchanged[name]["state"] is incoming[name]["state"]


def test_exchange_runs_in_a_real_coupled_trajectory(components):
    """The declarative exchanger survives being traced, scanned and jitted."""
    coupler = build_coupler(components, default_exchangers(components))
    initial = coupler.initialize()
    final, diagnostics = coupler.generate_trajectory_function(3)(initial)

    assert jax.eval_shape(lambda: final) == jax.eval_shape(lambda: initial)
    assert int(final.step) == 3
    for name, component_diagnostics in diagnostics.items():
        for leaf in jax.tree_util.tree_leaves(component_diagnostics):
            assert jnp.shape(leaf)[0] == 3, name
            assert bool(jnp.all(jnp.isfinite(leaf))), name
    # The exchange actually ran: the atmosphere's flux left zero, which it
    # cannot do until it has been given a surface temperature.
    assert not bool(
        jnp.allclose(final.components["atm"]["derived"].total_heat_flux, 0.0)
    )


def readme_style_exchange(components, time):
    """Move the standard fields, as the examples write the exchange by hand.

    Assembled from the exchangers in ``README.md``,
    ``examples/01_basic/01_aquaplanet.ipynb`` and
    ``examples/02_experimental/01_earth.ipynb`` -- the same eight fields the
    declarative default moves, in the same direction.
    """
    del time
    atm = components["atm"]
    ocn = components["ocn"]
    lnd = components["lnd"]
    seaice = components["seaice"]

    ocn = dict(ocn, forcing=ocn["forcing"].replace(
        total_heat_flux=atm["derived"].total_heat_flux,
    ))
    lnd = dict(lnd, forcing=lnd["forcing"].replace(
        total_heat_flux=atm["derived"].total_heat_flux,
    ))
    seaice = dict(seaice, forcing=seaice["forcing"].replace(
        ice_frazil_melt_energy=components["ocn"]["derived"].ice_frazil_melt_energy,
    ))
    atm = dict(atm, forcing=atm["forcing"].replace(
        sea_surface_temperature=components["ocn"]["state"].sea_surface_temperature,
        sice_am=seaice["derived"].ice_fraction,
        stl_am=lnd["state"].land_surface_temperature,
        snowc_am=lnd["state"].snowc,
        soilw_am=lnd["state"].soilw,
    ))
    return dict(components, atm=atm, ocn=ocn, lnd=lnd, seaice=seaice)


def test_default_exchange_matches_the_readme_exchanger():
    """The declarative default must reproduce the hand-written exchange exactly.

    Not "to a tolerance": the exchange is a copy, so the two runs share every
    operation and must agree bit for bit. This is what makes replacing the
    examples' hand-written exchangers with ``default_exchangers`` a
    refactoring rather than a change of model.
    """
    declarative = build_components()
    handwritten = build_components()

    declarative_coupler = build_coupler(declarative, default_exchangers(declarative))
    handwritten_coupler = build_coupler(
        handwritten, {"exchange": readme_style_exchange}
    )
    assert declarative_coupler.workflow == handwritten_coupler.workflow

    declarative_final, _ = declarative_coupler.generate_trajectory_function(3)(
        declarative_coupler.initialize()
    )
    handwritten_final, _ = handwritten_coupler.generate_trajectory_function(3)(
        handwritten_coupler.initialize()
    )

    expected_leaves, expected_treedef = jax.tree_util.tree_flatten(handwritten_final)
    actual_leaves, actual_treedef = jax.tree_util.tree_flatten(declarative_final)
    assert actual_treedef == expected_treedef
    for actual, expected in zip(actual_leaves, expected_leaves):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_exchange_reads_every_source_before_writing_any_destination():
    """An exchange is simultaneous: reordering the table cannot change it."""
    carry = {
        "a": {"state": _AtmosphereState(jnp.float32(1.0))},
        "b": {"state": _AtmosphereState(jnp.float32(2.0))},
    }
    specs = [
        ExchangeSpec("a.state.air_temperature", "b.state.air_temperature"),
        ExchangeSpec("b.state.air_temperature", "a.state.air_temperature"),
    ]
    swapped = Exchange(specs)(carry, None)
    reversed_order = Exchange(list(reversed(specs)))(carry, None)

    assert float(swapped["a"]["state"].air_temperature) == 2.0
    assert float(swapped["b"]["state"].air_temperature) == 1.0
    assert float(reversed_order["a"]["state"].air_temperature) == 2.0
    assert float(reversed_order["b"]["state"].air_temperature) == 1.0


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------


def test_regrid_spec_applies_the_named_regridder(components):
    """A spec's ``regrid`` is looked up and applied to the value it moves."""
    calls = []

    def doubling(field):
        calls.append(field)
        return field * 2.0

    exchange = Exchange(
        [ExchangeSpec("ocn.state.sea_surface_temperature",
                      "atm.forcing.sea_surface_temperature",
                      regrid="o2a_state")],
        regridders={"o2a_state": doubling},
    )
    coupler = build_coupler(components, {"exchange": exchange})
    carries = coupler.initialize().components

    exchanged = exchange(dict(carries), coupler.coupling_time(0))

    assert len(calls) == 1
    np.testing.assert_allclose(
        np.asarray(exchanged["atm"]["forcing"].sea_surface_temperature),
        2.0 * np.asarray(carries["ocn"]["state"].sea_surface_temperature),
    )


def test_default_exchanges_places_regridders_by_direction_and_kind():
    """The mixed-grid configuration is expressible as three named regridders.

    Fluxes and the areal ice fraction are regridded conservatively and the
    sea surface temperature bilinearly -- the choice
    ``examples/01_basic/04_jcm_slabs_mixed_grid_aqua_planet.ipynb`` makes by
    hand -- and the rows that stay on one grid get no regridder at all.
    """
    specs = default_exchanges(
        ("atm", "ocn", "lnd", "seaice"),
        regrid={
            "a2o_flux": "a2o_conserve",
            "o2a_flux": "o2a_conserve",
            "o2a_state": "o2a_bilinear",
        },
    )
    assert {(spec.src, spec.dst): spec.regrid for spec in specs} == {
        ("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux"): "a2o_conserve",
        ("atm.derived.total_heat_flux", "lnd.forcing.total_heat_flux"): None,
        ("ocn.derived.ice_frazil_melt_energy",
         "seaice.forcing.ice_frazil_melt_energy"): None,
        ("ocn.state.sea_surface_temperature",
         "atm.forcing.sea_surface_temperature"): "o2a_bilinear",
        ("seaice.derived.ice_fraction", "atm.forcing.sice_am"): "o2a_conserve",
        ("lnd.state.land_surface_temperature", "atm.forcing.stl_am"): None,
        ("lnd.state.snowc", "atm.forcing.snowc_am"): None,
        ("lnd.state.soilw", "atm.forcing.soilw_am"): None,
    }


# ---------------------------------------------------------------------------
# A Veros ocean gets the Veros table
# ---------------------------------------------------------------------------


def test_default_exchanges_uses_the_veros_table_for_a_veros_ocean(veros_components):
    """A Veros ocean is coupled through its own rows, not the slab table's.

    A Veros carry has none of the slab ocean's exchange fields -- no
    ``forcing.total_heat_flux``, no ``state.sea_surface_temperature`` -- so
    the slab table applied to it is not a degraded coupling but a broken one,
    which is what this replaces.
    """
    specs = default_exchanges(veros_components)
    assert tuple((spec.src, spec.dst) for spec in specs) == EXPECTED_VEROS_SPECS


def test_the_veros_table_validates_against_a_veros_carry(veros_components):
    """Every Veros row names a field the components really have.

    `Exchange.validate` is the pre-flight a runner pays for before anything is
    compiled; on the slab table applied to a Veros ocean it is exactly what
    used to fail.
    """
    exchange = default_exchangers(veros_components)["exchange"]
    carries = {
        name: component.initialize()
        for name, component in veros_components.items()
    }
    exchange.validate(carries)


def test_the_veros_wiring_runs_a_coupled_trajectory(veros_components):
    """Two coupled steps: the flux reaches the ocean and the SST comes back.

    One step would not catch a carry whose structure the exchange changed.
    Two also show the documented one-step lag: the exchange runs first, so
    the sea surface temperature the atmosphere holds after step 2 is the one
    the ocean reached at the end of step 1.
    """
    coupler = build_coupler(
        veros_components, default_exchangers(veros_components)
    )
    after_one, _ = coupler.generate_trajectory_function(1)(coupler.initialize())
    after_two, _ = coupler.generate_trajectory_function(2)(coupler.initialize())

    ocean = after_two.components["ocn"]
    # The ocean received the atmosphere's fluxes ...
    assert not np.allclose(np.asarray(ocean["forcing"].heat_flux), 0.0)
    assert not np.allclose(np.asarray(ocean["forcing"].freshwater_flux), 0.0)
    # ... and it is being cooled by them (the flux is upward positive).
    assert float(jnp.min(ocean["derived"].sea_surface_temperature)) < 285.0
    # The atmosphere is driven by the ocean's own temperature, one step back.
    np.testing.assert_allclose(
        np.asarray(after_two.components["atm"]["forcing"].sea_surface_temperature),
        np.asarray(after_one.components["ocn"]["derived"].sea_surface_temperature),
    )
    # Nothing wrote the fields no row covers: Veros' wind stress needs a drag
    # law, which no copy can express, so it is still what `initialize` set.
    np.testing.assert_array_equal(
        np.asarray(ocean["forcing"].surface_taux), 0.0
    )


def test_the_veros_sst_is_regridded_as_a_state_though_it_comes_from_derived(
    veros_components,
):
    """The row says flux or state; the carry section it is read from does not.

    Veros publishes its sea surface temperature from ``derived`` and a slab
    from ``state``, and it is the same intensive field either way -- so a
    rule that read the kind off the section would map it conservatively here
    and leave a conservative map's staircase in a smooth field.
    """
    specs = default_exchanges(
        veros_components,
        regrid={"a2o_flux": "a2o_conserve", "o2a_state": "o2a_bilinear"},
    )
    assert {(spec.src, spec.dst): spec.regrid for spec in specs} == {
        ("atm.derived.total_heat_flux", "ocn.forcing.heat_flux"): "a2o_conserve",
        ("atm.derived.total_freshwater_flux",
         "ocn.forcing.freshwater_flux"): "a2o_conserve",
        ("ocn.derived.sea_surface_temperature",
         "atm.forcing.sea_surface_temperature"): "o2a_bilinear",
    }


def test_a_slab_ocean_still_gets_the_slab_table(components, veros_module):
    """Registering the Veros wrapper changes nothing for a slab ocean."""
    del veros_module
    specs = default_exchanges(components)
    assert tuple((spec.src, spec.dst) for spec in specs) == EXPECTED_SPECS


def test_names_alone_cannot_select_the_veros_table(veros_module):
    """Called with names rather than components, the slab table is the answer.

    A name says nothing about a carry, and silently guessing "ocn" means Veros
    would break every test and script that wires a slab model by name.
    """
    del veros_module
    specs = default_exchanges(("atm", "ocn"))
    assert [spec.dst for spec in specs] == [
        "ocn.forcing.total_heat_flux", "atm.forcing.sea_surface_temperature",
    ]


def test_sea_ice_with_a_veros_ocean_is_warned_about(veros_components, caplog):
    """Veros publishes no freeze/melt potential, so the ice is undriven.

    The table has no row that could drive it, and a silently unforced sea-ice
    model is exactly the failure this warning exists to prevent.
    """
    grid = half_land_grid()
    components = dict(
        veros_components, seaice=SlabSeaiceModel(grid, name="seaice")
    )
    with caplog.at_level(logging.WARNING, logger="jem.exchangers"):
        default_exchanges(components)
    assert "no freeze/melt potential" in caplog.text


def test_the_veros_table_is_the_documented_one():
    """The constant and the rows this module expects agree, deliberately."""
    assert tuple(
        (src, dst) for src, dst, _ in VEROS_OCEAN_EXCHANGES
        if src.startswith(("atm.", "ocn.")) and dst.startswith(("atm.", "ocn."))
    ) == EXPECTED_VEROS_SPECS


def test_a_bare_direction_key_covers_both_kinds():
    """``regrid={"o2a": ...}`` regrids every field coming off the ocean grid."""
    specs = default_exchanges(("atm", "ocn", "seaice"), regrid={"o2a": "o2a"})
    by_route = {(spec.src, spec.dst): spec.regrid for spec in specs}
    assert by_route[
        ("ocn.state.sea_surface_temperature", "atm.forcing.sea_surface_temperature")
    ] == "o2a"
    assert by_route[("seaice.derived.ice_fraction", "atm.forcing.sice_am")] == "o2a"
    assert by_route[
        ("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux")
    ] is None


def test_default_exchangers_names_its_regridders_after_their_keys():
    """``default_exchangers`` takes the callables and derives the spec names."""
    identity = lambda field: field  # noqa: E731 - a stand-in regridder
    exchangers = default_exchangers(("atm", "ocn"), regrid={"a2o_flux": identity})
    exchange = exchangers["exchange"]
    assert exchange.regridders == {"a2o_flux": identity}
    assert [spec.regrid for spec in exchange.specs] == ["a2o_flux", None]


def test_unknown_regrid_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown regrid key"):
        default_exchanges(("atm", "ocn"), regrid={"atmosphere_to_ocean": "x"})


# ---------------------------------------------------------------------------
# Errors, named by spec
# ---------------------------------------------------------------------------


def test_exchange_unknown_field_raises_at_validate(components):
    """A field no section has is named, with the spec, before the run starts."""
    exchange = Exchange(
        [ExchangeSpec("ocn.state.surface_temperature",  # it is sea_surface_temperature
                      "atm.forcing.sea_surface_temperature")]
    )
    coupler = build_coupler(components, {"exchange": exchange})
    carries = coupler.initialize().components

    with pytest.raises(ValueError, match="surface_temperature"):
        exchange.validate(carries)
    # And the same failure at trace time, so a runner that skipped the
    # pre-flight still gets a message that names the spec.
    with pytest.raises(ValueError, match="ocn.state.surface_temperature"):
        exchange(dict(carries), coupler.coupling_time(0))


def test_exchange_unknown_destination_field_raises(components):
    exchange = Exchange(
        [ExchangeSpec("ocn.state.sea_surface_temperature", "atm.forcing.sst")]
    )
    coupler = build_coupler(components, {"exchange": exchange})
    carries = coupler.initialize().components
    with pytest.raises(ValueError, match="atm.forcing.sst"):
        exchange.validate(carries)
    with pytest.raises(ValueError, match="atm.forcing.sst"):
        exchange(dict(carries), coupler.coupling_time(0))


def test_exchange_unknown_component_raises(components):
    exchange = Exchange(
        [ExchangeSpec("sea.state.sea_surface_temperature",
                      "atm.forcing.sea_surface_temperature")]
    )
    coupler = build_coupler(components, {"exchange": exchange})
    with pytest.raises(KeyError, match="sea"):
        exchange.validate(coupler.initialize().components)


def test_exchange_unknown_section_raises(components):
    """A section the carry does not have is an error naming the sections it has."""
    exchange = Exchange(
        [ExchangeSpec("ocn.state.sea_surface_temperature",
                      "atm.forcing.sea_surface_temperature")]
    )
    carries = {"atm": {"forcing": _AtmosphereForcing(*([jnp.zeros(3)] * 5))},
               "ocn": {"derived": None}}
    with pytest.raises(KeyError, match="state"):
        exchange.validate(carries)


def test_exchange_on_a_carry_that_is_not_a_mapping_raises():
    exchange = Exchange([ExchangeSpec("a.state.x", "b.state.x")])
    with pytest.raises(TypeError, match="not a mapping"):
        exchange.validate({"a": 1.0, "b": {"state": None}})


def test_unknown_regridder_is_rejected_at_construction():
    with pytest.raises(KeyError, match="bilinear"):
        Exchange([ExchangeSpec("a.state.x", "b.state.x", regrid="bilinear")])


def test_two_specs_cannot_write_one_destination():
    with pytest.raises(ValueError, match="already writes"):
        Exchange([
            ExchangeSpec("a.state.x", "c.forcing.y"),
            ExchangeSpec("b.state.x", "c.forcing.y"),
        ])


@pytest.mark.parametrize(
    "path", ["atm.derived", "atm.derived.flux.extra", "atm..flux", ""]
)
def test_a_malformed_path_is_rejected_at_construction(path):
    with pytest.raises(ValueError, match="component.section.field"):
        ExchangeSpec(path, "ocn.forcing.total_heat_flux")


def test_an_unknown_section_name_is_rejected_at_construction():
    with pytest.raises(ValueError, match="carry section"):
        ExchangeSpec("atm.diagnostics.flux", "ocn.forcing.total_heat_flux")


# ---------------------------------------------------------------------------
# Construction and repr
# ---------------------------------------------------------------------------


def test_specs_may_be_written_as_mappings():
    """A coupling table can come straight out of YAML."""
    exchange = Exchange([
        {"src": "atm.derived.total_heat_flux", "dst": "ocn.forcing.total_heat_flux"},
        ExchangeSpec("ocn.state.sea_surface_temperature",
                     "atm.forcing.sea_surface_temperature"),
    ])
    assert exchange.specs == (
        ExchangeSpec("atm.derived.total_heat_flux", "ocn.forcing.total_heat_flux"),
        ExchangeSpec("ocn.state.sea_surface_temperature",
                     "atm.forcing.sea_surface_temperature"),
    )


def test_a_mapping_with_an_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        Exchange([{"src": "a.state.x", "dst": "b.state.x", "method": "conserve"}])


def test_exchange_spec_is_frozen_and_comparable():
    spec = ExchangeSpec("a.state.x", "b.forcing.y")
    assert dataclasses.is_dataclass(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.src = "c.state.x"
    assert spec == ExchangeSpec("a.state.x", "b.forcing.y")
    assert spec.src_parts == ("a", "state", "x")
    assert spec.dst_parts == ("b", "forcing", "y")


def test_repr_lists_the_specs():
    exchange = Exchange(
        [ExchangeSpec("a.state.x", "b.forcing.y", regrid="r")],
        regridders={"r": lambda field: field},
    )
    text = repr(exchange)
    assert "a.state.x -> b.forcing.y" in text
    assert "[r]" in text
    assert repr(Exchange([])) == "Exchange([])"


def test_a_near_miss_component_name_is_warned_about(caplog):
    """A sea-ice model left under its constructor default must not go quiet.

    ``SlabSeaiceModel``'s own default name is ``"ice"``, but the standard
    wiring is written for ``"seaice"``. Wiring ``"ice"`` anyway would be
    guesswork; saying nothing would leave the component silently uncoupled.
    """
    with caplog.at_level("WARNING", logger="jem.exchangers"):
        specs = default_exchanges(("atm", "ocn", "ice"))
    assert all("ice" not in spec.dst.split(".")[0] for spec in specs)
    assert "'seaice'" in caplog.text
    assert "'ice'" in caplog.text


def test_no_warning_when_both_names_are_present(caplog):
    """A model that genuinely has an extra "ice" component is not nagged."""
    with caplog.at_level("WARNING", logger="jem.exchangers"):
        default_exchanges(("atm", "ocn", "seaice", "ice"))
    assert caplog.text == ""
