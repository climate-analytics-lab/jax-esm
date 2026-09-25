"""Tests for :mod:`jem.runners` -- turning a composed config into objects.

The expensive checks here build a real T31 atmosphere, because that is the
only way to find out whether a composed option actually reached the object it
configures. The cheap ones -- the exchange wiring, the two spellings of a
coupling -- are done on a mapping of component *names*, since that is all the
default wiring reads.

The property that keeps this module honest is
``test_runners_has_no_component_kwargs``: the runner must stay generic, so a
new component is configured by adding a group file and never by adding a
branch to the runner.
"""

import dataclasses
import inspect
import logging

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

# Importing the config package registers the ${jcm_data:}/${jem_data:}
# resolvers the configurations use.
import jem.config  # noqa: F401
from jem import runners
from jem.base.component import Carry, CouplingTime
from jem.components.slab import (
    SlabAtmosphereParameters,
    SlabLandParameters,
    SlabOceanParameters,
    SlabSeaiceParameters,
)
from jem.exchangers import Exchange

CONFIG_MODULE = "jem.config"


def composed(overrides: list[str]):
    """Compose the primary config with ``overrides``."""
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def example_exchanger(components: dict[str, Carry], time: CouplingTime):
    """Return the carries unchanged; a config points at this by name."""
    del time
    return components


class RecordingExchanger:
    """An exchanger built from a config node, recording the `regrid=` it got.

    Stands in for a hand-written exchanger class (like
    `jem.fluxes.VerosExchange`) that needs the regridders `build_exchangers`
    injects -- something a bare dotted-path function cannot be given.
    """

    def __init__(self, regrid=None):
        """Record the `regrid` mapping the runner injected."""
        self.regrid = regrid

    def __call__(self, components: dict[str, Carry], time: CouplingTime):
        """Return the carries unchanged; only `__init__`'s recording matters."""
        del time
        return components


class SingleGridExchanger:
    """An exchanger class that takes no `regrid` argument at all.

    Stands in for a hand-written exchanger on a single shared grid, which
    `regrid=same_grid` composes to an empty mapping for -- such a class is
    not obliged to declare a `regrid` parameter at all, and `build_exchangers`
    must not inject one it does not accept.
    """

    def __init__(self, drag_coefficient: float = 1e-3):
        """Record the one keyword this class actually declares."""
        self.drag_coefficient = drag_coefficient

    def __call__(self, components: dict[str, Carry], time: CouplingTime):
        """Return the carries unchanged; only `__init__`'s recording matters."""
        del time
        return components


class TakesAGrid:
    """A component built *on* a grid, like every slab model. See below."""

    def __init__(self, grid, name="stub"):
        """Record the grid the runner injected."""
        self.grid = grid
        self.name = name


class BringsItsOwnGrid:
    """A component with a grid of its own, like the Veros ocean wrapper.

    ``from_setup`` mirrors :meth:`jem.components.VerosComponent.from_setup`
    in the one respect that matters here: it forwards every keyword it does
    not recognise to a setup factory, so a ``grid=`` the runner injected
    would not be quietly ignored -- it would reach the factory and be
    rejected there, which is the bug this stub pins.
    """

    def __init__(self, setup, **setup_kwargs):
        """Keep whatever the config node carried."""
        self.setup = setup
        self.setup_kwargs = setup_kwargs

    @classmethod
    def from_setup(cls, setup, **setup_kwargs):
        """Build from an importable setup path, as the Veros wrapper does."""
        if "grid" in setup_kwargs:
            raise TypeError(
                "generateVerosSetup() got an unexpected keyword argument 'grid'"
            )
        return cls(setup, **setup_kwargs)


@pytest.fixture
def restore_constants():
    """Undo a process-global ``jcm.constants`` override made by a test."""
    import jcm.constants

    saved = jcm.constants.physical_constants
    yield
    jcm.constants.set_constants(saved)


# ---------------------------------------------------------------------------
# The runner stays generic
# ---------------------------------------------------------------------------


def test_runners_has_no_component_kwargs():
    """No component parameter may be named in the runner, in any form.

    The runner injects the objects a config cannot name -- a grid, the
    regridders, the coupling timestep -- and instantiates everything else from
    the config node as it stands. The moment it mentions a *parameter* of a
    component, that component's configuration has two homes: the group file
    and a branch here, which is exactly the drift the config layer exists to
    prevent.
    """
    source = inspect.getsource(runners)
    parameter_fields = {
        field.name
        for parameters in (
            SlabOceanParameters,
            SlabLandParameters,
            SlabSeaiceParameters,
            SlabAtmosphereParameters,
        )
        for field in dataclasses.fields(parameters)
    }
    mentioned = sorted(name for name in parameter_fields if name in source)
    assert not mentioned, (
        f"jem/runners.py names the component parameter(s) {mentioned}. Configure "
        "them in the component's config group instead; the runner must not know "
        "what a component's parameters are."
    )


# ---------------------------------------------------------------------------
# Building the coupled model
# ---------------------------------------------------------------------------


def test_build_coupler_default():
    """The default config builds the model its group defaults describe."""
    coupler = runners.build_coupler(composed([]))

    # ocean=slab and seaice=slab are built; land=none is not, and its absence
    # is a missing component rather than a None in the mapping.
    assert set(coupler.components) == {"atm", "ocn", "seaice"}
    assert coupler.workflow == ("exchange", "atm", "ocn", "seaice")
    assert coupler.coupling_timestep == jdt.to_timedelta(1, "day")
    assert coupler.calendar == coupler.components["atm"].model.calendar
    assert coupler.start_date == coupler.components["atm"].model.start_date
    # The slabs were built on the atmosphere's grid, which is what
    # `regrid=same_grid` (no regridders) assumes.
    horizontal = coupler.components["atm"].model.coords.horizontal
    assert coupler.components["ocn"].grid.shape == (
        len(horizontal.nodal_axes[0]), len(horizontal.nodal_axes[1])
    )


def test_cli_and_python_construction_agree():
    """The README's Python construction and the equivalent config build one model.

    Two doors onto the same coupled model. If they drift, the documentation
    describes a model nobody runs from the command line, or the other way
    round -- so this compares what a coupled run is actually made of: the
    components and their types, the order they run in, the clock, and the
    structure of the carry the whole thing scans.
    """
    import jcm
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    from jem import Coupler, default_exchangers
    from jem.components import JCMComponent, SlabOceanModel
    from jem.components.slab import SlabGrid

    start_date = jdt.to_datetime("2000-01-01")
    model = jcm.model.Model(coords=get_speedy_coords(), start_date=start_date)
    atm = JCMComponent(model)
    components = {
        "atm": atm,
        "ocn": SlabOceanModel(SlabGrid.from_coords(model.coords.horizontal)),
    }
    by_hand = Coupler(
        components,
        default_exchangers(components),
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=start_date,
    )

    # The same model as a config: an aquaplanet slab ocean, no sea ice, no land.
    from_config = runners.build_coupler(composed(["+configuration=aquaplanet-slab",
                                                  "seaice=none"]))

    assert list(from_config.components) == list(by_hand.components)
    assert {name: type(component) for name, component in from_config.components.items()} == \
        {name: type(component) for name, component in by_hand.components.items()}
    assert from_config.workflow == by_hand.workflow
    assert from_config.coupling_timestep == by_hand.coupling_timestep
    assert from_config.start_date == by_hand.start_date
    assert from_config.calendar == by_hand.calendar
    assert jax.tree_util.tree_structure(from_config.initialize()) == \
        jax.tree_util.tree_structure(by_hand.initialize())


def test_mixed_grid_configuration_builds():
    """The mixed-grid configuration builds its own ocean grid and its regridders.

    Compose and build only: this is the configuration that proves the runner
    can put the surface components on a grid of their own, which is a wiring
    question, not a question about the trajectory.
    """
    cfg = composed(["+configuration=aquaplanet-slab-mixed-grid"])
    coupler = runners.build_coupler(cfg)

    atmosphere_shape = tuple(coupler.components["atm"].model.coords.nodal_shape[1:])
    ocean_grid = coupler.components["ocn"].grid
    assert ocean_grid.shape != atmosphere_shape
    # Both surface components are on the one ocean grid.
    assert coupler.components["seaice"].grid.shape == ocean_grid.shape
    # The land fraction came from the packaged mask file, not from the SCRIP
    # file's own all-ocean integer mask.
    assert float(ocean_grid.fractional_mask.max()) > 0.0

    exchange = coupler.exchangers["exchange"]
    crossing = {spec.regrid for spec in exchange.specs if spec.regrid is not None}
    assert crossing == {"a2o_flux", "o2a_flux", "o2a_state"}
    assert all(name in exchange.regridders for name in crossing)


def test_constants_override_reaches_the_build(restore_constants):
    """`+atmosphere.constants.<name>=` is applied before anything is built."""
    import jcm.constants

    assert jcm.constants.grav != 9.7
    runners.build_atmosphere(composed(["+atmosphere.constants.grav=9.7"]))
    assert jcm.constants.grav == pytest.approx(9.7)


# ---------------------------------------------------------------------------
# A grid is injected only into a component that takes one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "accepts"),
    [
        ("tests.unit.test_runners.TakesAGrid", True),
        ("tests.unit.test_runners.BringsItsOwnGrid", False),
        ("tests.unit.test_runners.BringsItsOwnGrid.from_setup", False),
        # Not resolvable at all: no grid, and `instantiate` reports the real
        # import error rather than this deciding what went wrong.
        ("tests.unit.test_runners.NoSuchComponent", False),
    ],
)
def test_accepts_grid_reads_the_target_signature(target, accepts):
    """Whether a grid is injected is a question about the target, not a name list.

    Both spellings of a `_target_` have to be resolved -- a class
    (`jem.components.SlabOceanModel`) and a classmethod
    (`jem.components.VerosComponent.from_setup`) -- and a `**kwargs`
    catch-all must not count as taking a grid: that is exactly the signature
    that swallows the keyword and fails somewhere else.
    """
    node = OmegaConf.create({"_target_": target})
    assert runners._accepts_grid(node) is accepts


@pytest.mark.parametrize("key", ["grid_file", "land_fraction_file"])
def test_a_grid_key_on_a_component_that_takes_no_grid_is_refused(key):
    """A grid described for a component that cannot use one is an error.

    `build_component` strips the runner-only keys before instantiating, so
    `ocean.grid_file=...` on a Veros node would otherwise be read by nothing
    at all: the run would go ahead on the ocean's own bathymetry while the
    user believes they replaced it.
    """
    node = OmegaConf.create(
        {"_target_": "tests.unit.test_runners.BringsItsOwnGrid.from_setup",
         "setup": "some_case.generateVerosSetup", key: "somewhere.nc"}
    )
    with pytest.raises(ValueError, match=key):
        runners._injected_grid(node, atm=None)


def test_a_broken_target_lookup_is_not_read_as_taking_no_grid(monkeypatch):
    """A lookup that is itself broken must raise, not answer "no grid".

    `_accepts_grid` swallows the errors a *lookup* raises, because an
    unresolvable target is `instantiate`'s to report. It must not swallow
    anything else: `hydra.utils.get_object` needs hydra-core >= 1.3, and on an
    older Hydra the resulting `AttributeError` would otherwise be read as
    "this component takes no grid" -- for EVERY component, slabs included,
    leaving the run to die inside `instantiate` with a message naming neither
    Hydra nor the grid.
    """
    def broken(path):
        raise AttributeError("module 'hydra.utils' has no attribute 'get_object'")

    monkeypatch.setattr(runners.hydra.utils, "get_object", broken)
    node = OmegaConf.create({"_target_": "tests.unit.test_runners.TakesAGrid"})
    with pytest.raises(AttributeError, match="get_object"):
        runners._accepts_grid(node)


def test_a_component_that_brings_its_own_grid_gets_none_built():
    """No grid is even built for a component that does not take one.

    `atm=None` is the point: `build_grid` would immediately fail on it, so
    this passing proves the grid is not merely built and dropped. A Veros
    ocean has its own bathymetry and land-sea mask, and a `SlabGrid` made
    from the atmosphere's geometry would describe a grid nothing runs on.
    """
    veros_like = OmegaConf.create(
        {"_target_": "tests.unit.test_runners.BringsItsOwnGrid.from_setup",
         "setup": "some_case.generateVerosSetup"}
    )
    assert runners._injected_grid(veros_like, atm=None) == {}

    built = runners.build_component(veros_like)
    assert built.setup == "some_case.generateVerosSetup"
    assert built.setup_kwargs == {}


def test_the_shipped_ocean_options_ask_for_what_they_take():
    """`ocean=veros` composes a node that takes no grid; the slabs take one.

    Composed from the shipped group files, so this fails if a config ever
    names a target whose signature disagrees with how the runner builds it.

    Resolving `jem.components.VerosComponent.from_setup` DOES import Veros --
    `jem.components.__getattr__` imports the wrapper module, which imports
    veros at module scope -- so without Veros installed `_accepts_grid` would
    answer False because the lookup failed, not because the signature says so,
    and the test would pass without checking anything. The signature is
    therefore read directly, and the assertion skipped where it cannot be.
    """
    for option in ("slab", "slab_relax", "slab_qflux"):
        assert runners._accepts_grid(composed([f"ocean={option}"]).ocean), option

    veros_node = composed(["ocean=veros"]).ocean
    pytest.importorskip("veros", reason="`ocean=veros`'s target cannot be resolved")
    resolved = runners.hydra.utils.get_object(str(veros_node._target_))
    assert runners.GRID_KEYWORD not in inspect.signature(resolved).parameters
    assert not runners._accepts_grid(veros_node)


# ---------------------------------------------------------------------------
# The exchange
# ---------------------------------------------------------------------------


def test_missing_component_filters_exchanges():
    """A component the config leaves out takes its exchange rows with it."""
    cfg = composed(["land=none"])
    assert cfg.land is None

    exchangers = runners.build_exchangers(
        cfg, {"atm": None, "ocn": None, "seaice": None}, {}
    )
    specs = exchangers["exchange"].specs
    assert specs, "the remaining components still have rows to exchange"
    assert not [spec for spec in specs if "lnd" in (spec.src + spec.dst)]

    with_land = runners.build_exchangers(
        composed([]), {"atm": None, "ocn": None, "lnd": None}, {}
    )
    assert [spec for spec in with_land["exchange"].specs
            if "lnd" in (spec.src + spec.dst)]


def test_exchanger_path_replaces_the_list():
    """`coupling.exchanger` names one function, used instead of the table."""
    cfg = composed(["coupling.exchanger=tests.unit.test_runners.example_exchanger"])
    exchangers = runners.build_exchangers(cfg, {"atm": None, "ocn": None}, {})
    assert exchangers == {"exchange": example_exchanger}


def test_exchanger_dotted_path_still_works():
    """Widening `coupling.exchanger` to accept a node leaves the dotted-path
    spelling exactly as `test_exchanger_path_replaces_the_list` above checks:
    a bare string still resolves through `hydra.utils.get_method`, not
    `instantiate`.
    """
    cfg = composed(["coupling.exchanger=tests.unit.test_runners.example_exchanger"])
    exchangers = runners.build_exchangers(cfg, {"atm": None, "ocn": None}, {})
    assert exchangers["exchange"] is example_exchanger


def test_exchanger_node_is_instantiated_with_the_regridders():
    """A `coupling.exchanger` mapping (a `_target_` node) is built, not just
    resolved, with the run's regridders injected as `regrid=`.

    `coupling.exchanger` defaults to `null`, so a CLI override cannot set a
    key *under* it (`coupling.exchanger._target_=...`) without `+` -- that
    spelling is refused with "Could not override ... To append to your
    config use +coupling.exchanger._target_=...", which is the override this
    test uses.
    """
    cfg = composed([
        "+coupling.exchanger._target_=tests.unit.test_runners.RecordingExchanger",
    ])
    regridders = {"a2o_flux": example_exchanger, "o2a_state": example_exchanger}
    exchangers = runners.build_exchangers(cfg, {"atm": None, "ocn": None}, regridders)
    built = exchangers["exchange"]
    assert isinstance(built, RecordingExchanger)
    assert built.regrid == regridders


def test_exchanger_node_without_regrid_parameter_is_not_given_one():
    """A single-grid exchanger class that takes no `regrid=` is a valid node.

    `regrid` is injected only when the target's own signature declares it
    (the same rule `_accepts_grid` applies to a component's `grid`), so a
    class like `SingleGridExchanger` -- which does not accept `regrid` at
    all -- is instantiated with none, instead of `instantiate` failing on an
    unexpected keyword argument.
    """
    cfg = composed([
        "+coupling.exchanger._target_=tests.unit.test_runners.SingleGridExchanger",
    ])
    regridders = {"a2o_flux": example_exchanger, "o2a_state": example_exchanger}
    exchangers = runners.build_exchangers(cfg, {"atm": None, "ocn": None}, regridders)
    built = exchangers["exchange"]
    assert isinstance(built, SingleGridExchanger)
    assert not hasattr(built, "regrid")


def test_exchanger_node_without_target_names_the_key():
    """A `coupling.exchanger` mapping with no `_target_` names no exchanger.

    Without this check, `hydra.utils.instantiate` would silently return the
    mapping as a plain `dict` -- a non-callable that only fails once the
    coupled step is traced, far from this call and naming nothing about the
    cause.
    """
    cfg = composed(["+coupling.exchanger.regrid=null"])
    with pytest.raises(ValueError, match="_target_"):
        runners.build_exchangers(cfg, {"atm": None, "ocn": None}, {})


def test_exchanger_and_exchangers_together_are_an_error():
    """Setting both leaves it undecided which couples the run, so it is refused."""
    cfg = composed([
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
        "+coupling.exchangers=[{src: 'ocn.state.sea_surface_temperature',"
        " dst: 'atm.forcing.sea_surface_temperature'}]",
    ])
    with pytest.raises(ValueError, match="are both") as excinfo:
        runners.build_exchangers(cfg, {"atm": None, "ocn": None}, {})
    # Names both spellings rather than dumping the whole DictConfig inline.
    assert "a dotted path, or a mapping" in str(excinfo.value)


def test_an_explicit_coupling_table_is_used_as_given():
    """A table written in YAML builds the exchange it describes, and only that."""
    cfg = composed([
        "+coupling.exchangers=[{src: 'ocn.state.sea_surface_temperature',"
        " dst: 'atm.forcing.sea_surface_temperature'}]",
    ])
    exchangers = runners.build_exchangers(cfg, {"atm": None, "ocn": None}, {})
    (spec,) = exchangers["exchange"].specs
    assert spec.src == "ocn.state.sea_surface_temperature"
    assert spec.regrid is None


# ---------------------------------------------------------------------------
# Which of the atmosphere's boundary conditions the coupling supplies
# ---------------------------------------------------------------------------


def test_earth_slab_couples_its_file_forcing(caplog):
    """The one shipped configuration with `forcing=from_file` builds and scans.

    `earth-slab` gives the atmosphere jax-gcm's packaged T30 climatology, in
    which every surface boundary condition is a time-varying `TimeSeries`,
    and then has the surface components overwrite five of them with plain
    arrays every step. The runner has to tell the atmosphere which five, or
    the carry changes pytree structure at the first exchange and no step can
    be scanned at all. Traced with `jax.eval_shape`: the structure check is a
    trace-time check, and tracing it costs no compilation.

    `earth-slab` does not set `coupling.exchanged_forcing`, so this also
    exercises the derived branch of `declare_exchanged_forcing` -- the one a
    user debugging a frozen climatology actually hits -- and its INFO log is
    the only place the derived set is visible without reading the table.
    """
    from jcm.forcing import TimeSeries

    with caplog.at_level(logging.INFO, logger="jem.runners"):
        coupler = runners.build_coupler(composed(["+configuration=earth-slab"]))

    # Read off the coupling table, not assumed: these are exactly the rows of
    # `jem.exchangers.STANDARD_EXCHANGES` that write into `atm.forcing`.
    assert set(coupler.components["atm"].exchanged_forcing) == {
        "sea_surface_temperature", "sice_am", "stl_am", "snowc_am", "soilw_am",
    }
    assert "derived from the exchanger table" in caplog.text
    for name in coupler.components["atm"].exchanged_forcing:
        assert name in caplog.text

    carry = coupler.initialize()
    forcing = carry.components["atm"]["forcing"]
    for name in coupler.components["atm"].exchanged_forcing:
        assert not isinstance(getattr(forcing, name), TimeSeries), name

    final, _ = jax.eval_shape(coupler.generate_trajectory_function(1), carry)
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)


def test_an_unexchanged_climatology_stays_a_climatology():
    """With no land model nothing supplies the land surface, so it keeps varying.

    The default configuration (`land=none`) with the same file forcing. This
    is why the set is derived from the coupling table rather than fixed:
    assuming the standard five would freeze this file's land climatology at
    the start date in every run built without a land model, which is most of
    them.
    """
    from jcm.forcing import TimeSeries

    coupler = runners.build_coupler(composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
    ]))

    assert set(coupler.components["atm"].exchanged_forcing) == {
        "sea_surface_temperature", "sice_am",
    }
    forcing = coupler.initialize().components["atm"]["forcing"]
    for name in ("stl_am", "snowc_am", "soilw_am"):
        assert isinstance(getattr(forcing, name), TimeSeries), name


def test_a_workflow_without_the_exchanger_leaves_forcing_unfrozen():
    """`coupling.workflow` may omit the exchanger, to run every component
    side by side with no coupling at all -- the supported way to compare
    against a coupled run (see the module docstring of `jem.exchangers`).

    Deriving `declare_exchanged_forcing`'s field set from every *registered*
    exchanger, rather than only the ones `workflow` actually runs, collapsed
    `sea_surface_temperature`/`sice_am` to their start-date value even though
    the "exchange" step that would write them never runs at all -- the
    climatology-frozen-with-no-symptom failure this function exists to avoid,
    just triggered by the workflow omitting the exchanger rather than a wrong
    field list. `land=none` (the default) keeps this to the two fields the
    default coupling table would otherwise still exchange.

    Fixing that alone would move the bug rather than remove it:
    `_validate_exchangers` would then call the never-run "exchange"
    exchanger's `.validate()` against the initial carry, in which the
    atmosphere's forcing fields are correctly still `TimeSeries` (nothing
    wrote them) while the surface components' fields are plain arrays --
    a structure mismatch that can only happen because of an exchange that
    never executes. So `_validate_exchangers` also has to skip an exchanger
    the resolved workflow does not run, which this test exercises by simply
    calling `build_coupler` at all: it would raise before returning if that
    skip were missing.
    """
    from jcm.forcing import TimeSeries

    coupler = runners.build_coupler(composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
        "coupling.workflow=[atm,ocn,seaice]",
    ]))

    # Nothing is declared: the only exchanger this coupler owns is
    # "exchange", and the workflow this test composed never runs it.
    assert coupler.components["atm"].exchanged_forcing == ()
    assert "exchange" not in coupler.workflow

    forcing = coupler.initialize().components["atm"]["forcing"]
    for name in ("sea_surface_temperature", "sice_am"):
        assert isinstance(getattr(forcing, name), TimeSeries), name

    # The one-step structure is stable: an uncoupled workflow scans exactly
    # as well as a coupled one, just without exchanging anything.
    carry = coupler.initialize()
    final, _ = jax.eval_shape(coupler.generate_trajectory_function(1), carry)
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)


def test_an_explicit_declaration_with_no_active_exchanger_leaves_forcing_unfrozen():
    """The same uncoupled-comparison workflow, for a hand-written exchanger.

    `coupling.exchanged_forcing` exists so a configuration built around a
    hand-written `coupling.exchanger` can say what it writes, since a Python
    function cannot be read off the way `jem.exchangers.Exchange`'s table
    can (see `declare_exchanged_forcing`'s docstring). Before the fix, only
    the *derived* branch was filtered by `active` (7f346c5): the explicit
    branch collapsed every declared field regardless of whether
    `coupling.workflow` actually runs the exchanger that is supposed to
    write them, so `coupling.workflow=[atm,ocn,seaice]` -- the supported way
    to run every component side by side with no coupling at all -- silently
    froze `sea_surface_temperature` at its start-date value even though
    nothing ever wrote it.
    """
    from jcm.forcing import TimeSeries

    coupler = runners.build_coupler(composed([
        "+configuration=earth-slab",
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
        "+coupling.exchanged_forcing=[sea_surface_temperature]",
        "coupling.workflow=[atm,ocn,seaice]",
    ]))

    assert "exchange" not in coupler.workflow
    # Nothing is active to write it, so the declaration is inert.
    assert coupler.components["atm"].exchanged_forcing == ()

    forcing = coupler.initialize().components["atm"]["forcing"]
    assert isinstance(forcing.sea_surface_temperature, TimeSeries)

    # And the trajectory still scans: an uncoupled workflow is exactly as
    # valid to trace as a coupled one.
    carry = coupler.initialize()
    final, _ = jax.eval_shape(coupler.generate_trajectory_function(1), carry)
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)


def test_a_declared_time_varying_field_with_no_writer_is_rejected():
    """Codex round 20 (P2): a declared field the active table never writes.

    Declaring `sice_am` for an atmosphere/ocean coupling that has no
    sea-ice component reproduces the reported gap exactly: `sice_am` really
    is time-varying (with `forcing=from_file`), so the `pinned` warning above
    cannot catch it -- that check only fires for a declared name that is
    *not* time-varying. Before the fix this fell through both safety nets
    and `atm.initialize()` silently collapsed `sice_am` to its start-date
    value for the rest of the run; only `sea_surface_temperature` is ever
    written here (the standalone `Exchange` below has no `seaice` row), so
    this must now be rejected outright rather than warned about.
    """
    cfg = composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
        "+coupling.exchanged_forcing=[sea_surface_temperature,sice_am]",
    ])
    atm = runners.build_atmosphere(cfg)
    # A fully inspectable table -- an `Exchange`, not a hand-written function
    # -- that only ever writes `sea_surface_temperature`, standing in for the
    # atmosphere/ocean-only coupling (`seaice=none`) the finding names.
    exchangers = {"exchange": Exchange([
        {"src": "ocn.state.sea_surface_temperature",
         "dst": "atm.forcing.sea_surface_temperature"},
    ])}

    # Deliberately not a bare `pytest.raises` block: if the guard regresses,
    # the failure here has to show the actual silent freeze (what the
    # reported bug looked like in practice), not just "no exception raised".
    from jcm.forcing import TimeSeries

    try:
        runners.declare_exchanged_forcing(cfg, atm, exchangers)
    except ValueError as error:
        assert "sice_am" in str(error)
        assert "sea_surface_temperature" in str(error)
        # The rejection has to actually prevent the freeze, not just announce
        # it: `atm.set_exchanged_forcing` must never have been reached, so
        # `sice_am` is still the time-varying `TimeSeries` it was built as,
        # not the array `initialize()` would otherwise have collapsed it to.
        assert atm.exchanged_forcing == ()
        assert isinstance(atm.initialize()["forcing"].sice_am, TimeSeries)
    else:
        pytest.fail(
            "declare_exchanged_forcing accepted a declared field "
            "('sice_am') that the active coupling table never writes; "
            f"atm.exchanged_forcing = {atm.exchanged_forcing!r} and "
            "atm.initialize()['forcing'].sice_am is now "
            f"{type(atm.initialize()['forcing'].sice_am).__name__} instead "
            f"of {TimeSeries.__name__} -- the exact silent freeze the "
            "Codex round 20 finding reported."
        )


def test_a_declared_field_with_no_writer_is_accepted_when_an_exchanger_is_opaque():
    """The negative case: an opaque active exchanger makes the set unprovable.

    `coupling.exchanged_forcing` exists precisely so a hand-written
    `coupling.exchanger` can say what it writes, since its body cannot be
    read off (see `declare_exchanged_forcing`'s docstring). So even though
    this test's only *readable* table -- there is none -- writes nothing,
    declaring `sice_am` here must stay silent: the hand-written exchanger
    might be the one writing it, and this function can never know that it
    is not.
    """
    cfg = composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
        "+coupling.exchanged_forcing=[sice_am]",
    ])
    atm = runners.build_atmosphere(cfg)
    runners.declare_exchanged_forcing(
        cfg, atm, {"exchange": example_exchanger}
    )
    assert atm.exchanged_forcing == ("sice_am",)


def test_a_declared_field_with_a_writer_is_accepted():
    """The base case: a declared field the active table does write is fine.

    Same shape as the reported-gap test above, but `sea_surface_temperature`
    is exactly what the standalone `Exchange` writes, so nothing should be
    rejected or warned about.
    """
    cfg = composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
        "+coupling.exchanged_forcing=[sea_surface_temperature]",
    ])
    atm = runners.build_atmosphere(cfg)
    exchangers = {"exchange": Exchange([
        {"src": "ocn.state.sea_surface_temperature",
         "dst": "atm.forcing.sea_surface_temperature"},
    ])}
    runners.declare_exchanged_forcing(cfg, atm, exchangers)
    assert atm.exchanged_forcing == ("sea_surface_temperature",)


def test_earth_slab_starts_its_sea_ice_from_the_observed_cover():
    """The ice the atmosphere is handed on step 0 is the file's, not zero.

    The exchange runs before the components and a `derived` field is only
    rewritten at the end of a step, so `seaice.initialize()`'s `ice_fraction`
    is what `atm.forcing.sice_am` holds for the first two coupling steps. An
    Earth-like run must not begin with ice-free poles.
    """
    coupler = runners.build_coupler(composed(["+configuration=earth-slab"]))
    carries = coupler.initialize().components

    ice_fraction = carries["seaice"]["derived"].ice_fraction
    assert float(jnp.max(ice_fraction)) > 0.9
    assert float(jnp.mean(ice_fraction)) > 0.01
    # Finite, because the fraction closure's inverse is capped: a fully
    # covered cell would otherwise be infinitely thick.
    thickness = carries["seaice"]["state"].ice_thickness
    assert bool(jnp.all(jnp.isfinite(thickness)))
    assert float(jnp.max(thickness)) <= float(
        coupler.components["seaice"].params.max_initial_ice_thickness
    )

    # And it reaches the atmosphere: the exchange puts it in `sice_am`.
    exchanged = coupler.exchangers["exchange"](dict(carries), coupler.coupling_time(0))
    np.testing.assert_allclose(
        np.asarray(exchanged["atm"]["forcing"].sice_am), np.asarray(ice_fraction)
    )


def test_earth_slab_ocean_starts_at_or_above_freezing():
    """The packaged SST climatology is sub-freezing under ice; the ocean is not.

    Taken verbatim the mixed layer would start tens of kelvin below the floor
    `step` holds it to, and the first step would turn the whole deficit into
    frazil ice.
    """
    from jem import constants

    coupler = runners.build_coupler(composed(["+configuration=earth-slab"]))
    carries = coupler.initialize().components

    ocean = np.asarray(coupler.components["ocn"].grid.binary_mask == 0.0)
    sea_surface_temperature = np.asarray(
        carries["ocn"]["state"].sea_surface_temperature
    )
    assert sea_surface_temperature[ocean].min() >= constants.seawater_freezing_point_K


def test_exchanged_forcing_can_be_declared_in_the_config():
    """A hand-written exchanger cannot be read, so the config says what it writes."""
    cfg = composed([
        "+configuration=earth-slab",
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
        "+coupling.exchanged_forcing=[sea_surface_temperature]",
    ])
    coupler = runners.build_coupler(cfg)
    assert coupler.components["atm"].exchanged_forcing == ("sea_surface_temperature",)


def test_a_hand_written_exchanger_over_a_file_forcing_is_warned_about(caplog):
    """Nothing to read and nothing declared: say so rather than guess.

    The warning is worth its noise only when something is actually still a
    time series, so it names the fields that are.
    """
    cfg = composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
    ])
    atm = runners.build_atmosphere(cfg)
    with caplog.at_level(logging.WARNING, logger="jem.runners"):
        runners.declare_exchanged_forcing(
            cfg, atm, {"exchange": example_exchanger}
        )
    assert "hand-written" in caplog.text
    assert "sea_surface_temperature" in caplog.text
    assert atm.exchanged_forcing == ()


def test_a_hand_written_exchanger_over_a_plain_forcing_is_not_warned_about(caplog):
    """With the default forcing every field is already an array: nothing to say."""
    cfg = composed([
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
    ])
    atm = runners.build_atmosphere(cfg)
    assert atm.time_varying_forcing == ()
    with caplog.at_level(logging.WARNING, logger="jem.runners"):
        runners.declare_exchanged_forcing(
            cfg, atm, {"exchange": example_exchanger}
        )
    assert "hand-written" not in caplog.text


def test_a_bare_string_exchanged_forcing_is_refused():
    """A string is an iterable of characters, so it is rejected by name.

    `exchanged_forcing=sice_am` is the spelling a user reaches for first; read
    as a list it declares seven one-letter fields and the error then names
    those instead of the missing brackets.
    """
    cfg = composed([
        "+configuration=earth-slab",
        "+coupling.exchanged_forcing=sice_am",
    ])
    atm = runners.build_atmosphere(cfg)
    with pytest.raises(ValueError, match=r"\[sice_am\]"):
        runners.declare_exchanged_forcing(cfg, atm, {})


def test_declaring_a_field_nothing_varies_is_warned_about(caplog):
    """A declared field that is not time-varying is pinned for nothing.

    The declaration mechanism can reintroduce, by hand, exactly the failure
    the derivation exists to avoid: a climatology held at its start-date value
    with no symptom. `alb0` is a plain annual-mean array in the packaged file,
    so declaring it says something the forcing cannot honour.
    """
    cfg = composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
        "+coupling.exchanged_forcing=[sea_surface_temperature,alb0]",
    ])
    atm = runners.build_atmosphere(cfg)
    with caplog.at_level(logging.WARNING, logger="jem.runners"):
        runners.declare_exchanged_forcing(cfg, atm, {})
    assert "alb0" in caplog.text
    assert "not time-varying" in caplog.text


def test_an_explicit_declaration_still_warns_about_what_it_left_out(caplog):
    """The safety net covers the declared branch too, not only the derived one."""
    cfg = composed([
        "forcing@atmosphere.forcing=from_file",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
        "+coupling.exchanged_forcing=[sea_surface_temperature]",
    ])
    atm = runners.build_atmosphere(cfg)
    with caplog.at_level(logging.WARNING, logger="jem.runners"):
        runners.declare_exchanged_forcing(
            cfg, atm, {"exchange": example_exchanger}
        )
    # The four it did not declare are still time series, and a hand-written
    # exchanger might be writing any of them.
    assert "stl_am" in caplog.text
    assert atm.exchanged_forcing == ("sea_surface_temperature",)


# ---------------------------------------------------------------------------
# The run settings
# ---------------------------------------------------------------------------


def test_coupled_run_keys_are_run_chunked_arguments():
    """Every `coupled_run` key is a `run_chunked` keyword, bar the log level.

    The group file is the run's complete schema, and `run_chunked` owns every
    default in it. A key here that the driver does not take would compose,
    override cleanly on the command line, and do nothing.
    """
    from jem import driver

    keys = set(OmegaConf.to_container(composed([]).coupled_run))
    arguments = set(inspect.signature(driver.run_chunked).parameters)
    assert keys - {"log_level"} <= arguments
    assert "log_level" not in arguments
