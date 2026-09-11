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

import jax
import jax_datetime as jdt
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

CONFIG_MODULE = "jem.config"


def composed(overrides: list[str]):
    """Compose the primary config with ``overrides``."""
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def example_exchanger(components: dict[str, Carry], time: CouplingTime):
    """Return the carries unchanged; a config points at this by name."""
    del time
    return components


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


def test_exchanger_and_exchangers_together_are_an_error():
    """Setting both leaves it undecided which couples the run, so it is refused."""
    cfg = composed([
        "coupling.exchanger=tests.unit.test_runners.example_exchanger",
        "+coupling.exchangers=[{src: 'ocn.state.sea_surface_temperature',"
        " dst: 'atm.forcing.sea_surface_temperature'}]",
    ])
    with pytest.raises(ValueError, match="are both"):
        runners.build_exchangers(cfg, {"atm": None, "ocn": None}, {})


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
