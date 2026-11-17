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
from pathlib import Path

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
    assert coupler.start_date == coupler.components["atm"].model.start_time
    # The slabs were built on the atmosphere's grid, which is what
    # `regrid=same_grid` (no regridders) assumes.
    horizontal = coupler.components["atm"].model.coords.horizontal
    assert coupler.components["ocn"].grid.shape == (
        len(horizontal.nodal_axes[0]), len(horizontal.nodal_axes[1])
    )


# The atmosphere overrides `python -m jem.main --help`, `jem/config/config.yaml`,
# `jem/main.py` and docs/source/getting_started.rst show a reader. Kept here so
# that the commands the documentation teaches are *built*, not just composed:
# an override can compose cleanly and still name a model that cannot be
# coupled, which is only found by building it.
DOCUMENTED_ATMOSPHERE_OVERRIDES = [
    ["diffusion@atmosphere.diffusion=strong", "atmosphere.run.time_step=15"],
    ["atmosphere.run.time_step=15"],
]


@pytest.mark.parametrize(
    "overrides", DOCUMENTED_ATMOSPHERE_OVERRIDES, ids=lambda o: " ".join(o)
)
def test_the_documented_atmosphere_overrides_build(overrides):
    """Every atmosphere override the documentation shows builds a coupled model."""
    coupler = runners.build_coupler(composed(overrides))

    assert coupler.components["atm"].model.dt_si.m == 15 * 60
    root = Path(runners.__file__).parent
    spelling = " ".join(overrides)
    documented = "".join(
        " ".join(path.read_text().replace("\\\n", " ").split())
        for path in (
            root / "config" / "config.yaml",
            root / "config" / "hydra" / "help" / "custom_help.yaml",
            root / "main.py",
            root.parent / "docs" / "source" / "getting_started.rst",
        )
    )
    for override in overrides:
        assert override in documented, (
            f"{override!r} is tested as a documented example ({spelling}) but"
            " no longer appears in the help text or the docs; keep"
            " DOCUMENTED_ATMOSPHERE_OVERRIDES in step with them."
        )


def test_an_atmosphere_without_a_surface_is_refused():
    """A physics package that publishes no surface fluxes cannot be coupled.

    Held-Suarez forcing resolves no surface at all, so there is nothing for
    an ocean to be driven with; the build says so rather than coupling a
    model whose exchange is all zeros.
    """
    cfg = composed([
        "physics@atmosphere.physics=held_suarez",
        "grid@atmosphere.grid=held_suarez_t31_l8",
    ])
    with pytest.raises(ValueError, match="surface_exchange"):
        runners.build_coupler(cfg)


def test_an_atmosphere_timestep_that_does_not_divide_the_coupling_step_is_refused():
    """Seven-minute steps do not fit a whole number of times into a day."""
    with pytest.raises(ValueError, match="not a whole multiple"):
        runners.build_coupler(composed(["atmosphere.run.time_step=7"]))


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
    model = jcm.model.Model(coords=get_speedy_coords(), start_time=start_date)
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
    # An aquaplanet: no land fraction file is given, so the grid's own all-active
    # SCRIP mask is used and every ocean and sea-ice cell is ocean. A packaged
    # land mask would leave cells that never leave their initial temperature.
    assert float(ocean_grid.fractional_mask.max()) == 0.0
    assert bool(np.all(np.asarray(ocean_grid.ocean_mask)))
    assert float(coupler.components["seaice"].grid.fractional_mask.max()) == 0.0
    # The ocean grid is the displaced-pole one: curvilinear, not the
    # atmosphere's separable Gaussian grid.
    assert not ocean_grid.is_separable

    exchange = coupler.exchangers["exchange"]
    crossing = {spec.regrid for spec in exchange.specs if spec.regrid is not None}
    assert crossing == {"a2o_flux", "o2a_flux", "o2a_state"}
    assert all(name in exchange.regridders for name in crossing)


@pytest.mark.parametrize(
    ("overrides", "keys", "hint"),
    [
        (["ocean=slab_relax"], ["ocean.sst_clim_file"], "${jcm_data:bc/t30/clim/forcing.nc}"),
        (
            ["regrid=esmf"],
            ["regrid.a2o_conserve", "regrid.a2o_bilinear",
             "regrid.o2a_conserve", "regrid.o2a_bilinear"],
            "${jem_data:weight_algo-conserve_JCM_T31_to_DisplacedPoleGrid.nc}",
        ),
        (["ocean=veros"], ["ocean.setup"], "jem.components.veros.setups."),
    ],
    ids=["slab_relax", "esmf", "veros"],
)
def test_a_missing_required_input_names_its_full_key_and_the_override(
    overrides, keys, hint
):
    """`???` surfaces as the full key and the override to add, not the bare leaf.

    OmegaConf alone says `Missing mandatory value: sst_clim_file` -- no group,
    no hint. The error names `ocean.sst_clim_file` and shows the line to add,
    spelled with the packaged-file resolver where the repo's docs use one.
    """
    from omegaconf.errors import MissingMandatoryValue

    # The component is built before anything that needs a grid or a file, so
    # no atmosphere is needed for the `ocean` cases; the regridders are built
    # on their own.
    cfg = composed(overrides)
    with pytest.raises(MissingMandatoryValue) as info:
        if overrides[0].startswith("regrid"):
            runners.build_regridders(cfg)
        else:
            runners.build_component(cfg.ocean, key="ocean")
    message = str(info.value)
    for key in keys:
        assert f"  {key}=" in message, message
    assert hint in message


def test_a_component_is_handed_the_local_copy_of_a_mirror_file(monkeypatch):
    """An ``hf://`` value reaches the constructor as the fetched file's path.

    jax-gcm's configuration bundles name their forcing on its data mirror, and
    a surface component that reads the same file is handed that name by
    interpolation (``sst_clim_file: ${atmosphere.forcing.file}``). xarray
    cannot open the scheme, so the runner fetches through jax-gcm's own
    ``fetch`` -- at any depth of the node, and leaving every other value alone.
    """
    import jcm.data.remote

    fetched = []

    def fake_fetch(relative_path):
        fetched.append(relative_path)
        return f"/cache/{relative_path}"

    monkeypatch.setattr(jcm.data.remote, "fetch", fake_fetch)
    node = OmegaConf.create(
        {
            "_target_": "builtins.dict",
            "sst_clim_file": "hf://bundles/t63/forcing_pd.nc",
            "local_file": "/somewhere/forcing.nc",
            "nested": {"files": ["hf://bundles/t63/terrain.nc", "plain.nc"]},
        }
    )
    built = runners.build_component(node, key="ocean")
    assert built == {
        "sst_clim_file": "/cache/bundles/t63/forcing_pd.nc",
        "local_file": "/somewhere/forcing.nc",
        "nested": {"files": ["/cache/bundles/t63/terrain.nc", "plain.nc"]},
    }
    assert fetched == ["bundles/t63/forcing_pd.nc", "bundles/t63/terrain.nc"]
    # The composed config a caller may still be reading keeps what it was given.
    assert node.sst_clim_file == "hf://bundles/t63/forcing_pd.nc"


def test_a_missing_input_is_named_by_the_coupler_build_too():
    """`build_coupler` passes each group's key, so the CLI error is the full key."""
    from omegaconf.errors import MissingMandatoryValue

    with pytest.raises(MissingMandatoryValue, match=r"ocean\.sst_clim_file"):
        runners.build_coupler(composed(["ocean=slab_relax"]))


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
    cfg = composed(["seaice=none"])
    assert cfg.seaice is None

    exchangers = runners.build_exchangers(cfg, {"atm": None, "ocn": None}, {})
    specs = exchangers["exchange"].specs
    assert specs, "the remaining components still have rows to exchange"
    assert not [spec for spec in specs if "seaice" in (spec.src + spec.dst)]

    with_ice = runners.build_exchangers(
        composed([]), {"atm": None, "ocn": None, "seaice": None}, {}
    )
    assert [spec for spec in with_ice["exchange"].specs
            if "seaice" in (spec.src + spec.dst)]


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


def test_earth_slab_with_the_bucket_land_couples_the_soil_water():
    """`land=bucket` on `earth-slab`: the land is built on the atmosphere's
    grid, with its cell areas, the atmosphere's `soilw_am` is declared
    exchanged (a plain array, no longer the file's time series), and a
    coupled step keeps the carry's structure.
    """
    from jcm.forcing import TimeSeries

    coupler = runners.build_coupler(composed([
        "+configuration=earth-slab", "land=bucket",
        "+land.soil_water_clim_file=${jcm_data:bc/t30/clim/forcing.nc}",
    ]))
    assert set(coupler.components) == {"atm", "ocn", "seaice", "lnd"}
    assert coupler.components["lnd"].grid.cell_solid_angle is not None
    assert coupler.components["atm"].publishes_land_evaporation
    assert set(coupler.components["atm"].exchanged_forcing) == {
        "sea_surface_temperature", "sice_am", "soilw_am",
    }
    carry = coupler.initialize()
    forcing = carry.components["atm"]["forcing"]
    assert not isinstance(forcing.soilw_am, TimeSeries)
    assert isinstance(forcing.stl_am, TimeSeries)

    final, _ = jax.eval_shape(coupler.generate_trajectory_function(1), carry)
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)


def test_earth_slab_winton_couples_the_ice_both_ways():
    """`+configuration=earth-slab-winton`: a clamping ocean, the Winton ice
    started from the observed cover, the atmosphere declared to take the
    ice's skin temperature (seeded, since jax-gcm leaves it None without an
    ice), and a coupled step that keeps the carry's structure.
    """
    from jem.components import WintonSeaiceModel

    coupler = runners.build_coupler(composed(["+configuration=earth-slab-winton"]))
    assert isinstance(coupler.components["seaice"], WintonSeaiceModel)
    assert coupler.components["ocn"].params.clamp_at_freezing
    assert set(coupler.components["atm"].exchanged_forcing) == {
        "sea_surface_temperature", "sice_am", "sea_ice_temperature",
    }
    carry = coupler.initialize()
    seaice = carry.components["seaice"]["state"]
    assert 0.0 < float(seaice.ice_fraction.mean()) < 1.0
    tice = carry.components["atm"]["forcing"].sea_ice_temperature
    assert tice is not None and tice.shape == seaice.ice_fraction.shape

    final, _ = jax.eval_shape(coupler.generate_trajectory_function(1), carry)
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)


def test_earth_slab_couples_its_file_forcing(caplog):
    """The one shipped configuration with `forcing=from_file` builds and scans.

    `earth-slab` gives the atmosphere jax-gcm's packaged T30 climatology, in
    which every surface boundary condition is a time-varying `TimeSeries`,
    and then has the ocean and sea ice overwrite two of them with plain
    arrays every step. The runner has to tell the atmosphere which two, or
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
    # The land fields stay the atmosphere's own climatology (SPEEDY's slab
    # land integrates its anomaly about `stl_am`), so nothing writes them.
    assert set(coupler.components["atm"].exchanged_forcing) == {
        "sea_surface_temperature", "sice_am",
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


def test_forcing_align_auto_raises_for_an_undeclared_user_file():
    """jax-gcm#884 (PR 877): `auto` no longer infers a file's kind, ever.

    `earth-slab`'s own forcing file resolves under `auto` because it is
    jax-gcm's packaged T30 climatology, which the data-mirror manifest
    records as `alignment: climatology` (see the previous test and
    `jem/components/jcm/contract.py`'s note on jax-gcm#884). This test is the
    other half: a *user's* file -- anything the manifest does not
    recognise -- must raise jax-gcm's own named error rather than silently
    guessing, so a user who points `atmosphere.forcing.file` of
    `+configuration=veros-double-drake` (which names its own file's `align`
    explicitly) at their own archive and leaves `atmosphere.forcing.align=auto`
    is told to declare it, not handed a silently mis-phased forcing.

    The file used here does not need to exist or even be a real netCDF: the
    manifest lookup (`jcm.data.input_resolution.manifest_alignment_for_paths`)
    only pattern-matches the *path*, and `resolve_align` raises before
    jax-gcm ever tries to open it.
    """
    cfg = composed([
        "+configuration=veros-double-drake",
        "atmosphere.forcing.align=auto",
        "atmosphere.forcing.file=${jem_data:terrain_double_drake_T31.nc}",
    ])
    with pytest.raises(ValueError, match="forcing.align=auto cannot resolve"):
        runners.build_atmosphere(cfg)


def test_forcing_align_auto_resolves_jcms_own_packaged_climatology():
    """The positive case for the same rule: a manifest-known file needs no align.

    Pointing the SAME `from_file` override at jax-gcm's own packaged T30
    climatology (`${jcm_data:bc/t30/clim/forcing.nc}`, the file
    `+configuration=earth-slab` uses) resolves under `auto` with no
    `atmosphere.forcing.align` override -- proving the failure above is about
    the file, not about composing `from_file` onto a Veros configuration.
    """
    cfg = composed([
        "+configuration=veros-double-drake",
        "atmosphere.forcing.align=auto",
        "atmosphere.forcing.file=${jcm_data:bc/t30/clim/forcing.nc}",
    ])
    atm = runners.build_atmosphere(cfg)
    assert atm.forcing is not None


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

    The exchange runs before the components, so `seaice.initialize()`'s
    `ice_fraction` -- not one diagnosed from an SST the sea ice has not been
    handed yet -- is what `atm.forcing.sice_am` holds on the atmosphere's
    first coupling step. An Earth-like run must not begin with ice-free
    poles.
    """
    coupler = runners.build_coupler(composed(["+configuration=earth-slab"]))
    carries = coupler.initialize().components

    ice_fraction = carries["seaice"]["derived"].ice_fraction
    assert float(jnp.max(ice_fraction)) > 0.9
    assert float(jnp.mean(ice_fraction)) > 0.01
    # A concentration: finite and within [0, 1] everywhere, land included.
    assert bool(jnp.all(jnp.isfinite(ice_fraction)))
    assert float(jnp.min(ice_fraction)) >= 0.0
    assert float(jnp.max(ice_fraction)) <= 1.0

    # And it reaches the atmosphere: the exchange puts it in `sice_am`.
    exchanged = coupler.exchangers["exchange"](
        dict(carries), coupler.coupling_time(0, coupler.start_date)
    )
    np.testing.assert_allclose(
        np.asarray(exchanged["atm"]["forcing"].sice_am), np.asarray(ice_fraction)
    )


def test_earth_slab_ocean_starts_at_or_above_freezing():
    """The packaged SST climatology is sub-freezing under ice; the ocean is not.

    Taken verbatim the mixed layer would start tens of kelvin below freezing,
    a heat deficit the run never had.
    """
    from jem import constants

    coupler = runners.build_coupler(composed(["+configuration=earth-slab"]))
    carries = coupler.initialize().components

    ocean = np.asarray(coupler.components["ocn"].grid.ocean_mask)
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


def test_initial_condition_reaches_the_run_through_the_runner(tmp_path, monkeypatch):
    """`coupled_run.initial_condition` starts a run under another start date.

    Goes through `runners.run`, the function `python -m jem.main` calls, on a
    two-slab model swapped in for `build_coupler` (the runner hands
    `run_chunked` whatever coupler it is given, and knows nothing of its
    components): a January checkpoint becomes the initial condition of a July
    run, and repeating the command resumes the run instead.
    """
    from tests.unit.test_driver import JULY_START, probed_slabs, spun_up_in_january

    spun_up = tmp_path / "spun-up"
    probed_slabs(jdt.to_datetime("2001-01-01")).save_carry(spun_up_in_january(), spun_up)
    monkeypatch.setattr(runners, "build_coupler", lambda cfg: probed_slabs(JULY_START))

    def launch(total_time: str):
        return runners.run(composed([
            f"coupled_run.total_time={total_time}", "coupled_run.chunk=1 day",
            f"coupled_run.initial_condition={spun_up}",
            f"coupled_run.output_dir={tmp_path / 'experiment'}",
        ]))

    first = launch("2 days")
    assert first.final_carry.time == JULY_START + jdt.to_timedelta(2, "day")
    assert int(first.final_carry.step) == 2

    second = launch("3 days")
    assert second.steps_completed == 3
    assert int(second.final_carry.step) == 3


def test_initial_condition_is_a_schema_key_that_defaults_to_none():
    """The key is in the group's schema, so it overrides without a `+`."""
    assert composed([]).coupled_run.initial_condition is None
    kwargs = runners.build_run_kwargs(composed(["coupled_run.initial_condition=/x/y"]))
    assert kwargs["initial_condition"] == "/x/y"


# ---------------------------------------------------------------------------
# The precision of a run is part of its configuration (`jax_enable_x64`)
# ---------------------------------------------------------------------------

#: `jax_enable_x64` as this process had it when the module was collected. The
#: last test of the module checks the process is left the way it was found.
_X64_WHEN_COLLECTED = jax.config.read("jax_enable_x64")


@pytest.fixture
def restore_x64():
    """Put `jax_enable_x64` back after a test that builds a coupler.

    The flag is process-global and `runners.build_coupler` changes it by
    design, so a test that builds one with a precision of its own would
    otherwise decide the precision of every test that runs after it.
    """
    before = jax.config.read("jax_enable_x64")
    yield
    jax.config.update("jax_enable_x64", before)


def _float_dtypes(tree, *, ignore_jcm_fixed_forcing: bool = False) -> set[str]:
    """Return the dtype names of every floating-point leaf of ``tree``.

    ``ignore_jcm_fixed_forcing`` skips the atmosphere's forcing leaves that
    are scalars or empty. jax-gcm builds the solar-geometry scalars and the
    empty analytic-ozone array as float32 whatever ``jax_enable_x64`` is
    (``jcm.forcing`` passes ``dtype=jnp.float32`` explicitly), so they are
    not something this layer can set.

    """
    found = set()
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    for path, leaf in leaves:
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            continue
        if (ignore_jcm_fixed_forcing and "forcing" in jax.tree_util.keystr(path)
                and (array.ndim == 0 or array.size == 0)):
            continue
        found.add(str(array.dtype))
    return found


def test_precision_key_is_null_by_default_and_a_schema_key():
    """Unset means "leave JAX alone", and the key overrides without a `+`."""
    assert composed([]).jax_enable_x64 is None
    assert composed(["jax_enable_x64=true"]).jax_enable_x64 is True
    assert composed(["jax_enable_x64=false"]).jax_enable_x64 is False


@pytest.mark.parametrize("overrides", [
    ["ocean=veros"],
    ["+configuration=veros-earth"],
    ["+configuration=veros-double-drake"],
])
def test_choosing_the_veros_ocean_selects_double_precision(overrides):
    """`ocean=veros` carries the setting, with or without a named configuration."""
    assert composed(overrides).jax_enable_x64 is True


@pytest.mark.parametrize("configuration", [
    "aquaplanet-slab", "aquaplanet-slab-mixed-grid", "earth-slab",
    "earth-slab-winton",
])
def test_slab_configurations_leave_the_precision_alone(configuration):
    assert composed([f"+configuration={configuration}"]).jax_enable_x64 is None


def test_a_command_line_setting_beats_the_ocean_options_selection():
    """An explicit `false` is still `false` in the composed config (the build refuses it)."""
    assert composed(["ocean=veros", "jax_enable_x64=false"]).jax_enable_x64 is False


def test_apply_precision_sets_the_flag_and_reports_what_it_must_stay(restore_x64):
    jax.config.update("jax_enable_x64", False)
    assert runners.apply_precision(composed(["jax_enable_x64=true"])) is True
    assert jax.config.read("jax_enable_x64") is True
    assert runners.apply_precision(composed(["jax_enable_x64=false"])) is False
    assert jax.config.read("jax_enable_x64") is False
    # null leaves it as the process has it, whichever way that is.
    jax.config.update("jax_enable_x64", True)
    assert runners.apply_precision(composed([])) is True
    assert jax.config.read("jax_enable_x64") is True


def test_apply_precision_refuses_a_non_boolean(restore_x64):
    cfg = composed([])
    cfg.jax_enable_x64 = "yes please"
    with pytest.raises(TypeError, match="jax_enable_x64 must be true, false or null"):
        runners.apply_precision(cfg)


@pytest.mark.parametrize("x64, dtype", [(True, "float64"), (False, "float32")])
def test_a_slab_run_is_the_precision_it_asks_for_from_the_start(
    x64, dtype, restore_x64
):
    """Atmosphere and slab ocean carries hold one float dtype, the requested one.

    The setting is applied before the atmosphere is built, so nothing is
    allocated in the other precision.
    """
    coupler = runners.build_coupler(composed([f"jax_enable_x64={str(x64).lower()}"]))
    carry = coupler.initialize()
    assert _float_dtypes(carry.components["atm"], ignore_jcm_fixed_forcing=True) == {dtype}
    assert _float_dtypes(carry.components["ocn"]) == {dtype}


@pytest.mark.slow
def test_a_float64_slab_run_scans_two_coupled_steps(restore_x64):
    """The carry that leaves a float64 step has the structure and dtypes it entered with."""
    coupler = runners.build_coupler(composed(["jax_enable_x64=true"]))
    carry = coupler.initialize()
    final, _ = coupler.generate_trajectory_function(2)(carry)
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)
    assert [jnp.asarray(leaf).dtype for leaf in jax.tree_util.tree_leaves(final)] == [
        jnp.asarray(leaf).dtype for leaf in jax.tree_util.tree_leaves(carry)
    ]
    assert _float_dtypes(final.components["atm"], ignore_jcm_fixed_forcing=True) == {"float64"}
    sst = final.components["ocn"]["state"].sea_surface_temperature
    assert bool(jnp.all(jnp.isfinite(sst)))


@pytest.fixture
def component_that_flips_x64(monkeypatch):
    """Make building the ocean turn x64 on, as the first Veros import does."""
    real = runners.build_component

    def flipping(node, **kwargs):
        built = real(node, **kwargs)
        jax.config.update("jax_enable_x64", True)
        return built

    monkeypatch.setattr(runners, "build_component", flipping)


@pytest.mark.parametrize("overrides", [["jax_enable_x64=false"], []])
def test_a_component_that_changes_the_precision_mid_build_is_refused(
    overrides, component_that_flips_x64, restore_x64
):
    """The mix is detected right after the build and the message names the key.

    Both ways in: a configuration that asked for float32, and one that asked
    for nothing and so expected the process's float32 to hold. The
    atmosphere was built before the flip in both.
    """
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(RuntimeError, match=r"jax_enable_x64=true"):
        runners.build_coupler(composed(overrides))


def test_a_component_that_agrees_with_the_requested_precision_is_accepted(
    component_that_flips_x64, restore_x64
):
    """Requesting `true` makes the same flip a no-op, so the build goes through."""
    coupler = runners.build_coupler(composed(["jax_enable_x64=true"]))
    assert "ocn" in coupler.components


def test_zz_the_process_is_left_with_the_precision_it_was_collected_with():
    """No test in this module leaks a changed `jax_enable_x64` into the next one.

    `build_coupler` changes the flag by design and the tests above restore it;
    composing a Veros configuration changes nothing (only building one does).
    Kept last in the file so that it sees what the others left behind.
    """
    composed(["ocean=veros"])
    assert jax.config.read("jax_enable_x64") == _X64_WHEN_COLLECTED
