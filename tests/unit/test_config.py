"""Tests for `jem.config`: the Hydra groups and the named configurations.

Three things are checked here, and they are the three ways this layer can be
wrong without anyone noticing:

* **Everything composes.** Every option of every group, and every named
  configuration, is composed -- including the JAX-GCM configuration bundles
  re-rooted under ``atmosphere``.
* **What composes takes effect.** Composing a group option that the builder
  ignores is worse than not having the option, so the atmosphere matrix is
  actually *built* and the built model interrogated.
* **The YAML stays thin.** ``test_config_has_no_python_defaults`` fails if a
  config file supplies a value that equals the target's own default -- the
  one rule that keeps defaults living in Python.
"""

import importlib
import inspect
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from hydra.utils import get_object
from omegaconf import DictConfig, ListConfig, OmegaConf

# Importing the package registers the ${jcm_data:} / ${jem_data:} resolvers the
# configurations use; anything that composes a JEM config has to do this first.
import jem.config

CONFIG_MODULE = "jem.config"
CONFIG_DIR = Path(jem.config.__file__).parent
JCM_CONFIG_DIR = Path(str(importlib.resources.files("jcm.config")))

#: Directories under `jem/config/` that are not option groups.
NOT_A_GROUP = {"hydra", "__pycache__"}

#: Keys a component node may carry that are NOT constructor arguments of its
#: `_target_`: `jem.runners` reads and removes them before instantiating. They
#: are exempt from the no-Python-defaults check for the same reason a `**kwargs`
#: argument is -- there is no signature default to compare against.
RUNNER_ONLY_KEYS = frozenset({"grid_file", "land_fraction_file"})


def _groups() -> list[str]:
    """Return JAX-ESM's own config groups, `configuration` included."""
    return sorted(
        p.name for p in CONFIG_DIR.iterdir()
        if p.is_dir() and p.name not in NOT_A_GROUP
    )


def _options(group: str) -> list[str]:
    """Return the option names of one group."""
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


def _group_override(group: str, option: str) -> str:
    """Return the command-line override selecting ``option`` of ``group``.

    Every JAX-ESM group composes at its own name, so one spelling serves all
    of them -- there is no package to carry.
    """
    return f"{group}={option}"


def _atmosphere_groups() -> dict[str, str]:
    """Return the JCM groups `config.yaml` re-roots, as group -> package.

    Read from the primary config's defaults list rather than hard-coded, so
    that a group added to (or dropped from) the re-rooted set is covered here
    without the test needing an edit.
    """
    defaults = OmegaConf.load(CONFIG_DIR / "config.yaml").defaults
    groups = {}
    for entry in defaults:
        if not isinstance(entry, DictConfig):
            continue
        for key in entry:
            group, _, package = str(key).partition("@")
            if package.startswith("atmosphere."):
                groups[group] = package
    return groups


def composed(overrides: list[str]) -> DictConfig:
    """Compose the primary config with ``overrides``."""
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _target_nodes(node, path: str = "") -> Iterator[tuple[str, DictConfig]]:
    """Yield every ``(path, node)`` under ``node`` that carries a ``_target_``."""
    if isinstance(node, DictConfig):
        if "_target_" in node:
            yield path, node
        for key in node:
            if OmegaConf.is_missing(node, key):
                continue
            yield from _target_nodes(node[key], f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for index, item in enumerate(node):
            yield from _target_nodes(item, f"{path}[{index}]")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("group", "option"),
    [(g, o) for g in _groups() if g != "configuration" for o in _options(g)],
)
def test_every_group_option_composes(group, option):
    """Every option of every JAX-ESM group composes onto the default config."""
    cfg = composed([_group_override(group, option)])
    assert group in cfg


@pytest.mark.parametrize("group", ["ocean", "land", "seaice", "regrid"])
def test_none_option_composes_to_none(group):
    """``<group>=none`` leaves the key literally null, not an empty node.

    The runner distinguishes "no component" from "a component with no
    arguments" by this, and an option file whose body is `null` composes to
    `{}` -- so the `none` options carry a `# @package _global_` header and
    assign the key from the root instead. Checked here because that is a
    non-obvious spelling that a later edit could quietly undo.
    """
    option = "none" if group != "regrid" else "same_grid"
    assert composed([f"{group}={option}"])[group] is None


@pytest.mark.parametrize("name", _options("configuration"))
def test_named_configuration_composes(name):
    """Every named coupled configuration composes, and sets something."""
    cfg = composed([f"+configuration={name}"])
    assert cfg.atmosphere.grid.spectral_truncation == 31
    # A configuration that composed but selected nothing would pass every
    # other check in this file; the ocean is the one component all five
    # configure.
    assert cfg.ocean is not None


def test_jcm_configuration_composes_under_the_atmosphere():
    """A JAX-GCM configuration bundle re-roots under ``atmosphere``.

    The bundles are ``# @package _global_`` files whose defaults are absolute
    (``override /physics: speedy``), and composing one with the group's
    package (``+configuration@atmosphere=...``) lands every one of those
    choices under ``atmosphere`` -- which is what lets JAX-ESM reuse JAX-GCM's
    validated atmospheres instead of restating them. Command-line overrides
    still apply on top.
    """
    cfg = composed(["+configuration@atmosphere=speedy-t31"])
    assert cfg.atmosphere.grid.spectral_truncation == 31
    assert "speedy_convection" in cfg.atmosphere.physics.terms
    # The bundle's own non-default settings land too.
    assert cfg.atmosphere.terrain.kind == "from_file"
    assert cfg.atmosphere.run.time_step == 15
    # ... and the top-level (coupled) config is untouched by it.
    assert cfg.coupled_run.total_time == "30 days"

    overridden = composed(
        ["+configuration@atmosphere=speedy-t31", "atmosphere.run.time_step=7"]
    )
    assert overridden.atmosphere.run.time_step == 7


def test_jcm_run_group_is_not_shadowed():
    """The coupled run group must not hide JCM's own ``run`` group.

    Hydra resolves a group option from the first search-path entry that has
    it, and this package comes before ``pkg://jcm.config``. A group called
    ``run`` here would therefore be composed into ``atmosphere.run`` in place
    of JCM's -- and, worse, silently: 16 of the 19 JAX-GCM configuration
    bundles say ``override /run: longrun``, so they would compose the coupler's
    run keys into the atmosphere. Hence the group is called ``coupled_run``,
    which is also how it reads: ``atmosphere.run`` is the atmosphere's run
    config, ``coupled_run`` the coupled model's, and neither claims a bare
    top-level ``run`` key.
    """
    assert not (CONFIG_DIR / "run").exists()
    assert "run" not in composed([])

    cfg = composed(["+configuration@atmosphere=t63-echam-1m"])
    # JCM's own run/longrun.yaml, not anything of ours.
    assert cfg.atmosphere.run.total_time == 365
    assert "time_step" in cfg.atmosphere.run
    assert cfg.coupled_run.total_time == "30 days"


def test_help_is_jem_s_own():
    """``--help`` describes JAX-ESM, not the atmosphere it composes.

    Both packages ship a `hydra/help/custom_help.yaml`, and this one has to
    win -- the same search-path precedence that makes the `run` group collide.
    """
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[], return_hydra_config=True)
    assert cfg.hydra.help.app_name == "JEM"
    assert "python -m jem.main" in cfg.hydra.help.template


def test_run_options_share_one_schema():
    """Every ``coupled_run`` option exposes the same keys.

    ``default`` is the complete schema and the others inherit it, so any
    ``coupled_run.<key>`` can be set on the command line without a ``+``
    whichever option is composed.
    """
    key_sets = {
        option: set(composed([_group_override("coupled_run", option)]).coupled_run)
        for option in _options("coupled_run")
    }
    assert len(set(map(frozenset, key_sets.values()))) == 1, key_sets


# ---------------------------------------------------------------------------
# The YAML stays thin
# ---------------------------------------------------------------------------

def _supplied_kwargs(node: DictConfig) -> dict:
    """Return the node's keyword arguments, dropping Hydra's own and MISSING."""
    return {
        key: node[key]
        for key in node
        if not str(key).startswith("_")
        and not OmegaConf.is_missing(node, key)
    }


@pytest.mark.parametrize(
    "overrides",
    [[]]
    + [[_group_override(g, o)] for g in _groups() if g != "configuration"
       for o in _options(g)]
    + [[f"+configuration={name}"] for name in _options("configuration")],
    ids=lambda ov: "+".join(ov) or "defaults",
)
def test_config_has_no_python_defaults(overrides):
    """No config file supplies a value that equals its target's default.

    A default repeated in YAML is a second place to change it, and the two
    drift. A key earns its place in a config file only by being ``_target_``,
    a required input (``???``), or a value that differs from the Python
    default *and* is what the named option is about.
    """
    cfg = composed(overrides)
    # The atmosphere is JAX-GCM's own config, tested in JAX-GCM.
    coupled = {key: cfg[key] for key in cfg if key != "atmosphere"}
    nodes = 0
    for path, node in _target_nodes(OmegaConf.create(coupled)):
        try:
            target = get_object(node._target_)
        except ImportError as exc:
            if "veros" in str(exc).lower():
                # Optional dependency: `_target_` cannot be resolved in an
                # environment without it. The `examples` CI job has Veros.
                continue
            raise
        signature = inspect.signature(target)
        for key, value in _supplied_kwargs(node).items():
            parameter = signature.parameters.get(key)
            if parameter is None:
                assert key in RUNNER_ONLY_KEYS or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in signature.parameters.values()
                ), f"{path}.{key} is not an argument of {node._target_}"
                continue
            if parameter.default is inspect.Parameter.empty:
                continue
            assert value != parameter.default, (
                f"{path}.{key} = {value!r} repeats the default of "
                f"{node._target_}; delete it (or, for a named configuration, "
                "pick the value that makes the configuration what it is)."
            )
        nodes += 1
    # Every configuration builds at least one component; a zero here would mean
    # the walk found nothing and the test passed on an empty loop.
    assert nodes > 0


def test_named_configurations_set_non_default_values():
    """The configurations that exist to pin a value really do pin one."""
    earth = composed(["+configuration=earth-slab"])
    assert earth.ocean.params.forcing_method == "relaxation"
    # 30 days, against the model's own 60.
    assert earth.ocean.params.relaxation_time == 30 * 86400
    assert earth.land.params.tdland == 86400.0


# ---------------------------------------------------------------------------
# Packaged data paths
# ---------------------------------------------------------------------------

def test_package_data_resolvers_give_existing_files():
    """``${jcm_data:}`` / ``${jem_data:}`` resolve to files that are there."""
    cfg = composed(["+configuration=earth-slab"])
    assert Path(cfg.ocean.sst_clim_file).is_file()
    assert Path(cfg.atmosphere.terrain.file).is_file()

    mixed = composed(["+configuration=aquaplanet-slab-mixed-grid"])
    for key in ("a2o_conserve", "a2o_bilinear", "o2a_conserve", "o2a_bilinear"):
        assert Path(mixed.regrid[key]).is_file()
    assert Path(mixed.ocean.grid_file).is_file()


def test_package_data_resolver_reports_a_missing_file():
    """A typo in a packaged path fails while composing, naming the path."""
    with pytest.raises(FileNotFoundError, match="no-such-file.nc"):
        jem.config.package_data_path("jem.data", "no-such-file.nc")


# ---------------------------------------------------------------------------
# Composed options take effect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("physics", ["speedy", "held_suarez"])
@pytest.mark.parametrize("terrain", ["aquaplanet", "from_file"])
def test_atmosphere_options_take_effect(physics, terrain):
    """The re-rooted JCM groups reach the built atmosphere.

    Composing is not enough: an option that the builder never reads would
    compose exactly as well as one that works. So the model is built and each
    option read back off it -- the timestep, the physics package, the
    resolution, and whether the orography is real.
    """
    from jcm.runners import build_model

    overrides = [
        f"physics@atmosphere.physics={physics}",
        f"grid@atmosphere.grid={physics}_t31_l8",
        f"terrain@atmosphere.terrain={terrain}",
        "atmosphere.run.time_step=7",
    ]
    if terrain == "from_file":
        overrides.append(
            "atmosphere.terrain.file=${jcm_data:bc/t30/clim/terrain.nc}"
        )
    cfg = composed(overrides)
    model = build_model(cfg.atmosphere)

    # `dt_si` is a pint quantity in seconds (`jcm.model.Model`).
    assert float(model.dt_si.m) == 7 * 60

    # The physics package is the one the option named, term for term.
    expected_terms = {
        str(term._target_).rsplit(".", 1)[-1]
        for term in cfg.atmosphere.physics.terms.values()
    }
    assert {type(term).__name__ for term in model.physics.terms} == expected_terms

    # T31 on both native grids.
    assert model.coords.horizontal.total_wavenumbers == 33

    orography = np.asarray(model.dycore.terrain.orog)
    if terrain == "aquaplanet":
        assert not orography.any()
    else:
        assert np.abs(orography).max() > 0.0


@pytest.mark.parametrize(
    ("group", "option"),
    [
        (group, path.stem)
        for group in _atmosphere_groups()
        for path in sorted((JCM_CONFIG_DIR / group).glob("*.yaml"))
    ],
)
def test_atmosphere_subgroup_options_match_jcm(group, option):
    """Every option JAX-GCM ships composes under ``atmosphere``.

    Composition only: building some of them needs an optional extra or a
    network fetch of boundary data, which is JAX-GCM's business and not this
    layer's. What this catches is a JAX-GCM group whose options this package's
    re-rooting cannot reach at all.
    """
    package = _atmosphere_groups()[group]
    cfg = composed([f"{group}@{package}={option}"])
    assert OmegaConf.select(cfg, package) is not None


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def test_installed_wheel_has_config(tmp_path):
    """The YAML is package data: an installed wheel must carry it.

    `python -m jem.main` composes through ``pkg://jem.config``, so a wheel
    built without the config groups installs a package that cannot run
    anything -- and the source tree gives no warning of it.
    """
    repository_root = Path(jem.__file__).resolve().parent.parent
    # Built from a copy, because a build writes a `build/` directory beside the
    # sources and a test must not leave one in the working tree. Only what the
    # build reads is copied (the version is an attribute of `jem/__init__`, so
    # the package itself has to come along).
    source = tmp_path / "src"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(repository_root / name, source / name)
    shutil.copytree(
        repository_root / "jem",
        source / "jem",
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    # The PEP 517 hook directly rather than `pip wheel`: it is the same build
    # setuptools would run, without pip's resolver, its index lookups or its
    # lock -- none of which this test is about, and any of which can hang a CI
    # job that has no network.
    output = tmp_path / "wheel"
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from setuptools import build_meta; "
            "print(build_meta.build_wheel(sys.argv[1]))",
            str(output),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1, wheels
    names = set(zipfile.ZipFile(wheels[0]).namelist())
    assert "jem/config/config.yaml" in names
    assert "jem/config/configuration/earth-slab.yaml" in names
