"""Load a validated configuration recipe as built Python objects (issue #131).

The *recipe door*: ``jem/config/configuration/*.yaml`` stays the single recipe
store (no parallel Python dict -- that is how jax-gcm's ``benchmark.PRESETS``
drifted, see ``jcm/configurations.py``, this module's reference formulation).
``load(name)`` composes that yaml through Hydra INTERNALLY and hands back a
frozen :class:`LoadedConfiguration` of built objects -- a
:class:`~jem.base.coupler.Coupler` and the ``run_kwargs`` a caller passes to
``jem.run_chunked`` -- with Hydra/omegaconf invisible to the caller (only a
plain-dict ``.config`` is exposed for introspection). So
``jem.run_chunked(exp.coupler, **exp.run_kwargs)`` reproduces
``python -m jem.main +configuration=<name>``'s single integration.

This door routes through the SAME :mod:`jem.runners` builders the CLI uses
(:func:`jem.runners.build_coupler` and :func:`jem.runners.build_run_kwargs`),
so a recipe means one thing whether it is composed from the shell or from a
notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The Hydra config root and its configuration group -- the single recipe store.
CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIGURATION_DIR = CONFIG_DIR / "configuration"


@dataclass(frozen=True)
class LoadedConfiguration:
    """A composed configuration, built and Hydra-free.

    ``run_kwargs`` is exactly what :func:`jem.runners.build_run_kwargs` would
    hand ``python -m jem.main`` for the same composed config, so
    ``jem.run_chunked(coupler, **run_kwargs)`` reproduces the CLI's
    integration. ``config`` is a plain resolved dict for introspection -- no
    ``DictConfig`` leaks out.
    """

    name: str
    coupler: Any
    run_kwargs: dict
    config: dict = field(default_factory=dict)


def _summary(path: Path) -> str:
    """First human comment line of a configuration yaml (its one-line summary).

    The yamls open with ``# @package _global_``, a blank ``#``, then the
    summary; return that first real comment, or ``""`` if none is present.
    """
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s.startswith("#"):
            break
        body = s[1:].strip()
        if not body or body.startswith("@package"):
            continue
        return body
    return ""


def available() -> dict[str, str]:
    """Map each configuration name to its one-line summary (sorted by name)."""
    return {p.stem: _summary(p)
            for p in sorted(CONFIGURATION_DIR.glob("*.yaml"))}


def _compose(name: str, overrides: list[str]):
    """Compose ``+configuration=<name>`` (+ dotted overrides) against the root.

    Uses ``initialize_config_module`` on ``jem.config`` -- the same call every
    other composer in this repository makes (``jem.main``, the test suite) --
    rather than ``initialize_config_dir`` on a physical path: jem's own
    convention is the package form, so this stays composable the same way
    even if the installed package's files move relative to this source file.

    ``initialize_config_module`` clears the global Hydra on exit, and we also
    clear a pre-existing one up front, so ``load`` is safe to call repeatedly.
    To not leave a *host* application's own initialised Hydra cleared, we
    snapshot all Hydra singletons first and restore them in a ``finally`` --
    the host's context still composes after ``load`` returns.
    """
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.singleton import Singleton

    saved = Singleton.get_state()
    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_module(config_module="jem.config", version_base="1.3"):
            return compose(config_name="config",
                           overrides=[f"+configuration={name}", *overrides])
    finally:
        Singleton.set_state(saved)


def _override_str(key: str, value: Any) -> str:
    """One Hydra override token from a ``**overrides`` item.

    ``None`` -> ``null``. A ``str`` value is emitted as a Hydra *quoted
    string* so grammar characters (commas, ``=``, braces -- ordinary in
    paths/filenames) are carried literally rather than parsed as
    list/sweep/assignment syntax; :meth:`QuotedString.with_quotes` is Hydra's
    own serializer, so the token parses back to the exact string (embedded
    quotes/backslashes handled). This also covers a config-GROUP selection
    such as ``seaice=none`` (a bare Python string, since a caller writes
    ``load(name, seaice="none")``) -- verified to compose identically quoted
    or not, since group resolution reads the parsed override's value, not its
    source spelling. Non-string scalars pass through unquoted so
    ``coupled_run.total_time=10`` stays the number ``10`` (and a Python
    list/dict keeps its native override meaning).
    """
    from hydra.core.override_parser.types import Quote, QuotedString

    if value is None:
        return f"{key}=null"
    if isinstance(value, str):
        return f"{key}={QuotedString(text=value, quote=Quote.single).with_quotes()}"
    return f"{key}={value}"


def load(name: str, **overrides: Any) -> LoadedConfiguration:
    """Compose a named configuration recipe and return its built objects.

    ``name`` is any :func:`available` key (a
    ``jem/config/configuration/*.yaml`` stem). ``**overrides`` is the optional
    escape hatch: Hydra overrides passed straight into compose -- a plain
    value override (``load("earth-slab", **{"coupled_run.total_time": 10})``,
    where the dict key carries the dots a Python kwarg cannot) or a
    config-group selection (``load("aquaplanet-slab", seaice="none")``,
    exactly the CLI's ``seaice=none``). The returned
    :class:`LoadedConfiguration` is Hydra-free: ``coupler`` is built and
    ``config`` is a plain resolved dict.

    ``jem.run_chunked(exp.coupler, **exp.run_kwargs)`` reproduces the CLI's
    single integration for the recipe.

    Other pre-build side effects of :func:`jem.runners.run`
    ---------------------------------------------------------
    Unlike jax-gcm's own door (:func:`jcm.configurations.load`), which has to
    separately call ``apply_constants_overrides`` before building the model,
    this door needs no equivalent step: every build-relevant side effect
    ``run()`` relies on is already inside :func:`jem.runners.build_coupler`
    itself, because ``build_coupler`` calls :func:`jem.runners.build_atmosphere`,
    which applies jax-gcm's ``+atmosphere.constants.*`` overrides and its
    config-trap warnings before constructing the model (see
    ``build_atmosphere``'s own docstring). So calling ``build_coupler(cfg)``
    here, exactly as ``run()`` does, already reproduces those effects with no
    separate call needed.

    The ONE thing ``run()`` does before ``build_coupler(cfg)`` that this door
    does not repeat is logging the composed config -- a convenience with no
    bearing on the objects built, and one this door's caller can reproduce
    for themselves from ``exp.config`` if wanted.
    ``run()``'s other work (choosing ``output_dir``, handing everything to
    ``driver.run_chunked``) is exactly what building ``run_kwargs`` and
    returning them, rather than starting the run, replaces -- the run itself
    is left to the caller, same as the CLI leaves it to
    ``driver.run_chunked``.

    Parameters
    ----------
    name : str
        A shipped configuration's name.
    **overrides
        Hydra overrides, dotted keys and config-group selections alike.

    Returns
    -------
    LoadedConfiguration

    Raises
    ------
    ValueError
        If ``name`` is not one of :func:`available`.

    """
    from omegaconf import OmegaConf

    from jem import runners

    names = available()
    if name not in names:
        raise ValueError(
            f"Unknown configuration {name!r}. Available: {sorted(names)}")

    overrides_list = [_override_str(k, v) for k, v in overrides.items()]
    cfg = _compose(name, overrides_list)

    coupler = runners.build_coupler(cfg)
    run_kwargs = runners.build_run_kwargs(cfg)
    # Plain resolved dict for introspection; a still-unfilled ``???`` key stays
    # a string rather than raising on this read-only copy. `cfg` is always a
    # mapping node (the whole composed config), so `to_container` always
    # returns a dict here -- the broader union in its return type covers a
    # list/leaf-valued *sub*-node, which this call never passes.
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    assert isinstance(config, dict)
    return LoadedConfiguration(name=name, coupler=coupler,
                               run_kwargs=run_kwargs, config=config)
