"""Load a validated configuration recipe as built Python objects (issue #131).

The *recipe door*: ``jem/config/configuration/*.yaml`` stays the single recipe
store (no parallel Python dict -- that is how jax-gcm's ``benchmark.PRESETS``
drifted, see ``jcm/configurations.py``, this module's reference formulation).
``load(name)`` composes that yaml through Hydra INTERNALLY and hands back a
frozen :class:`LoadedConfiguration` of built objects -- a
:class:`~jem.base.coupler.Coupler` and the ``run_kwargs`` a caller passes to
``jem.run_chunked`` -- with Hydra/omegaconf invisible to the caller (only a
plain-dict ``.config`` is exposed for introspection). So
``jem.run_chunked(exp.coupler, **exp.run_kwargs)`` reproduces the CLI's build
and its ``coupled_run`` settings; see :func:`load`'s docstring for the small,
enumerated list of things around that build which it does NOT reproduce
(``output_dir``'s exact path, the logger level).

This door routes through the SAME :mod:`jem.runners` builders the CLI uses
(:func:`jem.runners.build_coupler` and :func:`jem.runners.build_run_kwargs`),
so a recipe means one thing whether it is composed from the shell or from a
notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

#: The Hydra config root and its configuration group -- the single recipe store.
CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIGURATION_DIR = CONFIG_DIR / "configuration"


@dataclass(frozen=True)
class LoadedConfiguration:
    """A composed configuration, built and Hydra-free.

    ``run_kwargs`` reproduces the CLI's ``coupled_run`` settings exactly
    (whatever the recipe and any ``**overrides`` resolved them to), so
    ``jem.run_chunked(coupler, **run_kwargs)`` reproduces the CLI's build and
    integration -- with the exceptions :func:`load`'s docstring enumerates
    (``output_dir`` is a fresh directory this door manufactures, not the
    CLI's Hydra-managed one; no logger-level change). ``config`` is a plain
    resolved dict for introspection -- no ``DictConfig`` leaks out, but see
    :func:`load` on why a ``+atmosphere.constants.*`` override will not show
    up in a *later* call's ``config`` even though it is still silently
    affecting that later call's build.
    """

    name: str
    coupler: Any
    run_kwargs: dict
    config: dict = field(default_factory=dict)


#: Directories :func:`_fresh_output_dir` has already handed out in this
#: process, so two calls in the same wall-clock second still disambiguate
#: even though NEITHER has written anything to disk yet at the point they are
#: handed out (see that function's docstring for why checking the filesystem
#: alone is not enough). Never cleared -- the whole point is that a name once
#: given out is never given out again for the life of the process.
_HANDED_OUT_OUTPUT_DIRS: set[str] = set()


def _fresh_output_dir() -> str:
    """Return a fresh directory mirroring Hydra's own default -- not yet created.

    :func:`jem.runners.build_run_kwargs` falls back to a literal ``"outputs"``
    when a config names no ``output_dir`` -- correct for a single ``run(cfg)``
    call from a script, but wrong for the door: a caller can build several
    experiments in one process (a notebook cell re-run, a loop over
    configurations), and a literal constant would make the second ``load()``
    land in, and resume the checkpoint of, the first's directory. This
    manufactures the door's own ``outputs/<date>/<time>``, mirroring the
    ``hydra.run.dir`` pattern ``jem/config/config.yaml`` gives the CLI
    (``outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}``), with a numeric suffix if
    that path is already spoken for.

    "Already spoken for" checks two things, not just one: the directory
    genuinely existing on disk already (an earlier PROCESS's real output, or
    this process's own after a caller actually ran something into it), and
    :data:`_HANDED_OUT_OUTPUT_DIRS` (a name this function itself already
    returned, in THIS process, whether or not anything has been written to it
    yet). The second check is the one that matters in the ordinary case:
    ``load()`` never creates the directory it names -- only `run_chunked`
    writing the first chunk does that -- so two ``load()`` calls made within
    the same wall-clock second, before either one's caller has started a run,
    would otherwise see the SAME still-nonexistent path on disk and be handed
    the identical "fresh" directory twice. Not creating the directory here is
    deliberate: a caller building several ``LoadedConfiguration``s purely for
    introspection (comparing ``.config`` across recipes, say, never
    running any of them) should not litter the working directory with empty
    ones.

    This is deliberately NOT done in :func:`jem.runners.build_run_kwargs`
    itself: a caller of ``runners.run(cfg)`` directly (a script, most of this
    repository's own tests) is a single call per process and gets the
    documented, simple ``"outputs"`` fallback; only the door needs freshness,
    since only the door is meant to be called more than once in a process.

    Returns
    -------
    str

    """
    now = datetime.now()
    base = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    candidate = base
    suffix = 1
    while str(candidate) in _HANDED_OUT_OUTPUT_DIRS or candidate.exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1
    _HANDED_OUT_OUTPUT_DIRS.add(str(candidate))
    return str(candidate)


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

    Uses ``initialize_config_module`` on ``jem.config`` -- the same package
    name every other composer in this repository resolves against
    (``jem.main``, the test suite) -- rather than ``initialize_config_dir`` on
    a physical path, so this composition step is robust to the installed
    package's files moving relative to this source file. That is narrower
    than it sounds: ``available()``/``CONFIGURATION_DIR`` below still resolve
    the recipe *names* through a plain ``Path(__file__)``-relative path, not
    through this same package machinery, so the asymmetry is real -- a
    packaging change that broke one would not necessarily break the other.

    This is NOT "the same call ``jem.main`` makes": ``jem.main`` decorates its
    entry point with ``@hydra.main(version_base=None, config_path="config")``,
    which runs Hydra's full JOB lifecycle around the composition -- creating
    Hydra's own run directory and writing its own log file into it, on top of
    whatever ``coupled_run.output_dir``/``run_chunked`` do -- while
    ``initialize_config_module`` + ``compose`` is Hydra's separate, lighter
    *compose API*, meant for exactly this (calling from a notebook/script/test
    without a job run), and produces the identical resolved ``DictConfig``
    with none of that job machinery (see ``load``'s docstring for the two
    things that difference actually amounts to for a caller). The
    ``version_base`` given here (``"1.3"``) and ``jem.main``'s (``None``,
    which ``hydra.main`` resolves to the installed version -- also 1.3.6
    here) are IDENTICAL on this installed Hydra, so nothing about how an
    override composes differs between the two calls today; where a base
    below 1.2 would matter is deprecated *defaults-list* spellings
    (``optional: true``, ``_name_``, a bare ``hydra/`` default with no
    ``override`` keyword) that no jem or jax-gcm config uses.

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


def _hydra_literal(value: Any, key: str, nested: bool = False) -> str:
    """Spell one override value as a Hydra token, or raise ``TypeError``.

    jem builds the token itself rather than using ``str(value)`` or
    ``repr(value)``, because either lets an object that controls its own
    spelling compose to something other than the value it holds: an
    ``IntEnum`` (``<Level.LOW: 1>`` in a list), numpy 2's ``np.float64(1.5)``
    (``np.float64`` subclasses ``float``), an ``int`` subclass whose
    ``__str__`` says ``2`` while its value is ``1``, or a ``list`` subclass
    whose repr differs from its contents. So only EXACT built-in types are
    accepted, at the top level and at any depth inside a list, and each is
    spelled here from its value:

    - ``None`` -> ``null``;
    - ``bool`` -> ``true`` / ``false``;
    - ``int`` -> its decimal digits;
    - ``float`` -> the shortest round-trip form (``2.5``, ``1e-10``, ``inf``);
    - ``str`` -> Hydra's own ``QuotedString`` serializer, so quotes,
      backslashes, commas and ``=`` survive (a ``${...}`` interpolation is
      still resolved; see :func:`_override_str`);
    - ``list`` (exactly ``list``) -> ``[...]``, each element recursively.

    ``test_every_accepted_list_element_type_round_trips`` and
    ``test_every_accepted_top_level_type_round_trips`` pin that each of these
    parses back to the same value.
    """
    from hydra.core.override_parser.types import Quote, QuotedString

    kind = type(value)
    if value is None:
        return "null"
    if kind is bool:
        return "true" if value else "false"
    if kind is int:
        return str(value)
    if kind is float:
        return repr(value)
    if kind is str:
        return QuotedString(text=value, quote=Quote.single).with_quotes()
    if kind is list:
        return "[" + ", ".join(
            _hydra_literal(item, key, nested=True) for item in value) + "]"
    raise TypeError(_unrepresentable_message(key, kind, nested))


def _unrepresentable_message(key: str, kind: type, nested: bool) -> str:
    """Return the ``TypeError`` message for an override that is or holds ``kind``.

    The type is named with its module (``numpy.bool``, ``pathlib.PosixPath``)
    because a bare ``__name__`` can collide with the built-in it is being
    refused in favour of: numpy 2's boolean scalar is named ``bool``.
    """
    name = (kind.__qualname__ if kind.__module__ == "builtins"
            else f"{kind.__module__}.{kind.__qualname__}")
    where = "is a list containing (at some nesting depth)" if nested else "is"
    return (
        f"load() override {key!r} {where} a value of type {name}, which jem "
        "cannot spell as a faithful Hydra override value. An override value "
        "may be only None, an exact bool, int, float or str, or an exact "
        "list of those; convert a numpy scalar, an enum member or a Path to "
        "the matching built-in first -- bool(), int(), float() or str() -- "
        "and give one dotted override per field for a structured value."
    )


def _override_str(key: str, value: Any) -> str:
    r"""One Hydra override token from a ``**overrides`` item.

    ``None`` -> ``null``. A ``str`` value is emitted as a Hydra *quoted
    string* so grammar characters (commas, ``=``, braces -- ordinary in
    paths/filenames) are carried literally rather than parsed as
    list/sweep/assignment syntax; :meth:`QuotedString.with_quotes` is Hydra's
    own serializer, so the token parses back to the exact string (embedded
    quotes/backslashes handled). This also covers a config-GROUP selection
    such as ``seaice=none`` (a bare Python string, since a caller writes
    ``load(name, seaice="none")``) -- verified to compose identically quoted
    or not, since group resolution reads the parsed override's value, not its
    source spelling. A numeric-LOOKING string (``load(name,
    **{"coupled_run.subsample": "3"})``) stays a quoted string and so composes
    to the Python string ``"3"``, unlike the CLI's own bare ``subsample=3``,
    which composes to the integer -- pass the unquoted Python ``int``/``float``
    for that.

    Quoting protects a string from Hydra's *override* grammar only: a
    ``${...}`` inside it is still an OmegaConf interpolation, resolved when
    the composed configuration is read at build time, exactly as the CLI's
    ``key='${...}'`` is. The same holds for strings inside a list. That is
    deliberate. It keeps ``load()`` equivalent to the command line, and it
    is how a config names packaged data: ``load(name, ocean="slab_relax",
    **{"ocean.sst_clim_file": "${jcm_data:bc/t30/clim/forcing.nc}"})`` is
    the Python spelling of the documented CLI override. (A caller holding
    arbitrary runtime strings can pass
    :func:`jem.config.package_data_path` results instead and never write
    ``${``.) ``\${`` escapes a literal ``${`` in a value that is read
    straight off the config (``coupled_run.*``, ``exp.config``), and a
    literal backslash directly before ``${`` must be doubled. A component
    or exchanger field cannot hold a literal ``${`` through config at all,
    on the CLI or here: ``hydra.utils.instantiate`` resolves its node a
    second time, so the escaped text is read as an interpolation again.
    Construct that component in Python instead.

    Other values are spelled by :func:`_hydra_literal` from their value,
    never by their own ``str``/``repr``, so ``coupled_run.total_time=10``
    stays the number ``10`` and ``None`` becomes ``null``. Only exact
    built-in types are accepted -- ``None``, ``bool``, ``int``, ``float``,
    ``str`` and ``list`` -- so a subclass (a numpy scalar, an enum member)
    or any other object (a ``Path``) is refused with a ``TypeError`` rather
    than trusted to spell itself; convert it with ``float()``, ``int()`` or
    ``str()`` first. A ``dict`` is refused outright with a
    ``TypeError`` naming the key: there is no single override token that
    reproduces an arbitrary nested mapping, so silently stringifying one here
    would emit a token that composes to something else entirely (or that
    Hydra's parser rejects) rather than failing where the caller can see why.
    Give one dotted override per field instead, e.g. ``load(name,
    **{"ocean.params.forcing_method": "relaxation",
    "ocean.params.relaxation_time": 1e6})`` rather than ``load(name,
    **{"ocean.params": {"forcing_method": ...}})``. A ``tuple`` is refused
    the same way, for the same reason -- Hydra's grammar has no tuple literal
    -- but its message suggests a plain ``list`` instead (which composes
    fine, see below), since "one dotted override per field" is meaningless
    for a tuple that is not naming nested config keys.

    ``None`` spells ``null``, which is also how ``**overrides`` writes a
    bare deletion: ``**{"~coupled_run.subsample": None}`` is the CLI's
    ``~coupled_run.subsample``.

    A ``list`` follows the same rule at every depth: its elements compose to
    exactly the values given -- ``None`` to ``null``, strings through Hydra's
    own quoting -- and any other element, or a ``list`` subclass, is refused
    with a ``TypeError`` naming it.

    Parameters
    ----------
    key : str
        The (possibly dotted) override key, as a caller's ``**overrides``
        item's key.
    value : Any
        The override's value.

    Returns
    -------
    str

    Raises
    ------
    TypeError
        If ``value`` is not ``None``, an exact ``bool``/``int``/``float``/
        ``str``, or an exact ``list`` holding only those and nested exact
        lists at any depth.

    """
    if isinstance(value, dict):
        raise TypeError(
            f"load() override {key!r} is a dict, which has no single Hydra "
            "override token that reproduces it faithfully; give one dotted "
            f"override per field instead, e.g. "
            f"**{{{key + '.<field>'!r}: <value>, ...}} rather than "
            f"**{{{key!r}: {{'<field>': <value>, ...}}}}."
        )
    if isinstance(value, tuple):
        raise TypeError(
            f"load() override {key!r} is a tuple, which has no single Hydra "
            "override token that reproduces it faithfully; pass a list "
            f"instead, e.g. **{{{key!r}: [<value>, ...]}}."
        )
    return f"{key}={_hydra_literal(value, key)}"


def load(name: str, **overrides: Any) -> LoadedConfiguration:
    """Compose a named configuration recipe and return its built objects.

    ``name`` is any :func:`available` key (a
    ``jem/config/configuration/*.yaml`` stem). ``**overrides`` is the optional
    escape hatch: Hydra overrides passed straight into compose -- a plain
    value override (``load("earth-slab", **{"coupled_run.total_time": "60
    days"})`` -- the recipe's chunk stays its default 30 days, so this has to
    be a multiple of that, not the CLI's own arbitrary "10 days"; the dict key
    carries the dots a Python kwarg cannot) or a config-group selection
    (``load("aquaplanet-slab", seaice="none")``, exactly the CLI's
    ``seaice=none``). The returned :class:`LoadedConfiguration` is Hydra-free:
    ``coupler`` is built and ``config`` is a plain resolved dict.

    What ``run_kwargs`` reproduces of the CLI, and what it does not
    -----------------------------------------------------------------
    ``jem.run_chunked(exp.coupler, **exp.run_kwargs)`` reproduces the CLI's
    build and every ``coupled_run`` setting the recipe (plus any
    ``**overrides``) resolved to. The CLI does NOT change the process's
    working directory (``python -m jem.main +configuration=... --cfg hydra``
    shows ``chdir: null``, and with Hydra's resolved ``version_base`` of 1.3
    a ``null`` ``chdir`` means no chdir at all), so that is not something to
    enumerate here. Two things the CLI does around the build ARE deliberately
    NOT reproduced:

    - **``output_dir``.** When the recipe names none (the common case --
      every shipped configuration leaves it ``null``), the CLI gets a fresh
      Hydra-managed ``outputs/<date>/<time>`` job directory. This function
      cannot use that (there is no Hydra *job* here, only a *composition* --
      see :func:`_compose`'s docstring), so it manufactures its OWN fresh
      ``outputs/<date>/<time>`` directory instead (:func:`_fresh_output_dir`),
      unique per call: two successive ``load()`` calls in one process land in
      two different directories, and neither is the other's checkpoint
      location. This is intentionally different from
      :func:`jem.runners.build_run_kwargs`'s own non-door fallback (a literal
      ``"outputs"``, correct for a script's single ``runners.run(cfg)``
      call) -- only the door needs per-call freshness, since only the door is
      meant to be called more than once in a process. An explicit
      ``coupled_run.output_dir`` override (or one set in the recipe itself)
      is honoured exactly, with no substitution.
    - **The ``jem`` logger's level.** ``jem.main`` sets it from
      ``cfg.coupled_run.log_level`` (``logging.getLogger("jem").setLevel(...)``,
      see its module docstring); ``load()`` touches no logger.

    Constants overrides are process-global and persist after ``load()``
    ----------------------------------------------------------------------
    A ``+atmosphere.constants.*`` override reaches this door's build exactly
    as it reaches the CLI's -- see the paragraph below on why no separate step
    is needed for that. But :mod:`jcm.constants` is a **process-global
    singleton** (``jcm.runners.apply_constants_overrides`` calls
    ``jcm.constants.set_constants(...)``, which REBINDS the module-global
    ``physical_constants`` name to a new ``PhysicalConstants`` built by its
    own ``_replace`` -- it is a ``NamedTuple``, so nothing is mutated in
    place, but the
    process-wide effect is the same: every subsequent attribute access,
    ``c.grav`` included, now reads the new one), and neither this door nor
    jax-gcm's own (:func:`jcm.configurations.load`) resets or restores it
    afterwards. So:

    - the override OUTLIVES the ``load()`` call that applied it, and silently
      applies to every model built in the same process afterwards -- another
      ``load()`` of a DIFFERENT recipe with no constants override of its own
      still sees the earlier override, and its own
      ``.config["atmosphere"]["constants"]`` will show the empty ``{}`` a
      configuration with no override of its own composes to (not the values
      the live singleton actually holds) even though the live singleton (and
      so the model that ``load()`` call just built) is running with the
      earlier, overridden ones;
    - this is deliberate, not an oversight to fix here: restoring the
      singleton after ``load()`` returns would silently change an
      already-built, already-traced model's physics parameters out from under
      whatever holds a reference to it, and resetting it *before* building
      would silently clobber a caller's own deliberate
      ``jcm.constants.set_constants(...)`` made earlier in the process. Both
      are worse than the leak.

    The practical rule: use at most one constants configuration per process.
    A script or notebook that needs several different overrides should run
    each in its own process (or explicitly save/restore
    ``jcm.constants.physical_constants`` itself, understanding that this
    changes the physics of any model already built with the old value).

    Other pre-build side effects of :func:`jem.runners.run`
    ---------------------------------------------------------
    Unlike jax-gcm's own door (:func:`jcm.configurations.load`), which has to
    separately call ``apply_constants_overrides`` before building the model,
    this door needs no equivalent step for THAT: every build-relevant side
    effect ``run()`` relies on is already inside
    :func:`jem.runners.build_coupler` itself, because ``build_coupler`` calls
    :func:`jem.runners.build_atmosphere`, which applies jax-gcm's
    ``+atmosphere.constants.*`` overrides and its config-trap warnings before
    constructing the model (see ``build_atmosphere``'s own docstring). So
    calling ``build_coupler(cfg)`` here, exactly as ``run()`` does, already
    reproduces those effects with no separate call needed. What ``run()``
    does that this door does NOT reproduce is enumerated above (``output_dir``,
    the logger level) plus logging the composed config, which is a
    convenience with no bearing on the objects built and which this door's
    caller can reproduce for themselves from ``exp.config`` if wanted.

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
    TypeError
        If an override value is not ``None``, an exact
        ``bool``/``int``/``float``/``str``, or an exact ``list`` holding only
        those and nested exact lists at any depth (see
        :func:`_override_str`).

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
    # A fresh directory per call, UNLESS the recipe (or an override) named an
    # explicit one -- checked on the composed config itself, before
    # `build_run_kwargs` folded a `null` into its own "outputs" fallback, so
    # this cannot mistake an explicit `coupled_run.output_dir="outputs"` for
    # the unset case. See `load`'s own docstring for why this differs from
    # `build_run_kwargs`'s non-door fallback.
    if cfg.coupled_run.get("output_dir") is None:
        run_kwargs["output_dir"] = _fresh_output_dir()
    # Plain resolved dict for introspection; a still-unfilled ``???`` key stays
    # a string rather than raising on this read-only copy. `cfg` is always a
    # mapping node (the whole composed config), so `to_container` always
    # returns a dict here -- the broader union in its return type covers a
    # list/leaf-valued *sub*-node, which this call never passes.
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    assert isinstance(config, dict)
    return LoadedConfiguration(name=name, coupler=coupler,
                               run_kwargs=run_kwargs, config=config)
