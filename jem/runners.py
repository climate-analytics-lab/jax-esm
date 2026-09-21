"""From a composed Hydra config to a built coupled model, and a run of it.

This is the only module that reads the configuration. Everything it builds is
an ordinary Python object that could have been built by hand -- the point of
keeping it in one place is that the *config* layer stays a thin wiring layer:

- the YAML says which class, which required input files and which non-default
  choices define a named configuration, and nothing else;
- the objects that come from *other* objects -- a slab component's grid, the
  regridders an exchange needs, the coupling timestep -- are injected here,
  because no configuration file can name a live Python object;
- every physics default stays on the Python class that owns it.

Nothing in this module names a component's parameters. It knows that
``cfg.ocean`` builds something and that the something needs a grid; it does
not know what a mixed layer is. ``test_runners_has_no_component_kwargs``
enforces exactly that, by failing if any slab parameter's field name appears
in this source at all -- so a new component is configured by adding a group
file, never by adding a branch here.

The one table is :data:`GROUP_TO_NAME`: which config group becomes which
component name in the coupler, and so which rows of
:func:`jem.exchangers.default_exchanges` apply.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

import hydra.utils
import jax_datetime as jdt
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Importing the config package registers the ${jcm_data:}/${jem_data:}
# resolvers. A config composed elsewhere (`jem.main`, a test) has already done
# it, but a caller that hands this module a config built by hand has not, and
# the resolvers must exist before a value that uses one is read.
import jem.config  # noqa: F401
from jem import driver
from jem.base.coupler import Coupler
from jem.components.jcm import JCMComponent
from jem.components.slab import SlabGrid
from jem.exchangers import (
    DEFAULT_EXCHANGER_NAME,
    Exchange,
    default_exchangers,
)

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400

#: Which config group builds which component, under which name. The names are
#: the ones :func:`jem.exchangers.default_exchanges` wires; the atmosphere is
#: not here because it is not optional and carries its own name
#: (:class:`~jem.components.jcm.component.JCMComponent` is ``"atm"``).
GROUP_TO_NAME = {"ocean": "ocn", "land": "lnd", "seaice": "seaice"}

#: Keys a component's config node may carry that are **not** constructor
#: arguments: they describe the grid the component is to be built on, which is
#: an object the runner makes and injects. They are read by :func:`build_grid`
#: and removed before the node is instantiated.
RUNNER_ONLY_KEYS = ("grid_file", "land_fraction_file")

#: The keyword :func:`build_coupler` injects a built grid under, and the
#: parameter a component must declare to be given one. See
#: :func:`_accepts_grid`.
GRID_KEYWORD = "grid"

#: How the named regridders of the ``regrid`` group map onto the keys
#: :func:`jem.exchangers.default_exchanges` asks for. A flux or an areal
#: fraction is mapped conservatively so its budget survives the interface; a
#: state such as the sea surface temperature is mapped bilinearly, which does
#: not leave a conservative map's staircase in a smooth field. Both
#: vocabularies reach an exchange spec, so a hand-written coupling table in
#: YAML may name either.
REGRID_ROLES = {
    "a2o_flux": "a2o_conserve",
    "a2o_state": "a2o_bilinear",
    "o2a_flux": "o2a_conserve",
    "o2a_state": "o2a_bilinear",
}

#: Land-fraction variable name in the packaged mask files, and the fallback
#: rule when a file uses another name. The files are ECMWF-derived, where the
#: land-sea mask is ``lsm``.
LAND_FRACTION_VARIABLE = "lsm"


def build_atmosphere(cfg: DictConfig) -> JCMComponent:
    """Build the atmosphere component from ``cfg.atmosphere``.

    ``cfg.atmosphere`` is a jax-gcm config, group for group, so jax-gcm's own
    builders are what read it -- JAX-ESM never reimplements the atmosphere's
    wiring, and an option that works in ``python -m jcm.main`` works here.

    Physical-constant overrides are applied **first**, process-globally, as
    jax-gcm's own CLI does: the dynamical core reads the live
    :mod:`jcm.constants` singleton while it is being constructed. Since the
    surface components read the same singleton, and they are built after this,
    one ``+atmosphere.constants.grav=...`` moves the whole Earth system rather
    than only the atmosphere.

    Nothing here touches ``cfg.atmosphere.run``. The coupled run's length and
    output interval are the *coupler's* (``cfg.coupled_run``), and
    :class:`~jem.components.jcm.component.JCMComponent` integrates exactly one
    coupling interval per call with its own explicit arguments, so the
    atmosphere's ``run`` group governs only what it governs in an uncoupled
    run -- above all its timestep.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The whole composed config; only ``cfg.atmosphere`` is read.

    Returns
    -------
    jem.components.jcm.component.JCMComponent

    """
    from jcm.runners import (
        apply_constants_overrides,
        build_forcing,
        build_model,
        warn_on_config_traps,
    )

    atmosphere = cfg.atmosphere
    apply_constants_overrides(atmosphere)
    model = build_model(atmosphere)
    dycore = getattr(model, "dycore", None)
    forcing = build_forcing(atmosphere, model.coords, dycore=dycore)
    # jax-gcm's own cross-validation of combinations that run but mislead. It
    # only ever warns, and a JAX-ESM user should hear the same warnings a
    # jax-gcm user does.
    warn_on_config_traps(
        atmosphere, model.physics, forcing, coords=model.coords, dycore=dycore
    )
    return JCMComponent(model, forcing=forcing)


def build_grid(node: Any, atm: JCMComponent) -> SlabGrid:
    """Build the grid a surface component runs on.

    Two cases, decided by whether the node names a grid file:

    - **no ``grid_file``** -- the component shares the atmosphere's horizontal
      grid and its land fraction, which is the single-grid configuration every
      basic example uses. Taking both from the built atmosphere is what stops a
      surface component ending up on a grid that merely resembles it.
    - **``grid_file``** -- a SCRIP grid of the component's own (a
      displaced-pole ocean grid). Its land fraction comes from
      ``land_fraction_file`` if one is given, and otherwise from the SCRIP
      file's own integer mask.

    Parameters
    ----------
    node : omegaconf.DictConfig
        One component's config node.
    atm : jem.components.jcm.component.JCMComponent
        The already-built atmosphere, the source of the shared grid.

    Returns
    -------
    jem.components.slab.grid.SlabGrid

    """
    grid_file = _runner_only_value(node, "grid_file")
    if grid_file is None:
        return SlabGrid.from_coords(
            atm.model.coords.horizontal, atm.model.terrain.fmask
        )
    mask_file = _runner_only_value(node, "land_fraction_file")
    # 0.5 rather than `from_scrip`'s own default, so that a cell counts as
    # land on a SCRIP grid under the same rule as on the atmosphere's grid
    # (`from_coords` defaults to 0.5); a coupled run that applied two
    # different land/ocean splits either side of the interface would be
    # exchanging fluxes with cells the other side thinks are dry.
    return SlabGrid.from_scrip(
        str(grid_file),
        fractional_mask=None if mask_file is None else _land_fraction(mask_file),
        threshold=0.5,
    )


def build_component(node: Any, **injected: Any) -> Any:
    """Instantiate one component (or None) from its config node.

    ``hydra.utils.instantiate`` on everything but the runner-only keys, with
    the objects this module made passed as keyword arguments. A ``None`` node
    -- what ``ocean=none`` composes to -- builds nothing, so a configuration
    drops a component by selecting an option rather than by a flag here.

    Parameters
    ----------
    node : omegaconf.DictConfig or None
        The component's config node.
    **injected
        Objects the config cannot name, passed to the constructor.

    Returns
    -------
    Any or None
        Whatever ``_target_`` builds.

    Raises
    ------
    omegaconf.MissingMandatoryValue
        If the node leaves a required input (``???``) unset. The message names
        the full config key, so the fix is the override to add.

    """
    if node is None:
        return None
    # Resolved to plain Python here, rather than instantiated straight from
    # the config node, so that the runner-only keys can be dropped without
    # mutating the composed config a caller may still be reading (and
    # `python -m jem.main --cfg job` still prints what the run was given).
    config = OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    if not isinstance(config, dict):
        raise TypeError(
            f"A component config node must be a mapping with a `_target_`; got "
            f"{type(config).__name__}."
        )
    for key in RUNNER_ONLY_KEYS:
        config.pop(key, None)
    # `_convert_="object"` for two reasons: a constructor is handed plain
    # Python values rather than OmegaConf containers, which behave differently
    # under JAX and numpy; and an *injected* object survives as itself. The
    # grid is a dataclass, which OmegaConf recognises as a structured config
    # and `_convert_="all"` would hand over as a bare dict.
    return hydra.utils.instantiate(config, **injected, _convert_="object")


def build_regridders(cfg: DictConfig) -> dict[str, Callable[[Any], Any]]:
    """Build the named regridders, and their role aliases.

    ``regrid=same_grid`` composes to ``None`` -- every component is on one
    grid and an exchange copies fields straight across -- and gives an empty
    mapping. ``regrid=esmf`` builds a
    :class:`jem.regrid.ESMFRegridders` from its weight files.

    The result carries each regridder twice: under the name the config gave it
    (``o2a_bilinear``) and under the exchange role it plays
    (:data:`REGRID_ROLES`, ``o2a_state``). A hand-written coupling table can
    then name whichever reads better, and the default table -- which asks by
    role -- needs no naming convention imposed on the config.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The whole composed config; only ``cfg.regrid`` is read.

    Returns
    -------
    dict[str, Callable]

    """
    regridders = build_component(cfg.get("regrid"))
    if regridders is None:
        return {}
    named = dict(regridders)
    roles = {
        role: named[name] for role, name in REGRID_ROLES.items() if name in named
    }
    return {**named, **roles}


def build_exchangers(
    cfg: DictConfig,
    components: Mapping[str, Any],
    regridders: Mapping[str, Callable[[Any], Any]],
) -> dict[str, Any]:
    """Build the exchangers the coupler runs, from ``cfg.coupling``.

    Three spellings, in precedence order:

    - ``coupling.exchanger`` -- an importable dotted path to a single
      exchanger function, used when the coupling is something a table cannot
      express (a wind stress rotated onto another grid, a case-specific
      freshwater budget). It replaces the table entirely.
    - ``coupling.exchangers`` -- an explicit coupling table in YAML, a list of
      ``{src, dst, regrid}`` mappings.
    - neither (both ``null``, the default) --
      :func:`jem.exchangers.default_exchangers` for whichever components were
      built, which is where the standard wiring is written down once.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The whole composed config; only ``cfg.coupling`` is read.
    components : Mapping[str, Any]
        The built components, by the name they will be registered under.
    regridders : Mapping[str, Callable]
        What :func:`build_regridders` returned.

    Returns
    -------
    dict[str, jem.base.component.Exchanger]

    Raises
    ------
    ValueError
        If both ``exchanger`` and ``exchangers`` are set; they are two answers
        to one question, and guessing which was meant is worse than asking.

    """
    coupling = cfg.coupling
    path = coupling.get("exchanger")
    specs = coupling.get("exchangers")
    if path and specs:
        raise ValueError(
            f"coupling.exchanger ({path!r}) and coupling.exchangers are both "
            "set. `exchanger` names one Python function used INSTEAD of the "
            "table, so setting both leaves it undecided which couples the run; "
            "clear one (coupling.exchanger=null or coupling.exchangers=null)."
        )
    if path:
        logger.info("Coupling through the exchanger %s.", path)
        return {DEFAULT_EXCHANGER_NAME: hydra.utils.get_method(path)}
    if specs is None:
        roles = {
            role: regridders[role] for role in REGRID_ROLES if role in regridders
        }
        return default_exchangers(components, regrid=roles)
    table = OmegaConf.to_container(specs, resolve=True, throw_on_missing=True)
    if not isinstance(table, list):
        raise TypeError(
            "coupling.exchangers is the coupling table: a list of "
            "`{src, dst, regrid}` mappings, or null for the default. Got a "
            f"{type(table).__name__}."
        )
    return {DEFAULT_EXCHANGER_NAME: Exchange(table, regridders)}


def build_coupler(cfg: DictConfig) -> Coupler:
    """Build the whole coupled model from a composed config.

    The atmosphere first (it owns the clock every other component is checked
    against, and the grid they default to), then each surface group that is
    not ``none``, then the exchange and the coupler itself.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        A config composed from ``jem/config/config.yaml``.

    Returns
    -------
    jem.base.coupler.Coupler

    """
    atm = build_atmosphere(cfg)
    components: dict[str, Any] = {atm.name: atm}
    for group, name in GROUP_TO_NAME.items():
        node = cfg.get(group)
        if node is None:
            logger.debug("No %s component (%s=none).", name, group)
            continue
        components[name] = build_component(node, **_injected_grid(node, atm))

    regridders = build_regridders(cfg)
    exchangers = build_exchangers(cfg, components, regridders)
    workflow = cfg.coupling.get("workflow")
    coupler = Coupler(
        components,
        exchangers,
        coupling_timestep=_coupling_timestep(cfg, atm.model.calendar),
        start_date=atm.model.start_date,
        calendar=atm.model.calendar,
        workflow=None if workflow is None else list(workflow),
    )
    _validate_exchangers(coupler)
    logger.info("Built %r", coupler)
    return coupler


def run(cfg: DictConfig) -> driver.RunResult:
    """Build the coupled model from ``cfg`` and run it.

    ``cfg.coupled_run`` is the keyword arguments of
    :func:`jem.driver.run_chunked`, one for one, minus ``log_level`` --- which
    belongs with the run's other command-line settings but configures
    :mod:`jem.main`'s logger rather than the run. No default is repeated here:
    every one of them lives on ``run_chunked``.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        A config composed from ``jem/config/config.yaml``.

    Returns
    -------
    jem.driver.RunResult

    """
    logger.info("Composed config:\n%s", OmegaConf.to_yaml(cfg))
    kwargs = OmegaConf.to_container(cfg.coupled_run, resolve=True, throw_on_missing=True)
    if not isinstance(kwargs, dict):
        raise TypeError(
            f"cfg.coupled_run must be a mapping of run settings; got "
            f"{type(kwargs).__name__}."
        )
    settings = {str(key): value for key, value in kwargs.items()}
    settings.pop("log_level", None)
    if settings.get("output_dir") is None:
        settings["output_dir"] = _default_output_dir()
    logger.info("Writing output to %s", settings["output_dir"])
    return driver.run_chunked(build_coupler(cfg), **settings)


# -- the objects a config cannot name ---------------------------------------


def _injected_grid(node: Any, atm: JCMComponent) -> dict[str, SlabGrid]:
    """Return the ``grid=`` keyword for ``node``, or nothing at all.

    A component that brings its own grid -- an ocean GCM with its own
    bathymetry -- neither takes one nor needs one built, so no grid is made
    for it either: :func:`build_grid` would read the atmosphere's horizontal
    grid and its land fraction to produce something nothing would use.

    Such a node carrying a :data:`RUNNER_ONLY_KEYS` key is refused rather than
    ignored. Those keys describe the grid this function would have built, and
    :func:`build_component` drops them before instantiating, so a
    ``ocean.grid_file=...`` on a Veros node would otherwise reach nothing at
    all and the run would proceed on a grid the user believes they replaced.

    Parameters
    ----------
    node : omegaconf.DictConfig
        One component's config node.
    atm : jem.components.jcm.component.JCMComponent
        The already-built atmosphere, for :func:`build_grid`.

    Returns
    -------
    dict
        ``{"grid": SlabGrid}``, or ``{}``.

    Raises
    ------
    ValueError
        If the node describes a grid for a component that takes none.

    """
    if not _accepts_grid(node):
        unusable = [
            key for key in RUNNER_ONLY_KEYS
            if _runner_only_value(node, key) is not None
        ]
        if unusable:
            raise ValueError(
                f"{node.get('_target_')!r} does not take a grid, so "
                f"{unusable!r} would be read by nothing: these keys describe "
                "the grid the runner builds for a component that is built ON "
                "one (the slab models). A component that brings its own grid "
                "takes its geometry through its own constructor arguments -- "
                "for a Veros ocean, the keys its setup factory declares."
            )
        logger.debug(
            "%s takes no grid, so none is built for it.", node.get("_target_")
        )
        return {}
    return {GRID_KEYWORD: build_grid(node, atm)}


def _accepts_grid(node: Any) -> bool:
    """Return whether the thing ``node`` builds takes a ``grid=`` argument.

    The slab components are built **on** a grid, which is a live object no
    configuration file can name, so the runner makes one and injects it. A
    component that brings its own grid -- an ocean GCM with its own
    bathymetry and its own land-sea mask -- does not take one, and handing it
    a grid anyway is not a harmless extra: ``VerosComponent.from_setup``
    passes every keyword it does not recognise on to the Veros setup factory,
    which rejects an unknown ``grid``.

    So the question is asked of the target itself rather than answered by a
    list of component names here: the ``_target_`` is resolved to the class
    or function it names and its signature inspected for an explicit ``grid``
    parameter. A ``**kwargs`` catch-all does not count -- that is exactly the
    case that swallows the keyword and fails somewhere else.

    A target that cannot be resolved (a typo, or an optional dependency that
    is not installed) answers ``False``: no grid is built, and
    ``hydra.utils.instantiate`` then raises the real import error, which says
    far more than anything this could invent. Only the errors a *lookup*
    raises are swallowed for that -- ``ImportError`` and the ``ValueError``
    Hydra gives an invalid dotstring. Anything else means the lookup itself is
    broken rather than the target missing, and it must not be mistaken for
    "this component takes no grid": that answer is indistinguishable from the
    truthful one, and it would silently deny every slab component the grid it
    requires, leaving a run to die inside ``instantiate`` with a message
    naming nothing.

    Parameters
    ----------
    node : omegaconf.DictConfig or None
        One component's config node.

    Returns
    -------
    bool

    """
    target = None if node is None else node.get("_target_")
    if target is None:
        return False
    try:
        # `get_object` rather than `get_class`/`get_method`, because a
        # `_target_` is legitimately either -- `jem.components.SlabOceanModel`
        # is a class, `jem.components.VerosComponent.from_setup` a classmethod
        # -- and the typed lookups reject (and log an error about) the other.
        resolved = hydra.utils.get_object(str(target))
    except (ImportError, ValueError):
        logger.debug("Could not resolve _target_ %r; no grid injected.", target)
        return False
    try:
        signature = inspect.signature(resolved)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(GRID_KEYWORD)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    )


def _runner_only_value(node: Any, key: str) -> Any:
    """Return one of the :data:`RUNNER_ONLY_KEYS` from a node, or None."""
    if node is None or key not in node:
        return None
    return node[key]


def _land_fraction(path: str) -> Any:
    """Read a land fraction from a mask file, onto ``SlabGrid``'s layout.

    The packaged mask files hold a single ``(time, lat, lon)`` field: the
    first (and only) record is taken and transposed to the ``(n_lon, n_lat)``
    layout every grid and component in JAX-ESM uses. A file whose variable is
    named something other than :data:`LAND_FRACTION_VARIABLE` is accepted when
    it holds exactly one variable, because the name is the file's convention
    rather than anything the coupler decides.
    """
    dataset = xr.open_dataset(path)
    if LAND_FRACTION_VARIABLE in dataset:
        field = dataset[LAND_FRACTION_VARIABLE]
    elif len(dataset.data_vars) == 1:
        (only,) = dataset.data_vars
        field = dataset[only]
    else:
        raise ValueError(
            f"{path!r} has no {LAND_FRACTION_VARIABLE!r} variable and holds "
            f"{sorted(map(str, dataset.data_vars))!r}, so which of them is the "
            "land fraction is ambiguous."
        )
    return field.to_numpy()[0].transpose()


def _coupling_timestep(cfg: DictConfig, calendar: str) -> jdt.Timedelta:
    """Return ``cfg.coupling.timestep`` as the coupler's ``jdt.Timedelta``.

    The config spells the interval the way a run length is spelled ("1 day",
    "12 hours"), on the atmosphere's calendar; the coupler holds whole
    seconds.
    """
    from jcm.date import parse_duration_days

    spelling = cfg.coupling.timestep
    seconds = float(parse_duration_days(spelling, calendar)) * SECONDS_PER_DAY
    if abs(seconds - round(seconds)) > 1e-6 or round(seconds) < 1:
        raise ValueError(
            f"coupling.timestep={spelling!r} is {seconds:g} s, and a coupling "
            "timestep is a whole positive number of seconds (jax-esm#110)."
        )
    return jdt.to_timedelta(int(round(seconds)), "second")


def _validate_exchangers(coupler: Coupler) -> None:
    """Check every declarative exchange against the model's real carries.

    :meth:`jem.exchangers.Exchange.validate` turns a mistyped component,
    section, field or regridder into an error naming the spec, before the
    coupled step is traced. It needs real carries, so this pays for one
    ``coupler.initialize()`` -- the same call the run makes, whose compiled
    pieces the run then reuses -- to catch a broken coupling table before a
    model is integrated at all. Exchangers that are plain functions have
    nothing to check and are skipped.
    """
    checkable = [
        exchanger
        for exchanger in coupler.exchangers.values()
        if hasattr(exchanger, "validate")
    ]
    if not checkable:
        return
    carry = coupler.initialize()
    for exchanger in checkable:
        exchanger.validate(carry.components)


def _default_output_dir() -> str:
    """Return the directory a run writes into when the config names none.

    Hydra's run directory, so that each run is self-contained and two runs
    cannot overwrite each other's files -- and ``outputs`` when this is not a
    Hydra job at all (a script or a notebook calling :func:`run` directly).
    """
    try:
        return str(HydraConfig.get().runtime.output_dir)
    except ValueError:
        return "outputs"
