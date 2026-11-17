"""JAX-ESM's Hydra configuration package, and the resolvers its YAML uses.

The YAML in this package is *wiring only*: it names the Python objects a run
is built from and the inputs those objects cannot invent for themselves. No
physics parameter and no Python default is repeated here, so a default can
only ever be changed in one place -- the Python object that owns it.

Importing this module registers three OmegaConf resolvers. Two let a config
name a file shipped inside an installed package, rather than a path that is
only valid on the machine the config was written on:

``${jcm_data:<relative path>}``
    A file under :mod:`jcm.data`, e.g.
    ``${jcm_data:bc/t30/clim/forcing.nc}`` for the packaged T30 surface
    climatology.
``${jem_data:<relative path>}``
    A file under :mod:`jem.data`, e.g.
    ``${jem_data:DisplacedPoleGrid.SCRIP.nc}`` for the packaged SCRIP grid.

``${run_default:<keyword>}``
    The default of the named keyword argument of
    :func:`jem.driver.run_chunked`, read from its signature when the config is
    resolved. ``coupled_run/default.yaml`` is the schema of the coupled run --
    every key present, so any of them can be overridden without a ``+`` -- and
    a key whose value is ``run_chunked``'s own default says
    ``chunk: ${run_default:chunk}`` instead of repeating ``"30 days"``, so the
    default lives only on the signature and cannot drift from it.

``${jcm_groups:}``
    A multi-line listing of jax-gcm's own config groups and their options,
    spelled the way they are overridden under the atmosphere
    (``physics@atmosphere.physics: echam, speedy``). Only ``--help`` uses it:
    Hydra's ``$APP_CONFIG_GROUPS`` lists the groups of the primary config's
    package but not those on the ``hydra.searchpath`` entry that carries
    jax-gcm's, and the listing is read off the installed package rather than
    typed out so it cannot go stale.

The first two exist so the shipped configurations run **offline**: jax-gcm's own
configurations fetch their boundary data from an ``hf://`` mirror, which
needs the network and a warm cache, while everything referenced through
these resolvers is already on disk next to the code.

Anything that composes a JAX-ESM config (``jem.main``, ``jem.runners``, the
tests) must import this package first so the resolvers exist before a value
that uses one is read. Composing through ``hydra.initialize_config_module("jem.config")``
-- or through ``pkg://jem.config`` on another app's search path -- imports it
for free, because that is how Hydra reads the package's files.
"""

from importlib import resources
from typing import Any

from omegaconf import OmegaConf


def package_data_path(package: str, relative_path: str) -> str:
    """Return the absolute path of ``relative_path`` inside ``package``.

    Parameters
    ----------
    package : str
        Importable package holding the data, e.g. ``"jcm.data"``.
    relative_path : str
        Path of the file relative to that package, with ``/`` separators.

    Returns
    -------
    str
        The file's absolute path.

    Raises
    ------
    FileNotFoundError
        If the package has no such file. Raised here, while composing, so a
        typo in a config is reported with the name that caused it instead of
        surfacing much later as a netCDF open error.

    """
    path = resources.files(package).joinpath(relative_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{relative_path!r} is not a file shipped in {package!r} "
            f"(looked at {str(path)!r})."
        )
    return str(path)


def _jcm_data(relative_path: str) -> str:
    """Resolve a path inside jax-gcm's packaged boundary data."""
    return package_data_path("jcm.data", relative_path)


def _jem_data(relative_path: str) -> str:
    """Resolve a path inside JAX-ESM's packaged grid and weight files."""
    return package_data_path("jem.data", relative_path)


# ``replace=True`` because a process may import this package more than once
# through different entry points (the CLI, then a test, then a notebook); a
# second registration of the same name is an error otherwise.
def _run_default(keyword: str) -> Any:
    """Return the default of ``jem.driver.run_chunked``'s keyword ``keyword``.

    Imported inside the function: :mod:`jem.driver` pulls in the coupler and
    the atmosphere adapter, which composing a config that never resolves a
    ``run_default`` (the composition of any other group, ``--help``) should
    not pay for.

    Raises
    ------
    ValueError
        If ``run_chunked`` has no such keyword, or it has no default (a
        required argument has no value to take).

    """
    import inspect

    from jem.driver import run_chunked

    parameters = inspect.signature(run_chunked).parameters
    if keyword not in parameters:
        raise ValueError(
            f"run_chunked() has no keyword {keyword!r} to take a default from; "
            f"its keywords are {sorted(parameters)}."
        )
    default = parameters[keyword].default
    if default is inspect.Parameter.empty:
        raise ValueError(f"run_chunked()'s {keyword!r} has no default.")
    return default


def _jcm_groups() -> str:
    """List jax-gcm's config groups and options as ``--help`` should show them.

    The re-rooting package of a group is ``@atmosphere.<group>`` (the
    ``defaults`` list of ``config.yaml``); ``configuration`` is the exception,
    a whole-bundle group composed with ``+configuration@atmosphere=<option>``.
    """
    lines = []
    for group in sorted(resources.files("jcm.config").iterdir(), key=lambda g: g.name):
        if not group.is_dir() or group.name in ("hydra", "__pycache__"):
            continue
        options = sorted(
            option.name[: -len(".yaml")]
            for option in group.iterdir()
            if option.name.endswith(".yaml")
        )
        spelled = (
            f"+{group.name}@atmosphere"
            if group.name == "configuration"
            else f"{group.name}@atmosphere.{group.name}"
        )
        lines.append(f"{spelled}: {', '.join(options)}")
    return "\n".join(lines)


OmegaConf.register_new_resolver("jcm_data", _jcm_data, replace=True)
OmegaConf.register_new_resolver("jcm_groups", _jcm_groups, replace=True)
OmegaConf.register_new_resolver("jem_data", _jem_data, replace=True)
OmegaConf.register_new_resolver("run_default", _run_default, replace=True)

__all__ = ["package_data_path"]
