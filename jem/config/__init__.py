"""JAX-ESM's Hydra configuration package, and the resolvers its YAML uses.

The YAML in this package is *wiring only*: it names the Python objects a run
is built from and the inputs those objects cannot invent for themselves. No
physics parameter and no Python default is repeated here, so a default can
only ever be changed in one place -- the Python object that owns it.

Importing this module registers two OmegaConf resolvers that let a config
name a file shipped inside an installed package, rather than a path that is
only valid on the machine the config was written on:

``${jcm_data:<relative path>}``
    A file under :mod:`jcm.data`, e.g.
    ``${jcm_data:bc/t30/clim/forcing.nc}`` for the packaged T30 surface
    climatology.
``${jem_data:<relative path>}``
    A file under :mod:`jem.data`, e.g.
    ``${jem_data:DisplacedPoleGrid.SCRIP.nc}`` for the packaged SCRIP grid.

They exist so the shipped configurations run **offline**: jax-gcm's own
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
OmegaConf.register_new_resolver("jcm_data", _jcm_data, replace=True)
OmegaConf.register_new_resolver("jem_data", _jem_data, replace=True)

__all__ = ["package_data_path"]
