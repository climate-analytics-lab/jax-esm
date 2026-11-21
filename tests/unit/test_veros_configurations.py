"""What the two Veros configurations compose, read off the composed config.

Nothing here imports or builds Veros: it checks the configuration layer, so it
runs in the ordinary fast suite. ``test_veros_seaice.py`` and the
``test_config.py`` Veros tests build the configurations.

The configurations take SPEEDY's parameters from jax-gcm's ``speedy-t31``
bundle, which carries the values calibrated for coupled use at a 30-minute
step. They live in jax-gcm only; the tests read them from that file at test
time, so a retune there is followed here and no number is restated.
"""

import importlib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

import jem.config  # noqa: F401  (registers the ${jem_data:} / ${jcm_data:} resolvers)

JCM_BUNDLE = (
    Path(str(importlib.resources.files("jcm.config"))) / "configuration" / "speedy-t31.yaml"
)

#: Each Veros configuration and the terrain and forcing files that are its own
#: (basenames), so a stray ``hf://`` file from the bundle shows up.
FILES = {
    "veros-earth": ("terrain_veros_earth_T31.nc", "forcing.nc"),
    "veros-double-drake": ("terrain_double_drake_T31.nc", "forcing_idealised_T31.nc"),
}


def _leaves(node, path=()):
    """Yield ``(path, value)`` for every non-mapping value under ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, (*path, key))
    else:
        yield path, node


def _composed(configuration: str) -> DictConfig:
    with initialize_config_module(config_module="jem.config", version_base="1.3"):
        return compose(config_name="config", overrides=[f"+configuration={configuration}"])


@pytest.mark.parametrize("configuration", FILES)
def test_a_veros_configuration_carries_the_calibrated_speedy_parameters(configuration):
    """``atmosphere.physics.terms`` holds every value of the bundle's, none restated here."""
    bundle = OmegaConf.to_container(OmegaConf.load(JCM_BUNDLE), resolve=True)
    expected = bundle["physics"]["terms"]
    assert expected, "the bundle should carry calibrated terms"
    atmosphere = OmegaConf.to_container(_composed(configuration).atmosphere, resolve=True)
    composed = dict(_leaves(atmosphere["physics"]["terms"]))
    # Every leaf of the bundle is composed with the bundle's value. (The composed
    # terms also hold what the physics group adds, such as each term's `_target_`.)
    assert dict(_leaves(expected)).items() <= composed.items()
    # The bundle's grid and step arrive with it (the step the calibration was made at).
    assert atmosphere["run"]["time_step"] == bundle["run"]["time_step"]
    assert atmosphere["grid"]["spectral_truncation"] == 31


@pytest.mark.parametrize("configuration", FILES)
def test_a_veros_configuration_reads_its_own_terrain_and_forcing(configuration):
    """The bundle's ``hf://`` files are overridden: no network, and the right grid handling."""
    terrain, forcing = FILES[configuration]
    atmosphere = _composed(configuration).atmosphere
    assert Path(atmosphere.terrain.file).name == terrain
    assert Path(atmosphere.forcing.file).name == forcing
    assert "hf://" not in atmosphere.terrain.file
    assert "hf://" not in atmosphere.forcing.file
    # Both forcing files are twelve-record climatologies replayed every year,
    # and the terrain files are on the T31 grid, where resampling is the
    # identity that also brings them to the model's float64.
    assert atmosphere.terrain.interpolate is True
    assert atmosphere.forcing.align == "wrap_year"
