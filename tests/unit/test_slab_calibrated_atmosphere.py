"""The SPEEDY slab configurations run jax-gcm's coupled-calibrated parameters.

``aquaplanet-slab``, ``aquaplanet-slab-mixed-grid``, ``earth-slab`` and
``earth-slab-winton`` compose jax-gcm's ``speedy-t31`` configuration bundle
under ``atmosphere``. The parameter values live once, in jax-gcm; every
expectation here is read from the bundle file at test time, never restated, so
a retune in jax-gcm moves these tests with it instead of failing them.
"""

import importlib.resources
from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

import jem.config  # noqa: F401  (registers the resolvers a configuration uses)
from jem import configurations

SLAB_CONFIGURATIONS = [
    "aquaplanet-slab",
    "aquaplanet-slab-mixed-grid",
    "earth-slab",
    "earth-slab-winton",
]
AQUAPLANETS = ["aquaplanet-slab", "aquaplanet-slab-mixed-grid"]
EARTHS = ["earth-slab", "earth-slab-winton"]

_BUNDLE = Path(str(importlib.resources.files("jcm.config"))) / "configuration" / "speedy-t31.yaml"


def _bundle() -> DictConfig:
    return OmegaConf.load(_BUNDLE)


def _composed(name: str) -> DictConfig:
    with initialize_config_module(config_module="jem.config", version_base="1.3"):
        return compose(config_name="config", overrides=[f"+configuration={name}"])


def _leaves(node, path=()):
    """Yield ``(path, value)`` for every scalar under a plain container."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, (*path, key))
    else:
        yield path, node


@pytest.mark.parametrize("name", SLAB_CONFIGURATIONS)
def test_configuration_composes_the_bundles_calibrated_terms(name):
    """Every ``physics.terms`` value of the bundle lands under ``atmosphere``."""
    bundle = OmegaConf.to_container(_bundle(), resolve=True)
    atmosphere = OmegaConf.to_container(_composed(name).atmosphere, resolve=True)
    expected = list(_leaves(bundle["physics"]["terms"]))
    assert len(expected) > 100  # the walk found the calibration, not an empty node
    for path, value in expected:
        node = atmosphere["physics"]["terms"]
        for key in path:
            node = node[key]
        assert node == value, path
    # The step the calibration was made at, from the bundle.
    assert atmosphere["run"]["time_step"] == bundle["run"]["time_step"]
    assert atmosphere["grid"]["spectral_truncation"] == 31


@pytest.mark.parametrize("name", AQUAPLANETS)
def test_aquaplanets_keep_their_own_terrain_and_forcing(name):
    """No bundle file survives on an aquaplanet: nothing to fetch, nothing read."""
    atmosphere = _composed(name).atmosphere
    assert atmosphere.terrain.kind == "aquaplanet"
    assert atmosphere.terrain.file is None
    assert atmosphere.forcing.kind == "default"
    assert atmosphere.forcing.file is None


@pytest.mark.parametrize("name", EARTHS)
def test_earths_keep_the_packaged_climatology(name):
    """The Earth configurations read jax-gcm's packaged T30 files, not the bundle's mirror."""
    atmosphere = _composed(name).atmosphere
    assert atmosphere.terrain.kind == "from_file"
    assert atmosphere.forcing.kind == "from_file"
    for node in (atmosphere.terrain, atmosphere.forcing):
        assert not str(node.file).startswith("hf://")
        assert Path(str(node.file)).is_file()
    assert str(atmosphere.terrain.file).endswith("bc/t30/clim/terrain.nc")
    assert str(atmosphere.forcing.file).endswith("bc/t30/clim/forcing.nc")
    # A climatology is replayed every year, whichever way it is spelled.
    assert atmosphere.forcing.align == "wrap_year"
    # The T30 file is on the model's own grid, so this is a no-op, and it is the
    # group's default as well as the bundle's value.
    assert atmosphere.terrain.interpolate is True


def test_the_composed_parameters_reach_the_built_physics_and_a_scan_runs():
    """The bundle's values are on the built terms, and two coupled steps scan.

    Composing a value proves nothing about what the atmosphere runs on: the
    builder could drop it. So read each term's parameter record back off the
    built model and compare it with the bundle, then run the coupled step
    through ``lax.scan`` (the carry-structure check only a scan makes).
    """
    experiment = configurations.load("aquaplanet-slab")
    model = experiment.coupler.components["atm"].model
    terms = {term.name: term for term in model.physics.terms}
    bundle = OmegaConf.to_container(_bundle(), resolve=True)

    compared = 0
    for term_name, records in bundle["physics"]["terms"].items():
        for key, values in records.items():
            # A term with one record keeps it as `params`; one with several
            # (the shortwave's `sw_params` and `mod_radcon_params`) keeps each
            # under its own name.
            attribute = key if hasattr(terms[term_name], key) else "params"
            record = getattr(terms[term_name], attribute)
            record = record.get_value() if hasattr(record, "get_value") else record
            for field, value in values.items():
                np.testing.assert_allclose(
                    float(getattr(record, field)), value, rtol=1e-6,
                    err_msg=f"{term_name}.{key}.{field}",
                )
                compared += 1
    assert compared > 70

    assert model.dt_si.to_timedelta().total_seconds() == _bundle().run.time_step * 60

    carry = experiment.coupler.initialize()
    final, _ = experiment.coupler.generate_trajectory_function(2)(carry)
    assert jax.tree_util.tree_structure(final) == jax.tree_util.tree_structure(carry)
    sst = final.components["ocn"]["state"].sea_surface_temperature
    assert bool(np.isfinite(np.asarray(sst)).all())
