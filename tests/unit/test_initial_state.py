"""The atmosphere starts from the initial condition its ``init`` group names."""

import jax
import jax_datetime as jdt
import numpy as np
import pytest
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from omegaconf import OmegaConf

from jem.components.jcm import JCMComponent
from jem.runners import build_initial_state

START = "2000-01-01"


@pytest.fixture(scope="module")
def model():
    coords = get_speedy_coords(layers=5, spectral_truncation=21)
    return Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                 start_time=jdt.to_datetime(START))


def config(**init):
    return OmegaConf.create({"init": init})


def test_isothermal_is_jax_gcms_default(model):
    assert build_initial_state(config(kind="isothermal"), model) == (None, None)
    assert build_initial_state(OmegaConf.create({}), model) == (None, None)


def test_jw_reaches_the_component_carry(model):
    """A non-default state is what ``initialize`` puts in the carry."""
    state, physics = build_initial_state(config(kind="jw", rh=0.6), model)
    assert physics is None
    default = JCMComponent(model).initialize()["state"]
    started = JCMComponent(model, initial_state=state).initialize()["state"]
    assert not np.allclose(np.asarray(started.temperature_variation),
                           np.asarray(default.temperature_variation))


def test_era5_defaults_to_the_models_start_date(model, monkeypatch):
    """No ``init.date``: the analysis is the one for the day the run starts."""
    import jcm.initial_states

    calls = []
    monkeypatch.setattr(jcm.initial_states, "era5_state",
                        lambda coords, date: calls.append(date) or "era5-state", raising=False)
    assert build_initial_state(config(kind="era5", date=None), model) == ("era5-state", None)
    assert calls[0].startswith(START)
    build_initial_state(config(kind="era5", date="2022-12-24T06"), model)
    assert calls[1] == "2022-12-24T06"


def test_from_state_keeps_the_donors_physics_carry(model, monkeypatch, tmp_path):
    import jcm.initial_states

    monkeypatch.setattr(jcm.initial_states, "checkpoint_state",
                        lambda m, path, unstamped_scale=None: ("state", {"carry": path}, 300.0))
    state, physics = build_initial_state(config(kind="from_state", file=str(tmp_path / "x.ckpt")), model)
    assert state == "state" and physics == {"carry": str(tmp_path / "x.ckpt")}
    component = JCMComponent(model, initial_physics_state=jax.tree.map(lambda x: x, physics))
    assert component.initial_physics_state == physics


def test_unknown_kind_is_refused(model):
    with pytest.raises(ValueError, match="not one of jax-gcm's initial conditions"):
        build_initial_state(config(kind="nope"), model)
