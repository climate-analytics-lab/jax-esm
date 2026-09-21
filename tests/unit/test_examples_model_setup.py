"""Tests for the example drivers' JCM monkey-patches.

The drivers under ``examples/`` are exercised end to end by
``tests/examples``, which runs their notebooks and ``run.sh`` on pull
requests only. ``_freeze_season`` never runs there -- ``freeze_season_at_day``
defaults to ``None`` and no driver sets it -- yet it is the one place an
example reaches into ``jcm``'s own machinery, replacing the model's date
conversion on the instance. That makes it exactly the kind of patch a jax-gcm
rename turns into a silent no-op, so it gets a fast test of its own here: it
builds a model and calls the patched conversion directly, without integrating
anything.
"""

import importlib.util
import sys
from pathlib import Path

import jax_datetime as jdt
import numpy as np
import pytest
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

# The example imports its Veros setup module at import time, and that module
# imports `veros`, the optional jittable fork. Skip rather than fail where it
# is absent; CI's test job installs it.
pytest.importorskip("veros")

EXAMPLE_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_experimental"
    / "03_jcm_veros_earth"
)

START_DATE = jdt.to_datetime("2000-01-01")
CALENDAR = "365_day"

# T21 with 5 levels on an aquaplanet: the cheapest model SPEEDY physics
# accepts, and nothing here integrates it.
LAYERS = 5
TRUNCATION = 21

FREEZE_DAY = 80
# Well past the freeze day, and six hours into a day, so the assertions also
# fail a patch that froze the diurnal cycle along with the season.
ELAPSED_DAYS = 200
HOUR_OF_DAY_SECONDS = 6 * 3600
SIM_TIME = ELAPSED_DAYS * 86400.0 + HOUR_OF_DAY_SECONDS


@pytest.fixture(scope="module")
def model_setup():
    """Import the JCM-Veros example's ``model_setup`` module by path.

    The example is a directory of scripts, not a package, and its modules
    import each other by bare name (``veros_case_setup``), so the directory
    goes on ``sys.path`` for the duration of the import.
    """
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "jcm_veros_earth_model_setup", EXAMPLE_DIR / "model_setup.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
    return module


@pytest.fixture
def frozen_model(model_setup) -> Model:
    """Build a small model and freeze its season at :data:`FREEZE_DAY`.

    A fresh model per test: ``_freeze_season`` wraps whatever conversion the
    instance currently holds, so a shared model would have each test patching
    the previous test's patch.
    """
    coords = get_speedy_coords(layers=LAYERS, spectral_truncation=TRUNCATION)
    model = Model(
        coords=coords,
        terrain=TerrainData.aquaplanet(coords),
        start_date=START_DATE,
        calendar=CALENDAR,
    )
    model_setup._freeze_season(model, FREEZE_DAY)
    return model


def _expected_frozen_date() -> jdt.Datetime:
    """Build the date the freeze must pin every later instant to."""
    return jdt.Datetime.from_pydatetime(START_DATE) + jdt.Timedelta(
        days=FREEZE_DAY, seconds=HOUR_OF_DAY_SECONDS
    )


def _assert_same_instant(actual: jdt.Datetime, expected: jdt.Datetime) -> None:
    """Assert two ``jax_datetime`` datetimes stand for the same instant."""
    np.testing.assert_array_equal(
        np.asarray(actual.delta.days), np.asarray(expected.delta.days)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.delta.seconds), np.asarray(expected.delta.seconds)
    )


def test_freeze_season_pins_the_date_and_keeps_the_diurnal_cycle(frozen_model):
    """Check the season stops at the freeze day while the time of day runs on."""
    _assert_same_instant(
        frozen_model.date_from_sim_time(SIM_TIME).dt, _expected_frozen_date()
    )


def test_freeze_season_also_reaches_the_private_alias(frozen_model):
    """Check the compatibility alias resolves to the patched conversion.

    ``_freeze_season`` overrides the public ``date_from_sim_time`` on the
    instance because jax-gcm's ``_date_from_sim_time`` only delegates to it.
    Were the delegation to run the other way round, the override would leave
    jax-gcm's own callers on the unpatched method and the season would go on
    advancing silently instead of the run failing -- so the delegation the
    patch relies on is pinned here. The alias is a private jax-gcm name that
    JAX-ESM itself never calls, so a jax-gcm that drops it is not a break:
    the test then skips rather than failing the canary.
    """
    if not hasattr(Model, "_date_from_sim_time"):
        pytest.skip("this jax-gcm defines no _date_from_sim_time alias")
    _assert_same_instant(
        frozen_model._date_from_sim_time(SIM_TIME).dt, _expected_frozen_date()
    )
