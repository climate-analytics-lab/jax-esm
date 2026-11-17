"""The named ECHAM configuration composes, builds and steps.

``aquaplanet-slab-echam`` is the only shipped coupled configuration that runs
ECHAM physics, and ECHAM's default land (SLIM) refuses a grid it has no
surface dataset for, so the one-line composition ``physics@atmosphere.physics=
echam`` fails at build time. These tests pin that the named configuration
avoids that on its own, offline, and that the coupled system it builds is
actually coupled: the ocean's SST moves and the atmosphere's flux reaches it.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_module

import jem.runners
from jem import configurations

CONFIG_MODULE = "jem.config"
NAME = "aquaplanet-slab-echam"


def _compose(*overrides):
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[f"+configuration={NAME}", "coupled_run=short_run", *overrides],
        )


def test_configuration_is_listed_with_a_summary():
    assert "ECHAM" in configurations.available()[NAME]


def test_composes_the_echam_components():
    cfg = _compose()
    atmosphere = cfg.atmosphere
    assert "rrtmgp_radiation" in atmosphere.physics.terms
    assert "echam_surface_exchange" in atmosphere.physics.terms
    assert atmosphere.grid.vertical == "sigma"
    assert atmosphere.terrain.kind == "aquaplanet"
    assert cfg.ocean._target_.endswith("SlabOceanModel")
    assert cfg.seaice._target_.endswith("SlabSeaiceModel")
    assert cfg.land is None


def test_land_model_is_prescribed_because_slim_has_no_t42_dataset():
    """The default `slim` land is what the bare one-liner dies on."""
    cfg = _compose()
    assert cfg.atmosphere.physics.land_model == "prescribed"


def test_starts_from_the_dry_jw_state_not_isothermal_rest():
    """ECHAM's cold start is the dry Jablonowski-Williamson state."""
    cfg = _compose()
    assert cfg.atmosphere.init.kind == "jw"
    assert cfg.atmosphere.init.rh == 0.0


def test_atmosphere_step_divides_the_coupling_step():
    """A coupling step has to be a whole number of atmosphere steps."""
    from jcm.date import parse_duration_seconds

    cfg = _compose()
    coupling_seconds = parse_duration_seconds(cfg.coupling.timestep)
    step_seconds = 60 * cfg.atmosphere.run.time_step
    assert coupling_seconds % step_seconds == 0
    # RRTMGP is called every 2 h (the radiation parameter default).
    assert 7200 % step_seconds == 0


@pytest.mark.slow
def test_two_coupled_steps_through_the_configuration():
    """Two steps through ``generate_trajectory_function(2)``, as lax.scan sees them.

    A single component's step does not catch a carry-structure mismatch; only
    the scan does. The SST must move (the ocean saw the atmosphere's flux) and
    everything must be finite.
    """
    cfg = _compose()
    coupler = jem.runners.build_coupler(cfg)
    carry = coupler.initialize()
    sst0 = np.asarray(carry.components["ocn"]["state"].sea_surface_temperature)

    final, _ = coupler.generate_trajectory_function(2)(carry)

    sst = np.asarray(final.components["ocn"]["state"].sea_surface_temperature)
    assert np.all(np.isfinite(sst))
    assert np.max(np.abs(sst - sst0)) > 1e-3, "the SST never responded to the atmosphere"
    atm_derived = final.components["atm"]["derived"]
    for name in ("total_heat_flux", "precipitation", "evaporation"):
        assert bool(jnp.all(jnp.isfinite(getattr(atm_derived, name)))), name
