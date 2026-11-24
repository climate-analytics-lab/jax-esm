"""The ECHAM configurations are jax-gcm's own atmosphere over JEM's slab.

``aquaplanet-slab-echam`` and ``earth-slab-echam`` compose jax-gcm's
``t63-echam-1m`` configuration bundle whole under ``atmosphere`` rather than
restating an ECHAM setup of their own, so the atmosphere a coupled run
integrates is the one jax-gcm validates. These tests pin that: the composed
atmosphere is compared with the bundle composed on its own, key for key, and
swapping the bundle for another jax-gcm configuration is one override.

T63 with 47 levels takes minutes per simulated day on a CPU and reads
jax-gcm's data mirror, so no test integrates it. The one integration here
(``test_two_coupled_steps_on_the_small_grid``) takes the named configuration
onto jax-gcm's smallest ECHAM grid to check the plumbing -- that ECHAM's
surface fluxes reach the slab ocean and the carry survives ``lax.scan`` -- and
it is the only place the test suite integrates these configurations.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import jem.runners
from jem import configurations

CONFIG_MODULE = "jem.config"
BUNDLE = "t63-echam-1m"
AQUAPLANET = "aquaplanet-slab-echam"
EARTH = "earth-slab-echam"

#: What takes a named ECHAM configuration onto a grid a test can integrate:
#: jax-gcm's smallest ECHAM grid (T42, eight sigma layers), and the
#: prescribed-moisture land, because the default SLIM land has a surface
#: dataset only for T63 and above and refuses to build without one. Nothing
#: else changes: the physics terms, the initial state, the 12-minute step and
#: the sponge are the bundle's.
SMALL_GRID = (
    "grid@atmosphere.grid=echam_t42_l8_sigma",
    "atmosphere.physics.land_model=prescribed",
)


def _compose(*overrides):
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def _atmosphere(cfg):
    return OmegaConf.to_container(cfg.atmosphere, resolve=False)


@pytest.mark.parametrize("name", [AQUAPLANET, EARTH])
def test_configuration_is_listed_with_a_summary(name):
    assert "ECHAM" in configurations.available()[name]


def test_the_earth_atmosphere_is_the_bundle_s_unchanged():
    """``earth-slab-echam`` adds a surface to the bundle and alters nothing in it."""
    assert _atmosphere(_compose(f"+configuration={EARTH}")) == _atmosphere(
        _compose(f"+configuration@atmosphere={BUNDLE}")
    )


def test_the_aquaplanet_atmosphere_differs_from_the_bundle_only_in_its_surface():
    """The aquaplanet replaces the bundle's terrain and forcing, and nothing else."""
    ours = _atmosphere(_compose(f"+configuration={AQUAPLANET}"))
    bundle = _atmosphere(_compose(f"+configuration@atmosphere={BUNDLE}"))
    assert ours["terrain"]["kind"] == "aquaplanet"
    assert ours["forcing"]["kind"] == "default"
    for group in ("terrain", "forcing"):
        ours.pop(group)
        bundle.pop(group)
    assert ours == bundle


@pytest.mark.parametrize("name", [AQUAPLANET, EARTH])
def test_the_surface_components_are_the_slab_s(name):
    cfg = _compose(f"+configuration={name}")
    assert cfg.ocean._target_.endswith("SlabOceanModel")
    assert cfg.seaice._target_.endswith("SlabSeaiceModel")
    # ECHAM integrates its own land (SLIM), inside the atmosphere.
    assert cfg.land is None


def test_the_earth_ocean_and_ice_read_the_atmosphere_s_own_forcing_file():
    """One file, named once: the ocean's SST target is what the atmosphere reads."""
    cfg = _compose(f"+configuration={EARTH}")
    assert cfg.ocean.sst_clim_file == cfg.atmosphere.forcing.file
    assert cfg.seaice.ice_clim_file == cfg.atmosphere.forcing.file


@pytest.mark.parametrize("name", [AQUAPLANET, EARTH])
def test_another_jax_gcm_bundle_is_one_override(name):
    """``configuration@atmosphere=<bundle>`` swaps the whole atmosphere."""
    cfg = _compose(f"+configuration={name}", "configuration@atmosphere=t63-echam-2m")
    terms = cfg.atmosphere.physics.terms
    assert "lohmann_2m_microphysics" in terms
    assert "echam_1m_microphysics" not in terms
    assert cfg.ocean._target_.endswith("SlabOceanModel")


def test_atmosphere_step_divides_the_coupling_step():
    """A coupling step has to be a whole number of atmosphere steps."""
    from jcm.date import parse_duration_seconds

    cfg = _compose(f"+configuration={AQUAPLANET}")
    coupling_seconds = parse_duration_seconds(cfg.coupling.timestep)
    assert coupling_seconds % (60 * cfg.atmosphere.run.time_step) == 0


@pytest.mark.slow
def test_two_coupled_steps_on_the_small_grid():
    """Two steps through ``generate_trajectory_function(2)``, as lax.scan sees them.

    A single component's step does not catch a carry-structure mismatch; only
    the scan does. The SST must move (the ocean saw the atmosphere's flux) and
    everything must be finite.
    """
    cfg = _compose(f"+configuration={AQUAPLANET}", *SMALL_GRID)
    assert "echam_surface_exchange" in cfg.atmosphere.physics.terms
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
