"""Tests for the JCM adapter (`jem.components.jcm_component`).

These exercise the adapter against the installed `jcm` package, so they build a
real (T31, 8-layer) SPEEDY model. The model is built once per module because
construction plus the first `initialize()` dominates the runtime.
"""

import warnings

import jax.numpy as jnp
import jax_datetime as jdt
import pytest
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords

from jem.base.coupler import Coupler
from jem.components import JCM, SlabOceanModel
from jem.components.slab.grid import generate_slab_grid

START_DATETIME = jdt.to_datetime("2000-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")

# T31 with jcm's default (96, 48) nodal grid.
EXPECTED_GRID_SHAPE = (96, 48)


def _build_adapted_model() -> Model:
    """Build a T31 SPEEDY model and make it JEM-compatible."""
    model = Model(
        coords=get_speedy_coords(),
        start_date=START_DATETIME,
        log_level=50,
    )
    return JCM.make_jem_compatible(model, coupling_timestep=COUPLING_TIMESTEP)


@pytest.fixture(scope="module")
def adapted_model() -> Model:
    return _build_adapted_model()


def test_initialize_carry_structure(adapted_model):
    """`initialize()` returns the three-key carry the coupler expects."""
    carry = adapted_model.initialize()

    assert set(carry) == {"state", "derived", "forcing"}
    assert carry["derived"].total_heat_flux.shape == EXPECTED_GRID_SHAPE
    assert carry["derived"].total_freshwater_flux.shape == EXPECTED_GRID_SHAPE


def test_no_float64_warning(adapted_model):
    """Initialisation must not request float64 on a float32 JAX build.

    The adapter used to cast the whole carry to float64, which JAX silently
    truncates back to float32 while emitting a UserWarning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapted_model.initialize()

    float64_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "float64" in str(w.message)
    ]
    assert not float64_warnings, [str(w.message) for w in float64_warnings]


@pytest.mark.slow
def test_two_coupling_steps_run():
    """Two coupled atmosphere/slab-ocean steps run and stay finite."""
    atmosphere = _build_adapted_model()
    ocean = SlabOceanModel(
        grid=generate_slab_grid("JCM::T31"),
        start_datetime=START_DATETIME,
        timestep=86400.0,
    )

    def mapper(carry):
        carry["ocn"]["forcing"] = carry["ocn"]["forcing"].replace(
            total_heat_flux=carry["atm"]["derived"].total_heat_flux,
        )
        carry["atm"]["forcing"] = carry["atm"]["forcing"].replace(
            sea_surface_temperature=carry["ocn"]["state"].sea_surface_temperature,
        )
        return carry

    coupler = Coupler(
        components={"atm": atmosphere, "ocn": ocean},
        mappers={"mapper": mapper},
    )

    _, final_carry, _ = coupler.run(
        workflow=["mapper", "atm", "ocn"],
        iterations=2,
        show_progress=False,
        verbose=False,
    )

    total_heat_flux = final_carry["atm"]["derived"].total_heat_flux
    assert total_heat_flux.shape == EXPECTED_GRID_SHAPE
    assert bool(jnp.all(jnp.isfinite(total_heat_flux)))
