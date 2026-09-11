"""Tests for `jem.constants`.

The rule this file enforces is the one thing a coupler cannot get wrong
quietly: JEM must not carry its own copy of a constant the atmosphere already
defines, because the two copies would only ever be noticed as a slow energy
leak across the interface.
"""

import dataclasses

import pytest

import jcm.constants as jcm_constants
from jem import constants
from jem.constants import SurfaceConstants, set_constants

#: Names that used to live in `jem.constants` and now come from `jcm.constants`.
REMOVED_IN_FAVOUR_OF_JCM = {
    "g0": "grav",
    "stephan_boltzmann_const": "sbc",
    "solar_const": "solc",
    "atmosphere_specific_heat_capacity_under_constant_pressure": "cpd",
    "freezing_point_K": "tmelt",
    "ice_melting_point_K": "tmelt",
    "ice_density": "rhoi",
    "ice_latent_heat_fusion": "alhf",
}


@pytest.fixture(autouse=True)
def restore_constants():
    """Undo any override, so an override in one test cannot leak into another."""
    original = constants.surface_constants
    yield
    set_constants(original)


def test_constants_match_jcm():
    """Every constant JEM still owns is one jcm.constants does not define."""
    for field in dataclasses.fields(SurfaceConstants):
        jcm_value = getattr(jcm_constants, field.name, None)
        if jcm_value is not None:
            assert getattr(constants, field.name) == jcm_value, (
                f"{field.name} is defined by both jem.constants and jcm.constants "
                "with different values; jcm.constants is the single source of truth."
            )


def test_duplicated_constants_were_removed():
    """The old duplicate names are gone, so nothing can read a stale copy."""
    for removed in REMOVED_IN_FAVOUR_OF_JCM:
        assert not hasattr(constants, removed)


def test_jcm_owns_the_removed_names():
    """Each removed name has a live replacement in jcm.constants."""
    for removed, jcm_name in REMOVED_IN_FAVOUR_OF_JCM.items():
        assert getattr(jcm_constants, jcm_name) is not None, (
            f"{removed} was removed in favour of jcm.constants.{jcm_name}, which "
            "must exist."
        )


def test_attribute_access_honours_an_override():
    """`set_constants` is seen by module-attribute reads, as in jcm.constants."""
    original = constants.ocean_density
    set_constants(ocean_density=original + 5.0)

    assert constants.ocean_density == original + 5.0
    # Untouched fields keep their values.
    assert constants.ocean_specific_heat_capacity == 3985.0


def test_set_constants_replaces_the_whole_set():
    """A whole SurfaceConstants can be swapped in, as jcm.constants allows."""
    set_constants(SurfaceConstants(ocean_density=1.0))

    assert constants.ocean_density == 1.0
    assert constants.seawater_freezing_point_K == 271.35


def test_set_constants_rejects_nonsense():
    """A typo in a constant name is an error, not a silently ignored keyword."""
    with pytest.raises(ValueError, match="Unknown surface constant"):
        set_constants(oceanic_density=1.0)
    with pytest.raises(ValueError, match="not both"):
        set_constants(SurfaceConstants(), ocean_density=1.0)


def test_unknown_attribute_still_raises():
    """The forwarding `__getattr__` does not invent attributes."""
    with pytest.raises(AttributeError, match="no attribute"):
        constants.not_a_constant


def test_slab_models_see_an_override(tmp_path):
    """An override made before construction reaches the physics.

    This is why the models read `constants.<name>` rather than importing the
    value: a `from`-import would bind at import time and quietly ignore
    `set_constants`.
    """
    import jax.numpy as jnp
    import numpy as np

    from jem.components.slab.slab_ocean_model import SlabOceanModel
    from tests.unit.slab_test_utils import coupling_time, make_grid

    grid = make_grid()
    heat_flux = -1000.0

    def one_step_warming():
        model = SlabOceanModel(grid)
        carry = model.initialize()
        carry["forcing"] = carry["forcing"].replace(
            total_heat_flux=jnp.full(grid.shape, heat_flux)
        )
        stepped, _ = model.step(carry, coupling_time(0))
        return float(
            jnp.mean(
                stepped["state"].sea_surface_temperature
                - carry["state"].sea_surface_temperature
            )
        )

    reference = one_step_warming()
    set_constants(ocean_specific_heat_capacity=2.0 * 3985.0)
    doubled = one_step_warming()

    # Twice the heat capacity, half the warming.
    np.testing.assert_allclose(doubled, 0.5 * reference, rtol=1e-4)
