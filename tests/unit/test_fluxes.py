"""Tests for :mod:`jem.fluxes` -- the computed half of the Veros coupling.

The pure functions (:func:`~jem.fluxes.mask_fluxes_under_ice`,
:func:`~jem.fluxes.rotate_vector`,
:func:`~jem.fluxes.read_rotation_angles`) are tested directly on plain
arrays. :class:`~jem.fluxes.VerosExchange` is tested on small fake carries --
:mod:`flax.struct` dataclasses carrying exactly the field names it reads and
writes -- rather than on a real atmosphere or ocean, since nothing here needs
either model to be built.
"""

from importlib import resources

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
from flax import struct

from jem.base.component import CouplingTime
from jem.fluxes import (
    VerosExchange,
    mask_fluxes_under_ice,
    read_rotation_angles,
    rotate_vector,
)

DATA = resources.files("jem.data")
ROTATED_SCRIP_FILE = str(DATA / "RotatedGaussianLatLon.SCRIP.nc")

TIME = CouplingTime(
    step=jnp.int32(0),
    time=jdt.to_datetime("2001-01-01"),
    dt=86400.0,
)


# ---------------------------------------------------------------------------
# mask_fluxes_under_ice
# ---------------------------------------------------------------------------


def test_mask_fluxes_under_ice_lets_a_warming_flux_through():
    """A downward (warming, negative) heat flux survives at the freezing point."""
    sst = jnp.array([271.35])
    heat_flux = jnp.array([-50.0])
    freshwater_flux = jnp.array([1e-6])
    masked_heat, masked_fresh = mask_fluxes_under_ice(sst, heat_flux, freshwater_flux)
    np.testing.assert_allclose(masked_heat, heat_flux)
    np.testing.assert_allclose(masked_fresh, freshwater_flux)


def test_mask_fluxes_under_ice_blocks_a_cooling_flux_at_the_freezing_point():
    """A cooling flux is masked at/below freezing, and untouched above it.

    The freshwater flux follows the heat flux's mask exactly, since both are
    masked by the same `ice_free` condition.
    """
    sst = jnp.array([271.35, 280.0])
    heat_flux = jnp.array([50.0, 50.0])
    freshwater_flux = jnp.array([1e-6, 1e-6])
    masked_heat, masked_fresh = mask_fluxes_under_ice(sst, heat_flux, freshwater_flux)
    np.testing.assert_allclose(masked_heat, jnp.array([0.0, 50.0]))
    np.testing.assert_allclose(masked_fresh, jnp.array([0.0, 1e-6]))


# ---------------------------------------------------------------------------
# rotate_vector / read_rotation_angles
# ---------------------------------------------------------------------------


def test_rotate_vector_preserves_magnitude():
    """A rotation changes direction, never the vector's length."""
    u, v = jnp.array([3.0]), jnp.array([4.0])
    cos_angle, sin_angle = jnp.array([0.6]), jnp.array([0.8])
    x, y = rotate_vector(u, v, cos_angle, sin_angle)
    np.testing.assert_allclose(jnp.sqrt(x**2 + y**2), jnp.sqrt(u**2 + v**2))


def test_rotate_vector_is_the_identity_for_zero_angle():
    """cos=1, sin=0 (no rotation) returns (u, v) unchanged."""
    u, v = jnp.array([3.0, -1.0]), jnp.array([4.0, 2.0])
    x, y = rotate_vector(u, v, jnp.ones_like(u), jnp.zeros_like(u))
    np.testing.assert_allclose(x, u)
    np.testing.assert_allclose(y, v)


def test_read_rotation_angles_uses_the_scrip_layout():
    """Against the packaged RotatedGaussianLatLon grid: shape and unit norm."""
    cos_angle, sin_angle = read_rotation_angles(ROTATED_SCRIP_FILE)
    assert cos_angle.shape == (96, 48)
    assert sin_angle.shape == (96, 48)
    np.testing.assert_allclose(cos_angle**2 + sin_angle**2, 1.0, atol=1e-6)


def test_read_rotation_angles_names_the_file_when_unrotated():
    """An unrotated grid's SCRIP file has no rotation angles at all."""
    with pytest.raises(KeyError, match="JCM_T31.SCRIP.nc"):
        read_rotation_angles(str(DATA / "JCM_T31.SCRIP.nc"))


# ---------------------------------------------------------------------------
# VerosExchange
# ---------------------------------------------------------------------------


@struct.dataclass
class _AtmDerived:
    eastward_wind_stress: jnp.ndarray
    northward_wind_stress: jnp.ndarray
    total_heat_flux: jnp.ndarray
    total_freshwater_flux: jnp.ndarray


@struct.dataclass
class _AtmForcing:
    sea_surface_temperature: jnp.ndarray


@struct.dataclass
class _OcnForcing:
    surface_taux: jnp.ndarray
    surface_tauy: jnp.ndarray
    heat_flux: jnp.ndarray
    freshwater_flux: jnp.ndarray


@struct.dataclass
class _OcnDerived:
    sea_surface_temperature: jnp.ndarray


def _fake_components():
    shape = (4,)
    atm = {
        "derived": _AtmDerived(
            eastward_wind_stress=jnp.array([0.1, -0.05, 0.02, 0.0]),
            northward_wind_stress=jnp.array([-0.03, 0.04, 0.0, 0.01]),
            total_heat_flux=jnp.full(shape, 20.0),
            total_freshwater_flux=jnp.full(shape, 1e-6),
        ),
        "forcing": _AtmForcing(sea_surface_temperature=jnp.full(shape, 290.0)),
    }
    ocn = {
        "forcing": _OcnForcing(
            surface_taux=jnp.zeros(shape),
            surface_tauy=jnp.zeros(shape),
            heat_flux=jnp.zeros(shape),
            freshwater_flux=jnp.zeros(shape),
        ),
        "derived": _OcnDerived(sea_surface_temperature=jnp.full(shape, 285.0)),
    }
    return {"atm": atm, "ocn": ocn}


def test_veros_exchange_drives_the_ocean():
    """The exchange writes all four ocean forcing fields and the atmosphere's SST.

    The input carries are left unchanged (a pure exchanger), and the pytree
    structure of the result matches the input's exactly -- what `lax.scan`
    requires of every coupled step.
    """
    components = _fake_components()
    exchange = VerosExchange()
    result = exchange(components, TIME)

    assert jnp.any(result["ocn"]["forcing"].surface_taux != 0.0)
    assert jnp.any(result["ocn"]["forcing"].surface_tauy != 0.0)
    assert jnp.any(result["ocn"]["forcing"].heat_flux != 0.0)
    assert jnp.any(result["ocn"]["forcing"].freshwater_flux != 0.0)
    np.testing.assert_allclose(
        result["atm"]["forcing"].sea_surface_temperature,
        components["ocn"]["derived"].sea_surface_temperature,
    )

    # Purity: the carries passed in are untouched.
    np.testing.assert_allclose(
        components["ocn"]["forcing"].surface_taux, jnp.zeros(4)
    )
    np.testing.assert_allclose(
        components["atm"]["forcing"].sea_surface_temperature, jnp.full((4,), 290.0)
    )

    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(
        components
    )


def test_veros_exchange_casts_to_the_destination_carrys_dtype():
    """A value crossing the atm/ocn precision boundary matches what it lands in.

    Regression test for the real failure mode this closes: Veros runs in
    double precision (importing `veros.core` flips the process-global
    `jax_enable_x64` setting to True as a side effect), so its carry is
    entirely float64, while the atmosphere's carry is *mixed* -- whatever
    jax-gcm allocated before Veros was imported stays float32, and anything
    allocated after the flip (every per-step diagnostic) is float64 too, so
    which atmosphere fields are float32 depends on build order. This test
    uses the simplest instance of that -- an all-float32 atmosphere carry
    against an all-float64 ocean one -- since the cast this exercises is
    per-field regardless of which side of the split each one landed on.
    `jax.lax.scan` requires a step's *output* carry to match its *input*
    dtype exactly, so an uncast value crossing that boundary breaks the very
    first coupled step with an opaque error deep inside the coupler, not a
    physics one.

    `jax_enable_x64` is process-global, so this snapshots and restores it --
    the same guard jax-gcm's own `configurations_test.py` uses for the
    identical hazard (a physics term that flips it at construction) -- so a
    genuine float64 array can be built here without leaking the setting into
    every other test sharing this process.
    """
    x64_was_enabled = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        shape = (4,)
        atm = {
            "derived": _AtmDerived(
                eastward_wind_stress=jnp.full(shape, 0.1, dtype=jnp.float32),
                northward_wind_stress=jnp.full(shape, -0.03, dtype=jnp.float32),
                total_heat_flux=jnp.full(shape, 20.0, dtype=jnp.float32),
                total_freshwater_flux=jnp.full(shape, 1e-6, dtype=jnp.float32),
            ),
            "forcing": _AtmForcing(
                sea_surface_temperature=jnp.full(shape, 290.0, dtype=jnp.float32)
            ),
        }
        ocn = {
            "forcing": _OcnForcing(
                surface_taux=jnp.zeros(shape, dtype=jnp.float64),
                surface_tauy=jnp.zeros(shape, dtype=jnp.float64),
                heat_flux=jnp.zeros(shape, dtype=jnp.float64),
                freshwater_flux=jnp.zeros(shape, dtype=jnp.float64),
            ),
            "derived": _OcnDerived(
                sea_surface_temperature=jnp.full(shape, 285.0, dtype=jnp.float64)
            ),
        }
        components = {"atm": atm, "ocn": ocn}
        exchange = VerosExchange()
        result = exchange(components, TIME)

        assert result["ocn"]["forcing"].surface_taux.dtype == jnp.float64
        assert result["ocn"]["forcing"].surface_tauy.dtype == jnp.float64
        assert result["ocn"]["forcing"].heat_flux.dtype == jnp.float64
        assert result["ocn"]["forcing"].freshwater_flux.dtype == jnp.float64
        assert result["atm"]["forcing"].sea_surface_temperature.dtype == jnp.float32
    finally:
        jax.config.update("jax_enable_x64", x64_was_enabled)


def test_veros_exchange_uses_the_regridders_it_was_given():
    """`regrid["a2o_flux"]`/`["o2a_state"]` are called on the right fields."""
    a2o_calls = []
    o2a_calls = []

    def a2o_flux(value):
        a2o_calls.append(value)
        return value

    def o2a_state(value):
        o2a_calls.append(value)
        return value

    components = _fake_components()
    exchange = VerosExchange(regrid={"a2o_flux": a2o_flux, "o2a_state": o2a_state})
    exchange(components, TIME)

    # Both stress components and both fluxes cross the conservative a2o map
    # -- the stress is a momentum flux, so it is regridded like one.
    assert len(a2o_calls) == 4
    derived = components["atm"]["derived"]
    for field in (derived.eastward_wind_stress, derived.northward_wind_stress,
                  derived.total_heat_flux, derived.total_freshwater_flux):
        assert any(call is field for call in a2o_calls)
    # The sea surface temperature crosses o2a exactly once.
    assert len(o2a_calls) == 1
    np.testing.assert_allclose(
        o2a_calls[0], components["ocn"]["derived"].sea_surface_temperature
    )


def test_veros_exchange_without_regridders_is_the_identity_path():
    """No `regrid`, no `rotation_grid_file`: the single-grid double-drake case."""
    components = _fake_components()
    exchange = VerosExchange()
    assert exchange.rotation_grid_file is None
    result = exchange(components, TIME)
    # With no regridder and no rotation, the ocean receives exactly the stress
    # the atmosphere published -- no drag law, no rescaling, no sign flip.
    np.testing.assert_array_equal(
        result["ocn"]["forcing"].surface_taux,
        components["atm"]["derived"].eastward_wind_stress,
    )
    np.testing.assert_array_equal(
        result["ocn"]["forcing"].surface_tauy,
        components["atm"]["derived"].northward_wind_stress,
    )


def test_veros_exchange_passes_the_stress_sign_through():
    """A westerly (positive eastward) stress on the surface pushes Veros east.

    jax-gcm's `stress_u` is positive down -- the stress ON the surface -- and
    Veros' `surface_taux` is the stress on the water, positive eastward, so
    the sign must survive the exchange unchanged. Getting this wrong would
    spin every gyre backwards without failing anything else.
    """
    components = _fake_components()
    components["atm"]["derived"] = components["atm"]["derived"].replace(
        eastward_wind_stress=jnp.full((4,), 0.1),
        northward_wind_stress=jnp.full((4,), -0.2),
    )
    result = VerosExchange()(components, TIME)
    assert bool(jnp.all(result["ocn"]["forcing"].surface_taux > 0))
    assert bool(jnp.all(result["ocn"]["forcing"].surface_tauy < 0))


def test_veros_exchange_conserves_momentum_through_a_conservative_regrid():
    """What the atmosphere loses, the ocean gains, area-integrated.

    A toy first-order conservative map from two atmosphere cells onto four
    ocean cells of half the area (each ocean cell takes its parent's value)
    preserves each flux's area integral exactly; the exchange must pass the
    stress through that map without any rescaling of its own, so the
    area-integrated momentum on the ocean side equals the atmosphere's.
    """
    atm_area = jnp.array([2.0, 2.0])
    ocn_area = jnp.array([1.0, 1.0, 1.0, 1.0])

    def a2o_flux(value):
        return jnp.repeat(value, 2)

    components = _fake_components()
    components["atm"]["derived"] = _AtmDerived(
        eastward_wind_stress=jnp.array([0.12, -0.04]),
        northward_wind_stress=jnp.array([0.03, 0.05]),
        total_heat_flux=jnp.array([20.0, 30.0]),
        total_freshwater_flux=jnp.array([1e-6, 2e-6]),
    )
    components["atm"]["forcing"] = _AtmForcing(
        sea_surface_temperature=jnp.full((2,), 290.0))

    def o2a_state(value):
        return value.reshape(2, 2).mean(axis=1)

    exchange = VerosExchange(regrid={"a2o_flux": a2o_flux, "o2a_state": o2a_state})
    result = exchange(components, TIME)
    derived = components["atm"]["derived"]
    forcing = result["ocn"]["forcing"]
    np.testing.assert_allclose(
        jnp.sum(forcing.surface_taux * ocn_area),
        jnp.sum(derived.eastward_wind_stress * atm_area))
    np.testing.assert_allclose(
        jnp.sum(forcing.surface_tauy * ocn_area),
        jnp.sum(derived.northward_wind_stress * atm_area))


def test_veros_exchange_rotates_the_stress_into_the_ocean_frame(tmp_path):
    """On a rotated grid the stress is rotated as a vector, after the regrid.

    Uses a small SCRIP file carrying rotation angles only, so the exchange
    runs end to end on a shape the test controls: the result is
    `rotate_vector` of the published stress, and each cell keeps its stress
    magnitude (the same momentum, expressed in the grid's own frame).
    """
    import xarray as xr

    angle = np.array([0.0, np.pi / 6, np.pi / 2, -np.pi / 4])
    path = tmp_path / "rotated.SCRIP.nc"
    xr.Dataset({
        "grid_dims": ("grid_rank", np.array([4, 1], dtype=np.int32)),
        "grid_cos_angle": ("grid_size", np.cos(angle)),
        "grid_sin_angle": ("grid_size", np.sin(angle)),
    }).to_netcdf(path)

    components = _fake_components()
    components = dict(components, atm=dict(
        components["atm"],
        derived=_AtmDerived(
            eastward_wind_stress=jnp.full((4, 1), 0.1),
            northward_wind_stress=jnp.full((4, 1), 0.05),
            total_heat_flux=jnp.full((4, 1), 20.0),
            total_freshwater_flux=jnp.full((4, 1), 1e-6),
        ),
    ))
    result = VerosExchange(rotation_grid_file=str(path))(components, TIME)
    taux = result["ocn"]["forcing"].surface_taux
    tauy = result["ocn"]["forcing"].surface_tauy

    expected_x, expected_y = rotate_vector(
        jnp.full((4, 1), 0.1), jnp.full((4, 1), 0.05),
        jnp.asarray(np.cos(angle)[:, None]), jnp.asarray(np.sin(angle)[:, None]))
    np.testing.assert_allclose(taux, expected_x, rtol=1e-6)
    np.testing.assert_allclose(tauy, expected_y, rtol=1e-6)
    np.testing.assert_allclose(jnp.hypot(taux, tauy), jnp.hypot(0.1, 0.05),
                               rtol=1e-6)
    # Zero angle is the identity; a quarter turn puts the eastward stress on
    # the grid's -y axis and the northward stress on its +x axis.
    np.testing.assert_allclose(taux[0], 0.1, rtol=1e-6)
    np.testing.assert_allclose(taux[2], 0.05, rtol=1e-6)
    np.testing.assert_allclose(tauy[2], -0.1, rtol=1e-6)


def test_veros_exchange_stress_is_differentiable():
    """The ocean's stress has a unit, finite gradient in the atmosphere's.

    The exchange is linear in the stress, so the derivative of the total
    stress the ocean receives with respect to the published eastward stress
    is exactly one per cell -- the gradient path from the ocean's momentum
    forcing back into the atmosphere's surface closure is unbroken.
    """
    components = _fake_components()

    def total_ocean_stress(eastward):
        atm = dict(components["atm"], derived=components["atm"]["derived"].replace(
            eastward_wind_stress=eastward))
        result = VerosExchange()(dict(components, atm=atm), TIME)
        return jnp.sum(result["ocn"]["forcing"].surface_taux)

    gradient = jax.grad(total_ocean_stress)(
        components["atm"]["derived"].eastward_wind_stress)
    np.testing.assert_array_equal(gradient, jnp.ones(4))


def test_veros_exchange_rotates_when_given_a_rotation_grid_file():
    """A `rotation_grid_file` reads and holds real (non-identity) rotation angles.

    A full end-to-end call would need shapes that actually match the
    packaged grid, which is beside the point here: this only checks that
    `__init__` loaded real angles rather than staying on the no-rotation
    identity path.
    """
    exchange = VerosExchange(rotation_grid_file=ROTATED_SCRIP_FILE)
    assert exchange.rotation_grid_file == ROTATED_SCRIP_FILE
    assert exchange._rotation_angles is not None
    _, sin_angle = exchange._rotation_angles
    assert not bool(jnp.all(sin_angle == 0.0))


def test_veros_exchange_repr_names_its_regridders_and_rotation():
    """`repr` says how the ocean is driven -- which regridders, which rotation."""
    exchange = VerosExchange(rotation_grid_file=ROTATED_SCRIP_FILE)
    text = repr(exchange)
    assert "VerosExchange" in text
    assert ROTATED_SCRIP_FILE in text
    assert "a2o_flux=identity" in text  # no regrid was given to this instance
    assert "o2a_state=identity" in text
    assert "rotates=yes" in text
