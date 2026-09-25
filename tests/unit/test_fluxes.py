"""Tests for :mod:`jem.fluxes` -- the computed half of the Veros coupling.

The pure functions (:func:`~jem.fluxes.bulk_wind_stress`,
:func:`~jem.fluxes.mask_fluxes_under_ice`, :func:`~jem.fluxes.rotate_vector`,
:func:`~jem.fluxes.read_rotation_angles`) are tested directly on plain
arrays. :class:`~jem.fluxes.VerosExchange` is tested on small fake carries --
:mod:`flax.struct` dataclasses carrying exactly the field names it reads and
writes -- rather than on a real atmosphere or ocean, since nothing here needs
either model to be built.
"""

from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from jem.base.component import CouplingTime
from jem.fluxes import (
    VerosExchange,
    bulk_wind_stress,
    mask_fluxes_under_ice,
    read_rotation_angles,
    rotate_vector,
)

DATA = resources.files("jem.data")
ROTATED_SCRIP_FILE = str(DATA / "RotatedGaussianLatLon.SCRIP.nc")

TIME = CouplingTime(
    step=jnp.int32(0), sim_time=jnp.float32(0.0), dt=86400.0,
    year_offset_seconds=0.0, days_per_year=365.0,
)


# ---------------------------------------------------------------------------
# bulk_wind_stress
# ---------------------------------------------------------------------------


def test_bulk_wind_stress_is_quadratic_in_the_wind():
    """Doubling the wind quadruples the stress, away from the speed floor."""
    u, v = jnp.array([3.0]), jnp.array([4.0])
    taux, tauy = bulk_wind_stress(u, v)
    taux2, tauy2 = bulk_wind_stress(2 * u, 2 * v)
    np.testing.assert_allclose(taux2, 4 * taux, rtol=1e-6)
    np.testing.assert_allclose(tauy2, 4 * tauy, rtol=1e-6)


def test_bulk_wind_stress_is_aligned_with_the_wind():
    """The stress is parallel to (u, v), with magnitude Cd*rho*|U|^2."""
    u, v = jnp.array([3.0]), jnp.array([-4.0])
    drag_coefficient, air_density = 1e-3, 1.22
    taux, tauy = bulk_wind_stress(
        u, v, drag_coefficient=drag_coefficient, air_density=air_density
    )
    speed = jnp.sqrt(u**2 + v**2)
    expected_magnitude = drag_coefficient * air_density * speed**2
    np.testing.assert_allclose(
        jnp.sqrt(taux**2 + tauy**2), expected_magnitude, rtol=1e-6
    )
    # Parallel: the cross product of (u, v) and (taux, tauy) is zero.
    np.testing.assert_allclose(u * tauy - v * taux, 0.0, atol=1e-10)


def test_bulk_wind_stress_gradient_is_finite_at_zero_wind():
    """The regression test for the squared-speed floor.

    Without flooring the squared speed before the square root,
    `d sqrt(x)/dx` is unbounded at `x=0` and this gradient comes back NaN
    even though the primal stress at zero wind is exactly zero.
    """

    def total_stress(uv):
        taux, tauy = bulk_wind_stress(uv[0], uv[1])
        return taux + tauy

    gradient = jax.grad(total_stress)(jnp.array([0.0, 0.0]))
    assert jnp.all(jnp.isfinite(gradient)), gradient


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
    u0: jnp.ndarray
    v0: jnp.ndarray
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


def _fake_components(windless: bool = False):
    """Build fake ``atm``/``ocn`` carries.

    ``windless=True`` gives ``derived.u0``/``.v0`` as ``None`` -- what
    ``jem.components.jcm.exchange_fields.from_diagnostics`` now returns for a
    composed physics package with no near-surface wind *vector* (jax-esm#129,
    e.g. ECHAM) -- rather than a value that only a real ``jcm`` model could
    produce, since nothing here needs one built.
    """
    shape = (4,)
    atm = {
        "derived": _AtmDerived(
            u0=None if windless else jnp.full(shape, 5.0),
            v0=None if windless else jnp.full(shape, -3.0),
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
                u0=jnp.full(shape, 5.0, dtype=jnp.float32),
                v0=jnp.full(shape, -3.0, dtype=jnp.float32),
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

    # u0, v0, total_heat_flux, total_freshwater_flux all cross a2o.
    assert len(a2o_calls) == 4
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
    # With no regridder and no rotation, the wind stress is the bulk drag law
    # applied directly to (u0, v0).
    expected_taux, expected_tauy = bulk_wind_stress(
        components["atm"]["derived"].u0, components["atm"]["derived"].v0
    )
    np.testing.assert_allclose(result["ocn"]["forcing"].surface_taux, expected_taux)
    np.testing.assert_allclose(result["ocn"]["forcing"].surface_tauy, expected_tauy)


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


# ---------------------------------------------------------------------------
# jax-esm#129: no wind vector -- build-time failure, not a silent bad stress
# ---------------------------------------------------------------------------
#
# `jem.components.jcm.exchange_fields.from_diagnostics` returns
# `derived.u0`/`.v0` as `None`, not a value, for a composed atmosphere physics
# package that publishes no near-surface wind vector (today: everything but
# SPEEDY, e.g. ECHAM). `VerosExchange` is the one production consumer of that
# vector (`bulk_wind_stress`), so it -- not `bulk_wind_stress` itself, which
# has no carry to inspect -- is where the absence has to be caught, and it
# must be caught before a coupled run starts, not by computing a stress from
# `None` mid-step.


def test_veros_exchange_validate_rejects_a_windless_atmosphere():
    """Build-time failure: `validate()` names jax-esm#129 for a windless atmosphere.

    This is the check `jem.runners._validate_exchangers` runs, right after
    `Coupler.initialize()` and before a coupled run is ever compiled, for
    every exchanger with a `validate` method -- the same slot
    `jem.exchangers.Exchange.validate` fills for a declarative table.
    """
    components = _fake_components(windless=True)

    with pytest.raises(ValueError, match="jax-esm#129"):
        VerosExchange().validate(components)


def test_veros_exchange_validate_passes_for_a_windy_atmosphere():
    """The happy path: `validate()` is silent when the wind vector is present."""
    VerosExchange().validate(_fake_components())


def test_veros_exchange_validate_names_a_missing_atm_component():
    """A coupled model with no `atm` at all is named, not a bare `KeyError`."""
    with pytest.raises(KeyError, match="atm"):
        VerosExchange().validate({"ocn": _fake_components()["ocn"]})


def test_veros_exchange_call_also_rejects_a_windless_atmosphere():
    """Defence in depth: `__call__` refuses too, not only `validate`.

    A `Coupler` built by hand, bypassing `jem.runners.build_coupler` (the
    only caller of `validate` today), must still fail on its very first step
    rather than silently computing a stress from an absent wind.
    """
    components = _fake_components(windless=True)

    with pytest.raises(ValueError, match="jax-esm#129"):
        VerosExchange()(components, TIME)
