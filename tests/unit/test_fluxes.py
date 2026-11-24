"""Tests for :mod:`jem.fluxes` -- the computed half of the Veros coupling.

The pure functions (:func:`~jem.fluxes.mask_fluxes_under_ice`,
:func:`~jem.fluxes.rotate_vector`,
:func:`~jem.fluxes.read_rotation_angles`) are tested directly on plain
arrays. :class:`~jem.fluxes.VerosExchange` is tested on small fake carries --
:mod:`flax.struct` dataclasses carrying exactly the field names it reads and
writes -- rather than on a real atmosphere or ocean, since nothing here needs
either model to be built.
"""

import dataclasses
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


def test_the_ice_mask_uses_the_coupled_model_s_freezing_point(monkeypatch):
    """With no freezing point given, the mask reads
    ``jem.constants.seawater_freezing_point_K`` when it runs -- the freezing
    point the slab ocean and slab ice use -- so an override reaches it.
    """
    from jem import constants

    sst = jnp.array([271.30])
    heat_flux = jnp.array([50.0])
    freshwater_flux = jnp.array([1e-6])
    masked, _ = mask_fluxes_under_ice(sst, heat_flux, freshwater_flux)
    np.testing.assert_allclose(masked, 0.0)

    monkeypatch.setattr(
        constants, "surface_constants",
        dataclasses.replace(constants.surface_constants, seawater_freezing_point_K=271.2))
    masked, _ = mask_fluxes_under_ice(sst, heat_flux, freshwater_flux)
    np.testing.assert_allclose(masked, 50.0)


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
    sea_eastward_wind_stress: jnp.ndarray
    sea_northward_wind_stress: jnp.ndarray
    sea_total_heat_flux: jnp.ndarray
    sea_total_freshwater_flux: jnp.ndarray
    sea_fraction: jnp.ndarray


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
    sea_fraction: jnp.ndarray


def _fake_components():
    shape = (4,)
    atm = {
        "derived": _AtmDerived(
            sea_eastward_wind_stress=jnp.array([0.1, -0.05, 0.02, 0.0]),
            sea_northward_wind_stress=jnp.array([-0.03, 0.04, 0.0, 0.01]),
            sea_total_heat_flux=jnp.full(shape, 20.0),
            sea_total_freshwater_flux=jnp.full(shape, 1e-6),
            sea_fraction=jnp.ones(shape),
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
        "derived": _OcnDerived(sea_surface_temperature=jnp.full(shape, 285.0),
                               sea_fraction=jnp.ones(shape)),
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
                sea_eastward_wind_stress=jnp.full(shape, 0.1, dtype=jnp.float32),
                sea_northward_wind_stress=jnp.full(shape, -0.03, dtype=jnp.float32),
                sea_total_heat_flux=jnp.full(shape, 20.0, dtype=jnp.float32),
                sea_total_freshwater_flux=jnp.full(shape, 1e-6, dtype=jnp.float32),
                sea_fraction=jnp.ones(shape, dtype=jnp.float32),
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
                sea_surface_temperature=jnp.full(shape, 285.0, dtype=jnp.float64),
                sea_fraction=jnp.ones(shape, dtype=jnp.float64),
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
    # -- the stress is a momentum flux, so it is regridded like one -- each
    # weighted by the sea part of the cell: R(fraction), R(fraction * field)
    # and R(field), the fallback where no sea is mapped (regrid_weighted).
    assert len(a2o_calls) == 4 * 3
    derived = components["atm"]["derived"]
    for field in (derived.sea_eastward_wind_stress, derived.sea_northward_wind_stress,
                  derived.sea_total_heat_flux, derived.sea_total_freshwater_flux):
        assert any(call is field for call in a2o_calls)
    assert any(call is derived.sea_fraction for call in a2o_calls)
    # The sea surface temperature crosses o2a weighted by Veros' wet mask:
    # R(wet), R(wet * SST) and the fallback R(SST); the first-step mask check
    # maps the wet mask once more, R(wet).
    assert len(o2a_calls) == 4
    assert any(call is components["ocn"]["derived"].sea_surface_temperature
               for call in o2a_calls)


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
        components["atm"]["derived"].sea_eastward_wind_stress,
    )
    np.testing.assert_array_equal(
        result["ocn"]["forcing"].surface_tauy,
        components["atm"]["derived"].sea_northward_wind_stress,
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
        sea_eastward_wind_stress=jnp.full((4,), 0.1),
        sea_northward_wind_stress=jnp.full((4,), -0.2),
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
        sea_eastward_wind_stress=jnp.array([0.12, -0.04]),
        sea_northward_wind_stress=jnp.array([0.03, 0.05]),
        sea_total_heat_flux=jnp.array([20.0, 30.0]),
        sea_total_freshwater_flux=jnp.array([1e-6, 2e-6]),
        sea_fraction=jnp.ones((2,)),
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
        jnp.sum(derived.sea_eastward_wind_stress * atm_area))
    np.testing.assert_allclose(
        jnp.sum(forcing.surface_tauy * ocn_area),
        jnp.sum(derived.sea_northward_wind_stress * atm_area))


def test_veros_exchange_takes_nothing_from_an_atmosphere_cell_with_no_sea():
    """On a mixed grid a coastal ocean cell gets the flux over sea only.

    Two atmosphere cells map onto one ocean cell: one all sea, one all land
    whose "sea" flux was computed against a surface with no sea under it. The
    ocean takes the sea cell's flux and stress, not the area mean of the two.
    """
    def a2o_flux(value):
        return value.mean(keepdims=True)

    def o2a_state(value):
        return jnp.repeat(value, 2)

    land = 500.0
    components = {
        "atm": {
            "derived": _AtmDerived(
                sea_eastward_wind_stress=jnp.array([0.1, land]),
                sea_northward_wind_stress=jnp.array([-0.05, land]),
                sea_total_heat_flux=jnp.array([20.0, land]),
                sea_total_freshwater_flux=jnp.array([1e-6, land]),
                sea_fraction=jnp.array([1.0, 0.0]),
            ),
            "forcing": _AtmForcing(sea_surface_temperature=jnp.full((2,), 290.0)),
        },
        "ocn": {
            "forcing": _OcnForcing(*(jnp.zeros((1,)) for _ in range(4))),
            "derived": _OcnDerived(sea_surface_temperature=jnp.full((1,), 285.0),
                                   sea_fraction=jnp.ones((1,))),
        },
    }
    exchange = VerosExchange(regrid={"a2o_flux": a2o_flux, "o2a_state": o2a_state})
    forcing = exchange(components, TIME)["ocn"]["forcing"]
    np.testing.assert_allclose(forcing.surface_taux, [0.1])
    np.testing.assert_allclose(forcing.surface_tauy, [-0.05])
    np.testing.assert_allclose(forcing.heat_flux, [20.0])
    np.testing.assert_allclose(forcing.freshwater_flux, [1e-6])


def test_veros_exchange_hands_the_atmosphere_the_sst_of_the_water():
    """An atmosphere cell over one wet and one land Veros column takes the
    wet column's SST, not its mean with the land's fill value.
    """
    def a2o_flux(value):
        return jnp.repeat(value, 2)

    def o2a_state(value):
        return value.mean(keepdims=True)

    components = {
        "atm": {
            "derived": _AtmDerived(
                *(jnp.zeros((1,)) for _ in range(4)), sea_fraction=jnp.ones((1,))),
            "forcing": _AtmForcing(sea_surface_temperature=jnp.full((1,), 280.0)),
        },
        "ocn": {
            "forcing": _OcnForcing(*(jnp.zeros((2,)) for _ in range(4))),
            "derived": _OcnDerived(sea_surface_temperature=jnp.array([290.0, 273.15]),
                                   sea_fraction=jnp.array([1.0, 0.0])),
        },
    }
    exchange = VerosExchange(regrid={"a2o_flux": a2o_flux, "o2a_state": o2a_state})
    result = exchange(components, TIME)
    np.testing.assert_allclose(result["atm"]["forcing"].sea_surface_temperature, [290.0])


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
            sea_eastward_wind_stress=jnp.full((4, 1), 0.1),
            sea_northward_wind_stress=jnp.full((4, 1), 0.05),
            sea_total_heat_flux=jnp.full((4, 1), 20.0),
            sea_total_freshwater_flux=jnp.full((4, 1), 1e-6),
            sea_fraction=jnp.ones((4, 1)),
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
            sea_eastward_wind_stress=eastward))
        result = VerosExchange()(dict(components, atm=atm), TIME)
        return jnp.sum(result["ocn"]["forcing"].surface_taux)

    gradient = jax.grad(total_ocean_stress)(
        components["atm"]["derived"].sea_eastward_wind_stress)
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


@struct.dataclass
class _LandForcing:
    precipitation: jnp.ndarray
    evaporation: jnp.ndarray


@struct.dataclass
class _LandDerived:
    soil_water_availability: jnp.ndarray
    runoff_mass_rate: jnp.ndarray


@struct.dataclass
class _AtmLandDerived:
    sea_eastward_wind_stress: jnp.ndarray
    sea_northward_wind_stress: jnp.ndarray
    sea_total_heat_flux: jnp.ndarray
    sea_total_freshwater_flux: jnp.ndarray
    sea_fraction: jnp.ndarray
    precipitation: jnp.ndarray
    land_evaporation: jnp.ndarray


@struct.dataclass
class _AtmLandForcing:
    sea_surface_temperature: jnp.ndarray
    soilw_am: jnp.ndarray


@struct.dataclass
class _OcnRunoffForcing:
    surface_taux: jnp.ndarray
    surface_tauy: jnp.ndarray
    heat_flux: jnp.ndarray
    freshwater_flux: jnp.ndarray
    runoff_mass_rate: jnp.ndarray


def test_veros_exchange_couples_a_land_with_the_table_s_rows():
    """With a land present the exchanger makes the land rows the declarative
    Veros table has: precipitation and the land tile's evaporation to the
    land, its availability to the atmosphere, its runoff to the ocean.
    """
    shape = (4,)
    components = {
        "atm": {
            "derived": _AtmLandDerived(
                *(jnp.zeros(shape) for _ in range(4)), sea_fraction=jnp.ones(shape),
                precipitation=jnp.full(shape, 3e-5),
                land_evaporation=jnp.full(shape, 1e-5)),
            "forcing": _AtmLandForcing(jnp.full(shape, 290.0), jnp.zeros(shape)),
        },
        "ocn": {
            "forcing": _OcnRunoffForcing(*(jnp.zeros(shape) for _ in range(4)),
                                         runoff_mass_rate=jnp.zeros(())),
            "derived": _OcnDerived(sea_surface_temperature=jnp.full(shape, 285.0),
                                   sea_fraction=jnp.ones(shape)),
        },
        "lnd": {
            "forcing": _LandForcing(jnp.zeros(shape), jnp.zeros(shape)),
            "derived": _LandDerived(jnp.full(shape, 0.4), jnp.asarray(2.5e6)),
        },
    }
    result = VerosExchange()(components, TIME)
    np.testing.assert_allclose(result["lnd"]["forcing"].precipitation, 3e-5)
    np.testing.assert_allclose(result["lnd"]["forcing"].evaporation, 1e-5)
    np.testing.assert_allclose(result["atm"]["forcing"].soilw_am, 0.4)
    np.testing.assert_allclose(result["ocn"]["forcing"].runoff_mass_rate, 2.5e6)
    # The ocean rows are made as before.
    np.testing.assert_allclose(result["atm"]["forcing"].sea_surface_temperature, 285.0)
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(
        components)


@struct.dataclass
class _ThermalLandForcing:
    total_heat_flux: jnp.ndarray
    precipitation: jnp.ndarray
    evaporation: jnp.ndarray


@struct.dataclass
class _ThermalLandState:
    land_surface_temperature: jnp.ndarray
    snow_cover: jnp.ndarray


@struct.dataclass
class _AtmThermalLandDerived:
    sea_eastward_wind_stress: jnp.ndarray
    sea_northward_wind_stress: jnp.ndarray
    sea_total_heat_flux: jnp.ndarray
    sea_total_freshwater_flux: jnp.ndarray
    sea_fraction: jnp.ndarray
    precipitation: jnp.ndarray
    land_evaporation: jnp.ndarray
    land_total_heat_flux: jnp.ndarray


@struct.dataclass
class _AtmThermalLandForcing:
    sea_surface_temperature: jnp.ndarray
    soilw_am: jnp.ndarray
    stl_am: jnp.ndarray
    snowc_am: jnp.ndarray


def test_veros_exchange_couples_a_land_with_a_temperature_both_ways():
    """A land whose forcing has a heat flux evolves its own temperature: the
    exchanger hands it the land tile's flux and hands the atmosphere its
    temperature and snow cover, the rows of `SLAB_LAND_EXCHANGES`, on top of
    the soil-water rows every land gets. The ocean still takes the sea
    tile's flux, so the two surfaces share the cell's flux between them.
    """
    shape = (4,)
    components = {
        "atm": {
            "derived": _AtmThermalLandDerived(
                jnp.zeros(shape), jnp.zeros(shape),
                sea_total_heat_flux=jnp.full(shape, 40.0),
                sea_total_freshwater_flux=jnp.zeros(shape),
                sea_fraction=jnp.ones(shape),
                precipitation=jnp.full(shape, 3e-5),
                land_evaporation=jnp.full(shape, 1e-5),
                land_total_heat_flux=jnp.full(shape, -25.0)),
            "forcing": _AtmThermalLandForcing(
                jnp.full(shape, 290.0), jnp.zeros(shape),
                stl_am=jnp.full(shape, 280.0), snowc_am=jnp.zeros(shape)),
        },
        "ocn": {
            "forcing": _OcnRunoffForcing(*(jnp.zeros(shape) for _ in range(4)),
                                         runoff_mass_rate=jnp.zeros(())),
            "derived": _OcnDerived(sea_surface_temperature=jnp.full(shape, 285.0),
                                   sea_fraction=jnp.ones(shape)),
        },
        "lnd": {
            "state": _ThermalLandState(
                land_surface_temperature=jnp.full(shape, 301.5),
                snow_cover=jnp.full(shape, 0.25)),
            "forcing": _ThermalLandForcing(*(jnp.zeros(shape) for _ in range(3))),
            "derived": _LandDerived(jnp.full(shape, 0.4), jnp.asarray(2.5e6)),
        },
    }
    result = VerosExchange()(components, TIME)
    np.testing.assert_allclose(result["lnd"]["forcing"].total_heat_flux, -25.0)
    np.testing.assert_allclose(result["atm"]["forcing"].stl_am, 301.5)
    np.testing.assert_allclose(result["atm"]["forcing"].snowc_am, 0.25)
    # The rows every land gets, and the ocean's own, are made as well.
    np.testing.assert_allclose(result["lnd"]["forcing"].precipitation, 3e-5)
    np.testing.assert_allclose(result["lnd"]["forcing"].evaporation, 1e-5)
    np.testing.assert_allclose(result["atm"]["forcing"].soilw_am, 0.4)
    np.testing.assert_allclose(result["ocn"]["forcing"].heat_flux, 40.0)
    np.testing.assert_allclose(result["ocn"]["forcing"].runoff_mass_rate, 2.5e6)
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(
        components)


# ---------------------------------------------------------------------------
# VerosExchange with a sea ice (jax-esm#160)
# ---------------------------------------------------------------------------
#
# The fake ocean and atmosphere carry exactly the fields the ice rows read and
# write; the ice is a real component on a 4 x 3 grid, because what the exchanger
# has to get right is the real carry layouts of both ice models.


@struct.dataclass
class _AtmIceDerived:
    sea_eastward_wind_stress: jnp.ndarray
    sea_northward_wind_stress: jnp.ndarray
    sea_total_heat_flux: jnp.ndarray
    sea_total_freshwater_flux: jnp.ndarray
    sea_fraction: jnp.ndarray
    downward_shortwave_flux: jnp.ndarray
    downward_longwave_flux: jnp.ndarray
    sea_air_temperature: jnp.ndarray
    sea_air_specific_humidity: jnp.ndarray
    wind_speed: jnp.ndarray
    normalized_surface_pressure: jnp.ndarray
    sea_evaporation: jnp.ndarray
    rainfall: jnp.ndarray
    snowfall: jnp.ndarray


@struct.dataclass
class _AtmIceForcing:
    sea_surface_temperature: jnp.ndarray
    sice_am: jnp.ndarray
    sea_ice_temperature: jnp.ndarray


@struct.dataclass
class _OcnIceDerived:
    sea_surface_temperature: jnp.ndarray
    sea_fraction: jnp.ndarray
    ice_frazil_melt_energy: jnp.ndarray
    clamps_at_freezing: jnp.ndarray


def _ice_components(*, ice="winton", clamp=True, sea_surface_temperature=285.0):
    """Return a JCM-shaped atmosphere over a Veros-shaped ocean, with a real ice."""
    from jem.components.slab import SlabSeaiceModel, WintonSeaiceModel
    from tests.unit.slab_test_utils import make_grid

    grid = make_grid()
    shape = grid.shape
    full = lambda value: jnp.full(shape, value)  # noqa: E731
    atm = {
        "derived": _AtmIceDerived(
            sea_eastward_wind_stress=full(0.1),
            sea_northward_wind_stress=full(0.0),
            sea_total_heat_flux=full(30.0),
            sea_total_freshwater_flux=full(2e-6),
            sea_fraction=full(1.0),
            downward_shortwave_flux=full(100.0),
            downward_longwave_flux=full(250.0),
            sea_air_temperature=full(262.0),
            sea_air_specific_humidity=full(1e-3),
            wind_speed=full(5.0),
            normalized_surface_pressure=full(1.0),
            sea_evaporation=full(1e-5),
            rainfall=full(0.0),
            snowfall=full(8e-6),
        ),
        "forcing": _AtmIceForcing(full(290.0), full(0.0), full(271.0)),
    }
    ocn = {
        "forcing": _OcnRunoffForcing(*(jnp.zeros(shape) for _ in range(4)),
                                     runoff_mass_rate=jnp.zeros(())),
        "derived": _OcnIceDerived(
            sea_surface_temperature=full(sea_surface_temperature),
            sea_fraction=1.0 - grid.fractional_mask,
            ice_frazil_melt_energy=full(5e6),
            clamps_at_freezing=jnp.asarray(1.0 if clamp else 0.0),
        ),
    }
    model = (WintonSeaiceModel(grid) if ice == "winton"
             else SlabSeaiceModel(grid, name="seaice"))
    return {"atm": atm, "ocn": ocn, "seaice": model.initialize()}, model


def _with_ice_fluxes(components):
    """Give the Winton ice's derived heat and freshwater fluxes recognisable values."""
    ice = components["seaice"]
    shape = ice["derived"].ice_fraction.shape
    derived = ice["derived"].replace(
        ocean_heat_flux_up=jnp.full(shape, -7.0),
        ocean_freshwater_flux_up=jnp.full(shape, 3e-6),
        ice_fraction=jnp.full(shape, 0.4),
        ice_surface_temperature_K=jnp.full(shape, 255.0),
        ice_area_fraction=jnp.full(shape, 0.4),
    )
    return dict(components, seaice=dict(ice, derived=derived))


def test_veros_exchange_passes_a_winton_ice_s_fluxes_to_the_ocean_in_place_of_the_atmosphere_s():
    """The ocean's heat and freshwater are the ice's; the swamp mask is not applied.

    At the freezing point, with a cooling flux, the swamp mask would zero both;
    here a real ice takes the deficit, so the ocean gets what the ice passes
    on, whatever the SST.
    """
    components, _ = _ice_components(sea_surface_temperature=271.0)
    components = _with_ice_fluxes(components)
    result = VerosExchange()(components, TIME)
    np.testing.assert_array_equal(result["ocn"]["forcing"].heat_flux, -7.0)
    np.testing.assert_allclose(result["ocn"]["forcing"].freshwater_flux, 3e-6, rtol=1e-6)
    # The wind stress is still the atmosphere's.
    np.testing.assert_allclose(result["ocn"]["forcing"].surface_taux, 0.1)
    # The atmosphere gets the SST, and the ice's fraction and skin temperature.
    np.testing.assert_allclose(result["atm"]["forcing"].sea_surface_temperature, 271.0)
    np.testing.assert_allclose(result["atm"]["forcing"].sice_am, 0.4)
    np.testing.assert_allclose(result["atm"]["forcing"].sea_ice_temperature, 255.0)
    # The ice is forced by the atmosphere's state, the ocean's SST and its
    # freeze/melt energy.
    forcing = result["seaice"]["forcing"]
    np.testing.assert_allclose(forcing.rsds, 100.0)
    np.testing.assert_allclose(forcing.rlds, 250.0)
    np.testing.assert_allclose(forcing.air_temperature, 262.0)
    np.testing.assert_allclose(forcing.total_heat_flux, 30.0)
    np.testing.assert_allclose(forcing.evaporation, 1e-5)
    np.testing.assert_allclose(forcing.frozen_precipitation, 8e-6)
    np.testing.assert_allclose(forcing.sea_surface_temperature, 271.0)
    np.testing.assert_allclose(forcing.ice_frazil_melt_energy, 5e6)
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(components)


def test_veros_exchange_keeps_the_swamp_mask_for_the_slab_ice():
    """An SST-diagnosed ice has no energy budget: the ocean keeps the masked atmosphere flux."""
    components, _ = _ice_components(
        ice="slab", clamp=False, sea_surface_temperature=271.0)
    result = VerosExchange()(components, TIME)
    # +30 W m-2 would cool an ocean already at the freezing point: masked.
    np.testing.assert_array_equal(result["ocn"]["forcing"].heat_flux, 0.0)
    np.testing.assert_array_equal(result["ocn"]["forcing"].freshwater_flux, 0.0)
    np.testing.assert_allclose(
        result["seaice"]["forcing"].sea_surface_temperature, 271.0)
    np.testing.assert_allclose(
        result["atm"]["forcing"].sice_am, components["seaice"]["derived"].ice_fraction)
    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(components)


def test_veros_exchange_weights_the_ice_rows_like_the_default_table():
    """On a mixed grid the ice's fraction is weighted by its sea part, its skin by its ice area.

    The atmosphere's fluxes come over the sea tile, weighted by the
    atmosphere's sea fraction; the fraction is a flux-kind row from the ice
    and the skin temperature is mapped by the same conservative regridder,
    weighted by the ice's area share (``jem.exchangers.ROW_WEIGHTS``).
    """
    components, _ = _ice_components()
    components = _with_ice_fluxes(components)
    calls = []

    def regridder(name):
        def apply(value):
            calls.append((name, value))
            return value
        apply.__name__ = name
        return apply

    exchange = VerosExchange(regrid={
        "a2o_flux": regridder("a2o_flux"), "a2o_state": regridder("a2o_state"),
        "o2a_flux": regridder("o2a_flux"), "o2a_state": regridder("o2a_state"),
    })
    exchange(components, TIME)
    ice = components["seaice"]["derived"]
    used = {(name, id(value)) for name, value in calls}
    assert ("o2a_flux", id(ice.sea_fraction)) in used, "the fraction is weighted by the ice's sea part"
    assert ("o2a_flux", id(ice.ice_area_fraction)) in used, (
        "the skin temperature is weighted by the ice's share of the cell")
    atm = components["atm"]["derived"]
    assert ("a2o_flux", id(atm.downward_shortwave_flux)) in used
    assert ("a2o_flux", id(atm.sea_fraction)) in used


def test_veros_exchange_refuses_a_mixed_grid_without_the_regridders_an_ice_row_needs():
    components, _ = _ice_components()
    exchange = VerosExchange(regrid={"a2o_flux": lambda value: value})
    with pytest.raises(ValueError, match="o2a_flux"):
        exchange(components, TIME)


@pytest.mark.parametrize(
    ("ice", "clamp", "message"),
    [
        ("winton", False, r"clamp_at_freezing=True \(\+ocean\.clamp_at_freezing=true\)"),
        ("slab", True, "no energy budget"),
    ],
)
def test_veros_exchange_validate_refuses_an_ice_the_ocean_s_clamp_does_not_fit(
    ice, clamp, message,
):
    components, _ = _ice_components(ice=ice, clamp=clamp)
    with pytest.raises(ValueError, match=message):
        VerosExchange().validate(components)


def test_veros_exchange_validate_refuses_a_clamp_with_no_ice():
    components, _ = _ice_components()
    del components["seaice"]
    with pytest.raises(ValueError, match="no sea ice to take the deficit"):
        VerosExchange().validate(components)


def test_veros_exchange_validate_accepts_the_matched_pairs():
    for ice, clamp in (("winton", True), ("slab", False)):
        components, _ = _ice_components(ice=ice, clamp=clamp)
        VerosExchange().validate(components)
    components, _ = _ice_components(clamp=False)
    del components["seaice"]
    VerosExchange().validate(components)


def test_veros_exchange_refuses_an_ice_whose_sea_is_not_the_wet_mask():
    components, _ = _ice_components()
    ice = components["seaice"]
    bad = dict(components, seaice=dict(ice, derived=ice["derived"].replace(
        sea_fraction=0.5 * jnp.ones_like(ice["derived"].sea_fraction))))
    with pytest.raises(ValueError, match="differs from Veros' wet mask in 12 cell"):
        VerosExchange().validate(bad)
    VerosExchange.check_ice_mask(np.ones((4, 3)), np.ones((4, 3)))
    with pytest.raises(ValueError, match="wet mask"):
        VerosExchange.check_ice_mask(np.ones((4, 3)), np.ones((3, 4)))


def test_veros_exchange_checks_the_ice_mask_on_the_first_step_of_a_run():
    components, _ = _ice_components()
    ice = components["seaice"]
    bad = dict(components, seaice=dict(ice, derived=ice["derived"].replace(
        sea_fraction=0.5 * jnp.ones_like(ice["derived"].sea_fraction))))
    exchange = VerosExchange()
    with pytest.raises(Exception, match="wet mask"):
        jax.block_until_ready(jax.jit(exchange)(bad, TIME))
    jax.block_until_ready(jax.jit(exchange)(components, TIME))
