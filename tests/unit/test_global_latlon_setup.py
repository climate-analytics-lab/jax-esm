"""The observation-started global ocean: its vertical grid and initial state."""

import jax
import numpy as np
import pytest

veros = pytest.importorskip("veros")


@pytest.fixture(autouse=True)
def _restore_x64():
    """Leave `jax_enable_x64` as this test found it.

    Initialising Veros' JAX backend (building a setup, importing
    `veros.core`) switches it on process-wide, and the float32 tests that run
    after this one in the same worker must not inherit that -- the same
    snapshot-and-restore `test_fluxes.py` uses.
    """
    previous = jax.config.read("jax_enable_x64")
    yield
    jax.config.update("jax_enable_x64", previous)

from jem.components.veros.setups.global_latlon import (  # noqa: E402
    GLOBAL_LATLON_LAYER_CENTRES,
    blend_observed_sst,
    interpolate_columns,
    layer_thicknesses_from_centres,
    stretched_layer_centres,
)


def veros_t_points(thickness_surface_first):
    """Veros' own T-point depths for given thicknesses (`u_centered_grid`)."""
    from veros.core.numerics import u_centered_grid
    import jax.numpy as jnp

    dzt = jnp.asarray(np.asarray(thickness_surface_first)[::-1])
    _, zt, zw = u_centered_grid(dzt, jnp.zeros_like(dzt), jnp.zeros_like(dzt), jnp.zeros_like(dzt))
    return -(np.asarray(zt) - np.asarray(zw)[-1])[::-1]


def test_veros_puts_its_t_points_exactly_at_the_requested_centres():
    thickness = layer_thicknesses_from_centres(GLOBAL_LATLON_LAYER_CENTRES)
    assert np.all(thickness > 0)
    np.testing.assert_allclose(veros_t_points(thickness), GLOBAL_LATLON_LAYER_CENTRES, atol=1e-9)


def test_the_default_column_resolves_the_mixed_layer():
    centres = np.asarray(GLOBAL_LATLON_LAYER_CENTRES)
    np.testing.assert_allclose(centres[:5], [5, 15, 25, 35, 45])
    np.testing.assert_allclose(layer_thicknesses_from_centres(centres)[:4], 10.0)
    assert np.all(np.diff(centres) > 0) and centres[-1] > 4000
    assert stretched_layer_centres(n=3, n_uniform=2, growth=2.0) == (5.0, 15.0, 35.0)


def test_interpolation_carries_the_deepest_wet_value_down():
    source_depth = np.array([5.0, 15.0, 30.0])
    field = np.zeros((2, 1, 3))
    field[0, 0] = [20.0, 10.0, 0.0]  # the third level is dry in the source
    out = interpolate_columns(field, source_depth, np.array([0.0, 10.0, 40.0]))
    np.testing.assert_allclose(out[0, 0], [20.0, 15.0, 10.0])
    np.testing.assert_array_equal(out[1, 0], 0.0)  # land stays land


def test_observed_sst_replaces_the_mixed_layer_and_tapers_below():
    depth = np.array([5.0, 40.0, 75.0, 150.0])
    temperature = np.array([[[15.0, 14.0, 12.0, 8.0]]])
    sst = np.array([[17.0]])
    out = blend_observed_sst(temperature, sst, depth, mixed_layer_depth=50.0)
    np.testing.assert_allclose(out[0, 0], [17.0, 16.0, 12.0 + 2.0 * 0.5, 8.0])
    # No observation: the climatology is kept.
    np.testing.assert_allclose(blend_observed_sst(temperature, np.array([[np.nan]]), depth, 50.0), temperature)


@pytest.fixture(scope="module")
def climatology(tmp_path_factory):
    """Write a 10-degree stand-in for Veros' ``global_1deg`` asset, in the same layout."""
    import xarray as xr

    xt = 95.0 + 10.0 * np.arange(36)
    yt = -75.0 + 10.0 * np.arange(16)
    zt = -np.array([5.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    lat = yt[:, None] * np.ones((16, 36))
    bathymetry = np.where(np.abs(lat) < 60, -3000.0, 0.0)
    bathymetry[:, 5:8] = 0.0  # a continent
    bathymetry[:, 20] = -100.0  # a shelf
    depth = -zt[:, None, None]
    wet = (bathymetry[None] < 0) & (depth < -bathymetry[None] + 1e-9)
    temperature = np.where(wet, 25.0 - 20.0 * np.abs(lat[None]) / 90.0 - depth / 500.0, 0.0)
    salinity = np.where(wet, 35.0, 0.0)
    path = tmp_path_factory.mktemp("veros") / "forcing.nc"
    xr.Dataset(
        {
            "temperature": (("zt", "yt", "xt"), temperature.astype("float32")),
            "salinity": (("zt", "yt", "xt"), salinity.astype("float32")),
            "bathymetry": (("yt", "xt"), bathymetry.astype("float32")),
        },
        coords={"xt": xt, "yt": yt, "zt": zt},
    ).to_netcdf(path)
    return str(path), xt, yt, bathymetry


def test_setup_matches_the_climatology_grid_and_takes_the_observed_sst(climatology):
    from jem.components.veros.setups.global_latlon import global_latlon_setup

    path, xt, yt, bathymetry = climatology
    sst = np.full((36, 16), np.nan)
    sst[0, 8] = 30.0  # one observed cell
    setup = global_latlon_setup(
        climatology=path, sea_surface_temperature=sst,
        layer_centres=(5.0, 15.0, 25.0, 40.0, 70.0, 150.0, 400.0, 1000.0, 2500.0),
    )()
    setup.setup()
    vs = setup.state.variables
    interior = slice(2, -2)
    # The T points are the asset's own cell centres.
    np.testing.assert_allclose(np.asarray(vs.xt[interior]), xt)
    np.testing.assert_allclose(np.asarray(vs.yt[interior]), yt)
    np.testing.assert_allclose(-np.asarray(vs.zt)[::-1], (5.0, 15.0, 25.0, 40.0, 70.0, 150.0, 400.0, 1000.0, 2500.0))
    kbot = np.asarray(vs.kbot[interior, interior])
    is_ocean = bathymetry.T < 0
    np.testing.assert_array_equal(kbot > 0, is_ocean)
    # The 100 m shelf holds the layers centred above it (5..70 m): 5 of 9.
    assert kbot[20, 8] == 9 - 5 + 1
    top = np.asarray(vs.temp[interior, interior, -1, 1])
    assert top[0, 8] == pytest.approx(30.0)
    # Elsewhere the climatology's top value, interpolated to 5 m.
    assert top[1, 8] == pytest.approx(25.0 - 20.0 * 5.0 / 90.0 - 5.0 / 500.0, rel=1e-6)


def test_setup_refuses_an_sst_on_another_grid(climatology):
    from jem.components.veros.setups.global_latlon import global_latlon_setup

    with pytest.raises(ValueError, match="this grid is"):
        global_latlon_setup(climatology=climatology[0], sea_surface_temperature=np.zeros((10, 10)))
