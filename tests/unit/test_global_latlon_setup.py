"""The observation-started global ocean: its vertical grid and initial state."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

veros = pytest.importorskip("veros")


def _run_isolated(code: str) -> dict:
    """Run ``code`` in a fresh interpreter and return the dict it prints as JSON.

    Building a Veros setup (or importing ``veros.core``) initialises Veros'
    JAX backend, a once-per-process event that switches ``jax_enable_x64``
    on for the whole process. Doing that here, at a point in the suite that
    depends on test order, would leave every later float32 test running in
    float64 -- and switching it back off afterwards would leave every later
    Veros test running in float32. A subprocess touches neither.
    """
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])

from jem.components.veros.setups.global_latlon import (  # noqa: E402
    GLOBAL_LATLON_LAYER_CENTRES,
    blend_observed_sst,
    interpolate_columns,
    layer_thicknesses_from_centres,
    stretched_layer_centres,
)


def veros_t_points(thickness_surface_first):
    """Veros' own T-point depths for given thicknesses (`u_centered_grid`)."""
    out = _run_isolated(f"""
        import json
        import numpy as np
        from jem.components import veros_component  # Veros on the JAX backend
        import jax.numpy as jnp
        from veros.core.numerics import u_centered_grid
        dzt = jnp.asarray(np.asarray({list(map(float, thickness_surface_first))})[::-1])
        _, zt, zw = u_centered_grid(dzt, jnp.zeros_like(dzt), jnp.zeros_like(dzt), jnp.zeros_like(dzt))
        print(json.dumps({{"zt": (-(np.asarray(zt) - np.asarray(zw)[-1])[::-1]).tolist()}}))
    """)
    return np.asarray(out["zt"])


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
    path, xt, yt, bathymetry = climatology
    out = _run_isolated(f"""
        import json
        import numpy as np
        from jem.components.veros.setups.global_latlon import global_latlon_setup
        sst = np.full((36, 16), np.nan)
        sst[0, 8] = 30.0  # one observed cell
        setup = global_latlon_setup(
            climatology={str(path)!r}, sea_surface_temperature=sst,
            layer_centres=(5.0, 15.0, 25.0, 40.0, 70.0, 150.0, 400.0, 1000.0, 2500.0),
        )()
        setup.setup()
        vs = setup.state.variables
        interior = slice(2, -2)
        print(json.dumps(dict(
            xt=np.asarray(vs.xt[interior]).tolist(),
            yt=np.asarray(vs.yt[interior]).tolist(),
            zt=(-np.asarray(vs.zt)[::-1]).tolist(),
            kbot=np.asarray(vs.kbot[interior, interior]).tolist(),
            top=np.asarray(vs.temp[interior, interior, -1, 1]).tolist(),
        )))
    """)
    # The T points are the asset's own cell centres.
    np.testing.assert_allclose(out["xt"], xt)
    np.testing.assert_allclose(out["yt"], yt)
    np.testing.assert_allclose(out["zt"], (5.0, 15.0, 25.0, 40.0, 70.0, 150.0, 400.0, 1000.0, 2500.0))
    kbot = np.asarray(out["kbot"])
    is_ocean = bathymetry.T < 0
    np.testing.assert_array_equal(kbot > 0, is_ocean)
    # The 100 m shelf holds the layers centred above it (5..70 m): 5 of 9.
    assert kbot[20, 8] == 9 - 5 + 1
    top = np.asarray(out["top"])
    assert top[0, 8] == pytest.approx(30.0)
    # Elsewhere the climatology's top value, interpolated to 5 m.
    assert top[1, 8] == pytest.approx(25.0 - 20.0 * 5.0 / 90.0 - 5.0 / 500.0, rel=1e-6)


def test_setup_refuses_an_sst_on_another_grid(climatology):
    from jem.components.veros.setups.global_latlon import global_latlon_setup

    with pytest.raises(ValueError, match="this grid is"):
        global_latlon_setup(climatology=climatology[0], sea_surface_temperature=np.zeros((10, 10)))
