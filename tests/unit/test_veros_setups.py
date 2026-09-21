"""Tests for the packaged Veros case setups (:mod:`jem.components.veros.setups`).

Veros is an optional dependency, so the whole module skips when it is
absent. `jem.components.veros_component` is imported first -- before
anything that imports `veros.core` -- because importing it is what points
Veros at its JAX backend (see `test_veros_component.py` for the same
ordering requirement).

The setup modules themselves (`double_drake`, `earth`) are imported inside
each test that needs them, **not** at module scope, and deliberately not
through a shared fixture either. pytest-xdist has every worker *collect*
this whole file (import it, to discover test names) whether or not a given
test ends up assigned to that worker or deselected by `-m "not slow"` --
only *running* a test is filtered by markers, not importing the module that
defines it. Since importing a setup module imports `veros.core`, which flips
the process-global `jax_enable_x64` setting to `True` as a side effect
(Veros runs double precision internally -- see `jem.fluxes.VerosExchange`'s
docstring), a module-level import here would flip it in *every* worker's
process at collection time, before any test runs at all -- contaminating
whichever unrelated tests (`test_accumulate.py`, `test_coupler.py`, ...)
those workers happen to run afterward with a `jax.lax.scan` carry-dtype
mismatch that has nothing to do with Veros. A local import inside a test
function only runs when that specific test is actually selected and
executed, which confines the flip to the one worker that runs a slow Veros
test -- already isolated from the rest of the suite by running in a process
of its own (see `CLAUDE.md`, jax-esm#113).
"""

import jax
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("veros")

from jem.components import veros_component  # noqa: E402

DOUBLE_DRAKE_MASK_FILE = "jem/data/terrain_double_drake_T31.nc"
ROTATED_SCRIP_FILE = "jem/data/RotatedGaussianLatLon.SCRIP.nc"
ROTATED_LANDSEA_MASK_FILE = "jem/data/landsea_mask_fraction_RotatedGaussianLatLon.nc"

#: `jax_enable_x64` as it was before this process ever ran a test that
#: imports a Veros setup module -- i.e. before anything in this file, since
#: it only imports one inside a test function (see the module docstring).
_JAX_X64_BEFORE_ANY_VEROS_SETUP_IMPORT = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True, scope="module")
def _restore_jax_x64_after_this_module():
    """Restore `jax_enable_x64` once every test in this module has run.

    Whichever of this module's tests actually runs -- the one fast test
    alone under `-m "not slow"`, or all five together (the Veros slow-test
    gate, jax-esm#113) -- needs `jax_enable_x64` left exactly as Veros wants
    it (on) for every test *in this module*, so nothing restores it
    mid-module; restoring after each test individually broke the later slow
    tests, which then found Veros silently degraded to float32 precision.
    This restores it only once, in this fixture's teardown, which runs after
    the *last* selected test in the module finishes, protecting whichever
    unrelated test file shares this worker afterward without disturbing
    anything Veros does for the rest of this module's own run.
    """
    yield
    jax.config.update("jax_enable_x64", _JAX_X64_BEFORE_ANY_VEROS_SETUP_IMPORT)


def test_veros_lazy_alias_still_resolves():
    """`jem.components.Veros` still resolves after the setups package is imported.

    `jem.components.veros` (this setups package) and `jem.components.Veros`
    (the lazy alias for `jem.components.veros_component`) differ only in
    case; this checks that importing the former does not confuse the
    latter's `__getattr__` resolution. Importing `.earth` here (rather than
    the parent `jem.components.veros` package, which imports nothing) is
    what actually exercises the case-collision this test is named for, and
    is what flips `jax_enable_x64` -- see `_restore_jax_x64_after_this_module`.
    """
    import jem.components
    import jem.components.veros.setups.earth  # noqa: F401

    assert jem.components.Veros is veros_component


@pytest.mark.slow
def test_double_drake_setup_takes_its_shape_from_the_mask():
    """`nx`/`ny` come from the mask file's own shape, not an argument."""
    from jem.components.veros.setups._layers import LAYER_THICKNESSES
    from jem.components.veros.setups.double_drake import double_drake_setup

    setup_cls = double_drake_setup(land_sea_mask_file=DOUBLE_DRAKE_MASK_FILE)
    model = setup_cls()
    model.setup()
    settings = model.state.settings
    assert (int(settings.nx), int(settings.ny)) == (96, 48)
    assert int(settings.nz) == len(LAYER_THICKNESSES)


@pytest.mark.slow
def test_layer_thicknesses_can_be_shortened():
    """`layer_thicknesses=LAYER_THICKNESSES[:n]` gives an `n`-layer ocean."""
    from jem.components.veros.setups._layers import LAYER_THICKNESSES
    from jem.components.veros.setups.double_drake import double_drake_setup

    setup_cls = double_drake_setup(
        land_sea_mask_file=DOUBLE_DRAKE_MASK_FILE,
        layer_thicknesses=LAYER_THICKNESSES[:3],
    )
    model = setup_cls()
    model.setup()
    assert int(model.state.settings.nz) == 3


@pytest.mark.slow
def test_earth_setup_reproduces_the_native_axes():
    """The `_calibrate_origin` reasoning: `vs.yt`/`vs.xt` reproduce the SCRIP
    file's own native (pre-rotation) axis exactly.

    This is the test that protects `GridInfo`'s grid-spacing reconstruction,
    the most easily-lost part of moving this setup into the package.
    """
    from jem.components.veros.setups.earth import earth_setup

    setup_cls = earth_setup(
        scrip_grid_file=ROTATED_SCRIP_FILE,
        landsea_mask_file=ROTATED_LANDSEA_MASK_FILE,
    )
    model = setup_cls()
    model.setup()
    vs = model.state.variables

    grid = xr.open_dataset(ROTATED_SCRIP_FILE)
    native_lat = grid["native_lat"].to_numpy()
    native_lon = grid["native_lon"].to_numpy()

    np.testing.assert_allclose(np.asarray(vs.yt)[2:-2], native_lat, atol=1e-9)
    np.testing.assert_allclose(np.asarray(vs.xt)[2:-2], native_lon, atol=1e-9)


@pytest.mark.slow
def test_earth_setup_coriolis_uses_the_true_latitude():
    """`coriolis_t` follows `grid_center_lat` (true), not `native_lat` (rotated)."""
    from jem.components.veros.setups.earth import earth_setup

    setup_cls = earth_setup(
        scrip_grid_file=ROTATED_SCRIP_FILE,
        landsea_mask_file=ROTATED_LANDSEA_MASK_FILE,
    )
    model = setup_cls()
    model.setup()
    vs = model.state.variables
    settings = model.state.settings

    grid = xr.open_dataset(ROTATED_SCRIP_FILE)
    nlon, nlat = (int(n) for n in grid["grid_dims"].to_numpy())
    true_lat_xy = grid["grid_center_lat"].to_numpy().reshape(nlat, nlon).transpose()
    expected = 2 * settings.omega * np.sin(true_lat_xy / 180.0 * settings.pi)
    np.testing.assert_allclose(
        np.asarray(vs.coriolis_t)[2:-2, 2:-2], expected, atol=1e-12
    )

    # And *not* what the rotated (native) latitude would give, wherever the
    # rotation actually moves a cell.
    native_lat = grid["native_lat"].to_numpy()
    rotated_guess = 2 * settings.omega * np.sin(
        np.broadcast_to(native_lat[None, :], true_lat_xy.shape) / 180.0 * settings.pi
    )
    assert not np.allclose(
        np.asarray(vs.coriolis_t)[2:-2, 2:-2], rotated_guess, atol=1e-6
    )
