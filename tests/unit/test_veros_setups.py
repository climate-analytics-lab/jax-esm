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

from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("veros")

from jem.components import veros_component  # noqa: E402, F401

DATA = resources.files("jem.data")
DOUBLE_DRAKE_MASK_FILE = str(DATA / "terrain_double_drake_T31.nc")
ROTATED_SCRIP_FILE = str(DATA / "RotatedGaussianLatLon.SCRIP.nc")
ROTATED_LANDSEA_MASK_FILE = str(DATA / "landsea_mask_fraction_RotatedGaussianLatLon.nc")

#: `jax_enable_x64` as it was before this process ever ran a test that
#: imports a Veros setup module -- i.e. before anything in this file, since
#: it only imports one inside a test function (see the module docstring).
_JAX_X64_BEFORE_ANY_VEROS_SETUP_IMPORT = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True, scope="module")
def _restore_jax_x64_after_this_module():
    """Set `jax_enable_x64` for this module's tests, and restore it after.

    Whichever of this module's tests actually runs -- the one fast test
    alone under `-m "not slow"`, or all five together (the Veros slow-test
    gate, jax-esm#113) -- needs `jax_enable_x64` on for every test *in this
    module*, so nothing restores it mid-module; restoring after each test
    individually broke the later slow tests, which then found Veros silently
    degraded to float32 precision. This restores it only once, in this
    fixture's teardown, which runs after the *last* selected test in the
    module finishes, protecting whichever unrelated test file shares this
    worker afterward without disturbing anything Veros does for the rest of
    this module's own run.

    The setup half explicitly sets it to `True` too, not only restores it in
    teardown: under `-n N --dist load` (the xdist default), a worker that
    interleaves this module's tests with another's tears this fixture down
    and back up again between them, and the *first* test to run after such a
    re-entry would otherwise find `jax_enable_x64` however the previous
    fixture invocation's teardown left it (`False`, ordinarily) instead of
    the `True` Veros needs -- silently degrading precision rather than
    failing loudly, exactly the class of bug this module exists to avoid.
    """
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _JAX_X64_BEFORE_ANY_VEROS_SETUP_IMPORT)


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
def test_double_drake_setup_rejects_a_fractional_mask(tmp_path):
    """A non-binary `lsm` is refused, naming the file and the value range.

    `double_drake_setup` does not threshold a fractional mask the way
    `earth_setup` does, so `(1 - lsm)` on one would silently give a
    fractional `kbot`; this is checked before any of that arithmetic runs.
    """
    from jem.components.veros.setups.double_drake import double_drake_setup

    reference = xr.open_dataset(DOUBLE_DRAKE_MASK_FILE)
    fractional = reference.copy(deep=True)
    fractional["lsm"] = fractional["lsm"] * 0 + 0.5
    mask_file = tmp_path / "fractional_mask.nc"
    fractional.to_netcdf(mask_file)
    with pytest.raises(ValueError, match="binary 0/1"):
        double_drake_setup(land_sea_mask_file=str(mask_file))


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


@pytest.mark.slow
def test_earth_setup_declares_its_cells_true_positions():
    """The rotated setup publishes where its cells are, for the output to use.

    Veros' own axes are the pre-rotation ones, so `VerosComponent` can only
    label the output correctly if the setup says where each cell really is.
    """
    from jem.components.veros.setups.earth import earth_setup

    setup_cls = earth_setup(
        scrip_grid_file=ROTATED_SCRIP_FILE,
        landsea_mask_file=ROTATED_LANDSEA_MASK_FILE,
    )

    grid = xr.open_dataset(ROTATED_SCRIP_FILE)
    nlon, nlat = (int(n) for n in grid["grid_dims"].to_numpy())
    for declared, name in (
        (setup_cls.geographic_latitude, "grid_center_lat"),
        (setup_cls.geographic_longitude, "grid_center_lon"),
    ):
        expected = grid[name].to_numpy().reshape(nlat, nlon).transpose()
        np.testing.assert_array_equal(declared, expected)
    # And they are not the native axis: the pole is displaced.
    native_lat = grid["native_lat"].to_numpy()
    assert np.abs(setup_cls.geographic_latitude - native_lat[None, :]).max() > 5.0


@pytest.mark.slow
def test_earth_setup_coriolis_is_periodic_across_the_seam():
    """The Coriolis parameter's zonal halo holds the other side of the seam.

    Veros' Coriolis tendency for `u` averages `coriolis_t` with its eastern
    neighbour, which for the last interior column is a ghost cell. Veros
    never fills that halo itself, so a setup that writes the interior only
    leaves it at zero and halves the Coriolis force on one meridian of zonal
    velocities. With a uniform northward flow the tendency there must be the
    full ``f v``, as it is everywhere else.
    """
    from veros.core import momentum
    from veros.core.operators import at, update

    from jem.components.veros.setups.earth import earth_setup

    setup_cls = earth_setup(
        scrip_grid_file=ROTATED_SCRIP_FILE,
        landsea_mask_file=ROTATED_LANDSEA_MASK_FILE,
    )
    model = setup_cls()
    model.setup()
    state = model.state
    vs = state.variables

    coriolis = np.asarray(vs.coriolis_t)
    np.testing.assert_array_equal(coriolis[-2:, 2:-2], coriolis[2:4, 2:-2])
    np.testing.assert_array_equal(coriolis[:2, 2:-2], coriolis[-4:-2, 2:-2])
    assert np.abs(coriolis[-2:, 2:-2]).max() > 0.0

    with vs.unlock():
        vs.v = update(vs.v, at[...], 1.0 * vs.maskV[..., np.newaxis])
        vs.u = update(vs.u, at[...], 0.0)
    tendency = np.asarray(momentum.tend_coriolisf(state).du_cor)
    mask_v = np.asarray(vs.maskV)[..., -1]
    # Rows where the four V points around the seam's U point are all wet, so
    # the tendency is the plain average of f over the two T columns.
    seam = -3
    rows = [
        j for j in range(3, coriolis.shape[1] - 3)
        if mask_v[seam, j] and mask_v[seam + 1, j]
        and mask_v[seam, j - 1] and mask_v[seam + 1, j - 1]
        and np.asarray(vs.maskU)[seam, j, -1]
    ]
    assert rows, "the grid has no open-ocean row at the seam to test"
    expected = 0.5 * (coriolis[seam, rows] + coriolis[seam + 1, rows])
    np.testing.assert_allclose(tendency[seam, rows, -1], expected, rtol=1e-3)


@pytest.mark.slow
def test_echam_veros_earth_configuration_steps():
    """`+configuration=veros-earth physics@atmosphere.physics=echam` builds,
    initializes and steps twice, and every atmosphere and ocean field stays
    finite.

    ECHAM's land is the prescribed-moisture tile here: jax-gcm's default
    SLIM land reads surface properties that exist only on T63 and finer
    Gaussian grids, and `veros-earth`'s atmosphere is T31.

    Importing `veros` sets `jax_enable_x64` process-wide, so this traces ECHAM's
    Tiedtke-Nordeng convection and RRTMGP radiation in 64-bit mode -- the
    dtype-sensitive path a slab-coupled ECHAM step (float32) never exercises.
    """
    from hydra import compose, initialize_config_module

    import jem.config  # noqa: F401
    import jem.runners as runners

    with initialize_config_module(config_module="jem.config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[
            "+configuration=veros-earth",
            "physics@atmosphere.physics=echam",
            "atmosphere.physics.land_model=prescribed",
        ])
    coupler = runners.build_coupler(cfg)
    assert set(coupler.components) == {"atm", "ocn"}
    carry = coupler.initialize()
    final, _diagnostics = coupler.generate_trajectory_function(2)(carry)

    assert int(final.step) == 2
    for name, component_carry in final.components.items():
        for path, leaf in jax.tree_util.tree_leaves_with_path(component_carry):
            leaf = jnp.asarray(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.inexact):
                assert bool(jnp.all(jnp.isfinite(leaf))), (
                    f"{name}{jax.tree_util.keystr(path)} is not finite")
