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
ATMOSPHERE_SCRIP_FILE = str(DATA / "JCM_T31.SCRIP.nc")
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
def test_double_drake_setup_takes_its_shape_from_the_atmosphere_grid():
    """`nx`/`ny` come from the atmosphere grid file, not an argument."""
    from jem.components.veros.setups._layers import LAYER_THICKNESSES
    from jem.components.veros.setups.double_drake import double_drake_setup

    setup_cls = double_drake_setup(
        atmosphere_grid_file=ATMOSPHERE_SCRIP_FILE,
        land_sea_mask_file=DOUBLE_DRAKE_MASK_FILE,
    )
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
        double_drake_setup(
            atmosphere_grid_file=ATMOSPHERE_SCRIP_FILE, land_sea_mask_file=str(mask_file)
        )


@pytest.mark.slow
def test_layer_thicknesses_can_be_shortened():
    """`layer_thicknesses=LAYER_THICKNESSES[:n]` gives an `n`-layer ocean."""
    from jem.components.veros.setups._layers import LAYER_THICKNESSES
    from jem.components.veros.setups.double_drake import double_drake_setup

    setup_cls = double_drake_setup(
        atmosphere_grid_file=ATMOSPHERE_SCRIP_FILE,
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


def _jcm_t31_axes_in_degrees():
    """Return the T31 atmosphere's own longitude and latitude axes, as JCM computes them.

    The same ``radians * 180 / pi`` route `jcm.predictions` labels its output
    with, so this is what the atmosphere's dataset holds.
    """
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    lon, sin_lat = get_speedy_coords().horizontal.nodal_axes
    return np.asarray(lon) * 180.0 / np.pi, np.arcsin(np.asarray(sin_lat)) * 180.0 / np.pi


def _double_drake_model():
    from jem.components.veros.setups.double_drake import double_drake_setup

    model = double_drake_setup(
        atmosphere_grid_file=ATMOSPHERE_SCRIP_FILE,
        land_sea_mask_file=DOUBLE_DRAKE_MASK_FILE,
    )()
    model.setup()
    return model


def _earth_model():
    from jem.components.veros.setups.earth import earth_setup

    model = earth_setup(
        scrip_grid_file=ROTATED_SCRIP_FILE, landsea_mask_file=ROTATED_LANDSEA_MASK_FILE
    )()
    model.setup()
    return model


@pytest.mark.slow
def test_double_drake_ocean_cells_are_the_atmospheres_cells():
    """`vs.xt`/`vs.yt` are the atmosphere's longitude and latitude axes.

    The exchange is index-to-index, so an ocean cell has to be where the
    atmosphere cell with the same indices is: a uniform 180/ny latitude grid
    put the rows 2.8-4.7 degrees away (and the first one beyond the pole),
    which applied every flux at the wrong latitude and gave the ocean the
    wrong metric terms. Compared against JCM's own coordinates, not the
    SCRIP file the setup reads, so the file's fidelity is tested too.
    """
    vs = _double_drake_model().state.variables
    atmosphere_lon, atmosphere_lat = _jcm_t31_axes_in_degrees()
    np.testing.assert_allclose(np.asarray(vs.xt)[2:-2], atmosphere_lon, atol=1e-9)
    np.testing.assert_allclose(np.asarray(vs.yt)[2:-2], atmosphere_lat, atol=1e-9)
    # The metric terms follow: cos(lat) at the right rows, no row past a pole.
    assert np.abs(np.asarray(vs.yt)[2:-2]).max() < 90.0
    np.testing.assert_allclose(
        np.asarray(vs.cost)[2:-2], np.cos(np.deg2rad(atmosphere_lat)), atol=1e-9
    )


@pytest.mark.slow
def test_double_drake_land_mask_is_the_terrain_files_land_mask():
    """The ocean's wet cells are the cells the terrain file says are not land."""
    model = _double_drake_model()
    lsm = xr.open_dataset(DOUBLE_DRAKE_MASK_FILE)["lsm"].to_numpy()
    wet = np.asarray(model.state.variables.maskT)[2:-2, 2:-2, -1] > 0
    np.testing.assert_array_equal(wet, lsm == 0)
    assert wet.any() and not wet.all()


@pytest.mark.slow
def test_double_drake_refuses_a_mask_on_another_grid(tmp_path):
    """A terrain file whose shape is not the atmosphere grid's is refused by name."""
    from jem.components.veros.setups.double_drake import double_drake_setup

    coarse = xr.open_dataset(DOUBLE_DRAKE_MASK_FILE).isel(lon=slice(0, 48))
    mask_file = tmp_path / "half_the_longitudes.nc"
    coarse.to_netcdf(mask_file)
    with pytest.raises(ValueError, match="same grid"):
        double_drake_setup(
            atmosphere_grid_file=ATMOSPHERE_SCRIP_FILE, land_sea_mask_file=str(mask_file)
        )


@pytest.mark.slow
@pytest.mark.parametrize("build", [_double_drake_model, _earth_model])
def test_the_coupled_setups_leave_veros_nan_scanner_off(build):
    """Veros' own NaN scanner prints to stdout and instruments every step; the run's health gate is the check."""
    assert not build().state.settings.enable_nan_checks


@pytest.mark.slow
@pytest.mark.parametrize("build", [_double_drake_model, _earth_model])
def test_the_initial_profile_is_uniform_and_crosses_zero_above_the_deepest_layer(build):
    """The documented initial temperature: uniform in the horizontal, zero at |zw[0]|.

    ``T0 * (1 - z / zw[0])`` reaches zero at the top face of the deepest
    layer, not at the sea floor, so the deepest layer starts below zero. The
    setups' docstrings say so; this pins the numbers they quote.
    """
    vs = build().state.variables
    wet = np.asarray(vs.maskT)[2:-2, 2:-2, -1] > 0
    profile = np.asarray(vs.temp)[2:-2, 2:-2, :, 0][wet]
    np.testing.assert_allclose(profile, np.broadcast_to(profile[0], profile.shape), atol=1e-12)
    assert abs(float(vs.zw[0])) == pytest.approx(4510.0)
    assert profile[0, 0] == pytest.approx(-1.147, abs=1e-3)
    assert profile[0, -1] == pytest.approx(14.884, abs=1e-3)


@pytest.fixture(scope="module")
def _bathymetry_file(tmp_path_factory):
    """Write a depth file on the rotated grid from a synthetic source spanning every layer count.

    The source is a 1-degree field laid out like Veros' ``global_1deg`` asset,
    with a depth that varies smoothly from about 60 m to over 5200 m, so the
    setup is exercised across the minimum, the full depth and everything
    between without needing that asset (or its licence) in the test.
    """
    from jem.tools.ocean_bathymetry import ocean_bathymetry

    directory = tmp_path_factory.mktemp("bathymetry")
    lon = np.arange(90.5, 450.0, 1.0, dtype=np.float32)
    lat = np.arange(-79.5, 80.0, 1.0, dtype=np.float32)
    lat2, lon2 = np.meshgrid(lat, lon % 360, indexing="ij")
    depth = 60.0 + 5400.0 * np.abs(np.sin(np.deg2rad(2.0 * lat2)) * np.cos(np.deg2rad(lon2)))
    source = directory / "source.nc"
    xr.Dataset(
        {"bathymetry": (("yt", "xt"), (-depth).astype(np.float32))}, coords={"xt": lon, "yt": lat}
    ).to_netcdf(source)
    return str(
        ocean_bathymetry(source, ROTATED_SCRIP_FILE, ROTATED_LANDSEA_MASK_FILE, directory / "depth.nc")
    )


def _earth_model_with_bathymetry(bathymetry_file):
    from jem.components.veros.setups.earth import earth_setup

    model = earth_setup(
        scrip_grid_file=ROTATED_SCRIP_FILE,
        landsea_mask_file=ROTATED_LANDSEA_MASK_FILE,
        bathymetry_file=bathymetry_file,
    )()
    model.setup()
    return model


@pytest.mark.slow
def test_earth_flat_bottom_is_the_land_sea_mask_itself():
    """With no bathymetry file every wet column holds every layer: ``kbot`` is the mask, exactly."""
    vs = _earth_model().state.variables
    lsm = xr.open_dataset(ROTATED_LANDSEA_MASK_FILE)["lsm"].to_numpy().reshape(48, 96)
    expected = (~(lsm >= 0.5)).astype(int).T
    kbot = np.asarray(vs.kbot)[2:-2, 2:-2]
    np.testing.assert_array_equal(kbot, expected)
    mask_t = np.asarray(vs.maskT)[2:-2, 2:-2]
    np.testing.assert_array_equal(mask_t, np.broadcast_to(expected[..., None] > 0, mask_t.shape))


@pytest.mark.slow
def test_earth_bathymetry_leaves_the_wet_columns_exactly_the_land_sea_masks(_bathymetry_file):
    """Bathymetry sets how deep a wet column is, never whether it is wet."""
    flat = np.asarray(_earth_model().state.variables.kbot)[2:-2, 2:-2]
    deep = _earth_model_with_bathymetry(_bathymetry_file).state.variables
    kbot = np.asarray(deep.kbot)[2:-2, 2:-2]
    np.testing.assert_array_equal(kbot > 0, flat > 0)
    np.testing.assert_array_equal(np.asarray(deep.maskT)[2:-2, 2:-2, -1] > 0, flat > 0)
    # And the bottom really is variable: the test would otherwise pass on a flat one.
    assert len(np.unique(kbot[kbot > 0])) > 5


@pytest.mark.slow
def test_earth_kbot_matches_the_depths_and_the_wet_layers_the_depth_implies(_bathymetry_file):
    """``kbot`` counts the layers whose centre is above the floor (at least two), as global_latlon does."""
    from jem.components.veros.setups._layers import LAYER_THICKNESSES
    from jem.tools.ocean_bathymetry import read_depth

    vs = _earth_model_with_bathymetry(_bathymetry_file).state.variables
    kbot = np.asarray(vs.kbot)[2:-2, 2:-2]
    depth = read_depth(_bathymetry_file, 48, 96)
    wet = kbot > 0
    thickness = np.asarray(LAYER_THICKNESSES)
    bottom_face = np.cumsum(thickness)
    centre = bottom_face - thickness / 2

    n_wet = len(thickness) - kbot + 1
    for i, j in zip(*np.nonzero(wet)):
        expected = max(int((centre < depth[i, j]).sum()), 2)
        assert n_wet[i, j] == expected, (i, j, depth[i, j])
    # Within one layer of what the depth implies: no more than one layer
    # whose bottom face lies below the floor, none missing above it (the
    # two-layer minimum aside).
    fully_above = (bottom_face[None, :] <= depth[wet][:, None]).sum(1)
    assert np.all(n_wet[wet] >= fully_above)
    assert np.all(n_wet[wet] <= np.maximum(fully_above + 1, 2))
    # The model's own masks agree with kbot: layers from the surface down.
    mask_t = np.asarray(vs.maskT)[2:-2, 2:-2]
    np.testing.assert_array_equal(mask_t.sum(-1), n_wet * wet)
    assert mask_t[..., -1][wet].all()


def _composed_veros_earth(overrides=()):
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="jem.config", version_base="1.3"):
        return compose(config_name="config", overrides=["+configuration=veros-earth", *overrides])


def test_veros_earth_configuration_selects_the_packaged_bathymetry_and_the_flat_bottom_is_one_override():
    """The headline configuration reads the packaged ETOPO depths; ``ocean.bathymetry_file=null`` is the flat bottom."""
    packaged = str(DATA / "bathymetry_RotatedGaussianLatLon.nc")
    assert _composed_veros_earth().ocean.bathymetry_file == packaged
    assert _composed_veros_earth(["ocean.bathymetry_file=null"]).ocean.bathymetry_file is None


@pytest.mark.slow
def test_earth_with_the_packaged_bathymetry_gives_real_depths_on_exactly_the_mask_wet_columns():
    """The packaged ETOPO file drives ``kbot``: every wet column holds the layers its depth implies."""
    from jem.components.veros.setups._layers import LAYER_THICKNESSES
    from jem.tools.ocean_bathymetry import read_depth

    packaged = str(DATA / "bathymetry_RotatedGaussianLatLon.nc")
    vs = _earth_model_with_bathymetry(packaged).state.variables
    flat_kbot = np.asarray(_earth_model().state.variables.kbot)[2:-2, 2:-2]
    kbot = np.asarray(vs.kbot)[2:-2, 2:-2]
    np.testing.assert_array_equal(kbot > 0, flat_kbot > 0)
    n_wet = len(LAYER_THICKNESSES) - kbot + 1
    thickness = np.asarray(LAYER_THICKNESSES)
    centre = np.cumsum(thickness) - thickness / 2
    depth = read_depth(packaged, 48, 96)
    wet = kbot > 0
    expected = np.maximum((centre[None, None, :] < depth[..., None]).sum(-1), 2)
    np.testing.assert_array_equal(n_wet[wet], expected[wet])
    # Real relief: columns of every layer count from the two-layer minimum to the full 15.
    assert n_wet[wet].min() == 2 and n_wet[wet].max() == len(LAYER_THICKNESSES)


@pytest.mark.slow
def test_earth_initial_state_is_the_flat_profile_on_every_wet_layer_and_zero_below(_bathymetry_file):
    """A variable bottom does not change the profile; it only removes the layers under the floor."""
    flat = _earth_model().state.variables
    deep = _earth_model_with_bathymetry(_bathymetry_file).state.variables
    mask_t = np.asarray(deep.maskT)[2:-2, 2:-2] > 0
    for name in ("temp", "salt"):
        flat_field = np.asarray(getattr(flat, name))[2:-2, 2:-2, :, 0]
        field = np.asarray(getattr(deep, name))[2:-2, 2:-2, :, 0]
        np.testing.assert_array_equal(field[mask_t], flat_field[mask_t])
        assert np.all(field[~mask_t] == 0.0)


def _atmosphere_dataset(n_records=2):
    """Return a dataset with the T31 atmosphere's own coordinates, from jcm's writer."""
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.utils import data_to_xarray

    coords = get_speedy_coords(layers=8, spectral_truncation=31)
    return data_to_xarray(
        {"surface_pressure": np.zeros(
            (n_records,) + coords.horizontal.nodal_shape, dtype=np.float32)},
        coords=coords,
        times=1.0 * np.arange(n_records),
        serialize_coords_to_attrs=False,
    )


def test_double_drake_declares_the_atmospheres_axes_bit_for_bit():
    """The declared output axes are what the atmosphere's own dataset holds.

    Not close: equal. ``xr.merge(join="exact")`` compares bits, and the SCRIP
    file the setup is built from is ~1e-13 degrees away from them.
    """
    from jem.components.veros.setups._grid import read_scrip_axes
    from jem.components.veros.setups.double_drake import _atmosphere_output_axes

    scrip_longitude, scrip_latitude = read_scrip_axes(ATMOSPHERE_SCRIP_FILE)
    longitude, latitude = _atmosphere_output_axes(
        scrip_longitude, scrip_latitude, ATMOSPHERE_SCRIP_FILE)

    atmosphere = _atmosphere_dataset()
    np.testing.assert_array_equal(longitude, atmosphere["lon"].values)
    np.testing.assert_array_equal(latitude, atmosphere["lat"].values)
    # The reason the declaration exists: the file's own centres are not those.
    assert not np.array_equal(scrip_longitude, atmosphere["lon"].values)


def test_double_drake_refuses_a_grid_that_is_not_the_atmospheres():
    from jem.components.veros.setups._grid import read_scrip_axes
    from jem.components.veros.setups.double_drake import _atmosphere_output_axes

    longitude, latitude = read_scrip_axes(ATMOSPHERE_SCRIP_FILE)
    with pytest.raises(ValueError, match="not a Gaussian grid"):
        _atmosphere_output_axes(longitude[:-1], latitude, "shape.nc")
    with pytest.raises(ValueError, match="not the grid the atmosphere runs on"):
        _atmosphere_output_axes(longitude + 0.5, latitude, "shifted.nc")


@pytest.mark.slow
def test_double_drake_output_merges_exactly_with_the_atmospheres():
    """A double-drake ocean's dataset shares ``lon``/``lat`` with the atmosphere's exactly.

    Without the setup's declared axes the ocean is labelled with Veros'
    reconstructed ``xt``/``yt``, ~1e-13 degrees from the atmosphere's, and
    ``xr.merge([atm, ocn], join="exact")`` raises.
    """
    import jax_datetime as jdt

    from jem.base.component import TimeAxis

    model = _double_drake_model()
    component = veros_component.VerosComponent(model)
    nx, ny = component.horizontal_shape
    nz = int(component.dzt.shape[0])
    n_records = 2
    surface = jnp.zeros((n_records, nx, ny))
    diagnostics = {name: jnp.zeros((n_records, nx, ny, nz))
                   for name in ("temp", "salt", "u", "v")}
    diagnostics.update({
        name: surface for name in (
            "psi", "sea_surface_temperature", "sea_surface_salinity",
            "sea_surface_u", "sea_surface_v", "surface_air_temperature",
            "surface_taux", "surface_tauy", "heat_flux", "freshwater_flux",
            "ice_frazil_melt_energy")
    })
    diagnostics["runoff_mass_rate"] = jnp.zeros((n_records,))
    diagnostics["ssh"] = surface
    diagnostics["B1_gm"] = jnp.zeros((n_records, nx, ny, nz))
    time = TimeAxis(
        jdt.to_datetime("2001-01-01"), np.arange(n_records), jdt.to_timedelta(1, "day"))
    dataset = component.to_xarray(diagnostics, time)

    atmosphere = _atmosphere_dataset(n_records)
    atmosphere["time"] = dataset["time"]
    merged = xr.merge([atmosphere, dataset], join="exact", compat="no_conflicts")

    np.testing.assert_array_equal(merged["lon"].values, atmosphere["lon"].values)
    np.testing.assert_array_equal(merged["lat"].values, atmosphere["lat"].values)
    assert merged["sea_surface_temperature"].dims == ("time", "lon", "lat")
    assert merged["surface_pressure"].dims == ("time", "lon", "lat")
