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
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("veros")

from jem.components import veros_component  # noqa: E402

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
def test_double_drake_veros_component_declares_its_internal_stepping_rate():
    """``SupportsInternalStepping`` extended to Veros.

    Veros' own ``itt`` iteration counter (``state.variables.itt``) is
    declared ``dtype="int32"`` in ``veros.variables`` and is a genuine
    ``jax.lax.fori_loop``-carried pytree leaf (``veros_variables_pytree
    _flatten`` flattens every one of ``VerosVariables``' fields, ``itt``
    included -- confirmed directly: ``model.state.variables.itt.dtype ==
    jnp.int32`` below), incremented once per internal Veros step
    (``veros.veros.py``'s own ``vs.itt = vs.itt + 1``) --
    ``self._steps_per_coupling_step`` times per coupled step. That is
    exactly the same shape of raw-counter risk as JCM's ``RunState.step``,
    on a real double-drake configuration (the shipped coupled ocean setup,
    not the smaller ``acc_basic`` test fixture): with the default
    ``dt_tracer=3600`` s and a 1 day coupling timestep,
    ``self._steps_per_coupling_step == 24``, so ``itt`` wraps at a coupled
    step count of about ``2**31 / 24``, roughly 245,000 simulated years for
    daily coupling.

    ``VerosComponent.internal_steps_per_call`` reports that rate, so
    ``jem.driver._max_element_rate``/``_check_step_counters_fit_int32``
    cover it the same way they cover JCM's own, and this checks the
    boundary directly (not the whole of ``run_chunked``, which would then
    have to build and run a many-million-step trajectory).
    """
    import jax.numpy as jnp
    import jax_datetime as jdt

    from jem.base.coupler import Coupler
    from jem.components.veros.setups.double_drake import double_drake_setup
    from jem.components.veros_component import VerosComponent
    from jem.driver import (
        _check_step_counters_fit_int32,
        _max_element_rate,
        _max_safe_coupled_steps,
    )

    setup_cls = double_drake_setup(land_sea_mask_file=DOUBLE_DRAKE_MASK_FILE)
    model = setup_cls()
    model.setup()
    # The premise this test exists to cover: a real, traced int32 counter.
    assert model.state.variables.itt.dtype == jnp.int32
    assert float(model.state.settings.dt_tracer) == 3600.0

    component = VerosComponent(model)
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler = Coupler(
        {"ocn": component}, {}, coupling_timestep=coupling_timestep,
        start_date=jdt.to_datetime("2000-01-01"), calendar="365_day",
    )
    assert component.internal_steps_per_call() == 24  # 86400 s / 3600 s
    assert _max_element_rate(coupler) == 24

    # A freshly built model: `itt` starts at 0, so this exercises the same
    # bound the rate alone would predict -- the pre-stepped case (a model
    # integrated before it was ever bound) is
    # `test_double_drake_veros_component_refuses_a_run_that_would_wrap_a_pre_stepped_itt`'s
    # job.
    carries = {"ocn": component.initialize()}
    limit = _max_safe_coupled_steps(coupler)
    assert limit < 2**31 - 2  # sanity: strictly tighter than the rate-1 floor
    _check_step_counters_fit_int32(coupler, 0, limit + 1, carries)  # last step == limit: fine
    with pytest.raises(ValueError, match="largest this coupler's own clock can hold"):
        _check_step_counters_fit_int32(coupler, 0, limit + 2, carries)  # last step == limit + 1


@pytest.mark.slow
def test_double_drake_veros_component_refuses_a_run_that_would_wrap_a_pre_stepped_itt():
    """A model integrated before it was bound is refused where ITS OWN itt would wrap.

    ``VerosComponent.bind``'s own docstring explicitly allows wrapping a
    model that was already integrated before it was ever registered with a
    coupler: "a setup that was already integrated before it was wrapped
    therefore starts the coupled run at its own current time rather than
    being declared wrong". Such a model's ``itt`` does not start at 0, so a
    coupled-step-count-only rate (``_max_element_rate``'s own) cannot bound
    it -- only reading ``itt`` off the concrete starting carry can. Built on
    a real double-drake ``VerosComponent`` (the shipped coupled ocean setup),
    with its own ``itt`` set to a large value before ``bind`` -- the
    reproduction the review's own probe used -- and checked at the exact
    boundary where advancing it further would silently wrap.
    """
    import jax.numpy as jnp
    import jax_datetime as jdt

    from jem.base.coupler import Coupler
    from jem.components.veros.setups.double_drake import double_drake_setup
    from jem.components.veros_component import VerosComponent
    from jem.driver import _check_step_counters_fit_int32

    setup_cls = double_drake_setup(land_sea_mask_file=DOUBLE_DRAKE_MASK_FILE)
    model = setup_cls()
    model.setup()

    rate = 24  # 86400 s coupling / 3600 s dt_tracer
    steps_to_boundary = 100
    # Chosen so the boundary is exact: `starting_itt + 100 * rate == 2**31 -
    # 1` precisely -- no remainder to obscure the "one step past is refused"
    # edge. Simulates a model that was run standalone (`model.step(state)`,
    # or an earlier coupled run) before this wrapper or coupler ever existed.
    starting_itt = 2**31 - 1 - steps_to_boundary * rate
    with model.state.variables.unlock():
        model.state.variables.itt = jnp.int32(starting_itt)

    component = VerosComponent(model)
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler = Coupler(
        {"ocn": component}, {}, coupling_timestep=coupling_timestep,
        start_date=jdt.to_datetime("2000-01-01"), calendar="365_day",
    )
    carry = coupler.initialize()
    assert int(carry.components["ocn"]["state"].variables.itt) == starting_itt
    carries = carry.components

    _check_step_counters_fit_int32(
        coupler, 0, steps_to_boundary, carries
    )  # itt reaches 2**31 - 1 exactly: fine
    with pytest.raises(ValueError, match="own internal counter"):
        _check_step_counters_fit_int32(
            coupler, 0, steps_to_boundary + 1, carries
        )  # one step past


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
