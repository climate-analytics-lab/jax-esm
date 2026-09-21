"""Tests for `jem.components.slab.slab_seaice_model`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import jcm.constants as jcm_constants
from jem import constants
from jem.components.slab.base import MASKED_SURFACE_TEMPERATURE
from jem.components.slab.slab_seaice_model import SlabSeaiceModel, SlabSeaiceParameters
from tests.unit.slab_test_utils import (
    LATITUDE_DEGREES,
    LONGITUDE_DEGREES,
    coupling_time,
    make_grid,
    tree_signature,
)

#: Energy (J/m2) that freezes one metre of ice, for the constants JCM owns.
ENERGY_PER_METRE = jcm_constants.rhoi * jcm_constants.alhf


@pytest.fixture
def uniform_grid():
    """Return a tiny all-ocean 4x3 lon-lat grid the tests own."""
    return make_grid()


@pytest.fixture
def half_land_grid():
    """Return a 4x3 grid whose eastern half is land."""
    shape = (len(LONGITUDE_DEGREES), len(LATITUDE_DEGREES))
    fractional_mask = jnp.where(
        jnp.arange(shape[0])[:, None] >= shape[0] // 2,
        jnp.ones(shape),
        jnp.zeros(shape),
    )
    return make_grid(fractional_mask=fractional_mask)


def test_positive_frazil_energy_grows_ice(uniform_grid):
    """The freeze/melt potential is an energy per coupling step, applied as-is."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(uniform_grid.shape, 0.5 * ENERGY_PER_METRE)
    )

    stepped, _ = model.step(carry, coupling_time(0))

    np.testing.assert_allclose(
        np.asarray(stepped["state"].ice_thickness), 0.5, rtol=1e-5
    )
    # Half a metre against the 0.5 m fill-in scale: 1 - exp(-1).
    np.testing.assert_allclose(
        np.asarray(stepped["derived"].ice_fraction), 1.0 - np.exp(-1.0), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(stepped["state"].ice_surface_temperature), jcm_constants.tmelt
    )


def test_melt_cannot_drive_thickness_negative(uniform_grid):
    """Surplus ocean heat melts ice, and stops at open water."""
    model = SlabSeaiceModel(
        uniform_grid, SlabSeaiceParameters(initial_ice_thickness=0.1)
    )
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(uniform_grid.shape, -1.0 * ENERGY_PER_METRE)
    )

    stepped, _ = model.step(carry, coupling_time(0))

    np.testing.assert_allclose(np.asarray(stepped["state"].ice_thickness), 0.0)
    np.testing.assert_allclose(
        np.asarray(stepped["state"].ice_surface_temperature),
        constants.seawater_freezing_point_K,
    )


def test_land_cells_carry_no_ice(half_land_grid):
    """Only ocean cells are integrated; land reports the masked temperature."""
    model = SlabSeaiceModel(
        half_land_grid, SlabSeaiceParameters(initial_ice_thickness=1.0)
    )
    carry = model.initialize()
    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(half_land_grid.shape, ENERGY_PER_METRE)
    )
    stepped, _ = model.step(carry, coupling_time(0))

    land = np.asarray(half_land_grid.binary_mask) == 1.0
    thickness = np.asarray(stepped["state"].ice_thickness)
    temperature = np.asarray(stepped["state"].ice_surface_temperature)
    assert np.all(thickness[land] == 0.0)
    assert np.all(thickness[~land] > 0.0)
    np.testing.assert_allclose(temperature[land], MASKED_SURFACE_TEMPERATURE)


def test_invalid_parameters_are_rejected(uniform_grid):
    """A thickness scale of zero would make the closures undefined."""
    with pytest.raises(ValueError, match="min_ice_thickness"):
        SlabSeaiceModel(uniform_grid, SlabSeaiceParameters(min_ice_thickness=0.0))
    with pytest.raises(ValueError, match="ice_fraction_thickness_scale"):
        SlabSeaiceModel(
            uniform_grid, SlabSeaiceParameters(ice_fraction_thickness_scale=0.0)
        )
    with pytest.raises(ValueError, match="initial_ice_thickness"):
        SlabSeaiceModel(uniform_grid, SlabSeaiceParameters(initial_ice_thickness=-1.0))


@pytest.mark.parametrize(
    "field, value",
    [
        ("min_ice_thickness", 0.0),
        ("min_ice_thickness", -1.0),
        ("min_ice_thickness", np.nan),
        ("min_ice_thickness", np.inf),
        ("ice_fraction_thickness_scale", 0.0),
        ("ice_fraction_thickness_scale", -0.5),
        ("ice_fraction_thickness_scale", np.nan),
        ("ice_fraction_thickness_scale", np.inf),
        ("initial_ice_thickness", -1.0),
        ("initial_ice_thickness", np.nan),
        ("initial_ice_thickness", np.inf),
    ],
)
def test_non_finite_thicknesses_are_rejected(uniform_grid, field, value):
    """Every thickness must be finite as well as correctly signed.

    A NaN compares False against every threshold and an infinite fraction scale
    makes ``1 - exp(-h / scale)`` zero everywhere, so an unvalidated one gives a
    run with no ice rather than an error.
    """
    with pytest.raises(ValueError, match=field):
        SlabSeaiceModel(uniform_grid, SlabSeaiceParameters(**{field: value}))


def test_params_default_equivalence(uniform_grid):
    """Constructing with no params is constructing with the defaults."""
    implicit = SlabSeaiceModel(uniform_grid).initialize()
    explicit = SlabSeaiceModel(
        uniform_grid, SlabSeaiceParameters.default()
    ).initialize()

    assert jax.tree_util.tree_structure(implicit) == jax.tree_util.tree_structure(
        explicit
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(implicit), jax.tree_util.tree_leaves(explicit)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_state_has_no_clock(uniform_grid):
    """State carries no time, and the step does not depend on the date.

    The sea-ice model integrates an energy that already has the coupling step
    folded into it (the ocean's ``frzmlt``), so unlike the ocean and land
    models it reads nothing from the clock at all -- which is only checkable
    now that the clock arrives as an argument rather than living in the state.
    """
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    assert "sim_time" not in carry["state"].asdict()

    carry["forcing"] = carry["forcing"].replace(
        ice_frazil_melt_energy=jnp.full(uniform_grid.shape, 0.25 * ENERGY_PER_METRE)
    )
    first, _ = model.step(carry, coupling_time(0))
    later, _ = model.step(carry, coupling_time(182))

    np.testing.assert_array_equal(
        np.asarray(first["state"].ice_thickness),
        np.asarray(later["state"].ice_thickness),
    )


def test_step_shapes_and_dtypes_stable(uniform_grid):
    """A step returns exactly the carry structure it received (lax.scan's rule)."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()
    new_carry, _ = model.step(carry, coupling_time(0))

    assert tree_signature(new_carry) == tree_signature(carry)


def test_step_is_differentiable(uniform_grid):
    """Reverse-mode gradients of one step agree with finite differences."""
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()

    def mean_ice_fraction(energy_scale):
        # One unit is the energy that freezes a metre of ice, so the ice
        # fraction responds at O(1) to a unit change.
        forced = carry["forcing"].replace(
            ice_frazil_melt_energy=jnp.full(
                uniform_grid.shape, ENERGY_PER_METRE * energy_scale
            )
        )
        stepped, _ = model.step({**carry, "forcing": forced}, coupling_time(0))
        return jnp.mean(stepped["derived"].ice_fraction)

    check_grads(
        mean_ice_fraction, (0.5,), order=1, modes=["rev"], eps=1e-3, atol=1e-3, rtol=1e-3
    )


def test_thickness_scale_is_differentiable(uniform_grid):
    """The ice-fraction closure's scale is a pytree leaf of the carry."""
    model = SlabSeaiceModel(uniform_grid)

    def mean_ice_fraction(scale):
        carry = model.initialize()
        carry["params"] = carry["params"].replace(ice_fraction_thickness_scale=scale)
        carry["forcing"] = carry["forcing"].replace(
            ice_frazil_melt_energy=jnp.full(uniform_grid.shape, ENERGY_PER_METRE)
        )
        carry, _ = model.step(carry, coupling_time(0))
        return jnp.mean(carry["derived"].ice_fraction)

    gradient = jax.grad(mean_ice_fraction)(jnp.float32(0.5))
    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 0.0


def test_initialize_takes_parameters_and_defaults_to_the_models_own(uniform_grid):
    """``initialize(params)`` starts from them; no argument starts as before."""
    model = SlabSeaiceModel(uniform_grid)

    default = model.initialize()
    explicit = model.initialize(model.params)
    thicker = model.initialize(
        SlabSeaiceParameters(initial_ice_thickness=2.0)
    )

    # No argument is the construction-time initial state, unchanged.
    assert jax.tree_util.tree_structure(default) == jax.tree_util.tree_structure(
        explicit
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(default), jax.tree_util.tree_leaves(explicit)
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_allclose(np.asarray(default["state"].ice_thickness), 0.0)

    # Parameters given here build the state *and* travel in the carry, so the
    # two cannot come from different objects.
    np.testing.assert_allclose(np.asarray(thicker["state"].ice_thickness), 2.0)
    assert float(thicker["params"].initial_ice_thickness) == 2.0
    np.testing.assert_allclose(
        np.asarray(thicker["derived"].ice_fraction),
        1.0 - np.exp(-2.0 / 0.5),
        rtol=1e-5,
    )


def test_replacing_the_carried_leaf_cannot_move_the_initial_state(uniform_grid):
    """Why ``initialize`` takes parameters: the initial condition is spent.

    ``initial_ice_thickness`` has already been copied into the state by the
    time a carry exists, and ``step`` never reads it, so replacing that leaf
    in the carry does nothing -- which is exactly why an initial condition is
    varied through ``initialize`` instead.
    """
    model = SlabSeaiceModel(uniform_grid)
    carry = model.initialize()

    carry["params"] = carry["params"].replace(initial_ice_thickness=3.0)
    stepped, _ = model.step(carry, coupling_time(0))

    np.testing.assert_allclose(np.asarray(carry["state"].ice_thickness), 0.0)
    np.testing.assert_allclose(np.asarray(stepped["state"].ice_thickness), 0.0)


def test_grad_wrt_initial_thickness_reaches_the_trajectory(uniform_grid):
    """The initial thickness is differentiable through ``initialize``."""
    model = SlabSeaiceModel(uniform_grid)

    def mean_thickness(initial_ice_thickness):
        params = model.params.replace(initial_ice_thickness=initial_ice_thickness)
        carry = model.initialize(params)
        carry["forcing"] = carry["forcing"].replace(
            # Growth well clear of the clip at zero, so the derivative is not
            # measuring the clip.
            ice_frazil_melt_energy=jnp.full(
                uniform_grid.shape, 0.25 * ENERGY_PER_METRE
            )
        )
        for step in range(3):
            carry, _ = model.step(carry, coupling_time(step))
        return jnp.mean(carry["state"].ice_thickness)

    gradient = jax.grad(mean_thickness)(jnp.float32(1.0))

    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 0.0
    # Basal growth is additive, so a metre of initial ice is a metre at the end.
    np.testing.assert_allclose(float(gradient), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Starting from an observed ice concentration
# ---------------------------------------------------------------------------
#
# Under the standard workflow the exchange runs before the components, and a
# `derived` field is only rewritten at the end of a step, so
# `initialize()`'s `ice_fraction` is what an atmosphere is handed for its
# first two coupling steps. Starting from zero hands an Earth-like run
# ice-free poles, and the heat loss that causes comes back as a freeze/melt
# potential that grows implausibly thick ice in one step. These pin the
# climatological start that avoids it.


#: A January-peaking concentration: fully covered in January, ice-free in July.
MONTHLY_ICE_FRACTION = 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(12) / 12.0)


@pytest.fixture
def ice_clim_file(tmp_path, uniform_grid):
    """Write a 12-month `icec` concentration climatology the tests own."""
    from tests.unit.slab_test_utils import write_climatology

    shape = uniform_grid.shape[::-1]  # (lat, lon), the file's order
    seasonal = [np.full(shape, value, dtype=np.float32)
                for value in MONTHLY_ICE_FRACTION]
    return write_climatology(tmp_path / "icec.nc", "icec", np.array(seasonal))


def _bind(model, start="2000-04-01"):
    """Bind a model to a one-day coupler starting on ``start``."""
    import jax_datetime as jdt

    model.bind(
        coupling_timestep=jdt.to_timedelta(1, "day"),
        start_date=jdt.to_datetime(start),
        calendar="365_day",
    )
    return model


def test_initial_fraction_reproduces_the_climatology_at_the_start_date(
    uniform_grid, ice_clim_file
):
    """The fraction `initialize` publishes is the file's, at the run's month.

    Round-tripped through the closure and its inverse, so this checks the
    inversion as well as the sampling: away from saturation the two are exact
    to float32. The climatology is read here the way the slab ocean reads its
    SST -- `load_monthly_climatology` then `evaluate_cyclic_linear` -- so the
    file's own month-to-month interpolation is what is compared against.
    """
    from jem.components.slab.base import load_monthly_climatology
    from jem.utils.cycles import evaluate_cyclic_linear

    model = _bind(SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file))

    fraction = np.asarray(model.initialize()["derived"].ice_fraction)
    expected = np.asarray(evaluate_cyclic_linear(
        model.start_year_fraction,
        load_monthly_climatology(ice_clim_file, "icec", uniform_grid),
    ))
    assert 0.0 < expected.max() < 0.95, expected.max()
    np.testing.assert_allclose(fraction, expected, atol=1e-6)

    # And it is the START date that selected it, not simply the first month:
    # a January start of the same file gives a different (saturated) cover.
    january = _bind(
        SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",
    ).initialize()["derived"].ice_fraction
    assert float(np.asarray(january).min()) > fraction.max()


def test_a_fully_covered_cell_is_capped_rather_than_infinite(
    uniform_grid, ice_clim_file
):
    """Concentration 1 inverts to infinity, so it is given a real depth instead.

    The closure saturates, so a fully covered cell's thickness is not
    recoverable from its fraction; `max_initial_ice_thickness` is what such a
    cell gets. It must still *read back* as effectively fully covered.
    """
    model = _bind(
        SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",  # the month the climatology is 1.0
    )
    carry = model.initialize()

    thickness = np.asarray(carry["state"].ice_thickness)
    np.testing.assert_allclose(
        thickness, float(model.params.max_initial_ice_thickness), rtol=1e-6
    )
    assert np.isfinite(thickness).all()
    assert np.asarray(carry["derived"].ice_fraction).min() > 0.99


def test_without_a_climatology_the_ice_starts_from_the_parameter(uniform_grid):
    """No file is the unchanged behaviour: a uniform thickness, zero by default."""
    bare = _bind(SlabSeaiceModel(uniform_grid)).initialize()
    assert float(np.asarray(bare["state"].ice_thickness).max()) == 0.0
    assert float(np.asarray(bare["derived"].ice_fraction).max()) == 0.0

    uniform = _bind(
        SlabSeaiceModel(uniform_grid, SlabSeaiceParameters(initial_ice_thickness=0.5))
    ).initialize()
    np.testing.assert_allclose(
        np.asarray(uniform["state"].ice_thickness), 0.5, rtol=1e-6
    )


def test_land_cells_carry_no_climatological_ice(half_land_grid, ice_clim_file):
    """The ocean mask still wins: a land cell has no ice whatever the file says."""
    model = _bind(
        SlabSeaiceModel(half_land_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",
    )
    carry = model.initialize()

    land = np.asarray(half_land_grid.binary_mask == 1.0)
    assert np.asarray(carry["state"].ice_thickness)[land].max() == 0.0
    assert np.asarray(carry["derived"].ice_fraction)[land].max() == 0.0


def test_a_missing_climatology_file_is_reported_at_construction(uniform_grid):
    """A bad path fails where the traceback still points at the caller."""
    with pytest.raises(FileNotFoundError, match="no-such-file.nc"):
        SlabSeaiceModel(uniform_grid, ice_clim_file="no-such-file.nc")


def test_a_climatology_with_nans_over_ocean_is_refused(tmp_path, uniform_grid):
    """NaNs over ocean mean the file's land mask and the grid's disagree."""
    from tests.unit.slab_test_utils import write_climatology

    shape = (12,) + uniform_grid.shape[::-1]
    values = np.zeros(shape, dtype=np.float32)
    values[0, 0, 0] = np.nan
    path = write_climatology(tmp_path / "nan.nc", "icec", values)

    with pytest.raises(ValueError, match="NaNs over ocean"):
        SlabSeaiceModel(uniform_grid, ice_clim_file=path)


def test_max_initial_ice_thickness_is_validated_and_differentiable(
    uniform_grid, ice_clim_file
):
    """It is a depth like the others, and a gradient reaches it through the state."""
    with pytest.raises(ValueError, match="max_initial_ice_thickness"):
        SlabSeaiceModel(
            uniform_grid, SlabSeaiceParameters(max_initial_ice_thickness=-1.0)
        )

    model = _bind(
        SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",  # saturated, so the cap is what sets the thickness
    )

    def total_thickness(cap):
        params = SlabSeaiceParameters(max_initial_ice_thickness=cap)
        return jnp.sum(model.initialize(params)["state"].ice_thickness)

    gradient = float(jax.grad(total_thickness)(3.0))
    assert gradient == pytest.approx(float(np.prod(uniform_grid.shape)))


def test_saturated_cell_gradient_is_finite_not_nan(uniform_grid, ice_clim_file):
    """Regression: a fully-covered cell must not poison the closure's gradient.

    ``_thickness_from_fraction`` inverts ``f = 1 - exp(-h / scale)`` with
    ``log1p(-f)``. At ``f == 1`` that is ``log1p(-1) = -inf``, and although
    the outer ``jnp.where`` replaces the *primal* with the cap for such a
    cell, reverse-mode AD still evaluates the VJP of the unselected branch
    before zeroing its cotangent -- ``0 * inf = nan``. Every ocean cell of
    ``ice_clim_file`` is exactly 1.0 in January, so this exercises the bug
    with real data rather than a hand-built edge case.
    """
    model = _bind(
        SlabSeaiceModel(uniform_grid, ice_clim_file=ice_clim_file),
        start="2000-01-01",  # the month the climatology is 1.0 everywhere
    )

    def total_thickness(scale):
        params = SlabSeaiceParameters(ice_fraction_thickness_scale=scale)
        return jnp.sum(model.initialize(params)["state"].ice_thickness)

    thickness = np.asarray(model.initialize()["state"].ice_thickness)
    gradient = jax.grad(total_thickness)(jnp.float32(0.5))

    # The primal is unchanged by the fix: every cell is still capped.
    np.testing.assert_allclose(
        thickness, float(model.params.max_initial_ice_thickness), rtol=1e-6
    )
    assert bool(jnp.isfinite(gradient))
    # Every cell is saturated, so the cap -- not the scale -- sets the
    # thickness everywhere: the gradient is exactly zero, not merely finite.
    assert float(gradient) == 0.0
