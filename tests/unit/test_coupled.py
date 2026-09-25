"""Coupled runs -- the seams a single-component test cannot see.

Every test here runs a real :class:`~jem.base.coupler.Coupler` over more than
one step, because that is the only thing that exercises what this project is:
that two independently written components agree on the carry structure a
``lax.scan`` demands, on the clock, on the sign and units of the fields they
exchange, and on how they label their output so the two datasets can be read
as one.

The atmosphere here is a real JCM model at the smallest configuration SPEEDY
physics supports (T21 on its (64, 32) nodal grid, 5 levels -- the fewest its
convective cloud-top search accepts), on an aquaplanet. Building it and
compiling one coupled step is the bulk of the runtime, so the model, the
coupler and the two-step trajectory are module-scoped and shared. A second,
smaller section further down couples an ECHAM atmosphere the same way (the
regression test for jax-esm#129: ECHAM publishes no near-surface wind vector,
which must not stop a coupled step that never needs one).
"""

import dataclasses

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
import xarray as xr
from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

from jem.base.coupler import Coupler
from jem.components.jcm import JCMComponent
from jem.components.slab import (
    SlabAtmosphereModel,
    SlabGrid,
    SlabLandModel,
    SlabOceanModel,
    SlabOceanParameters,
    SlabSeaiceModel,
)
from jem.components.slab.grid import to_degrees

START_DATE = jdt.to_datetime("2000-01-01")
CALENDAR = "365_day"
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")

LAYERS = 5
TRUNCATION = 21
GRID_SHAPE = (64, 32)

#: Relaxation timescale (s) of the slab ocean in the all-slab test. Short
#: enough that three coupling steps leave a gradient with respect to it well
#: clear of float32 noise; still a timescale a real run might use.
RELAXATION_TIME = 5.0 * 86400.0


# ---------------------------------------------------------------------------
# Exchangers
# ---------------------------------------------------------------------------


def atmosphere_ocean_exchange(components, time):
    """Move the surface heat flux down and the sea surface temperature up.

    The minimal exchange that closes the loop between a JCM atmosphere and a
    slab ocean, and the one every coupled JCM run starts from:

    - ``atm.derived.total_heat_flux`` -> ``ocn.forcing.total_heat_flux``. The
      JCM wrapper has already converted JCM's downward-positive ``hfluxn``
      into JEM's upward-positive convention, which is the sense
      :class:`~jem.components.slab.slab_ocean_model.SlabOceanModel` cools its
      mixed layer with, so this is a copy and not a sign flip.
    - ``ocn.state.sea_surface_temperature`` ->
      ``atm.forcing.sea_surface_temperature``, the field JCM's surface flux
      scheme reads (it is a prescribed boundary condition in an uncoupled JCM
      run; here the ocean provides it).

    Both fields are ``(ix, il)`` on the atmosphere's own nodal grid -- the
    slab grid is built from it with ``SlabGrid.from_coords`` -- so no
    regridding is involved.

    A new carry mapping is returned; nothing is written in place, which is
    what the coupler requires of an exchanger.
    """
    del time
    atmosphere = components["atm"]
    ocean = components["ocn"]
    return dict(
        components,
        atm=dict(
            atmosphere,
            forcing=atmosphere["forcing"].replace(
                sea_surface_temperature=ocean["state"].sea_surface_temperature
            ),
        ),
        ocn=dict(
            ocean,
            forcing=ocean["forcing"].replace(
                total_heat_flux=atmosphere["derived"].total_heat_flux
            ),
        ),
    )


def slab_exchange(components, time):
    """Wire the four slab models to each other.

    The slab atmosphere computes the surface heat flux itself, from the
    surface temperatures the ocean and the land hand it, so it is the source
    of the flux the other two are forced with. The sea ice is driven purely by
    the ocean's freeze/melt potential.

    ``atm.forcing.total_heat_flux`` is set from the atmosphere's own
    ``derived.internal_total_heat_flux`` for one reason only: that forcing
    slot is what the atmosphere writes to its output as ``total_heat_flux``,
    and leaving it at zero would report a flux of zero for a run that has one.
    """
    del time
    atmosphere = components["atm"]
    ocean = components["ocn"]
    land = components["lnd"]
    seaice = components["ice"]

    heat_flux = atmosphere["derived"].internal_total_heat_flux
    return dict(
        components,
        atm=dict(
            atmosphere,
            forcing=atmosphere["forcing"].replace(
                sea_surface_temperature=ocean["state"].sea_surface_temperature,
                land_surface_temperature=land["state"].land_surface_temperature,
                total_heat_flux=heat_flux,
            ),
        ),
        ocn=dict(ocean, forcing=ocean["forcing"].replace(total_heat_flux=heat_flux)),
        lnd=dict(land, forcing=land["forcing"].replace(total_heat_flux=heat_flux)),
        ice=dict(
            seaice,
            forcing=seaice["forcing"].replace(
                ice_frazil_melt_energy=ocean["derived"].ice_frazil_melt_energy
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def speedy_coords():
    """Return the coordinate system the JCM model and the slab grids share."""
    return get_speedy_coords(layers=LAYERS, spectral_truncation=TRUNCATION)


@pytest.fixture(scope="module")
def jcm_model(speedy_coords) -> Model:
    """Return a T21L5 SPEEDY aquaplanet -- the smallest atmosphere that runs."""
    return Model(
        coords=speedy_coords,
        terrain=TerrainData.aquaplanet(speedy_coords),
        start_date=START_DATE,
        calendar=CALENDAR,
    )


@pytest.fixture(scope="module")
def atmosphere_ocean(jcm_model) -> Coupler:
    """JCM plus a slab ocean on the atmosphere's own grid."""
    grid = SlabGrid.from_coords(jcm_model.coords.horizontal)
    return Coupler(
        {"atm": JCMComponent(jcm_model), "ocn": SlabOceanModel(grid)},
        {"exchange": atmosphere_ocean_exchange},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )


@pytest.fixture(scope="module")
def two_steps(atmosphere_ocean):
    """Two coupled steps of the JCM/slab-ocean model, run once for several tests."""
    initial = atmosphere_ocean.initialize()
    final, diagnostics = atmosphere_ocean.generate_trajectory_function(2)(initial)
    return initial, final, diagnostics


def assert_carries_close(actual, expected, rtol=1e-6, atol=1e-6):
    """Compare two coupled carries leaf by leaf.

    Integer and boolean leaves -- the coupled step counter, and the flags and
    counters inside JCM's physics carry -- must match exactly: they are
    decisions, not measurements, and a tolerance on them would hide a run that
    took a different branch. Float leaves are compared to a tolerance.
    """
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert actual_structure == expected_structure

    for index, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves)
    ):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        if expected_array.dtype.kind in "biu":
            np.testing.assert_array_equal(
                actual_array, expected_array, err_msg=f"leaf {index}"
            )
        else:
            np.testing.assert_allclose(
                actual_array, expected_array, rtol=rtol, atol=atol,
                err_msg=f"leaf {index}",
            )


def write_sst_climatology(path, grid: SlabGrid) -> str:
    """Write a month-varying, spatially uniform SST climatology on ``grid``.

    The coordinates are the grid's own, because
    :func:`~jem.components.slab.base.load_monthly_climatology` rejects a file
    that is not already on the model grid.
    """
    n_lon, n_lat = grid.shape
    monthly = 285.0 + 8.0 * np.cos(2 * np.pi * np.arange(12) / 12.0)
    values = np.broadcast_to(monthly[:, None, None], (12, n_lat, n_lon))
    dataset = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.array(values, dtype=np.float32))},
        coords={
            "time": np.arange(12),
            "lat": to_degrees(grid.latitude_axis_radian),
            "lon": to_degrees(grid.longitude_axis_radian),
        },
    )
    dataset.to_netcdf(path)
    return str(path)


# ---------------------------------------------------------------------------
# JCM coupled to a slab ocean
# ---------------------------------------------------------------------------


def test_two_coupled_steps_thread_the_carry(two_steps):
    """Two coupled steps run, and the carry that comes out is the one that went in.

    This is the check that ``lax.scan`` itself performs and that a
    single-component test cannot: the atmosphere's dycore state, its
    cross-step physics carry, the ocean's state and both components' forcing
    must come back with exactly the structure, shapes and dtypes they were
    handed, or the coupled model cannot be stepped a second time.
    """
    initial, final, diagnostics = two_steps

    assert jax.eval_shape(lambda: final) == jax.eval_shape(lambda: initial)
    assert int(final.step) == 2
    assert set(diagnostics) == {"atm", "ocn"}
    for name, component_diagnostics in diagnostics.items():
        for leaf in jax.tree_util.tree_leaves(component_diagnostics):
            assert jnp.shape(leaf)[0] == 2, name

    # And the run is physical, not merely well-shaped: two coupled steps of a
    # real atmosphere leave a finite surface flux on its own grid.
    heat_flux = final.components["atm"]["derived"].total_heat_flux
    assert heat_flux.shape == GRID_SHAPE
    assert bool(jnp.all(jnp.isfinite(heat_flux)))


def test_the_exchange_actually_moves_the_fields(two_steps):
    """The two components see each other, not their own initial conditions.

    Coupling is lagged: the exchanger at step *n* moves what each component
    produced during step *n-1*, so the ocean's first step is forced with the
    zero flux from ``initialize()`` and its second with the atmosphere's real
    one. Over two steps that is enough for both directions to have been used.
    """
    initial, final, diagnostics = two_steps

    # Down: the ocean is being forced with a real, finite atmospheric heat
    # flux (its own first step saw the zero one from `initialize()`), and it
    # responded -- the second record's SST is not the first's.
    ocean_flux = np.asarray(final.components["ocn"]["forcing"].total_heat_flux)
    assert ocean_flux.shape == GRID_SHAPE
    assert np.all(np.isfinite(ocean_flux))
    assert np.any(ocean_flux != 0.0)

    sea_surface_temperature = np.asarray(
        diagnostics["ocn"]["state"].sea_surface_temperature
    )
    assert np.any(sea_surface_temperature[1] != sea_surface_temperature[0])

    # Up: the atmosphere is running on the ocean's SST, not on the prescribed
    # climatology `default_forcing` gave it -- and on exactly the one the
    # ocean finished the PREVIOUS step with, which is what lagged coupling
    # means. The final exchange copied it before the ocean stepped again, so
    # the equality is exact rather than approximate.
    atmosphere_sst = np.asarray(
        final.components["atm"]["forcing"].sea_surface_temperature
    )
    prescribed = np.asarray(
        initial.components["atm"]["forcing"].sea_surface_temperature
    )
    assert atmosphere_sst.shape == GRID_SHAPE
    assert np.any(atmosphere_sst != prescribed)
    np.testing.assert_array_equal(atmosphere_sst, sea_surface_temperature[0])


def test_component_datasets_merge_on_one_time_and_grid(atmosphere_ocean, two_steps):
    """The atmosphere's and the ocean's output are one dataset, not two.

    ``join="exact"`` is the whole point: it refuses to align by taking a
    union, so it passes only if the two components wrote *identical*
    coordinate values -- the same ``datetime64[ns]`` instants (labelled at the
    end of each coupling interval, JCM's convention) and the same longitude
    and latitude in degrees. Getting either subtly wrong turns a merged
    coupled dataset into a mostly-empty outer join, which is what happens when
    a component invents its own time or grid encoding.
    """
    _, _, diagnostics = two_steps
    datasets = atmosphere_ocean.to_xarray(diagnostics)

    atmosphere = datasets["atm"]
    ocean = datasets["ocn"]

    assert atmosphere.time.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(atmosphere.time.values, ocean.time.values)
    np.testing.assert_array_equal(
        atmosphere.time.values,
        np.array(["2000-01-02", "2000-01-03"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(atmosphere.lon.values, ocean.lon.values)
    np.testing.assert_array_equal(atmosphere.lat.values, ocean.lat.values)

    merged = xr.merge([atmosphere, ocean], join="exact", compat="no_conflicts")

    assert merged.sizes["time"] == 2
    assert (merged.sizes["lon"], merged.sizes["lat"]) == GRID_SHAPE
    assert "sea_surface_temperature" in merged


@pytest.mark.slow
def test_continuous_equals_chunked_with_jcm(atmosphere_ocean):
    """Four steps in one call and two calls of two must give the same model.

    A chunked run only works because the clock lives in ``CoupledCarry.step``
    rather than in the ``lax.scan`` index, which restarts at zero on every
    call. If any component kept time of its own, or the coupler derived the
    date from the scan index, the two runs here would diverge -- and a
    production run written in chunks would silently be a different experiment
    from the same run written in one.
    """
    initial = atmosphere_ocean.initialize()

    continuous, _ = atmosphere_ocean.generate_trajectory_function(4)(initial)

    two = atmosphere_ocean.generate_trajectory_function(2)
    halfway, _ = two(initial)
    chunked, _ = two(halfway)

    assert int(continuous.step) == 4
    assert int(chunked.step) == 4
    assert_carries_close(chunked, continuous)


# ---------------------------------------------------------------------------
# An ECHAM atmosphere coupled to a slab ocean (jax-esm#129)
# ---------------------------------------------------------------------------
#
# ECHAM publishes no near-surface wind *vector* (only a speed) anywhere in its
# diagnostics, so `atm.derived.u0`/`.v0` are `None` throughout this coupled
# model (`jem.components.jcm.exchange_fields`, `JCMDerived`). Before jax-esm#129
# that made `JCMComponent.step` raise on the very first coupled step of ANY
# ECHAM configuration -- whatever the exchanger, including this one, which
# never reads `u0`/`v0` at all. This section is the regression test: an ECHAM
# atmosphere coupled to the slab ocean must complete several coupled steps,
# with finite heat and water fluxes, exactly as the SPEEDY case above does.


#: ECHAM's own stable-step default (``Model``'s 12-minute production step,
#: 720 s, verified against a real build rather than assumed). The coupling
#: timestep below must be a whole multiple of this (``JCMComponent.bind``'s
#: own requirement) -- but a bare multiple is not enough: it must ALSO be a
#: power-of-two fraction of a day, or the atmosphere's own time labels and
#: the ones ``jem.base.component.TimeAxis`` computes for the ocean part
#: company by up to 128 ns (that class's own docstring: "a 10- or 20-minute
#: coupling step puts roughly half the labels 128 ns off", tracked upstream
#: as jax-gcm#862) -- which breaks the ``xr.merge(..., join="exact")`` this
#: test's own output check relies on. 21600 s (6 hours, 1/4 of a day) is the
#: smallest such interval above the model step -- 30 atmospheric steps per
#: coupled step, exactly divisible both ways.
_ECHAM_MODEL_TIMESTEP_SECONDS = 720
ECHAM_COUPLING_TIMESTEP = jdt.to_timedelta(6 * 3600, "second")


def _build_echam_model(echam_coords) -> Model:
    """T31L47 ECHAM on an aquaplanet -- the smallest hybrid-vertical-coordinate
    setup jax-gcm's own test suite runs a real model on
    (jax-gcm's ``model_test.py``, ``test_echam_hybrid_model_output_averages``:
    "smallest hybrid setup that exercises the same code path"). ECHAM's
    hybrid levels are a fixed table (``get_echam_levels`` only defines 47 or
    95 levels), so unlike SPEEDY's ``layers=5`` there is no coarser built-in
    vertical resolution to drop to; T31 is the coarsest horizontal truncation
    used for it.
    """
    from jcm.physics.echam.echam_terms import echam_physics

    model = Model(
        coords=echam_coords,
        terrain=TerrainData.aquaplanet(echam_coords),
        physics=echam_physics(radiation_scheme="grey", checkpoint_terms=False),
        start_date=START_DATE,
        calendar=CALENDAR,
    )
    actual = int(model.dt_si.to_timedelta().total_seconds())
    assert actual == _ECHAM_MODEL_TIMESTEP_SECONDS, (
        f"ECHAM's stable-step default changed ({actual}s, expected "
        f"{_ECHAM_MODEL_TIMESTEP_SECONDS}s) -- re-check ECHAM_COUPLING_TIMESTEP "
        "still divides it evenly before trusting this test's timing."
    )
    return model


@pytest.fixture(scope="module")
def echam_coords():
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.utils import get_coords

    return get_coords(get_echam_levels(47), spectral_truncation=31)


@pytest.fixture(scope="module")
def echam_model(echam_coords) -> Model:
    return _build_echam_model(echam_coords)


@pytest.fixture(scope="module")
def echam_atmosphere_ocean(echam_model) -> Coupler:
    """Couple an ECHAM atmosphere to a slab ocean with the same minimal
    exchange as the SPEEDY case above (heat flux down, SST up) -- it never
    reads ``derived.u0``/``.v0``, so their absence must not stop the run.
    """
    grid = SlabGrid.from_coords(echam_model.coords.horizontal)
    return Coupler(
        {"atm": JCMComponent(echam_model), "ocn": SlabOceanModel(grid)},
        {"exchange": atmosphere_ocean_exchange},
        coupling_timestep=ECHAM_COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )


@pytest.mark.slow
def test_echam_coupled_to_a_slab_ocean_completes_several_steps(echam_atmosphere_ocean):
    """jax-esm#129: an ECHAM-composed coupled model must complete coupled steps.

    Checks, on a real ECHAM model, that a coupled step actually completes:

    - the initial and the post-step carry both have ``derived.u0``/``.v0`` as
      ``None`` (the static absence design), and the carry's overall pytree
      structure survives the step (what ``lax.scan`` itself enforces, and
      what the previous ``NotImplementedError`` never let this model reach);
    - the heat flux that reaches the slab ocean's forcing, and the
      atmosphere's own derived heat and freshwater fluxes, are finite and
      non-trivial after several steps;
    - the run's output still serializes and merges across components
      (``Coupler.to_xarray``), the same round trip the SPEEDY case's
      ``test_component_datasets_merge_on_one_time_and_grid`` checks.

    The freshwater flux is checked on ``atm.derived`` rather than on the
    ocean's own forcing: ``jem.components.slab.slab_ocean_model.SlabOceanModel``
    has no freshwater intake at all (its ``OceanForcing`` carries only
    ``total_heat_flux``/``q_flux`` -- it models temperature, not salinity), so
    only the heat flux is actually exchanged here, exactly as in the SPEEDY
    case's own ``atmosphere_ocean_exchange``. That the atmosphere's
    freshwater flux is finite and computed is still evidence #754's water
    flux works for ECHAM; it is jax-esm#128, not #129, that would give a slab
    or Veros ocean somewhere to put it.
    """
    initial = echam_atmosphere_ocean.initialize()
    assert initial.components["atm"]["derived"].u0 is None
    assert initial.components["atm"]["derived"].v0 is None

    n_steps = 3
    trajectory = echam_atmosphere_ocean.generate_trajectory_function(n_steps)
    final, diagnostics = trajectory(initial)

    assert jax.eval_shape(lambda: final) == jax.eval_shape(lambda: initial)
    assert int(final.step) == n_steps
    assert final.components["atm"]["derived"].u0 is None
    assert final.components["atm"]["derived"].v0 is None

    ocean_heat_flux = np.asarray(final.components["ocn"]["forcing"].total_heat_flux)
    assert np.all(np.isfinite(ocean_heat_flux))
    assert np.any(ocean_heat_flux != 0.0)

    atm_derived = final.components["atm"]["derived"]
    for name in ("total_heat_flux", "total_freshwater_flux",
                 "evaporation", "precipitation"):
        field = np.asarray(getattr(atm_derived, name))
        assert np.all(np.isfinite(field)), name

    datasets = echam_atmosphere_ocean.to_xarray(diagnostics)
    merged = xr.merge(
        [datasets["atm"], datasets["ocn"]], join="exact", compat="no_conflicts"
    )
    assert merged.sizes["time"] == n_steps


# ---------------------------------------------------------------------------
# The four slab models coupled to each other
# ---------------------------------------------------------------------------


@pytest.fixture
def slab_coupler(speedy_coords, tmp_path) -> Coupler:
    """All four slab models on a half-land T21 grid, ocean relaxing to a climatology.

    Half the grid is land so that both branches of every mask run: the ocean
    and the sea ice integrate the western half, the land model the eastern.
    """
    n_lon, n_lat = speedy_coords.horizontal.nodal_shape
    fractional_mask = jnp.where(
        jnp.arange(n_lon)[:, None] >= n_lon // 2,
        jnp.ones((n_lon, n_lat)),
        jnp.zeros((n_lon, n_lat)),
    )
    grid = SlabGrid.from_coords(
        speedy_coords.horizontal, fractional_mask=fractional_mask
    )
    ocean = SlabOceanModel(
        grid,
        SlabOceanParameters(
            forcing_method="relaxation", relaxation_time=RELAXATION_TIME
        ),
        sst_clim_file=write_sst_climatology(tmp_path / "sst.nc", grid),
    )
    return Coupler(
        {
            "atm": SlabAtmosphereModel(grid),
            "ocn": ocean,
            "lnd": SlabLandModel(grid),
            "ice": SlabSeaiceModel(grid),
        },
        {"exchange": slab_exchange},
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )


def test_four_slab_components_run_coupled(slab_coupler):
    """Four components, one exchanger, three steps under ``jax.jit``.

    ``generate_trajectory_function`` jits the whole coupled trajectory, so a
    component that branched on a traced value, allocated a shape that depends
    on one, or returned a carry that differs from the one it was given would
    fail here rather than at the first long run.
    """
    initial = slab_coupler.initialize()
    assert slab_coupler.workflow == ("exchange", "atm", "ocn", "lnd", "ice")

    final, diagnostics = slab_coupler.generate_trajectory_function(3)(initial)

    assert jax.eval_shape(lambda: final) == jax.eval_shape(lambda: initial)
    assert int(final.step) == 3
    assert set(diagnostics) == {"atm", "ocn", "lnd", "ice"}
    for name, component_diagnostics in diagnostics.items():
        for leaf in jax.tree_util.tree_leaves(component_diagnostics):
            assert jnp.shape(leaf)[0] == 3, name
            assert bool(jnp.all(jnp.isfinite(leaf))), name


def test_all_four_slab_datasets_merge(slab_coupler):
    """Every component of a coupled run merges into one dataset.

    Three of these four components are forced with a heat flux and two of them
    see the ocean's freeze/melt potential, so without a naming convention the
    same physical name would arrive from several components carrying different
    values -- coupling is lagged, so a received copy is a step behind the
    original -- and ``xr.merge`` would refuse the lot. A field a component was
    *given* is therefore written with the ``forcing_`` prefix, and only the
    component that produced a field writes it plain.
    """
    initial = slab_coupler.initialize()
    _, diagnostics = slab_coupler.generate_trajectory_function(3)(initial)

    datasets = slab_coupler.to_xarray(diagnostics)
    assert set(datasets) == {"atm", "ocn", "lnd", "ice"}

    merged = xr.merge(datasets.values(), join="exact", compat="no_conflicts")

    assert merged.sizes["time"] == 3
    assert (merged.sizes["lon"], merged.sizes["lat"]) == GRID_SHAPE
    # One variable per component survives under its own name, and the fields
    # the components received are there too, prefixed.
    for name in (
        "mean_air_temperature",
        "sea_surface_temperature",
        "land_surface_temperature",
        "ice_thickness",
        "internal_total_heat_flux",
        "total_heat_flux",
        "forcing_total_heat_flux",
        "forcing_ice_frazil_melt_energy",
    ):
        assert name in merged, name


def test_ocean_output_follows_the_method_the_coupled_run_was_started_with(
    slab_coupler,
):
    """A coupled run switched into Q-flux mode publishes the Q-flux it applied.

    The ocean of ``slab_coupler`` is *built* for relaxation, but the run is
    started from Q-flux parameters, so its steps apply a Q-flux and its output
    has to say so. Going through ``generate_trajectory_function`` is the point:
    the Q-flux snapshot rides as a conditional key of the ocean's diagnostics,
    and only ``lax.scan`` checks that the structure it stacks is the same on
    every step.
    """
    ocean_params = slab_coupler.components["ocn"].params
    initial = slab_coupler.initialize(
        {"ocn": ocean_params.replace(forcing_method="qflux")}
    )

    _, diagnostics = slab_coupler.generate_trajectory_function(2)(initial)
    ocean = slab_coupler.to_xarray(diagnostics)["ocn"]

    assert "forcing_q_flux" in ocean
    assert ocean["forcing_q_flux"].sizes["time"] == 2
    # The ocean built for relaxation still publishes no Q-flux when the run
    # keeps that method.
    _, relaxation_diagnostics = slab_coupler.generate_trajectory_function(2)(
        slab_coupler.initialize()
    )
    assert "forcing_q_flux" not in slab_coupler.to_xarray(relaxation_diagnostics)["ocn"]


def test_grad_of_sst_wrt_relaxation_time_through_the_coupler(slab_coupler):
    """A slab parameter stays differentiable through a whole coupled run.

    The parameters travel in the component's carry rather than in a closure
    over the model object, precisely so this works with no special casing in
    the coupler: perturb one leaf of the initial coupled carry, run three
    coupled steps of four components and an exchanger, and differentiate the
    result with respect to it.
    """
    initial = slab_coupler.initialize()
    trajectory = slab_coupler.generate_trajectory_function(3)

    def mean_sea_surface_temperature(relaxation_time):
        ocean = initial.components["ocn"]
        params = ocean["params"].replace(relaxation_time=relaxation_time)
        components = dict(initial.components, ocn=dict(ocean, params=params))
        final, _ = trajectory(dataclasses.replace(initial, components=components))
        return jnp.mean(final.components["ocn"]["state"].sea_surface_temperature)

    gradient = jax.grad(mean_sea_surface_temperature)(jnp.float32(RELAXATION_TIME))

    assert np.isfinite(gradient)
    # The relaxation pulls the mixed layer back towards its climatology, so a
    # longer timescale leaves more of the step's warming in place; the
    # magnitude is small (kelvin per second of timescale) but must not be the
    # zero that a parameter hidden from the gradient would produce.
    assert abs(float(gradient)) > 1e-12
