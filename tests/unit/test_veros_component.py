"""Tests for the Veros adapter (:mod:`jem.components.veros_component`).

Veros is an optional dependency, so the whole module skips when it is
absent. The model is Veros' own ``acc_basic`` example setup (30 x 42 x 15),
the smallest one it ships; it is built once per module because ``setup()``
plus the first compiled step dominates the runtime.
"""

import logging
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest

# Importing the adapter is what points Veros at its JAX backend, and it has
# to happen before anything imports veros.core -- including the setup module
# below. Import it first, and skip the module when Veros is not installed.
pytest.importorskip("veros")

from jem.base.component import (  # noqa: E402
    Component,
    CouplingTime,
    SupportsBind,
    SupportsCheckpoint,
    SupportsXarray,
    TimeAxis,
    forcing_variable,
)
from jem.components.veros_component import (  # noqa: E402
    GHOST_CELLS,
    VerosComponent,
    VerosDerived,
    VerosForcing,
    configure_veros_runtime,
)

START_DATE = jdt.to_datetime("2000-01-01")
CALENDAR = "365_day"
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")


def _acc_basic_model(workdir, **setting_overrides):
    """Yield a set-up ``acc_basic``, with Veros' own output diagnostics off.

    Two reasons for turning them off, both of which apply to a real coupled
    run as much as to this test: a coupled run's output comes from the
    coupler, not from each component writing its own files; and Veros'
    NetCDF writer goes through h5netcdf, which fails inside a process that
    has already loaded another HDF5 binding (importing ``jcm``/``xarray``
    pulls in ``netCDF4``) -- so leaving them on makes this module pass alone
    and fail in a suite.

    ``setting_overrides`` are applied after the setup's own
    ``set_parameter``, which is how the free-surface variant of the case is
    built.
    """
    from veros import veros_routine
    from veros.setups.acc_basic.acc_basic import ACCBasicSetup

    class CoupledACCBasic(ACCBasicSetup):
        @veros_routine
        def set_parameter(self, state):
            super().set_parameter(state)
            for name, value in setting_overrides.items():
                setattr(state.settings, name, value)

        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    # Belt and braces: run from a scratch directory so anything Veros does
    # write (a restart, say) lands there rather than in the repository.
    previous_directory = os.getcwd()
    os.chdir(workdir)
    try:
        model = CoupledACCBasic()
        model.setup()
        yield model
    finally:
        os.chdir(previous_directory)


@pytest.fixture(scope="module")
def veros_model(tmp_path_factory):
    yield from _acc_basic_model(tmp_path_factory.mktemp("veros_acc_basic"))


@pytest.fixture(scope="module")
def free_surface_veros_model(tmp_path_factory):
    """``acc_basic`` again, but solving the external mode for a free surface.

    Veros' own default -- which ``acc_basic`` keeps -- is to solve for a
    barotropic streamfunction, while every Veros setup shipped with JEM
    turns that off. The two branches of the ``psi`` output are genuinely
    different code paths reading different Veros fields, so covering the one
    the shipped setups take needs a second model.
    """
    yield from _acc_basic_model(
        tmp_path_factory.mktemp("veros_free_surface"),
        enable_streamfunction=False,
    )


@pytest.fixture(scope="module")
def grid_shape(veros_model):
    return (veros_model.state.dimensions["xt"],
            veros_model.state.dimensions["yt"])


def _bound(model) -> VerosComponent:
    """Wrap ``model`` and bind it to this module's coupler clock."""
    wrapper = VerosComponent(model)
    wrapper.bind(
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )
    return wrapper


@pytest.fixture
def component(veros_model) -> VerosComponent:
    return _bound(veros_model)


@pytest.fixture
def free_surface_component(free_surface_veros_model) -> VerosComponent:
    return _bound(free_surface_veros_model)


def _coupling_time(step: int) -> CouplingTime:
    """Build the clock the coupler hands a component on step ``step``."""
    return CouplingTime(
        step=jnp.int32(step),
        sim_time=jnp.float32(step * 86400.0),
        dt=86400.0,
        year_offset_seconds=0.0,
        days_per_year=365.0,
    )


def _labelling_diagnostics(component, n_records):
    """Return ``n_records`` of zero-filled diagnostics in ``step``'s layout.

    ``to_xarray`` labels what it is handed and reads nothing else, so this
    layout is all it needs -- and building it here rather than by
    integrating keeps the labelling checks in the fast suite, where the
    others of their kind cost a Veros integration apiece.
    """
    nx, ny = component.horizontal_shape
    nz = int(component.dzt.shape[0])
    surface = jnp.zeros((n_records, nx, ny))
    volume = jnp.zeros((n_records, nx, ny, nz))
    diagnostics = {name: volume for name in ("temp", "salt", "u", "v")}
    diagnostics.update({
        name: surface
        for name in (
            "psi",
            "sea_surface_temperature", "sea_surface_salinity",
            "sea_surface_u", "sea_surface_v",
            "surface_air_temperature", "surface_taux", "surface_tauy",
            "heat_flux", "freshwater_flux",
        )
    })
    return diagnostics


def _step_under_wind(component, n_steps):
    """Step ``component`` under a zonal wind stress; return the last step.

    The coupler's forcing replaces the setup's own, so an ocean stepped from
    a zero-filled carry stays exactly at rest -- and a streamfunction of
    zeros agrees with anything. The stress is the shape ``acc_basic`` drives
    itself with, applied afresh each step because the exchangers a real run
    has are not in the loop here.
    """
    carry = component.initialize()
    nx, ny = component.horizontal_shape
    latitude = np.asarray(component.latitude)
    taux = jnp.asarray(np.broadcast_to(
        0.1 * np.sin(np.pi * (latitude - latitude.min()) / np.ptp(latitude)),
        (nx, ny),
    ))
    diagnostics = None
    for step in range(n_steps):
        carry = dict(carry,
                     forcing=carry["forcing"].replace(surface_taux=taux))
        carry, diagnostics = component.step(carry, _coupling_time(step))
    return carry, diagnostics


def _integrated_transport(component, zonal_velocity):
    """Integrate the depth-integrated zonal transport northwards, on the host.

    The reference the ``psi`` tests check against, written as the recurrence
    Veros' barotropic-mode update inverts -- ``psi[j] = psi[j-1]
    - U[j] * dyt[j]``, from a southern boundary where psi vanishes -- rather
    than as a second cumulative sum, so that it checks the axes and the
    starting point as well as the arithmetic.
    """
    transport = np.sum(
        np.asarray(zonal_velocity)
        * np.asarray(component.mask_U)
        * np.asarray(component.dzt),
        axis=-1,
    )
    dyt = np.asarray(component.dlatitude)
    psi = np.zeros_like(transport)
    southern_neighbour = np.zeros(transport.shape[0])
    for j in range(transport.shape[1]):
        southern_neighbour = southern_neighbour - transport[:, j] * dyt[j]
        psi[:, j] = southern_neighbour
    return psi


def test_configure_veros_runtime_is_idempotent():
    """Re-running the backend selection after Veros locked it is a no-op."""
    from veros import runtime_settings

    configure_veros_runtime()
    assert runtime_settings.backend == "jax"


def test_component_satisfies_protocols(component):
    """The wrapper is what the coupler tests for with ``isinstance``."""
    assert isinstance(component, Component)
    assert isinstance(component, SupportsBind)
    assert isinstance(component, SupportsXarray)
    assert isinstance(component, SupportsCheckpoint)
    assert component.name == "ocn"


def test_construction_disables_the_setups_own_forcing(veros_model):
    """Veros calls set_forcing every step; a coupled run must neutralise it."""
    VerosComponent(veros_model)
    assert veros_model.set_forcing(veros_model.state) is None


def test_grid_metadata_drops_the_halo(component, veros_model, grid_shape):
    """The exchanged fields are the interior, without Veros' ghost cells."""
    nx, ny = grid_shape
    assert component.mask_T.shape[:2] == (nx, ny)
    assert component.longitude.shape == (nx,)
    assert component.latitude.shape == (ny,)
    assert (component.mask_T.shape[0]
            == veros_model.state.variables.maskT.shape[0] - 2 * GHOST_CELLS)


def test_to_xarray_publishes_the_barotropic_streamfunction(component, grid_shape):
    """`psi` is always in the output, labelled, and says which psi it is."""
    nx, ny = grid_shape
    diagnostics = _labelling_diagnostics(component, 2)
    psi = jnp.asarray(
        np.linspace(-1e6, 1e6, 2 * nx * ny).reshape(2, nx, ny))
    diagnostics["psi"] = psi

    dataset = component.to_xarray(
        diagnostics,
        TimeAxis(START_DATE, np.arange(2), COUPLING_TIMESTEP, CALENDAR),
    )

    assert dataset.psi.dims == ("time", "lon", "lat")
    assert dataset.psi.attrs["units"] == "m3 s-1"
    assert dataset.psi.attrs["long_name"] == "barotropic streamfunction"
    assert dataset.psi.attrs["jem_role"] == "derived"
    np.testing.assert_array_equal(dataset.psi.values, np.asarray(psi))
    assert np.isfinite(dataset.psi.values).all()

    # Nothing in the numbers says whether this is Veros' own prognostic
    # streamfunction or the diagnosis that stands in for it under a free
    # surface, so the comment has to -- along with the staggering, which the
    # `lon`/`lat` labels do not carry either.
    comment = dataset.psi.attrs["comment"]
    assert ("prognostic" in comment) is component.enable_streamfunction
    assert ("diagnosed" in comment) is not component.enable_streamfunction
    assert "zeta" in comment


def test_the_diagnosed_streamfunction_integrates_the_zonal_transport(
    component, grid_shape
):
    """The diagnosis is the northward integral of the zonal transport."""
    nx, ny = grid_shape
    nz = int(component.dzt.shape[0])
    rng = np.random.default_rng(20250918)
    zonal_velocity = jnp.asarray(rng.standard_normal((nx, ny, nz)))

    psi = np.asarray(component._barotropic_streamfunction(zonal_velocity))

    expected = _integrated_transport(component, zonal_velocity)
    np.testing.assert_allclose(psi, expected, rtol=1e-5)
    assert np.abs(psi).max() > 0
    # The integration starts from a boundary where psi vanishes, so the
    # southernmost emitted row holds one cell's worth of transport and no
    # accumulated history.
    np.testing.assert_allclose(psi[:, 0], expected[:, 0], rtol=1e-5)


def test_initialize_carry_structure(component, grid_shape):
    """``initialize()`` returns the three-key carry, without integrating."""
    carry = component.initialize()

    assert set(carry) == {"state", "derived", "forcing"}
    assert isinstance(carry["derived"], VerosDerived)
    assert isinstance(carry["forcing"], VerosForcing)
    assert carry["derived"].sea_surface_temperature.shape == grid_shape
    assert carry["forcing"].heat_flux.shape == grid_shape
    assert carry["state"] is component.model.state


def test_step_before_bind_raises(veros_model):
    """Stepping an unregistered component names the fix."""
    wrapper = VerosComponent(veros_model)
    with pytest.raises(RuntimeError, match="bind"):
        wrapper.step(wrapper.initialize(), _coupling_time(0))


def test_bind_rejects_non_multiple_timestep(veros_model):
    """The coupling interval must be a whole number of tracer timesteps."""
    wrapper = VerosComponent(veros_model)
    tracer_seconds = int(veros_model.state.settings.dt_tracer)
    with pytest.raises(ValueError, match="whole multiple"):
        wrapper.bind(
            coupling_timestep=jdt.to_timedelta(tracer_seconds + 1, "second"),
            start_date=START_DATE,
            calendar=CALENDAR,
        )


def test_bind_sets_the_internal_step_count(component, veros_model):
    """One coupling day is a whole number of Veros tracer steps."""
    tracer_seconds = int(veros_model.state.settings.dt_tracer)
    assert component._steps_per_coupling_step == 86400 // tracer_seconds


def test_bind_records_the_veros_clock_zero_point(component, veros_model):
    """The Veros time at registration is what the coupler's start date means."""
    assert component._veros_time_zero == float(veros_model.state.variables.time)


def _carry_at_veros_time(component, coupler_seconds):
    """Return a carry whose Veros clock reads ``coupler_seconds`` into the run.

    ``_report_clock_drift`` reads exactly one field, so the whole Veros state
    can be stood in for here: building a real drifted ``VerosState`` would
    mean integrating the ocean to get one, which is what the check exists to
    make unnecessary.
    """
    veros_time = component._veros_time_zero + coupler_seconds
    return {"state": SimpleNamespace(
        variables=SimpleNamespace(time=jnp.float32(veros_time)))}


def test_clock_drift_is_silent_when_the_clocks_agree(component, caplog):
    """An ocean one day into the run, on the coupler's second step, is fine."""
    carry = _carry_at_veros_time(component, 86400.0)

    with caplog.at_level(logging.ERROR, logger="jem.components.veros_component"):
        component._report_clock_drift(carry, _coupling_time(1))
        jax.effects_barrier()

    assert caplog.text == ""


def test_clock_drift_of_a_day_is_reported(component, caplog):
    """An ocean a day ahead of the coupler names itself and both clocks."""
    carry = _carry_at_veros_time(component, 2 * 86400.0)

    with caplog.at_level(logging.ERROR, logger="jem.components.veros_component"):
        component._report_clock_drift(carry, _coupling_time(1))
        jax.effects_barrier()

    assert "ocn" in caplog.text
    # The drift, and both clocks in the coupler's frame: 172800 s of ocean
    # against 86400 s of coupler.
    assert "86400" in caplog.text
    assert "172800" in caplog.text
    assert caplog.records and caplog.records[0].levelno == logging.ERROR


def test_make_jem_compatible_is_deprecated(veros_model):
    """The old entry point still works and warns."""
    from jem.components import veros_component

    with pytest.warns(DeprecationWarning, match="VerosComponent"):
        wrapper = veros_component.make_jem_compatible(
            veros_model, COUPLING_TIMESTEP)

    assert isinstance(wrapper, VerosComponent)
    assert wrapper.model is veros_model


@pytest.mark.slow
def test_step_advances_and_returns_a_stackable_carry(component, grid_shape):
    """A step returns the carry structure it received, with finite fields."""
    carry0 = component.initialize()
    carry1, diagnostics = component.step(carry0, _coupling_time(0))

    assert set(carry1) == set(carry0)
    assert jax.tree.structure(carry1) == jax.tree.structure(carry0)
    sst = carry1["derived"].sea_surface_temperature
    assert sst.shape == grid_shape
    assert bool(jnp.all(jnp.isfinite(sst)))
    assert diagnostics["temp"].shape[:2] == grid_shape


@pytest.mark.slow
def test_to_xarray_has_time_axis_of_length_n(component, grid_shape):
    """Stacked per-step diagnostics label a time axis one record per step."""
    carry = component.initialize()
    carry, first = component.step(carry, _coupling_time(0))
    _, second = component.step(carry, _coupling_time(1))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), first, second)

    dataset = component.to_xarray(
        stacked, TimeAxis(START_DATE, np.arange(2), COUPLING_TIMESTEP, CALENDAR))

    assert dataset.sizes["time"] == 2
    assert dataset.sizes["lon"], dataset.sizes["lat"] == grid_shape
    # Fields the ocean was given carry the forcing_ prefix, as the slab
    # models' do, so an atmosphere dataset and this one merge.
    assert dataset.forcing_heat_flux.attrs["units"] == "W/m^2"
    assert set(dataset.data_vars) >= {
        forcing_variable(name)
        for name in ("heat_flux", "freshwater_flux", "surface_taux",
                     "surface_tauy", "surface_air_temperature")
    }
    # ...and what the ocean computed keeps its plain name.
    assert "sea_surface_temperature" in dataset.data_vars
    # The one time coordinate of the run, not a bare 0..n-1 index: an ocean
    # dataset has to merge with the atmosphere's on the same instants.
    time_axis = TimeAxis(START_DATE, np.arange(2), COUPLING_TIMESTEP, CALENDAR)
    np.testing.assert_array_equal(dataset.time.values, time_axis.datetimes())
    assert dataset.time.attrs == time_axis.attrs


@pytest.mark.slow
def test_psi_is_veros_own_streamfunction_when_the_run_solves_for_one(component):
    """In streamfunction mode `psi` is Veros', and the diagnosis recovers it.

    This is the check that the diagnosis' sign and metric factors are the
    ones Veros uses: the wrapper publishes Veros' own field here, and the
    diagnosis -- what a free-surface run gets instead -- has to come back to
    it. It can only do so up to a constant, because a streamfunction is
    defined up to one and the two fix it differently: Veros holds its first
    island at zero, the diagnosis the southern boundary. In this channel
    setup the difference is the throughflow, which is why the check is on
    the spread of the difference rather than on the difference.
    """
    assert component.enable_streamfunction

    carry, diagnostics = _step_under_wind(component, n_steps=2)

    variables = carry["state"].variables
    interior = slice(GHOST_CELLS, -GHOST_CELLS)
    psi = np.asarray(diagnostics["psi"])
    np.testing.assert_array_equal(
        psi, np.asarray(variables.psi[interior, interior, variables.tau]))
    assert np.isfinite(psi).all()
    # The wind has spun something up, so what follows is not two fields of
    # zeros agreeing with each other.
    assert np.abs(psi).max() > 1.0

    diagnosed = np.asarray(
        component._barotropic_streamfunction(diagnostics["u"]))
    gauge = diagnosed - psi
    np.testing.assert_allclose(
        gauge, gauge.mean(), rtol=0, atol=1e-5 * np.abs(psi).max())
    # ...and the host-side reference is the same field, so the relation the
    # diagnosis implements is the one written down in its docstring.
    np.testing.assert_allclose(
        diagnosed, _integrated_transport(component, diagnostics["u"]),
        rtol=1e-5)


@pytest.mark.slow
def test_psi_is_diagnosed_when_the_run_solves_a_free_surface(
    free_surface_component
):
    """Under a free surface `psi` is diagnosed, not read from Veros.

    Which matters more than a missing field would: in this mode Veros reuses
    `variables.psi` for the surface pressure (m^2 s^-2, on the T grid), so
    publishing it would have published a different quantity under the
    streamfunction's name. Every Veros setup shipped with JEM runs this way.
    """
    component = free_surface_component
    assert not component.enable_streamfunction

    carry, diagnostics = _step_under_wind(component, n_steps=1)

    psi = np.asarray(diagnostics["psi"])
    assert np.isfinite(psi).all()
    assert np.abs(psi).max() > 1.0
    np.testing.assert_allclose(
        psi, _integrated_transport(component, diagnostics["u"]), rtol=1e-5)

    variables = carry["state"].variables
    interior = slice(GHOST_CELLS, -GHOST_CELLS)
    surface_pressure = np.asarray(
        variables.psi[interior, interior, variables.tau])
    assert not np.allclose(psi, surface_pressure)


@pytest.mark.slow
def test_to_xarray_rejects_a_mismatched_time_axis(component):
    """A time axis that does not match the records is a coupler-side bug."""
    carry = component.initialize()
    _, diagnostics = component.step(carry, _coupling_time(0))
    stacked = jax.tree.map(lambda x: jnp.stack([x]), diagnostics)

    with pytest.raises(ValueError, match="output records"):
        component.to_xarray(
            stacked,
            TimeAxis(START_DATE, np.arange(3), COUPLING_TIMESTEP, CALENDAR))


@pytest.mark.slow
def test_a_drifted_clock_is_reported_and_the_step_still_runs(component, caplog):
    """A mismatch is loud but never aborts: the run may still be salvageable."""
    carry = component.initialize()

    with caplog.at_level(logging.ERROR, logger="jem.components.veros_component"):
        # The carry is the ocean at the start of the run; the clock says five
        # days have passed, as a restart paired with the wrong step counter
        # would.
        new_carry, diagnostics = component.step(carry, _coupling_time(5))
        jax.effects_barrier()

    assert "model clock is" in caplog.text
    assert set(new_carry) == set(carry)
    assert bool(jnp.all(jnp.isfinite(
        new_carry["derived"].sea_surface_temperature)))
    assert "temp" in diagnostics


def test_rebinding_to_a_different_timestep_is_rejected(component):
    """One instance belongs to one coupled model; a conflicting second bind raises."""
    component.bind(
        coupling_timestep=COUPLING_TIMESTEP, start_date=START_DATE, calendar=CALENDAR
    )
    with pytest.raises(ValueError, match="already bound"):
        component.bind(
            coupling_timestep=COUPLING_TIMESTEP * 2,
            start_date=START_DATE,
            calendar=CALENDAR,
        )


@pytest.mark.slow
def test_save_state_and_load_state_round_trip(component, grid_shape, tmp_path):
    """The carry survives the split between the HDF5 restart and the carry file.

    The ``VerosState`` goes through Veros' own restart writer and the rest of
    the carry through :mod:`jem.checkpoint`, and only a round trip shows that
    the two halves come back as one carry -- with the restart file where the
    loader looks for it, and the pytree half restored leaf for leaf.
    """
    from jem.checkpoint import CARRY_FILENAME
    from jem.components.veros_component import VEROS_RESTART_FILENAME

    carry = component.initialize()
    carry = dict(
        carry,
        derived=carry["derived"].replace(
            sea_surface_temperature=jnp.full(grid_shape, 290.5)
        ),
        forcing=carry["forcing"].replace(heat_flux=jnp.full(grid_shape, -12.5)),
    )

    directory = tmp_path / "ocn"
    component.save_state(carry, directory)
    assert (directory / VEROS_RESTART_FILENAME).exists()
    # The carry file is written last: it is the completion marker of this
    # component's directory as much as of a coupled checkpoint.
    assert (directory / CARRY_FILENAME).exists()

    loaded = component.load_state(directory)

    assert set(loaded) == {"state", "derived", "forcing"}
    # Veros' reader mutates the model's state in place, so the restored carry
    # shares that object rather than holding a copy of it.
    assert loaded["state"] is component.model.state
    np.testing.assert_array_equal(
        np.asarray(loaded["derived"].sea_surface_temperature),
        np.asarray(carry["derived"].sea_surface_temperature),
    )
    np.testing.assert_array_equal(
        np.asarray(loaded["forcing"].heat_flux),
        np.asarray(carry["forcing"].heat_flux),
    )


def test_a_runtime_setting_is_restored_even_when_the_block_raises():
    """The process-global Veros setting goes back however the block ends.

    `load_state` has to turn `force_overwrite` off to read a restart, and the
    rest of a coupled run needs it on. The settings are process-global, so a
    failed read -- a missing or mismatched HDF5 file -- must not leave the
    flag flipped: the next thing to write an output would then fail for a
    reason with no connection to what actually went wrong.
    """
    from veros import runtime_settings

    from jem.components.veros_component import _veros_runtime_setting

    before = runtime_settings.force_overwrite
    locked_before = getattr(runtime_settings, "__locked__", False)

    with pytest.raises(RuntimeError, match="restart is missing"):
        with _veros_runtime_setting("force_overwrite", not before):
            assert runtime_settings.force_overwrite is (not before)
            raise RuntimeError("the restart is missing")

    assert runtime_settings.force_overwrite is before
    # And the lock the settings were under is put back as it was, rather than
    # assumed: this module runs before and after `veros.core` is imported.
    assert getattr(runtime_settings, "__locked__", False) is locked_before


@pytest.mark.slow
def test_load_state_restores_force_overwrite_when_the_restart_is_missing(
    component, tmp_path
):
    """A failed restart read leaves the runtime settings as it found them."""
    from veros import runtime_settings

    before = runtime_settings.force_overwrite
    with pytest.raises(Exception):
        component.load_state(tmp_path / "not-a-checkpoint")
    assert runtime_settings.force_overwrite is before
