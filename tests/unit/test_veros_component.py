"""Tests for the Veros adapter (:mod:`jem.components.veros_component`).

Veros is an optional dependency, so the whole module skips when it is
absent. The model is Veros' own ``acc_basic`` example setup (30 x 42 x 15),
the smallest one it ships; it is built once per module because ``setup()``
plus the first compiled step dominates the runtime.
"""

import os

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


@pytest.fixture(scope="module")
def veros_model(tmp_path_factory):
    from veros import veros_routine
    from veros.setups.acc_basic.acc_basic import ACCBasicSetup

    class CoupledACCBasic(ACCBasicSetup):
        """``acc_basic`` with Veros' own output diagnostics switched off.

        Two reasons, both of which apply to a real coupled run as much as
        to this test: a coupled run's output comes from the coupler, not
        from each component writing its own files; and Veros' NetCDF writer
        goes through h5netcdf, which fails inside a process that has
        already loaded another HDF5 binding (importing ``jcm``/``xarray``
        pulls in ``netCDF4``) -- so leaving them on makes this module pass
        alone and fail in a suite.
        """

        @veros_routine
        def set_diagnostics(self, state):
            state.diagnostics.clear()

    # Belt and braces: run from a scratch directory so anything Veros does
    # write (a restart, say) lands there rather than in the repository.
    workdir = tmp_path_factory.mktemp("veros_acc_basic")
    previous_directory = os.getcwd()
    os.chdir(workdir)
    try:
        model = CoupledACCBasic()
        model.setup()
        yield model
    finally:
        os.chdir(previous_directory)


@pytest.fixture(scope="module")
def grid_shape(veros_model):
    return (veros_model.state.dimensions["xt"],
            veros_model.state.dimensions["yt"])


@pytest.fixture
def component(veros_model) -> VerosComponent:
    wrapper = VerosComponent(veros_model)
    wrapper.bind(
        coupling_timestep=COUPLING_TIMESTEP,
        start_date=START_DATE,
        calendar=CALENDAR,
    )
    return wrapper


def _coupling_time(step: int) -> CouplingTime:
    """Build the clock the coupler hands a component on step ``step``."""
    return CouplingTime(
        step=jnp.int32(step),
        sim_time=jnp.float32(step * 86400.0),
        dt=86400.0,
        year_offset_seconds=0.0,
        days_per_year=365.0,
    )


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
