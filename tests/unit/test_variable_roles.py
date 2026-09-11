"""The ``jem_role`` attribute on a packaged component's output variables.

A coupled run's datasets are meant to be read together, and two mechanisms say
which part of a component a variable came from: the ``forcing_`` name prefix,
which exists so ``xr.merge`` does not see two different variables under one
name, and this attribute, which says the same thing without a name to parse
(:func:`jem.base.component.role_attrs`). These tests hold the two to each
other -- in the slab models' output the prefixed variables are exactly the
ones with ``jem_role = "forcing"`` -- and check that the attribute survives
the merge, which is the whole point of having it.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from jem.base.component import (
    FORCING_VARIABLE_PREFIX,
    ROLE_ATTRIBUTE,
    ROLES,
    role_attrs,
)
from jem.components.slab import (
    SlabAtmosphereModel,
    SlabGrid,
    SlabLandModel,
    SlabOceanModel,
    SlabOceanParameters,
    SlabSeaiceModel,
)
from tests.unit.slab_test_utils import make_grid, run_steps, time_axis

N_STEPS = 3


def half_land_grid():
    """Return the 4x3 test grid with its two eastern columns land."""
    fraction = np.zeros((4, 3))
    fraction[2:, :] = 1.0
    return make_grid(fractional_mask=fraction)


def slab_models() -> dict:
    """Return one of every slab model on a shared half-land grid.

    The ocean is built for Q-flux forcing (with no file, so the Q-flux is
    zero) because that is the configuration whose output carries
    ``forcing_q_flux`` -- the one forcing variable that does not come from the
    carry's forcing section, and so the one most likely to be missed.
    """
    grid = half_land_grid()
    return {
        "atm": SlabAtmosphereModel(grid),
        "ocn": SlabOceanModel(
            grid, SlabOceanParameters(forcing_method="qflux"), name="ocn"
        ),
        "lnd": SlabLandModel(grid),
        "seaice": SlabSeaiceModel(grid, name="seaice"),
    }


def dataset_for(model) -> xr.Dataset:
    """Return ``N_STEPS`` steps of ``model``'s output as a labelled Dataset."""
    _, diagnostics = run_steps(model, model.initialize(), N_STEPS)
    return model.to_xarray(diagnostics, time_axis(N_STEPS))


@pytest.fixture(scope="module")
def datasets() -> dict:
    return {name: dataset_for(model) for name, model in slab_models().items()}


def test_every_slab_variable_carries_a_role(datasets):
    """No packaged slab variable may be left untagged."""
    for name, dataset in datasets.items():
        assert dataset.data_vars, name
        for variable_name, variable in dataset.data_vars.items():
            assert ROLE_ATTRIBUTE in variable.attrs, f"{name}.{variable_name}"
            assert variable.attrs[ROLE_ATTRIBUTE] in ROLES, f"{name}.{variable_name}"


def test_the_roles_are_the_ones_each_variable_actually_has(datasets):
    """Spot-check the role of every variable the slab models write.

    Written out rather than derived, so that moving a field from ``state`` to
    ``derived`` (or writing a received field as if the component had computed
    it) has to be acknowledged here.
    """
    expected = {
        "atm": {
            "forcing_total_heat_flux": "forcing",
            "internal_total_heat_flux": "derived",
            "mean_air_temperature": "state",
            "mean_zonal_wind_velocity": "state",
            "mean_meridional_wind_velocity": "state",
        },
        "ocn": {
            "sea_surface_temperature": "state",
            "mixed_layer_depth": "derived",
            "total_heat_flux": "derived",
            "ice_frazil_melt_energy": "derived",
            "forcing_q_flux": "forcing",
        },
        "lnd": {
            "land_surface_temperature": "state",
            "snowc": "state",
            "soilw": "state",
            "forcing_total_heat_flux": "forcing",
        },
        "seaice": {
            "ice_thickness": "state",
            "ice_surface_temperature": "state",
            "forcing_ice_frazil_melt_energy": "forcing",
            "ice_fraction": "derived",
        },
    }
    for name, roles in expected.items():
        dataset = datasets[name]
        assert set(dataset.data_vars) == set(roles), name
        for variable_name, role in roles.items():
            assert dataset[variable_name].attrs[ROLE_ATTRIBUTE] == role, (
                f"{name}.{variable_name}"
            )


def test_the_prefixed_variables_are_exactly_the_forcing_ones(datasets):
    """The name prefix and the attribute must not be able to disagree."""
    for name, dataset in datasets.items():
        prefixed = {
            str(variable)
            for variable in dataset.data_vars
            if str(variable).startswith(FORCING_VARIABLE_PREFIX)
        }
        tagged = set(map(str, dataset.filter_by_attrs(**{ROLE_ATTRIBUTE: "forcing"})))
        assert prefixed == tagged, name
        assert prefixed, name


def test_the_roles_survive_a_merge_of_all_four_slabs(datasets):
    """A merged coupled dataset is still queryable by role.

    ``xr.merge`` keeps a variable's attributes, so the role is what a reader
    of the merged file uses to tell the sea surface temperature the ocean
    computed from the copy another component was given.
    """
    merged = xr.merge(datasets.values(), join="exact", compat="no_conflicts")

    for name, dataset in datasets.items():
        for variable_name, variable in dataset.data_vars.items():
            assert merged[variable_name].attrs[ROLE_ATTRIBUTE] == (
                variable.attrs[ROLE_ATTRIBUTE]
            ), f"{name}.{variable_name}"

    forcing = set(map(str, merged.filter_by_attrs(**{ROLE_ATTRIBUTE: "forcing"})))
    assert forcing == {
        "forcing_total_heat_flux",
        "forcing_q_flux",
        "forcing_ice_frazil_melt_energy",
    }
    assert "sea_surface_temperature" in map(
        str, merged.filter_by_attrs(**{ROLE_ATTRIBUTE: "state"})
    )


def test_role_attrs_returns_a_fresh_dict_each_time():
    """Two variables must not end up sharing one attributes dict."""
    first, second = role_attrs("state"), role_attrs("state")
    assert first == second == {ROLE_ATTRIBUTE: "state"}
    assert first is not second


def test_role_attrs_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="Unknown variable role"):
        role_attrs("diagnostic")


def test_a_curvilinear_grid_keeps_the_role_beside_the_coordinates_attribute():
    """The curvilinear path rewrites every variable's attrs; the role must stay."""
    # Genuinely curvilinear: longitude varies down the latitude axis and
    # latitude along the longitude axis, so no 1-D axes can be recovered.
    longitude = np.deg2rad(
        np.array([0.0, 90.0, 180.0, 270.0])[:, None] + np.arange(3)[None, :]
    )
    latitude = np.deg2rad(
        np.array([-60.0, 0.0, 60.0])[None, :] + np.arange(4)[:, None]
    )
    curvilinear = SlabGrid(
        fractional_mask=jnp.zeros((4, 3)),
        latitude_radian=jnp.asarray(latitude),
        longitude_radian=jnp.asarray(longitude),
        threshold=0.5,
    )
    assert not curvilinear.is_separable
    model = SlabSeaiceModel(curvilinear, name="seaice")
    dataset = dataset_for(model)

    for variable in dataset.data_vars.values():
        assert variable.attrs["coordinates"] == "lat lon"
        assert variable.attrs[ROLE_ATTRIBUTE] in ROLES
