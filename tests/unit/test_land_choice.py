"""Which land integrates the temperature: the atmosphere's own, or JEM's.

A coupled model has one land temperature and two places that can evolve it:
the atmosphere's own physics (SPEEDY's slab land) and a JEM land component
(:class:`~jem.components.slab.slab_land_model.SlabLandModel`). The choice is
made in two places -- the components dict and the atmosphere's physics -- and
:mod:`jem.exchangers` is where they are held together. These tests are about
that seam alone, so the atmosphere is a stand-in that only answers the two
questions the coupling asks of it; ``tests/unit/test_runners.py`` builds the
real pairing.
"""

import pytest

from jem.components.slab import BucketLandModel, SlabLandModel
from jem.exchangers import (
    SLAB_LAND_EXCHANGES,
    check_land_coupling,
    default_exchanges,
    land_temperature_source,
)
from tests.unit.slab_test_utils import make_grid

SOIL_WATER_ROWS = {
    ("atm.derived.precipitation", "lnd.forcing.precipitation"),
    ("atm.derived.land_evaporation", "lnd.forcing.evaporation"),
    ("lnd.derived.soil_water_availability", "atm.forcing.soilw_am"),
}
TEMPERATURE_ROWS = {
    ("atm.derived.land_total_heat_flux", "lnd.forcing.total_heat_flux"),
    ("lnd.state.land_surface_temperature", "atm.forcing.stl_am"),
    ("lnd.state.snow_cover", "atm.forcing.snowc_am"),
}


class _Atmosphere:
    """An atmosphere that only says what the land coupling asks of it."""

    name = "atm"

    def __init__(self, *, integrates_land_temperature, publishes_land_tile=True):
        self.integrates_land_temperature = integrates_land_temperature
        self.publishes_land_evaporation = publishes_land_tile


def _prescribed_atmosphere():
    return _Atmosphere(integrates_land_temperature=False)


def _atmosphere_with_its_own_land():
    return _Atmosphere(integrates_land_temperature=True)


def _land_rows(components):
    return {(spec.src, spec.dst): spec.regrid
            for spec in default_exchanges(components)
            if "lnd" in spec.src + spec.dst}


@pytest.fixture
def grid():
    return make_grid()


def test_the_slab_lands_rows_are_the_documented_ones():
    assert {row[:2] for row in SLAB_LAND_EXCHANGES} == TEMPERATURE_ROWS


def test_a_slab_land_takes_the_land_tiles_heat_flux_and_hands_back_its_temperature(grid):
    """The land that evolves its temperature gets the temperature rows on top
    of the soil-water rows every land gets, none of them regridded.
    """
    rows = _land_rows({"atm": _prescribed_atmosphere(), "lnd": SlabLandModel(grid)})
    assert set(rows) == SOIL_WATER_ROWS | TEMPERATURE_ROWS
    assert set(rows.values()) == {None}


def test_a_bucket_land_leaves_the_temperature_with_the_atmosphere(grid):
    """The bucket evolves soil water only: no row carries a heat flux or a
    temperature, whichever land the atmosphere runs.
    """
    for atmosphere in (_prescribed_atmosphere(), _atmosphere_with_its_own_land()):
        rows = _land_rows({"atm": atmosphere, "lnd": BucketLandModel(grid)})
        assert set(rows) == SOIL_WATER_ROWS


def test_names_alone_get_the_soil_water_rows():
    """A name says nothing about a carry, so it cannot select the temperature rows."""
    assert set(_land_rows(("atm", "lnd"))) == SOIL_WATER_ROWS


def test_two_lands_integrating_the_temperature_are_refused(grid):
    """An atmosphere with its own slab land adds its anomaly to the `stl_am`
    the component writes, so both slabs would be warmed by the same flux. The
    error names both ways out.
    """
    components = {"atm": _atmosphere_with_its_own_land(), "lnd": SlabLandModel(grid)}
    with pytest.raises(ValueError, match="Two lands would integrate") as raised:
        default_exchanges(components)
    message = str(raised.value)
    assert 'land_model="prescribed"' in message
    assert "land=slab" in message and "land=none" in message
    with pytest.raises(ValueError, match="Two lands would integrate"):
        check_land_coupling(components)


def test_a_slab_land_under_a_prescribed_atmosphere_land_is_accepted(grid):
    check_land_coupling({"atm": _prescribed_atmosphere(), "lnd": SlabLandModel(grid)})


def test_a_slab_land_under_an_atmosphere_with_no_land_tile_is_refused(grid):
    """Without a land tile the land's heat flux would be the grid-cell mean."""
    atmosphere = _Atmosphere(integrates_land_temperature=False, publishes_land_tile=False)
    with pytest.raises(ValueError, match="no land flux tile"):
        default_exchanges({"atm": atmosphere, "lnd": SlabLandModel(grid)})


def test_an_atmosphere_that_cannot_be_asked_is_not_refused(grid):
    """A stand-in with neither attribute is coupled, as it is to a bucket land."""
    rows = _land_rows({"atm": object(), "lnd": SlabLandModel(grid)})
    assert set(rows) == SOIL_WATER_ROWS | TEMPERATURE_ROWS


@pytest.mark.parametrize(
    ("atmosphere", "land", "phrase"),
    [
        (_atmosphere_with_its_own_land, None,
         "evolved by the atmosphere's own land"),
        (_atmosphere_with_its_own_land, BucketLandModel,
         "evolved by the atmosphere's own land"),
        (_prescribed_atmosphere, SlabLandModel,
         "evolved by the land component (SlabLandModel)"),
        (_prescribed_atmosphere, None, "prescribed"),
        (_prescribed_atmosphere, BucketLandModel, "prescribed"),
        (_atmosphere_with_its_own_land, SlabLandModel, "evolved twice"),
    ],
)
def test_the_land_temperatures_source_is_said_in_words(grid, atmosphere, land, phrase):
    components = {"atm": atmosphere()}
    if land is not None:
        components["lnd"] = land(grid)
    assert phrase in land_temperature_source(components)


def test_the_source_of_names_alone_is_unknown():
    assert "unknown" in land_temperature_source(("atm", "lnd"))
