"""Restricting the forcing to a window changes no value the model reads."""

import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
from jcm.date import DateData
from jcm.forcing import BY_DATE_INTERP, WRAP_YEAR, ForcingData, TimeSeries, make_time_series

from jem.components.jcm.forcing_window import restrict_forcing_to_window

SHAPE = (4, 3)


def daily_climatology():
    dates = np.arange("2001-01-01", "2002-01-01", dtype="datetime64[D]").astype("datetime64[s]")
    values = np.arange(365, dtype=float)[:, None, None] + np.zeros(SHAPE)
    return make_time_series(values, dates, align_mode=WRAP_YEAR)


def monthly_climatology():
    dates = np.arange("2001-01", "2002-01", dtype="datetime64[M]").astype("datetime64[s]")
    values = 100.0 + np.arange(12, dtype=float)[:, None, None] + np.zeros(SHAPE)
    return make_time_series(values, dates, align_mode=WRAP_YEAR)


def dated_series():
    dates = np.arange(np.datetime64("2022-12-01T00"), np.datetime64("2023-02-01T00"), np.timedelta64(6, "h")).astype("datetime64[s]")
    values = np.arange(dates.size, dtype=float)[:, None, None] * np.ones(SHAPE)
    return make_time_series(values, dates, align_mode=BY_DATE_INTERP)


@pytest.fixture
def forcing():
    base = ForcingData.zeros(SHAPE)
    return base.replace(stl_am=daily_climatology(), sice_am=monthly_climatology(), soilw_am=dated_series())


@pytest.mark.parametrize("start, end", [("2022-12-24", "2023-01-05"), ("2023-02-26", "2023-03-03")])
def test_every_step_in_the_window_reads_the_same_values(forcing, start, end):
    """Across a year boundary and across a month boundary, every hour matches."""
    if start.startswith("2023-02"):
        forcing = forcing.replace(soilw_am=jnp.zeros(SHAPE))  # dated series ends in January
    restricted = restrict_forcing_to_window(forcing, start, end)
    for hour in np.arange(np.datetime64(start, "h"), np.datetime64(end, "h") + 1, 5):
        date = DateData(jdt.to_datetime(hour.astype("datetime64[s]")), jnp.int32(0), 3600.0)
        full, cut = forcing.select(date), restricted.select(date)
        for name in ("stl_am", "sice_am", "soilw_am"):
            np.testing.assert_array_equal(np.asarray(getattr(cut, name)), np.asarray(getattr(full, name)),
                                          err_msg=f"{name} at {hour}")


def test_the_window_holds_only_the_records_it_needs(forcing):
    restricted = restrict_forcing_to_window(forcing, "2022-12-24", "2023-01-05")
    assert restricted.stl_am.values.shape[0] == 13  # one per midnight, both ends included
    assert restricted.sice_am.values.shape[0] == 13
    assert restricted.soilw_am.values.shape[0] == 12 * 4 + 1
    assert isinstance(restricted.sea_surface_temperature, type(forcing.sea_surface_temperature))


def test_a_reversed_window_is_refused(forcing):
    with pytest.raises(ValueError):
        restrict_forcing_to_window(forcing, "2023-01-05", "2022-12-24")


def test_a_table_not_indexed_by_date_is_kept_whole():
    """A WRAP_YEAR table of another length (MACv2-SP's weekly cycle) is not cut."""
    weekly = TimeSeries(
        values=jnp.zeros((52,) + SHAPE),
        times=jdt.to_datetime(np.arange(np.datetime64("2001-01-01"), np.datetime64("2001-12-31"), np.timedelta64(7, "D"))[:52].astype("datetime64[s]")),
        align_mode=jnp.asarray(WRAP_YEAR, dtype=jnp.int32),
    )
    forcing = ForcingData.zeros(SHAPE).replace(stl_am=weekly)
    assert restrict_forcing_to_window(forcing, "2022-12-24", "2023-01-05").stl_am is weekly
