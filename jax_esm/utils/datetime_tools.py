import jax_datetime as jdt
from datetime import datetime, timedelta


def days_of_year_in_date(dt: jdt.Datetime):
    pydt = dt.to_pydatetime()
    next_year = datetime(pydt.year + 1, 1, 1)
    this_year = datetime(pydt.year, 1, 1)
    return int((next_year - this_year) / timedelta(days=1))
