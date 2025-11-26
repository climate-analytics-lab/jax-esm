import jax_datetime as jdt
from datetime import datetime, timedelta


def days_of_year_in_date(dt: jdt.Datetime):
    pydt = dt.to_pydatetime()
    next_year = datetime(pydt.year + 1, 1, 1)
    this_year = datetime(pydt.year, 1, 1)
    return int((next_year - this_year) / timedelta(days=1))
 
def days_of_month_in_date(dt: jdt.Datetime):
    pydt = dt.to_pydatetime()
    year_of_next_month = pydt.year + pydt.month // 12
    next_month = datetime(year_of_next_month, pydt.month % 12 + 1, 1)
    this_month = datetime(pydt.year, pydt.month, 1)
    return int((next_month - this_month) / timedelta(days=1))
