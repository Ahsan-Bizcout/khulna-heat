"""Data adapters.

Every adapter returns the same long-format frame, so the formula layer never
knows or cares where the numbers came from:

    unit_id | date | tmax_c | dewpoint_c (or rh_pct)

Which one to use:

    tabular    Fastest route if you already export ERA5-Land zonal means from
               Earth Engine. Point it at that CSV and you are done.
    gee        Runs the ERA5-Land extraction for you, if earthengine-api is
               authenticated. ~9 km, 1981-present.
    openweather 30-day forward forecast only. Cannot do climate projection --
               see the README before reaching for this.
    synthetic  Plausible fake data so the dashboard runs before you have
               credentials or a shapefile. Never use it for results.
"""

from .synthetic import synthetic_series
from .tabular import load_tabular

__all__ = ["synthetic_series", "load_tabular", "load_gee", "load_openweather"]


def load_gee(*args, **kwargs):
    from .gee import load_gee as _f

    return _f(*args, **kwargs)


def load_openweather(*args, **kwargs):
    from .openweather import load_openweather as _f

    return _f(*args, **kwargs)
