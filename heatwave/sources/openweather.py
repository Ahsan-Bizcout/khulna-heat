"""OpenWeather Climatic Forecast 30 Days adapter.

Endpoint: https://pro.openweathermap.org/data/2.5/forecast/climate
Requires a paid plan. One call per unit centroid, so 59 upazilas = 59 calls.

SCOPE WARNING
-------------
This is a 30-day weather forecast. It is not, and cannot be turned into, a
climate projection. There is no emissions scenario behind it and forecast
skill decays over days, not decades. If you need 2030/2050/2100 heatwave
projections, use NEX-GDDP-CMIP6 (see README) -- and compute the percentile
threshold from that model's own historical run, never from ERA5.

Two data caveats worth carrying into any write-up:

  * `humidity` in the response is a daily aggregate, not the humidity at the
    hour of peak temperature. Pairing it with temp.max slightly overstates
    the heat index. This is the same approximation the thesis already makes
    with daily-mean dewpoint, so at least it is consistent.
  * temp.max here is a model forecast on a coarse grid, not an observation.
    It will not match a BMD station reading.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

ENDPOINT = "https://pro.openweathermap.org/data/2.5/forecast/climate"


def fetch_unit(
    lat: float,
    lon: float,
    api_key: str,
    days: int = 30,
    timeout: int = 20,
    session: requests.Session | None = None,
) -> list[dict]:
    """One unit's forecast. Returns the raw `list` array from the response."""
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "cnt": days,
    }
    http = session or requests
    resp = http.get(ENDPOINT, params=params, timeout=timeout)

    if resp.status_code == 401:
        raise RuntimeError(
            "OpenWeather rejected the key (401). The 30-day climatic forecast "
            "is not on the free tier -- check your plan covers it."
        )
    if resp.status_code == 429:
        raise RuntimeError("OpenWeather rate limit hit (429). Slow down or upgrade.")
    resp.raise_for_status()
    return resp.json().get("list", [])


def load_openweather(
    centroids: pd.DataFrame,
    api_key: str,
    days: int = 30,
    unit_col: str = "unit_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    pause: float = 0.35,
    progress=None,
) -> pd.DataFrame:
    """Fetch every unit and return the standard long frame.

    `centroids` needs columns [unit_id, lat, lon] -- zones.centroids() builds
    this from your shapefile.
    """
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    session = requests.Session()
    total = len(centroids)

    for i, rec in enumerate(centroids.itertuples(index=False), start=1):
        unit = getattr(rec, unit_col)
        try:
            entries = fetch_unit(
                getattr(rec, lat_col), getattr(rec, lon_col),
                api_key, days, session=session,
            )
        except Exception as exc:  # noqa: BLE001 - one bad unit must not kill the run
            failures.append((str(unit), str(exc)))
            continue

        for entry in entries:
            rows.append(
                {
                    "unit_id": unit,
                    "date": pd.to_datetime(entry["dt"], unit="s").normalize(),
                    "tmax_c": entry["temp"]["max"],
                    "tmin_c": entry["temp"].get("min"),
                    "rh_pct": entry.get("humidity"),
                    "pressure_hpa": entry.get("pressure"),
                }
            )

        if progress:
            progress(i / total, f"{unit} ({i}/{total})")
        time.sleep(pause)

    if not rows:
        raise RuntimeError(
            "No forecast data returned for any unit. First error was: "
            + (failures[0][1] if failures else "unknown")
        )

    out = pd.DataFrame(rows).sort_values(["unit_id", "date"]).reset_index(drop=True)
    out.attrs["source"] = "openweather-forecast30"
    out.attrs["failures"] = failures
    return out
