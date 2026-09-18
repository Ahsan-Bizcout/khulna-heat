"""Earth Engine adapter for ERA5-Land daily aggregates.

Needs `earthengine-api` installed and authenticated:

    pip install earthengine-api
    earthengine authenticate

For a 44-year record over 59 polygons this is a large computation. Prefer
Export.table.toDrive (see tabular.ERA5_GEE_SNIPPET) for the full history and
use this adapter for shorter windows or spot checks.
"""

from __future__ import annotations

import pandas as pd

COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
BANDS = ["temperature_2m_max", "dewpoint_temperature_2m"]


def load_gee(
    geojson: dict,
    start: str,
    end: str,
    unit_property: str = "GID_3",
    season: tuple[int, int] | None = (3, 6),
    scale: int = 9000,
    chunk_days: int = 365,
) -> pd.DataFrame:
    """Zonal-mean daily Tmax and dewpoint per polygon.

    `geojson` is a FeatureCollection dict -- zones.to_geojson() gives you one.
    """
    try:
        import ee
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "earthengine-api is not installed. Either `pip install "
            "earthengine-api` and authenticate, or export the table from the "
            "Earth Engine Code Editor and use sources.load_tabular()."
        ) from exc

    try:
        ee.Initialize()
    except Exception:  # pragma: no cover
        ee.Authenticate()
        ee.Initialize()

    zones = ee.FeatureCollection(geojson)
    frames = []

    for chunk_start, chunk_end in _chunks(start, end, chunk_days):
        coll = (
            ee.ImageCollection(COLLECTION)
            .filterDate(chunk_start, chunk_end)
            .select(BANDS)
        )
        if season:
            coll = coll.filter(
                ee.Filter.calendarRange(season[0], season[1], "month")
            )

        def _reduce(img):
            stats = img.reduceRegions(
                collection=zones,
                reducer=ee.Reducer.mean(),
                scale=scale,
                tileScale=4,
            )
            date = img.date().format("YYYY-MM-dd")
            return stats.map(lambda f: f.set("date", date))

        info = coll.map(_reduce).flatten().getInfo()
        for feat in info.get("features", []):
            props = feat["properties"]
            if props.get("temperature_2m_max") is None:
                continue
            frames.append(
                {
                    "unit_id": props.get(unit_property),
                    "date": props["date"],
                    "tmax_c": props["temperature_2m_max"] - 273.15,
                    "dewpoint_c": (
                        props["dewpoint_temperature_2m"] - 273.15
                        if props.get("dewpoint_temperature_2m") is not None
                        else None
                    ),
                }
            )

    if not frames:
        raise RuntimeError("Earth Engine returned no rows. Check dates and asset.")

    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["date"])
    df.attrs["source"] = "gee:ERA5_LAND/DAILY_AGGR"
    return df.sort_values(["unit_id", "date"]).reset_index(drop=True)


def _chunks(start: str, end: str, days: int):
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur < stop:
        nxt = min(cur + pd.Timedelta(days=days), stop)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt
