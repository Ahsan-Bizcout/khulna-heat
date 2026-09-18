"""Import daily series you already exported (the fastest path).

If your Earth Engine script already writes a per-upazila daily table, you do
not need this project to talk to any API. Point it at the file.

Accepted columns, case-insensitive, with common aliases handled:

    unit id     unit_id | GID_3 | ADM3_PCODE | upazila | name
    date        date | system:time_start | time
    max temp    tmax_c | tmax | temperature_2m_max | maximum_2m_air_temperature
    humidity    dewpoint_c | dewpoint_2m | rh_pct | humidity

Kelvin is detected and converted automatically -- ERA5-Land comes in Kelvin
and forgetting that is the single most common import bug.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ALIASES = {
    "unit_id": ["unit_id", "gid_3", "adm3_pcode", "upazila", "upazila_name",
                "name_3", "name", "zone", "id"],
    "date": ["date", "system:time_start", "time", "datetime", "day"],
    "tmax_c": ["tmax_c", "tmax", "temperature_2m_max", "t2m_max",
               "maximum_2m_air_temperature", "tasmax", "temp_max"],
    "dewpoint_c": ["dewpoint_c", "dewpoint", "dewpoint_2m",
                   "dewpoint_temperature_2m", "td"],
    "rh_pct": ["rh_pct", "rh", "humidity", "relative_humidity", "hurs"],
}


def _resolve(columns) -> dict[str, str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    found = {}
    for canonical, options in ALIASES.items():
        for opt in options:
            if opt in lowered:
                found[canonical] = lowered[opt]
                break
    return found


def load_tabular(path: str | Path, unit_col: str | None = None) -> pd.DataFrame:
    """Read a CSV/parquet daily series into the standard long frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    mapping = _resolve(raw.columns)
    if unit_col:
        mapping["unit_id"] = unit_col

    missing = [k for k in ("unit_id", "date", "tmax_c") if k not in mapping]
    if missing:
        raise ValueError(
            f"Could not find column(s) for {missing} in {path.name}. "
            f"Columns present: {list(raw.columns)[:12]}. "
            "Rename them or extend ALIASES in sources/tabular.py."
        )

    df = raw.rename(columns={v: k for k, v in mapping.items()})
    keep = [c for c in ("unit_id", "date", "tmax_c", "dewpoint_c", "rh_pct")
            if c in df.columns]
    df = df[keep].copy()

    # Earth Engine exports system:time_start as epoch milliseconds.
    if pd.api.types.is_numeric_dtype(df["date"]):
        scale = "ms" if df["date"].max() > 1e11 else "s"
        df["date"] = pd.to_datetime(df["date"], unit=scale)
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "tmax_c"])
    df["date"] = df["date"].dt.normalize()

    for col in ("tmax_c", "dewpoint_c"):
        if col in df and df[col].median() > 100:
            df[col] = df[col] - 273.15  # Kelvin -> Celsius

    df = df.sort_values(["unit_id", "date"]).reset_index(drop=True)
    df.attrs["source"] = f"tabular:{path.name}"
    return df


ERA5_GEE_SNIPPET = '''
// Earth Engine (JavaScript API) -- ERA5-Land daily zonal means per upazila.
// Run this, export the CSV, then load it with sources.load_tabular().

var zones = ee.FeatureCollection('users/YOUR_ASSET/khulna_upazilas');
var era5  = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
              .filterDate('1981-01-01', '2025-01-01')
              .filter(ee.Filter.calendarRange(3, 6, 'month'))
              .select(['temperature_2m_max', 'dewpoint_temperature_2m']);

var rows = era5.map(function (img) {
  var stats = img.reduceRegions({
    collection: zones,
    reducer: ee.Reducer.mean(),
    scale: 9000,
    tileScale: 4
  });
  return stats.map(function (f) {
    return f.set('date', img.date().format('YYYY-MM-dd'));
  });
}).flatten();

Export.table.toDrive({
  collection: rows,
  description: 'khulna_era5land_daily',
  fileFormat: 'CSV',
  selectors: ['GID_3', 'date', 'temperature_2m_max', 'dewpoint_temperature_2m']
});
'''
