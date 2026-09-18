"""Plausible fake climate data, so the app runs before you have credentials.

Reproduces the qualitative behaviour of the real Khulna record: a warm-season
temperature cycle, an inland-to-coastal gradient, year-to-year variability and
a mild warming trend. It is NOT real data. Every screen that uses it says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_series(
    units: list[str],
    start_year: int = 1981,
    end_year: int = 2024,
    season: tuple[int, int] = (3, 6),
    coastal_index: dict[str, float] | None = None,
    warming_c_per_decade: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    """Daily Tmax and dewpoint for each unit across the warm season.

    coastal_index maps unit -> 0 (fully inland) .. 1 (fully coastal). Coastal
    units run cooler in Tmax but more humid, which is the real pattern and the
    reason the coast is not automatically low-risk.
    """
    rng = np.random.default_rng(seed)
    coastal_index = coastal_index or {}

    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    dates = dates[(dates.month >= season[0]) & (dates.month <= season[1])]
    doy = dates.dayofyear.to_numpy()
    year = dates.year.to_numpy()

    # Seasonal shape: peaks in late April.
    seasonal = 5.0 * np.exp(-((doy - 115) ** 2) / (2 * 38.0 ** 2))
    trend = warming_c_per_decade * (year - start_year) / 10.0

    frames = []
    for i, unit in enumerate(units):
        coast = float(coastal_index.get(unit, rng.uniform(0, 1)))
        base = 30.4 + 2.4 * (1.0 - coast) + rng.normal(0, 0.35)

        # Year-level anomaly gives realistic clustering of hot years.
        year_anom = rng.normal(0, 0.75, size=end_year - start_year + 1)
        anom = year_anom[year - start_year]

        # AR(1) weather noise so hot days clump into runs, which is what makes
        # the >=3-day rule meaningful at all.
        noise = np.zeros(len(dates))
        e = rng.normal(0, 1.35, size=len(dates))
        for t in range(1, len(dates)):
            noise[t] = 0.74 * noise[t - 1] + e[t]

        tmax = base + seasonal + trend + anom + noise
        dew = (
            19.8
            + 5.4 * coast
            + 0.28 * seasonal
            + rng.normal(0, 1.1, size=len(dates))
        )
        dew = np.minimum(dew, tmax - 1.0)

        frames.append(
            pd.DataFrame(
                {
                    "unit_id": unit,
                    "date": dates,
                    "tmax_c": np.round(tmax, 2),
                    "dewpoint_c": np.round(dew, 2),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    out.attrs["synthetic"] = True
    return out


def synthetic_socioeconomic(
    units: list[str], coastal_index: dict[str, float] | None = None, seed: int = 7
) -> pd.DataFrame:
    """Fake census-style indicators matching the thesis indicator names."""
    rng = np.random.default_rng(seed)
    coastal_index = coastal_index or {}
    rows = []
    for unit in units:
        coast = float(coastal_index.get(unit, rng.uniform(0, 1)))
        urban = rng.beta(2, 5)
        rows.append(
            {
                "unit_id": unit,
                "Population": int(rng.lognormal(12.2, 0.45)),
                "PopDensity": round(300 + 2200 * urban + rng.normal(0, 120), 1),
                "BuiltUp_pct": round(2 + 30 * urban + rng.normal(0, 1.5), 2),
                "Elevation": round(1.5 + 11 * (1 - coast) + rng.normal(0, 1.2), 2),
                "Elderly_pct": round(rng.normal(7.4, 1.1), 2),
                "Children_pct": round(rng.normal(9.6, 1.3), 2),
                "AgriWorkers_pct": round(58 - 34 * urban + rng.normal(0, 4), 2),
                "IndustryWorkers_pct": round(6 + 22 * urban + rng.normal(0, 3), 2),
                "Kutcha_pct": round(62 - 30 * urban + rng.normal(0, 6), 2),
                "Poverty_pct": round(30 - 12 * urban + rng.normal(0, 5), 2),
                "Literacy_pct": round(58 + 16 * urban + rng.normal(0, 5), 2),
                "PipedWater_pct": round(4 + 45 * urban + rng.normal(0, 6), 2),
                "NDVI": round(0.44 + 0.14 * coast - 0.2 * urban + rng.normal(0, 0.04), 3),
                "NDWI": round(-0.05 + 0.3 * coast + rng.normal(0, 0.05), 3),
                "Mobile_pct": round(62 + 14 * urban + rng.normal(0, 5), 2),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["synthetic"] = True
    return df
