"""Heatwave detection and the six hazard indices.

This is the formula layer. Everything here follows the Perkins & Alexander
(2013) framework that the thesis methodology uses:

  1. per-unit percentile threshold from a fixed baseline period
  2. runs of >= N consecutive exceedance days are events
  3. six indices summarise the events

Two threshold modes are supported, because the choice materially changes what
the index measures:

  season_wide  - one 90th percentile across the whole warm season.
                 Detected events cluster in the hottest weeks (Apr-May) and
                 March/June effectively never qualify.
  day_of_year  - Perkins & Alexander's own approach: a 15-day moving window
                 centred on each calendar day. Detects heat that is anomalous
                 *for the time of year*.

The thesis as written uses season_wide. Run both and report the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .humidity import heat_index, relative_humidity


@dataclass
class HeatwaveConfig:
    """Every tunable knob in the detection step, in one place."""

    percentile: float = 0.90
    min_duration: int = 3          # consecutive days to qualify as an event
    baseline: tuple[int, int] = (1991, 2020)   # WMO climate normal
    season: tuple[int, int] = (3, 6)           # inclusive month range, Mar-Jun
    threshold_mode: str = "season_wide"        # or "day_of_year"
    window_days: int = 15                      # only for day_of_year mode
    intensity_over_event_years_only: bool = True
    unit_col: str = "unit_id"
    date_col: str = "date"
    tmax_col: str = "tmax_c"

    def describe(self) -> str:
        mode = (
            f"{self.window_days}-day moving window"
            if self.threshold_mode == "day_of_year"
            else "single season-wide percentile"
        )
        return (
            f"P{int(self.percentile * 100)} threshold ({mode}), "
            f"baseline {self.baseline[0]}-{self.baseline[1]}, "
            f"events = runs of >= {self.min_duration} days, "
            f"season = months {self.season[0]}-{self.season[1]}"
        )


def in_season(dates, season):
    months = pd.DatetimeIndex(dates).month
    return (months >= season[0]) & (months <= season[1])


def compute_thresholds(df: pd.DataFrame, cfg: HeatwaveConfig) -> pd.DataFrame:
    """Percentile threshold per unit (and per day-of-year, if that mode is on).

    Returns a frame with columns [unit_id, threshold_c] or
    [unit_id, doy, threshold_c].
    """
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    years = d[cfg.date_col].dt.year
    mask = (
        in_season(d[cfg.date_col], cfg.season)
        & (years >= cfg.baseline[0])
        & (years <= cfg.baseline[1])
    )
    base = d.loc[mask]

    if base.empty:
        raise ValueError(
            f"No data inside baseline {cfg.baseline} and season {cfg.season}. "
            "Check the date column and that your record covers the baseline."
        )

    n_years = base[cfg.date_col].dt.year.nunique()
    if n_years < 20:
        import warnings

        warnings.warn(
            f"Baseline has only {n_years} years. A 90th percentile from a short "
            "record is unstable; 30 years is the standard.",
            stacklevel=2,
        )

    if cfg.threshold_mode == "season_wide":
        out = (
            base.groupby(cfg.unit_col)[cfg.tmax_col]
            .quantile(cfg.percentile)
            .rename("threshold_c")
            .reset_index()
        )
        return out

    if cfg.threshold_mode != "day_of_year":
        raise ValueError(f"Unknown threshold_mode: {cfg.threshold_mode!r}")

    # Moving-window percentile per calendar day.
    half = cfg.window_days // 2
    base = base.assign(doy=base[cfg.date_col].dt.dayofyear)
    rows = []
    for unit, grp in base.groupby(cfg.unit_col):
        doys = np.sort(grp["doy"].unique())
        values = grp[["doy", cfg.tmax_col]].to_numpy()
        for doy in doys:
            # Circular window so the year boundary does not create a gap.
            offsets = np.arange(-half, half + 1)
            window = ((doy - 1 + offsets) % 366) + 1
            sel = values[np.isin(values[:, 0], window), 1]
            if sel.size:
                rows.append((unit, int(doy), float(np.quantile(sel, cfg.percentile))))
    return pd.DataFrame(rows, columns=[cfg.unit_col, "doy", "threshold_c"])


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True. Returns (start, end) index pairs, end exclusive."""
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def flag_exceedances(
    df: pd.DataFrame, thresholds: pd.DataFrame, cfg: HeatwaveConfig
) -> pd.DataFrame:
    """Attach threshold_c and a boolean `hot` column to every in-season day."""
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    d = d.loc[in_season(d[cfg.date_col], cfg.season)].copy()

    if "doy" in thresholds.columns:
        d["doy"] = d[cfg.date_col].dt.dayofyear
        d = d.merge(thresholds, on=[cfg.unit_col, "doy"], how="left")
    else:
        d = d.merge(thresholds, on=cfg.unit_col, how="left")

    d["hot"] = d[cfg.tmax_col] > d["threshold_c"]
    return d.sort_values([cfg.unit_col, cfg.date_col]).reset_index(drop=True)


def detect_events(flagged: pd.DataFrame, cfg: HeatwaveConfig) -> pd.DataFrame:
    """One row per heatwave event.

    Runs are broken at season boundaries, so a hot spell cannot be stitched
    across the gap between one June and the next March. This is the boundary
    bug worth checking in any implementation of this method.
    """
    records = []
    group_cols = [cfg.unit_col, flagged[cfg.date_col].dt.year.rename("year")]

    for (unit, year), grp in flagged.groupby(group_cols, sort=True):
        grp = grp.sort_values(cfg.date_col)
        dates = grp[cfg.date_col].to_numpy()
        hot = grp["hot"].to_numpy()

        # Split on calendar discontinuities so a missing day breaks the run.
        gaps = np.concatenate(
            ([0], (np.diff(dates).astype("timedelta64[D]").astype(int) != 1).cumsum())
        )
        for _, seg_idx in pd.Series(range(len(grp))).groupby(gaps):
            idx = seg_idx.to_numpy()
            for start, end in _runs(hot[idx]):
                length = end - start
                if length < cfg.min_duration:
                    continue
                sl = grp.iloc[idx[start:end]]
                records.append(
                    {
                        cfg.unit_col: unit,
                        "year": int(year),
                        "start": sl[cfg.date_col].iloc[0],
                        "end": sl[cfg.date_col].iloc[-1],
                        "duration": int(length),
                        "peak_tmax": float(sl[cfg.tmax_col].max()),
                        "mean_tmax": float(sl[cfg.tmax_col].mean()),
                        "mean_hi": (
                            float(sl["hi_c"].mean()) if "hi_c" in sl else np.nan
                        ),
                    }
                )

    cols = [cfg.unit_col, "year", "start", "end", "duration",
            "peak_tmax", "mean_tmax", "mean_hi"]
    return pd.DataFrame(records, columns=cols)


def annual_indices(
    events: pd.DataFrame, flagged: pd.DataFrame, cfg: HeatwaveConfig
) -> pd.DataFrame:
    """The six indices, per unit per year. Years with no event get zeros for
    the frequency family and NaN for the intensity family."""
    units = flagged[cfg.unit_col].unique()
    years = sorted(flagged[cfg.date_col].dt.year.unique())
    scaffold = pd.MultiIndex.from_product(
        [units, years], names=[cfg.unit_col, "year"]
    ).to_frame(index=False)

    if events.empty:
        out = scaffold.assign(HWN=0, HWF=0, HWD=0,
                              HWA=np.nan, HWM=np.nan, HI=np.nan)
        return out

    # Weight intensity by event length so HWM is the mean across heatwave
    # *days*, not the mean of per-event means.
    ev = events.assign(_wsum=events["mean_tmax"] * events["duration"])
    if "mean_hi" in events:
        ev["_hsum"] = events["mean_hi"] * events["duration"]

    agg = ev.groupby([cfg.unit_col, "year"]).agg(
        HWN=("duration", "size"),
        HWF=("duration", "sum"),
        HWD=("duration", "max"),
        HWA=("peak_tmax", "max"),
        _wsum=("_wsum", "sum"),
        _hsum=("_hsum", "sum") if "_hsum" in ev else ("duration", "sum"),
    ).reset_index()

    agg["HWM"] = agg["_wsum"] / agg["HWF"]
    agg["HI"] = agg["_hsum"] / agg["HWF"] if "_hsum" in ev else np.nan
    agg = agg.drop(columns=["_wsum", "_hsum"])

    out = scaffold.merge(agg, on=[cfg.unit_col, "year"], how="left")
    out[["HWN", "HWF", "HWD"]] = out[["HWN", "HWF", "HWD"]].fillna(0)
    return out


def climatology(annual: pd.DataFrame, cfg: HeatwaveConfig) -> pd.DataFrame:
    """Collapse the per-year table to one row per unit.

    Frequency indices average over every year in the record, including years
    with no event. Intensity indices average over event years only, so that a
    quiet year does not dilute the severity of events that did occur. That
    asymmetry is the thesis's own choice; flip it with
    intensity_over_event_years_only=False.
    """
    freq = annual.groupby(cfg.unit_col)[["HWN", "HWF", "HWD"]].mean()

    if cfg.intensity_over_event_years_only:
        source = annual.loc[annual["HWN"] > 0]
    else:
        source = annual
    inten = source.groupby(cfg.unit_col)[["HWA", "HWM", "HI"]].mean()

    n_years = annual.groupby(cfg.unit_col)["year"].nunique().rename("n_years")
    n_event_years = (
        annual.loc[annual["HWN"] > 0]
        .groupby(cfg.unit_col)["year"]
        .nunique()
        .rename("n_event_years")
    )

    return (
        freq.join(inten, how="left")
        .join(n_years, how="left")
        .join(n_event_years, how="left")
        .fillna({"n_event_years": 0})
        .reset_index()
    )


def add_heat_index(
    df: pd.DataFrame,
    dewpoint_col: str | None = "dewpoint_c",
    rh_col: str | None = None,
    tmax_col: str = "tmax_c",
) -> pd.DataFrame:
    """Add rh_pct and hi_c columns from whichever humidity variable you have."""
    d = df.copy()
    if rh_col and rh_col in d:
        d["rh_pct"] = d[rh_col]
    elif dewpoint_col and dewpoint_col in d:
        d["rh_pct"] = relative_humidity(d[tmax_col], d[dewpoint_col])
    else:
        d["rh_pct"] = np.nan
    hi, flag = heat_index(d[tmax_col], d["rh_pct"], return_flag=True)
    d["hi_c"] = hi
    d["hi_capped"] = flag
    n = int(flag.sum())
    if n:
        import warnings
        warnings.warn(
            f"{n} of {len(d)} rows ({100*n/len(d):.1f}%) fell outside the NOAA "
            "heat-index chart and were capped at 58.3 C. That usually means the "
            "humidity variable is wrong for the paired temperature, not that "
            "the heat is unprecedented -- check the dewpoint/RH column.",
            stacklevel=2,
        )
    return d


def forecast_window_summary(
    flagged: pd.DataFrame, events: pd.DataFrame, cfg: HeatwaveConfig
) -> pd.DataFrame:
    """Summarise a short forward window (e.g. the 30-day forecast).

    Deliberately NOT the same product as `climatology`. Over 30 days you
    cannot estimate an annual rate, so this reports what is actually
    observable in the window: how many exceedance days, whether a qualifying
    run occurs, and how hot it gets. Do not plot these numbers on the same
    axis as the 1981-2024 climatology.
    """
    base = flagged.groupby(cfg.unit_col).agg(
        window_days=(cfg.date_col, "size"),
        exceedance_days=("hot", "sum"),
        peak_tmax=(cfg.tmax_col, "max"),
        threshold_c=("threshold_c", "first"),
    )
    if "hi_c" in flagged:
        base["peak_hi"] = flagged.groupby(cfg.unit_col)["hi_c"].max()

    if events.empty:
        base["events"] = 0
        base["heatwave_days"] = 0
        base["longest_event"] = 0
        base["first_event_start"] = pd.NaT
        base["hottest_event_tmax"] = np.nan
    else:
        ev = events.groupby(cfg.unit_col).agg(
            events=("duration", "size"),
            heatwave_days=("duration", "sum"),
            longest_event=("duration", "max"),
            first_event_start=("start", "min"),
            hottest_event_tmax=("peak_tmax", "max"),
        )
        base = base.join(ev, how="left")
        base[["events", "heatwave_days", "longest_event"]] = base[
            ["events", "heatwave_days", "longest_event"]
        ].fillna(0).astype(int)

    base["exceedance_days"] = base["exceedance_days"].astype(int)
    return base.reset_index()
