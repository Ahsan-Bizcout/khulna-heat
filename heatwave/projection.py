"""Future heatwave hazard and HWRI from NEX-GDDP-CMIP6.

Method (matches the historical pipeline; nothing new is invented here):

  1. Daily tasmax (and hurs, if exported) for a future window, per upazila,
     per GCM, per SSP scenario -- from a Google Earth Engine export of
     NASA/GDDP-CMIP6 (see CMIP6_GEE_SNIPPET).
  2. The *historical* per-upazila percentile threshold is carried forward
     unchanged. Recomputing it on 2031-2040 would move the goalposts with the
     warming and hide part of the change.
  3. The same >= N consecutive-day rule, the same six indices, per GCM.
  4. Ensemble: median across GCMs per upazila, with the inter-model spread
     reported (p25-p75 and min-max). Indices are computed per model and
     summarised afterwards; model daily series are never averaged first,
     because different models put the same heatwave on different days.
  5. Future HWRI: the historical equations with the hazard block replaced
     by the ensemble-median future indices, and exposure / adaptive-capacity
     assumptions applied to the indicator table.

Normalisation caveat, stated once here and again in the UI: min-max rescaling
within a scenario makes HWRI a ranking, so two scenarios cannot be compared
on HWRI alone. Compare them on the absolute hazard change (delta HWF, delta
HWN) or scale against the historical indicator ranges (`reference`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import indices, risk

SCENARIOS = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
DEFAULT_PERIOD = (2031, 2040)

# NEX-GDDP-CMIP6 models in the Earth Engine catalogue that carry tasmax for
# both scenarios. A representative spread of climate sensitivities; swap in
# what your supervisor prefers.
DEFAULT_MODELS = [
    "ACCESS-CM2", "EC-Earth3", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR",
    "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-MM", "UKESM1-0-LL",
]

ALIASES = {
    "unit_id": ["unit_id", "gid_3", "adm3_pcode", "upazila", "name_3", "id"],
    "date": ["date", "system:time_start", "time", "day"],
    "tmax_c": ["tasmax", "tmax_c", "tmax", "tasmax_c"],
    "rh_pct": ["hurs", "rh_pct", "rh", "relative_humidity"],
    "model": ["model", "gcm", "source_id"],
    "scenario": ["scenario", "ssp", "experiment", "experiment_id"],
}


# ------------------------------------------------------------------ ingestion
def _resolve(columns) -> dict[str, str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    found = {}
    for canonical, options in ALIASES.items():
        for opt in options:
            if opt in lowered:
                found[canonical] = lowered[opt]
                break
    return found


def _infer_from_name(name: str) -> tuple[str | None, str | None]:
    """Pull scenario and model out of a file name like ssp585_EC-Earth3.csv."""
    low = name.lower()
    scen = next((s for s in SCENARIOS if s in low), None)
    model = next((m for m in DEFAULT_MODELS if m.lower() in low), None)
    return scen, model


def load_cmip6_tabular(paths) -> pd.DataFrame:
    """Read one or more Earth Engine exports into the long frame

        unit_id | date | tmax_c | rh_pct | model | scenario

    Kelvin is detected and converted. Scenario and model may be columns or
    encoded in the file name.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in map(Path, paths):
        raw = pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
        m = _resolve(raw.columns)
        missing = [k for k in ("unit_id", "date", "tmax_c") if k not in m]
        if missing:
            raise ValueError(f"{p.name}: missing column(s) for {missing}. "
                             f"Present: {list(raw.columns)[:12]}")
        df = raw.rename(columns={v: k for k, v in m.items()})
        scen_from_name, model_from_name = _infer_from_name(p.name)
        if "scenario" not in df:
            if scen_from_name is None:
                raise ValueError(f"{p.name}: no scenario column and none in the file name.")
            df["scenario"] = scen_from_name
        if "model" not in df:
            df["model"] = model_from_name or p.stem
        keep = [c for c in ("unit_id", "date", "tmax_c", "rh_pct", "model", "scenario") if c in df]
        frames.append(df[keep])
    out = pd.concat(frames, ignore_index=True)

    if pd.api.types.is_numeric_dtype(out["date"]):
        scale = "ms" if out["date"].max() > 1e11 else "s"
        out["date"] = pd.to_datetime(out["date"], unit=scale)
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "tmax_c"])
    out["date"] = out["date"].dt.normalize()
    if out["tmax_c"].median() > 100:
        out["tmax_c"] = out["tmax_c"] - 273.15
    out["scenario"] = out["scenario"].astype(str).str.lower().str.replace("-", "").str.replace(".", "")
    out["scenario"] = out["scenario"].str.extract(r"(ssp\d{3})", expand=False).fillna(out["scenario"])
    out.attrs["synthetic"] = False
    return out.sort_values(["scenario", "model", "unit_id", "date"]).reset_index(drop=True)


# ------------------------------------------------------------------ synthetic
def synthetic_cmip6(
    units: list[str],
    coastal_index: dict[str, float] | None = None,
    period: tuple[int, int] = DEFAULT_PERIOD,
    models: list[str] | None = None,
    warming: dict[str, float] | None = None,
    seed: int = 2035,
) -> pd.DataFrame:
    """Plausible stand-in so the projection view renders before the export exists.

    Built from the same generator as the historical demo series, shifted by a
    scenario-dependent warming relative to the 1991-2020 synthetic baseline,
    with a per-model offset and variance factor so the ensemble has a spread.
    Regional warming by the 2030s differs little between SSP2-4.5 and
    SSP5-8.5; the divergence comes later in the century. NOT data.
    """
    from .sources.synthetic import synthetic_series

    models = models or DEFAULT_MODELS[:6]
    warming = warming or {"ssp245": 1.35, "ssp585": 1.65}
    rng = np.random.default_rng(seed)
    frames = []
    for scen, dT in warming.items():
        for k, model in enumerate(models):
            s = synthetic_series(units, period[0], period[1], coastal_index=coastal_index,
                                 warming_c_per_decade=0.0, seed=seed + 17 * k)
            offset = dT + rng.normal(0, 0.28)          # model climate-sensitivity spread
            spread = rng.uniform(0.9, 1.15)            # model variance factor
            base = s.groupby("unit_id")["tmax_c"].transform("mean")
            s["tmax_c"] = base + (s["tmax_c"] - base) * spread + offset
            s["rh_pct"] = indices.relative_humidity(s["tmax_c"], s["dewpoint_c"])
            s["model"] = model
            s["scenario"] = scen
            frames.append(s[["unit_id", "date", "tmax_c", "rh_pct", "model", "scenario"]])
    out = pd.concat(frames, ignore_index=True)
    out.attrs["synthetic"] = True
    return out


# ------------------------------------------------------------------- hazard
def future_hazard(
    future_daily: pd.DataFrame,
    thresholds: pd.DataFrame,
    cfg: indices.HeatwaveConfig,
    period: tuple[int, int] = DEFAULT_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six indices per (scenario, model, unit), then the ensemble summary.

    Returns (per_model, ensemble). `ensemble` has one row per (scenario, unit)
    with <index>, <index>_p25, <index>_p75, <index>_min, <index>_max and
    n_models. Thresholds are the historical ones, unchanged.
    """
    d = future_daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    yrs = d["date"].dt.year
    d = d.loc[(yrs >= period[0]) & (yrs <= period[1])]
    if d.empty:
        raise ValueError(f"No future rows inside {period}. Check the export window.")

    has_rh = "rh_pct" in d.columns and d["rh_pct"].notna().any()
    if has_rh:
        d = indices.add_heat_index(d, dewpoint_col=None, rh_col="rh_pct")

    rows = []
    for (scen, model), grp in d.groupby(["scenario", "model"], sort=True):
        flagged = indices.flag_exceedances(grp, thresholds, cfg)
        events = indices.detect_events(flagged, cfg)
        annual = indices.annual_indices(events, flagged, cfg)
        clim = indices.climatology(annual, cfg)
        clim["scenario"] = scen
        clim["model"] = model
        rows.append(clim)
    per_model = pd.concat(rows, ignore_index=True)

    idx_cols = ["HWN", "HWF", "HWD", "HWA", "HWM", "HI"]
    g = per_model.groupby(["scenario", "unit_id"])
    ens = g[idx_cols].median()
    for q, suffix in ((0.25, "_p25"), (0.75, "_p75")):
        ens = ens.join(g[idx_cols].quantile(q).add_suffix(suffix))
    ens = ens.join(g[idx_cols].min().add_suffix("_min")).join(g[idx_cols].max().add_suffix("_max"))
    ens["n_models"] = g["model"].nunique()
    ens = ens.reset_index()
    if not has_rh:
        ens["HI"] = np.nan
    return per_model, ens


# --------------------------------------------------------------------- risk
def adjust_indicators(
    socio: pd.DataFrame,
    ac_improvement: float = 0.0,
    exposure_growth: float = 0.0,
    future_socio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the exposure / adaptive-capacity assumption to the indicator table.

    `future_socio` (one row per unit, any subset of indicator columns) takes
    precedence and is the defensible route: a real 2035 population table.
    The scalar assumptions are uniform across upazilas, which means they
    cancel out under within-scenario min-max scaling and only matter when
    scaling against historical reference ranges. The UI says so.
    """
    out = socio.copy()
    if future_socio is not None:
        fut = future_socio.set_index("unit_id")
        for col in fut.columns:
            if col in out.columns:
                out[col] = out["unit_id"].map(fut[col]).fillna(out[col])
    if exposure_growth:
        for col in ("Population", "PopDensity"):
            if col in out:
                out[col] = out[col] * (1.0 + exposure_growth)
    if ac_improvement:
        f = 1.0 + ac_improvement
        for col in ("Literacy_pct", "PipedWater_pct", "Mobile_pct"):
            if col in out:
                out[col] = np.clip(out[col] * f, 0, 100)
        for col in ("NDVI", "NDWI"):
            if col in out:
                out[col] = out[col] * f
        if "Poverty_pct" in out:
            out["Poverty_pct"] = np.clip(out["Poverty_pct"] / f, 0, 100)
    return out


def historical_reference(table: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Min/max of every indicator in the historical table."""
    ref = {}
    for name in risk.DEFAULT_POLARITY:
        if name in table.columns:
            col = pd.to_numeric(table[name], errors="coerce")
            if col.notna().any():
                ref[name] = (float(col.min()), float(col.max()))
    return ref


def historical_anchor(hist_table: pd.DataFrame, rcfg: risk.RiskConfig) -> dict:
    """Everything needed to score a future table on the historical yardstick.

    Indicator ranges, raw component / HWRI ranges, and the four historical
    HWRI quintile cut points. A future value beyond the historical maximum
    saturates at 1; that is a finding ("beyond anything in 1981-2024"), not a
    bug, and the UI says so.
    """
    res, _ = risk.build_indices(hist_table, cfg=rcfg)
    edges = np.quantile(res["HWRI"].dropna().to_numpy(), [0.2, 0.4, 0.6, 0.8]).tolist()
    return {
        "indicators": historical_reference(hist_table),
        "components": res.attrs["ranges"],
        "class_edges": edges,
    }


def future_hwri(
    ensemble: pd.DataFrame,
    scenario: str,
    socio: pd.DataFrame,
    historical_clim: pd.DataFrame,
    rcfg: risk.RiskConfig,
    anchor: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """HWRI for one scenario from the ensemble-median hazard and an indicator table.

    * `anchor=None` -- min-max within the future table (the thesis procedure
      applied to the future set). HWRI is a ranking inside the scenario and is
      NOT comparable with another scenario or with the historical run.
    * `anchor=historical_anchor(...)` -- scored on the historical ranges with
      fixed class cut points, so scenarios and the baseline are comparable.

    If the future export carried no humidity, HI is carried forward from the
    historical climatology per unit (static) rather than dropped, so the
    Hazard block keeps the same six indicators as the historical run.
    """
    haz = ensemble.loc[ensemble["scenario"] == scenario,
                       ["unit_id", "HWN", "HWF", "HWD", "HWA", "HWM", "HI"]].copy()
    hi_carried = bool(haz["HI"].isna().all())
    if hi_carried:
        hist_hi = historical_clim.set_index("unit_id")["HI"]
        haz["HI"] = haz["unit_id"].map(hist_hi)
    table = haz.merge(socio, on="unit_id", how="left")

    if anchor:
        result, weights = risk.build_indices(
            table, cfg=rcfg, reference=anchor["indicators"],
            component_reference=anchor["components"], class_edges=anchor["class_edges"],
        )
    else:
        result, weights = risk.build_indices(table, cfg=rcfg)
    result = pd.concat([table[["unit_id"]], result], axis=1)
    result = result.merge(haz, on="unit_id", how="left")
    result["HI_carried"] = hi_carried
    result["scenario"] = scenario
    return result, weights


def hazard_change(ensemble: pd.DataFrame, historical_clim: pd.DataFrame) -> pd.DataFrame:
    """Absolute change of each index versus the historical climatology, per scenario."""
    hist = historical_clim.set_index("unit_id")[["HWN", "HWF", "HWD", "HWA", "HWM", "HI"]]
    out = ensemble[["scenario", "unit_id", "n_models"]].copy()
    for col in hist.columns:
        h = out["unit_id"].map(hist[col])
        out[f"{col}_hist"] = h
        out[f"{col}_fut"] = ensemble[col].to_numpy()
        out[f"d{col}"] = ensemble[col].to_numpy() - h.to_numpy()
        if f"{col}_p25" in ensemble:
            out[f"d{col}_p25"] = ensemble[f"{col}_p25"].to_numpy() - h.to_numpy()
            out[f"d{col}_p75"] = ensemble[f"{col}_p75"].to_numpy() - h.to_numpy()
    return out


# ------------------------------------------------------------ Earth Engine
CMIP6_GEE_SNIPPET = r'''
// NEX-GDDP-CMIP6: daily tasmax (+ hurs) for 2031-2040, per upazila, per GCM,
// for SSP2-4.5 and SSP5-8.5. Paste into the Earth Engine Code Editor.
//
// 1. Upload data/boundaries/study_area_upazilas.gpkg (or the .shp) as an
//    asset and put its path below. The GID_3 property is the join key.
// 2. Run; one export task per scenario appears under Tasks. Each CSV has
//    columns GID_3, date, model, scenario, tasmax, hurs.
// 3. Drop the CSVs anywhere under data/ (not boundaries/ or cache/). The
//    dashboard lists them under "Projection exports".
//
// Do not average models before running this: the dashboard computes the six
// indices per model and summarises the ensemble afterwards.

var zones = ee.FeatureCollection('users/YOUR_ASSET/study_area_upazilas');

var START = '2031-01-01', END = '2041-01-01';   // 2031-2040 inclusive
var SEASON = [3, 6];                             // warm season, months
var MODELS = ['ACCESS-CM2', 'EC-Earth3', 'GFDL-ESM4', 'INM-CM5-0',
              'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0',
              'NorESM2-MM', 'UKESM1-0-LL'];
var SCENARIOS = ['ssp245', 'ssp585'];

// 0.25 deg cells (~27 km) are larger than most upazilas; a zonal mean at the
// native scale is the honest reduction. Expect spatially smooth hazard.
var SCALE = 27830;

function exportScenario(scenario) {
  var perModel = MODELS.map(function (model) {
    var coll = ee.ImageCollection('NASA/GDDP-CMIP6')
      .filter(ee.Filter.eq('model', model))
      .filter(ee.Filter.eq('scenario', scenario))
      .filterDate(START, END)
      .filter(ee.Filter.calendarRange(SEASON[0], SEASON[1], 'month'))
      .select(['tasmax', 'hurs']);

    return coll.map(function (img) {
      var stats = img.reduceRegions({
        collection: zones,
        reducer: ee.Reducer.mean(),
        scale: SCALE,
        tileScale: 4
      });
      var date = img.date().format('YYYY-MM-dd');
      return stats.map(function (f) {
        return f.set({date: date, model: model, scenario: scenario});
      });
    }).flatten();
  });

  var rows = ee.FeatureCollection(perModel).flatten();

  Export.table.toDrive({
    collection: rows,
    description: 'cmip6_' + scenario + '_2031_2040_khulna',
    fileNamePrefix: 'cmip6_' + scenario + '_2031_2040_khulna',
    fileFormat: 'CSV',
    selectors: ['GID_3', 'date', 'model', 'scenario', 'tasmax', 'hurs']
  });
}

SCENARIOS.forEach(exportScenario);

// Optional sanity check: one model, one day, on the map.
var sample = ee.ImageCollection('NASA/GDDP-CMIP6')
  .filter(ee.Filter.eq('model', 'EC-Earth3'))
  .filter(ee.Filter.eq('scenario', 'ssp585'))
  .filterDate('2035-04-15', '2035-04-16').first().select('tasmax');
Map.centerObject(zones, 8);
Map.addLayer(sample.subtract(273.15).clip(zones.geometry()),
             {min: 30, max: 42, palette: ['ffe5dd', 'e87652', '7a2302']}, 'tasmax degC');
Map.addLayer(zones.style({color: '14181D', fillColor: '00000000', width: 1}), {}, 'upazilas');
'''
