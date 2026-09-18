#!/usr/bin/env python3
"""Headless pipeline: daily series in, HWRI table and GeoPackage out.

Use this when you want reproducible results for the thesis rather than an
interactive session. The dashboard is for exploring; this is for the record.

    python scripts/run_pipeline.py --daily data/era5_daily.csv \\
        --socio data/indicators_2022.csv --out outputs/

    python scripts/run_pipeline.py --demo --out outputs/

Writes:
    hwri.csv            one row per unit, all components and the final index
    hwri.gpkg           the same joined to geometry
    hazard_annual.csv   per unit per year, for trend analysis
    events.csv          one row per detected heatwave event
    weights.csv         indicator weights under the chosen scheme
    robustness.csv      Spearman rho between weighting schemes
    run_config.txt      exactly how this run was parameterised
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heatwave import indices, risk, zones  # noqa: E402
from heatwave.sources.synthetic import (  # noqa: E402
    synthetic_series,
    synthetic_socioeconomic,
)
from heatwave.sources.tabular import load_tabular  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily", help="CSV/parquet of daily Tmax per unit")
    ap.add_argument("--socio", help="CSV of socio-economic indicators per unit")
    ap.add_argument("--boundaries", help="Shapefile / GeoPackage / GeoJSON")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--demo", action="store_true",
                    help="Run on synthetic data (produces no valid results)")
    ap.add_argument("--percentile", type=float, default=0.90)
    ap.add_argument("--min-duration", type=int, default=3)
    ap.add_argument("--threshold-mode", default="season_wide",
                    choices=["season_wide", "day_of_year"])
    ap.add_argument("--baseline", default="1991,2020")
    ap.add_argument("--season", default="3,6")
    ap.add_argument("--weighting", default="equal",
                    choices=["equal", "pca", "entropy"])
    ap.add_argument("--floor", type=float, default=0.05,
                    help="Min-max lower bound. 0.0 reproduces the structural "
                         "zeros of plain [0,1] scaling.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    lo, hi = (int(x) for x in args.baseline.split(","))
    s0, s1 = (int(x) for x in args.season.split(","))
    cfg = indices.HeatwaveConfig(
        percentile=args.percentile, min_duration=args.min_duration,
        threshold_mode=args.threshold_mode, baseline=(lo, hi), season=(s0, s1),
    )

    # ---- boundaries
    if args.boundaries:
        gdf = zones.load_zones(args.boundaries)
    elif args.demo:
        gdf = zones.demo_zones(8, 8)
    else:
        found = zones.find_boundary_file()
        gdf = zones.load_zones(found) if found else zones.demo_zones(8, 8)
    units = gdf["unit_id"].tolist()
    print(f"[1/6] {len(units)} spatial units")

    # ---- daily climate series
    if args.daily:
        daily = load_tabular(args.daily)
    else:
        print("      no --daily given, generating synthetic series")
        daily = synthetic_series(units, lo - 10, 2024,
                                 coastal_index=zones.coastal_index(gdf))
    daily = indices.add_heat_index(daily)
    print(f"[2/6] {len(daily):,} daily records, "
          f"{daily.date.min():%Y-%m-%d} to {daily.date.max():%Y-%m-%d}")

    # ---- detection
    thresholds = indices.compute_thresholds(daily, cfg)
    flagged = indices.flag_exceedances(daily, thresholds, cfg)
    events = indices.detect_events(flagged, cfg)
    annual = indices.annual_indices(events, flagged, cfg)
    clim = indices.climatology(annual, cfg)
    print(f"[3/6] {len(events):,} events; mean HWF "
          f"{clim.HWF.mean():.2f} days/yr, mean HWD {clim.HWD.mean():.2f} days")

    # ---- socio-economic
    if args.socio:
        socio = pd.read_csv(args.socio)
        idcol = next((c for c in ("unit_id", "GID_3", "ADM3_PCODE")
                      if c in socio.columns), socio.columns[0])
        socio = socio.rename(columns={idcol: "unit_id"})
    else:
        print("      no --socio given, generating synthetic indicators")
        socio = synthetic_socioeconomic(units, zones.coastal_index(gdf))

    table = clim.merge(socio, on="unit_id", how="left")
    missing = table.drop(columns=["unit_id"]).isna().sum()
    if missing.any():
        print("      missing cells per indicator:",
              missing[missing > 0].to_dict())

    # ---- index
    rcfg = risk.RiskConfig(floor=args.floor, weighting=args.weighting)
    result, weights = risk.build_indices(table, cfg=rcfg)
    result = pd.concat([table[["unit_id"]], result], axis=1)
    result = result.merge(gdf[["unit_id", "label", "district"]],
                          on="unit_id", how="left")
    zeros = int((result["HWRI"] <= 1e-9).sum())
    print(f"[4/6] HWRI {result.HWRI.min():.3f}-{result.HWRI.max():.3f}, "
          f"{zeros} exact zeros")
    if zeros:
        print("      WARNING: exact zeros present. With multiplicative "
              "aggregation these come from the [0,1] scaling, not the data. "
              "Consider --floor 0.05.")

    # ---- robustness
    scores, rho = risk.compare_schemes(table, cfg=rcfg)
    print("[5/6] Spearman rho between schemes:")
    print(rho.to_string().replace("\n", "\n      "))

    # ---- write
    result.to_csv(outdir / "hwri.csv", index=False)
    annual.to_csv(outdir / "hazard_annual.csv", index=False)
    events.to_csv(outdir / "events.csv", index=False)
    pd.DataFrame(weights).to_csv(outdir / "weights.csv")
    rho.to_csv(outdir / "robustness.csv")
    gdf.merge(result, on="unit_id", how="left").to_file(
        outdir / "hwri.gpkg", driver="GPKG"
    )
    (outdir / "run_config.txt").write_text(
        f"{cfg.describe()}\nweighting: {args.weighting}\nfloor: {args.floor}\n"
        f"units: {len(units)}\ndaily source: {args.daily or 'synthetic'}\n"
        f"socio source: {args.socio or 'synthetic'}\n"
        f"synthetic: {args.demo or not args.daily}\n"
    )
    print(f"[6/6] wrote {len(list(outdir.iterdir()))} files to {outdir}/")

    if args.demo or not args.daily:
        print("\nNOTE: this run used synthetic data. Not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
