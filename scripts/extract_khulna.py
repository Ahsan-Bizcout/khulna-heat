"""Extract Khulna Division boundaries for the dashboard.

Outputs land in data/boundaries/:

  study_area_upazilas.gpkg        your own upazila shapefile, standardised
                                  (--study-area). Preferred by the app when
                                  present; the division polygon is not used.
  khulna_upazilas.gpkg            64 upazila polygons (GADM v4.1 level 3,
                                  filtered to NAME_1 == "Khulna"); fallback
  khulna_division_outline.geojson the division polygon, for reference only

Usage:

  python scripts/extract_khulna.py --study-area ~/Downloads/Studyarea/Upazilas.shp --drop-city-thanas
  python scripts/extract_khulna.py --study-area ~/Downloads/Studyarea/Upazilas.shp --district Khulna
  python scripts/extract_khulna.py                       # download GADM
  python scripts/extract_khulna.py --gadm path/to/gadm41_BGD_3.json
  python scripts/extract_khulna.py --divisions ~/Downloads/bd_shp/bd.shp

GADM is fetched once into data/cache/ and reused. The upazila file is what
zones.find_boundary_file() picks up; the outline is drawn on the map as a
frame and is never used for analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BGD_3.json"
CACHE = ROOT / "data" / "cache" / "gadm41_BGD_3.json"
OUT_DIR = ROOT / "data" / "boundaries"
UPAZILA_OUT = OUT_DIR / "khulna_upazilas.gpkg"
STUDY_AREA_OUT = OUT_DIR / "study_area_upazilas.gpkg"
OUTLINE_OUT = OUT_DIR / "khulna_division_outline.geojson"

KEEP = ["GID_3", "GID_2", "GID_1", "NAME_1", "NAME_2", "NAME_3",
        "TYPE_3", "CC_3", "geometry"]

# GADM level 3 lists the five Khulna City Corporation thanas (metropolitan
# police areas) alongside the upazilas. They are not upazilas, and dropping
# them takes the division from 64 features to the official 59.
KHULNA_CITY_THANAS = {
    "BGD.4.5.3_1": "Daulatpur (Khulna)",
    "BGD.4.5.6_1": "Khalishpur",
    "BGD.4.5.7_1": "Khan Jahan Ali",
    "BGD.4.5.8_1": "Khulna Sadar",
    "BGD.4.5.13_1": "Sonadanga",
}


def fetch_gadm(dest: Path) -> Path:
    if dest.exists():
        return dest
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GADM level 3 for Bangladesh -> {dest}")
    with requests.get(GADM_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
    return dest


def extract_upazilas(gadm_path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(gadm_path)
    if "NAME_1" not in g.columns:
        sys.exit("Expected a GADM level-3 file with a NAME_1 column.")
    k = g.loc[g["NAME_1"] == "Khulna", [c for c in KEEP if c in g.columns]]
    if k.empty:
        sys.exit("No features with NAME_1 == 'Khulna' in that file.")
    k = k.to_crs("EPSG:4326").reset_index(drop=True)
    if not k["GID_3"].is_unique:
        sys.exit("GID_3 is not unique; refusing to write an ambiguous file.")
    return k


def extract_outline(divisions_path: Path | None,
                    upazilas: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    if divisions_path and divisions_path.exists():
        d = gpd.read_file(divisions_path).to_crs("EPSG:4326")
        name_col = next((c for c in ("name", "NAME_1", "NAME", "division")
                         if c in d.columns), None)
        if name_col is not None:
            hit = d[d[name_col].astype(str).str.strip().str.lower() == "khulna"]
            if len(hit) == 1:
                out = hit[[name_col, "geometry"]].rename(columns={name_col: "name"})
                out["source"] = str(divisions_path.name)
                return out.reset_index(drop=True), f"division shapefile {divisions_path}"
        print(f"Could not find a single 'Khulna' feature in {divisions_path}; "
              "falling back to a dissolve of the upazilas.")
    merged = upazilas.dissolve(by="NAME_1").reset_index()[["NAME_1", "geometry"]]
    merged = merged.rename(columns={"NAME_1": "name"})
    merged["source"] = "dissolved from GADM level 3"
    return merged, "dissolve of the upazila polygons"


def import_study_area(src: Path, district: str | None = None,
                      drop_city_thanas: bool = False) -> gpd.GeoDataFrame:
    """Standardise a user-supplied upazila layer (GADM-style columns expected).

    `district` keeps only that NAME_2 (e.g. "Khulna" -> the 14 upazilas of
    Khulna District, not the 64 of the division).
    """
    g = gpd.read_file(src)
    if district:
        col = next((c for c in ("NAME_2", "district", "ADM2_EN") if c in g.columns), None)
        if col is None:
            sys.exit("No district column (NAME_2) to filter on.")
        mask = g[col].astype(str).str.strip().str.lower() == district.strip().lower()
        if not mask.any():
            sys.exit(f"No rows with {col} == {district!r}. Values: {sorted(g[col].unique())}")
        g = g.loc[mask]
    if drop_city_thanas:
        before = len(g)
        g = g.loc[~g["GID_3"].isin(KHULNA_CITY_THANAS)]
        print(f"Dropped {before - len(g)} Khulna City Corporation thanas: "
              + ", ".join(KHULNA_CITY_THANAS.values()))
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g.to_crs("EPSG:4326")
    keep = [c for c in KEEP if c in g.columns]
    if "GID_3" not in keep:
        sys.exit(f"{src.name} has no GID_3 column; the app joins on it.")
    g = g[keep].reset_index(drop=True)
    if not g["GID_3"].is_unique:
        sys.exit("GID_3 is not unique in the study-area file.")
    # QGIS exports carry survey-grade detail (~6 MB for 64 polygons). The map
    # re-sends the geometry on every rerun, so simplify to ~30 m, preserving
    # topology so shared borders stay shared.
    g["geometry"] = g.geometry.simplify(0.0003, preserve_topology=True)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study-area", type=Path, default=None,
                    help="Your upazila shapefile; becomes the zone layer the app uses.")
    ap.add_argument("--district", default=None,
                    help="With --study-area: keep only this district (NAME_2), e.g. Khulna.")
    ap.add_argument("--drop-city-thanas", action="store_true",
                    help="With --study-area: remove the 5 Khulna City Corporation thanas "
                         "so the division has its official 59 upazilas.")
    ap.add_argument("--gadm", type=Path, default=None,
                    help="GADM v4.1 level-3 file (json/shp/gpkg). Downloaded if omitted.")
    ap.add_argument("--divisions", type=Path, default=None,
                    help="Division-level shapefile to take the outline from.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.study_area:
        sa = import_study_area(args.study_area, args.district, args.drop_city_thanas)
        STUDY_AREA_OUT.unlink(missing_ok=True)
        sa.to_file(STUDY_AREA_OUT, driver="GPKG", layer="study_area_upazilas")
        print(f"Wrote study area: {len(sa)} upazilas across "
              f"{sa['NAME_2'].nunique() if 'NAME_2' in sa else '?'} districts -> {STUDY_AREA_OUT}")
        return

    if UPAZILA_OUT.exists() and not args.force:
        print(f"{UPAZILA_OUT} exists; use --force to rebuild.")
    else:
        src = args.gadm if args.gadm else fetch_gadm(CACHE)
        upz = extract_upazilas(src)
        upz.to_file(UPAZILA_OUT, driver="GPKG", layer="khulna_upazilas")
        print(f"Wrote {len(upz)} upazilas across "
              f"{upz['NAME_2'].nunique()} districts -> {UPAZILA_OUT}")

    upz = gpd.read_file(UPAZILA_OUT)
    outline, how = extract_outline(args.divisions, upz)
    # A frame line does not need survey-grade detail; ~50 m tolerance keeps
    # the GeoJSON small enough to ship to the browser on every rerun.
    outline["geometry"] = outline.geometry.simplify(0.0005, preserve_topology=True)
    outline.to_file(OUTLINE_OUT, driver="GeoJSON")
    print(f"Wrote division outline ({how}) -> {OUTLINE_OUT}")


if __name__ == "__main__":
    main()
