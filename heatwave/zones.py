"""Shapefile handling.

Run `python scripts/extract_khulna.py` once and this module will find the
resulting Khulna upazila file in data/boundaries/. Anything geopandas can read
works: .shp, .gpkg, .geojson. A GADM v4.1 level-3 file for all of Bangladesh
also works; it is filtered to Khulna Division on load.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BOUNDARY_DIR = Path("data/boundaries")
OUTLINE_FILE = BOUNDARY_DIR / "khulna_division_outline.geojson"

# GADM level-3 names, in preference order.
ID_CANDIDATES = ["GID_3", "unit_id", "ADM3_PCODE", "gid_3"]
NAME_CANDIDATES = ["NAME_3", "upazila", "ADM3_EN", "name", "NAME_2"]
DISTRICT_CANDIDATES = ["NAME_2", "district", "ADM2_EN"]
DIVISION_CANDIDATES = ["NAME_1", "division", "ADM1_EN"]

DIVISION_NAME = "Khulna"

# Jessore was officially renamed Jashore in 2018. GADM still carries the old
# spelling, so both are accepted and the display name is standardised.
DISTRICT_ALIASES = {"Jessore": "Jashore", "Jhenaidaha": "Jhenaidah",
                    "Chuadanga": "Chuadanga"}

KHULNA_DISTRICTS = [
    "Bagerhat", "Chuadanga", "Jashore", "Jhenaidah", "Khulna",
    "Kushtia", "Magura", "Meherpur", "Narail", "Satkhira",
]
_ACCEPTED_DISTRICTS = set(KHULNA_DISTRICTS) | set(DISTRICT_ALIASES)


def find_boundary_file(directory: str | Path = BOUNDARY_DIR) -> Path | None:
    """First usable polygon file in the boundary directory.

    The division outline is deliberately skipped: it is a single polygon for
    drawing a frame on the map and would otherwise be mistaken for the zones.
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    for pattern in ("*.gpkg", "*.shp", "*.geojson", "*.json"):
        hits = sorted(
            p for p in directory.glob(pattern) if "outline" not in p.name.lower()
        )
        if hits:
            # A user-supplied study area wins over the generic GADM cut-out.
            study = [p for p in hits if "study" in p.name.lower()]
            return (study or hits)[0]
    return None


def load_zones(
    path: str | Path | None = None,
    id_col: str | None = None,
    name_col: str | None = None,
    district_col: str | None = None,
    filter_division: bool = True,
) -> gpd.GeoDataFrame:
    """Load boundaries and standardise to [unit_id, unit_name, district, geometry]."""
    path = Path(path) if path else find_boundary_file()
    if path is None:
        raise FileNotFoundError(
            "No boundary file found in data/boundaries/. Run "
            "`python scripts/extract_khulna.py` to build one from GADM v4.1, "
            "or run the app in demo mode."
        )

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    id_col = id_col or _first_present(gdf, ID_CANDIDATES)
    name_col = name_col or _first_present(gdf, NAME_CANDIDATES)
    district_col = district_col or _first_present(gdf, DISTRICT_CANDIDATES)
    division_col = _first_present(gdf, DIVISION_CANDIDATES)

    if id_col is None:
        gdf = gdf.reset_index().rename(columns={"index": "unit_id"})
        gdf["unit_id"] = "U" + gdf["unit_id"].astype(str).str.zfill(3)
        id_col = "unit_id"

    out = gdf.rename(columns={id_col: "unit_id"})
    out["unit_name"] = out[name_col] if name_col else out["unit_id"]
    out["district"] = out[district_col] if district_col else "Unknown"
    out["district"] = out["district"].astype(str).str.strip().replace(DISTRICT_ALIASES)

    if filter_division:
        if division_col is not None:
            mask = out[division_col].astype(str).str.strip() == DIVISION_NAME
        elif district_col is not None:
            mask = out["district"].isin(_ACCEPTED_DISTRICTS)
        else:
            mask = pd.Series(True, index=out.index)
        if mask.any():
            out = out.loc[mask]

    out = out[["unit_id", "unit_name", "district", "geometry"]].copy()
    # GADM glues compound names together ("BagerhatSadar", "KhanJahanAli").
    out["unit_name"] = (
        out["unit_name"].astype(str).str.strip()
        .str.replace(r"([a-z])([A-Z])", r"\1 \2", regex=True)
    )

    # GADM calls Jashore's sadar upazila by its thana name.
    out.loc[(out["unit_name"] == "Kotwali") & (out["district"] == "Jashore"), "unit_name"] = "Jashore Sadar"

    # Duplicate upazila names are real in this division: there is a Kaliganj in
    # both Jhenaidah and Satkhira, and a Daulatpur in both Kushtia and Khulna.
    # Joining on name silently merges them. Always join on unit_id, and flag
    # the collisions so they are visible rather than silent.
    dupes = out["unit_name"].duplicated(keep=False)
    out["name_is_ambiguous"] = dupes
    out["label"] = np.where(
        dupes, out["unit_name"] + " (" + out["district"] + ")", out["unit_name"]
    )

    if out["unit_id"].duplicated().any():
        raise ValueError(
            "unit_id is not unique in the boundary file. Pick a different "
            "id_col -- joins will silently corrupt otherwise."
        )
    return out.sort_values(["district", "unit_name"]).reset_index(drop=True)


def study_area_boundary(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """Outer boundary of the loaded zones, for framing the choropleth.

    Deliberately derived from the upazilas themselves rather than a separate
    division polygon, so the frame always matches exactly what is analysed.
    """
    if zones is None or not len(zones):
        return None
    return gpd.GeoDataFrame(geometry=[zones.geometry.union_all()], crs="EPSG:4326")


def load_division_outline(zones=None, path: str | Path = OUTLINE_FILE):
    """Kept for the headless pipeline; the dashboard uses study_area_boundary()."""
    path = Path(path)
    if path.exists():
        g = gpd.read_file(path)
        return (g.set_crs("EPSG:4326") if g.crs is None else g).to_crs("EPSG:4326")[["geometry"]]
    return study_area_boundary(zones)


def district_boundaries(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One polygon per district, dissolved from the zones, with a label point."""
    d = zones[["district", "geometry"]].dissolve(by="district").reset_index()
    pts = d.geometry.representative_point()
    d["lat"] = pts.y
    d["lon"] = pts.x
    d["n_units"] = zones.groupby("district").size().reindex(d["district"]).to_numpy()
    return d


def centroids(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Representative point per polygon, for point-based APIs.

    Uses representative_point() rather than centroid so the coordinate is
    guaranteed to fall inside the polygon -- centroids of concave coastal
    upazilas can land in the water.
    """
    pts = zones.geometry.representative_point()
    return pd.DataFrame(
        {
            "unit_id": zones["unit_id"].to_numpy(),
            "label": zones["label"].to_numpy(),
            "lat": pts.y.to_numpy(),
            "lon": pts.x.to_numpy(),
        }
    )


def to_geojson(zones: gpd.GeoDataFrame) -> dict:
    import json

    return json.loads(zones[["unit_id", "label", "geometry"]].to_json())


def coastal_index(zones: gpd.GeoDataFrame) -> dict[str, float]:
    """0 (northernmost) to 1 (southernmost), a crude inland-coastal proxy.

    Only used to make demo data look like Khulna. Not an analytical variable.
    """
    lat = zones.geometry.representative_point().y.to_numpy()
    lo, hi = lat.min(), lat.max()
    scaled = (hi - lat) / (hi - lo) if hi > lo else np.zeros_like(lat)
    return dict(zip(zones["unit_id"], scaled))


def demo_zones(n_rows: int = 8, n_cols: int = 8) -> gpd.GeoDataFrame:
    """A synthetic polygon grid over Khulna Division's bounding box.

    Lets the dashboard render before a real shapefile exists. Shapes are
    rectangles, not upazilas.
    """
    from shapely.geometry import box

    lon0, lon1 = 88.55, 89.90
    lat0, lat1 = 21.60, 24.10
    dlon = (lon1 - lon0) / n_cols
    dlat = (lat1 - lat0) / n_rows

    records = []
    k = 0
    for r in range(n_rows):
        for c in range(n_cols):
            k += 1
            records.append(
                {
                    "unit_id": f"DEMO{k:03d}",
                    "unit_name": f"Zone {k:02d}",
                    "district": KHULNA_DISTRICTS[r % len(KHULNA_DISTRICTS)],
                    "geometry": box(
                        lon0 + c * dlon, lat1 - (r + 1) * dlat,
                        lon0 + (c + 1) * dlon, lat1 - r * dlat,
                    ),
                }
            )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["name_is_ambiguous"] = False
    gdf["label"] = gdf["unit_name"]
    gdf.attrs["synthetic"] = True
    return gdf


def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None
