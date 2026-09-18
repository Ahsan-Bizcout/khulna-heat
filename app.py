"""Khulna Heatwave Risk Dashboard.

    streamlit run app.py

Boundaries are read from data/boundaries/ (run `python scripts/extract_khulna.py`
once to build them from GADM v4.1). Climate and socio-economic inputs are chosen
in the sidebar; anything not supplied falls back to clearly labelled synthetic
data so the interface always renders.
"""

from __future__ import annotations

import io
from pathlib import Path

import altair as alt
import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from heatwave import indices, projection, risk, zones
from heatwave.humidity import HEAT_INDEX_BANDS, heat_index_band, relative_humidity
from heatwave.sources.synthetic import synthetic_series, synthetic_socioeconomic
from heatwave.sources.tabular import ALIASES, load_tabular

st.set_page_config(
    page_title="Khulna heatwave risk",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ design tokens
# Chrome is neutral so the choropleth carries all the colour. Data colours come
# from one validated heat ramp (magnitude), a five-step ordinal cut of it (risk
# classes), and a blue/orange categorical pair (forecast series).
FONT = "'Source Sans Pro', 'Source Sans 3', system-ui, -apple-system, 'Segoe UI', sans-serif"
INK = "#14181D"
INK2 = "#525A63"
MUTED = "#8A9099"
HAIR = "#E4E6E9"
RULE = "#C9CDD2"
PLANE = "#F4F5F7"
MAP_BG = "#EEF1F4"
NA_FILL = "#D9DBDF"
DEEMPH = "#D3D6DB"

HEAT = ["#ffe5dd", "#fbc2b0", "#f49e84", "#e87652", "#cd4f24", "#a73301", "#7a2302"]
CLASS_RAMP = ["#f49e84", "#e4734f", "#ca4b20", "#a23102", "#762100"]
VIRIDIS = ["#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725"]
# Diverging: blue <-> red with a neutral grey midpoint, for change layers.
DIVERGING = ["#1c5cab", "#5598e7", "#b7d3f6", "#f0efec", "#fbc2b0", "#e87652", "#a73301"]
ACCENT = "#cd4f24"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, WARNING = "#0ca30c", "#fab219"

CLASS_COLOURS = dict(zip(risk.RISK_CLASSES, CLASS_RAMP))
BAND_COLOURS = {
    "None": "#E3E6EA", "Caution": CLASS_RAMP[1], "Extreme caution": CLASS_RAMP[2],
    "Danger": CLASS_RAMP[3], "Extreme danger": CLASS_RAMP[4],
}
COMPONENT_NAMES = {
    "H": "Hazard", "E": "Exposure", "S": "Sensitivity",
    "AC": "Adaptive capacity", "V": "Vulnerability", "HWRI": "Risk index",
}
LAYERS = {
    "Risk index": "HWRI", "Risk class": "risk_class", "Hazard": "H",
    "Exposure": "E", "Sensitivity": "S", "Adaptive capacity": "AC",
    "Vulnerability": "V",
}

st.markdown(
    f"""
    <style>
    .stApp {{ background: #FFFFFF; color: {INK}; }}
    .block-container {{ padding-top: 4.3rem; padding-bottom: 2.5rem; max-width: 1560px; }}
    [data-testid="stDecoration"] {{ display: none; }}
    header[data-testid="stHeader"] {{ background: #FFFFFF; }}
    #MainMenu, footer {{ visibility: hidden; }}

    section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {HAIR}; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}
    section[data-testid="stSidebar"] label p {{ font-size: 0.84rem; color: {INK}; }}
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{ font-weight: 600; font-size: 0.86rem; }}
    .brand {{ font-weight: 700; font-size: 1.02rem; letter-spacing: -0.01em; color: {INK}; line-height: 1.15; }}
    .brand span {{ display: block; font-weight: 500; font-size: 0.74rem; color: {MUTED}; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 3px; }}
    .eyebrow {{ font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: {MUTED}; font-weight: 600; margin: 1.1rem 0 0.25rem; }}

    .title {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.022em; color: {INK}; line-height: 1.15; margin: 0 0 0.35rem; }}
    .h2 {{ font-size: 1.12rem; font-weight: 650; color: {INK}; letter-spacing: -0.01em; margin: 0.3rem 0 0.6rem; }}
    .subhead {{ color: {INK2}; font-size: 0.9rem; line-height: 1.45; }}
    .ph {{ font-size: 0.82rem; font-weight: 650; color: {INK}; margin: 0.2rem 0 0.35rem; }}
    .ph small {{ font-weight: 400; color: {MUTED}; margin-left: 6px; }}
    .cap {{ color: {MUTED}; font-size: 0.76rem; line-height: 1.45; margin-top: 4px; }}

    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .chip {{ display: inline-flex; align-items: center; gap: 7px; font-size: 0.76rem; color: {INK2};
             background: {PLANE}; border: 1px solid {HAIR}; border-radius: 999px; padding: 4px 11px 4px 9px; white-space: nowrap; }}
    .chip i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }}
    .chip b {{ color: {INK}; font-weight: 600; }}

    .note {{ border-left: 3px solid {WARNING}; background: #FFF8E8; color: #5C4300; padding: 0.6rem 0.9rem;
             border-radius: 0 6px 6px 0; font-size: 0.84rem; line-height: 1.45; margin: 12px 0 4px; }}
    .note b {{ color: #3E2D00; }}

    .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; margin: 14px 0 8px; }}
    .tile {{ background: #FFFFFF; border: 1px solid {HAIR}; border-radius: 10px; padding: 13px 16px 12px; min-width: 0; }}
    .tile .tl {{ font-size: 0.76rem; color: {INK2}; margin-bottom: 7px; }}
    .tile .tv {{ font-size: 1.62rem; font-weight: 650; letter-spacing: -0.02em; color: {INK}; line-height: 1.05;
                 overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .tile .tv small {{ font-size: 0.95rem; font-weight: 500; color: {INK2}; margin-left: 3px; letter-spacing: 0; }}
    .tile .ts {{ font-size: 0.75rem; color: {MUTED}; margin-top: 7px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

    .legend {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-size: 0.76rem; color: {INK2}; margin-top: 8px; }}
    .legend .bar {{ height: 10px; width: 240px; border-radius: 5px; border: 1px solid rgba(0,0,0,0.06); }}
    .legend .sw {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend .sw i {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; border: 1px solid rgba(0,0,0,0.08); }}
    .legend .t {{ font-weight: 600; color: {INK}; margin-right: 2px; }}

    .kv {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; font-size: 0.8rem; margin: 6px 0 10px; }}
    .kv div {{ display: flex; justify-content: space-between; gap: 8px; border-bottom: 1px solid {HAIR}; padding: 4px 0; }}
    .kv span {{ color: {MUTED}; }}
    .kv b {{ color: {INK}; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }}
    .kv i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 6px; vertical-align: -1px; }}

    .st-key-viewnav {{ border-bottom: 1px solid {HAIR}; margin: 6px 0 14px; }}
    .st-key-viewnav [data-testid="stButtonGroup"] > div {{ gap: 4px; border: none; background: transparent; }}
    .st-key-viewnav [data-testid="stButtonGroup"] button {{
        background: transparent; border: none; border-radius: 0; padding: 8px 12px 10px; margin: 0;
        border-bottom: 2px solid transparent; color: {INK2}; }}
    .st-key-viewnav [data-testid="stButtonGroup"] button p {{ font-size: 0.92rem; font-weight: 500; }}
    .st-key-viewnav [data-testid="stButtonGroup"] button:hover {{ color: {INK}; background: transparent; }}
    .st-key-viewnav [data-testid="stBaseButton-segmented_controlActive"] {{
        border-bottom-color: {ACCENT}; color: {INK}; }}
    .st-key-viewnav [data-testid="stBaseButton-segmented_controlActive"] p {{ font-weight: 650; }}
    [data-testid="stButtonGroup"] button p {{ font-size: 0.82rem; }}
    div[data-testid="stRadio"] label p {{ font-size: 0.84rem; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {HAIR}; border-radius: 8px; }}
    iframe {{ border-radius: 10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

MAP_CSS = f"""
<style>
.leaflet-container {{ background: {MAP_BG}; font-family: {FONT}; }}
.leaflet-tooltip {{ font-family: {FONT}; font-size: 12px; color: {INK}; border: 1px solid {HAIR};
                    border-radius: 6px; box-shadow: 0 4px 14px rgba(20,24,29,0.12); padding: 8px 10px; }}
.leaflet-tooltip th {{ color: {MUTED}; font-weight: 500; text-align: left; padding-right: 10px; }}
.leaflet-tooltip td {{ font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }}
.dlabel {{ font: 600 10px/1 {FONT}; letter-spacing: 0.1em; text-transform: uppercase; color: {INK};
           text-align: center; white-space: nowrap; pointer-events: none; opacity: 0.85;
           text-shadow: 0 0 3px #fff, 0 0 3px #fff, 0 0 5px #fff; }}
.ulabel {{ font: 500 10.5px/1 {FONT}; color: {INK}; text-align: center; white-space: nowrap;
           pointer-events: none; text-shadow: 0 0 3px #fff, 0 0 3px #fff, 0 0 5px #fff; }}
.leaflet-control-scale-line {{ font-family: {FONT}; }}
</style>
"""


# ------------------------------------------------------------------ small helpers
def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def chart(c: alt.TopLevelMixin) -> None:
    st.altair_chart(c, use_container_width=True, theme=None)


def styled(c):
    """House style for every Altair chart: hairline grid, muted axes, one font."""
    return (
        c.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=HAIR, gridWidth=1, domainColor=RULE, tickColor=RULE,
            labelColor=INK2, titleColor=INK2, labelFont=FONT, titleFont=FONT,
            labelFontSize=11, titleFontSize=11, titleFontWeight=500, labelPadding=6,
        )
        .configure_legend(
            labelColor=INK2, titleColor=INK2, labelFont=FONT, titleFont=FONT,
            labelFontSize=11, titleFontSize=11, symbolStrokeWidth=2, symbolSize=120,
        )
        .configure_title(font=FONT, fontSize=12, fontWeight=600, color=INK, anchor="start")
    )


def tiles(items: list[dict]) -> None:
    cells = []
    for it in items:
        unit = f"<small>{it['unit']}</small>" if it.get("unit") else ""
        cells.append(
            f"<div class='tile'><div class='tl'>{it['label']}</div>"
            f"<div class='tv'>{it['value']}{unit}</div>"
            f"<div class='ts'>{it.get('sub', '')}</div></div>"
        )
    html("<div class='tiles'>" + "".join(cells) + "</div>")


def chip(label: str, value: str, real: bool) -> str:
    dot = GOOD if real else WARNING
    return f"<span class='chip'><i style='background:{dot}'></i>{label} <b>{value}</b></span>"


def legend_html(title: str, *, categorical=False, colours=None, ramp=None,
                vmin=0.0, vmax=1.0, order=None, fmt="{:.2f}") -> str:
    if categorical:
        keys = order or list(colours)
        sw = "".join(
            f"<span class='sw'><i style='background:{colours[k]}'></i>{k}</span>" for k in keys
        )
        return f"<div class='legend'><span class='t'>{title}</span>{sw}</div>"
    grad = ", ".join(ramp)
    return (
        f"<div class='legend'><span class='t'>{title}</span><span>{fmt.format(vmin)}</span>"
        f"<span class='bar' style='background:linear-gradient(90deg,{grad})'></span>"
        f"<span>{fmt.format(vmax)}</span>"
        f"<span class='sw'><i style='background:{NA_FILL}'></i>not assessed</span></div>"
    )


def fmt3(x) -> str:
    return "–" if pd.isna(x) else f"{x:.3f}"


def fmt1(x) -> str:
    return "–" if pd.isna(x) else f"{x:.1f}"


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def discover_inputs(root: str = "data") -> dict:
    """Find candidate climate (date column), CMIP6 projection (tasmax / scenario
    columns) and socio-economic (indicator columns) files under data/."""
    climate, socio, proj = [], [], []
    date_names = set(ALIASES["date"])
    proj_names = {"tasmax", "scenario", "ssp", "experiment", "experiment_id"}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in (".csv", ".parquet"):
            continue
        if any(part in ("boundaries", "cache") for part in p.parts):
            continue
        try:
            if p.suffix.lower() == ".csv":
                cols = list(pd.read_csv(p, nrows=0).columns)
            else:
                cols = list(pd.read_parquet(p).columns)
        except Exception:  # noqa: BLE001 - unreadable files are simply not offered
            continue
        low = {str(c).strip().lower() for c in cols}
        if (low & proj_names) or any(k in p.name.lower() for k in ("ssp245", "ssp585", "cmip6")):
            proj.append(str(p))
        elif low & date_names:
            climate.append(str(p))
        elif any(c in cols for c in risk.DEFAULT_POLARITY):
            socio.append(str(p))
    return {"climate": climate, "socio": socio, "projection": proj}


@st.cache_data(show_spinner=False)
def get_zones(path: str | None, demo: bool):
    if demo or path is None:
        return zones.demo_zones(8, 8)
    return zones.load_zones(path)


@st.cache_data(show_spinner=False)
def get_map_context(path: str | None, demo: bool) -> dict:
    """Division outline, district polygons and label points for the map frame."""
    gdf = get_zones(path, demo)
    districts = zones.district_boundaries(gdf)
    outline = zones.study_area_boundary(gdf)
    pts = gdf.geometry.representative_point()
    area = gdf.geometry.to_crs("EPSG:3857").area.to_numpy()
    keep = area >= 0.2 * np.median(area)
    return {
        "districts": districts,
        "districts_json": districts[["district", "geometry"]].to_json(),
        "outline": outline,
        "outline_json": None if outline is None else outline.to_json(),
        # Upazila labels for a small study area, skipping the tiniest polygons
        # (city wards) whose labels would only collide with their neighbours.
        "labels": (list(zip(*[c[keep] for c in (gdf["label"].to_numpy(), pts.y.to_numpy(), pts.x.to_numpy())]))
                   if len(gdf) <= 20 else
                   list(zip(districts["district"], districts["lat"], districts["lon"]))),
        "label_kind": "upazila" if len(gdf) <= 20 else "district",
    }


@st.cache_data(show_spinner="Building the daily series...")
def get_daily(path: str | None, units: tuple, coastal: tuple, start: int, end: int):
    if path is None:
        return synthetic_series(list(units), start, end, coastal_index=dict(coastal))
    return load_tabular(path)


@st.cache_data(show_spinner=False)
def get_socio(path: str | None, units: tuple, coastal: tuple) -> pd.DataFrame:
    if path is None:
        return synthetic_socioeconomic(list(units), dict(coastal))
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    id_col = next(
        (c for c in df.columns if str(c).strip().lower() in ALIASES["unit_id"]), None
    )
    if id_col is None:
        raise ValueError(f"{p.name} has no unit id column (expected one of {ALIASES['unit_id']}).")
    df = df.rename(columns={id_col: "unit_id"})
    keep = ["unit_id"] + [c for c in risk.DEFAULT_POLARITY if c in df.columns]
    return df[keep].copy()


@st.cache_data(show_spinner="Detecting heatwaves...")
def get_climatology(daily: pd.DataFrame, percentile: float, min_dur: int,
                    mode: str, base_lo: int, base_hi: int):
    cfg = indices.HeatwaveConfig(
        percentile=percentile, min_duration=min_dur,
        threshold_mode=mode, baseline=(base_lo, base_hi),
    )
    d = indices.add_heat_index(daily)
    th = indices.compute_thresholds(d, cfg)
    flagged = indices.flag_exceedances(d, th, cfg)
    events = indices.detect_events(flagged, cfg)
    annual = indices.annual_indices(events, flagged, cfg)
    return indices.climatology(annual, cfg), th, annual, events, cfg


@st.cache_data(show_spinner="Loading CMIP6 exports...")
def get_future_daily(paths: tuple, units: tuple, coastal: tuple, period: tuple):
    if not paths:
        return projection.synthetic_cmip6(list(units), dict(coastal), period=period)
    return projection.load_cmip6_tabular(list(paths))


@st.cache_data(show_spinner="Detecting future heatwaves, model by model...")
def get_future_hazard(future_daily: pd.DataFrame, thresholds: pd.DataFrame, percentile: float,
                      min_dur: int, mode: str, period: tuple):
    cfg = indices.HeatwaveConfig(percentile=percentile, min_duration=min_dur, threshold_mode=mode)
    return projection.future_hazard(future_daily, thresholds, cfg, period)


# ------------------------------------------------------------------------ sidebar
inputs = discover_inputs()
boundary_path = zones.find_boundary_file()

with st.sidebar:
    html("<div class='brand'>Khulna heatwave risk<span>Upazila risk index explorer</span></div>")

    html("<div class='eyebrow'>Data</div>")
    if boundary_path is None:
        st.caption("No boundary file found. Run `python scripts/extract_khulna.py` "
                   "to build one from GADM v4.1. Using a synthetic grid meanwhile.")
        demo_geometry = True
    else:
        demo_geometry = st.toggle(
            "Use synthetic grid instead of upazilas", value=False,
            help=f"Boundaries found: {boundary_path.name}. Turn this on to see the "
                 "interface on a rectangular grid instead.",
        )
    climate_options = ["Synthetic (demo)"] + inputs["climate"]
    climate_choice = st.selectbox(
        "Climate series", climate_options,
        format_func=lambda s: s if s.startswith("Synthetic") else Path(s).name,
        help="Drop an ERA5-Land export (CSV or parquet with unit id, date, Tmax "
             "and dewpoint or RH) anywhere under data/ and it appears here.",
    )
    socio_options = ["Synthetic (demo)"] + inputs["socio"]
    socio_choice = st.selectbox(
        "Socio-economic indicators", socio_options,
        format_func=lambda s: s if s.startswith("Synthetic") else Path(s).name,
        help="One row per unit with the indicator names in heatwave/risk.py.",
    )
    climate_path = None if climate_choice.startswith("Synthetic") else climate_choice
    socio_path = None if socio_choice.startswith("Synthetic") else socio_choice
    if inputs["projection"]:
        proj_choice = st.multiselect(
            "Projection exports (CMIP6)", inputs["projection"], default=inputs["projection"],
            format_func=lambda s: Path(s).name,
            help="Earth Engine exports of NASA/GDDP-CMIP6 tasmax per upazila, model and "
                 "scenario. Deselect all for the synthetic stand-in.",
        )
    else:
        proj_choice = []
        st.caption("No CMIP6 exports under data/ yet; the 2035 view uses a synthetic "
                   "stand-in. The Earth Engine script is at the bottom of that view.")
    proj_paths = tuple(proj_choice)

    html("<div class='eyebrow'>Map</div>")
    ramp_choice = st.radio("Colour ramp", ["Heat", "Viridis (thesis)"], horizontal=True,
                           help="Heat is a single-hue ramp, light to dark. Viridis "
                                "matches the thesis figures.")
    basemap_choice = st.selectbox("Basemap", ["None (plain)", "OpenStreetMap"], index=0)
    show_districts = st.toggle("District borders", value=True)
    show_labels = st.toggle("Place labels", value=True,
                            help="Upazila names for a small study area, district names otherwise.")
    opacity = st.slider("Fill opacity", 0.3, 1.0, 0.9, 0.02)

    with st.expander("Heatwave definition", expanded=True):
        percentile = st.slider("Percentile threshold", 0.80, 0.99, 0.90, 0.01)
        min_duration = st.slider("Minimum consecutive days", 2, 6, 3)
        threshold_mode = st.selectbox(
            "Threshold basis", ["season_wide", "day_of_year"],
            help="season_wide is what the thesis uses. day_of_year follows Perkins & "
                 "Alexander with a 15-day moving window and detects heat that is "
                 "anomalous for the time of year.",
        )
        base_lo, base_hi = st.select_slider(
            "Baseline period", options=list(range(1981, 2025)), value=(1991, 2020)
        )

    with st.expander("Index construction", expanded=False):
        weighting = st.selectbox("Weighting", ["equal", "pca", "entropy"], index=0)
        floor = st.slider(
            "Normalization floor", 0.0, 0.20, 0.05, 0.01,
            help="Min-max to [floor, 1] instead of [0, 1]. At 0.00 the lowest unit "
                 "scores exactly zero on every component, which forces HWRI to zero "
                 "through the multiplicative aggregation. That is a scaling artefact, "
                 "not a finding.",
        )

    with st.expander("Renderer", expanded=False):
        RENDERER = st.radio(
            "Map renderer", ["Interactive (Leaflet)", "Static (matplotlib)"], index=0,
            help="Static needs no CDN or tile server; use it on a restricted network "
                 "or to preview the figure you will export.",
        )

BASEMAP = None if basemap_choice.startswith("None") else "OpenStreetMap"
RAMP = HEAT if ramp_choice == "Heat" else VIRIDIS

# --------------------------------------------------------------------------- data
bpath = None if demo_geometry else str(boundary_path)
gdf = get_zones(bpath, demo_geometry)
ctx = get_map_context(bpath, demo_geometry)
unit_ids = tuple(gdf["unit_id"])
coastal = tuple(zones.coastal_index(gdf).items())

daily = get_daily(climate_path, unit_ids, coastal, 1981, 2024)
if climate_path is not None:
    matched = daily["unit_id"].isin(unit_ids).mean()
    if matched < 0.5:
        st.warning(
            f"Only {matched:.0%} of rows in {Path(climate_path).name} match a unit id "
            "in the boundary file. Check that the series is keyed on GID_3."
        )

clim, thresholds, annual, events, cfg = get_climatology(
    daily, percentile, min_duration, threshold_mode, base_lo, base_hi
)
socio = get_socio(socio_path, unit_ids, coastal)
table = clim.merge(socio, on="unit_id", how="left")

rcfg = risk.RiskConfig(floor=floor, weighting=weighting)
result, weights = risk.build_indices(table, cfg=rcfg)
result = pd.concat([table[["unit_id"]], result], axis=1)
result = result.merge(gdf[["unit_id", "label", "district"]], on="unit_id", how="left")
result = result.merge(clim, on="unit_id", how="left")
result["rank"] = result["HWRI"].rank(ascending=False, method="min").astype(int)
result = result.sort_values("rank").reset_index(drop=True)

n_units = len(result)
n_districts = result["district"].nunique()
single_district = n_districts == 1
REGION = (f"{result['district'].iloc[0]} District" if single_district and boundary_path is not None and not demo_geometry
          else "Khulna Division")
SCOPE = "district" if single_district else "division"   # "within the district"
top = result.iloc[0]

# ------------------------------------------------------------------------- header
hl, hr = st.columns([1.55, 1], vertical_alignment="center")
with hl:
    html(f"<div class='title'>Upazila heatwave risk, {REGION}</div>")
    html(f"<div class='subhead'>{cfg.describe()} &nbsp;·&nbsp; {weighting} weighting "
         f"&nbsp;·&nbsp; floor {floor:.2f}</div>")
with hr:
    boundaries_real = not demo_geometry
    boundary_source = ("study area" if boundary_path and "study" in boundary_path.name.lower()
                       else "GADM v4.1")
    html("<div class='chips'>"
         + chip("Boundaries", f"{boundary_source} · {n_units} upazilas" if boundaries_real else "synthetic grid", boundaries_real)
         + chip("Climate", Path(climate_path).name if climate_path else "synthetic", climate_path is not None)
         + chip("Socio-economic", Path(socio_path).name if socio_path else "synthetic", socio_path is not None)
         + chip("CMIP6", f"{len(proj_paths)} export(s)" if proj_paths else "synthetic", bool(proj_paths))
         + "</div>")

synthetic_bits = []
if not boundaries_real:
    synthetic_bits.append("the polygons are a rectangular grid, not upazilas")
if climate_path is None:
    synthetic_bits.append("temperatures are generated, not observed")
if socio_path is None:
    synthetic_bits.append("census-style indicators are generated")
if synthetic_bits:
    html("<div class='note'><b>Partly synthetic.</b> "
         + "; ".join(synthetic_bits).capitalize()
         + ". Scores are illustrative and not a result. Add an ERA5-Land export and "
           "an indicator table under data/ to use the dashboard for real.</div>")

hwf_mean = result["HWF"].mean()
hwa_row = result.loc[result["HWA"].idxmax()] if result["HWA"].notna().any() else None
hi_mean = result["HI"].mean()
hi_band = heat_index_band(np.array([hi_mean]))[0] if pd.notna(hi_mean) else "–"
tiles([
    {"label": "Upazilas assessed", "value": f"{n_units}",
     "sub": f"{n_districts} district{'s' if n_districts != 1 else ''} · {(boundary_source + ' upazilas') if boundaries_real else 'synthetic grid'}"},
    {"label": "Highest risk", "value": top["label"],
     "sub": f"HWRI {top['HWRI']:.3f} · {top['district']} · {top['risk_class']}"},
    {"label": "Heatwave days per year", "value": f"{hwf_mean:.1f}", "unit": "days",
     "sub": f"{SCOPE} mean · range {result['HWF'].min():.1f}–{result['HWF'].max():.1f}"},
    {"label": "Hottest event peak", "value": f"{hwa_row['HWA']:.1f}" if hwa_row is not None else "–", "unit": "°C",
     "sub": f"{hwa_row['label']} · event-year mean" if hwa_row is not None else ""},
    {"label": "Heat index on heatwave days", "value": f"{hi_mean:.1f}" if pd.notna(hi_mean) else "–", "unit": "°C",
     "sub": f"{SCOPE} mean · {hi_band} band"},
])

VIEWS = ["Risk map", "Upazilas & components" if single_district else "Districts & components",
         "30-day outlook", "2035 projection", "Diagnostics", "Export"]
with st.container(key="viewnav"):
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], label_visibility="collapsed") or VIEWS[0]


# ---------------------------------------------------------------------- map layer
def build_map(gdf, values, unit_ids, layer_name, *, categorical=False, colours=None,
              vmin=0.0, vmax=1.0, props: pd.DataFrame | None = None, extra_fields=(),
              ramp=None, focus_id=None, height_hint=640):
    ramp = ramp or RAMP
    merged = gdf[["unit_id", "label", "district", "geometry"]].merge(
        pd.DataFrame({"unit_id": list(unit_ids), "_v": np.asarray(values)}),
        on="unit_id", how="left",
    )
    if props is not None:
        merged = merged.merge(props, on="unit_id", how="left")
    if categorical:
        merged["_vs"] = merged["_v"].fillna("not assessed").astype(str)
    else:
        merged["_vs"] = pd.to_numeric(merged["_v"], errors="coerce").map(fmt3)

    b = merged.total_bounds
    m = folium.Map(
        location=[(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], zoom_start=8,
        tiles=BASEMAP, control_scale=True, zoom_control=True,
        zoomSnap=0.25, zoomDelta=0.5,
    )
    m.get_root().header.add_child(folium.Element(MAP_CSS))

    if categorical:
        colours = colours or CLASS_COLOURS

        def style(feat):
            v = feat["properties"]["_v"]
            missing = v is None or (isinstance(v, float) and np.isnan(v))
            return {
                "fillColor": NA_FILL if missing else colours.get(v, NA_FILL),
                "color": "#FFFFFF", "weight": 0.8,
                "fillOpacity": 0.55 if missing else opacity,
            }
    else:
        cmap = cm.LinearColormap(ramp, vmin=vmin, vmax=vmax)

        def style(feat):
            v = feat["properties"]["_v"]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return {"fillColor": NA_FILL, "color": "#FFFFFF", "weight": 0.8, "fillOpacity": 0.55}
            return {"fillColor": cmap(float(v)), "color": "#FFFFFF", "weight": 0.8, "fillOpacity": opacity}

    fields = ["label", "district", "_vs"] + [f for f, _ in extra_fields]
    aliases = ["Upazila", "District", layer_name] + [a for _, a in extra_fields]
    folium.GeoJson(
        merged.to_json(),
        name=layer_name,
        style_function=style,
        highlight_function=lambda f: {"weight": 2.4, "color": INK, "fillOpacity": min(1.0, opacity + 0.06)},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True, labels=True),
    ).add_to(m)

    if show_districts and ctx["districts_json"] is not None:
        folium.GeoJson(
            ctx["districts_json"], name="Districts", interactive=False,
            style_function=lambda f: {"fill": False, "color": "#3B4149", "weight": 1.3, "opacity": 0.85},
        ).add_to(m)
    if ctx["outline_json"] is not None:
        folium.GeoJson(
            ctx["outline_json"], name="Division", interactive=False,
            style_function=lambda f: {"fill": False, "color": INK, "weight": 2.2, "opacity": 1.0},
        ).add_to(m)
    if focus_id is not None:
        focus = merged.loc[merged["unit_id"] == focus_id]
        if len(focus):
            folium.GeoJson(
                focus[["unit_id", "geometry"]].to_json(), name="Focus", interactive=False,
                style_function=lambda f: {"fill": False, "color": INK, "weight": 3.2, "opacity": 1.0},
            ).add_to(m)
    if show_labels:
        for name, lat, lon in ctx["labels"]:
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f"<div class='{'ulabel' if ctx.get('label_kind') == 'upazila' else 'dlabel'}'>{name}</div>",
                    icon_size=(140, 14), icon_anchor=(70, 7)),
            ).add_to(m)

    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]], padding=(6, 6))
    return m


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    return plt, LinearSegmentedColormap


def draw_frame(ax, linewidth_scale=1.0):
    if show_districts and ctx["districts"] is not None:
        ctx["districts"].boundary.plot(ax=ax, color="#3B4149", linewidth=0.6 * linewidth_scale)
    if ctx["outline"] is not None:
        ctx["outline"].boundary.plot(ax=ax, color=INK, linewidth=1.1 * linewidth_scale)
    if show_labels:
        for name, lat, lon in ctx["labels"]:
            ax.text(lon, lat, name.upper() if ctx.get("label_kind") != "upazila" else name,
                    fontsize=5.2 * linewidth_scale, ha="center", va="center",
                    color=INK, alpha=0.85, fontweight="semibold",
                    path_effects=None)


def static_choropleth(gdf, values, unit_ids, layer_name, categorical=False, colours=None,
                      vmin=0.0, vmax=1.0, dpi=150, ramp=None, title=None):
    """Matplotlib choropleth. No CDN, no tile server, no JavaScript."""
    plt, LSC = _mpl()
    import matplotlib.patches as mpatches

    ramp = ramp or RAMP
    merged = gdf.merge(
        pd.DataFrame({"unit_id": list(unit_ids), "_v": np.asarray(values)}),
        on="unit_id", how="left",
    )
    fig, ax = plt.subplots(figsize=(6.4, 8.0), dpi=dpi)
    fig.patch.set_facecolor("white")

    if categorical:
        colours = colours or CLASS_COLOURS
        order = [k for k in colours if (merged["_v"] == k).any()]
        for key in order:
            merged[merged["_v"] == key].plot(ax=ax, color=colours[key], edgecolor="white", linewidth=0.4)
        missing = merged[merged["_v"].isna()]
        if len(missing):
            missing.plot(ax=ax, color=NA_FILL, edgecolor="white", linewidth=0.4)
        ax.legend(handles=[mpatches.Patch(color=colours[k], label=k) for k in order],
                  loc="lower left", frameon=False, fontsize=7.5)
    else:
        cmap = LSC.from_list("ramp", ramp)
        merged.plot(
            column="_v", cmap=cmap, ax=ax, legend=True, edgecolor="white", linewidth=0.4,
            vmin=vmin, vmax=vmax,
            legend_kwds={"shrink": 0.42, "label": layer_name, "pad": 0.01},
            missing_kwds={"color": NA_FILL, "label": "Not assessed"},
        )
    draw_frame(ax)
    ax.set_axis_off()
    ax.set_title(title or layer_name, fontsize=10.5, loc="left", color=INK, fontweight="semibold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def render_map(gdf, values, unit_ids, layer_name, categorical=False, colours=None,
               vmin=0.0, vmax=1.0, height=640, props=None, extra_fields=(), focus_id=None,
               ramp=None):
    if RENDERER.startswith("Static"):
        st.image(static_choropleth(gdf, values, unit_ids, layer_name, categorical, colours,
                                   vmin, vmax, ramp=ramp), width="stretch")
    else:
        fmap = build_map(gdf, values, unit_ids, layer_name, categorical=categorical,
                         colours=colours, vmin=vmin, vmax=vmax, props=props,
                         extra_fields=extra_fields, focus_id=focus_id, ramp=ramp)
        st_folium(fmap, height=height, use_container_width=True, returned_objects=[])


@st.cache_data(show_spinner=False)
def component_panels(_gdf, values: pd.DataFrame, ramp: tuple, cols: tuple, districts_on: bool,
                     labels_on: bool, geometry_key: tuple) -> bytes:
    """Small multiples of the components. `_gdf` is not hashed; geometry_key stands in."""
    plt, LSC = _mpl()
    cmap = LSC.from_list("ramp", list(ramp))
    merged = _gdf.merge(values, on="unit_id", how="left")
    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 4.4), dpi=160)
    for ax, col in zip(np.atleast_1d(axes), cols):
        merged.plot(column=col, cmap=cmap, ax=ax, vmin=0, vmax=1, edgecolor="white", linewidth=0.25,
                    missing_kwds={"color": NA_FILL})
        if districts_on and ctx["districts"] is not None:
            ctx["districts"].boundary.plot(ax=ax, color="#3B4149", linewidth=0.4)
        if ctx["outline"] is not None:
            ctx["outline"].boundary.plot(ax=ax, color=INK, linewidth=0.8)
        ax.set_axis_off()
        ax.set_title(f"{COMPONENT_NAMES.get(col, col)} ({col})", fontsize=9.5, loc="left",
                     color=INK, fontweight="semibold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=list(np.atleast_1d(axes)), orientation="horizontal",
                        fraction=0.035, pad=0.02, aspect=60)
    cbar.set_label("component score, 0–1 (min–max within the study area)", fontsize=8, color=INK2)
    cbar.ax.tick_params(labelsize=7.5, colors=INK2)
    cbar.outline.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------------ risk map
if view == "Risk map":
    layer_label = st.radio("Map layer", list(LAYERS), index=0, horizontal=True)
    layer = LAYERS[layer_label]

    left, right = st.columns([1.75, 1], gap="large")

    with right:
        html("<div class='ph'>Focus upazila</div>")
        labels_ranked = result["label"].tolist()
        pick = st.selectbox("Focus upazila", labels_ranked, index=0, label_visibility="collapsed")
        row = result.loc[result["label"] == pick].iloc[0]
        focus_id = row["unit_id"]

    tooltip_props = result[["unit_id"]].copy()
    tooltip_props["hwri_s"] = result["HWRI"].map(fmt3)
    tooltip_props["class_s"] = result["risk_class"].astype(str)
    tooltip_props["h_s"] = result["H"].map(fmt3)
    tooltip_props["e_s"] = result["E"].map(fmt3)
    tooltip_props["v_s"] = result["V"].map(fmt3)
    tooltip_props["hwf_s"] = result["HWF"].map(fmt1)
    tooltip_props["rank_s"] = result["rank"].map(lambda r: f"{r} of {n_units}")
    extra = [("rank_s", "Rank"), ("hwri_s", "HWRI"), ("class_s", "Class"),
             ("h_s", "Hazard"), ("e_s", "Exposure"), ("v_s", "Vulnerability"),
             ("hwf_s", "Heatwave days / yr")]
    if layer in ("HWRI", "risk_class"):
        extra = [e for e in extra if e[0] not in ("hwri_s", "class_s")] if layer == "HWRI" \
            else [e for e in extra if e[0] != "class_s"]

    with left:
        if layer == "risk_class":
            render_map(gdf, result["risk_class"], result["unit_id"], "Risk class",
                       categorical=True, props=tooltip_props, extra_fields=extra, focus_id=focus_id)
            html(legend_html("Risk class", categorical=True, colours=CLASS_COLOURS,
                             order=risk.RISK_CLASSES))
            html(f"<div class='cap'>Quintiles are relative to this {SCOPE}: a 'Very low' "
                 "upazila is the least at risk here, not safe in absolute terms.</div>")
        else:
            name = f"{COMPONENT_NAMES[layer]} ({layer})"
            render_map(gdf, result[layer], result["unit_id"], name, vmin=0.0, vmax=1.0,
                       props=tooltip_props, extra_fields=extra, focus_id=focus_id)
            html(legend_html(name, ramp=RAMP, vmin=0.0, vmax=1.0))
            if layer == "HWRI":
                html("<div class='cap'>HWRI = (H · E · V)<sup>1/3</sup> with V = S · (1 − AC), "
                     f"each min–max scaled within the {SCOPE}. The outlined polygon is the "
                     "focus upazila; hover any upazila for its full profile.</div>")
            else:
                html(f"<div class='cap'>{COMPONENT_NAMES[layer]} component, min–max scaled to "
                     f"[{floor:.2f}, 1] within the {SCOPE}. Hover for the other components.</div>")

    with right:
        cls_col = CLASS_COLOURS.get(row["risk_class"], NA_FILL)
        html(
            "<div class='kv'>"
            f"<div><span>District</span><b>{row['district']}</b></div>"
            f"<div><span>Rank</span><b>{row['rank']} of {n_units}</b></div>"
            f"<div><span>Risk index</span><b>{row['HWRI']:.3f}</b></div>"
            f"<div><span>Class</span><b><i style='background:{cls_col}'></i>{row['risk_class']}</b></div>"
            f"<div><span>Heatwave days / yr</span><b>{fmt1(row['HWF'])}</b></div>"
            f"<div><span>Events / yr</span><b>{row['HWN']:.2f}</b></div>"
            f"<div><span>Peak event Tmax</span><b>{fmt1(row['HWA'])} °C</b></div>"
            f"<div><span>Heat index</span><b>{fmt1(row['HI'])} °C</b></div>"
            "</div>"
        )

        html(f"<div class='ph'>Components<small>bars: focus upazila · ticks: {SCOPE} median</small></div>")
        comp_order = ["H", "E", "S", "AC", "V", "HWRI"]
        prof = pd.DataFrame({
            "component": [COMPONENT_NAMES[c] for c in comp_order],
            "value": [float(row[c]) for c in comp_order],
            "median": [float(result[c].median()) for c in comp_order],
        })
        names_order = [COMPONENT_NAMES[c] for c in comp_order]
        base = alt.Chart(prof).encode(
            y=alt.Y("component:N", sort=names_order, title=None, axis=alt.Axis(labelLimit=140)),
        )
        bars = base.mark_bar(size=14, cornerRadiusEnd=4, color=ACCENT).encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                    axis=alt.Axis(format=".1f", tickCount=5)),
            tooltip=[alt.Tooltip("component:N", title="Component"),
                     alt.Tooltip("value:Q", format=".3f", title=pick),
                     alt.Tooltip("median:Q", format=".3f", title=f"{SCOPE.capitalize()} median")],
        )
        ticks = base.mark_tick(color=INK, thickness=2, size=22).encode(x="median:Q")
        vals = base.mark_text(align="left", baseline="middle", dx=6, fontSize=11, color=INK2,
                              font=FONT).encode(x="value:Q", text=alt.Text("value:Q", format=".2f"))
        chart(styled((bars + ticks + vals).properties(height=alt.Step(26))))

        html(f"<div class='ph'>Ranking<small>{'all ' + str(n_units) if n_units <= 15 else 'top 15'} by HWRI</small></div>")
        top_n = result.head(15)
        if pick not in top_n["label"].values:
            top_n = pd.concat([top_n, result.loc[result["label"] == pick]])
        rank_base = alt.Chart(top_n).encode(
            y=alt.Y("label:N", sort=alt.EncodingSortField("HWRI", order="descending"),
                    title=None, axis=alt.Axis(labelLimit=170)),
            x=alt.X("HWRI:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                    axis=alt.Axis(format=".1f", tickCount=5)),
        )
        rank_bars = rank_base.mark_bar(size=14, cornerRadiusEnd=4).encode(
            color=alt.condition(alt.datum.label == pick, alt.value(ACCENT), alt.value(DEEMPH)),
            tooltip=[alt.Tooltip("rank:Q", title="Rank"), alt.Tooltip("label:N", title="Upazila"),
                     alt.Tooltip("district:N", title="District"),
                     alt.Tooltip("HWRI:Q", format=".3f"), alt.Tooltip("risk_class:N", title="Class")],
        )
        rank_text = rank_base.mark_text(align="left", baseline="middle", dx=6, fontSize=11,
                                        color=INK2, font=FONT).encode(
            text=alt.Text("HWRI:Q", format=".3f"))
        chart(styled((rank_bars + rank_text).properties(height=alt.Step(22))))

    st.markdown("")
    with st.expander(f"All {n_units} upazilas: components and hazard indices", expanded=False):
        show = result[["rank", "label", "district", "risk_class", "HWRI", "H", "E", "S", "AC", "V",
                       "HWN", "HWF", "HWD", "HWA", "HWM", "HI", "n_event_years"]].rename(columns={
            "rank": "Rank", "label": "Upazila", "district": "District", "risk_class": "Class",
            "n_event_years": "Event years"})
        st.dataframe(
            show, hide_index=True, width="stretch", height=420,
            column_config={
                "HWRI": st.column_config.ProgressColumn("HWRI", min_value=0, max_value=1, format="%.3f"),
                **{c: st.column_config.NumberColumn(c, format="%.3f") for c in ["H", "E", "S", "AC", "V"]},
                "HWN": st.column_config.NumberColumn("HWN", format="%.2f", help="Heatwave events per year"),
                "HWF": st.column_config.NumberColumn("HWF", format="%.1f", help="Heatwave days per year"),
                "HWD": st.column_config.NumberColumn("HWD", format="%.1f", help="Longest event, days"),
                "HWA": st.column_config.NumberColumn("HWA °C", format="%.1f", help="Peak Tmax in event years"),
                "HWM": st.column_config.NumberColumn("HWM °C", format="%.1f", help="Mean Tmax on heatwave days"),
                "HI": st.column_config.NumberColumn("HI °C", format="%.1f", help="Heat index on heatwave days"),
            },
        )


# ------------------------------------------------------------- districts & parts
if view in ("Districts & components", "Upazilas & components"):
    d_sum = (
        result.groupby("district")
        .agg(upazilas=("unit_id", "size"), mean_HWRI=("HWRI", "mean"), max_HWRI=("HWRI", "max"),
             high_share=("risk_class", lambda s: s.isin(["High", "Very high"]).mean()),
             HWF=("HWF", "mean"))
        .reset_index()
    )
    top_by_d = result.sort_values("HWRI", ascending=False).drop_duplicates("district")[["district", "label"]]
    d_sum = d_sum.merge(top_by_d.rename(columns={"label": "top_upazila"}), on="district")
    d_sum = d_sum.sort_values("mean_HWRI", ascending=False).reset_index(drop=True)
    d_order = d_sum["district"].tolist()

    c1, c2 = st.columns([1.15, 1], gap="large")
    if single_district:
        with c1:
            html(f"<div class='ph'>All {n_units} upazilas by risk index<small>bars: HWRI · ticks: component H</small></div>")
            ub = alt.Chart(result).encode(
                y=alt.Y("label:N", sort=alt.EncodingSortField("HWRI", order="descending"), title=None,
                        axis=alt.Axis(labelLimit=170)),
                x=alt.X("HWRI:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format=".1f", tickCount=5)))
            ubars = ub.mark_bar(size=14, cornerRadiusEnd=4, color=ACCENT).encode(
                tooltip=[alt.Tooltip("label:N", title="Upazila"), alt.Tooltip("HWRI:Q", format=".3f"),
                         alt.Tooltip("H:Q", format=".3f", title="Hazard"), alt.Tooltip("risk_class:N", title="Class")])
            uticks = ub.mark_tick(color=INK, thickness=2, size=20).encode(x="H:Q")
            utext = ub.mark_text(align="left", baseline="middle", dx=6, fontSize=11, color=INK2, font=FONT).encode(
                text=alt.Text("HWRI:Q", format=".3f"))
            chart(styled((ubars + uticks + utext).properties(height=alt.Step(24))))
            html("<div class='cap'>Every upazila of the district. The tick marks the Hazard component, so "
                 "a long bar with a low tick is risk driven by exposure and vulnerability rather than heat.</div>")
        with c2:
            html("<div class='ph'>Class breakdown</div>")
            cls = result["risk_class"].value_counts().reindex(risk.RISK_CLASSES).fillna(0).astype(int)
            html("<div class='kv'>" + "".join(
                f"<div><span><i style='background:{CLASS_COLOURS[c]}'></i>{c}</span><b>{n}</b></div>"
                for c, n in cls.items()) + "</div>")
            html(f"<div class='cap'>Quintiles are equal-count by construction: {n_units} upazilas split as "
                 f"evenly as possible into five classes. Use the components below to see what drives the order.</div>")
    else:
        with c1:
            html("<div class='ph'>Risk index by district<small>dots: upazilas · ticks: district mean</small></div>")
            dots = alt.Chart(result).mark_circle(size=80, color=ACCENT, opacity=0.95,
                                                 stroke="#FFFFFF", strokeWidth=2).encode(
                y=alt.Y("district:N", sort=d_order, title=None),
                x=alt.X("HWRI:Q", scale=alt.Scale(domain=[0, 1]), title="HWRI",
                        axis=alt.Axis(format=".1f", tickCount=6)),
                tooltip=[alt.Tooltip("label:N", title="Upazila"), alt.Tooltip("district:N", title="District"),
                         alt.Tooltip("HWRI:Q", format=".3f"), alt.Tooltip("risk_class:N", title="Class"),
                         alt.Tooltip("rank:Q", title="Rank")],
            )
            means = alt.Chart(d_sum).mark_tick(color=INK, thickness=2, size=26).encode(
                y=alt.Y("district:N", sort=d_order), x="mean_HWRI:Q",
                tooltip=[alt.Tooltip("district:N", title="District"),
                         alt.Tooltip("mean_HWRI:Q", format=".3f", title="Mean HWRI")],
            )
            chart(styled((dots + means).properties(height=alt.Step(34))))
            html("<div class='cap'>Districts sorted by mean HWRI. Spread within a district is "
                 "often wider than the gap between districts, which is why the index is built "
                 "at upazila level.</div>")
        with c2:
            html("<div class='ph'>District summary</div>")
            st.dataframe(
                d_sum[["district", "upazilas", "mean_HWRI", "high_share", "top_upazila"]]
                .assign(high_share=lambda d: (d["high_share"] * 100).round(0))
                .rename(columns={"district": "District", "upazilas": "Upazilas", "mean_HWRI": "Mean HWRI",
                                 "high_share": "≥ High", "top_upazila": "Highest-risk upazila"}),
                hide_index=True, width="stretch", height=int(35 * (len(d_sum) + 1) + 3),
                column_config={
                    "Upazilas": st.column_config.NumberColumn(width="small"),
                    "Mean HWRI": st.column_config.ProgressColumn("Mean HWRI", min_value=0, max_value=1,
                                                                 format="%.3f", width="small"),
                    "≥ High": st.column_config.NumberColumn(
                        format="%.0f%%", width="small",
                        help="Share of the district's upazilas classed High or Very high"),
                },
            )

    html("<div class='ph' style='margin-top:14px'>Component maps<small>same scale on every panel</small></div>")
    comp_cols = ("H", "E", "S", "AC", "V")
    st.image(component_panels(gdf[["unit_id", "geometry"]], result[["unit_id", *comp_cols]],
                              tuple(RAMP), comp_cols, show_districts, show_labels,
                              (bpath, demo_geometry)), width="stretch")
    html("<div class='cap'>Hazard comes from the climate series; Exposure, Sensitivity and "
         "Adaptive capacity from the indicator table. Vulnerability V = S · (1 − AC). "
         "Where a component is nearly uniform, it contributes little to the ranking.</div>")


# -------------------------------------------------------------- forecast view
if view == "30-day outlook":
    html("<div class='h2'>Next 30 days</div>")
    html("<div class='note'>This is a <b>weather forecast</b>, not a climate projection. "
         "It asks whether a heatwave is likely in the coming month against each upazila's "
         "own historical threshold. It says nothing about 2040 or 2100; that needs CMIP6 "
         "scenario data (see the README).</div>")

    k1, k2 = st.columns([2, 1], vertical_alignment="bottom")
    api_key = k1.text_input(
        "OpenWeather API key", type="password",
        help="Requires a paid plan: the 30-day climatic forecast is not on the free tier. "
             "Leave blank to preview with a synthetic forecast.",
    )
    go = k2.button("Fetch forecast", type="primary", width="stretch")

    fc_daily = None
    if go and api_key:
        from heatwave.sources.openweather import load_openweather

        bar = st.progress(0.0, "Starting")
        try:
            fc_daily = load_openweather(
                zones.centroids(gdf), api_key,
                progress=lambda p, m: bar.progress(min(p, 1.0), m),
            )
            bar.empty()
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"Forecast fetch failed: {exc}")
    elif go:
        st.info("No key entered, showing a synthetic forecast instead.")

    forecast_synthetic = fc_daily is None
    if fc_daily is None:
        rng = np.random.default_rng(11)
        fdates = pd.date_range(pd.Timestamp.today().normalize(), periods=30)
        blocks = []
        for u in unit_ids:
            base_t = float(thresholds.loc[thresholds.unit_id == u, "threshold_c"].iloc[0]) \
                if "doy" not in thresholds.columns else float(thresholds.threshold_c.mean())
            bump = np.zeros(30)
            s = rng.integers(3, 22)
            bump[s:s + rng.integers(2, 6)] = rng.uniform(1.4, 3.2)
            tmax = base_t - 1.4 + bump + rng.normal(0, 0.7, 30)
            dew = np.clip(rng.normal(24.5, 1.6, 30), 16.0, tmax - 1.5)
            blocks.append(pd.DataFrame({
                "unit_id": u, "date": fdates, "tmax_c": tmax,
                "rh_pct": relative_humidity(tmax, dew),
            }))
        fc_daily = pd.concat(blocks, ignore_index=True)

    fc = indices.add_heat_index(fc_daily, dewpoint_col=None, rh_col="rh_pct")
    fflag = indices.flag_exceedances(
        fc, thresholds,
        indices.HeatwaveConfig(min_duration=min_duration, season=(1, 12), percentile=percentile),
    )
    fev = indices.detect_events(
        fflag, indices.HeatwaveConfig(min_duration=min_duration, season=(1, 12))
    )
    summary = indices.forecast_window_summary(
        fflag, fev, indices.HeatwaveConfig(min_duration=min_duration)
    )
    summary = summary.merge(gdf[["unit_id", "label", "district"]], on="unit_id")
    summary["band"] = heat_index_band(summary["peak_hi"].to_numpy()) if "peak_hi" in summary else "None"

    n_hw = int((summary["events"] > 0).sum())
    pk = summary.loc[summary["peak_tmax"].idxmax()]
    first = summary.loc[summary["events"] > 0, "first_event_start"].min() if n_hw else pd.NaT
    fc_tiles = [
        {"label": "Upazilas with a forecast heatwave", "value": f"{n_hw}",
         "sub": f"of {n_units} · ≥{min_duration}-day runs over own threshold"},
        {"label": "Peak forecast Tmax", "value": f"{pk['peak_tmax']:.1f}", "unit": "°C",
         "sub": f"{pk['label']} · threshold {pk['threshold_c']:.1f} °C"},
        {"label": f"Exceedance days, {SCOPE} total", "value": f"{int(summary['exceedance_days'].sum())}",
         "sub": "upazila-days above threshold"},
        {"label": "Earliest forecast event", "value": first.strftime("%d %b") if pd.notna(first) else "none",
         "sub": "first day of the first qualifying run"},
    ]
    if "peak_hi" in summary:
        hp = summary.loc[summary["peak_hi"].idxmax()]
        fc_tiles.insert(2, {"label": "Peak heat index", "value": f"{hp['peak_hi']:.1f}", "unit": "°C",
                            "sub": f"{hp['label']} · {hp['band']}"})
    tiles(fc_tiles)
    if forecast_synthetic:
        html("<div class='cap'>Synthetic forecast: generated around each upazila's own threshold "
             "so the interface can be explored without an API key.</div>")

    fc_layer = st.segmented_control(
        "Colour by", ["Heat-index band (absolute)", "Heatwave days (relative)"],
        default="Heat-index band (absolute)", label_visibility="collapsed",
    ) or "Heat-index band (absolute)"

    fprops = summary[["unit_id"]].copy()
    fprops["ex_s"] = summary["exceedance_days"].astype(int).astype(str)
    fprops["ev_s"] = summary["events"].astype(int).astype(str)
    fprops["lo_s"] = summary["longest_event"].astype(int).astype(str)
    fprops["pt_s"] = summary["peak_tmax"].map(fmt1)
    fprops["band_s"] = summary["band"].astype(str)
    fextra = [("ex_s", "Days above threshold"), ("ev_s", "Events"), ("lo_s", "Longest run"),
              ("pt_s", "Peak Tmax °C"), ("band_s", "Peak HI band")]

    cA, cB = st.columns([1.75, 1], gap="large")
    with cA:
        if fc_layer.startswith("Heat-index"):
            render_map(gdf, summary["band"], summary["unit_id"], "Peak heat-index band",
                       categorical=True, colours=BAND_COLOURS, height=560, props=fprops,
                       extra_fields=[e for e in fextra if e[0] != "band_s"])
            html(legend_html("Peak heat-index band", categorical=True, colours=BAND_COLOURS,
                             order=[b for _, _, b in HEAT_INDEX_BANDS]))
            html("<div class='cap'>Absolute NWS apparent-temperature bands. Unlike the quintile "
                 "map, these mean the same thing every day, so colours are comparable between runs.</div>")
        else:
            vmax_days = max(1.0, float(summary["heatwave_days"].max()))
            render_map(gdf, summary["heatwave_days"], summary["unit_id"], "Heatwave days in window",
                       vmin=0.0, vmax=vmax_days, height=560, props=fprops, extra_fields=fextra)
            html(legend_html("Heatwave days in the next 30 days", ramp=RAMP, vmin=0.0, vmax=vmax_days))
    with cB:
        html("<div class='ph'>Forecast detail</div>")
        st.dataframe(
            summary.sort_values(["heatwave_days", "exceedance_days"], ascending=False)[
                ["label", "exceedance_days", "events", "longest_event", "peak_tmax", "band"]
            ].rename(columns={
                "label": "Upazila", "exceedance_days": "Days > threshold", "events": "Events",
                "longest_event": "Longest", "peak_tmax": "Peak °C", "band": "HI band",
            }),
            height=560, hide_index=True, width="stretch",
            column_config={"Peak °C": st.column_config.NumberColumn(format="%.1f")},
        )

    html("<div class='ph' style='margin-top:12px'>Daily trace</div>")
    default_pick = summary.sort_values("heatwave_days", ascending=False)["label"].iloc[0]
    t1, _ = st.columns([1, 2])
    trace_pick = t1.selectbox("Upazila", summary["label"].tolist(),
                              index=summary["label"].tolist().index(default_pick),
                              label_visibility="collapsed")
    uid = summary.loc[summary.label == trace_pick, "unit_id"].iloc[0]
    trace = fflag.loc[fflag.unit_id == uid, ["date", "tmax_c", "hi_c", "threshold_c", "hot"]].copy()

    series_names = {"tmax_c": "Forecast Tmax", "hi_c": "Heat index", "threshold_c": "Heatwave threshold"}
    long = trace.melt(id_vars=["date"], value_vars=list(series_names), var_name="series", value_name="value")
    long["series"] = long["series"].map(series_names)
    order = list(series_names.values())
    colour = alt.Color("series:N", scale=alt.Scale(domain=order, range=[BLUE, ORANGE, MUTED]),
                       legend=alt.Legend(orient="top", title=None, direction="horizontal"))
    x_enc = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=10))
    y_enc = alt.Y("value:Q", title="°C", scale=alt.Scale(zero=False, nice=True))

    hot_days = trace.loc[trace["hot"], ["date"]].assign(end=lambda d: d["date"] + pd.Timedelta(days=1))
    shade = alt.Chart(hot_days).mark_rect(color=ORANGE, opacity=0.12).encode(x="date:T", x2="end:T")
    lines = alt.Chart(long).mark_line(strokeWidth=2, strokeJoin="round", strokeCap="round").encode(
        x=x_enc, y=y_enc, color=colour,
        strokeDash=alt.condition(alt.datum.series == "Heatwave threshold", alt.value([6, 4]), alt.value([1, 0])),
    )
    nearest = alt.selection_point(nearest=True, on="mouseover", fields=["date"], empty=False)
    points = alt.Chart(long).mark_point(size=70, filled=True, stroke="#FFFFFF", strokeWidth=2).encode(
        x=x_enc, y=y_enc, color=colour, opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
    )
    rule = (
        alt.Chart(long).transform_pivot("series", value="value", groupby=["date"])
        .mark_rule(color=RULE, strokeWidth=1.5)
        .encode(
            x="date:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b")]
                    + [alt.Tooltip(f"{s}:Q", format=".1f", title=f"{s} °C") for s in order],
        )
        .add_params(nearest)
    )
    chart(styled(alt.layer(shade, lines, points, rule).properties(height=280)))
    html("<div class='cap'>Shaded days exceed the upazila's own threshold; a heatwave needs "
         f"{min_duration} or more in a row. Heat index follows the NOAA Rothfusz regression "
         "and is capped at the chart maximum of 58.3 °C.</div>")


# ---------------------------------------------------------------- projection
if view == "2035 projection":
    html("<div class='h2'>2035 projection: 2031–2040 under SSP2-4.5 and SSP5-8.5</div>")
    html("<div class='note'>Same method, future climate. Daily tasmax from <b>NASA NEX-GDDP-CMIP6</b> "
         "(0.25°, bias-corrected) is tested against each upazila's <b>historical</b> threshold, so the "
         "change is measured against today's climate. The six indices are computed per model and only "
         "then summarised as the ensemble median; the inter-model spread is reported, never hidden. "
         "Regional warming by the 2030s differs little between the two pathways; the divergence comes "
         "later in the century.</div>")

    c_a, c_b, c_c = st.columns([1.15, 1.25, 1], vertical_alignment="bottom")
    with c_a:
        per_lo, per_hi = st.select_slider(
            "Projection window", options=list(range(2025, 2061)), value=(2031, 2040),
            help="Ten years around 2035. Heatwave statistics from a single year are unstable.")
    with c_b:
        scaling = st.radio(
            "HWRI scaling", ["Within scenario", "Anchored to historical"], horizontal=True,
            help="Within scenario: min–max inside the future table, exactly as the historical run; "
                 "HWRI is then a ranking and cannot be compared across scenarios. Anchored: scored on "
                 "the historical indicator, component and HWRI ranges with the historical quintile cut "
                 "points, so scenarios and the baseline are comparable; values beyond the historical "
                 "maximum saturate at 1.")
    with c_c:
        ac_improve = st.slider("Adaptive-capacity improvement by 2035", 0, 40, 0, 5, format="%d%%",
                               help="Uniform uplift of literacy, piped water, mobile access, NDVI/NDWI "
                                    "and a matching poverty reduction. Uniform changes cancel under "
                                    "within-scenario scaling; they only register when anchored.")
    d_a, d_b = st.columns([1.15, 2.25], vertical_alignment="bottom")
    with d_a:
        fut_socio_choice = st.selectbox(
            "2035 indicator table (exposure)", ["Same as baseline"] + inputs["socio"],
            format_func=lambda s: s if s.startswith("Same") else Path(s).name,
            help="A per-upazila 2035 table (population, density, built-up). A uniform growth factor "
                 "cancels out under min–max scaling, so only a per-upazila table changes Exposure.")
    fut_socio_path = None if fut_socio_choice.startswith("Same") else fut_socio_choice

    period = (int(per_lo), int(per_hi))
    future_daily = get_future_daily(proj_paths, unit_ids, coastal, period)
    proj_synthetic = not proj_paths
    try:
        per_model, ens = get_future_hazard(future_daily, thresholds, percentile, min_duration,
                                           threshold_mode, period)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    scenarios = [sc for sc in ("ssp245", "ssp585") if sc in set(ens["scenario"])]
    if not scenarios:
        st.error("No SSP2-4.5 or SSP5-8.5 rows found in the export.")
        st.stop()
    change = projection.hazard_change(ens, clim)
    n_models = int(ens["n_models"].max())

    if proj_synthetic:
        html("<div class='cap'>Synthetic stand-in: the historical demo generator warmed by a "
             "scenario-dependent offset with a per-model spread. Replace it with the Earth Engine "
             "export (script at the bottom of this view).</div>")
    if proj_paths:
        matched = future_daily["unit_id"].isin(unit_ids).mean()
        if matched < 0.5:
            st.warning(f"Only {matched:.0%} of export rows match a unit id in the boundary file; "
                       "check that the asset carried GID_3.")

    fut_socio = get_socio(fut_socio_path, unit_ids, coastal) if fut_socio_path else None
    socio_adj = projection.adjust_indicators(socio, ac_improvement=ac_improve / 100.0,
                                             future_socio=fut_socio)
    anchor = projection.historical_anchor(table, rcfg)
    use_anchor = scaling.startswith("Anchored")
    fut = {}
    for sc in scenarios:
        r, _ = projection.future_hwri(ens, sc, socio_adj, clim, rcfg,
                                      anchor=anchor if use_anchor else None)
        fut[sc] = r.merge(gdf[["unit_id", "label", "district"]], on="unit_id", how="left")
    hi_carried = bool(next(iter(fut.values()))["HI_carried"].iloc[0])

    # ---- KPI row
    def _scen_mean(sc, col):
        return float(change.loc[change["scenario"] == sc, col].mean())
    ktiles = [{"label": "GCMs in ensemble", "value": f"{n_models}",
               "sub": f"{period[0]}–{period[1]} · indices per model, then median"}]
    for sc in scenarios:
        sub_df = change.loc[change["scenario"] == sc]
        doubled = int((sub_df["HWF_fut"] >= 2 * sub_df["HWF_hist"]).sum())
        ktiles.append({
            "label": f"Δ heatwave days / yr · {projection.SCENARIOS[sc]}",
            "value": f"{_scen_mean(sc, 'dHWF'):+.1f}", "unit": "days",
            "sub": f"spread {sub_df['dHWF_p25'].mean():+.1f} to {sub_df['dHWF_p75'].mean():+.1f} · "
                   f"doubles in {doubled}/{n_units}",
        })
    hot_sc = scenarios[-1]
    ktiles.append({"label": f"Δ peak Tmax · {projection.SCENARIOS[hot_sc]}",
                   "value": f"{_scen_mean(hot_sc, 'dHWA'):+.1f}", "unit": "°C",
                   "sub": "hottest event, ensemble median vs 1981–2024"})
    if use_anchor:
        sat = {sc: int((fut[sc]["H"] >= 0.999).sum()) for sc in scenarios}
        ktiles.append({"label": "Hazard beyond historical maximum",
                       "value": f"{max(sat.values())}", "unit": f"of {n_units}",
                       "sub": " · ".join(f"{projection.SCENARIOS[sc]} {v}" for sc, v in sat.items())})
    tiles(ktiles)

    # ---- maps, one per scenario, shared scale
    PLAYERS = {
        "Future HWRI": ("HWRI", "fut", False, RAMP, (0.0, 1.0), "{:.2f}"),
        "Risk class": ("risk_class", "fut", True, None, None, None),
        "Δ heatwave days / yr": ("dHWF", "chg", False, None, None, "{:+.1f}"),
        "Future heatwave days / yr": ("HWF", "fut", False, RAMP, None, "{:.1f}"),
        "Δ events / yr": ("dHWN", "chg", False, None, None, "{:+.2f}"),
        "Δ peak Tmax (°C)": ("dHWA", "chg", False, None, None, "{:+.1f}"),
        "Hazard (H)": ("H", "fut", False, RAMP, (0.0, 1.0), "{:.2f}"),
    }
    p_layer = st.radio("Projection layer", list(PLAYERS), index=0, horizontal=True)
    col, src, is_cat, ramp, fixed, fmt = PLAYERS[p_layer]
    frames = {sc: (fut[sc] if src == "fut" else change.loc[change["scenario"] == sc]) for sc in scenarios}
    if not is_cat:
        allv = pd.concat([pd.to_numeric(frames[sc][col], errors="coerce") for sc in scenarios])
        if fixed:
            vmin, vmax = fixed
        elif src == "chg":
            m = float(np.nanmax(np.abs(allv))) or 1.0
            vmin, vmax = -m, m
            ramp = DIVERGING
        else:
            vmin, vmax = 0.0, float(np.nanmax(allv)) or 1.0

    mcols = st.columns(len(scenarios), gap="medium")
    for mcol, sc in zip(mcols, scenarios):
        f = frames[sc]
        props = fut[sc][["unit_id"]].copy()
        chg_sc = change.loc[change["scenario"] == sc].set_index("unit_id")
        props["hwf_h"] = props["unit_id"].map(chg_sc["HWF_hist"]).map(fmt1)
        props["hwf_f"] = props["unit_id"].map(chg_sc["HWF_fut"]).map(fmt1)
        props["hwf_d"] = props["unit_id"].map(chg_sc["dHWF"]).map(lambda x: "–" if pd.isna(x) else f"{x:+.1f}")
        props["spread"] = [f"{a:.1f}–{b:.1f}" for a, b in zip(
            props["unit_id"].map(chg_sc["HWF_fut"] - chg_sc["dHWF"] + chg_sc["dHWF_p25"]),
            props["unit_id"].map(chg_sc["HWF_fut"] - chg_sc["dHWF"] + chg_sc["dHWF_p75"]))]
        props["hwa_d"] = props["unit_id"].map(chg_sc["dHWA"]).map(lambda x: "–" if pd.isna(x) else f"{x:+.1f} °C")
        props["hwri_s"] = fut[sc]["HWRI"].map(fmt3)
        props["cls_s"] = fut[sc]["risk_class"].astype(str)
        extra = [("hwf_h", "Heatwave days / yr, 1981–2024"), ("hwf_f", f"Heatwave days / yr, {period[0]}–{period[1]}"),
                 ("spread", "Model p25–p75"), ("hwf_d", "Δ days / yr"), ("hwa_d", "Δ peak Tmax"),
                 ("hwri_s", "Future HWRI"), ("cls_s", "Class")]
        with mcol:
            html(f"<div class='ph'>{projection.SCENARIOS[sc]}<small>{period[0]}–{period[1]} · "
                 f"{int(ens.loc[ens['scenario'] == sc, 'n_models'].max())} models</small></div>")
            if is_cat:
                render_map(gdf, f["risk_class"], f["unit_id"], "Risk class", categorical=True,
                           height=600, props=props, extra_fields=[e for e in extra if e[0] != "cls_s"])
            else:
                render_map(gdf, f[col], f["unit_id"], p_layer, vmin=vmin, vmax=vmax, height=600,
                           props=props, extra_fields=extra, ramp=ramp)
    if is_cat:
        html(legend_html("Risk class", categorical=True, colours=CLASS_COLOURS, order=risk.RISK_CLASSES))
        html("<div class='cap'>" + ("Fixed cut points from the historical HWRI quintiles: 'Very high' means "
             "inside the historical top-quintile range, so counts are comparable across scenarios."
             if use_anchor else
             "Quintiles within each scenario: always 20% per class, so the map shows where risk "
             "concentrates, not whether it rose. Switch to anchored scaling to compare.") + "</div>")
    else:
        html(legend_html(p_layer, ramp=ramp, vmin=vmin, vmax=vmax, fmt=fmt))
        if col == "HWRI":
            html("<div class='cap'>" + ("Anchored to the historical ranges: a value of 1 means at or beyond the "
                 "1981–2024 maximum. Hazard saturates for most upazilas, so the remaining variation comes "
                 "from exposure and vulnerability." if use_anchor else
                 "Min–max within each scenario, as in the historical run. Compare scenarios on the "
                 "Δ layers, not on this one.") + "</div>")
        elif src == "chg":
            html("<div class='cap'>Ensemble median minus the 1981–2024 climatology. Hover for the "
                 "inter-model p25–p75 range. Expect spatial smoothness: one 0.25° cell spans two or "
                 "three upazilas, so the projection's contribution is the change through time, not new "
                 "spatial detail.</div>")
    if hi_carried:
        html("<div class='cap'>The export carried no humidity, so the heat-index indicator is held at its "
             "historical value per upazila. Export <code>hurs</code> alongside tasmax to project it.</div>")

    # ---- ensemble spread for a focus upazila + scenario matrix
    s1, s2 = st.columns([1.05, 1], gap="large")
    with s1:
        html("<div class='ph'>Model spread<small>dots: GCMs · tick: ensemble median · dashed: 1981–2024</small></div>")
        f1, f2 = st.columns([1.4, 1])
        p_focus = f1.selectbox("Upazila", result["label"].tolist(), index=0, key="proj_focus",
                               label_visibility="collapsed")
        metric_lbl = f2.radio("Metric", ["Heatwave days / yr", "Events / yr", "Peak Tmax °C"],
                              horizontal=True, label_visibility="collapsed")
        metric = {"Heatwave days / yr": "HWF", "Events / yr": "HWN", "Peak Tmax °C": "HWA"}[metric_lbl]
        f_uid = result.loc[result["label"] == p_focus, "unit_id"].iloc[0]
        pm = per_model.loc[per_model["unit_id"] == f_uid, ["scenario", "model", metric]].copy()
        pm["Scenario"] = pm["scenario"].map(projection.SCENARIOS)
        med = ens.loc[ens["unit_id"] == f_uid, ["scenario", metric]].copy()
        med["Scenario"] = med["scenario"].map(projection.SCENARIOS)
        hist_v = float(clim.loc[clim["unit_id"] == f_uid, metric].iloc[0])
        sc_order = [projection.SCENARIOS[sc] for sc in scenarios]
        x_enc = alt.X(f"{metric}:Q", title=metric_lbl, scale=alt.Scale(zero=False, nice=True))
        dots = alt.Chart(pm).mark_circle(size=90, color=ACCENT, opacity=0.9, stroke="#FFFFFF", strokeWidth=2).encode(
            y=alt.Y("Scenario:N", sort=sc_order, title=None), x=x_enc,
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("Scenario:N"),
                     alt.Tooltip(f"{metric}:Q", format=".2f", title=metric_lbl)])
        ticks = alt.Chart(med).mark_tick(color=INK, thickness=2.5, size=30).encode(
            y=alt.Y("Scenario:N", sort=sc_order), x=f"{metric}:Q",
            tooltip=[alt.Tooltip("Scenario:N"), alt.Tooltip(f"{metric}:Q", format=".2f", title="Ensemble median")])
        hist_rule = alt.Chart(pd.DataFrame({"v": [hist_v]})).mark_rule(color=MUTED, strokeDash=[6, 4], strokeWidth=1.5).encode(x="v:Q")
        chart(styled(alt.layer(hist_rule, dots, ticks).properties(height=150)))
        html(f"<div class='cap'>{p_focus}: historical {metric_lbl.lower()} {hist_v:.2f}. Each dot is one GCM's "
             f"{period[0]}–{period[1]} climatology against the same historical threshold.</div>")
    with s2:
        html("<div class='ph'>Scenario matrix<small>anchored to historical ranges</small></div>")
        ac_alt = ac_improve / 100.0 if ac_improve else 0.15
        rows = []
        for sc in scenarios:
            for ac_label, ac_val in (("Static AC", 0.0), (f"Improved AC (+{ac_alt:.0%})", ac_alt)):
                s_adj = projection.adjust_indicators(socio, ac_improvement=ac_val, future_socio=fut_socio)
                r_m, _ = projection.future_hwri(ens, sc, s_adj, clim, rcfg, anchor=anchor)
                rows.append({"Climate": projection.SCENARIOS[sc], "AC": ac_label,
                             "Mean HWRI": float(r_m["HWRI"].mean()),
                             "Very high": int((r_m["risk_class"] == "Very high").sum()),
                             "Δ days/yr": _scen_mean(sc, "dHWF")})
        hist_anchor_res, _ = risk.build_indices(table, cfg=rcfg)
        rows.insert(0, {"Climate": "Baseline", "AC": "Observed (1981–2024)",
                        "Mean HWRI": float(hist_anchor_res["HWRI"].mean()),
                        "Very high": int((hist_anchor_res["risk_class"] == "Very high").sum()),
                        "Δ days/yr": 0.0})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     height=int(35 * (len(rows) + 1) + 3),
                     column_config={
                         "Climate": st.column_config.TextColumn(width="small"),
                         "AC": st.column_config.TextColumn(width="small"),
                         "Mean HWRI": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f", width="small"),
                         "Very high": st.column_config.NumberColumn(width="small", help="Upazilas inside the historical top-quintile HWRI range"),
                         "Δ days/yr": st.column_config.NumberColumn(format="%+.1f", width="small"),
                     })
        html("<div class='cap'>Every row is scored on the same historical yardstick, so the columns are "
             "comparable. With the hazard saturated in both pathways, the adaptive-capacity assumption "
             "moves the index more than the choice of SSP does by the 2030s.</div>")

    # ---- per-upazila table + downloads + GEE code
    with st.expander(f"All {n_units} upazilas: historical vs {period[0]}–{period[1]}", expanded=False):
        tbl = gdf[["unit_id", "label", "district"]].merge(clim[["unit_id", "HWF", "HWA"]], on="unit_id")
        tbl = tbl.rename(columns={"HWF": "HWF 1981–2024", "HWA": "HWA 1981–2024"})
        for sc in scenarios:
            c_sc = change.loc[change["scenario"] == sc].set_index("unit_id")
            f_sc = fut[sc].set_index("unit_id")
            tag = projection.SCENARIOS[sc]
            tbl[f"HWF {tag}"] = tbl["unit_id"].map(c_sc["HWF_fut"])
            tbl[f"p25–p75 {tag}"] = tbl["unit_id"].map(
                lambda u, c=c_sc: f"{c.loc[u, 'HWF_fut'] - c.loc[u, 'dHWF'] + c.loc[u, 'dHWF_p25']:.1f}–"
                                  f"{c.loc[u, 'HWF_fut'] - c.loc[u, 'dHWF'] + c.loc[u, 'dHWF_p75']:.1f}")
            tbl[f"Δ HWA {tag}"] = tbl["unit_id"].map(c_sc["dHWA"])
            tbl[f"HWRI {tag}"] = tbl["unit_id"].map(f_sc["HWRI"])
            tbl[f"Class {tag}"] = tbl["unit_id"].map(f_sc["risk_class"])
        st.dataframe(tbl.drop(columns=["unit_id"]).rename(columns={"label": "Upazila", "district": "District"}),
                     hide_index=True, width="stretch", height=420,
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in tbl.columns
                                    if c.startswith(("HWF", "HWA", "Δ"))}
                     | {c: st.column_config.NumberColumn(format="%.3f") for c in tbl.columns if c.startswith("HWRI")})
        x1, x2 = st.columns(2)
        x1.download_button("Future results per scenario (CSV)",
                           pd.concat(fut.values()).drop(columns=["label", "district"]).merge(
                               change.drop(columns=["n_models"]), on=["scenario", "unit_id"]).to_csv(index=False).encode(),
                           file_name=f"khulna_hwri_{period[0]}_{period[1]}.csv", mime="text/csv", width="stretch")
        x2.download_button("Per-model indices (CSV)", per_model.to_csv(index=False).encode(),
                           file_name=f"khulna_cmip6_per_model_{period[0]}_{period[1]}.csv", mime="text/csv", width="stretch")

    with st.expander("Earth Engine extraction script (NEX-GDDP-CMIP6 → CSV)", expanded=False):
        st.markdown(
            "Upload `data/boundaries/study_area_upazilas.gpkg` as an Earth Engine asset, set its path "
            "on the first line, run, and start the two export tasks. Each CSV has `GID_3, date, model, "
            "scenario, tasmax, hurs`. Drop them under `data/` and they appear in the sidebar. The whole "
            "NEX-GDDP archive is about 38 TB; this pulls only Khulna, the warm season and 2031–2040.")
        st.code(projection.CMIP6_GEE_SNIPPET.strip(), language="javascript")
        st.download_button("Download gee_cmip6_extract.js", projection.CMIP6_GEE_SNIPPET.strip().encode(),
                           file_name="gee_cmip6_extract.js", mime="text/javascript")


# ----------------------------------------------------------------- diagnostics
if view == "Diagnostics":
    html("<div class='h2'>Index diagnostics</div>")

    html(f"<div class='ph'>Indicator weights by component<small>{weighting} weighting</small></div>")
    wcols = st.columns(len(weights))
    for wcol, (name, w) in zip(wcols, weights.items()):
        wdf = w.rename("weight").rename_axis("indicator").reset_index()
        wbase = alt.Chart(wdf, title=alt.TitleParams(text=f"{COMPONENT_NAMES.get(name, name)} ({name})")).encode(
            y=alt.Y("indicator:N", sort="-x", title=None, axis=alt.Axis(labelLimit=150)),
            x=alt.X("weight:Q", title=None, axis=alt.Axis(format=".2f", tickCount=4),
                    scale=alt.Scale(domain=[0, max(0.5, float(w.max()) * 1.25)])),
        )
        wbars = wbase.mark_bar(size=13, cornerRadiusEnd=3, color=ACCENT).encode(
            tooltip=[alt.Tooltip("indicator:N", title="Indicator"), alt.Tooltip("weight:Q", format=".4f")])
        wtext = wbase.mark_text(align="left", baseline="middle", dx=5, fontSize=10.5, color=INK2,
                                font=FONT).encode(text=alt.Text("weight:Q", format=".3f"))
        with wcol:
            chart(styled((wbars + wtext).properties(height=alt.Step(21))))
    html("<div class='cap'>Weights sum to one within each component. With equal weighting "
         "every indicator in a component carries the same weight, so the panels above are flat by "
         "construction; switch to PCA or entropy in the sidebar to see data-driven weights.</div>")

    g1, g2 = st.columns([1, 1.6], gap="large")
    with g1:
        html("<div class='ph'>Sampling adequacy (KMO)</div>")
        kmo_rows = []
        for name, cols in risk.DEFAULT_COMPONENTS.items():
            present = [c for c in cols if c in table.columns]
            if len(present) < 2:
                continue
            z = risk.normalize_indicators(table, present, cfg=rcfg)
            overall, per = risk.sampling_adequacy(z)
            kmo_rows.append({"Component": f"{COMPONENT_NAMES.get(name, name)} ({name})",
                             "KMO": round(overall, 3), "Indicators": len(present)})
        st.dataframe(pd.DataFrame(kmo_rows), hide_index=True, width="stretch",
                     column_config={"KMO": st.column_config.NumberColumn(format="%.3f")})
        html("<div class='cap'>Below 0.5 means the indicators are not interchangeable measures "
             "of one latent construct. For Hazard and Sensitivity that is expected; they bundle "
             "physically and socially distinct processes. Read it as a conceptual result, not a "
             "data fault.</div>")

        html("<div class='ph' style='margin-top:14px'>Effect of the normalization floor</div>")
        rows = []
        for f in (0.0, 0.02, 0.05, 0.10):
            res_f, _ = risk.build_indices(table, cfg=risk.RiskConfig(floor=f, weighting=weighting))
            rows.append({"Floor": f, "Exact zeros": int((res_f["HWRI"] <= 1e-9).sum()),
                         "Min HWRI": round(float(res_f["HWRI"].min()), 4),
                         "Units at V = 0": int((res_f["V"] <= 1e-9).sum())})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Floor": st.column_config.NumberColumn(format="%.2f")})

    with g2:
        html("<div class='ph'>Robustness to weighting scheme<small>Spearman rank correlation of HWRI</small></div>")
        scores, rho = risk.compare_schemes(table, cfg=rcfg)
        r1, r2 = st.columns([1, 1.4])
        r1.dataframe(rho.style.format("{:.3f}"), width="stretch")
        sc_df = scores.assign(unit=result["label"].to_numpy(), district=result["district"].to_numpy())
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            color=RULE, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        sc = alt.Chart(sc_df).mark_circle(size=70, color=ACCENT, stroke="#FFFFFF", strokeWidth=2).encode(
            x=alt.X("pca:Q", scale=alt.Scale(domain=[0, 1]), title="HWRI with PCA weights",
                    axis=alt.Axis(format=".1f", tickCount=5)),
            y=alt.Y("entropy:Q", scale=alt.Scale(domain=[0, 1]), title="HWRI with entropy weights",
                    axis=alt.Axis(format=".1f", tickCount=5)),
            tooltip=[alt.Tooltip("unit:N", title="Upazila"), alt.Tooltip("district:N", title="District"),
                     alt.Tooltip("equal:Q", format=".3f", title="Equal"),
                     alt.Tooltip("pca:Q", format=".3f", title="PCA"),
                     alt.Tooltip("entropy:Q", format=".3f", title="Entropy")],
        )
        with r2:
            chart(styled((diag + sc).properties(height=290)))
        html("<div class='cap'>If PCA and equal weighting correlate at ρ ≈ 1.00, PCA is adding "
             "machinery without adding information, and equal weighting is the simpler defensible "
             "choice. Points on the dashed line rank identically under both schemes.</div>")


# ---------------------------------------------------------------------- export
if view == "Export":
    html("<div class='h2'>Export</div>")
    export = result.drop(columns=["rank"]).copy()

    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "Results table (CSV)", export.to_csv(index=False).encode(),
        file_name="khulna_hwri.csv", mime="text/csv", width="stretch",
    )
    geo = gdf.merge(export, on="unit_id", how="left")
    gbuf = io.BytesIO()
    geo.to_file(gbuf, driver="GeoJSON")
    e2.download_button(
        "Results with geometry (GeoJSON)", gbuf.getvalue(),
        file_name="khulna_hwri.geojson", mime="application/geo+json", width="stretch",
    )
    if boundary_path is not None and not demo_geometry:
        e3.download_button(
            "Khulna upazila boundaries (GeoPackage)", Path(boundary_path).read_bytes(),
            file_name=Path(boundary_path).name, mime="application/geopackage+sqlite3", width="stretch",
        )

    html("<div class='ph' style='margin-top:16px'>Publication figure<small>static choropleth at 600 dpi</small></div>")
    f1, f2 = st.columns([1, 2])
    fig_ramp = f1.radio("Figure ramp", ["Viridis (thesis)", "Heat"], horizontal=True)
    if f1.button("Render figure", type="primary"):
        png = static_choropleth(
            gdf, export["HWRI"], export["unit_id"], "HWRI",
            vmin=0.0, vmax=1.0, dpi=600, ramp=VIRIDIS if fig_ramp.startswith("Viridis") else HEAT,
            title=f"Heatwave Risk Index, {REGION}",
        )
        f2.image(png, width="stretch")
        f1.download_button("Download PNG (600 dpi)", png, file_name="hwri_map.png", mime="image/png")

    with st.expander("Method summary", expanded=False):
        st.markdown(
            f"""
            **Detection.** {cfg.describe()}. Events are runs of at least {min_duration}
            consecutive exceedance days; runs break at season boundaries and missing days.

            **Hazard indices.** HWN (events / yr), HWF (heatwave days / yr), HWD (longest event),
            HWA (peak Tmax), HWM (mean Tmax on heatwave days), HI (NOAA heat index on heatwave days).
            Frequency indices average over all years, intensity indices over event years only.

            **Index.** Each indicator is min–max scaled to [{floor:.2f}, 1] (inverted where polarity
            is negative), combined with {weighting} weights into H, E, S and AC; then
            V = S · (1 − AC) and HWRI = (H · E · V)^(1/3), rescaled and cut into quintiles.

            **Boundaries.** {boundary_source + ' upazilas, ' + REGION + ' (' + str(n_units) + ' units). Joins use GID_3, never names.' if boundaries_real else 'Synthetic rectangular grid.'}
            """
        )
