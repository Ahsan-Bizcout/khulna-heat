# Khulna heatwave risk dashboard

An interactive dashboard that computes the upazila-level Heatwave Risk Index
for Khulna Division, maps it onto your shapefile, and runs the same detection
logic forward over a 30-day forecast.

Implements the methodology from the thesis: percentile-threshold heatwave
detection, six hazard indices, min–max normalization, PCA / entropy / equal
weighting, `V = S(1 − AC)` and `HWRI = (H · E · V)^(1/3)`.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/extract_khulna.py --study-area path/to/Studyarea/Upazilas.shp --district Khulna
streamlit run app.py
```

`--study-area` imports your own upazila layer (GADM-style columns, `GID_3`
as the key), simplifies it to ~30 m and writes
`data/boundaries/study_area_upazilas.gpkg`, which the app prefers over any
other boundary file. `--district Khulna` keeps only Khulna District's 14
upazilas (`NAME_2 == "Khulna"`); omit it for the whole division. Only the
upazila layer is used; the division polygon is not drawn. The map frame is
the outer boundary of the upazilas themselves, and with a single district the
map labels upazilas and the "Districts & components" view becomes "Upazilas &
components".
Without `--study-area`:

The extraction step downloads GADM level 3 for Bangladesh once (cached under
`data/cache/`), filters it to Khulna Division's 64 upazilas and writes
`data/boundaries/khulna_upazilas.gpkg`, plus a division outline the map uses
as a frame. If you have a division-level shapefile (for example simplemaps'
`bd.shp`), pass it with `--divisions path/to/bd.shp` to take the outline from
there instead of dissolving the upazilas.

Without that step the app still boots on a synthetic rectangular grid. Every
input that is synthetic (boundaries, climate series, indicator table) is
flagged in the header, and the scores are never a result until all three are
real.

For a headless run that writes files instead:

```bash
python scripts/run_pipeline.py --demo --out outputs/
```

---

## Wiring in your own data

### 1. Boundaries

`python scripts/extract_khulna.py` does this for you. Alternatively download
GADM v4.1 for Bangladesh, level 3, from gadm.org and unzip into
`data/boundaries/`; the app finds it automatically and filters to Khulna
Division (by `NAME_1`, falling back to the district list). GADM still spells
Jessore the old way; the loader normalises it to Jashore and splits glued
names such as "BagerhatSadar". Files with "outline" in the name are treated as
the division frame, never as zones.

It joins on `GID_3`, never on name. That matters here: there is a Kaliganj in
both Jhenaidah and Satkhira, and a Daulatpur in both Kushtia and Khulna city.
Any name-based join silently merges them. Ambiguous names are flagged in the
`name_is_ambiguous` column and shown with their district appended.

### 2. Climate series

Three routes, in order of how quickly they get you running.

**Import a CSV you already have (recommended).** If your Earth Engine script
already exports per-upazila daily ERA5-Land means, drop that file anywhere
under `data/` (except `boundaries/` and `cache/`). The sidebar lists every CSV
or parquet that has a date column under *Climate series*; pick it and the
header chip turns green. Column names are matched flexibly and Kelvin is
detected and converted:

```python
from heatwave.sources.tabular import load_tabular
daily = load_tabular("data/era5_daily.csv")
```

`heatwave.sources.tabular.ERA5_GEE_SNIPPET` holds a ready-to-paste Earth
Engine script if you need to regenerate the export.

**Pull it through Earth Engine directly.** Needs `earthengine-api` installed
and authenticated. Fine for short windows; use the Code Editor export for the
full 44-year record.

**OpenWeather.** Forecast only. See the scope note below.

### 3. Socio-economic indicators

A CSV with one row per unit (keyed on `GID_3` / `unit_id`) and the indicator
names in `heatwave/risk.py: DEFAULT_COMPONENTS`. Put it under `data/` and it
appears in the sidebar under *Socio-economic indicators*; for the headless
pipeline pass it with `--socio`. Anything missing is imputed with the
indicator mean and reported at run time.

---

## What the formulas actually do

`heatwave/indices.py` is the formula layer.

1. **Threshold.** For each unit, the 90th percentile of warm-season daily Tmax
   over the 1991–2020 baseline.
2. **Exceedance.** Every warm-season day where Tmax exceeds that unit's
   threshold.
3. **Events.** Runs of three or more consecutive exceedance days. Runs break at
   season boundaries and across missing days, so a spell cannot be stitched
   from one June to the next March.
4. **Indices.** HWN, HWF, HWD, HWA, HWM and HI per unit per year.
5. **Climatology.** Frequency indices averaged over all years; intensity
   indices over event years only.

Two threshold modes are available:

| Mode | What it measures |
|---|---|
| `season_wide` | One percentile across March–June. Events cluster in the hottest weeks; March and June rarely qualify. This is what the thesis uses. |
| `day_of_year` | 15-day moving window per calendar day, following Perkins & Alexander (2013). Detects heat anomalous *for the time of year*. |

Run both and report the difference. They are not the same measurement.

**A consequence worth understanding.** Because the threshold is per-unit, every
upazila exceeds its own threshold on about 10% of days by construction. Near
uniformity in HWF across the division is therefore arithmetic, not climate.
The heat index is computed only on days that already qualified; it plays no
part in detection.

---

## Why there is a normalization floor

Plain min–max sets the lowest unit to exactly 0 on every component. Because
HWRI multiplies H, E and V, that guarantees an exact-zero risk score somewhere,
and `V = S(1 − AC)` guarantees more. Those zeros are artefacts of the scaling,
not findings about the place.

The default `floor=0.05` rescales to [0.05, 1], preserving the full ordering
while removing structural zeros. Set `floor=0.0` to reproduce plain [0,1]
behaviour exactly. The Diagnostics tab shows how many zeros each setting
produces.

---

## Scope of the 30-day forecast

The outlook tab answers one question: **is a heatwave likely in the next month,
measured against each upazila's own historical threshold?** It is a weather
forecast.

It is not a climate projection and cannot be turned into one. There is no
emissions scenario behind it, and forecast skill decays over days.

Two colour scales are offered, and the distinction is the point:

- **Heat-index band (absolute).** NWS apparent-temperature categories. These
  mean the same thing every day, so colours are comparable between runs.
  Use this one operationally.
- **Heatwave days (relative).** Within-window comparison across upazilas.

The forecast summary deliberately reports different quantities from the
historical climatology. Over 30 days you cannot estimate an annual rate, so it
reports exceedance days, whether a qualifying run occurs, and peak heat. Do not
plot these on the same axis as the 1981–2024 climatology.

---

## 2035 projection (SSP2-4.5 and SSP5-8.5)

The **2035 projection** view applies the historical method to future climate.
Data come from **NEX-GDDP-CMIP6** (`ee.ImageCollection("NASA/GDDP-CMIP6")`):
daily `tasmax` and `hurs`, bias-corrected and downscaled to 0.25°, for both
SSP245 and SSP585.

Workflow:

1. **Extract.** Open `scripts/gee_cmip6_extract.js` (also shown inside the
   view) in the Earth Engine Code Editor, point it at your uploaded upazila
   asset, and run the two export tasks. Each CSV has
   `GID_3, date, model, scenario, tasmax, hurs` for 2031–2040, March–June,
   ten GCMs. Don't download the archive; it is ~38 TB.
2. **Drop the CSVs under `data/`.** They appear in the sidebar under
   *Projection exports*. Scenario and model may be columns or part of the
   file name (`..._ssp585_EC-Earth3.csv`). Kelvin is converted.
3. **"2035" = 2031–2040.** The window is a slider; a single year is too
   unstable for heatwave statistics.
4. **Thresholds are carried forward.** Each upazila's historical percentile
   threshold is applied to the future series unchanged, so the result is the
   change relative to today's climate. Recomputing it on 2031–2040 would move
   the goalposts with the warming.
5. **Per model, then ensemble.** The ≥3-day rule and the six indices run per
   GCM; the ensemble median is reported with the p25–p75 and min–max spread.
   Model daily series are never averaged first.
6. **Future HWRI.** The hazard block is replaced by the ensemble median; the
   indicator table can be swapped for a 2035 table (exposure) and adaptive
   capacity uplifted by a slider. `V = S(1 − AC)`, `HWRI = (H·E·V)^(1/3)`.

Two scaling modes, because min–max is a ranking:

| Mode | What HWRI means |
|---|---|
| Within scenario | Thesis procedure applied to the future set. Comparable *within* a scenario only. |
| Anchored to historical | Scored on the historical indicator, component and HWRI ranges with the historical quintile cut points. Comparable across scenarios and with the baseline. Values beyond the historical maximum saturate at 1. |

Compare SSP2-4.5 with SSP5-8.5 on the Δ layers (Δ heatwave days, Δ events,
Δ peak Tmax), which are absolute. By the 2030s the two pathways differ by a
few tenths of a degree; the adaptive-capacity assumption usually moves the
index more than the SSP choice does. The scenario matrix in the view makes
that explicit.

`heatwave/projection.py` holds the ingestion, ensemble and anchoring logic;
`risk.build_indices` gained `component_reference` and `class_edges` for the
anchored mode.

## Notes on projection data

One rule matters more than the rest:

> Compute the percentile threshold from **that model's own historical run**,
> never from ERA5-Land. Every GCM carries a temperature bias. Testing a warm
> model against an ERA5-derived threshold manufactures a heatwave increase that
> is entirely model bias.

Run each model separately and report the ensemble median with its spread.
Inter-model spread will likely exceed the gap between adjacent scenarios.

Expect the projected hazard layer to be close to spatially uniform: at 0.25°,
one grid cell spans two or three upazilas. The contribution of a projection is
the temporal change signal, not new spatial detail. Say so up front.

---

## Layout

```
app.py                      Streamlit dashboard
.streamlit/config.toml      theme (accent colour, light base)
scripts/extract_khulna.py   study-area import, or GADM download + Khulna cut-out
scripts/gee_cmip6_extract.js Earth Engine export of NEX-GDDP-CMIP6 per upazila
heatwave/
  indices.py                thresholds, run detection, the six indices
  humidity.py               Magnus relation, NOAA Rothfusz heat index
  risk.py                   normalization, weighting, HWRI, KMO, robustness, anchoring
  projection.py             CMIP6 ingestion, per-GCM indices, ensemble, future HWRI
  zones.py                  shapefile loading, centroids, demo geometry
  sources/
    tabular.py              CSV/parquet import, GEE export snippet
    gee.py                  ERA5-Land via Earth Engine
    openweather.py          30-day climatic forecast
    synthetic.py            demo data
scripts/run_pipeline.py     headless run, writes CSV + GeoPackage
data/boundaries/            study_area_upazilas.gpkg (preferred), khulna_upazilas.gpkg
data/cache/                 raw GADM download
outputs/                    results land here
```

## Deploying

The app is a plain Streamlit script with no database or secrets, so
**Streamlit Community Cloud** (free) is the simplest host:

1. Push this repo to GitHub (the 14-upazila boundaries are committed, so the
   map works out of the box; climate and indicator inputs stay synthetic until
   you commit or upload real exports under `data/`).
2. Go to https://share.streamlit.io, sign in with GitHub, choose *New app*,
   pick the repo, branch `main`, main file `app.py`, and deploy. Python 3.12
   is pinned in `.python-version`; `requirements.txt` installs everything,
   including GDAL through the `pyogrio` wheels.
3. First boot takes a couple of minutes while geopandas builds its wheels'
   cache; later reloads are fast because the heavy steps are `st.cache_data`.

Alternatives that need no code change: a Hugging Face Space with the
*Streamlit* SDK (add `sdk: streamlit` and `app_file: app.py` to the Space
README front matter), or any container host with
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Dashboard views

- **Risk map** - choropleth of HWRI or any component, with district borders,
  the division outline and a focus upazila whose components are compared with
  the division median. Hover any upazila for its rank, class and hazard
  indices. The full 64-row table sits in an expander below the map.
- **Districts & components** - upazila HWRI by district as a strip plot with
  district means, a district summary table, and small-multiple maps of H, E,
  S, AC and V on one shared scale.
- **2035 projection** - SSP2-4.5 and SSP5-8.5 side by side for 2031–2040:
  future HWRI, risk class, Δ heatwave days, Δ events, Δ peak Tmax; model
  spread per upazila; a scenario × adaptive-capacity matrix; the Earth Engine
  script.
- **30-day outlook**, **Diagnostics** and **Export** as described below.

Views are a rerun-based switcher rather than client-side tabs: a Leaflet map
that mounts inside a hidden tab collapses to zero height, so only the active
view is rendered.

Colour is one single-hue heat ramp (light to dark = low to high) for every
magnitude layer, a five-step cut of the same ramp for the risk classes and the
NWS heat-index bands, and a blue/orange pair for the forecast trace. Viridis
is still available in the sidebar and is the default for the 600 dpi export,
to match the thesis figures.

## Map renderers

Two options in the sidebar:

- **Interactive (Leaflet)** — pan, zoom, hover tooltips. Loads Leaflet from a
  CDN, so it needs outbound internet.
- **Static (matplotlib)** — no CDN, no tile server, no JavaScript. Use it on a
  restricted network, or to preview exactly what the Export tab will render at
  600 dpi.

Set `HWRI_STATIC_MAPS=1` to make static the default, which is what you want for
headless screenshots or CI.

CartoDB is deliberately not offered as a basemap: it now requires an API key.
The default is a plain background, which also matches the thesis figures.

## A note on the heat index

The Rothfusz regression is a curve fit to the NWS heat-index chart, valid for
roughly 80–110 °F with a chart maximum of 137 °F (58.3 °C). It is a high-order
polynomial, so outside that range it diverges rather than degrading. At 43 °C
with 64% RH it returns about 75 °C.

That input combination is itself unphysical — 43 °C air at 64% RH implies a
dewpoint near 34 °C — so in real data it signals a humidity problem upstream,
not unprecedented heat. Unguarded, it silently corrupts the HI indicator, and
since HI feeds Hazard it propagates to the final index.

`heat_index()` therefore caps at the chart maximum and reports how many cells
were affected. If you see that warning on your own ERA5 data, check the
dewpoint column before trusting the HI values.

## Verification

The heat index implementation reproduces the published NWS table (32.2 °C at
70% RH gives 105.9 °F against a published 106 °F). With a 90th-percentile
threshold the baseline exceedance rate comes out at 9.97%, which is the
arithmetic check that detection is wired correctly. If your run departs far
from ~10%, the baseline period or the season filter is wrong.
