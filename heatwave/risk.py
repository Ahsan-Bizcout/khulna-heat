"""Index construction: normalize, weight, aggregate, classify.

Implements the thesis pipeline:

    Z    = min-max normalized indicator (inverted where polarity is negative)
    C    = sum_j w_j * Z_j           for each of H, E, S, AC
    V    = S * (1 - AC)
    HWRI = (H * E * V) ^ (1/3)

with one deliberate deviation, controlled by `floor`.

WHY THE FLOOR EXISTS
--------------------
Plain min-max sets the lowest unit to exactly 0 on every component. Because
HWRI multiplies H, E and V, that guarantees an exact-zero risk score for at
least one unit, and V = S(1-AC) guarantees more. Those zeros are an artefact
of the scaling, not a finding about the place. `floor=0.05` rescales to
[0.05, 1] instead, which preserves the full ordering while removing the
structural zeros.

Set floor=0.0 to reproduce the original behaviour exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Indicator -> polarity. +1 raises its parent component, -1 lowers it.
DEFAULT_POLARITY = {
    # Hazard
    "HWN": 1, "HWF": 1, "HWD": 1, "HWA": 1, "HWM": 1, "HI": 1,
    # Exposure
    "Population": 1, "PopDensity": 1, "BuiltUp_pct": 1, "Elevation": -1,
    # Sensitivity
    "Elderly_pct": 1, "Children_pct": 1, "AgriWorkers_pct": 1,
    "IndustryWorkers_pct": 1, "Kutcha_pct": 1,
    # Adaptive capacity
    "Poverty_pct": -1, "Literacy_pct": 1, "PipedWater_pct": 1,
    "NDVI": 1, "NDWI": 1, "Mobile_pct": 1,
}

DEFAULT_COMPONENTS = {
    "H": ["HWN", "HWF", "HWD", "HWA", "HWM", "HI"],
    "E": ["Population", "PopDensity", "BuiltUp_pct", "Elevation"],
    "S": ["Elderly_pct", "Children_pct", "AgriWorkers_pct",
          "IndustryWorkers_pct", "Kutcha_pct"],
    "AC": ["Poverty_pct", "Literacy_pct", "PipedWater_pct",
           "NDVI", "NDWI", "Mobile_pct"],
}

RISK_CLASSES = ["Very low", "Low", "Moderate", "High", "Very high"]


@dataclass
class RiskConfig:
    floor: float = 0.05
    weighting: str = "equal"        # "pca" | "entropy" | "equal"
    impute: str = "mean"            # "mean" | "median" | "drop"
    n_components: int | None = None  # None -> Kaiser rule (eigenvalue > 1)


def minmax(
    values, polarity: int = 1, floor: float = 0.0, reference=None
) -> np.ndarray:
    """Rescale to [floor, 1], inverting when polarity is negative.

    `reference` pins the min and max to an external range. Pass the historical
    range when normalizing a forecast window, otherwise today's colours mean
    something different from yesterday's and the map is not comparable
    across runs.
    """
    x = np.asarray(values, dtype=float)
    lo, hi = (np.nanmin(x), np.nanmax(x)) if reference is None else reference
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return np.full(x.shape, (1.0 + floor) / 2.0)
    z = (x - lo) / (hi - lo)
    if polarity < 0:
        z = 1.0 - z
    z = np.clip(z, 0.0, 1.0)
    return floor + z * (1.0 - floor)


def normalize_indicators(
    df: pd.DataFrame,
    indicators: list[str],
    polarity: dict[str, int] | None = None,
    cfg: RiskConfig | None = None,
    reference: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Impute, then min-max normalize every indicator column."""
    cfg = cfg or RiskConfig()
    polarity = polarity or DEFAULT_POLARITY
    reference = reference or {}
    out = pd.DataFrame(index=df.index)

    for name in indicators:
        if name not in df:
            continue
        col = df[name].astype(float)
        if col.isna().any():
            if cfg.impute == "mean":
                col = col.fillna(col.mean())
            elif cfg.impute == "median":
                col = col.fillna(col.median())
        out[name] = minmax(
            col, polarity.get(name, 1), cfg.floor, reference.get(name)
        )
    return out


def equal_weights(columns) -> pd.Series:
    n = len(columns)
    return pd.Series(np.full(n, 1.0 / n), index=list(columns))


def entropy_weights(z: pd.DataFrame) -> pd.Series:
    """Shannon entropy weighting.

    Rewards indicators that discriminate strongly between units. That is a
    statement about variance in this dataset, not about real-world importance
    -- which is why the thesis keeps it as a sensitivity check only.
    """
    x = z.to_numpy(dtype=float)
    x = np.clip(x, 1e-12, None)
    p = x / x.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(len(x))
    entropy = -k * (p * np.log(p)).sum(axis=0)
    divergence = 1.0 - entropy
    if np.isclose(divergence.sum(), 0.0):
        return equal_weights(z.columns)
    return pd.Series(divergence / divergence.sum(), index=z.columns)


def pca_weights(
    z: pd.DataFrame, n_components: int | None = None, squared: bool = False
) -> pd.Series:
    """Loading-based PCA weights.

    Default reproduces the thesis formula: sum over retained components of
    |loading| * variance explained, normalized to sum to 1.

    `squared=True` uses squared loadings instead. That is the more standard
    choice: squaring is sign-agnostic by construction and expresses each
    indicator's share of the component's variance, whereas taking an absolute
    value discards the sign of a negative loading, so an indicator that pulls
    the component down is weighted as if it pushed it up.
    """
    x = z.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    x = x / sd

    _, s, vt = np.linalg.svd(x, full_matrices=False)
    eigenvalues = (s ** 2) / (len(x) - 1)
    var_explained = eigenvalues / eigenvalues.sum()

    if n_components is None:
        n_components = max(1, int((eigenvalues > 1.0).sum()))
    n_components = min(n_components, len(eigenvalues))

    loadings = vt[:n_components].T * np.sqrt(eigenvalues[:n_components])
    contrib = loadings ** 2 if squared else np.abs(loadings)
    raw = (contrib * var_explained[:n_components]).sum(axis=1)

    if np.isclose(raw.sum(), 0.0):
        return equal_weights(z.columns)
    return pd.Series(raw / raw.sum(), index=z.columns)


def get_weights(z: pd.DataFrame, scheme: str, cfg: RiskConfig | None = None):
    cfg = cfg or RiskConfig()
    scheme = scheme.lower()
    if scheme == "equal":
        return equal_weights(z.columns)
    if scheme == "entropy":
        return entropy_weights(z)
    if scheme == "pca":
        return pca_weights(z, cfg.n_components)
    raise ValueError(f"Unknown weighting scheme: {scheme!r}")


def component_index(
    z: pd.DataFrame, weights: pd.Series, floor: float = 0.0
) -> np.ndarray:
    """Weighted linear sum, rescaled to [floor, 1]."""
    cols = [c for c in weights.index if c in z.columns]
    w = weights[cols] / weights[cols].sum()
    raw = (z[cols].to_numpy() * w.to_numpy()).sum(axis=1)
    return minmax(raw, 1, floor)


def build_indices(
    df: pd.DataFrame,
    components: dict[str, list[str]] | None = None,
    cfg: RiskConfig | None = None,
    polarity: dict[str, int] | None = None,
    reference: dict[str, tuple[float, float]] | None = None,
    component_reference: dict[str, tuple[float, float]] | None = None,
    class_edges: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Run the whole chain. Returns (results, weights-by-component).

    `reference` pins indicator min/max; `component_reference` pins the raw
    range of each component (H, E, S, AC, V) and of HWRI_raw ("HWRI");
    `class_edges` are four fixed HWRI cut points. Together they let a future
    scenario be scored on the historical yardstick instead of on its own
    range. The raw ranges of this run are returned in `result.attrs["ranges"]`
    so a historical run can supply them.
    """
    cfg = cfg or RiskConfig()
    components = components or DEFAULT_COMPONENTS
    comp_ref = component_reference or {}

    result = pd.DataFrame(index=df.index)
    weights: dict[str, pd.Series] = {}
    normalized: dict[str, pd.DataFrame] = {}
    ranges: dict[str, tuple[float, float]] = {}

    for name, indicators in components.items():
        present = [c for c in indicators if c in df.columns]
        if not present:
            continue
        z = normalize_indicators(df, present, polarity, cfg, reference)
        w = get_weights(z, cfg.weighting, cfg)
        raw = component_raw(z, w)
        ranges[name] = _range(raw)
        result[name] = minmax(raw, 1, cfg.floor, comp_ref.get(name))
        weights[name] = w
        normalized[name] = z

    if "S" in result and "AC" in result:
        v_raw = result["S"] * (1.0 - result["AC"])
        ranges["V"] = _range(v_raw)
        result["V"] = minmax(v_raw, 1, cfg.floor, comp_ref.get("V"))

    if {"H", "E", "V"}.issubset(result.columns):
        product = result["H"] * result["E"] * result["V"]
        result["HWRI_raw"] = np.cbrt(product)
        ranges["HWRI"] = _range(result["HWRI_raw"])
        result["HWRI"] = minmax(result["HWRI_raw"], 1, cfg.floor, comp_ref.get("HWRI"))
        result["risk_class"] = quintile_class(result["HWRI"], edges=class_edges)

    result.attrs["normalized"] = normalized
    result.attrs["ranges"] = ranges
    return result, weights


def component_raw(z: pd.DataFrame, weights: pd.Series) -> np.ndarray:
    """Weighted linear sum of normalized indicators, before rescaling."""
    cols = [c for c in weights.index if c in z.columns]
    w = weights[cols] / weights[cols].sum()
    return (z[cols].to_numpy() * w.to_numpy()).sum(axis=1)


def _range(values) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    return float(np.nanmin(x)), float(np.nanmax(x))


def quintile_class(values, labels: list[str] | None = None,
                   edges: list[float] | None = None) -> pd.Series:
    """Five classes. Equal-count within this dataset by default; with `edges`
    (four ascending HWRI cut points) the classes are fixed, so two runs can be
    compared on 'how many upazilas are Very high'."""
    labels = labels or RISK_CLASSES
    s = pd.Series(np.asarray(values, dtype=float))
    if edges is not None:
        bins = [-np.inf, *sorted(edges), np.inf]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)
    try:
        return pd.qcut(s.rank(method="first"), 5, labels=labels).astype(str)
    except ValueError:
        return pd.Series([labels[2]] * len(s), index=s.index)


def compare_schemes(
    df: pd.DataFrame,
    components: dict[str, list[str]] | None = None,
    cfg: RiskConfig | None = None,
    schemes: tuple[str, ...] = ("pca", "entropy", "equal"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute HWRI under each scheme and cross-correlate the rankings."""
    cfg = cfg or RiskConfig()
    scores = {}
    for scheme in schemes:
        local = RiskConfig(
            floor=cfg.floor, weighting=scheme,
            impute=cfg.impute, n_components=cfg.n_components,
        )
        res, _ = build_indices(df, components, local)
        if "HWRI" in res:
            scores[scheme] = res["HWRI"].to_numpy()

    table = pd.DataFrame(scores)
    names = list(table.columns)
    rho = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = spearmanr(table[a], table[b]).statistic
            rho.loc[a, b] = rho.loc[b, a] = round(float(r), 4)
    return table, rho


def sampling_adequacy(z: pd.DataFrame) -> tuple[float, pd.Series]:
    """Kaiser-Meyer-Olkin overall and per-indicator MSA.

    Below 0.5 means the indicators do not behave as interchangeable measures
    of one latent construct. For Hazard and Sensitivity that is expected --
    they bundle genuinely distinct processes -- so read a low value as a
    conceptual result, not a data fault.
    """
    x = z.dropna().to_numpy(dtype=float)
    if x.shape[0] < 3 or x.shape[1] < 2:
        return float("nan"), pd.Series(dtype=float)

    corr = np.corrcoef(x, rowvar=False)
    try:
        inv = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        return float("nan"), pd.Series(dtype=float)

    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    np.fill_diagonal(corr, 0.0)

    r2 = (corr ** 2).sum()
    p2 = (partial ** 2).sum()
    overall = r2 / (r2 + p2) if (r2 + p2) else float("nan")

    per = (corr ** 2).sum(axis=0) / (
        (corr ** 2).sum(axis=0) + (partial ** 2).sum(axis=0)
    )
    return float(overall), pd.Series(per, index=z.columns)
