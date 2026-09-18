"""Relative humidity and apparent temperature.

Two entry points, because the two data sources give you different things:

  ERA5-Land  -> Tmax and dewpoint      -> relative_humidity() then heat_index()
  OpenWeather-> Tmax and RH directly   -> heat_index() straight away

Both end at the NOAA Rothfusz regression, which is what the methodology
chapter specifies.
"""

from __future__ import annotations

import numpy as np

# Magnus-Tetens coefficients over water (Alduchov & Eskridge 1996 variant of
# the form used in the thesis).
_MAGNUS_A = 17.67
_MAGNUS_B = 243.5  # degrees C
_E0 = 6.112  # hPa

# Bounds of the published NWS heat-index chart that Rothfusz was fitted to.
# Beyond these the polynomial diverges rather than degrading gracefully.
CHART_MAX_T_F = 112.0   # ~44.4 C
CHART_MAX_HI_F = 137.0  # ~58.3 C, the chart maximum


def saturation_vapour_pressure(t_c):
    """Saturation vapour pressure (hPa) at temperature t_c (deg C)."""
    t = np.asarray(t_c, dtype=float)
    return _E0 * np.exp(_MAGNUS_A * t / (t + _MAGNUS_B))


def relative_humidity(tmax_c, dewpoint_c, clip=True):
    """Relative humidity (%) from air temperature and dewpoint, both deg C.

    RH = 100 * e(Td) / es(T)

    Note the approximation carried over from the thesis: ERA5-Land gives a
    daily *mean* dewpoint, which is paired here with the daily *maximum*
    temperature. Dewpoint is far more stable through the day than temperature,
    so this is defensible, but it is an approximation and it biases the heat
    index slightly high. Document it, do not hide it.
    """
    e_actual = saturation_vapour_pressure(dewpoint_c)
    e_sat = saturation_vapour_pressure(tmax_c)
    rh = 100.0 * e_actual / e_sat
    return np.clip(rh, 0.0, 100.0) if clip else rh


def dewpoint_from_rh(t_c, rh_pct):
    """Inverse of the above. Useful when a source gives RH but you want Td."""
    t = np.asarray(t_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1e-6, 100.0)
    gamma = np.log(rh / 100.0) + (_MAGNUS_A * t) / (t + _MAGNUS_B)
    return (_MAGNUS_B * gamma) / (_MAGNUS_A - gamma)


def heat_index(t_c, rh_pct, cap=True, return_flag=False):
    """NOAA Rothfusz apparent temperature, in deg C.

    Full NWS implementation, including the low-heat simple form and the two
    Rothfusz adjustment terms. The regression is defined in Fahrenheit, so we
    convert in and out.

    DOMAIN GUARD (important)
    ------------------------
    Rothfusz is a curve fit to the NWS heat-index chart, which spans roughly
    T = 80-110 F and tops out at HI = 137 F. It is a high-order polynomial, so
    outside that domain it does not degrade gracefully -- it diverges. At
    43 C with 64% RH it returns about 167 F (75 C), which is not a temperature
    anyone has ever felt.

    That combination is itself unphysical (43 C air holding 64% RH implies a
    dewpoint near 34 C), so in real data it signals a humidity error upstream
    rather than genuine extreme heat. Left unguarded it silently poisons the
    HI indicator, and because HI feeds the Hazard component it would propagate
    all the way to the final index.

    With cap=True (default) results are clamped to the chart maximum and the
    out-of-range cells are reported via return_flag=True.

    Reference: Rothfusz (1990), NWS Technical Attachment SR 90-23.
    """
    t_f = np.asarray(t_c, dtype=float) * 9.0 / 5.0 + 32.0
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)

    # Steadman simple form, used when it is not actually hot.
    simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    use_simple = ((simple + t_f) / 2.0) < 80.0

    # Rothfusz regression.
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f * t_f
        - 0.05481717 * rh * rh
        + 0.00122874 * t_f * t_f * rh
        + 0.00085282 * t_f * rh * rh
        - 0.00000199 * t_f * t_f * rh * rh
    )

    dry = (rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    # Clip before the sqrt: outside 78-112 F the bracket goes negative and
    # numpy emits a domain warning even though `dry` masks the result away.
    dry_bracket = np.clip((17.0 - np.abs(t_f - 95.0)) / 17.0, 0.0, None)
    adj_dry = ((13.0 - rh) / 4.0) * np.sqrt(dry_bracket)
    hi_f = np.where(dry, hi_f - adj_dry, hi_f)

    # Humid adjustment: add when air is very humid and moderately hot.
    # This one matters for coastal Khulna.
    humid = (rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    adj_humid = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    hi_f = np.where(humid, hi_f + adj_humid, hi_f)

    hi_f = np.where(use_simple, simple, hi_f)

    out_of_range = (~use_simple) & ((t_f > CHART_MAX_T_F) | (hi_f > CHART_MAX_HI_F))
    if cap:
        hi_f = np.where(out_of_range, CHART_MAX_HI_F, hi_f)

    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    if return_flag:
        return hi_c, out_of_range
    return hi_c


# NWS apparent-temperature risk bands, in deg C. Used for the absolute colour
# scale on the forecast map, as opposed to the relative quintile scale.
HEAT_INDEX_BANDS = [
    (-np.inf, 27.0, "None"),
    (27.0, 32.0, "Caution"),
    (32.0, 41.0, "Extreme caution"),
    (41.0, 54.0, "Danger"),
    (54.0, np.inf, "Extreme danger"),
]


def heat_index_band(hi_c):
    """Label each heat index value with its NWS risk band."""
    hi = np.asarray(hi_c, dtype=float)
    out = np.full(hi.shape, "None", dtype=object)
    for lo, hi_bound, label in HEAT_INDEX_BANDS:
        out = np.where((hi >= lo) & (hi < hi_bound), label, out)
    return out
