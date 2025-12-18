import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import math
import altair as alt

from cs_modulator import (
    apply_cs_modulation,
    calibrate_k_for_target_t90,
    predict_t90_for_reference,
)
from fatigue_model import (
    zone_points_paired,
    zone_points_count_sorted,
    fit_v_of_hr,
    fatigue_index_series,
    global_zone_points_all_activities,
    fit_v_of_hr_global,
)

# ---------------------------------------------------------
# НАСТРОЙКИ
# ---------------------------------------------------------
T_SEG = 7.0            # дължина на сегмента [s]
MIN_D_SEG = 5.0        # минимум хоризонтална дистанция [m]
MIN_T_SEG = 4.0        # минимум продължителност [s]
MAX_ABS_SLOPE = 15.0   # макс. наклон [%]
V_JUMP_KMH = 15.0      # праг за "скачане" на скоростта между сегменти
V_JUMP_MIN = 20.0      # гледаме спайкове само над тази скорост [km/h]

GLIDE_POLY_DEG = 2     # степен на полинома за плъзгаемост
SLOPE_POLY_DEG = 2     # степен на полинома за наклон

# Зонна система като % от критичната скорост
ZONE_BOUNDS = [0.0, 0.75, 0.85, 0.95, 1.05, 1.15, np.inf]
ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


# ---------------------------------------------------------
# ВСПОМОГАТЕЛНИ ФУНКЦИИ
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def poly_to_str(poly, var="s"):
    if poly is None:
        return "няма модел (недостатъчно данни)"
    coeffs = poly.coefficients
    deg = poly.order

    def fmt_coef(c):
        return f"{c:.4f}"

    if deg == 2:
        a, b, c = coeffs
        return (f"{fmt_coef(a)}·{var}² "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}·{var} "
                f"{'+ ' if c >= 0 else '- '}{fmt_coef(abs(c))}")
    elif deg == 1:
        a, b = coeffs
        return (f"{fmt_coef(a)}·{var} "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}")
    else:
        return " + ".join(
            f"{fmt_coef(c)}·{var}^{p}"
            for p, c in zip(range(deg, -1, -1), coeffs)
        )


def seconds_to_hhmmss(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def clean_speed_for_cs(g, v_max_cs=50.0):
    v = g["v_flat_eq"].to_numpy(dtype=float)
    v = np.clip(v, 0.0, v_max_cs)

    if "speed_spike" in g.columns:
        is_spike = g["speed_spike"].to_numpy(dtype=bool)
    else:
        is_spike = np.zeros_like(v, dtype=bool)

    if not is_spike.any():
        return v

    v_clean = v.copy()
    idx = np.arange(len(v_clean))

    good = ~is_spike
    if good.sum() == 0:
        return v_clean
    if good.sum() == 1:
        v_clean[is_spike] = v_clean[good][0]
        return v_clean

    v_clean[is_spike] = np.interp(idx[is_spike], idx[good], v_clean[good])
    return v_clean


def estimate_seg_dt_seconds(df_seg: pd.DataFrame) -> float:
    if df_seg.empty or "dt_s" not in df_seg.columns:
        return T_SEG
    v = df_seg["dt_s"].dropna()
    if v.empty:
        return T_SEG
    return float(np.median(v))


def apply_hr_lag(seg_df: pd.DataFrame, lag_s: float, hr_col="hr_mean") -> pd.DataFrame:
    """
    speed изпреварва HR с lag_s:
      HR_aligned(t) = HR(t + lag_s)
    Реализация: shift(-shift_n) върху HR.
    """
    df = seg_df.copy()
    dt_est = estimate_seg_dt_seconds(df)
    shift_n = int(round(max(lag_s, 0.0) / max(dt_est, 1e-6)))

    df["hr_aligned"] = df.groupby("activity")[hr_col].shift(-shift_n) if shift_n > 0 else df[hr_col]
    df["hr_lag_s_used"] = float(lag_s)
    df["hr_shift_n"] = int(shift_n)
    return df


# ---------------------------------------------------------
# TCX PARSER – с пулс
# ---------------------------------------------------------
def parse_tcx(file, activity_label):
    content = file.read()
    tree = ET.parse(BytesIO(content))
    root = tree.getroot()

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    rows = []
    for lap in root.findall(".//tcx:Lap", ns):
        for tp in lap.findall(".//tcx:Trackpoint", ns):
            t_el = tp.find("tcx:Time", ns)
            if t_el is None:
                continue
            time = datetime.fromisoformat(t_el.text.replace("Z", "+00:00"))

            pos_el = tp.find("tcx:Position", ns)
            lat = lon = None
            if pos_el is not None:
                lat_el = pos_el.find("tcx:LatitudeDegrees", ns)
                lon_el = pos_el.find("tcx:LongitudeDegrees", ns)
                if lat_el is not None and lon_el is not None:
                    lat = float(lat_el.text)
                    lon = float(lon_el.text)

            alt_el = tp.find("tcx:AltitudeMeters", ns)
            elev = float(alt_el.text) if alt_el is not None else None

            dist_el = tp.find("tcx:DistanceMeters", ns)
            dist = float(dist_el.text) if dist_el is not None else None

            hr_el = tp.find(".//tcx:HeartRateBpm/tcx:Value", ns)
            hr = float(hr_el.text) if hr_el is not None else np.nan

            rows.append({
                "activity": activity_label,
                "time": time,
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "dist": dist,
                "hr": hr,
            })

    if not rows:
        return pd.DataFrame(columns=["activity", "time", "lat", "lon", "elev", "dist", "hr"])

    df = pd.DataFrame(rows)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ако няма дистанция – смятаме грубо от lat/lon
    if df["dist"].isna().all():
        df["dist"] = 0.0
        for i in range(1, len(df)):
            if None in (df.at[i-1, "lat"], df.at[i-1, "lon"],
                        df.at[i, "lat"], df.at[i, "lon"]):
                df.at[i, "dist"] = df.at[i-1, "dist"]
                continue
            d = haversine(
                df.at[i-1, "lat"], df.at[i-1, "lon"],
                df.at[i, "lat"], df.at[i, "lon"]
            )
            df.at[i, "dist"] = df.at[i-1, "dist"] + d

    df["dist"] = df["dist"].ffill()
    return df


# ---------------------------------------------------------
# СЕГМЕНТИРАНЕ НА 7 s (с hr_mean)
# ---------------------------------------------------------
def build_segments(df_activity, activity_label):
    if df_activity.empty:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh", "hr_mean"
        ])

    df_activity = df_activity.sort_values("time").reset_index(drop=True)

    times = df_activity["time"].to_numpy()
    elevs = df_activity["elev"].to_numpy()
    dists = df_activity["dist"].to_numpy()
    hrs = df_activity["hr"].to_numpy()

    n = len(df_activity)
    start_idx = 0
    seg_idx = 0
    seg_rows = []

    while start_idx < n - 1:
        t0 = times[start_idx]

        end_idx = start_idx + 1
        while end_idx < n:
            dt_tmp = (times[end_idx] - t0) / np.timedelta64(1, "s")
            if dt_tmp >= T_SEG:
                break
            end_idx += 1

        if end_idx >= n:
            break

        t1 = times[end_idx]
        dt = (t1 - t0) / np.timedelta64(1, "s")

        d0 = dists[start_idx]
        d1 = dists[end_idx]
        elev0 = elevs[start_idx]
        elev1 = elevs[end_idx]
        d_m = max(0.0, d1 - d0)

        if dt < MIN_T_SEG or d_m < MIN_D_SEG:
            start_idx = end_idx
            continue

        if elev0 is None or elev1 is None or np.isnan(elev0) or np.isnan(elev1):
            slope = np.nan
        else:
            slope = (elev1 - elev0) / d_m * 100.0 if d_m > 0 else np.nan

        v_kmh = (d_m / dt) * 3.6
        hr_mean = float(np.nanmean(hrs[start_idx:end_idx + 1]))

        seg_rows.append({
            "activity": activity_label,
            "seg_idx": seg_idx,
            "t_start": pd.to_datetime(t0),
            "t_end": pd.to_datetime(t1),
            "dt_s": float(dt),
            "d_m": float(d_m),
            "slope_pct": float(slope) if not np.isnan(slope) else np.nan,
            "v_kmh": float(v_kmh),
            "hr_mean": hr_mean
        })

        seg_idx += 1
        start_idx = end_idx

    if not seg_rows:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh", "hr_mean"
        ])

    return pd.DataFrame(seg_rows)


# ---------------------------------------------------------
# ФИЛТРИ ЗА НЕРЕАЛИСТИЧНИ СЕГМЕНТИ
# ---------------------------------------------------------
def apply_basic_filters(segments):
    seg = segments.copy()

    valid_slope = seg["slope_pct"].between(-MAX_ABS_SLOPE, MAX_ABS_SLOPE)
    valid_slope &= seg["slope_pct"].notna()
    seg["valid_basic"] = valid_slope

    def mark_speed_spikes(group):
        group = group.sort_values("seg_idx").copy()
        spike = np.zeros(len(group), dtype=bool)
        v = group["v_kmh"].values
        for i in range(1, len(group)):
            dv = abs(v[i] - v[i-1])
            vmax = max(v[i], v[i-1])
            if dv > V_JUMP_KMH and vmax > V_JUMP_MIN:
                spike[i] = True
        group["speed_spike"] = spike
        return group

    seg = seg.groupby("activity", group_keys=False).apply(mark_speed_spikes)
    seg["speed_spike"] = seg["speed_spike"].fillna(False)
    seg.loc[seg["speed_spike"], "valid_basic"] = False

    return seg


# ---------------------------------------------------------
# МОДЕЛ ЗА ПЛЪЗГАЕМОСТ (GLIDE)
# ---------------------------------------------------------
def get_glide_training_segments(seg):
    df = seg.copy()
    df["prev_slope"] = df.groupby("activity")["slope_pct"].shift(1)
    df["prev_valid"] = df.groupby("activity")["valid_basic"].shift(1)

    cond = (
        df["valid_basic"] &
        df["prev_valid"].fillna(False) &
        (df["slope_pct"] <= -5.0) &
        (df["prev_slope"] <= -5.0)
    )

    train = df[cond].copy()
    return train


def fit_glide_poly(train_df):
    if train_df.empty:
        return None
    x = train_df["slope_pct"].values.astype(float)
    y = train_df["v_kmh"].values.astype(float)
    if len(x) <= GLIDE_POLY_DEG:
        return None
    coeffs = np.polyfit(x, y, GLIDE_POLY_DEG)
    return np.poly1d(coeffs)


def compute_glide_coefficients(seg, glide_poly, DAMP_GLIDE):
    train = get_glide_training_segments(seg)
    if glide_poly is None or train.empty:
        return {}

    coeffs = {}
    for act, g in train.groupby("activity"):
        s_mean = g["slope_pct"].mean()
        v_real = g["v_kmh"].mean()
        if v_real <= 0:
            continue
        v_model = float(glide_poly(s_mean))
        if v_model <= 0:
            continue

        k_raw = v_model / v_real
        k_clipped = max(0.9, min(1.25, k_raw))
        k_final = 1.0 + DAMP_GLIDE * (k_clipped - 1.0)
        coeffs[act] = k_final

    return coeffs


def apply_glide_modulation(seg, glide_coeffs):
    seg = seg.copy()
    seg["K_glide"] = seg["activity"].map(glide_coeffs).fillna(1.0)
    seg["v_glide"] = seg["v_kmh"] * seg["K_glide"]
    return seg


# ---------------------------------------------------------
# МОДЕЛ ЗА НАКЛОН
# ---------------------------------------------------------
def compute_flat_ref_speeds(seg_glide):
    flat_refs = {}
    for act, g in seg_glide.groupby("activity"):
        mask_flat = g["slope_pct"].between(-1.0, 1.0) & g["valid_basic"]
        g_flat = g[mask_flat]
        if g_flat.empty:
            continue
        v_flat = g_flat["v_glide"].mean()
        if v_flat > 0:
            flat_refs[act] = v_flat
    return flat_refs


def get_slope_training_data(seg_glide, flat_refs):
    df = seg_glide.copy()
    df["V_flat_ref"] = df["activity"].map(flat_refs)
    mask = (
        df["valid_basic"] &
        df["slope_pct"].between(-15.0, 15.0) &
        df["V_flat_ref"].notna() &
        (df["v_glide"] > 0)
    )
    train = df[mask].copy()
    if train.empty:
        return pd.DataFrame(columns=["slope_pct", "F"])
    train["F"] = train["V_flat_ref"] / train["v_glide"]
    return train[["slope_pct", "F"]]


def fit_slope_poly(train_df):
    if train_df.empty:
        return None
    x = train_df["slope_pct"].values.astype(float)
    y = train_df["F"].values.astype(float)
    if len(x) <= SLOPE_POLY_DEG:
        return None
    coeffs = np.polyfit(x, y, SLOPE_POLY_DEG)
    return np.poly1d(coeffs)


def apply_slope_modulation(seg_glide, slope_poly, V_crit):
    df = seg_glide.copy()
    if slope_poly is None:
        df["v_flat_eq"] = df["v_glide"]
        return df

    slopes = df["slope_pct"].values.astype(float)
    F_vals = slope_poly(slopes)
    F_vals = np.clip(F_vals, 0.7, 1.7)

    mask_mid = np.abs(slopes) <= 1.0
    F_vals[mask_mid] = 1.0

    mask_down = slopes < -1.0
    F_vals[mask_down] = np.minimum(F_vals[mask_down], 1.0)

    mask_up = slopes > 1.0
    F_vals[mask_up] = np.maximum(F_vals[mask_up], 1.0)

    v_flat_eq = df["v_glide"].values * F_vals

    if V_crit is not None and V_crit > 0:
        idx_below = df["slope_pct"] < -3.0
        v_flat_eq[idx_below] = 0.7 * V_crit

    df["v_flat_eq"] = v_flat_eq
    return df


# ---------------------------------------------------------
# ОБОБЩЕНИЕ ПО АКТИВНОСТИ
# ---------------------------------------------------------
def build_activity_summary(
    segments_f,
    train_glide,
    seg_glide,
    seg_slope,
    seg_slope_cs,
    glide_coeffs
):
    activities = sorted(segments_f["activity"].unique())
    summary = pd.DataFrame({"activity": activities})

    if not train_glide.empty:
        glide_train_agg = train_glide.groupby("activity").agg(
            slope_glide_mean=("slope_pct", "mean"),
            v_glide_train_mean=("v_kmh", "mean"),
        ).reset_index()
        summary = summary.merge(glide_train_agg, on="activity", how="left")
    else:
        summary["slope_glide_mean"] = np.nan
        summary["v_glide_train_mean"] = np.nan

    if glide_coeffs:
        K_glide_df = pd.DataFrame(
            {"activity": list(glide_coeffs.keys()),
             "K_glide": list(glide_coeffs.values())}
        )
        summary = summary.merge(K_glide_df, on="activity", how="left")
    else:
        summary["K_glide"] = np.nan

    real_agg = segments_f[segments_f["valid_basic"]].groupby("activity").agg(
        v_real_mean=("v_kmh", "mean")
    ).reset_index()
    summary = summary.merge(real_agg, on="activity", how="left")

    glide_agg = seg_glide[seg_glide["valid_basic"]].groupby("activity").agg(
        v_glide_mean=("v_glide", "mean")
    ).reset_index()
    summary = summary.merge(glide_agg, on="activity", how="left")

    slope_agg = seg_slope[seg_slope["valid_basic"]].groupby("activity").agg(
        v_flat_mean=("v_flat_eq", "mean")
    ).reset_index()
    summary = summary.merge(slope_agg, on="activity", how="left")

    cs_agg = seg_slope_cs[seg_slope_cs["valid_basic"]].groupby("activity").agg(
        v_flat_cs_mean=("v_flat_eq_cs", "mean")
    ).reset_index()
    summary = summary.merge(cs_agg, on="activity", how="left")

    summary["K_slope_eff"] = summary["v_flat_mean"] / summary["v_glide_mean"]

    summary = summary[
        [
            "activity",
            "slope_glide_mean",
            "v_glide_train_mean",
            "K_glide",
            "v_real_mean",
            "v_glide_mean",
            "v_flat_mean",
            "v_flat_cs_mean",
            "K_slope_eff",
        ]
    ]

    summary = summary.rename(columns={
        "activity": "Активност",
        "slope_glide_mean": "Среден наклон на спусканията за модел [%]",
        "v_glide_train_mean": "Средна скорост на спусканията за модел [km/h]",
        "K_glide": "Коефициент плъзгаемост K_glide",
        "v_real_mean": "Средна реална скорост [km/h]",
        "v_glide_mean": "Средна скорост след плъзгаемост [km/h]",
        "v_flat_mean": "Средна скорост еквив. на равно [km/h]",
        "v_flat_cs_mean": "Средна скорост след CS модулация [km/h]",
        "K_slope_eff": "Ефективен коефициент наклон K_slope",
    })

    return summary


# ---------------------------------------------------------
# ЗОНИ ПО СКОРОСТ
# ---------------------------------------------------------
def assign_speed_zones(seg_any_speed, V_crit, speed_col="v_flat_eq"):
    df = seg_any_speed.copy()
    if V_crit is None or V_crit <= 0:
        df["rel_crit"] = np.nan
        df["zone"] = None
        return df

    df["rel_crit"] = df[speed_col] / V_crit

    zones = []
    for r in df["rel_crit"]:
        if pd.isna(r):
            zones.append(None)
            continue
        z_name = None
        for i in range(len(ZONE_NAMES)):
            if ZONE_BOUNDS[i] <= r < ZONE_BOUNDS[i + 1]:
                z_name = ZONE_NAMES[i]
                break
        zones.append(z_name)
    df["zone"] = zones
    return df


def summarize_speed_zones(seg_zones, speed_col="v_flat_eq"):
    df = seg_zones.dropna(subset=["zone"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["zone", "n_segments", "total_time_s", "mean_speed"])
    agg = df.groupby("zone").agg(
        n_segments=("seg_idx", "count"),
        total_time_s=("dt_s", "sum"),
        mean_speed=(speed_col, "mean"),
    ).reset_index()
    agg = agg.sort_values("zone")
    return agg


# ---------------------------------------------------------
# HR по зони – 2 схеми
# ---------------------------------------------------------
def zone_hr_scheme_A_paired(seg_zones, hr_col="hr_aligned"):
    df = seg_zones.copy()
    out = []
    for z in ZONE_NAMES:
        g = df[df["zone"] == z]
        if g.empty:
            out.append({"zone": z, "mean_hr_zone": np.nan})
            continue
        vals = g[hr_col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            out.append({"zone": z, "mean_hr_zone": np.nan})
            continue
        vals.sort()
        if len(vals) >= 20:
            k = int(round(0.10 * len(vals)))
            vals = vals[k:len(vals)-k] if len(vals) - 2*k > 0 else vals
        out.append({"zone": z, "mean_hr_zone": float(np.mean(vals))})
    return pd.DataFrame(out)


def zone_hr_scheme_B_count_sorted(seg_df, zone_counts, hr_col="hr_aligned"):
    df_hr = seg_df.copy()
    if "speed_spike" in df_hr.columns:
        df_hr = df_hr[~df_hr["speed_spike"].fillna(False)]
    df_hr = df_hr.dropna(subset=[hr_col]).copy()

    if df_hr.empty:
        return pd.DataFrame([{"zone": z, "mean_hr_zone": np.nan} for z in ZONE_NAMES])

    df_hr = df_hr.sort_values(hr_col).reset_index(drop=True)

    results = []
    start_idx = 0
    for z in ZONE_NAMES:
        n = int(zone_counts.get(z, 0))
        if n <= 0 or start_idx >= len(df_hr):
            results.append({"zone": z, "mean_hr_zone": np.nan})
            continue
        end_idx = min(start_idx + n, len(df_hr))
        subset = df_hr.iloc[start_idx:end_idx]
        mean_hr = subset[hr_col].mean() if not subset.empty else np.nan
        results.append({"zone": z, "mean_hr_zone": float(mean_hr) if not pd.isna(mean_hr) else np.nan})
        start_idx = end_idx

    return pd.DataFrame(results)


def build_zone_speed_hr_table(seg_zones, speed_col, hr_scheme="A", hr_col="hr_aligned", activity=None):
    if activity is not None:
        df = seg_zones[seg_zones["activity"] == activity].copy()
    else:
        df = seg_zones.copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "Зона", "Брой сегменти", "Време [ч:мм:сс]",
            "Средна скорост [km/h]", "Среден пулс [bpm]"
        ])

    speed_summary = summarize_speed_zones(df, speed_col=speed_col)
    if speed_summary.empty:
        return pd.DataFrame(columns=[
            "Зона", "Брой сегменти", "Време [ч:мм:сс]",
            "Средна скорост [km/h]", "Среден пулс [bpm]"
        ])

    zone_counts = dict(zip(speed_summary["zone"], speed_summary["n_segments"]))

    if hr_scheme.upper() == "A":
        hr_summary = zone_hr_scheme_A_paired(df, hr_col=hr_col)
    else:
        hr_summary = zone_hr_scheme_B_count_sorted(df, zone_counts, hr_col=hr_col)

    merged = pd.merge(speed_summary, hr_summary, on="zone", how="left")
    merged["time_hhmmss"] = merged["total_time_s"].apply(seconds_to_hhmmss)

    merged = merged.rename(columns={
        "zone": "Зона",
        "n_segments": "Брой сегменти",
        "time_hhmmss": "Време [ч:мм:сс]",
        "mean_speed": "Средна скорост [km/h]",
        "mean_hr_zone": "Среден пулс [bpm]",
    })

    merged = merged[[
        "Зона", "Брой сегменти", "Време [ч:мм:сс]",
        "Средна скорост [km/h]", "Среден пулс [bpm]"
    ]]

    return merged


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
st.set_page_config(page_title="Ski Glide & Slope Model", layout="wide")
st.title("Модел за плъзгаемост, наклон и кислороден дълг + HR lag тест")

# ---------- Sidebar: основни параметри ----------
st.sidebar.header("Параметри на наклона и плъзгаемостта")

V_crit = st.sidebar.number_input(
    "Критична скорост V_crit [km/h]",
    min_value=5.0,
    max_value=40.0,
    value=20.0,
    step=0.5,
)

DAMP_GLIDE = st.sidebar.slider(
    "Омекотяване на плъзгаемостта α",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
)

st.sidebar.markdown("---")

# ---------- Sidebar: HR lag + HR схема ----------
st.sidebar.header("HR lag и зониране по 2 схеми")

hr_lag_s = st.sidebar.slider(
    "Lag на пулса спрямо скоростта (speed leads HR) [s]",
    min_value=0,
    max_value=120,
    value=40,
    step=5,
)

hr_source_for_tables = st.sidebar.selectbox(
    "Кой пулс да се ползва за зонните таблици?",
    ["hr_aligned (с lag)", "hr_mean (без lag)"],
    index=0
)
hr_col_used = "hr_aligned" if "aligned" in hr_source_for_tables else "hr_mean"

st.sidebar.markdown("---")

# ---------- Sidebar: CS / кислороден дълг ----------
st.sidebar.header("CS модел (кислороден „дълг“)")

use_vcrit_as_cs = st.sidebar.checkbox("Използвай V_crit като CS", value=True)
CS = V_crit if use_vcrit_as_cs else st.sidebar.number_input("Критична скорост CS [km/h]", 5.0, 40.0, 18.0, 0.5)

tau_min = st.sidebar.number_input("τ_min (s)", 5.0, 120.0, 25.0, 1.0)
k_par = st.sidebar.number_input("k (τ растеж)", 0.0, 500.0, 35.0, 1.0)
q_par = st.sidebar.number_input("q (нелинейност)", 0.1, 3.0, 1.3, 0.1)

gamma_cs = st.sidebar.slider("γ (влияние на дълга)", 0.0, 1.0, 1.0, 0.05)

st.sidebar.subheader("Калибрация по референтен сценарий")
ref_percent = st.sidebar.number_input("Референтна интензивност (% от CS)", 101.0, 200.0, 105.0, 0.5)
target_t90 = st.sidebar.number_input("Желано t₉₀ (s)", 10.0, 1200.0, 60.0, 5.0)

do_calibrate = st.sidebar.button("Приложи калибрация (пресметни k)")

uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла:",
    type=["tcx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Качи поне един TCX файл, за да започнем.")
    st.stop()

# 1) Парсване
all_points = []
for f in uploaded_files:
    label = f.name
    df_act = parse_tcx(f, label)
    if not df_act.empty:
        all_points.append(df_act)

if not all_points:
    st.error("Не успях да извлека данни от файловете.")
    st.stop()

points = pd.concat(all_points, ignore_index=True)

# 2) Сегментиране
seg_list = []
for act, g in points.groupby("activity"):
    seg_df = build_segments(g, act)
    if not seg_df.empty:
        seg_list.append(seg_df)

segments = pd.concat(seg_list, ignore_index=True) if seg_list else pd.DataFrame()
if segments.empty:
    st.error("Не успях да създам сегменти. Провери TCX файловете.")
    st.stop()

# 3) Базови филтри
segments_f = apply_basic_filters(segments)

# 4) Glide модел
train_glide = get_glide_training_segments(segments_f)
glide_poly = fit_glide_poly(train_glide)

if glide_poly is None:
    glide_coeffs = {}
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)
else:
    glide_coeffs = compute_glide_coefficients(segments_f, glide_poly, DAMP_GLIDE)
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)

# 5) Slope модел
flat_refs = compute_flat_ref_speeds(seg_glide)
slope_train = get_slope_training_data(seg_glide, flat_refs)
raw_slope_poly = fit_slope_poly(slope_train)

if raw_slope_poly is None:
    slope_poly = None
    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_crit)
else:
    F0 = float(raw_slope_poly(0.0))
    offset = F0 - 1.0
    coeffs = raw_slope_poly.coefficients.copy()
    coeffs[-1] -= offset
    slope_poly = np.poly1d(coeffs)
    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_crit)

# 5a) чистене от спайкове
seg_slope = seg_slope.sort_values(["activity", "t_start"]).reset_index(drop=True)
for act, g_act in seg_slope.groupby("activity"):
    v_clean = clean_speed_for_cs(g_act, v_max_cs=50.0)
    seg_slope.loc[g_act.index, "v_flat_eq"] = v_clean

# 5b) HR lag alignment (на сегментно ниво)
seg_slope = apply_hr_lag(seg_slope, lag_s=float(hr_lag_s), hr_col="hr_mean")

# 6) CS модулация върху v_flat_eq
if do_calibrate:
    k_par = calibrate_k_for_target_t90(CS, ref_percent, tau_min, q_par, target_t90)
    st.sidebar.success(f"Нов k = {k_par:.2f} (приложен)")

seg_slope = seg_slope.sort_values(["activity", "t_start"]).reset_index(drop=True)
seg_slope["time_s"] = seg_slope.groupby("activity")["dt_s"].cumsum() - seg_slope["dt_s"]

cs_rows = []
for act, g in seg_slope.groupby("activity"):
    v_clean = clean_speed_for_cs(g, v_max_cs=50.0)
    dt_arr = g["dt_s"].to_numpy(dtype=float)

    out_cs = apply_cs_modulation(
        v=v_clean,
        dt=dt_arr,
        CS=CS,
        tau_min=tau_min,
        k_par=k_par,
        q_par=q_par,
        gamma=gamma_cs,
    )

    g_cs = g.copy()
    g_cs["v_flat_eq_cs"] = out_cs["v_mod"]
    g_cs["delta_v_plus_kmh"] = out_cs["delta_v_plus"]
    g_cs["r_kmh"] = out_cs["r"]
    g_cs["tau_s"] = out_cs["tau_s"]
    cs_rows.append(g_cs)

seg_slope_cs = pd.concat(cs_rows, ignore_index=True)

# t90 диагностична метрика
dv_ref, tau_ref_now, t90_now = predict_t90_for_reference(CS, ref_percent, tau_min, k_par, q_par)
st.caption(
    f"CS модел: Δv_ref = {dv_ref:.2f} km/h, τ_ref ≈ {tau_ref_now:.1f} s, "
    f"t₉₀ ≈ {t90_now:.0f} s при {ref_percent:.1f}% от CS."
)

# 7) Обобщение по активности
summary_df = build_activity_summary(
    segments_f,
    train_glide,
    seg_glide,
    seg_slope,
    seg_slope_cs,
    glide_coeffs
)

st.subheader("Обобщение по активности (след нормализация + CS + HR lag инфо)")
st.dataframe(summary_df, use_container_width=True)

# ---------------------------------------------------------
# ГРАФИКА: скорости + HR (две оси)
# ---------------------------------------------------------
st.subheader("Времева серия: скорости (реална/модулирана) и пулс (с/без lag)")

act_list = sorted(seg_slope_cs["activity"].unique())
act_selected = st.selectbox("Избери активност за графика:", act_list, key="plot_act_select")
g_plot = seg_slope_cs[seg_slope_cs["activity"] == act_selected].copy()

if not g_plot.empty:
    dt_est = estimate_seg_dt_seconds(g_plot)
    shift_n = int(round(hr_lag_s / max(dt_est, 1e-6)))
    st.caption(f"dt≈{dt_est:.2f}s | HR lag={hr_lag_s}s → shift_n={shift_n} сегмента")

    df_plot = g_plot[[
        "time_s", "v_kmh", "v_glide", "v_flat_eq", "v_flat_eq_cs", "hr_mean", "hr_aligned"
    ]].copy()

    speed_long = df_plot.melt(
        id_vars=["time_s"],
        value_vars=["v_kmh", "v_glide", "v_flat_eq", "v_flat_eq_cs"],
        var_name="series",
        value_name="value"
    )
    speed_names = {
        "v_kmh": "V_real",
        "v_glide": "V_glide",
        "v_flat_eq": "V_flat_eq",
        "v_flat_eq_cs": "V_flat_eq_cs",
    }
    speed_long["series"] = speed_long["series"].map(speed_names)

    speed_chart = alt.Chart(speed_long).mark_line().encode(
        x=alt.X("time_s:Q", title="Време [s]"),
        y=alt.Y("value:Q", title="Скорост [km/h]"),
        color=alt.Color("series:N", title="Скоростни серии"),
        tooltip=["time_s:Q", "series:N", "value:Q"]
    )

    hr_long = df_plot.melt(
        id_vars=["time_s"],
        value_vars=["hr_mean", "hr_aligned"],
        var_name="hr_series",
        value_name="hr_value"
    )
    hr_names = {"hr_mean": "HR_raw", "hr_aligned": f"HR_aligned(+{hr_lag_s}s)"}
    hr_long["hr_series"] = hr_long["hr_series"].map(hr_names)

    hr_chart = alt.Chart(hr_long).mark_line(strokeDash=[6, 3]).encode(
        x="time_s:Q",
        y=alt.Y("hr_value:Q", title="Пулс [bpm]", axis=alt.Axis(orient="right")),
        color=alt.Color("hr_series:N", title="HR серии"),
        tooltip=["time_s:Q", "hr_series:N", "hr_value:Q"]
    )

    layered = alt.layer(speed_chart, hr_chart).resolve_scale(y="independent")
    st.altair_chart(layered, use_container_width=True)

# ---------------------------------------------------------
# ✅ ЕДНА ОБЩА ТАБЛИЦА: скорости (всички) + HR (raw + lag)
# ---------------------------------------------------------
st.subheader("Една обща таблица: V_real, V_glide, V_flat_eq, V_flat_eq_cs + HR (raw и lag)")

cols_big = [
    "activity",
    "seg_idx",
    "time_s",
    "dt_s",
    "d_m",
    "slope_pct",
    "v_kmh",
    "v_glide",
    "v_flat_eq",
    "v_flat_eq_cs",
    "delta_v_plus_kmh",
    "r_kmh",
    "tau_s",
    "hr_mean",
    "hr_aligned",
    "hr_lag_s_used",
    "hr_shift_n",
    "valid_basic",
    "speed_spike",
]
cols_big = [c for c in cols_big if c in seg_slope_cs.columns]

big_table = seg_slope_cs[seg_slope_cs["activity"] == act_selected][cols_big].copy()

big_table = big_table.rename(columns={
    "seg_idx": "seg",
    "time_s": "t [s]",
    "dt_s": "dt [s]",
    "d_m": "d [m]",
    "slope_pct": "slope [%]",
    "v_kmh": "V_real [km/h]",
    "v_glide": "V_glide [km/h]",
    "v_flat_eq": "V_flat_eq [km/h]",
    "v_flat_eq_cs": "V_flat_eq_cs (x3) [km/h]",
    "delta_v_plus_kmh": "Δv+ [km/h]",
    "r_kmh": "r [km/h]",
    "tau_s": "τ_cs [s]",
    "hr_mean": "HR_raw [bpm]",
    "hr_aligned": f"HR_lagged(+{hr_lag_s}s) [bpm]",
    "hr_lag_s_used": "HR lag [s]",
    "hr_shift_n": "HR shift [seg]",
    "valid_basic": "valid",
    "speed_spike": "spike",
})

st.dataframe(big_table, use_container_width=True, height=520)

# ---------------------------------------------------------
# ЗОНИ – 4 таблици + ЯСНО дали HR е lag-нат + debug ред
# ---------------------------------------------------------
st.subheader("Зони: сравнение на 2 HR-схеми (A: paired | B: count-sorted)")

# ---- HR debug label (показва ясно дали е lag или не) ----
if hr_col_used == "hr_aligned":
    hr_label = f"HR_lagged (+{hr_lag_s}s)"
else:
    hr_label = "HR_raw (без lag)"

# без CS: zonning по v_flat_eq
seg_zones = assign_speed_zones(seg_slope.copy(), V_crit, speed_col="v_flat_eq")

# с CS: zonning по v_flat_eq_cs
seg_zones_cs = assign_speed_zones(seg_slope_cs.copy(), V_crit, speed_col="v_flat_eq_cs")

col1, col2 = st.columns(2)

with col1:
    st.markdown("## Без CS (speed = v_flat_eq)")
    st.markdown(f"### Схема A (paired) — {hr_label}")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones, "v_flat_eq", "A", hr_col_used, activity=None),
        use_container_width=True
    )
    st.markdown(f"### Схема B (count-sorted) — {hr_label}")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones, "v_flat_eq", "B", hr_col_used, activity=None),
        use_container_width=True
    )

with col2:
    st.markdown("## С CS (speed = v_flat_eq_cs)")
    st.markdown(f"### Схема A (paired) — {hr_label}")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones_cs, "v_flat_eq_cs", "A", hr_col_used, activity=None),
        use_container_width=True
    )
    st.markdown(f"### Схема B (count-sorted) — {hr_label}")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones_cs, "v_flat_eq_cs", "B", hr_col_used, activity=None),
        use_container_width=True
    )

# ---- HR lag debug info (за да няма чудене) ----
dt_est_dbg = estimate_seg_dt_seconds(seg_slope_cs)
shift_n_dbg = int(round(hr_lag_s / max(dt_est_dbg, 1e-6)))

st.caption(
    f"DEBUG: HR колона = {hr_col_used} | "
    f"lag = {hr_lag_s} s | "
    f"dt ≈ {dt_est_dbg:.2f} s | "
    f"shift_n = {shift_n_dbg} сегмента"
)
# ---------------------------------------------------------
# NEW: ГЛОБАЛНИ V = f(HR) по всички активности + индекс на умора за избрана
# ---------------------------------------------------------
st.subheader("V = f(HR) (глобално от всички активности) + индекс на умора (отклонение за активност)")

hr_for_models = hr_col_used               # hr_aligned (lag) или hr_mean (raw) според sidebar
speed_for_models = "v_flat_eq_cs"         # тройно модулирана скорост

deg = st.selectbox("Степен на регресията V=f(HR)", [1, 2], index=0)

# 1) Глобални точки по зони от всички активности
zp_global_B = global_zone_points_all_activities(
    seg_all=seg_slope_cs,
    assign_speed_zones_fn=assign_speed_zones,
    V_crit=V_crit,
    speed_col=speed_for_models,
    hr_col=hr_for_models,
    mode="B"  # count-sorted
)

zp_global_A = global_zone_points_all_activities(
    seg_all=seg_slope_cs,
    assign_speed_zones_fn=assign_speed_zones,
    V_crit=V_crit,
    speed_col=speed_for_models,
    hr_col=hr_for_models,
    mode="A"  # paired
)

colg1, colg2 = st.columns(2)
with colg1:
    st.markdown("### Глобални точки — Модел 1 (count-sorted HR, всички активности)")
    st.dataframe(zp_global_B, use_container_width=True, height=280)
with colg2:
    st.markdown("### Глобални точки — Модел 2 (paired HR, всички активности)")
    st.dataframe(zp_global_A, use_container_width=True, height=280)

# 2) Фит на глобалните регресии
poly_B, dfB_fit = fit_v_of_hr_global(zp_global_B, deg=deg)
poly_A, dfA_fit = fit_v_of_hr_global(zp_global_A, deg=deg)

if poly_B is None or poly_A is None:
    st.warning("Няма достатъчно pooled точки (всички активности) за регресия. Увеличи броя TCX или намали степента (deg=1).")
else:
    st.markdown("### Уравнения на глобалните регресии")
    st.write(f"Глобален Модел 1 (count-sorted):  V = {poly_B}")
    st.write(f"Глобален Модел 2 (paired):       V = {poly_A}")

    # 3) Индекс на умора за избраната активност спрямо глобалния модел
    g_act = seg_slope_cs[seg_slope_cs["activity"] == act_selected].copy()
    if g_act.empty:
        st.info("Няма сегменти за избраната активност.")
    else:
        fi_B = fatigue_index_series(
            seg_df=g_act,
            poly=poly_B,
            speed_real_col=speed_for_models,
            hr_input_col=hr_for_models,
        )
        fi_A = fatigue_index_series(
            seg_df=g_act,
            poly=poly_A,
            speed_real_col=speed_for_models,
            hr_input_col=hr_for_models,
        )

        st.markdown("### Динамика на индекса на умора (отклонение спрямо глобалния модел)")

        df_plot_fi = pd.concat([
            fi_B.assign(model="Global_Model1_count_sorted"),
            fi_A.assign(model="Global_Model2_paired")
        ], ignore_index=True)

        chart_fi = alt.Chart(df_plot_fi).mark_line().encode(
            x=alt.X("time_s:Q", title="Време [s]"),
            y=alt.Y("fatigue_index:Q", title="Индекс на умора (km/h) = 2*V_real - V_pred"),
            color=alt.Color("model:N", title="Глобален модел"),
            tooltip=["time_s:Q", "model:N", "fatigue_index:Q", "v_real:Q", "v_pred:Q", "delta_v:Q", "hr_used:Q"]
        )
        st.altair_chart(chart_fi, use_container_width=True)

        # 4) Scatter pooled точки + линии (за прозрачност)
        st.markdown("### Scatter (всички активности): V_flat_eq_cs vs HR_used + линии на регресиите")

        hr_min = float(np.nanmin(pd.concat([dfB_fit["mean_hr"], dfA_fit["mean_hr"]], ignore_index=True)))
        hr_max = float(np.nanmax(pd.concat([dfB_fit["mean_hr"], dfA_fit["mean_hr"]], ignore_index=True)))
        hr_grid = np.linspace(hr_min, hr_max, 250)

        df_line_B = pd.DataFrame({"hr": hr_grid, "v": poly_B(hr_grid), "model": "Global_Model1_count_sorted"})
        df_line_A = pd.DataFrame({"hr": hr_grid, "v": poly_A(hr_grid), "model": "Global_Model2_paired"})

        pts_B = dfB_fit.rename(columns={"mean_hr": "hr", "mean_speed": "v"}).assign(model="Global_Model1_count_sorted")
        pts_A = dfA_fit.rename(columns={"mean_hr": "hr", "mean_speed": "v"}).assign(model="Global_Model2_paired")

        scatter = alt.Chart(pd.concat([pts_B, pts_A], ignore_index=True)).mark_circle(size=45).encode(
            x=alt.X("hr:Q", title=f"HR използван ({hr_for_models})"),
            y=alt.Y("v:Q", title="V по зони (v_flat_eq_cs) [km/h]"),
            color=alt.Color("model:N", title="Модел"),
            tooltip=["model:N", "hr:Q", "v:Q", "activity:N", "zone:N", "n:Q"]
        )

        lines = alt.Chart(pd.concat([df_line_B, df_line_A], ignore_index=True)).mark_line().encode(
            x="hr:Q", y="v:Q", color="model:N"
        )

        st.altair_chart(scatter + lines, use_container_width=True)

        # 5) Export
        st.markdown("### Експорт на индекс на умора (2 глобални модела)")
        fi_export = df_plot_fi[["activity", "time_s", "seg_idx", "model", "hr_used", "v_real", "v_pred", "delta_v", "fatigue_index"]].copy()
        st.download_button(
            "Свали fatigue index като CSV",
            data=fi_export.to_csv(index=False).encode("utf-8"),
            file_name=f"fatigue_index_global_{act_selected}.csv".replace(" ", "_"),
            mime="text/csv"
        )


    # Модел 2: paired HR (Схема A) + mean speed по speed-зоните
    zp_A = zone_points_paired(
        seg_zones=g_act_z,
        speed_col=speed_for_models,
        hr_col=hr_for_models,
    )

    colm1, colm2 = st.columns(2)
    with colm1:
        st.markdown("### Точки по зони — Модел 1 (count-sorted HR)")
        st.dataframe(zp_B, use_container_width=True)
    with colm2:
        st.markdown("### Точки по зони — Модел 2 (paired HR)")
        st.dataframe(zp_A, use_container_width=True)

    # ---- Фит на V=f(HR) ----
    deg = st.selectbox("Степен на регресията V=f(HR)", [1, 2], index=0)
    poly_B, zpB_fit = fit_v_of_hr(zp_B, deg=deg)
    poly_A, zpA_fit = fit_v_of_hr(zp_A, deg=deg)

    if poly_B is None or poly_A is None:
        st.warning("Няма достатъчно точки по зони за регресия (трябват поне deg+1 валидни зони).")
    else:
        st.markdown("### Уравнения на регресиите")
        st.write(f"Модел 1 (count-sorted):  V = {poly_B}")
        st.write(f"Модел 2 (paired):       V = {poly_A}")

        # ---- Индекс на умора (по сегменти) ----
        # HR за индекса: искаш „изместения 40 sec пулс“ → това е hr_aligned,
        # но ако си избрал raw, ще работи с hr_mean.
        fi_B = fatigue_index_series(
            seg_df=g_act,
            poly=poly_B,
            speed_real_col=speed_for_models,
            hr_input_col=hr_for_models,
        )
        fi_A = fatigue_index_series(
            seg_df=g_act,
            poly=poly_A,
            speed_real_col=speed_for_models,
            hr_input_col=hr_for_models,
        )

        # ---- Графики: fatigue_index(t) ----
        import altair as alt

        st.markdown("### Динамика на индекса на умора (2 модела)")
        df_plot_fi = pd.concat([
            fi_B.assign(model="Model1_count_sorted"),
            fi_A.assign(model="Model2_paired")
        ], ignore_index=True)

        chart_fi = alt.Chart(df_plot_fi).mark_line().encode(
            x=alt.X("time_s:Q", title="Време [s]"),
            y=alt.Y("fatigue_index:Q", title="Индекс на умора (km/h)"),
            color=alt.Color("model:N", title="Модел"),
            tooltip=["time_s:Q", "model:N", "fatigue_index:Q", "v_real:Q", "v_pred:Q", "delta_v:Q", "hr_used:Q"]
        )
        st.altair_chart(chart_fi, use_container_width=True)

        # ---- Допълнително: scatter HR vs V + регресионни линии ----
        st.markdown("### Scatter: V_flat_eq_cs vs HR_used + регресии")
        hr_min = float(np.nanmin(pd.concat([zpB_fit["mean_hr"], zpA_fit["mean_hr"]], ignore_index=True)))
        hr_max = float(np.nanmax(pd.concat([zpB_fit["mean_hr"], zpA_fit["mean_hr"]], ignore_index=True)))
        hr_grid = np.linspace(hr_min, hr_max, 200)

        df_line_B = pd.DataFrame({"hr": hr_grid, "v": poly_B(hr_grid), "model": "Model1_count_sorted"})
        df_line_A = pd.DataFrame({"hr": hr_grid, "v": poly_A(hr_grid), "model": "Model2_paired"})

        df_pts_B = zpB_fit.rename(columns={"mean_hr": "hr", "mean_speed": "v"}).assign(model="Model1_count_sorted")
        df_pts_A = zpA_fit.rename(columns={"mean_hr": "hr", "mean_speed": "v"}).assign(model="Model2_paired")

        scatter = alt.Chart(pd.concat([df_pts_B, df_pts_A], ignore_index=True)).mark_circle(size=70).encode(
            x=alt.X("hr:Q", title=f"HR използван ({hr_for_models})"),
            y=alt.Y("v:Q", title="V по зони (v_flat_eq_cs) [km/h]"),
            color=alt.Color("model:N", title="Модел"),
            tooltip=["model:N", "hr:Q", "v:Q"]
        )

        lines = alt.Chart(pd.concat([df_line_B, df_line_A], ignore_index=True)).mark_line().encode(
            x="hr:Q", y="v:Q", color="model:N"
        )

        st.altair_chart(scatter + lines, use_container_width=True)

        # ---- Експорт на FI series ----
        st.markdown("### Експорт на индекс на умора")
        fi_export = df_plot_fi[["activity", "time_s", "seg_idx", "model", "hr_used", "v_real", "v_pred", "delta_v", "fatigue_index"]].copy()
        st.download_button(
            "Свали fatigue index като CSV",
            data=fi_export.to_csv(index=False).encode("utf-8"),
            file_name=f"fatigue_index_{act_selected}.csv".replace(" ", "_"),
            mime="text/csv"
        )


# ---------------------------------------------------------
# ЗОНИ по избрана активност (A vs B)
# ---------------------------------------------------------
st.subheader("Зони по избрана активност (A vs B)")

act_zone = st.selectbox(
    "Избери активност за зонен анализ:",
    act_list,
    key="zone_act_select"
)

col3, col4 = st.columns(2)
with col3:
    st.markdown("### Без CS (speed = v_flat_eq)")
    st.markdown(f"**Схема A — {hr_label}:**")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones, "v_flat_eq", "A", hr_col_used, activity=act_zone),
        use_container_width=True
    )
    st.markdown(f"**Схема B — {hr_label}:**")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones, "v_flat_eq", "B", hr_col_used, activity=act_zone),
        use_container_width=True
    )

with col4:
    st.markdown("### С CS (speed = v_flat_eq_cs)")
    st.markdown(f"**Схема A — {hr_label}:**")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones_cs, "v_flat_eq_cs", "A", hr_col_used, activity=act_zone),
        use_container_width=True
    )
    st.markdown(f"**Схема B — {hr_label}:**")
    st.dataframe(
        build_zone_speed_hr_table(seg_zones_cs, "v_flat_eq_cs", "B", hr_col_used, activity=act_zone),
        use_container_width=True
    )

st.caption(
    "Интерпретация: A(paired) = HR от сегментите в реалната speed-зона; "
    "B(count-sorted) = HR се разпределя по рангове според броя сегменти във всяка speed-зона."
)
