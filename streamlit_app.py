import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import math
import altair as alt

from cs_modulator import apply_cs_modulation, calibrate_k_for_target_t90, predict_t90_for_reference

from fatigue_model import (
    zone_points_paired,
    zone_points_count_sorted,
    fit_v_of_hr_global,
    fatigue_index_series,
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
T_SEG = 7.0
MIN_D_SEG = 5.0
MIN_T_SEG = 4.0
MAX_ABS_SLOPE = 15.0
V_JUMP_KMH = 15.0
V_JUMP_MIN = 20.0

GLIDE_POLY_DEG = 2
SLOPE_POLY_DEG = 2

ZONE_BOUNDS = [0.0, 0.75, 0.85, 0.95, 1.05, 1.15, np.inf]
ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


# ---------------------------------------------------------
# HELPERS
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
        return "няма модел"

    coeffs = poly.coefficients
    deg = poly.order

    def fmt_coef(c):
        return f"{c:.4f}"

    if deg == 2:
        a, b, c = coeffs
        return (f"{fmt_coef(a)}·{var}² "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}·{var} "
                f"{'+ ' if c >= 0 else '- '}{fmt_coef(abs(c))}")
    if deg == 1:
        a, b = coeffs
        return (f"{fmt_coef(a)}·{var} "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}")
    return str(poly)


def seconds_to_hhmmss(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def estimate_seg_dt_seconds(seg_df: pd.DataFrame) -> float:
    if seg_df.empty or "dt_s" not in seg_df.columns:
        return T_SEG
    v = seg_df["dt_s"].dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return T_SEG
    return float(np.median(v))


def clean_speed_for_cs(g, v_max_cs=50.0):
    v = g["v_flat_eq"].to_numpy(dtype=float)
    v = np.clip(v, 0.0, v_max_cs)

    is_spike = g["speed_spike"].to_numpy(dtype=bool) if "speed_spike" in g.columns else np.zeros_like(v, dtype=bool)
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


# ---------------------------------------------------------
# TCX PARSER
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

    if df["dist"].isna().all():
        df["dist"] = 0.0
        for i in range(1, len(df)):
            if None in (df.at[i-1, "lat"], df.at[i-1, "lon"], df.at[i, "lat"], df.at[i, "lon"]):
                df.at[i, "dist"] = df.at[i-1, "dist"]
                continue
            d = haversine(df.at[i-1, "lat"], df.at[i-1, "lon"], df.at[i, "lat"], df.at[i, "lon"])
            df.at[i, "dist"] = df.at[i-1, "dist"] + d

    df["dist"] = df["dist"].ffill()
    return df


# ---------------------------------------------------------
# SEGMENTATION (7s)
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

    # допълнителен базов филтър: сегменти със скорост < 2 km/h (спирания) не влизат в моделите
    seg.loc[seg["v_kmh"] < 2.0, "valid_basic"] = False

    return seg


# ---------------------------------------------------------
# GLIDE MODEL
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
    return df[cond].copy()


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
# SLOPE MODEL
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
# HR ALIGNMENT (lag)
# ---------------------------------------------------------
def add_hr_aligned(seg_df: pd.DataFrame, hr_lag_s: float) -> pd.DataFrame:
    """
    HR_lagged (+40s) означава:
      speed(t) -> HR(t + lag)
    т.е. за сегмент i взимаме HR от бъдещ сегмент i+shift_n
    => pandas shift(-shift_n)
    """
    df = seg_df.copy()
    dt_est = estimate_seg_dt_seconds(df)
    shift_n = int(round(hr_lag_s / max(dt_est, 1e-6)))
    if shift_n < 0:
        shift_n = 0

    df["hr_aligned"] = df.groupby("activity")["hr_mean"].shift(-shift_n)
    return df


# ---------------------------------------------------------
# ZONING
# ---------------------------------------------------------
def assign_speed_zones(seg_df: pd.DataFrame, V_crit: float, speed_col: str = "v_flat_eq") -> pd.DataFrame:
    df = seg_df.copy()
    if V_crit is None or V_crit <= 0 or speed_col not in df.columns:
        df["rel_crit"] = np.nan
        df["zone"] = None
        return df

    df["rel_crit"] = df[speed_col] / V_crit
    zones = []
    for r in df["rel_crit"].to_numpy(dtype=float):
        if np.isnan(r):
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


def build_zone_table(seg_df: pd.DataFrame, speed_col: str, scheme: str, hr_col: str) -> pd.DataFrame:
    """
    Връща:
      Зона | Брой сегменти | Време | Средна скорост | Среден пулс
    scheme: "A" paired или "B" count-sorted
    """
    df = seg_df.dropna(subset=["zone"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Зона", "Брой сегменти", "Време [ч:мм:сс]", "Средна скорост [km/h]", "Среден пулс [bpm]"])

    # speed summary по зоните
    speed_summary = df.groupby("zone").agg(
        n_segments=("seg_idx", "count"),
        total_time_s=("dt_s", "sum"),
        mean_speed=(speed_col, "mean"),
    ).reset_index()

    speed_summary["time_hhmmss"] = speed_summary["total_time_s"].apply(seconds_to_hhmmss)

    if scheme.upper() == "A":
        hr_summary = df.groupby("zone")[hr_col].mean().reset_index().rename(columns={hr_col: "mean_hr"})
    else:
        # count-sorted
        counts = dict(zip(speed_summary["zone"], speed_summary["n_segments"]))

        df_hr = seg_df.copy()
        if "speed_spike" in df_hr.columns:
            df_hr = df_hr[~df_hr["speed_spike"].fillna(False)]
        df_hr = df_hr.dropna(subset=[hr_col]).copy()
        df_hr = df_hr.sort_values(hr_col).reset_index(drop=True)

        rows = []
        start = 0
        for z in ZONE_NAMES:
            n = int(counts.get(z, 0))
            if n <= 0 or start >= len(df_hr):
                rows.append({"zone": z, "mean_hr": np.nan})
                continue
            end = min(start + n, len(df_hr))
            subset = df_hr.iloc[start:end]
            rows.append({"zone": z, "mean_hr": float(subset[hr_col].mean()) if not subset.empty else np.nan})
            start = end
        hr_summary = pd.DataFrame(rows)

    out = speed_summary.merge(hr_summary, left_on="zone", right_on="zone", how="left")

    out = out.rename(columns={
        "zone": "Зона",
        "n_segments": "Брой сегменти",
        "time_hhmmss": "Време [ч:мм:сс]",
        "mean_speed": "Средна скорост [km/h]",
        "mean_hr": "Среден пулс [bpm]",
    })

    out = out[["Зона", "Брой сегменти", "Време [ч:мм:сс]", "Средна скорост [km/h]", "Среден пулс [bpm]"]]
    return out.sort_values("Зона")


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Ski Glide & Slope Model", layout="wide")
st.title("Модел за плъзгаемост, наклон и кислороден дълг + HR lag + V=f(HR) + Индекс на умора")

# Sidebar parameters
st.sidebar.header("Параметри (наклон/плъзгаемост/зони)")
V_crit = st.sidebar.number_input("Критична скорост V_crit [km/h]", min_value=5.0, max_value=40.0, value=20.0, step=0.5)
DAMP_GLIDE = st.sidebar.slider("Омекотяване на плъзгаемостта α", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Корекция на скоростта от TCX")
speed_scale = st.sidebar.slider(
    "Коефициент за корекция на скоростта",
    min_value=0.80,
    max_value=1.20,
    value=1.00,
    step=0.01,
    help="Ако Garmin показва по-ниска средна скорост от тази тук, избери коефициент < 1.0 (напр. 0.95)."
)

st.sidebar.markdown("---")
st.sidebar.header("HR синхронизация (lag)")
hr_lag_s = st.sidebar.slider("HR lag [s] (speed изпреварва пулса)", min_value=0, max_value=120, value=40, step=1)
hr_col_used = st.sidebar.selectbox("Кой пулс да се ползва за модели/таблици?", ["hr_aligned", "hr_mean"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("CS модел (кислороден „дълг“)")

use_vcrit_as_cs = st.sidebar.checkbox("Използвай V_crit като CS", value=True)
CS = V_crit if use_vcrit_as_cs else st.sidebar.number_input("CS [km/h]", min_value=5.0, max_value=40.0, value=18.0, step=0.5)

tau_min = st.sidebar.number_input("τ_min (s)", min_value=5.0, max_value=120.0, value=25.0, step=1.0)
k_par = st.sidebar.number_input("k (τ растеж)", min_value=0.0, max_value=500.0, value=35.0, step=1.0)
q_par = st.sidebar.number_input("q (нелинейност)", min_value=0.1, max_value=3.0, value=1.3, step=0.1)
gamma_cs = st.sidebar.slider("γ (влияние на дълга)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

st.sidebar.subheader("Калибрация по референтен сценарий")
ref_percent = st.sidebar.number_input("Референтна интензивност (% от CS)", min_value=101.0, max_value=200.0, value=105.0, step=0.5)
target_t90 = st.sidebar.number_input("Желано t₉₀ (s)", min_value=10.0, max_value=1200.0, value=60.0, step=5.0)
do_calibrate = st.sidebar.button("Приложи калибрация (пресметни k)")

uploaded_files = st.file_uploader("Качи един или няколко TCX файла:", type=["tcx"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Качи поне един TCX файл, за да започнем.")
    st.stop()

# Parse TCX
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

# Segment
seg_list = []
for act, g in points.groupby("activity"):
    seg_df = build_segments(g, act)
    if not seg_df.empty:
        seg_list.append(seg_df)

segments = pd.concat(seg_list, ignore_index=True) if seg_list else pd.DataFrame()
if segments.empty:
    st.error("Не успях да създам сегменти.")
    st.stop()

# запазваме суровата скорост и прилагаме корекцията от TCX коефициента
segments["v_kmh_raw"] = segments["v_kmh"]
segments["v_kmh"] = segments["v_kmh_raw"] * speed_scale

segments_f = apply_basic_filters(segments)

# Glide
train_glide = get_glide_training_segments(segments_f)
glide_poly = fit_glide_poly(train_glide)

if glide_poly is None:
    glide_coeffs = {}
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)
else:
    glide_coeffs = compute_glide_coefficients(segments_f, glide_poly, DAMP_GLIDE)
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)

# Slope
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

# Clean v_flat_eq spikes
seg_slope = seg_slope.sort_values(["activity", "t_start"]).reset_index(drop=True)
for act, g_act in seg_slope.groupby("activity"):
    v_clean = clean_speed_for_cs(g_act, v_max_cs=50.0)
    seg_slope.loc[g_act.index, "v_flat_eq"] = v_clean

# Add HR aligned (lag)
seg_slope = add_hr_aligned(seg_slope, hr_lag_s=hr_lag_s)

# CS calibration
if do_calibrate:
    k_par = calibrate_k_for_target_t90(CS, ref_percent, tau_min, q_par, target_t90)
    st.sidebar.success(f"Нов k = {k_par:.2f} (приложен)")

# CS modulation on v_flat_eq
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

# сегментите с реална скорост < 2 km/h ги махаме от крайната модулирана скорост (NaN)
seg_slope_cs.loc[seg_slope_cs["v_kmh"] < 2.0, "v_flat_eq_cs"] = np.nan

# Re-add aligned HR to CS DF (keeps same alignment)
seg_slope_cs["hr_aligned"] = seg_slope_cs["hr_aligned"]

# CS diagnostics
dv_ref, tau_ref_now, t90_now = predict_t90_for_reference(CS, ref_percent, tau_min, k_par, q_par)
st.caption(f"CS модел: Δv_ref={dv_ref:.2f} km/h, τ_ref≈{tau_ref_now:.1f} s, t90≈{t90_now:.0f} s при {ref_percent:.1f}% от CS.")

with st.expander("ℹ️ Обяснение на CS модела и нормализиране на скоростта"):
    st.markdown(
        """
**CS моделът** описва как се натрупва „кислороден дълг“ при скорости над критичната скорост (CS).  
Нормализираната скорост `v_flat_eq_cs` отчита:
- плъзгаемостта (глайд),
- наклона,
- и ефекта на метаболитното натоварване спрямо CS.

По този начин скоростите от различни трасета и дни стават сравними на една и съща физиологична скала.
        """
    )

st.subheader("Обобщение по активности (средни скорости + среден пулс)")

# какъв пулс да обобщаваме (същия, който ползваш за моделите/таблиците)
hr_summary_col = hr_col_used  # "hr_aligned" или "hr_mean"

summary_cols_needed = ["activity", "dt_s", "v_kmh", "v_glide", "v_flat_eq", "v_flat_eq_cs", hr_summary_col]
miss = [c for c in summary_cols_needed if c not in seg_slope_cs.columns]

if miss:
    st.warning(f"Липсват колони за summary таблицата: {miss}")
else:
    df_sum = seg_slope_cs.copy()

    # по избор: изключи speed_spike ако искаш
    if "speed_spike" in df_sum.columns:
        df_sum = df_sum[~df_sum["speed_spike"].fillna(False)].copy()

    # по избор: филтър за спирания (примерно под 2 km/h) – за визуализация/summary
    min_speed_summary = st.slider("Филтър за обобщение: минимална реална скорост [km/h]", 0.0, 5.0, 0.0, 0.5)
    if min_speed_summary > 0:
        df_sum = df_sum[df_sum["v_kmh"] >= min_speed_summary].copy()

    df_activity_summary = (
        df_sum.groupby("activity", as_index=False)
        .agg(
            duration_s=("dt_s", "sum"),
            V_real_mean=("v_kmh", "mean"),
            V_mod1_glide_mean=("v_glide", "mean"),
            V_mod2_slope_mean=("v_flat_eq", "mean"),
            V_mod3_cs_mean=("v_flat_eq_cs", "mean"),
            HR_mean=(hr_summary_col, "mean"),
        )
    )

    df_activity_summary["duration_h"] = df_activity_summary["duration_s"] / 3600.0
    df_activity_summary = df_activity_summary.drop(columns=["duration_s"])
    df_activity_summary = df_activity_summary.sort_values("activity").reset_index(drop=True)

    st.dataframe(df_activity_summary, use_container_width=True, height=360)

    st.download_button(
        "Свали summary CSV",
        data=df_activity_summary.to_csv(index=False).encode("utf-8"),
        file_name="activity_summary.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# Charts for glide/slope (optional)
# ---------------------------------------------------------
st.subheader("Модели: плъзгаемост и наклон (диагностика)")

colm_a, colm_b = st.columns(2)

with colm_a:
    st.markdown("### Плъзгаемост (спускания под -5%)")
    if not train_glide.empty and glide_poly is not None:
        s_min = train_glide["slope_pct"].min()
        s_max = train_glide["slope_pct"].max()
        s_grid = np.linspace(s_min, s_max, 200)
        df_glide_curve = pd.DataFrame({"slope_pct": s_grid, "v_model": glide_poly(s_grid)})

        chart_points = alt.Chart(train_glide).mark_circle(size=30).encode(
            x=alt.X("slope_pct", title="Наклон [%]"),
            y=alt.Y("v_kmh", title="Скорост [km/h]"),
            color="activity:N"
        )
        chart_curve = alt.Chart(df_glide_curve).mark_line().encode(x="slope_pct", y="v_model")
        st.altair_chart(chart_points + chart_curve, use_container_width=True)
        st.caption(f"v(s) = {poly_to_str(glide_poly, var='s')}")
    else:
        st.info("Няма достатъчно данни за glide модел.")

with colm_b:
    st.markdown("### Наклон: F(slope)")
    if not slope_train.empty and slope_poly is not None:
        s_min2 = slope_train["slope_pct"].min()
        s_max2 = slope_train["slope_pct"].max()
        s_grid2 = np.linspace(s_min2, s_max2, 200)
        df_slope_curve = pd.DataFrame({"slope_pct": s_grid2, "F_model": slope_poly(s_grid2)})

        chart_points2 = alt.Chart(slope_train).mark_circle(size=30).encode(
            x=alt.X("slope_pct", title="Наклон [%]"),
            y=alt.Y("F", title="F = V_flat_ref / v_glide"),
        )
        chart_curve2 = alt.Chart(df_slope_curve).mark_line().encode(x="slope_pct", y="F_model")
        st.altair_chart(chart_points2 + chart_curve2, use_container_width=True)
        st.caption(f"F(s) = {poly_to_str(slope_poly, var='s')}")
    else:
        st.info("Няма достатъчно данни за slope модел.")

# ---------------------------------------------------------
# ZONES TABLES (4)
# ---------------------------------------------------------
st.subheader("Зони (4 таблици): Без CS / С CS  ×  Схема A / Схема B")

# Create zones for both speeds
seg_no_cs = assign_speed_zones(seg_slope, V_crit, speed_col="v_flat_eq")
seg_with_cs = assign_speed_zones(seg_slope_cs, V_crit, speed_col="v_flat_eq_cs")

# Labels
hr_label = f"HR_lagged (+{hr_lag_s}s)" if hr_col_used == "hr_aligned" else "HR_raw (без lag)"

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"### Без CS (speed=v_flat_eq) — {hr_label}")
    st.markdown(f"**Схема A (paired) — {hr_label}**")
    st.dataframe(build_zone_table(seg_no_cs, speed_col="v_flat_eq", scheme="A", hr_col=hr_col_used), use_container_width=True)
    st.markdown(f"**Схема B (count-sorted) — {hr_label}**")
    st.dataframe(build_zone_table(seg_no_cs, speed_col="v_flat_eq", scheme="B", hr_col=hr_col_used), use_container_width=True)

with c2:
    st.markdown(f"### С CS (speed=v_flat_eq_cs) — {hr_label}")
    st.markdown(f"**Схема A (paired) — {hr_label}**")
    st.dataframe(build_zone_table(seg_with_cs, speed_col="v_flat_eq_cs", scheme="A", hr_col=hr_col_used), use_container_width=True)
    st.markdown(f"**Схема B (count-sorted) — {hr_label}**")
    st.dataframe(build_zone_table(seg_with_cs, speed_col="v_flat_eq_cs", scheme="B", hr_col=hr_col_used), use_container_width=True)

# Debug lag
dt_est_dbg = estimate_seg_dt_seconds(seg_slope_cs)
shift_n_dbg = int(round(hr_lag_s / max(dt_est_dbg, 1e-6)))
st.caption(f"DEBUG: HR колона={hr_col_used} | lag={hr_lag_s}s | dt≈{dt_est_dbg:.2f}s | shift_n={shift_n_dbg} сегмента")

# ---------------------------------------------------------
# Select activity
# ---------------------------------------------------------
st.subheader("Избор на активност (за графики/индекс)")
act_list = sorted(seg_slope_cs["activity"].unique())
act_selected = st.selectbox("Избери активност:", act_list)

g_act = seg_slope_cs[seg_slope_cs["activity"] == act_selected].copy()
st.subheader("Динамика на скоростите (4 линии) за избраната активност")

needed_cols = ["time_s", "v_kmh", "v_glide", "v_flat_eq", "v_flat_eq_cs"]
missing = [c for c in needed_cols if c not in g_act.columns]

if missing:
    st.warning(f"Липсват колони за графиката: {missing}. Провери дали се пренасят през pipeline-а.")
else:
    df_speed = g_act[needed_cols].copy()

    # по желание: махни очевидни нули/спирания от визуализацията
    min_plot_speed = st.slider("Минимална скорост за графиката [km/h]", 0.0, 10.0, 0.0, 0.5)
    if min_plot_speed > 0:
        df_speed = df_speed[df_speed["v_kmh"] >= min_plot_speed].copy()

    df_long = df_speed.melt(id_vars=["time_s"], var_name="series", value_name="v")

    name_map = {
        "v_kmh": "V_real (raw)",
        "v_glide": "V_glide (mod 1)",
        "v_flat_eq": "V_flat_eq (mod 2)",
        "v_flat_eq_cs": "V_flat_eq_cs (mod 3 / CS)",
    }
    df_long["series"] = df_long["series"].map(name_map).fillna(df_long["series"])

    chart_speed = alt.Chart(df_long).mark_line().encode(
        x=alt.X("time_s:Q", title="Време [s]"),
        y=alt.Y("v:Q", title="Скорост [km/h]"),
        color=alt.Color("series:N", title="Серия"),
        tooltip=["time_s:Q", "series:N", "v:Q"],
    )
    st.altair_chart(chart_speed, use_container_width=True)


# ---------------------------------------------------------
# Global V=f(HR) models from ALL activities
# ---------------------------------------------------------
st.subheader("Глобални V=f(HR) модели (от всички активности) + индекс на умора за избраната")

deg = st.selectbox("Степен на регресията V=f(HR)", [1, 2], index=0)

# Build pooled zone points over ALL activities (CS speed)
rows_B = []
rows_A = []
for act, g in seg_slope_cs.groupby("activity"):
    g_local = g.copy()
    g_local_z = assign_speed_zones(g_local, V_crit, speed_col="v_flat_eq_cs").dropna(subset=["zone"]).copy()
    if g_local_z.empty:
        continue

    zpB = zone_points_count_sorted(seg_df=g_local, seg_zones=g_local_z, speed_col="v_flat_eq_cs", hr_col=hr_col_used)
    zpB["activity"] = act
    rows_B.append(zpB)

    zpA = zone_points_paired(seg_zones=g_local_z, speed_col="v_flat_eq_cs", hr_col=hr_col_used)
    zpA["activity"] = act
    rows_A.append(zpA)

zp_global_B = pd.concat(rows_B, ignore_index=True) if rows_B else pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n", "activity"])
zp_global_A = pd.concat(rows_A, ignore_index=True) if rows_A else pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n", "activity"])

colp1, colp2 = st.columns(2)
with colp1:
    st.markdown("### Pooled точки — Модел 1 (count-sorted)")
    st.dataframe(zp_global_B, use_container_width=True, height=260)
with colp2:
    st.markdown("### Pooled точки — Модел 2 (paired)")
    st.dataframe(zp_global_A, use_container_width=True, height=260)

poly_B, dfB_fit = fit_v_of_hr_global(zp_global_B, deg=deg)
poly_A, dfA_fit = fit_v_of_hr_global(zp_global_A, deg=deg)

if poly_B is None or poly_A is None:
    st.warning("Недостатъчно pooled точки за регресия. Добави още активности или ползвай deg=1.")
else:
    st.markdown("### Уравнения")
    st.write(f"Глобален Модел 1 (count-sorted):  V = {poly_B}")
    st.write(f"Глобален Модел 2 (paired):       V = {poly_A}")

    if g_act.empty:
        st.info("Няма сегменти за избраната активност.")
    else:
        # Индекс на умора за избраната активност спрямо глобалните модели
        fi_B = fatigue_index_series(
            g_act,
            poly_B,
            speed_real_col="v_flat_eq_cs",
            hr_input_col=hr_col_used,
            cs_value=CS,
        )
        fi_A = fatigue_index_series(
            g_act,
            poly_A,
            speed_real_col="v_flat_eq_cs",
            hr_input_col=hr_col_used,
            cs_value=CS,
        )

        st.markdown("### Индекс на умора (2 глобални модела) – динамика по време")
        df_plot = pd.concat([
            fi_B.assign(model="Global_count_sorted"),
            fi_A.assign(model="Global_paired"),
        ], ignore_index=True)

        chart_fi = alt.Chart(df_plot).mark_line().encode(
            x=alt.X("time_s:Q", title="Време [s]"),
            y=alt.Y(
                "fatigue_index:Q",
                title="Индекс на умора (km/h) = CS + (V_real - V_pred)",
            ),
            color=alt.Color("model:N", title="Модел"),
            tooltip=["time_s:Q", "model:N", "fatigue_index:Q", "v_real:Q", "v_pred:Q", "delta_v:Q", "hr_used:Q"]
        )
        st.altair_chart(chart_fi, use_container_width=True)

        with st.expander("ℹ️ Обяснение на индекса на умора"):
            st.markdown(
                """
Индексът на умора тук се дефинира като:

\\[
FI = CS + (V_{real} - V_{pred})
\\]

- `V_real` е реалната нормализирана скорост (`v_flat_eq_cs`) за сегмента.
- `V_pred` е скоростта, която глобалният модел V=f(HR) „очаква“ при същия HR.
- Ако `V_real = V_pred`, тогава `FI = CS`.
- Ако `V_real > V_pred` → бегачът/скиорът е по-свеж и ефективен → FI > CS.
- Ако `V_real < V_pred` → има повече умора/неефективност → FI < CS.

Тъй като FI е в km/h и е „закован“ около CS, можем да сравняваме умората между различни дни и трасета.
                """
            )

        # Scatter pooled + lines
        st.markdown("### Scatter (всички активности) + линии на глобалните регресии")
        hr_min = float(np.nanmin(pd.concat([dfB_fit["mean_hr"], dfA_fit["mean_hr"]], ignore_index=True)))
        hr_max = float(np.nanmax(pd.concat([dfB_fit["mean_hr"], dfA_fit["mean_hr"]], ignore_index=True)))
        hr_grid = np.linspace(hr_min, hr_max, 250)

        df_line_B = pd.DataFrame({"hr": hr_grid, "v": poly_B(hr_grid), "model": "Global_count_sorted"})
        df_line_A = pd.DataFrame({"hr": hr_grid, "v": poly_A(hr_grid), "model": "Global_paired"})

        pts_B = dfB_fit.rename(columns={"mean_hr": "hr", "mean_speed": "v"}).assign(model="Global_count_sorted")
        pts_A = dfA_fit.rename(columns={"mean_hr": "hr", "mean_speed": "v"}).assign(model="Global_paired")

        scatter = alt.Chart(pd.concat([pts_B, pts_A], ignore_index=True)).mark_circle(size=45).encode(
            x=alt.X("hr:Q", title=f"HR използван ({hr_col_used})"),
            y=alt.Y("v:Q", title="V по зони (v_flat_eq_cs) [km/h]"),
            color=alt.Color("model:N", title="Модел"),
            tooltip=["model:N", "activity:N", "zone:N", "hr:Q", "v:Q", "n:Q"]
        )

        lines = alt.Chart(pd.concat([df_line_B, df_line_A], ignore_index=True)).mark_line().encode(
            x="hr:Q", y="v:Q", color="model:N"
        )
        st.altair_chart(scatter + lines, use_container_width=True)

        # Export
        st.markdown("### Експорт (fatigue index)")
        st.download_button(
            "Свали fatigue index CSV",
            data=df_plot.to_csv(index=False).encode("utf-8"),
            file_name=f"fatigue_index_{act_selected}.csv".replace(" ", "_"),
            mime="text/csv"
        )
