# fatigue_model.py
import numpy as np
import pandas as pd


ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


def _robust_mean(x: pd.Series) -> float:
    """10% trimmed mean ако има достатъчно данни."""
    vals = x.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan
    vals.sort()
    if len(vals) >= 20:
        k = int(round(0.10 * len(vals)))
        if len(vals) - 2 * k > 0:
            vals = vals[k:len(vals) - k]
    return float(np.mean(vals))


def zone_points_paired(seg_zones: pd.DataFrame, speed_col: str, hr_col: str) -> pd.DataFrame:
    """
    Схема A (paired): взимаме средните стойности по зони от същите сегменти.
    """
    rows = []
    for z in ZONE_NAMES:
        g = seg_zones[seg_zones["zone"] == z]
        if g.empty:
            rows.append({"zone": z, "mean_speed": np.nan, "mean_hr": np.nan, "n": 0})
            continue
        rows.append({
            "zone": z,
            "mean_speed": float(g[speed_col].mean()) if speed_col in g.columns else np.nan,
            "mean_hr": _robust_mean(g[hr_col]) if hr_col in g.columns else np.nan,
            "n": int(len(g)),
        })
    return pd.DataFrame(rows)


def zone_points_count_sorted(seg_df: pd.DataFrame, seg_zones: pd.DataFrame, speed_col: str, hr_col: str) -> pd.DataFrame:
    """
    Схема B (count-sorted):
      - speed зоните определят броя сегменти във всяка зона (N_z)
      - HR се взима чрез сортиране по hr_col и разпределяне на първите N_Z1, после N_Z2, ...
    speed средната си остава по speed зоната (както в summarize_speed_zones).
    """
    # 1) counts по зони
    counts = seg_zones.groupby("zone").size().to_dict()

    # 2) mean speed по зони (от реалните сегменти в зоната)
    speed_means = seg_zones.groupby("zone")[speed_col].mean().to_dict() if speed_col in seg_zones.columns else {}

    # 3) HR сортиране
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
            rows.append({"zone": z, "mean_speed": float(speed_means.get(z, np.nan)), "mean_hr": np.nan, "n": n})
            continue
        end = min(start + n, len(df_hr))
        subset = df_hr.iloc[start:end]
        rows.append({
            "zone": z,
            "mean_speed": float(speed_means.get(z, np.nan)),
            "mean_hr": _robust_mean(subset[hr_col]),
            "n": n
        })
        start = end

    return pd.DataFrame(rows)


def fit_v_of_hr(zone_points: pd.DataFrame, deg: int = 1):
    """
    Фитва V = f(HR) като полином (deg=1 линейно, deg=2 квадратично).
    Връща np.poly1d или None.
    """
    df = zone_points.copy()
    df = df.dropna(subset=["mean_speed", "mean_hr"])
    # трябва ни поне deg+1 точки
    if len(df) <= deg:
        return None, df

    x = df["mean_hr"].to_numpy(dtype=float)     # HR
    y = df["mean_speed"].to_numpy(dtype=float)  # V

    coeffs = np.polyfit(x, y, deg)
    return np.poly1d(coeffs), df


def predict_v(poly: np.poly1d, hr: np.ndarray) -> np.ndarray:
    hr = np.asarray(hr, dtype=float)
    return poly(hr)


def fatigue_index_series(seg_df: pd.DataFrame, poly: np.poly1d, speed_real_col: str, hr_input_col: str) -> pd.DataFrame:
    """
    За всеки сегмент:
      V_pred = f(HR_input)
      FI = 2*V_real - V_pred   (твоята методика)
    """
    df = seg_df.copy()
    df = df.sort_values(["activity", "t_start"]).reset_index(drop=True)

    hr = df[hr_input_col].to_numpy(dtype=float)
    v_real = df[speed_real_col].to_numpy(dtype=float)

    v_pred = predict_v(poly, hr)
    delta = v_real - v_pred
    fi = v_real + delta  # = 2*v_real - v_pred

    out = pd.DataFrame({
        "activity": df["activity"].values,
        "time_s": df["time_s"].values if "time_s" in df.columns else np.arange(len(df), dtype=float),
        "seg_idx": df["seg_idx"].values if "seg_idx" in df.columns else np.arange(len(df)),
        "hr_used": hr,
        "v_real": v_real,
        "v_pred": v_pred,
        "delta_v": delta,
        "fatigue_index": fi,
    })

    return out
