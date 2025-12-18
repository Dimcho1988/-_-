# fatigue_model.py
from __future__ import annotations

import numpy as np
import pandas as pd

ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def zone_points_paired(
    seg_zones: pd.DataFrame,
    speed_col: str = "v_flat_eq_cs",
    hr_col: str = "hr_aligned",
) -> pd.DataFrame:
    if seg_zones is None or seg_zones.empty:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    df = seg_zones.copy()
    df = _ensure_cols(df, ["zone", speed_col, hr_col])
    df = df.dropna(subset=["zone"])

    if df.empty:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    out = (
        df.groupby("zone")
        .agg(
            mean_speed=(speed_col, "mean"),
            mean_hr=(hr_col, "mean"),
            n=("zone", "count"),
        )
        .reset_index()
    )

    out["zone"] = pd.Categorical(out["zone"], categories=ZONE_NAMES, ordered=True)
    out = out.sort_values("zone").reset_index(drop=True)
    out["zone"] = out["zone"].astype(str)

    return out


def zone_points_count_sorted(
    seg_df: pd.DataFrame,
    seg_zones: pd.DataFrame,
    speed_col: str = "v_flat_eq_cs",
    hr_col: str = "hr_aligned",
) -> pd.DataFrame:
    if seg_df is None or seg_zones is None:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    z = seg_zones.copy()
    z = _ensure_cols(z, ["zone", speed_col])
    z = z.dropna(subset=["zone"])

    if z.empty:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    speed_summary = (
        z.groupby("zone")
        .agg(
            mean_speed=(speed_col, "mean"),
            n=("zone", "count"),
        )
        .reset_index()
    )

    df_hr = seg_df.copy()
    df_hr = _ensure_cols(df_hr, [hr_col])

    if "speed_spike" in df_hr.columns:
        df_hr = df_hr[~df_hr["speed_spike"].fillna(False)]

    df_hr = df_hr.dropna(subset=[hr_col]).sort_values(hr_col).reset_index(drop=True)

    rows = []
    start = 0
    for zname in ZONE_NAMES:
        n = int(speed_summary.loc[speed_summary["zone"] == zname, "n"].sum())
        if n <= 0:
            rows.append({"zone": zname, "mean_hr": np.nan})
            continue

        end = min(start + n, len(df_hr))
        chunk = df_hr.iloc[start:end]
        rows.append({"zone": zname, "mean_hr": chunk[hr_col].mean()})
        start = end

    hr_summary = pd.DataFrame(rows)

    out = speed_summary.merge(hr_summary, on="zone", how="right")
    out["zone"] = pd.Categorical(out["zone"], categories=ZONE_NAMES, ordered=True)
    out = out.sort_values("zone").reset_index(drop=True)
    out["zone"] = out["zone"].astype(str)

    return out


def fit_v_of_hr_global(zp_global: pd.DataFrame, deg: int = 1):
    if zp_global is None or zp_global.empty:
        return None, zp_global

    df = zp_global.copy()
    df = _ensure_cols(df, ["mean_hr", "mean_speed", "n"])
    df = df.dropna(subset=["mean_hr", "mean_speed"])

    if len(df) <= deg:
        return None, df

    x = df["mean_hr"].to_numpy(dtype=float)
    y = df["mean_speed"].to_numpy(dtype=float)
    w = df["n"].fillna(1.0).to_numpy(dtype=float)

    try:
        poly = np.poly1d(np.polyfit(x, y, deg, w=w))
        return poly, df
    except Exception:
        return None, df


def fatigue_index_series(
    seg_df: pd.DataFrame,
    poly_v_of_hr: np.poly1d,
    speed_real_col: str = "v_flat_eq_cs",
    hr_input_col: str = "hr_aligned",
) -> pd.DataFrame:
    if seg_df is None or seg_df.empty or poly_v_of_hr is None:
        return pd.DataFrame()

    df = seg_df.copy()
    df = _ensure_cols(df, ["time_s", speed_real_col, hr_input_col, "dt_s"])

    if df["time_s"].isna().all():
        dt = df["dt_s"].fillna(0.0).to_numpy()
        df["time_s"] = np.cumsum(dt) - dt

    hr = df[hr_input_col].to_numpy(dtype=float)
    v_real = df[speed_real_col].to_numpy(dtype=float)

    v_pred = np.full_like(v_real, np.nan)
    mask = ~np.isnan(hr)
    v_pred[mask] = poly_v_of_hr(hr[mask])

    delta_v = v_real - v_pred
    fatigue_index = 2.0 * v_real - v_pred

    return pd.DataFrame(
        {
            "time_s": df["time_s"],
            "hr_used": hr,
            "v_real": v_real,
            "v_pred": v_pred,
            "delta_v": delta_v,
            "fatigue_index": fatigue_index,
        }
    )
