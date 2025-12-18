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
    """
    "Scheme A (paired)" точки по зони:
      - mean_speed = средната скорост в зоната (от seg_zones)
      - mean_hr    = средния пулс в зоната (от seg_zones)
      - n          = брой сегменти в зоната (от seg_zones)
    """
    if seg_zones is None or seg_zones.empty:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    df = seg_zones.copy()
    df = _ensure_cols(df, ["zone", speed_col, hr_col])

    df = df.dropna(subset=["zone"])
    if df.empty:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    grp = (
        df.groupby("zone", dropna=True)
        .agg(
            mean_speed=(speed_col, "mean"),
            mean_hr=(hr_col, "mean"),
            n=("zone", "count"),
        )
        .reset_index()
    )

    # стабилен ред Z1..Z6
    grp["zone"] = pd.Categorical(grp["zone"], categories=ZONE_NAMES, ordered=True)
    grp = grp.sort_values("zone").reset_index(drop=True)
    grp["zone"] = grp["zone"].astype(str)

    return grp[["zone", "mean_speed", "mean_hr", "n"]]


def zone_points_count_sorted(
    seg_df: pd.DataFrame,
    seg_zones: pd.DataFrame,
    speed_col: str = "v_flat_eq_cs",
    hr_col: str = "hr_aligned",
) -> pd.DataFrame:
    """
    "Scheme B (count-sorted)" точки по зони:
      - mean_speed = средната скорост в зоната (от seg_zones)
      - mean_hr    = пулс по разпределение по count:
                    сортираме всички HR (валидни) и раздаваме по зони
                    според броя сегменти във всяка speed-зона.
      - n          = брой сегменти във зоната (count от seg_zones)
    """
    if seg_df is None or seg_df.empty or seg_zones is None or seg_zones.empty:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    z = seg_zones.copy()
    z = _ensure_cols(z, ["zone", speed_col])
    z = z.dropna(subset=["zone"])
    if z.empty:
        return pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n"])

    # mean_speed и counts от speed зоните
    speed_summary = (
        z.groupby("zone", dropna=True)
        .agg(
            mean_speed=(speed_col, "mean"),
            n=("zone", "count"),
        )
        .reset_index()
    )

    # подготвяме HR списък (по възможност без speed_spike)
    df_hr = seg_df.copy()
    df_hr = _ensure_cols(df_hr, [hr_col])

    if "speed_spike" in df_hr.columns:
        df_hr = df_hr[~df_hr["speed_spike"].fillna(False)]

    df_hr = df_hr.dropna(subset=[hr_col]).copy()
    if df_hr.empty:
        out = speed_summary.copy()
        out["mean_hr"] = np.nan
        out["zone"] = pd.Categorical(out["zone"], categories=ZONE_NAMES, ordered=True)
        out = out.sort_values("zone").reset_index(drop=True)
        out["zone"] = out["zone"].astype(str)
        return out[["zone", "mean_speed", "mean_hr", "n"]]

    df_hr = df_hr.sort_values(hr_col).reset_index(drop=True)

    # раздаване по зони според n
    counts = dict(zip(speed_summary["zone"], speed_summary["n"]))
    rows = []
    start = 0
    for zone in ZONE_NAMES:
        n_zone = int(counts.get(zone, 0))
        if n_zone <= 0:
            rows.append({"zone": zone, "mean_hr": np.nan})
            continue

        end = min(start + n_zone, len(df_hr))
        chunk = df_hr.iloc[start:end]
        mean_hr = float(chunk[hr_col].mean()) if not chunk.empty else np.nan
        rows.append({"zone": zone, "mean_hr": mean_hr})
        start = end

    hr_summary = pd.DataFrame(rows)

    out = speed_summary.merge(hr_summary, on="zone", how="right")
    out["mean_speed"] = out["mean_speed"].astype(float)

    out["zone"] = pd.Categorical(out["zone"], categories=ZONE_NAMES, ordered=True)
    out = out.sort_values("zone").reset_index(drop=True)
    out["zone"] = out["zone"].astype(str)

    # Ако някоя зона няма mean_speed (няма speed сегменти), оставяме NaN
    return out[["zone", "mean_speed", "mean_hr", "n"]]


def fit_v_of_hr_global(zp_global: pd.DataFrame, deg: int = 1):
    """
    Фитва глобален модел V=f(HR) върху pooled zone точки.
    Очаквани колони: mean_hr, mean_speed, n (n може да липсва).
    Връща: (poly1d или None, df_used)
    """
    if zp_global is None or zp_global.empty:
        return None, pd.DataFrame(columns=["zone", "mean_speed", "mean_hr", "n", "activity"])

    df = zp_global.copy()
    df = _ensure_cols(df, ["mean_hr", "mean_speed", "n"])
    df = df.dropna(subset=["mean_hr", "mean_speed"]).copy()

    if df.empty:
        return None, df

    x = df["mean_hr"].to_numpy(dtype=float)
    y = df["mean_speed"].to_numpy(dtype=float)

    # тежести по n (ако има)
    w = df["n"].fillna(1.0).to_numpy(dtype=float)
    w = np.clip(w, 1.0, np.inf)

    # минимални точки
    if len(x) <= deg:
        return None, df

    try:
        coeffs = np.polyfit(x, y, deg, w=w)
        poly = np.poly1d(coeffs)
        return poly, df
    except Exception:
        return None, df


def fatigue_index_series(
    seg_df: pd.DataFrame,
    poly_v_of_hr: np.poly1d,
    speed_real_col: str = "v_flat_eq_cs",
    hr_input_col: str = "hr_aligned",
) -> pd.DataFrame:
    """
    За всеки сегмент:
      v_pred = poly(HR)
      v_real = speed_real_col
      delta_v = v_real - v_pred
      fatigue_index = 2*v_real - v_pred   (както го визуализираш в app-а)

    Връща DF с колони:
      time_s, hr_used, v_real, v_pred, delta_v, fatigue_index
    """
    if seg_df is None or seg_df.empty or poly_v_of_hr is None:
        return pd.DataFrame(columns=["time_s", "hr_used", "v_real", "v_pred", "delta_v", "fatigue_index"])

    df = seg_df.copy()
    df = _ensure_cols(df, [speed_real_col, hr_input_col, "dt_s", "time_s"])

    # ако time_s липсва или е NaN, правим кумулативно време
    if df["time_s"].isna().all():
        dt = df["dt_s"].fillna(0.0).to_numpy(dtype=float)
        df["time_s"] = np.cumsum(dt) - dt

    hr = df[hr_input_col].to_numpy(dtype=float)
    v_real = df[speed_real_col].to_numpy(dtype=float)

    v_pred = np.full_like(v_real, np.nan, dtype=float)
    mask = ~np.isnan(hr)
    if mask.any():
        v_pred[mask] = poly_v_of_hr(hr[mask])

    delta_v = v_real - v_pred
    fatigue_index = 2.0 * v_real - v_pred

    out = pd.DataFrame(
        {
            "time_s": df["time_s"].to_numpy(dtype=float),
            "hr_used": hr,
            "v_real": v_real,
            "v_pred": v_pred,
            "delta_v": delta_v,
            "fatigue_index": fatigue_index,
        }
    )
    return out
