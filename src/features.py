"""
Feature engineering for one-hour-ahead net_flow prediction.

Produces a modelling table with:
  - temporal features: hour, dow, is_weekend
  - autoregressive lags: net_flow at t-1 … t-3, t-24, t-48
  - departure / arrival lags at t-1, t-24
  - spatial-neighbour mean lag (mean net_flow of K nearest stations at t-1)
  - time-based train / test split flag
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Constants ────────────────────────────────────────────────────────────────

LAG_HOURS = [1, 2, 3, 24, 48]
NEIGHBOUR_K = 5
TRAIN_END = "2018-02-20 23:00:00"  # first 7 weeks inclusive


# ── Spatial neighbours ───────────────────────────────────────────────────────

def _build_neighbour_map(
    panel: pd.DataFrame, k: int = NEIGHBOUR_K
) -> dict[int, list[int]]:
    """
    For each geocoded station, find its k nearest neighbours (by Euclidean
    distance on lat/lon).  Returns {station_id: [neighbour_ids]}.
    """
    geo = (
        panel[["station_id", "lat", "lon"]]
        .drop_duplicates("station_id")
        .dropna(subset=["lat", "lon"])
        .reset_index(drop=True)
    )
    coords = geo[["lat", "lon"]].values
    tree = cKDTree(coords)
    # k+1 because the station itself is the closest
    _, idx = tree.query(coords, k=k + 1)

    neighbour_map: dict[int, list[int]] = {}
    ids = geo["station_id"].values
    for i, sid in enumerate(ids):
        neighbour_map[sid] = [int(ids[j]) for j in idx[i, 1:]]
    return neighbour_map


# ── Feature builder ──────────────────────────────────────────────────────────

def build_features(
    panel: pd.DataFrame,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """
    Build the full modelling table from the station-hour panel.

    Returns a DataFrame with one row per (station_id, datetime_hour),
    dropping rows where any lag feature is unavailable (first 48 hours).
    """
    df = panel.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df.sort_values(["station_id", "datetime_hour"], inplace=True)

    # ── Temporal features ──
    df["hour"] = df["datetime_hour"].dt.hour
    df["dow"] = df["datetime_hour"].dt.dayofweek          # 0=Mon … 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # ── Autoregressive lags (within each station) ──
    grp = df.groupby("station_id")
    for h in LAG_HOURS:
        df[f"nf_lag_{h}"] = grp["net_flow"].shift(h)
    df["dep_lag_1"] = grp["departures"].shift(1)
    df["arr_lag_1"] = grp["arrivals"].shift(1)
    df["dep_lag_24"] = grp["departures"].shift(24)
    df["arr_lag_24"] = grp["arrivals"].shift(24)

    # ── Spatial-neighbour mean net_flow at t-1 ──
    neighbour_map = _build_neighbour_map(panel)

    # Vectorised: pivot nf_lag_1 to a station×time matrix, then average
    # the K nearest neighbours for each station.
    nf1_pivot = df.pivot(
        index="datetime_hour", columns="station_id", values="nf_lag_1"
    )
    nb_mean_parts = []
    for sid, nbs in neighbour_map.items():
        present_nbs = [n for n in nbs if n in nf1_pivot.columns]
        if present_nbs:
            s = nf1_pivot[present_nbs].mean(axis=1).rename(sid)
        else:
            s = pd.Series(np.nan, index=nf1_pivot.index, name=sid)
        nb_mean_parts.append(s)
    nb_mean_df = pd.concat(nb_mean_parts, axis=1)  # datetime_hour × station_id

    # Melt back and merge
    nb_long = (
        nb_mean_df.stack()
        .rename("neighbour_nf_lag1_mean")
        .reset_index()
        .rename(columns={"level_1": "station_id"})
    )
    # For stations without geocoords, the column will remain NaN after merge
    df = df.merge(nb_long, on=["datetime_hour", "station_id"], how="left")

    # ── Drop rows with missing lag features (first 48h of each station) ──
    temporal_lag_cols = [f"nf_lag_{h}" for h in LAG_HOURS] + [
        "dep_lag_1", "arr_lag_1", "dep_lag_24", "arr_lag_24",
    ]
    df.dropna(subset=temporal_lag_cols, inplace=True)

    # Non-geocoded stations have no neighbour feature; fill with 0
    df["neighbour_nf_lag1_mean"] = df["neighbour_nf_lag1_mean"].fillna(0.0)

    # ── Train / test flag ──
    df["split"] = np.where(
        df["datetime_hour"] <= pd.Timestamp(train_end), "train", "test"
    )

    return df.reset_index(drop=True)


# ── Convenience accessors ────────────────────────────────────────────────────

FEATURE_COLS = (
    ["hour", "dow", "is_weekend"]
    + [f"nf_lag_{h}" for h in LAG_HOURS]
    + ["dep_lag_1", "arr_lag_1", "dep_lag_24", "arr_lag_24"]
    + ["neighbour_nf_lag1_mean"]
)

TARGET_COL = "net_flow"
