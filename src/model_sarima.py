"""
SARIMA baseline for one-hour-ahead net_flow prediction.

Model: SARIMA(1,0,1)(1,0,1)[24]
  - Justified by ACF/PACF evidence: strong 24h periodicity, rapid PACF decay
    beyond lag 1-2 with secondary spike at lag 24.
  - Non-differenced (d=0, D=0) because net_flow is already stationary
    (mean ≈ 0 network-wide, bounded fluctuations per station).

Applied to a representative subset of ~15 stations spanning high / medium /
low volume to keep runtime manageable.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── Configuration ────────────────────────────────────────────────────────────

ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 0, 1, 24)

TRAIN_END = "2018-02-20 23:00:00"

REPRESENTATIVE_STATIONS = [
    154,   # Waterloo 3 — highest mean net flow (major sender)
    194,   # Hop Exchange — large negative net flow (major receiver)
    66,    # Holborn Circus — strong receiver
    14,    # Argyle Street — highest total volume
    101,   # Queen Street 1 — City hub
    251,   # Brushfield Street — Liverpool Street area
    191,   # Hyde Park Corner — leisure hub, near-zero net flow
    73,    # Old Street Station — tech-corridor station
    341,   # Craven Street — moderate volume, near-zero net flow
    132,   # Bethnal Green Road — east residential
    12,    # Malet Street — Bloomsbury / university area
    361,   # Waterloo 2 — strong sender
    427,   # Cheapside — City core
    732,   # Duke Street Hill — London Bridge
    215,   # Moorfields — Moorgate area
]


# ── Fit & forecast ───────────────────────────────────────────────────────────

def fit_sarima_station(
    ts_train: pd.Series,
    ts_test: pd.Series,
) -> pd.DataFrame:
    """
    Fit SARIMA on ts_train, then produce one-step-ahead forecasts over
    ts_test using iterative extension (no full refit each step).

    Returns a DataFrame with columns [datetime_hour, actual, predicted].
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            ts_train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)

        predictions = []
        for t_idx in range(len(ts_test)):
            yhat = fit.forecast(steps=1).iloc[0]
            predictions.append(yhat)
            # Extend with actual observed value for next one-step forecast
            fit = fit.append([ts_test.iloc[t_idx]], refit=False)

    return pd.DataFrame({
        "datetime_hour": ts_test.index,
        "actual": ts_test.values,
        "predicted": predictions,
    })


def run_sarima(
    panel: pd.DataFrame,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """
    Run SARIMA on the representative station subset.

    Returns a long DataFrame: station_id, datetime_hour, actual, predicted.
    """
    panel = panel.copy()
    panel["datetime_hour"] = pd.to_datetime(panel["datetime_hour"])

    results = []
    t0 = time.time()

    for sid in REPRESENTATIVE_STATIONS:
        sdf = (
            panel.loc[panel["station_id"] == sid]
            .sort_values("datetime_hour")
            .set_index("datetime_hour")["net_flow"]
        )
        train = sdf[sdf.index <= pd.Timestamp(train_end)]
        test = sdf[sdf.index > pd.Timestamp(train_end)]

        if len(test) == 0:
            continue

        sname = panel.loc[panel["station_id"] == sid, "station_name"].iloc[0]
        print(f"  SARIMA  station {sid:>4d}  ({sname}) ... ", end="", flush=True)

        sub_t0 = time.time()
        res = fit_sarima_station(train, test)
        res["station_id"] = sid
        results.append(res)
        elapsed = time.time() - sub_t0
        print(f"{elapsed:.1f}s")

    total = time.time() - t0
    print(f"  SARIMA total elapsed: {total:.0f}s")

    return pd.concat(results, ignore_index=True)
