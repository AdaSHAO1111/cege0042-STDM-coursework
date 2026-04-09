"""
Random Forest regression baseline for one-hour-ahead net_flow prediction.

Trains a single global model across all stations, using the feature set from
src/features.py (temporal + autoregressive lags + spatial-neighbour lag).

XGBoost is used if available (faster, usually marginally better); otherwise
falls back to scikit-learn RandomForestRegressor.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, TARGET_COL

# ── Configuration ────────────────────────────────────────────────────────────

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=42,
)

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    n_jobs=-1,
    random_state=42,
)


# ── Model selection ──────────────────────────────────────────────────────────

def _get_model():
    """Return (model_instance, model_name)."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(**XGB_PARAMS), "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**RF_PARAMS), "RandomForest"


# ── Train & predict ──────────────────────────────────────────────────────────

def run_rf(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, object, str]:
    """
    Train a global tree-based model and return test-set predictions.

    Returns:
        preds_df:  DataFrame with station_id, datetime_hour, actual, predicted
        model:     fitted model object (for feature importance)
        model_name: "XGBoost" or "RandomForest"
    """
    train = feat_df[feat_df["split"] == "train"]
    test = feat_df[feat_df["split"] == "test"]

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    model, model_name = _get_model()
    print(f"  Training {model_name} on {len(train):,} rows, "
          f"{len(FEATURE_COLS)} features ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    y_pred = model.predict(X_test)

    preds = test[["station_id", "datetime_hour"]].copy()
    preds["actual"] = y_test
    preds["predicted"] = y_pred

    return preds, model, model_name
