#!/usr/bin/env python3
"""
Step 3 — Modelling pipeline for one-hour-ahead station-level net-flow prediction.

Study period : 2018-01-03 to 2018-03-06 (63 days)
Train window : 2018-01-03 to 2018-02-20 (first 7 weeks)
Test window  : 2018-02-21 to 2018-03-06 (final 2 weeks)

Models:
  1. SARIMA(1,0,1)(1,0,1)[24] — fitted on 15 representative stations
  2. Global XGBoost / RandomForest — temporal + spatial lag features

Usage:
    python run_models.py
"""
from __future__ import annotations

import os

import pandas as pd

from src.features import build_features, FEATURE_COLS
from src.model_sarima import run_sarima, REPRESENTATIVE_STATIONS
from src.model_rf import run_rf
from src.model_eval import (
    station_metrics,
    overall_comparison,
    plot_model_comparison_bar,
    plot_station_rmse_comparison,
    plot_forecast_timeseries,
    plot_feature_importance,
    plot_residual_by_hour,
)

# ── Paths ────────────────────────────────────────────────────────────────────

PANEL_PATH = "data/processed/station_hour_netflow.csv"
FIG_DIR = "outputs/figures"
TBL_DIR = "outputs/tables"

TRAIN_END = "2018-02-20 23:00:00"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _station_win_summary(
    per_station: pd.DataFrame,
    rf_name: str,
    tbl_dir: str,
) -> None:
    """Write CSV and text summaries of which model wins per station."""
    rf_col = f"rmse_{rf_name.lower()}"
    per_station = per_station.copy()
    per_station["winner"] = per_station.apply(
        lambda r: rf_name if r[rf_col] < r["rmse_sarima"] else "SARIMA",
        axis=1,
    )

    csv_path = os.path.join(tbl_dir, "station_win_summary.csv")
    per_station.to_csv(csv_path, index=False)

    n_rf_wins = (per_station["winner"] == rf_name).sum()
    n_sarima_wins = (per_station["winner"] == "SARIMA").sum()
    n_total = len(per_station)

    lines = [
        "── STATION-LEVEL MODEL WINNER SUMMARY ──",
        "",
        f"Stations compared: {n_total}",
        f"{rf_name} wins  : {n_rf_wins}  ({100*n_rf_wins/n_total:.0f}%)",
        f"SARIMA wins      : {n_sarima_wins}  ({100*n_sarima_wins/n_total:.0f}%)",
        "",
        "Per-station breakdown:",
    ]
    for _, r in per_station.iterrows():
        lines.append(
            f"  {int(r['station_id']):>4d}  {r['station_name']:<42s}  "
            f"SARIMA={r['rmse_sarima']:.3f}  {rf_name}={r[rf_col]:.3f}  "
            f"→ {r['winner']}"
        )

    txt_path = os.path.join(tbl_dir, "station_win_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)

    # ── 1. Load panel ──
    print("\n══ Loading panel ══")
    panel = pd.read_csv(PANEL_PATH, parse_dates=["datetime_hour"])
    print(f"  {len(panel):,} rows, {panel['station_id'].nunique()} stations")
    print(f"  Hour range: {panel['datetime_hour'].min()} → {panel['datetime_hour'].max()}")
    print(f"  Train/test split at: {TRAIN_END}")

    # ── 2. Feature engineering ──
    print("\n══ Building features ══")
    feat = build_features(panel, train_end=TRAIN_END)
    print(f"  Modelling table: {len(feat):,} rows")
    print(f"  Features: {FEATURE_COLS}")
    train_n = (feat["split"] == "train").sum()
    test_n = (feat["split"] == "test").sum()
    print(f"  Train: {train_n:,}  |  Test: {test_n:,}")

    # ── 3. SARIMA baseline ──
    print("\n══ SARIMA baseline ══")
    sarima_preds = run_sarima(panel, train_end=TRAIN_END)
    sarima_preds.to_csv(
        os.path.join(TBL_DIR, "sarima_predictions.csv"), index=False
    )
    print(f"  Predictions saved: {len(sarima_preds):,} rows")

    # ── 4. Random Forest / XGBoost ──
    print("\n══ Tree model ══")
    rf_preds, rf_model, rf_name = run_rf(feat)
    rf_preds.to_csv(
        os.path.join(TBL_DIR, "rf_predictions.csv"), index=False
    )
    print(f"  {rf_name} predictions saved: {len(rf_preds):,} rows")

    # ── 5. Evaluation ──
    print("\n══ Evaluation ══")

    sarima_sm = station_metrics(sarima_preds, "SARIMA")
    rf_sm_sub = station_metrics(
        rf_preds[rf_preds["station_id"].isin(REPRESENTATIVE_STATIONS)],
        rf_name,
    )
    rf_sm_all = station_metrics(rf_preds, rf_name)

    comp = overall_comparison(sarima_preds, rf_preds, rf_name)
    comp.to_csv(os.path.join(TBL_DIR, "model_comparison.csv"), index=False)
    print("\n  Model comparison:")
    print(comp.to_string(index=False))

    names = (
        panel[["station_id", "station_name"]]
        .drop_duplicates()
        .set_index("station_id")["station_name"]
    )
    per_station = sarima_sm[["station_id", "rmse", "mae"]].merge(
        rf_sm_sub[["station_id", "rmse", "mae"]],
        on="station_id", suffixes=("_sarima", f"_{rf_name.lower()}"),
    )
    per_station.insert(
        1, "station_name",
        per_station["station_id"].map(lambda s: names.get(s, ""))
    )
    per_station.to_csv(
        os.path.join(TBL_DIR, "per_station_comparison.csv"), index=False
    )
    print("\n  Per-station comparison (SARIMA subset):")
    print(per_station.to_string(index=False))

    _station_win_summary(per_station, rf_name, TBL_DIR)
    print(f"\n  ✓ {TBL_DIR}/station_win_summary.csv")
    print(f"  ✓ {TBL_DIR}/station_win_summary.txt")

    # ── 6. Figures ──
    print("\n══ Generating figures ══")

    path = plot_model_comparison_bar(comp, fig_dir=FIG_DIR)
    print(f"  ✓ {path}")

    path = plot_station_rmse_comparison(
        sarima_sm, rf_sm_sub, panel, fig_dir=FIG_DIR
    )
    print(f"  ✓ {path}")

    for sid in [154, 194]:
        path = plot_forecast_timeseries(
            sarima_preds, rf_preds, sid, panel, rf_name, fig_dir=FIG_DIR
        )
        print(f"  ✓ {path}")

    path = plot_feature_importance(rf_model, rf_name, fig_dir=FIG_DIR)
    if path:
        print(f"  ✓ {path}")

    path = plot_residual_by_hour(rf_preds, rf_name, fig_dir=FIG_DIR)
    print(f"  ✓ {path}")

    # ── 7. Summary text ──
    train_start = panel["datetime_hour"].min().strftime("%Y-%m-%d")
    test_end = panel["datetime_hour"].max().strftime("%Y-%m-%d")
    train_end_date = pd.Timestamp(TRAIN_END).strftime("%Y-%m-%d")
    test_start_date = (pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")

    summary_lines = [
        "══ MODELLING RESULTS SUMMARY ══",
        "",
        "Target: one-hour-ahead station-level net_flow (dep − arr)",
        f"Study period: {train_start} to {test_end}",
        f"Train: {train_start} to {train_end_date} (7 weeks)  |  "
        f"Test: {test_start_date} to {test_end} (2 weeks)  |  {test_n:,} test rows",
        "",
        "Model 1: SARIMA(1,0,1)(1,0,1)[24]",
        f"  Stations: {len(REPRESENTATIVE_STATIONS)} representative",
        f"  Test RMSE: {comp.loc[comp['scope'].str.contains('SARIMA —'), 'rmse'].values[0]:.4f}",
        f"  Test MAE:  {comp.loc[comp['scope'].str.contains('SARIMA —'), 'mae'].values[0]:.4f}",
        "",
        f"Model 2: {rf_name} (global, all stations)",
        f"  Features: {len(FEATURE_COLS)} ({', '.join(FEATURE_COLS)})",
        f"  Test RMSE (all):    {comp.iloc[0]['rmse']:.4f}",
        f"  Test MAE  (all):    {comp.iloc[0]['mae']:.4f}",
        f"  Test RMSE (subset): {comp.iloc[1]['rmse']:.4f}",
        f"  Test MAE  (subset): {comp.iloc[1]['mae']:.4f}",
        "",
        "── LIKE-FOR-LIKE COMPARISON NOTE ──",
        "",
        "SARIMA is fitted on 15 representative stations only.",
        f"{rf_name} is trained globally on all stations.",
        "The 'SARIMA subset' row restricts the tree model to the",
        "same 15 stations for a fair apples-to-apples comparison.",
        f"The 'all stations' row shows {rf_name} on all {rf_preds['station_id'].nunique()} stations.",
    ]
    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(TBL_DIR, "modelling_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\n  ✓ {summary_path}")
    print(f"\n{summary_text}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
