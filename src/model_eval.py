"""
Evaluation utilities: metrics, comparison tables, and report-quality figures.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS
from src.model_sarima import REPRESENTATIVE_STATIONS

# ── Paths ────────────────────────────────────────────────────────────────────

FIG_DIR = "outputs/figures"
TBL_DIR = "outputs/tables"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Metrics ──────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Per-station metrics ──────────────────────────────────────────────────────

def station_metrics(preds: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Compute RMSE and MAE per station for a predictions DataFrame."""
    rows = []
    for sid, grp in preds.groupby("station_id"):
        rows.append({
            "station_id": sid,
            "model": model_name,
            "rmse": rmse(grp["actual"], grp["predicted"]),
            "mae": mae(grp["actual"], grp["predicted"]),
            "n_test": len(grp),
        })
    return pd.DataFrame(rows)


# ── Comparison table ─────────────────────────────────────────────────────────

def overall_comparison(
    sarima_preds: pd.DataFrame,
    rf_preds: pd.DataFrame,
    rf_model_name: str,
) -> pd.DataFrame:
    """
    Produce a comparison table with overall and per-station-subset metrics.

    The SARIMA results cover only the representative subset, so we compare
    both models on exactly the same station×hour pairs.
    """
    sarima_sids = set(sarima_preds["station_id"])

    # Global RF metrics (all stations)
    rf_all_rmse = rmse(rf_preds["actual"], rf_preds["predicted"])
    rf_all_mae = mae(rf_preds["actual"], rf_preds["predicted"])

    # Restrict RF to SARIMA stations for fair comparison
    rf_sub = rf_preds[rf_preds["station_id"].isin(sarima_sids)]

    rows = [
        {"scope": f"{rf_model_name} — all stations",
         "rmse": rf_all_rmse, "mae": rf_all_mae,
         "n_stations": rf_preds["station_id"].nunique(),
         "n_rows": len(rf_preds)},
        {"scope": f"{rf_model_name} — SARIMA subset",
         "rmse": rmse(rf_sub["actual"], rf_sub["predicted"]),
         "mae": mae(rf_sub["actual"], rf_sub["predicted"]),
         "n_stations": rf_sub["station_id"].nunique(),
         "n_rows": len(rf_sub)},
        {"scope": "SARIMA — SARIMA subset",
         "rmse": rmse(sarima_preds["actual"], sarima_preds["predicted"]),
         "mae": mae(sarima_preds["actual"], sarima_preds["predicted"]),
         "n_stations": sarima_preds["station_id"].nunique(),
         "n_rows": len(sarima_preds)},
    ]
    return pd.DataFrame(rows)


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_model_comparison_bar(
    comp: pd.DataFrame,
    fig_dir: str = FIG_DIR,
) -> str:
    """Grouped bar chart of RMSE and MAE by model/scope."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    x = np.arange(len(comp))
    labels = comp["scope"].values

    for i, metric in enumerate(["rmse", "mae"]):
        ax = axes[i]
        bars = ax.barh(x, comp[metric], color=["#3498db", "#2ecc71", "#e74c3c"])
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"Test-set {metric.upper()}")
        ax.grid(alpha=0.2, axis="x")
        for b in bars:
            ax.text(b.get_width() + 0.02, b.get_y() + b.get_height()/2,
                    f"{b.get_width():.3f}", va="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(fig_dir, "model_comparison_bar.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_station_rmse_comparison(
    sarima_metrics: pd.DataFrame,
    rf_metrics: pd.DataFrame,
    panel: pd.DataFrame,
    fig_dir: str = FIG_DIR,
) -> str:
    """
    Scatter: SARIMA RMSE vs RF RMSE for the representative stations.
    Points above the diagonal → RF is better.
    """
    merged = sarima_metrics.merge(
        rf_metrics, on="station_id", suffixes=("_sarima", "_rf")
    )
    names = (
        panel[["station_id", "station_name"]]
        .drop_duplicates()
        .set_index("station_id")["station_name"]
    )
    merged["label"] = merged["station_id"].map(
        lambda s: names.get(s, str(s)).split(",")[0]
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(merged["rmse_rf"], merged["rmse_sarima"],
               s=40, alpha=0.85, edgecolors="0.3", linewidths=0.4)
    lo = 0
    hi = max(merged["rmse_sarima"].max(), merged["rmse_rf"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=0.8, label="y = x")
    for _, r in merged.iterrows():
        ax.annotate(r["label"], (r["rmse_rf"], r["rmse_sarima"]),
                    fontsize=6, ha="left", va="bottom")
    ax.set(xlabel="RF / XGBoost RMSE", ylabel="SARIMA RMSE",
           title="Per-station RMSE: SARIMA vs tree model")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path = os.path.join(fig_dir, "station_rmse_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_forecast_timeseries(
    sarima_preds: pd.DataFrame,
    rf_preds: pd.DataFrame,
    station_id: int,
    panel: pd.DataFrame,
    rf_model_name: str,
    fig_dir: str = FIG_DIR,
) -> str:
    """Overlay actual vs predicted time series for one station."""
    s_sar = sarima_preds[sarima_preds["station_id"] == station_id].sort_values("datetime_hour")
    s_rf = rf_preds[rf_preds["station_id"] == station_id].sort_values("datetime_hour")
    sname = panel.loc[panel["station_id"] == station_id, "station_name"].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(s_sar["datetime_hour"], s_sar["actual"],
            lw=0.8, color="black", alpha=0.7, label="Actual")
    ax.plot(s_sar["datetime_hour"], s_sar["predicted"],
            lw=0.8, color="#e74c3c", alpha=0.8, label="SARIMA")
    ax.plot(s_rf["datetime_hour"], s_rf["predicted"],
            lw=0.8, color="#3498db", alpha=0.8, label=rf_model_name)
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set(xlabel="", ylabel="Net flow (dep − arr)",
           title=f"One-hour-ahead forecast — {sname}  (station {station_id})")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.15)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    path = os.path.join(fig_dir, f"forecast_ts_{station_id}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importance(
    model,
    model_name: str,
    fig_dir: str = FIG_DIR,
) -> str:
    """Bar chart of feature importances from the tree model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return ""

    order = np.argsort(imp)[::-1]
    names = [FEATURE_COLS[i] for i in order]
    vals = imp[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(names)), vals[::-1], color="#3498db", height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance (impurity decrease)")
    ax.set_title(f"{model_name} — Feature importance")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    path = os.path.join(fig_dir, "feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residual_by_hour(
    rf_preds: pd.DataFrame,
    rf_model_name: str,
    fig_dir: str = FIG_DIR,
) -> str:
    """Box-plot of prediction residuals by hour of day."""
    df = rf_preds.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"] = df["datetime_hour"].dt.hour
    df["residual"] = df["actual"] - df["predicted"]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    df.boxplot(column="residual", by="hour", ax=ax,
               showfliers=False, widths=0.6,
               boxprops=dict(color="#3498db"),
               medianprops=dict(color="#e74c3c"))
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set(xlabel="Hour of day",
           ylabel="Residual (actual − predicted)",
           title=f"{rf_model_name} residuals by hour of day")
    fig.suptitle("")  # remove pandas auto-title
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    path = os.path.join(fig_dir, "residual_by_hour.png")
    fig.savefig(path)
    plt.close(fig)
    return path
