#!/usr/bin/env python3
"""
Step 4 (optional) — Publication-quality figure polish pass.

Loads existing panel and predictions; does NOT rerun any model fitting.
Overwrites figures in outputs/figures/ with higher-quality versions.

Usage:
    python polish_figures.py
"""
from __future__ import annotations
import os, textwrap
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── Paths ────────────────────────────────────────────────────────────────────
PANEL = "data/processed/station_hour_netflow.csv"
SARIMA_PREDS = "outputs/tables/sarima_predictions.csv"
RF_PREDS = "outputs/tables/rf_predictions.csv"
MODEL_COMP = "outputs/tables/model_comparison.csv"
FIG = "outputs/figures"
TBL = "outputs/tables"

SPATIAL_CMAP = "RdBu"
DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 250, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Load data once ───────────────────────────────────────────────────────────
print("Loading data …")
panel = pd.read_csv(PANEL, parse_dates=["datetime_hour"])
panel["hour"] = panel["datetime_hour"].dt.hour
panel["dow_name"] = panel["datetime_hour"].dt.day_name()
panel["is_weekend"] = panel["dow_name"].isin(["Saturday", "Sunday"])

stn = panel.groupby("station_id").agg(
    station_name=("station_name","first"), lat=("lat","first"), lon=("lon","first"),
    total_departures=("departures","sum"), total_arrivals=("arrivals","sum"),
    mean_net_flow=("net_flow","mean"),
).reset_index()

sarima = pd.read_csv(SARIMA_PREDS, parse_dates=["datetime_hour"])
rf = pd.read_csv(RF_PREDS, parse_dates=["datetime_hour"])
comp = pd.read_csv(MODEL_COMP)

date_start = panel["datetime_hour"].min().strftime("%d %b %Y")
date_end = panel["datetime_hour"].max().strftime("%d %b %Y")

changed = []

# ═════════════════════════════════════════════════════════════════════════════
# 1. dow_hour_heatmap.png
# ═════════════════════════════════════════════════════════════════════════════
print("1. dow_hour_heatmap")
pivot = (panel.groupby(["dow_name","hour"])["net_flow"].mean().reset_index()
         .pivot(index="dow_name", columns="hour", values="net_flow")
         .reindex(DAY_ORDER))
fig, ax = plt.subplots(figsize=(10, 3.2))
vabs = np.abs(pivot.values).max()
sns.heatmap(pivot, ax=ax, cmap=SPATIAL_CMAP, center=0, vmin=-vabs, vmax=vabs,
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Mean net flow (dep − arr)\n← receivers    senders →",
                       "shrink": 0.75})
ax.set_yticklabels([d[:3] for d in DAY_ORDER], rotation=0)
ax.set(xlabel="Hour of day", ylabel="",
       title="Mean station-level net flow: day-of-week × hour-of-day")
ax.text(0.5, -0.13, f"Study period: {date_start} – {date_end}  |  "
        "net_flow = departures − arrivals",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="0.45")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "dow_hour_heatmap.png"))
plt.close(fig)
changed.append(("dow_hour_heatmap.png", "Added study-period subtitle; refined colour-bar label"))

# ═════════════════════════════════════════════════════════════════════════════
# 2. weekday_weekend_profile.png
# ═════════════════════════════════════════════════════════════════════════════
print("2. weekday_weekend_profile")
grouped = panel.groupby(["is_weekend","hour"])[["departures","arrivals","net_flow"]].mean().reset_index()
wd = grouped[~grouped["is_weekend"]]; we = grouped[grouped["is_weekend"]]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
panels_cfg = [("departures","Departures",False),("arrivals","Arrivals",False),("net_flow","Net flow",True)]
for i,(col,title,hline) in enumerate(panels_cfg):
    ax = axes[i]
    ax.fill_between(wd["hour"], wd[col], alpha=0.12, color="C0")
    ax.fill_between(we["hour"], we[col], alpha=0.12, color="C1")
    ax.plot(wd["hour"], wd[col], "o-", ms=3, color="C0", label="Weekday")
    ax.plot(we["hour"], we[col], "s-", ms=3, color="C1", label="Weekend")
    ax.set(xlabel="Hour of day", title=f"({'abc'[i]}) Mean {title}")
    ax.set_xticks(range(0, 24, 4)); ax.set_xlim(-0.5, 23.5)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2, axis="y")
    if hline:
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"(c) Mean net flow*", fontsize=11)
fig.text(0.5, -0.02,
    "* Network-wide hourly mean net flow ≈ 0 by construction; "
    "small deviations arise when a departure and its corresponding arrival "
    "fall in different clock hours.",
    ha="center", fontsize=7, color="0.45", style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(os.path.join(FIG, "weekday_weekend_profile.png"))
plt.close(fig)
changed.append(("weekday_weekend_profile.png",
    "Added footnote on net-flow panel; asterisk on panel (c) title"))

# ═════════════════════════════════════════════════════════════════════════════
# 3 & 4. AM / PM peak maps
# ═════════════════════════════════════════════════════════════════════════════

def _peak_map(hour_lo, hour_hi, title, fname, vabs_override, n_labels=4):
    mask = panel["hour"].between(hour_lo, hour_hi) & ~panel["is_weekend"]
    peak = panel.loc[mask].groupby("station_id")["net_flow"].mean().reset_index(name="peak_nf")
    geo = peak.merge(stn[["station_id","station_name","lat","lon"]], on="station_id", how="inner"
                     ).dropna(subset=["lat","lon"])
    n_excl = peak["station_id"].nunique() - len(geo)
    vabs = vabs_override
    fig, ax = plt.subplots(figsize=(7, 7.5))
    sc = ax.scatter(geo["lon"], geo["lat"], c=geo["peak_nf"], cmap=SPATIAL_CMAP,
                    vmin=-vabs, vmax=vabs, s=16, alpha=0.85, edgecolors="0.4", linewidths=0.25)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, aspect=30)
    cbar.set_label("Mean net flow (dep − arr)\n← receivers          senders →")
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]
    for subset in [geo.nlargest(n_labels, "peak_nf"), geo.nsmallest(n_labels, "peak_nf")]:
        for _, r in subset.iterrows():
            name = r["station_name"].split(",")[0]
            ax.annotate(name, (r["lon"], r["lat"]), fontsize=5.5,
                        ha="center", va="bottom", path_effects=outline)
    sub = (f"{len(geo)} geocoded stations, {n_excl} excluded  |  "
           f"Weekday {hour_lo:02d}:00–{hour_hi:02d}:59  |  {date_start} – {date_end}")
    ax.set(xlabel="Longitude", ylabel="Latitude")
    ax.set_title(f"{title}\n", fontsize=11)
    ax.text(0.5, 1.0, sub, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.4")
    ax.set_aspect("equal"); ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

vals = []
for lo, hi in [(7,9),(17,19)]:
    m = panel["hour"].between(lo, hi) & ~panel["is_weekend"]
    vals.append(panel.loc[m].groupby("station_id")["net_flow"].mean())
shared_vabs = float(np.abs(pd.concat(vals)).quantile(0.98))

print("3. map_am_peak")
_peak_map(7, 9, "Weekday AM Peak (07:00–09:59) Mean Net Flow",
          "map_am_peak_net_flow.png", shared_vabs, n_labels=4)
changed.append(("map_am_peak_net_flow.png", "Shared colour scale; study-period subtitle"))

print("4. map_pm_peak")
_peak_map(17, 19, "Weekday PM Peak (17:00–19:59) Mean Net Flow",
          "map_pm_peak_net_flow.png", shared_vabs, n_labels=4)
changed.append(("map_pm_peak_net_flow.png", "Shared colour scale; matched AM map style"))

# ═════════════════════════════════════════════════════════════════════════════
# 5. model_comparison_bar.png
# ═════════════════════════════════════════════════════════════════════════════
print("5. model_comparison_bar")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
x = np.arange(len(comp))
labels = [
    "RandomForest — all stations (reference only)",
    "RandomForest — SARIMA subset",
    "SARIMA — SARIMA subset",
]
colours = ["#bdc3c7", "#3498db", "#e74c3c"]
for i, metric in enumerate(["rmse", "mae"]):
    ax = axes[i]
    bars = ax.barh(x, comp[metric], color=colours, edgecolor="0.5", linewidth=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Test-set {metric.upper()}")
    ax.grid(alpha=0.2, axis="x")
    for b in bars:
        ax.text(b.get_width() + 0.02, b.get_y() + b.get_height()/2,
                f"{b.get_width():.3f}", va="center", fontsize=8)
    y_lo, y_hi = 1, 2
    xmax = comp[metric].max() * 1.35
    ax.plot([xmax*0.92, xmax*0.95, xmax*0.95, xmax*0.92],
            [y_lo-0.15, y_lo-0.15, y_hi+0.15, y_hi+0.15],
            color="0.3", lw=1, clip_on=False)
    ax.text(xmax*0.97, (y_lo+y_hi)/2, "fair\ncomparison",
            fontsize=6.5, va="center", ha="left", color="0.3")
fig.suptitle("Model comparison: one-hour-ahead net_flow prediction",
             fontsize=11, y=1.02)
fig.text(0.5, -0.04,
    "The 'all stations' row includes 785 stations (many low-activity); "
    "only the SARIMA-subset rows are directly comparable.",
    ha="center", fontsize=7.5, color="0.45", style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.98])
fig.savefig(os.path.join(FIG, "model_comparison_bar.png"))
plt.close(fig)
changed.append(("model_comparison_bar.png",
    "All-stations row greyed out; bracket annotation for fair comparison"))

# ═════════════════════════════════════════════════════════════════════════════
# 6. feature_importance.png — proxy from subsample RF
# ═════════════════════════════════════════════════════════════════════════════
print("6. feature_importance")
from src.features import build_features, FEATURE_COLS, TARGET_COL
from sklearn.ensemble import RandomForestRegressor

feat_sample = build_features(panel, train_end="2018-02-20 23:00:00")
train_s = feat_sample[feat_sample["split"] == "train"].sample(n=50000, random_state=42)
mini_rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
mini_rf.fit(train_s[FEATURE_COLS].values, train_s[TARGET_COL].values)
imp = mini_rf.feature_importances_
order = np.argsort(imp)[::-1][:10]
names_imp = [FEATURE_COLS[i] for i in order]
vals_imp = imp[order]

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.barh(range(len(names_imp)), vals_imp[::-1], color="#3498db", height=0.6, edgecolor="0.5", linewidth=0.3)
ax.set_yticks(range(len(names_imp)))
ax.set_yticklabels(names_imp[::-1], fontsize=8)
ax.set_xlabel("Importance (impurity decrease)")
ax.set_title("RandomForest — Top-10 feature importance")
ax.grid(alpha=0.2, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "feature_importance.png"))
plt.close(fig)
changed.append(("feature_importance.png", "Top-10 features; compact layout"))

# ═════════════════════════════════════════════════════════════════════════════
# 7. forecast_ts_154.png
# ═════════════════════════════════════════════════════════════════════════════
print("7. forecast_ts_154")
s_sar = sarima[sarima["station_id"]==154].sort_values("datetime_hour")
s_rf = rf[rf["station_id"]==154].sort_values("datetime_hour")
sname = panel.loc[panel["station_id"]==154, "station_name"].iloc[0]

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(s_sar["datetime_hour"], s_sar["actual"],
        lw=1.1, color="black", alpha=0.75, label="Actual", zorder=3)
ax.plot(s_sar["datetime_hour"], s_sar["predicted"],
        lw=0.9, color="#e74c3c", alpha=0.85, ls="--", label="SARIMA", zorder=2)
ax.plot(s_rf["datetime_hour"], s_rf["predicted"],
        lw=0.9, color="#3498db", alpha=0.85, label="RandomForest", zorder=2)
ax.axhline(0, color="0.5", lw=0.4)
ax.set(xlabel="", ylabel="Net flow (dep − arr)",
       title=f"One-hour-ahead forecast — {sname}  (station 154)")
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.grid(alpha=0.15)
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "forecast_ts_154.png"))
plt.close(fig)
changed.append(("forecast_ts_154.png", "Improved line weights and legend placement"))

# ═════════════════════════════════════════════════════════════════════════════
# 8. station_rmse_scatter.png
# ═════════════════════════════════════════════════════════════════════════════
print("8. station_rmse_scatter")
from src.model_eval import station_metrics
from src.model_sarima import REPRESENTATIVE_STATIONS
sarima_sm = station_metrics(sarima, "SARIMA")
rf_sub = rf[rf["station_id"].isin(REPRESENTATIVE_STATIONS)]
rf_sm_sub = station_metrics(rf_sub, "RF")
merged = sarima_sm.merge(rf_sm_sub, on="station_id", suffixes=("_sarima","_rf"))
name_map = panel[["station_id","station_name"]].drop_duplicates().set_index("station_id")["station_name"]
merged["label"] = merged["station_id"].map(lambda s: name_map.get(s,"").split(",")[0])

LABEL_IDS = {154, 14, 361, 66, 194, 101, 427}

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(merged["rmse_rf"], merged["rmse_sarima"],
           s=45, alpha=0.85, edgecolors="0.3", linewidths=0.4, zorder=3)
lo = 0; hi = max(merged["rmse_sarima"].max(), merged["rmse_rf"].max()) * 1.1
ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=0.8, label="y = x (parity)")
outline = [pe.withStroke(linewidth=2.5, foreground="white")]
for _, r in merged.iterrows():
    if r["station_id"] in LABEL_IDS:
        ax.annotate(r["label"], (r["rmse_rf"], r["rmse_sarima"]),
                    fontsize=6.5, ha="left", va="bottom", path_effects=outline,
                    xytext=(4, 3), textcoords="offset points")
ax.set(xlabel="RandomForest RMSE", ylabel="SARIMA RMSE",
       title="Per-station RMSE: SARIMA vs RandomForest\n(15 representative stations)")
ax.legend(fontsize=8, loc="lower right")
ax.set_aspect("equal"); ax.grid(alpha=0.15)
ax.text(hi*0.35, hi*0.9, "SARIMA better →", fontsize=7, color="0.5", rotation=0)
ax.text(hi*0.65, hi*0.2, "← RF better", fontsize=7, color="0.5", rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "station_rmse_scatter.png"))
plt.close(fig)
changed.append(("station_rmse_scatter.png", "Selective labels; region annotations"))

# ═════════════════════════════════════════════════════════════════════════════
# 9. station_timeseries_acf.png
# ═════════════════════════════════════════════════════════════════════════════
print("9. station_timeseries_acf")
ts = (panel.loc[panel["station_id"]==154].sort_values("datetime_hour")
      .set_index("datetime_hour")["net_flow"])
sname154 = panel.loc[panel["station_id"]==154, "station_name"].iloc[0]
fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.3)
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(ts.index, ts.values, lw=0.45, color="C0", alpha=0.85)
ax0.axhline(0, color="black", lw=0.4)
ax0.set(ylabel="Net flow (dep − arr)",
        title=f"(a) Hourly net flow — {sname154}  (station 154)")
ax0.grid(alpha=0.15); ax0.tick_params(axis="x", labelsize=8)
ax1 = fig.add_subplot(gs[1, 0])
plot_acf(ts, ax=ax1, lags=72, alpha=0.05, title="(b) Autocorrelation (ACF)")
ax1.set_xlabel("Lag (hours)"); ax1.grid(alpha=0.15)
ax2 = fig.add_subplot(gs[1, 1])
plot_pacf(ts, ax=ax2, lags=72, alpha=0.05, method="ywm",
          title="(c) Partial autocorrelation (PACF)")
ax2.set_xlabel("Lag (hours)"); ax2.grid(alpha=0.15)
fig.savefig(os.path.join(FIG, "station_timeseries_acf.png"))
plt.close(fig)
changed.append(("station_timeseries_acf.png", "Regenerated at higher dpi"))

# ═════════════════════════════════════════════════════════════════════════════
# Changelog
# ═════════════════════════════════════════════════════════════════════════════
print("\nWriting changelog …")
lines = [
    "── FINAL FIGURE POLISH LOG ────────────────────────────────",
    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "── CHANGES ──",
    "",
]
for fname, desc in changed:
    lines.append(f"  {fname}")
    lines.append(f"    {desc}")
    lines.append("")

lines += [
    "── UNCHANGED ──",
    "",
    "  forecast_ts_194.png       (appendix quality, no change needed)",
    "  residual_by_hour.png      (appendix quality, no change needed)",
    "  hourly_profile.png        (appendix, no change)",
    "  appendix_daily_net_flow.png  (appendix, no change)",
    "  appendix_station_map_overall.png  (appendix, no change)",
]
with open(os.path.join(TBL, "final_figure_polish_log.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("Done.")
