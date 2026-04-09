"""
Exploratory Spatial-Temporal Data Analysis (ESDA) for the Santander Cycles
station-hour panel.

Outputs:
  outputs/figures/   – publication-quality figures
  outputs/tables/    – CSV ranking tables & interpretation notes
"""
from __future__ import annotations

import os
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── Plotting defaults ────────────────────────────────────────────────────────

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

FIG_DIR = "outputs/figures"
TBL_DIR = "outputs/tables"
PANEL_PATH = "data/processed/station_hour_netflow.csv"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Consistent diverging colour map across all spatial plots
SPATIAL_CMAP = "RdBu"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _add_time_cols(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"] = df["datetime_hour"].dt.hour
    df["dow_name"] = df["datetime_hour"].dt.day_name()
    df["is_weekend"] = df["dow_name"].isin(["Saturday", "Sunday"])
    return df


def station_summary(panel: pd.DataFrame) -> pd.DataFrame:
    agg = panel.groupby("station_id").agg(
        station_name=("station_name", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        total_departures=("departures", "sum"),
        total_arrivals=("arrivals", "sum"),
        mean_departures=("departures", "mean"),
        mean_arrivals=("arrivals", "mean"),
        mean_net_flow=("net_flow", "mean"),
        std_net_flow=("net_flow", "std"),
    ).reset_index()
    return agg


# ── 1. Polished temporal figures (kept) ──────────────────────────────────────

def plot_hourly_profile(df: pd.DataFrame, fig_dir: str = FIG_DIR) -> str:
    """Average departures, arrivals, and net_flow by hour of day."""
    hourly = df.groupby("hour")[["departures", "arrivals", "net_flow"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    ax.fill_between(hourly.index, hourly["departures"], alpha=0.15, color="C0")
    ax.fill_between(hourly.index, hourly["arrivals"], alpha=0.15, color="C1")
    ax.plot(hourly.index, hourly["departures"], "o-", ms=4,
            color="C0", label="Departures")
    ax.plot(hourly.index, hourly["arrivals"], "s-", ms=4,
            color="C1", label="Arrivals")
    ax.set(xlabel="Hour of day", ylabel="Mean count per station-hour",
           title="(a) Average hourly departures & arrivals")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    colours = ["#c0392b" if v < 0 else "#27ae60" for v in hourly["net_flow"]]
    ax.bar(hourly.index, hourly["net_flow"], color=colours, width=0.75)
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xlabel="Hour of day", ylabel="Mean net flow",
           title="(b) Mean net flow by hour")
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlim(-0.5, 23.5)
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    path = os.path.join(fig_dir, "hourly_profile.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_weekday_weekend_profile(df: pd.DataFrame, fig_dir: str = FIG_DIR) -> str:
    """Weekday vs weekend hourly departures, arrivals, net_flow."""
    grouped = (
        df.groupby(["is_weekend", "hour"])[["departures", "arrivals", "net_flow"]]
        .mean().reset_index()
    )
    wd = grouped[~grouped["is_weekend"]]
    we = grouped[grouped["is_weekend"]]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
    panels = [
        ("departures", "Departures", False),
        ("arrivals", "Arrivals", False),
        ("net_flow", "Net flow", True),
    ]
    for i, (col, title, hline) in enumerate(panels):
        ax = axes[i]
        ax.fill_between(wd["hour"], wd[col], alpha=0.12, color="C0")
        ax.fill_between(we["hour"], we[col], alpha=0.12, color="C1")
        ax.plot(wd["hour"], wd[col], "o-", ms=3, color="C0", label="Weekday")
        ax.plot(we["hour"], we[col], "s-", ms=3, color="C1", label="Weekend")
        ax.set(xlabel="Hour of day",
               title=f"({'abc'[i]}) Mean {title}")
        ax.set_xticks(range(0, 24, 4))
        ax.set_xlim(-0.5, 23.5)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.2, axis="y")
        if hline:
            ax.axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    path = os.path.join(fig_dir, "weekday_weekend_profile.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dow_hour_heatmap(df: pd.DataFrame, fig_dir: str = FIG_DIR) -> str:
    """Heatmap: mean net_flow by day-of-week × hour-of-day."""
    pivot = (
        df.groupby(["dow_name", "hour"])["net_flow"]
        .mean().reset_index()
        .pivot(index="dow_name", columns="hour", values="net_flow")
        .reindex(DAY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 3.2))
    vabs = np.abs(pivot.values).max()
    sns.heatmap(
        pivot, ax=ax, cmap=SPATIAL_CMAP, center=0,
        vmin=-vabs, vmax=vabs,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Mean net flow (dep − arr)", "shrink": 0.75},
    )
    ax.set_yticklabels([d[:3] for d in DAY_ORDER], rotation=0)
    ax.set(xlabel="Hour of day", ylabel="",
           title="Mean station-level net flow: day-of-week × hour-of-day")
    fig.tight_layout()
    path = os.path.join(fig_dir, "dow_hour_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── 2. Ranking tables ───────────────────────────────────────────────────────

def ranking_tables(stn: pd.DataFrame, tbl_dir: str = TBL_DIR) -> dict[str, pd.DataFrame]:
    cols_meta = ["station_id", "station_name", "lat", "lon"]
    rankings: dict[str, pd.DataFrame] = {}

    specs = [
        ("top15_highest_mean_net_flow", "mean_net_flow", False),
        ("top15_lowest_mean_net_flow",  "mean_net_flow", True),
        ("top15_total_departures",      "total_departures", False),
        ("top15_total_arrivals",        "total_arrivals", False),
    ]
    for name, col, ascending in specs:
        tbl = (
            stn.sort_values(col, ascending=ascending)
            .head(15)[cols_meta + [col]]
            .reset_index(drop=True)
        )
        tbl.index = tbl.index + 1
        tbl.index.name = "rank"
        tbl.to_csv(os.path.join(tbl_dir, f"{name}.csv"))
        rankings[name] = tbl
    return rankings


# ── 3. Spatial peak-flow maps ────────────────────────────────────────────────

def _label_extremes(ax, geo: pd.DataFrame, col: str, n: int = 5) -> None:
    """Annotate the n most positive and n most negative stations."""
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]
    for subset in [geo.nlargest(n, col), geo.nsmallest(n, col)]:
        for _, r in subset.iterrows():
            name = r["station_name"].split(",")[0]
            ax.annotate(
                name, (r["lon"], r["lat"]),
                fontsize=5.5, ha="center", va="bottom",
                path_effects=outline,
            )


def _plot_peak_map(
    df_time: pd.DataFrame,
    stn: pd.DataFrame,
    hour_lo: int,
    hour_hi: int,
    weekday_only: bool,
    title: str,
    fname: str,
    vabs_override: float | None = None,
    fig_dir: str = FIG_DIR,
) -> tuple[str, pd.DataFrame]:
    """
    Compute mean net_flow for hours [hour_lo, hour_hi] inclusive,
    merge geocoords, and produce a labelled scatter map.

    Returns (saved_path, geo_dataframe) so the caller can extract
    vabs for a shared colour scale.
    """
    mask = df_time["hour"].between(hour_lo, hour_hi)
    if weekday_only:
        mask &= ~df_time["is_weekend"]
    peak = (
        df_time.loc[mask]
        .groupby("station_id")["net_flow"]
        .mean()
        .reset_index(name="peak_net_flow")
    )
    geo = peak.merge(
        stn[["station_id", "station_name", "lat", "lon",
             "total_departures", "total_arrivals"]],
        on="station_id", how="inner",
    ).dropna(subset=["lat", "lon"])

    n_excluded = peak["station_id"].nunique() - len(geo)

    if vabs_override is not None:
        vabs = vabs_override
    else:
        vabs = np.abs(geo["peak_net_flow"]).quantile(0.98)

    fig, ax = plt.subplots(figsize=(7, 7.5))
    sc = ax.scatter(
        geo["lon"], geo["lat"],
        c=geo["peak_net_flow"], cmap=SPATIAL_CMAP,
        vmin=-vabs, vmax=vabs,
        s=16, alpha=0.85, edgecolors="0.4", linewidths=0.25,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, aspect=30)
    cbar.set_label("Mean net flow (dep − arr)\n"
                   "← net inflow (receivers)          net outflow (senders) →")

    _label_extremes(ax, geo, "peak_net_flow", n=5)

    subtitle = (f"{len(geo)} geocoded stations, {n_excluded} excluded  |  "
                f"Weekday {hour_lo:02d}:00–{hour_hi:02d}:59")
    ax.set(xlabel="Longitude", ylabel="Latitude")
    ax.set_title(f"{title}\n", fontsize=11)
    ax.text(0.5, 1.0, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="0.4")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path = os.path.join(fig_dir, fname)
    fig.savefig(path)
    plt.close(fig)
    return path, geo


def _compute_shared_vabs(
    df_time: pd.DataFrame,
    stn: pd.DataFrame,
    windows: list[tuple[int, int]],
) -> float:
    """Compute a single symmetric colour limit across multiple peak windows."""
    vals = []
    for hour_lo, hour_hi in windows:
        mask = df_time["hour"].between(hour_lo, hour_hi) & ~df_time["is_weekend"]
        peak = df_time.loc[mask].groupby("station_id")["net_flow"].mean()
        vals.append(peak)
    combined = pd.concat(vals)
    return float(np.abs(combined).quantile(0.98))


def plot_am_peak_map(df_time: pd.DataFrame, stn: pd.DataFrame,
                     vabs: float | None = None,
                     fig_dir: str = FIG_DIR) -> str:
    path, _ = _plot_peak_map(
        df_time, stn, 7, 9, weekday_only=True,
        title="Weekday AM Peak (07:00–09:59) Mean Net Flow",
        fname="map_am_peak_net_flow.png",
        vabs_override=vabs,
        fig_dir=fig_dir,
    )
    return path


def plot_pm_peak_map(df_time: pd.DataFrame, stn: pd.DataFrame,
                     vabs: float | None = None,
                     fig_dir: str = FIG_DIR) -> str:
    path, _ = _plot_peak_map(
        df_time, stn, 17, 19, weekday_only=True,
        title="Weekday PM Peak (17:00–19:59) Mean Net Flow",
        fname="map_pm_peak_net_flow.png",
        vabs_override=vabs,
        fig_dir=fig_dir,
    )
    return path


# ── 4. Time-series diagnostic for one station ───────────────────────────────

DIAGNOSTIC_STATION_ID = 154   # Waterloo Station 3 — see selection rationale

STATION_SELECTION_RATIONALE = (
    "Waterloo Station 3 (id 154) was selected because it is the second-"
    "highest-volume station (12,901 total trips) and has the largest "
    "positive mean net flow (+0.47 dep/hr), meaning it is a major net "
    "sender (net outflow) — a commuter rail interchange where riders "
    "pick up bikes in the morning. Its high variability (std ≈ 4.3) "
    "makes it a good stress-test for forecasting models."
)


def plot_station_timeseries(panel: pd.DataFrame, fig_dir: str = FIG_DIR) -> str:
    """Hourly net_flow time series + ACF + PACF for the diagnostic station."""
    ts = (
        panel.loc[panel["station_id"] == DIAGNOSTIC_STATION_ID]
        .sort_values("datetime_hour")
        .set_index("datetime_hour")["net_flow"]
    )
    sname = panel.loc[
        panel["station_id"] == DIAGNOSTIC_STATION_ID, "station_name"
    ].iloc[0]

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.3)

    # (a) Full time series
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(ts.index, ts.values, lw=0.45, color="C0", alpha=0.85)
    ax0.axhline(0, color="black", lw=0.4)
    ax0.set(ylabel="Net flow (dep − arr)",
            title=f"(a) Hourly net flow — {sname}  (station {DIAGNOSTIC_STATION_ID})")
    ax0.grid(alpha=0.15)
    ax0.tick_params(axis="x", labelsize=8)

    # (b) ACF
    ax1 = fig.add_subplot(gs[1, 0])
    plot_acf(ts, ax=ax1, lags=72, alpha=0.05,
             title="(b) Autocorrelation (ACF)")
    ax1.set_xlabel("Lag (hours)")
    ax1.grid(alpha=0.15)

    # (c) PACF
    ax2 = fig.add_subplot(gs[1, 1])
    plot_pacf(ts, ax=ax2, lags=72, alpha=0.05, method="ywm",
              title="(c) Partial autocorrelation (PACF)")
    ax2.set_xlabel("Lag (hours)")
    ax2.grid(alpha=0.15)

    path = os.path.join(fig_dir, "station_timeseries_acf.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── 5. Demoted figures (appendix) ───────────────────────────────────────────

def plot_daily_net_flow(df: pd.DataFrame, fig_dir: str = FIG_DIR) -> str:
    """Network-wide mean net_flow by day of week (appendix only)."""
    daily = df.groupby("dow_name")["net_flow"].mean().reindex(DAY_ORDER)

    fig, ax = plt.subplots(figsize=(5, 3))
    colours = ["#c0392b" if v < 0 else "#27ae60" for v in daily]
    ax.bar(range(7), daily, color=colours, width=0.6)
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in DAY_ORDER])
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="Day of week", ylabel="Mean net flow",
           title="[Appendix] Network-wide mean net flow by day of week")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    path = os.path.join(fig_dir, "appendix_daily_net_flow.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_overall_station_map(stn: pd.DataFrame, fig_dir: str = FIG_DIR) -> str:
    """Overall mean net_flow scatter map (appendix only)."""
    geo = stn.dropna(subset=["lat", "lon"]).copy()
    n_excl = len(stn) - len(geo)
    vabs = np.abs(geo["mean_net_flow"]).quantile(0.98)

    fig, ax = plt.subplots(figsize=(7, 7.5))
    sc = ax.scatter(
        geo["lon"], geo["lat"],
        c=geo["mean_net_flow"], cmap=SPATIAL_CMAP,
        vmin=-vabs, vmax=vabs,
        s=14, alpha=0.8, edgecolors="0.4", linewidths=0.2,
    )
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02,
                 label="Mean net flow (dep − arr)")
    _label_extremes(ax, geo, "mean_net_flow", n=5)
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"[Appendix] Overall mean net flow  "
                 f"({len(geo)} stations, {n_excl} excluded)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path = os.path.join(fig_dir, "appendix_station_map_overall.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── 6. Recommendations & notes ──────────────────────────────────────────────

REPORT_FIGURES = textwrap.dedent("""\
    ── RECOMMENDED FIGURES FOR THE 6-PAGE REPORT ──────────────

    NOTE: net_flow = departures − arrivals, therefore:
      positive (blue on RdBu maps) = net outflow / net sender
      negative (red  on RdBu maps) = net inflow  / net receiver

    1. dow_hour_heatmap.png
       Most information-dense figure. Shows the full Mon–Sun × 0–23h
       commuter tidal pattern in a single compact view.

    2. weekday_weekend_profile.png
       Reveals the temporal regime shift: bimodal commuter weekdays
       vs unimodal leisure weekends across departures, arrivals, and
       net flow.

    3. map_am_peak_net_flow.png
       Spatial snapshot of the morning commuter flow (weekday
       07:00–09:59). Blue stations (net outflow / senders) are
       typically rail termini and residential fringes where
       commuters depart; red stations (net inflow / receivers)
       cluster around City/Holborn employment zones where
       commuters arrive.

    4. map_pm_peak_net_flow.png
       Mirror image of the AM peak (weekday 17:00–19:59). The same
       employment-zone stations now show net outflow (blue) as
       commuters depart; residential and rail stations turn red
       (net inflow) as bikes return. Both maps share a common
       colour scale for direct comparison.

    5. station_timeseries_acf.png
       Diagnostic for the modelling section: strong 24h periodicity
       in ACF confirms a seasonal ARIMA / STL-based approach is
       warranted; PACF decay guides AR order selection.

    ── APPENDIX FIGURES ───────────────────────────────────────

    A1. hourly_profile.png
        Partly redundant with Figures 1 and 2, but the clearest
        single-panel view of the AM/PM tidal bar pattern.

    A2. appendix_daily_net_flow.png
        Sanity check: network-wide net flow is ≈ 0 every day
        (bikes are conserved). Minimal analytical content.

    A3. appendix_station_map_overall.png
        All-hours mean net flow map; superseded by the more
        informative peak-specific maps in Figures 3–4.
""")


FINAL_CAPTIONS_TEMPLATE = textwrap.dedent("""\
    ── FINAL FIGURE CAPTIONS ──────────────────────────────────

    Study period: {date_start} to {date_end} ({n_days} days).
    Convention: net_flow = departures − arrivals.
    Positive values (blue) indicate net outflow (senders);
    negative values (red) indicate net inflow (receivers).

    dow_hour_heatmap.png
      Heatmap of mean station-level net flow by day of week and
      hour of day ({date_start} – {date_end}). Weekday mornings
      (07:00–09:59) show strong positive net flow (net outflow)
      as commuters depart residential stations, while the evening
      peak (17:00–19:59) reverses the pattern. Weekend activity is
      weaker and more symmetric, consistent with leisure rather
      than commuter use.

    weekday_weekend_profile.png
      Average hourly departures, arrivals, and net flow split by
      weekday and weekend ({date_start} – {date_end}). Weekdays
      display a bimodal commuter signature with sharp AM/PM peaks;
      weekends show a single midday hump. The net flow panel
      confirms that the weekday tidal asymmetry is largely absent
      on weekends.

    map_am_peak_net_flow.png
      Spatial distribution of mean net flow during the weekday AM
      peak (07:00–09:59). Blue (positive) stations are net
      senders — predominantly rail termini and residential fringes
      where commuters pick up bikes. Red (negative) stations are
      net receivers concentrated around City and Holborn employment
      zones where commuters arrive. Colour scale is symmetric and
      shared with the PM peak map for direct comparison.

    map_pm_peak_net_flow.png
      Spatial distribution of mean net flow during the weekday PM
      peak (17:00–19:59). The pattern reverses: employment-zone
      stations turn blue (net outflow) as commuters depart, while
      rail termini and residential stations turn red (net inflow)
      as bikes return. The shared colour scale confirms the AM/PM
      reversal is near-symmetric in magnitude.

    station_timeseries_acf.png
      Hourly net flow time series for Waterloo Station 3 (id 154)
      over {n_days} days ({date_start} – {date_end}).
      The ACF shows strong periodicity at 24-hour lags, confirming
      diurnal cyclicity. The PACF shows rapid decay beyond lag 1–2
      with a secondary spike near lag 24, suggesting a seasonal
      ARIMA or STL decomposition approach is appropriate for
      one-hour-ahead forecasting.
""")


# ── Main entry point ─────────────────────────────────────────────────────────

def main(
    panel_path: str = PANEL_PATH,
    fig_dir: str = FIG_DIR,
    tbl_dir: str = TBL_DIR,
) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    print("\n── Loading panel ──")
    panel = pd.read_csv(panel_path)
    panel["datetime_hour"] = pd.to_datetime(panel["datetime_hour"])
    print(f"  {len(panel):,} rows, {panel['station_id'].nunique()} stations")

    # Station summary
    print("\n── Station-level summary ──")
    stn = station_summary(panel)
    stn.to_csv(os.path.join(tbl_dir, "station_summary.csv"), index=False)
    print(f"  {len(stn)} stations")

    df = _add_time_cols(panel)
    n_no_geo = stn[["lat", "lon"]].isna().any(axis=1).sum()
    print(f"  Stations without coordinates (excluded from maps): {n_no_geo}")

    # ── Report figures ──
    print("\n── Report figures ──")

    for func, args in [
        (plot_hourly_profile, (df, fig_dir)),
        (plot_weekday_weekend_profile, (df, fig_dir)),
        (plot_dow_hour_heatmap, (df, fig_dir)),
    ]:
        path = func(*args)
        print(f"  ✓ {path}")

    shared_vabs = _compute_shared_vabs(df, stn, [(7, 9), (17, 19)])
    for func, args in [
        (plot_am_peak_map, (df, stn, shared_vabs, fig_dir)),
        (plot_pm_peak_map, (df, stn, shared_vabs, fig_dir)),
    ]:
        path = func(*args)
        print(f"  ✓ {path}")

    path = plot_station_timeseries(panel, fig_dir=fig_dir)
    print(f"  ✓ {path}")

    # ── Appendix figures ──
    print("\n── Appendix figures ──")
    for func, args in [
        (plot_daily_net_flow, (df, fig_dir)),
        (plot_overall_station_map, (stn, fig_dir)),
    ]:
        path = func(*args)
        print(f"  ✓ {path}")

    # ── Ranking tables ──
    print("\n── Ranking tables ──")
    rankings = ranking_tables(stn, tbl_dir=tbl_dir)
    for name in rankings:
        print(f"  ✓ {tbl_dir}/{name}.csv")

    # ── Diagnostic station rationale ──
    print(f"\n  Station selection: {STATION_SELECTION_RATIONALE}")

    # ── Save notes ──
    notes_path = os.path.join(tbl_dir, "esda_figure_notes.txt")
    with open(notes_path, "w") as f:
        f.write(REPORT_FIGURES)
        f.write("\n── DIAGNOSTIC STATION SELECTION ────────────────────────\n")
        f.write(STATION_SELECTION_RATIONALE + "\n")
    print(f"\n  ✓ {notes_path}")

    # ── Final polished captions ──
    date_start = panel["datetime_hour"].min().strftime("%Y-%m-%d")
    date_end = panel["datetime_hour"].max().strftime("%Y-%m-%d")
    n_days = (panel["datetime_hour"].max() - panel["datetime_hour"].min()).days + 1
    final_captions = FINAL_CAPTIONS_TEMPLATE.format(
        date_start=date_start, date_end=date_end, n_days=n_days,
    )
    captions_path = os.path.join(tbl_dir, "final_figure_captions.txt")
    with open(captions_path, "w") as f:
        f.write(final_captions)
    print(f"  ✓ {captions_path}")

    print(f"\n{REPORT_FIGURES}")
    print("Done.\n")


if __name__ == "__main__":
    main()
