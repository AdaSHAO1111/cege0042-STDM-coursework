"""
Build the balanced station-hour panel for London Santander Cycles.

Reads raw journey CSVs and BikePoint JSON, produces:
  - data/interim/station_metadata.csv
  - data/processed/station_hour_netflow.csv
  - outputs/tables/data_audit_summary.txt

The core logic is in ``run_pipeline`` so callers can supply custom paths
without duplicating code.
"""
import glob as _glob
import os

import pandas as pd

from src.utils import (
    ensure_dirs,
    load_journeys,
    load_station_metadata,
    print_diagnostics,
)

# ── Default paths ────────────────────────────────────────────────────────────

JOURNEY_DIR = "data/raw/journeys"
BIKEPOINT_JSON = "data/raw/bikepoint_all.json"

OUT_METADATA = "data/interim/station_metadata.csv"
OUT_PANEL = "data/processed/station_hour_netflow.csv"
OUT_AUDIT = "outputs/tables/data_audit_summary.txt"


# ── Panel construction ───────────────────────────────────────────────────────

def build_station_hour_panel(journeys: pd.DataFrame) -> pd.DataFrame:
    """
    Build a **complete balanced** station × hour panel.

    1. Compute hourly departure / arrival counts from journey data.
    2. Create the full Cartesian product of every station ID that appears
       in the journey data × every hour from the global min to global max.
    3. Left-join the counts onto this grid and fill gaps with 0.
    4. Recompute net_flow = departures − arrivals.
    """
    journeys["start_hour"] = journeys["Start Date"].dt.floor("h")
    journeys["end_hour"] = journeys["End Date"].dt.floor("h")

    # --- aggregate counts ---------------------------------------------------
    departures = (
        journeys.groupby(["StartStation Id", "start_hour"])
        .size()
        .reset_index(name="departures")
        .rename(columns={"StartStation Id": "station_id", "start_hour": "datetime_hour"})
    )

    arrivals = (
        journeys.groupby(["EndStation Id", "end_hour"])
        .size()
        .reset_index(name="arrivals")
        .rename(columns={"EndStation Id": "station_id", "end_hour": "datetime_hour"})
    )

    # --- full Cartesian grid ------------------------------------------------
    all_station_ids = sorted(
        set(journeys["StartStation Id"]) | set(journeys["EndStation Id"])
    )
    hour_min = min(journeys["start_hour"].min(), journeys["end_hour"].min())
    hour_max = max(journeys["start_hour"].max(), journeys["end_hour"].max())
    all_hours = pd.date_range(hour_min, hour_max, freq="h")

    grid = pd.MultiIndex.from_product(
        [all_station_ids, all_hours], names=["station_id", "datetime_hour"]
    ).to_frame(index=False)

    # --- merge counts onto grid, fill zeros ---------------------------------
    grid = grid.merge(departures, on=["station_id", "datetime_hour"], how="left")
    grid = grid.merge(arrivals, on=["station_id", "datetime_hour"], how="left")

    grid["departures"] = grid["departures"].fillna(0).astype(int)
    grid["arrivals"] = grid["arrivals"].fillna(0).astype(int)
    grid["net_flow"] = grid["departures"] - grid["arrivals"]

    grid.sort_values(["station_id", "datetime_hour"], inplace=True)
    grid.reset_index(drop=True, inplace=True)

    return grid


# ── Reusable pipeline runner ─────────────────────────────────────────────────

def run_pipeline(
    journey_dir: str,
    bikepoint_json: str,
    out_metadata: str,
    out_panel: str,
    out_audit: str,
    trim_start: str | None = None,
) -> None:
    """
    End-to-end pipeline: load journeys, build balanced panel, run QA, save.

    All paths are explicit parameters so the same logic serves both the
    original 4-week dataset and the expanded 10-week dataset.

    Parameters
    ----------
    trim_start : str or None
        If given (e.g. ``"2018-01-03"``), drop every journey whose
        *Start Date* is before this timestamp **before** building the panel.
        The audit summary records both the raw and trimmed datetime ranges.
    """
    ensure_dirs(
        os.path.dirname(out_metadata),
        os.path.dirname(out_panel),
        os.path.dirname(out_audit),
    )

    # 1. Load journey data
    print("\n── Loading journey CSVs ──")
    journeys = load_journeys(journey_dir)
    n_csv_files = len(sorted(_glob.glob(os.path.join(journey_dir, "*.csv"))))

    raw_rows = len(journeys)
    raw_start_min = journeys["Start Date"].min()
    raw_start_max = journeys["Start Date"].max()
    raw_end_min = journeys["End Date"].min()
    raw_end_max = journeys["End Date"].max()

    # 1b. Optional datetime trim
    if trim_start is not None:
        trim_ts = pd.Timestamp(trim_start)
        before = len(journeys)
        journeys = journeys[
            (journeys["Start Date"] >= trim_ts)
            & (journeys["End Date"] >= trim_ts)
        ].copy()
        after = len(journeys)
        print(f"\n  Trim applied: Start Date >= {trim_ts}")
        print(f"  Rows before trim: {before:,}  →  after trim: {after:,}  "
              f"(dropped {before - after:,})")

    # 2. Load station metadata
    print("\n── Loading station metadata ──")
    stations = load_station_metadata(bikepoint_json)
    print(f"  {len(stations)} stations parsed from BikePoint JSON")

    # 3. Diagnostics (journey-level)
    print()
    audit_text = print_diagnostics(journeys, stations)

    # 4. Save station metadata
    stations.to_csv(out_metadata, index=False)
    print(f"\n✓ Station metadata saved → {out_metadata}")

    # 5. Build balanced station-hour panel
    print("\n── Building balanced station-hour panel ──")
    panel = build_station_hour_panel(journeys)

    n_stations = panel["station_id"].nunique()
    n_hours = panel["datetime_hour"].nunique()
    expected_rows = n_stations * n_hours
    actual_rows = len(panel)

    print(f"  Unique stations : {n_stations}")
    print(f"  Unique hours    : {n_hours}")
    print(f"  Expected rows   : {expected_rows:,}  ({n_stations} × {n_hours})")
    print(f"  Actual rows     : {actual_rows:,}")
    print(f"  Balanced check  : {'PASS ✓' if actual_rows == expected_rows else 'FAIL ✗'}")
    print(f"  Hour range      : {panel['datetime_hour'].min()} → {panel['datetime_hour'].max()}")

    zero_activity = ((panel["departures"] == 0) & (panel["arrivals"] == 0)).sum()
    print(f"  Zero-activity station-hours: {zero_activity:,} "
          f"({100 * zero_activity / actual_rows:.1f}%)")

    dep_na = panel["departures"].isna().sum()
    arr_na = panel["arrivals"].isna().sum()
    print(f"  Missing departures: {dep_na}  |  Missing arrivals: {arr_na}")

    # 6. Merge metadata onto panel
    panel = panel.merge(
        stations[["station_id", "station_name", "lat", "lon"]],
        on="station_id",
        how="left",
    )
    n_with_meta = panel["lat"].notna().sum()
    n_without_meta = actual_rows - n_with_meta
    print(f"  Rows with metadata: {n_with_meta:,} / {actual_rows:,} "
          f"({100 * n_with_meta / actual_rows:.1f}%)")

    # 7. QA: random spot-check that net_flow == departures - arrivals
    sample = panel.sample(n=5, random_state=42)
    check_ok = (sample["net_flow"] == sample["departures"] - sample["arrivals"]).all()

    # 8. Save panel
    panel.to_csv(out_panel, index=False)
    print(f"\n✓ Station-hour panel saved → {out_panel}")

    # 9. Compute global QA numbers
    total_dep = int(panel["departures"].sum())
    total_arr = int(panel["arrivals"].sum())
    total_net = int(panel["net_flow"].sum())
    n_dupes = int(panel.duplicated(subset=["station_id", "datetime_hour"]).sum())

    all_csv_ids = set(journeys["StartStation Id"]) | set(journeys["EndStation Id"])
    json_ids = set(stations["station_id"])
    matched = all_csv_ids & json_ids
    unmatched = all_csv_ids - json_ids

    # 10. Build extended audit text and save
    trim_applied = trim_start is not None
    extra_lines = [
        "",
        "── BALANCED PANEL QA ──────────────────────────────────────",
        f"Weekly CSV files loaded   : {n_csv_files}",
        f"Total journey rows loaded : {raw_rows:,}",
        f"Raw datetime range (before any trim):",
        f"  Start Date : {raw_start_min}  →  {raw_start_max}",
        f"  End Date   : {raw_end_min}  →  {raw_end_max}",
    ]
    if trim_applied:
        extra_lines += [
            f"Trim applied             : Start Date >= {trim_start}",
            f"Rows after trim          : {len(journeys):,}  "
            f"(dropped {raw_rows - len(journeys):,})",
            f"Analysis datetime range (after trim):",
            f"  Start Date : {journeys['Start Date'].min()}  →  {journeys['Start Date'].max()}",
            f"  End Date   : {journeys['End Date'].min()}  →  {journeys['End Date'].max()}",
        ]
    else:
        extra_lines.append("Trim applied             : none")
    extra_lines += [
        f"Panel hour range         : {panel['datetime_hour'].min()} → {panel['datetime_hour'].max()}",
        f"Unique journey stations  : {len(all_csv_ids)}",
        f"Matched metadata stations: {len(matched)} / {len(all_csv_ids)}",
        f"Unmatched journey stations: {len(unmatched)}"
        + (f"  {sorted(unmatched)}" if unmatched else ""),
        "",
        f"Unique stations          : {n_stations}",
        f"Unique hourly timestamps : {n_hours}",
        f"Expected balanced rows   : {expected_rows:,}  ({n_stations} × {n_hours})",
        f"Actual balanced rows     : {actual_rows:,}",
        f"Balanced check           : {'PASS' if actual_rows == expected_rows else 'FAIL'}",
        f"Zero-activity station-hrs: {zero_activity:,} ({100 * zero_activity / actual_rows:.1f}%)",
        f"Missing departures       : {dep_na}",
        f"Missing arrivals         : {arr_na}",
        "",
        "── GLOBAL QA ─────────────────────────────────────────────",
        f"Total departures         : {total_dep:,}",
        f"Total arrivals           : {total_arr:,}",
        f"Total net_flow           : {total_net:,}",
        f"Duplicate panel rows     : {n_dupes:,}",
        f"dep == arr == journey rows: {'PASS' if total_dep == total_arr == len(journeys) else 'FAIL'}"
        f"  ({total_dep:,} vs {total_arr:,} vs {len(journeys):,})",
        f"net_flow sums to zero    : {'PASS' if total_net == 0 else 'FAIL'}  (got {total_net:,})",
        f"No duplicate rows        : {'PASS' if n_dupes == 0 else 'FAIL'}",
        "",
        "── NET_FLOW SPOT CHECK (5 random rows, seed=42) ──────────",
    ]
    for _, row in sample.iterrows():
        status = "OK" if row["net_flow"] == row["departures"] - row["arrivals"] else "FAIL"
        extra_lines.append(
            f"  station {int(row['station_id']):>4d}  "
            f"{row['datetime_hour']}  "
            f"dep={int(row['departures'])}  arr={int(row['arrivals'])}  "
            f"net={int(row['net_flow'])}  → {status}"
        )
    extra_lines.append(f"  Overall: {'ALL PASSED ✓' if check_ok else 'SOME FAILED ✗'}")
    extra_lines.append("")
    extra_lines.append("=" * 60)

    full_audit = audit_text + "\n" + "\n".join(extra_lines)
    with open(out_audit, "w") as f:
        f.write(full_audit)
    print(f"✓ Audit summary saved   → {out_audit}")

    print("\nDone.\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    run_pipeline(
        journey_dir=JOURNEY_DIR,
        bikepoint_json=BIKEPOINT_JSON,
        out_metadata=OUT_METADATA,
        out_panel=OUT_PANEL,
        out_audit=OUT_AUDIT,
    )


if __name__ == "__main__":
    main()
