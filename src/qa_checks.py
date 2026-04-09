"""
Global QA checks on the final station-hour panel.

Validates panel integrity: conservation (departures == arrivals == journey rows),
no duplicate rows, non-negative integers, balanced panel, and activity breakdown.

All paths are parameterised so the same logic can serve different datasets.
"""
import glob
import csv

import pandas as pd

PANEL_PATH = "data/processed/station_hour_netflow.csv"
JOURNEY_DIR = "data/raw/journeys"
AUDIT_PATH = "outputs/tables/data_audit_summary.txt"


def count_journey_rows(journey_dir: str) -> int:
    total = 0
    for p in sorted(glob.glob(f"{journey_dir}/*.csv")):
        with open(p) as f:
            total += sum(1 for _ in f) - 1
    return total


def run_global_qa(
    panel_path: str = PANEL_PATH,
    journey_dir: str = JOURNEY_DIR,
) -> str:
    lines: list[str] = []
    all_pass = True

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        suffix = f"  ({detail})" if detail else ""
        log(f"  [{status}] {name}{suffix}")

    log("")
    log("── GLOBAL QA CHECKS ──────────────────────────────────────")

    panel = pd.read_csv(panel_path, parse_dates=["datetime_hour"])
    n_journeys = count_journey_rows(journey_dir)

    # 1–3. Totals
    total_dep = int(panel["departures"].sum())
    total_arr = int(panel["arrivals"].sum())
    total_net = int(panel["net_flow"].sum())

    log(f"\n  Total departures : {total_dep:,}")
    log(f"  Total arrivals   : {total_arr:,}")
    log(f"  Total net_flow   : {total_net:,}")
    log(f"  Journey rows     : {n_journeys:,}")
    log("")

    # 4. Conservation checks
    check("total departures == journey rows",
          total_dep == n_journeys,
          f"{total_dep:,} vs {n_journeys:,}")
    check("total arrivals == journey rows",
          total_arr == n_journeys,
          f"{total_arr:,} vs {n_journeys:,}")
    check("total net_flow == 0",
          total_net == 0,
          f"got {total_net:,}")

    # 5. No duplicate (station_id, datetime_hour) rows
    n_dupes = panel.duplicated(subset=["station_id", "datetime_hour"]).sum()
    check("no duplicate (station_id, datetime_hour)",
          n_dupes == 0,
          f"{n_dupes:,} duplicates found" if n_dupes else "0 duplicates")

    # 6. Non-negative integers
    dep_nonneg = (panel["departures"] >= 0).all()
    arr_nonneg = (panel["arrivals"] >= 0).all()
    dep_int = (panel["departures"] == panel["departures"].astype(int)).all()
    arr_int = (panel["arrivals"] == panel["arrivals"].astype(int)).all()
    check("departures are non-negative integers",
          dep_nonneg and dep_int)
    check("arrivals are non-negative integers",
          arr_nonneg and arr_int)

    # 7. Every station has the same number of hourly rows (balanced panel)
    rows_per_station = panel.groupby("station_id").size()
    expected_hours = panel["datetime_hour"].nunique()
    all_equal = (rows_per_station == expected_hours).all()
    if not all_equal:
        bad = rows_per_station[rows_per_station != expected_hours]
        detail = f"{len(bad)} stations deviate: {dict(bad.head(5))}"
    else:
        detail = (
            f"all {rows_per_station.nunique()} distinct counts == {expected_hours}"
        )
    check(
        f"every station has exactly {expected_hours} rows",
        all_equal,
        detail,
    )

    # 8. Activity category counts
    dep_gt0 = panel["departures"] > 0
    arr_gt0 = panel["arrivals"] > 0
    cat_dep_only = int((dep_gt0 & ~arr_gt0).sum())
    cat_arr_only = int((~dep_gt0 & arr_gt0).sum())
    cat_both = int((dep_gt0 & arr_gt0).sum())
    cat_neither = int((~dep_gt0 & ~arr_gt0).sum())
    total_rows = len(panel)

    log("")
    log("  Station-hour activity breakdown:")
    log(f"    dep>0, arr=0 : {cat_dep_only:>8,}  ({100*cat_dep_only/total_rows:5.1f}%)")
    log(f"    dep=0, arr>0 : {cat_arr_only:>8,}  ({100*cat_arr_only/total_rows:5.1f}%)")
    log(f"    dep>0, arr>0 : {cat_both:>8,}  ({100*cat_both/total_rows:5.1f}%)")
    log(f"    dep=0, arr=0 : {cat_neither:>8,}  ({100*cat_neither/total_rows:5.1f}%)")
    log(f"    total        : {cat_dep_only+cat_arr_only+cat_both+cat_neither:>8,}")
    check("category counts sum to panel rows",
          cat_dep_only + cat_arr_only + cat_both + cat_neither == total_rows)

    # 9. Hour-range note
    h_min = panel["datetime_hour"].min()
    h_max = panel["datetime_hour"].max()
    log("")
    log("  Note on panel hour range:")
    log("  Timestamps are floored to whole hours before aggregation.")
    log(f"  Panel spans {h_min} to {h_max} inclusive,")
    log(f"  giving {expected_hours} hourly slots per station.")

    # Final verdict
    log("")
    if all_pass:
        log("GLOBAL QA PASSED.")
    else:
        log("GLOBAL QA FAILED — see individual checks above.")
    log("")
    log("=" * 60)

    return "\n".join(lines)


def main(
    panel_path: str = PANEL_PATH,
    journey_dir: str = JOURNEY_DIR,
    audit_path: str = AUDIT_PATH,
) -> None:
    qa_text = run_global_qa(panel_path=panel_path, journey_dir=journey_dir)

    with open(audit_path, "a") as f:
        f.write(qa_text)

    print(f"\n✓ Appended to {audit_path}")


if __name__ == "__main__":
    main()
