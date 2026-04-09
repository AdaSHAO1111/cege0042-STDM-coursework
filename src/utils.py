"""
Shared utility functions for the STDM bike-share pipeline.
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd


# ── I/O helpers ──────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_journeys(journey_dir: str) -> pd.DataFrame:
    """
    Read all CSV files in *journey_dir*, concatenate, and parse datetimes.

    The TfL journey CSVs use day-first UK format (dd/mm/yyyy HH:MM).
    """
    csv_paths = sorted(glob.glob(os.path.join(journey_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {journey_dir}")

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        frames.append(df)
        print(f"  loaded {Path(p).name}: {len(df):,} rows")

    journeys = pd.concat(frames, ignore_index=True)

    # Strip whitespace from column names (TfL files sometimes have trailing spaces)
    journeys.columns = journeys.columns.str.strip()

    # Parse datetimes — UK day-first format
    journeys["Start Date"] = pd.to_datetime(
        journeys["Start Date"], dayfirst=True, format="mixed"
    )
    journeys["End Date"] = pd.to_datetime(
        journeys["End Date"], dayfirst=True, format="mixed"
    )

    # Ensure station IDs are integers (drop rows where they are missing)
    for col in ("StartStation Id", "EndStation Id"):
        journeys[col] = pd.to_numeric(journeys[col], errors="coerce")
    journeys.dropna(subset=["StartStation Id", "EndStation Id"], inplace=True)
    journeys["StartStation Id"] = journeys["StartStation Id"].astype(int)
    journeys["EndStation Id"] = journeys["EndStation Id"].astype(int)

    return journeys


def load_station_metadata(json_path: str) -> pd.DataFrame:
    """
    Parse TfL BikePoint JSON into a clean station metadata table.

    The JSON ``id`` field is ``"BikePoints_<n>"``; the numeric suffix ``<n>``
    matches the journey CSV ``StartStation Id`` / ``EndStation Id``.
    """
    with open(json_path) as f:
        raw = json.load(f)

    records = []
    for bp in raw:
        # Extract numeric station ID from e.g. "BikePoints_1"
        station_id = int(bp["id"].replace("BikePoints_", ""))

        # NbDocks lives inside additionalProperties
        nb_docks = None
        for prop in bp.get("additionalProperties", []):
            if prop["key"] == "NbDocks":
                nb_docks = int(prop["value"]) if prop["value"] else None
                break

        records.append(
            {
                "station_id": station_id,
                "station_name": bp["commonName"],
                "lat": bp["lat"],
                "lon": bp["lon"],
                "nb_docks": nb_docks,
            }
        )

    return pd.DataFrame(records)


# ── Diagnostics ──────────────────────────────────────────────────────────────

def print_diagnostics(journeys: pd.DataFrame, stations: pd.DataFrame) -> str:
    """
    Print and return a text summary of data-quality diagnostics.
    """
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 60)
    log("DATA AUDIT SUMMARY")
    log("=" * 60)

    log(f"\nTotal journey rows loaded: {len(journeys):,}")
    log(f"Unique Start Station IDs : {journeys['StartStation Id'].nunique()}")
    log(f"Unique End Station IDs   : {journeys['EndStation Id'].nunique()}")

    all_csv_ids = set(journeys["StartStation Id"]) | set(journeys["EndStation Id"])
    log(f"Unique station IDs (union): {len(all_csv_ids)}")

    log(f"\nDatetime range:")
    log(f"  Start Date min: {journeys['Start Date'].min()}")
    log(f"  Start Date max: {journeys['Start Date'].max()}")
    log(f"  End Date min  : {journeys['End Date'].min()}")
    log(f"  End Date max  : {journeys['End Date'].max()}")

    log(f"\nMissing values per column:")
    for col in journeys.columns:
        n_miss = journeys[col].isna().sum()
        if n_miss > 0:
            log(f"  {col}: {n_miss:,}")
    if journeys.isna().sum().sum() == 0:
        log("  (none)")

    log(f"\nStation metadata rows: {len(stations)}")
    json_ids = set(stations["station_id"])
    matched = all_csv_ids & json_ids
    csv_only = sorted(all_csv_ids - json_ids)
    log(f"Stations matched to metadata: {len(matched)} / {len(all_csv_ids)}")

    if csv_only:
        log(f"\n⚠ {len(csv_only)} journey station IDs have NO metadata match:")
        log(f"  {csv_only}")

    json_only = sorted(json_ids - all_csv_ids)
    if json_only:
        log(f"\n  {len(json_only)} metadata stations have no journeys in this period")
        log(f"  (these are likely newer/removed stations — not a problem)")

    log("\n" + "=" * 60)
    return "\n".join(lines)
