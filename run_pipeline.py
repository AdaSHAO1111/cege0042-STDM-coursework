#!/usr/bin/env python3
"""
Step 1 — Build the balanced station-hour panel from raw TfL journey data.

Study period: 2018-01-03 to 2018-03-06 (10 weekly CSV files).
A datetime trim drops journeys before 2018-01-03 so the panel starts cleanly.

Usage:
    python run_pipeline.py
"""
from src.build_panel import run_pipeline

JOURNEY_DIR = "data/raw/journeys"
BIKEPOINT_JSON = "data/raw/bikepoint_all.json"

OUT_METADATA = "data/interim/station_metadata.csv"
OUT_PANEL = "data/processed/station_hour_netflow.csv"
OUT_AUDIT = "outputs/tables/data_audit_summary.txt"

TRIM_START = "2018-01-03"

if __name__ == "__main__":
    run_pipeline(
        journey_dir=JOURNEY_DIR,
        bikepoint_json=BIKEPOINT_JSON,
        out_metadata=OUT_METADATA,
        out_panel=OUT_PANEL,
        out_audit=OUT_AUDIT,
        trim_start=TRIM_START,
    )
