#!/usr/bin/env python3
"""
Step 2 — Exploratory spatial-temporal data analysis (ESDA).

Produces publication-quality figures and station-ranking tables from the
balanced station-hour panel built by run_pipeline.py.

Usage:
    python run_esda.py
"""
from src.esda import main

if __name__ == "__main__":
    main(
        panel_path="data/processed/station_hour_netflow.csv",
        fig_dir="outputs/figures",
        tbl_dir="outputs/tables",
    )
