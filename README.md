# Spatio-Temporal Analysis of London’s Santander Cycles Bike-Sharing System

CEGE0042 Spatial and Temporal Data Analysis and Data Mining — Coursework

## Project Summary

This repository contains a reproducible analysis pipeline for modelling
short-term station-level imbalance in the London Santander Cycles network.
The workflow builds a balanced station × hour panel from raw TfL journey
records, conducts exploratory spatio-temporal data analysis (ESDA), and
compares two one-hour-ahead forecasting approaches for station-level
**net flow** (departures minus arrivals):

1. **SARIMA(1,0,1)(1,0,1)[24]** — fitted independently on 15 representative stations  
2. **Global RandomForest** — trained across all stations with temporal,
   autoregressive, and spatial-neighbour lag features

## Research Objective

To investigate the spatio-temporal structure of bike-sharing activity across
London and to evaluate the predictive performance of a classical time-series
model (SARIMA) against a global machine-learning approach for one-hour-ahead
station-level net-flow forecasting.

## Study Period

**2018-01-03 to 2018-03-06** (63 days, approximately 9 weeks)

- **Training window:** 2018-01-03 to 2018-02-20 (7 weeks)
- **Test window:** 2018-02-21 to 2018-03-06 (2 weeks)

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Weekly journey files | [TfL Cycling Data](https://cycling.data.tfl.gov.uk/) | 10 weekly CSV extracts covering 27 Dec 2017 to 6 Mar 2018 |
| BikePoint metadata | [TfL Unified API](https://api.tfl.gov.uk/BikePoint) | Station coordinates, names, and dock counts |

## Repository Structure

```text
├── run_pipeline.py          Step 1: build balanced station-hour panel
├── run_esda.py              Step 2: exploratory spatio-temporal analysis
├── run_models.py            Step 3: SARIMA + RandomForest modelling pipeline
├── polish_figures.py        Step 4 (optional): publication-quality figure polish
├── run_all.py               Convenience wrapper — runs Steps 1–4 sequentially
│
├── src/                     Core analysis modules
│   ├── utils.py               I/O helpers (load journeys, metadata, diagnostics)
│   ├── build_panel.py         Panel construction (balanced station × hour grid)
│   ├── qa_checks.py           Global QA checks on the finished panel
│   ├── features.py            Feature engineering (lags, spatial neighbours, split)
│   ├── esda.py                ESDA figures and station-ranking tables
│   ├── model_sarima.py        SARIMA baseline on 15 representative stations
│   ├── model_rf.py            Global RandomForest regressor
│   └── model_eval.py          Evaluation metrics and comparison figures
│
├── data/
│   ├── raw/
│   │   ├── journeys/          User-supplied TfL journey CSVs (not tracked in git)
│   │   └── bikepoint_all.json TfL BikePoint station metadata
│   ├── interim/               Generated: parsed station metadata
│   └── processed/             Generated: balanced station-hour panel
│
├── outputs/
│   ├── figures/               Final report figures and supplementary diagnostics
│   └── tables/                Summary tables, model predictions, and QA logs
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment Setup

**Python 3.10+** is required.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## How to Run

The pipeline consists of four sequential steps.
Each step depends on the outputs of the previous one.

```bash
python run_pipeline.py       # Step 1 — build balanced station-hour panel
python run_esda.py           # Step 2 — exploratory spatial-temporal analysis
python run_models.py         # Step 3 — SARIMA + RandomForest modelling
python polish_figures.py     # Step 4 — publication-quality figure polish (optional)
```

Or run everything at once:

```bash
python run_all.py
```

> **Note:** Step 3 (SARIMA fitting on 15 stations) may take several minutes
> depending on hardware.

## Expected Outputs

After a full run, the `outputs/` directory will contain:

**Figures** (`outputs/figures/`) — publication-quality figures including
temporal profiles, spatial peak-flow maps, ACF/PACF diagnostics, model
comparison charts, forecast overlays, feature-importance plots, and residual diagnostics.

**Tables** (`outputs/tables/`) — model comparison metrics
(`model_comparison.csv`, `per_station_comparison.csv`), station-level
summaries, ranking tables (`top15_*.csv`), prediction files, and QA logs.

## Reproducibility Notes

- All random seeds are fixed (`random_state=42`) for the RandomForest model.
- SARIMA fitting uses `maxiter=200`; convergence warnings are suppressed for
  stations where the optimiser does not fully converge.
- The spatial-neighbour feature uses KNN (`k=5`) on station lat/lon
  coordinates (Euclidean approximation, acceptable at the London scale).
- The reported analysis in this repository uses RandomForest as the tree-based model.

## Data Availability

The raw TfL weekly journey CSV files are not stored in this repository due to
file size. To reproduce the analysis from scratch:

1. Download the 10 weekly journey extracts (files 90–99, covering
   27 Dec 2017 to 6 Mar 2018) from
   [TfL Cycling Data](https://cycling.data.tfl.gov.uk/).
2. Place the CSV files in `data/raw/journeys/`.
3. The BikePoint metadata file (`data/raw/bikepoint_all.json`) is included.

Pre-built outputs in `outputs/` are provided so that figures and tables can
be inspected without re-running the pipeline.

## Quick Reference

```bash
python run_pipeline.py       # build panel
python run_esda.py           # ESDA figures & tables
python run_models.py         # fit models & evaluate
python polish_figures.py     # polish figures (optional)
```
