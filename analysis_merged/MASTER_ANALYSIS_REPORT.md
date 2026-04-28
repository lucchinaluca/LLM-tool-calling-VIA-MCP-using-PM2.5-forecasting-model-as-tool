# Master Analysis Report

## Overview

- This file merges all analysis completed so far into one document.
- Source data used for analysis: `classified_datasets`.
- Main objective: establish baseline understanding of traffic, weather, and pollution; then deepen analysis with pollution as the target.

## What Was Done End-to-End

### 1) Dataset cleaning and structuring

- Dataset files were filtered and cleaned from mixed folders.
- Non-dataset files were removed from `new_datasets`.
- Duplicate dataset files were removed against the whole workspace and within `new_datasets`.
- A classified structure was created in `classified_datasets` with domain groups:
  - `weather` (`daily_weather`, `aggregated_weather`)
  - `pollution` (`city_wide`, `road_wise`)
  - `traffic` (`anpr`, `atc`, `google_travel_time`, `other_traffic`)
  - `combined` (cross-domain combined sets)

### 2) Initial cross-domain analysis

- A first exploratory pass profiled all categories by file-level and dataset-level metrics.
- Outputs generated in `analysis_output_20260326_191734`:
  - `INITIAL_ANALYSIS_REPORT.md`
  - `file_inventory.csv`
  - `category_summary.csv`
  - plots for files/rows distribution, largest files, and sample trends.

### 3) Deep pollution-focused analysis

- A deeper, data-level analysis was run with pollution as the target.
- Initial deep output was improved and corrected in:
  - `analysis_output_20260326_192621_pollution_deep_v2`
- Corrections applied:
  - removed sentinel/extreme placeholder values (`>= 1e9`)
  - excluded false pollutant matches (for example, `Count` as pollutant)

## Initial Cross-Domain Findings

Category-level summary:


| Category  | Files | Readable | Total rows | Avg columns | Avg missing % |
| --------- | ----- | -------- | ---------- | ----------- | ------------- |
| combined  | 16    | 16       | 55,424     | 9.56        | 13.76         |
| pollution | 10    | 10       | 1,089,882  | 9.80        | 18.74         |
| traffic   | 101   | 101      | 13,597,739 | 5.56        | 18.91         |
| weather   | 156   | 156      | 84,923     | 11.99       | 6.11          |


Key points:

- Largest file: `traffic/anpr/observations.csv` (~4.03M rows).
- Read failures in initial run: 0.
- Data volume is dominated by traffic; weather is fragmented across many daily files.
- Combined datasets are sufficient for initial cross-domain correlation checks.

## Pollution-Focused Results (Target Variable)

Coverage after correction:

- Pollution files scanned: 10
- Readable: 9
- With usable datetime: 9
- With pollutant columns: 9

Pollutant summary (corrected):


| Pollutant | N         | Mean   | Median | P95    | Max    |
| --------- | --------- | ------ | ------ | ------ | ------ |
| NOx       | 1,056,915 | 44.09  | 30.30  | 131.40 | 257.00 |
| NO        | 1,056,569 | 27.01  | 14.80  | 97.00  | 210.30 |
| NO2       | 1,055,856 | 17.33  | 15.30  | 37.50  | 57.10  |
| CO        | 432,257   | 659.26 | 999.00 | 999.00 | 999.00 |
| PM10      | 207,101   | 47.72  | 22.22  | 60.67  | 999.76 |
| O3        | 21,963    | 19.40  | 20.00  | 33.50  | 49.25  |


Interpretation:

- `NO2`, `NO`, and `NOx` provide strong, high-volume target signals.
- `PM10`, `NO`, and `NOx` show high variability, indicating episodic events.
- CO/PM10 still show capped-looking values (`999` region), so unit/QA harmonization is still needed before final modeling.

## Pollution vs Traffic Relationships

Strong examples from combined datasets:

- `NO2` vs `Count`: `r = 0.649`
- `NO2` vs `speed(km/h)`: `r = -0.648` and `r = -0.593`
- `NO2` vs `TravelTime`: `r = 0.619` and `r = 0.557`
- `NOx` vs `Count`: `r = 0.582`

Conclusion:

- Higher traffic intensity and congestion are consistently associated with higher roadside pollution in key segments.
- Lower speeds align with higher pollution concentration, consistent with congestion/queueing behavior.

## Final Conclusions So Far

- The dataset is now organized and analysis-ready by domain and subtype.
- Pollution can be treated as a viable target variable, with `NO2` recommended as first target.
- There is clear empirical signal linking traffic burden and pollution, supporting predictive modeling and intervention studies.
- Before production modeling, timestamp and unit harmonization should be completed, especially for capped pollutant values.

## Recommended Next Step

- Build an hourly, site-level modeling table for `NO2`:
  - features: traffic counts/speed/travel time + weather variables + lag terms
  - target: `NO2`
  - outputs: train/validation splits, baseline model, feature importance, and scenario simulation.

## Where All Outputs Live

- Initial EDA: `analysis_output_20260326_191734`
- Deep pollution analysis (corrected): `analysis_output_20260326_192621_pollution_deep_v2`
- This merged report: `analysis_merged/MASTER_ANALYSIS_REPORT.md`

