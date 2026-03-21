# Faza I Data Preparation Report

## Dataset Overview
- Source files: `hourly-2021.csv, hourly-2022.csv, hourly-2023.csv, hourly-2024.csv, hourly-2025.csv`
- Merged hourly shape: `43824 rows x 12 columns`
- Daily merged shape: `1826 rows x 16 columns`
- Cleaned shape: `1826 rows x 30 columns`
- Target source column: `Carbon intensity gCO₂eq/kWh (direct)`

## Step 1: Initial Data Overview
- Columns audited: `16`
- High-cardinality columns flagged: `6`

## Step 2: Core Data Cleaning
- Constraint filtering actions: `0`
- Imputation actions: `0`

## Step 3: Class Definition & Imbalance Metrics
```text
target_quantile_class  count   pct  minority_flag
                  low    609 33.35          False
                 high    609 33.35          False
               medium    608 33.30          False
```

## Step 4: Sampling & Balancing
- Holdout strategy: `train_test_split(stratify=y, test_size=0.2, random_state=42)`
- Resampler used: `Skipped (already balanced)`
```text
target_quantile_class  count   pct  minority_flag
                  low    487 33.36          False
                 high    487 33.36          False
               medium    486 33.29          False
```

## Step 5: Data Aggregation
- Aggregated rows: `60`
- Aggregated columns: `50`

## Step 6: Subsets & Transformations
- Added temporal features: `hour`, `day_of_week`, `is_weekend`, `month`, `day`
- Added derived features when supported by the dataset schema.
- Standardization applied inside the modelling matrix builder.

## Step 7: Multi-Method Outlier Detection
- Outlier columns saved in the prepared dataset:
- `iqr_outlier_count`
- `zscore_outlier_count`
- `isolation_forest_flag`
- `outlier_consensus_count`
- `outlier_consensus_flag`

## Step 8: Presentation & Reporting
- Null heatmap and class-balance plots were exported to `output/plots`.
- Structured logs and prepared datasets were exported to `output/`.
