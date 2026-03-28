# Hapi 4: Detektimi i Outliers

## Qëllimi
Ky hap identifikon anomalitë me tri metoda: IQR, Z-score dhe Isolation Forest. Rreshtat vetëm flag-ohen, nuk fshihen ende.

## Input
Skedari hyrës:
- `../Hapi 3 - Diskretizimi/dataset_with_target.csv`

Kolonat në input:
```text
day
month
year
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
estimated_hour_flag
hourly_row_count
estimated_hour_share
hour
day_of_week
is_weekend
carbon_intensity_gap
cfe_re_gap
target_quantile_class
```

## Output
Skedari i gjeneruar:
- `outlier_flags_dataset.csv`

Kolonat në `outlier_flags_dataset.csv`:
```text
day
month
year
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
estimated_hour_flag
hourly_row_count
estimated_hour_share
hour
day_of_week
is_weekend
carbon_intensity_gap
cfe_re_gap
target_quantile_class
iqr_outlier_count
zscore_outlier_count
isolation_forest_flag
outlier_consensus_count
outlier_consensus_flag
```

## Dataseti që vazhdon në hapin tjetër
Skedari kryesor për Hapin 5 është `outlier_flags_dataset.csv`.

## Ekzekutimi
```bash
python step_4.py
```
