# Hapi 7: Finalizimi i Datasetit

## Qëllimi
Ky hap prodhon datasetin final ditor të Fazës I, i cili përmban të gjitha veçoritë e inxhinieruara dhe targetin e klasifikimit, pa kolonat e auditimit të outliers.

## Input
Skedari hyrës:
- `../Hapi 4 - Detektimi i Outliers/outlier_flags_dataset.csv`

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
iqr_outlier_count
zscore_outlier_count
isolation_forest_flag
outlier_consensus_count
outlier_consensus_flag
```

## Output
Skedari i gjeneruar:
- `feature_engineered_dataset.csv`

Kolonat në `feature_engineered_dataset.csv`:
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

## Dataseti final
Ky është dataseti final kryesor i Fazës I:
- `feature_engineered_dataset.csv`

Ky skedar është versioni më i plotë i datasetit ditor për analizë, krahasime statistikore dhe si referencë e master dataset-it të përgatitur.

## Ekzekutimi
```bash
python step_7.py
```
