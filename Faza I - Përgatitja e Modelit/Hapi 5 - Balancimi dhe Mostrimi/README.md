# Hapi 5: Balancimi dhe Mostrimi

## Qëllimi
Ky hap krijon ndarjen Train/Test dhe gjeneron matricat e gatshme për modelim. Në këtë run, klasat ishin tashmë të balancuara, prandaj nuk u krijuan rreshta sintetikë shtesë.

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
Skedarët e gjeneruar:
- `train_balanced_features.csv`
- `train_balanced_target.csv`
- `test_features.csv`
- `test_target.csv`
- `class_distribution_after_balancing.csv`

Kolonat në `train_balanced_features.csv` dhe `test_features.csv`:
```text
numeric__day
numeric__month
numeric__year
numeric__Carbon intensity gCO₂eq/kWh (direct)
numeric__Carbon intensity gCO₂eq/kWh (Life cycle)
numeric__Carbon-free energy percentage (CFE%)
numeric__Renewable energy percentage (RE%)
numeric__Data estimated
numeric__estimated_hour_flag
numeric__hourly_row_count
numeric__estimated_hour_share
numeric__hour
numeric__day_of_week
numeric__is_weekend
numeric__carbon_intensity_gap
numeric__cfe_re_gap
categorical__Data source_Electricity Maps Estimation
categorical__Data source_entsoe.eu
categorical__Data source_kostt.com
categorical__Data estimation method_Unknown
```

Kolona në `train_balanced_target.csv` dhe `test_target.csv`:
```text
target_quantile_class
```

Kolonat në `class_distribution_after_balancing.csv`:
```text
target_quantile_class
count
```

## Datasetet që përdoren më pas
Këto skedarë përdoren direkt për modelim në Fazën II:
- `train_balanced_features.csv`
- `train_balanced_target.csv`
- `test_features.csv`
- `test_target.csv`

## Ekzekutimi
```bash
python step_5.py
```
