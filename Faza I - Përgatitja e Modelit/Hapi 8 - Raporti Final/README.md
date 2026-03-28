# Hapi 8: Raporti Final

## Qëllimi
Ky hap përmbledh Fazën I në formë raporti dhe grafike.

## Input
Skedarët hyrës kryesorë:
- `../Hapi 1 - Ngarkimi dhe Bashkimi/merged_daily_dataset.csv`
- `../Hapi 3 - Diskretizimi/class_distribution.csv`
- `../Hapi 5 - Balancimi dhe Mostrimi/class_distribution_after_balancing.csv`
- `../Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

Kolonat në `merged_daily_dataset.csv`:
```text
Country
Zone name
Zone id
date
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
Datetime (UTC)
estimated_hour_share
```

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

## Output
Skedarët e gjeneruar:
- `null_heatmap.png`
- `class_balance_comparison.png`
- `phase1_report.md`
- `phase1_report.pdf`

Ky hap nuk gjeneron dataset të ri tabelor, prandaj nuk ka kolona të reja output për CSV.

## Ekzekutimi
```bash
python step_8.py
```
