# Hapi 1: Ngarkimi dhe Bashkimi

## Qëllimi
Ky hap lexon të gjithë skedarët orarë `hourly-2021.csv` deri `hourly-2025.csv`, i bashkon në një dataset të vetëm dhe i agregon nga nivel orar në nivel ditor.

## Input
Skedarët hyrës:
- `../hourly-2021.csv`
- `../hourly-2022.csv`
- `../hourly-2023.csv`
- `../hourly-2024.csv`
- `../hourly-2025.csv`

Kolonat në skedarët hyrës:
```text
Datetime (UTC)
Country
Zone name
Zone id
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
```

## Output
Skedarët e gjeneruar:
- `merged_hourly_dataset.csv`
- `merged_daily_dataset.csv`
- `schema_audit.csv`

Kolonat në `merged_hourly_dataset.csv`:
```text
Datetime (UTC)
Country
Zone name
Zone id
Carbon intensity gCO₂eq/kWh (direct)
Carbon intensity gCO₂eq/kWh (Life cycle)
Carbon-free energy percentage (CFE%)
Renewable energy percentage (RE%)
Data source
Data estimated
Data estimation method
source_file
```

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

Kolonat në `schema_audit.csv`:
```text
column
dtype
null_count
null_pct
unique_count
unique_ratio
sample_value
high_cardinality_flag
```

## Dataseti që vazhdon në hapin tjetër
Skedari kryesor për Hapin 2 është `merged_daily_dataset.csv`.

## Ekzekutimi
```bash
python step_1.py
```
