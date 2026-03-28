# Hapi 2: Pastrimi i të Dhënave

## Qëllimi
Ky hap kontrollon kufijtë logjikë, filtron rreshtat e pavlefshëm dhe aplikon imputim nëse ka vlera munguese.

## Input
Skedari hyrës:
- `../Hapi 1 - Ngarkimi dhe Bashkimi/merged_daily_dataset.csv`

Kolonat në input:
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

## Output
Skedarët e gjeneruar:
- `cleaned_dataset.csv`
- `cleaning_log.csv`
- `imputation_log.csv`
- `logical_bounds.json`

Kolonat në `cleaned_dataset.csv`:
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

Kolonat në `cleaning_log.csv`:
```text
step
detail
rows_before
rows_after
rows_removed
```

Kolonat në `imputation_log.csv`:
```text
column
strategy
fill_value
```

## Dataseti që vazhdon në hapin tjetër
Skedari kryesor për Hapin 3 është `cleaned_dataset.csv`.

## Ekzekutimi
```bash
python step_2.py
```
