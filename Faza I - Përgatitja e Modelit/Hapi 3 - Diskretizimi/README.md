# Hapi 3: Diskretizimi dhe Inxhinieria e Veçorive

## Qëllimi
Ky hap ndërton targetin e klasifikimit dhe krijon veçoritë kohore dhe të derivuara që përdoren në hapat pasues.

## Input
Skedari hyrës:
- `../Hapi 2 - Pastrimi i të dhënave/cleaned_dataset.csv`

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
- `dataset_with_target.csv`
- `class_distribution.csv`

Kolonat në `dataset_with_target.csv`:
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

Kolonat në `class_distribution.csv`:
```text
target_quantile_class
count
```

## Dataseti që vazhdon në hapin tjetër
Skedari kryesor për Hapin 4 është `dataset_with_target.csv`.

## Ekzekutimi
```bash
python step_3.py
```
