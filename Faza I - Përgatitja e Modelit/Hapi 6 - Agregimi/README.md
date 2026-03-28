# Hapi 6: Agregimi

## Qëllimi
Ky hap krijon një pamje mujore të datasetit ditor për analiza makro dhe krahasim të trendeve sezonale.

## Input
Skedari hyrës:
- `../Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

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
- `aggregated_dataset.csv`

Kolonat në `aggregated_dataset.csv`:
```text
year
month
day_mean
day_sum
Carbon intensity gCO₂eq/kWh (direct)_mean
Carbon intensity gCO₂eq/kWh (direct)_sum
Carbon intensity gCO₂eq/kWh (Life cycle)_mean
Carbon intensity gCO₂eq/kWh (Life cycle)_sum
Carbon-free energy percentage (CFE%)_mean
Carbon-free energy percentage (CFE%)_sum
Renewable energy percentage (RE%)_mean
Renewable energy percentage (RE%)_sum
Data estimated_mean
Data estimated_sum
estimated_hour_flag_mean
estimated_hour_flag_sum
hourly_row_count_mean
hourly_row_count_sum
estimated_hour_share_mean
estimated_hour_share_sum
hour_mean
hour_sum
day_of_week_mean
day_of_week_sum
is_weekend_mean
is_weekend_sum
carbon_intensity_gap_mean
carbon_intensity_gap_sum
cfe_re_gap_mean
cfe_re_gap_sum
row_count_in_month_count
```

## Përdorimi
Ky skedar është dataset analitik për vizualizime dhe analiza mujore, jo dataseti final kryesor për modelim.

## Ekzekutimi
```bash
python step_6.py
```
