# Faza I: Përgatitja dhe Inxhinieria e të Dhënave

Kjo fazë e transformon datasetin orar të karbonit për Kosovën në një dataset ditor të pastruar, të inxhinieruar dhe gati për përdorim analitik dhe modelim.

## Inputi fillestar i Fazës I
Skedarët hyrës:
- `hourly-2021.csv`
- `hourly-2022.csv`
- `hourly-2023.csv`
- `hourly-2024.csv`
- `hourly-2025.csv`

Kolonat fillestare:
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

## Rrjedha e 8 hapave

### Hapi 1 - Ngarkimi dhe Bashkimi
- Input: 11 kolona orare
- Output kryesor: `merged_daily_dataset.csv`

Kolonat e output-it:
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

### Hapi 2 - Pastrimi i të Dhënave
- Input: `merged_daily_dataset.csv`
- Output kryesor: `cleaned_dataset.csv`

Kolonat e output-it:
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

### Hapi 3 - Diskretizimi
- Input: `cleaned_dataset.csv`
- Output kryesor: `dataset_with_target.csv`

Kolonat e output-it:
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

### Hapi 4 - Detektimi i Outliers
- Input: `dataset_with_target.csv`
- Output kryesor: `outlier_flags_dataset.csv`

Kolonat shtesë të output-it:
```text
iqr_outlier_count
zscore_outlier_count
isolation_forest_flag
outlier_consensus_count
outlier_consensus_flag
```

### Hapi 5 - Balancimi dhe Mostrimi
- Input: `outlier_flags_dataset.csv`
- Output-e kryesore:
  - `train_balanced_features.csv`
  - `train_balanced_target.csv`
  - `test_features.csv`
  - `test_target.csv`

Kolonat e matricës së feature-ve:
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

Kolona e target-it:
```text
target_quantile_class
```

### Hapi 6 - Agregimi
- Input: `feature_engineered_dataset.csv`
- Output kryesor: `aggregated_dataset.csv`

Kolonat e output-it:
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

### Hapi 7 - Finalizimi i Datasetit
- Input: `outlier_flags_dataset.csv`
- Output kryesor: `feature_engineered_dataset.csv`

Kolonat e output-it:
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

### Hapi 8 - Raporti Final
- Input: rezultatet nga Hapi 1, Hapi 3, Hapi 5 dhe Hapi 7
- Output: `null_heatmap.png`, `class_balance_comparison.png`, `phase1_report.md`, `phase1_report.pdf`

Ky hap nuk prodhon dataset të ri CSV.

## Cili është dataseti final?
Dataseti final kryesor i Fazës I është:
- `Hapi 7 - Finalizimi i Datasetit/feature_engineered_dataset.csv`

Ky është master dataset-i ditor i përgatitur për analiza të mëtejshme dhe si referencë kryesore e Fazës I.

Datasetet e gatshme për modelim në Fazën II janë:
- `Hapi 5 - Balancimi dhe Mostrimi/train_balanced_features.csv`
- `Hapi 5 - Balancimi dhe Mostrimi/train_balanced_target.csv`
- `Hapi 5 - Balancimi dhe Mostrimi/test_features.csv`
- `Hapi 5 - Balancimi dhe Mostrimi/test_target.csv`
