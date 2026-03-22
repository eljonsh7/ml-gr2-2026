import pandas as pd
from pathlib import Path
import json
import numpy as np

# Path configurations
INPUT_CSV = Path(__file__).resolve().parent.parent / "Hapi 1 - Ngarkimi dhe Bashkimi" / "merged_daily_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    datetime_columns = []
    for column in df.columns:
        normalized = column.lower()
        if "date" in normalized or "time" in normalized:
            datetime_columns.append(column)
    return datetime_columns

def detect_numeric_and_categorical(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    return numeric_columns, categorical_columns

def build_logical_bounds(df: pd.DataFrame) -> dict[str, dict[str, any]]:
    bounds = {}
    for column in df.columns:
        lowered = column.lower()
        if "percentage" in lowered or "%" in lowered:
            bounds[column] = {"min": 0.0, "max": 100.0, "rule": "percentage_range"}
        elif "intensity" in lowered:
            bounds[column] = {"min": 0.0, "max": None, "rule": "non_negative_intensity"}
        elif "count" in lowered:
            bounds[column] = {"min": 0.0, "max": None, "rule": "non_negative_count"}
    for column in detect_datetime_columns(df):
        bounds[column] = {"min": None, "max": pd.Timestamp.now("UTC"), "rule": "not_in_future"}
    return bounds

def apply_constraint_filtering(df: pd.DataFrame, bounds: dict[str, dict[str, any]]) -> tuple[pd.DataFrame, list[dict[str, any]]]:
    filtered = df.copy()
    logs = []
    for column, rule in bounds.items():
        if column not in filtered.columns:
            continue
        before = len(filtered)
        if isinstance(rule["max"], pd.Timestamp) or isinstance(rule["min"], pd.Timestamp):
            comparable = pd.to_datetime(filtered[column], errors="coerce", utc=True)
        else:
            comparable = pd.to_numeric(filtered[column], errors="coerce")
        
        mask = pd.Series(True, index=filtered.index)
        non_null_mask = comparable.notna()
        if rule["min"] is not None:
            mask &= (~non_null_mask) | (comparable >= rule["min"])
        if rule["max"] is not None:
            mask &= (~non_null_mask) | (comparable <= rule["max"])
        
        filtered[column] = comparable.combine_first(filtered[column])
        filtered = filtered.loc[mask].copy()
        after = len(filtered)
        if before != after:
            logs.append({"step": "constraint_filter", "detail": f"{column}: {rule['rule']}", "rows_before": before, "rows_after": after, "rows_removed": before - after})
    return filtered, logs

def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, any]]]:
    filled = df.copy()
    actions = []
    numeric_columns, categorical_columns = detect_numeric_and_categorical(filled)
    for column in numeric_columns:
        if filled[column].isna().any():
            value = float(filled[column].median())
            filled[column] = filled[column].fillna(value)
            actions.append({"column": column, "strategy": "median", "fill_value": value})
    for column in categorical_columns:
        if filled[column].isna().any():
            mode = filled[column].mode(dropna=True)
            value = "Unknown" if mode.empty else str(mode.iloc[0])
            filled[column] = filled[column].fillna(value)
            actions.append({"column": column, "strategy": "mode", "fill_value": value})
    return filled, actions

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Previous step output not found at {INPUT_CSV}")
        return

    daily_df = pd.read_csv(INPUT_CSV)
    
    # Ensure datetime columns are parsed
    datetime_columns = detect_datetime_columns(daily_df)
    for column in datetime_columns:
        daily_df[column] = pd.to_datetime(daily_df[column], errors="coerce", utc=True)

    print("Applying logical bounds and constraint filtering...")
    bounds = build_logical_bounds(daily_df)
    cleaned_df, filter_logs = apply_constraint_filtering(daily_df, bounds)
    
    print("Imputing missing values if any...")
    cleaned_df, imputation_log = impute_missing_values(cleaned_df)
    
    # Save outputs
    cleaned_df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    pd.DataFrame(filter_logs).to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False)
    pd.DataFrame(imputation_log).to_csv(OUTPUT_DIR / "imputation_log.csv", index=False)
    
    with open(OUTPUT_DIR / "logical_bounds.json", "w") as f:
        json.dump(bounds, f, default=str, indent=2)

    print(f"Step 2 completed. Cleaned shape: {cleaned_df.shape}")

if __name__ == "__main__":
    main()
