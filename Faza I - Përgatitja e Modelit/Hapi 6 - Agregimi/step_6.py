import pandas as pd
import numpy as np
from pathlib import Path

# Path configurations
INPUT_CSV = Path(__file__).resolve().parent.parent / "Hapi 3 - Diskretizimi" / "dataset_with_target.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def aggregate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    aggregating = df.copy()
    group_columns = ["year", "month"]
    # Check if year/month exist, otherwise use what is there
    group_columns = [c for c in group_columns if c in aggregating.columns]
    
    group_columns = ["year", "month"]
    numeric_columns = aggregating.select_dtypes(include=["number"]).columns.tolist()
    
    # Exclude columns that don't make sense to mean/sum (like hour, day, is_weekend if we want)
    # But usually we aggregate all numerics.
    aggregation_map = {col: ["mean", "sum"] for col in numeric_columns if col not in group_columns}
    aggregating["row_count_in_month"] = 1
    aggregation_map["row_count_in_month"] = ["count"]
    
    grouped = aggregating.groupby(group_columns, dropna=False).agg(aggregation_map)
    grouped.columns = ["_".join(part for part in column if part) for column in grouped.columns.to_flat_index()]
    return grouped.reset_index()

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Previous step output not found at {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print("Aggregating dataset to monthly level (grouping by year, month)...")
    agg_df = aggregate_dataset(df)
    
    # Save output
    agg_df.to_csv(OUTPUT_DIR / "aggregated_dataset.csv", index=False)
    print(f"Step 6 completed. Aggregated shape: {agg_df.shape}")

if __name__ == "__main__":
    main()
