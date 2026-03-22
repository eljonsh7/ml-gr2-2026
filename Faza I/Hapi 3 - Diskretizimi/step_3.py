import pandas as pd
import numpy as np
from pathlib import Path

# Path configurations
INPUT_CSV = Path(__file__).resolve().parent.parent / "Hapi 2 - Pastrimi i te dhenave" / "cleaned_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = "Datetime (UTC)"
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df["day"] = df[time_col].dt.day
        df["month"] = df[time_col].dt.month
        df["year"] = df[time_col].dt.year
        df["hour"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Remove original time column and any intermediate 'date' column
        df = df.drop(columns=[time_col])
        if "date" in df.columns: df = df.drop(columns=["date"])
        
        # Reorder to put day, month, year at the front
        front_cols = ["day", "month", "year"]
        other_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + other_cols]
        
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Carbon Gap: Life cycle minus direct
    if "Carbon intensity gCO₂eq/kWh (Life cycle)" in df.columns and "Carbon intensity gCO₂eq/kWh (direct)" in df.columns:
        df["carbon_intensity_gap"] = df["Carbon intensity gCO₂eq/kWh (Life cycle)"] - df["Carbon intensity gCO₂eq/kWh (direct)"]
    
    # Renewable Gap check
    if "Carbon-free energy percentage (CFE%)" in df.columns and "Renewable energy percentage (RE%)" in df.columns:
        df["cfe_re_gap"] = df["Carbon-free energy percentage (CFE%)"] - df["Renewable energy percentage (RE%)"]
    
    return df

def select_target_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "intensity" in c.lower() and "direct" in c.lower()]
    return candidates[0] if candidates else df.columns[0]

def create_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    # Create 3 quantiles: Low, Medium, High
    df["target_quantile_class"] = pd.qcut(df[target_col], q=3, labels=["Low", "Medium", "High"])
    return df

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Previous step output not found at {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    
    print("Removing redundant columns (Country, Zone name, Zone id)...")
    cols_to_drop = ["Country", "Zone name", "Zone id"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    print("Adding temporal (day, month, year) and derived features...")
    df = add_temporal_features(df)
    df = add_derived_features(df)
    
    print("Creating target classes (Low, Medium, High)...")
    target_col = select_target_column(df)
    df = create_target(df, target_col)
    
    # Save outputs
    df.to_csv(OUTPUT_DIR / "dataset_with_target.csv", index=False)
    
    counts = df["target_quantile_class"].value_counts().reset_index()
    counts.columns = ["target_quantile_class", "count"]
    counts.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
    
    print(f"Step 3 completed. Target column: {target_col}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    main()
