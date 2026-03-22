import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Path configurations
INPUT_CSV = Path(__file__).resolve().parent.parent / "Hapi 3 - Diskretizimi" / "dataset_with_target.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def detect_numeric_and_categorical(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    return numeric_columns, categorical_columns

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns, _ = detect_numeric_and_categorical(df)
    candidates = [c for c in numeric_columns if c not in ["is_weekend", "hour", "day_of_week", "month", "day", "year"]]
    numeric_frame = df[candidates].copy()
    outliers = pd.DataFrame(index=df.index)
    
    if numeric_frame.empty:
        for col in ["iqr_outlier_count", "zscore_outlier_count", "isolation_forest_flag", "outlier_consensus_count", "outlier_consensus_flag"]:
            outliers[col] = 0
        return outliers

    # IQR
    q1, q3 = numeric_frame.quantile(0.25), numeric_frame.quantile(0.75)
    iqr = q3 - q1
    iqr_flags = ((numeric_frame < (q1 - 1.5 * iqr)) | (numeric_frame > (q3 + 1.5 * iqr))).astype(int)
    
    # Z-score
    std = numeric_frame.std(ddof=0).replace(0, np.nan)
    z_flags = (((numeric_frame - numeric_frame.mean()) / std).abs() > 3.0).fillna(False).astype(int)
    
    # Isolation Forest
    scaled = StandardScaler().fit_transform(numeric_frame.fillna(0))
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso_flag = (iso.fit_predict(scaled) == -1).astype(int)
    
    outliers["iqr_outlier_count"] = iqr_flags.sum(axis=1)
    outliers["zscore_outlier_count"] = z_flags.sum(axis=1)
    outliers["isolation_forest_flag"] = iso_flag
    outliers["outlier_consensus_count"] = (outliers["iqr_outlier_count"] > 0).astype(int) + (outliers["zscore_outlier_count"] > 0).astype(int) + outliers["isolation_forest_flag"]
    outliers["outlier_consensus_flag"] = (outliers["outlier_consensus_count"] >= 2).astype(int)
    return outliers

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Previous step output not found at {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print("Performing multi-method outlier detection and flagging...")
    
    outlier_flags = detect_outliers(df)
    df_with_outliers = pd.concat([df, outlier_flags], axis=1)
    
    # Save output
    df_with_outliers.to_csv(OUTPUT_DIR / "outlier_flags_dataset.csv", index=False)
    print(f"Step 4 completed. Consensus outliers identified: {int(outlier_flags['outlier_consensus_flag'].sum())}")

if __name__ == "__main__":
    main()
