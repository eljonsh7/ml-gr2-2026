import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Path configurations
INPUT_CSV = Path(__file__).resolve().parent.parent / "Hapi 3 - Diskretizimi" / "dataset_with_target.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    candidates = [c for c in numeric_columns if c not in ["is_weekend", "hour", "day_of_week", "month", "day", "year"]]
    numeric_frame = df[candidates].copy()
    
    if numeric_frame.empty: return df

    # IQR
    q1, q3 = numeric_frame.quantile(0.25), numeric_frame.quantile(0.75)
    iqr = q3 - q1
    iqr_flags = ((numeric_frame < (q1 - 1.5 * iqr)) | (numeric_frame > (q3 + 1.5 * iqr))).sum(axis=1) > 0
    
    # Z-score
    std = numeric_frame.std(ddof=0).replace(0, np.nan)
    z_flags = (((numeric_frame - numeric_frame.mean()) / std).abs() > 3.0).fillna(False).sum(axis=1) > 0
    
    # Isolation Forest
    scaled = StandardScaler().fit_transform(numeric_frame.fillna(0))
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso_flags = iso.fit_predict(scaled) == -1
    
    # Consensus: 2 out of 3
    consensus_mask = (iqr_flags.astype(int) + z_flags.astype(int) + iso_flags.astype(int)) >= 2
    return df[~consensus_mask].copy()

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Previous step output not found at {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print("Finalizing dataset: Removing outliers and ensuring clean structure...")
    
    df_clean = remove_outliers(df)
    
    # Save output
    df_clean.to_csv(OUTPUT_DIR / "feature_engineered_dataset.csv", index=False)
    print(f"Step 7 completed. Final dataset rows: {len(df_clean)} (Outliers removed)")

if __name__ == "__main__":
    main()
