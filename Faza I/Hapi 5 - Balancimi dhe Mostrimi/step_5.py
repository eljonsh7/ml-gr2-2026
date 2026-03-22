import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import ADASYN, SMOTE

# Path configurations
INPUT_CSV = Path(__file__).resolve().parent.parent / "Hapi 3 - Diskretizimi" / "dataset_with_target.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def detect_numeric_and_categorical(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    return numeric_columns, categorical_columns

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns, _ = detect_numeric_and_categorical(df)
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
    
    print(f"Removing {consensus_mask.sum()} outliers based on consensus...")
    return df[~consensus_mask].copy()

def build_model_matrix(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    # Drop non-feature columns
    cols_to_drop = [target_column]
    if "source_file" in df.columns: cols_to_drop.append("source_file")
    
    modelling = df.drop(columns=cols_to_drop).copy()
    y = df[target_column].copy()
    
    numeric_cols, categorical_cols = detect_numeric_and_categorical(modelling)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    transformed = preprocessor.fit_transform(modelling)
    X = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out(), index=modelling.index)
    return X, y

def balance_data(X_train, y_train):
    counts = y_train.value_counts()
    share = counts / counts.sum()
    if share.min() >= 0.2: return X_train, y_train, "Skipped (already balanced)"
    sampler = ADASYN(random_state=42, n_neighbors=max(1, counts.min()-1)) if counts.min() < 6 else SMOTE(random_state=42)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name), type(sampler).__name__

def main():
    if not INPUT_CSV.exists():
        print(f"Error: Previous step output not found at {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print("Detecting and removing outliers...")
    df_clean = remove_outliers(df)
    
    print("Building model matrix and splitting...")
    X, y = build_model_matrix(df_clean, "target_quantile_class")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Balancing training set...")
    X_train_bal, y_train_bal, sampler = balance_data(X_train, y_train)
    
    # Save outputs
    X_train_bal.to_csv(OUTPUT_DIR / "train_balanced_features.csv", index=False)
    pd.DataFrame({"target_quantile_class": y_train_bal}).to_csv(OUTPUT_DIR / "train_balanced_target.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "test_features.csv", index=False)
    pd.DataFrame({"target_quantile_class": y_test}).to_csv(OUTPUT_DIR / "test_target.csv", index=False)
    
    counts_after = pd.DataFrame({"target_quantile_class": y_train_bal})["target_quantile_class"].value_counts().reset_index()
    counts_after.columns = ["target_quantile_class", "count"]
    counts_after.to_csv(OUTPUT_DIR / "class_distribution_after_balancing.csv", index=False)
    
    print(f"Step 5 completed. Train size: {len(y_train_bal)} rows")

if __name__ == "__main__":
    main()
