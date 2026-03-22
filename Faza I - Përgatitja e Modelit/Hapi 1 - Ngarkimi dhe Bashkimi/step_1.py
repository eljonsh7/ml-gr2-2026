import pandas as pd
from pathlib import Path
import json
import numpy as np

# Path configurations
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATTERN = "hourly-20*.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def load_dataset(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    datetime_columns = []
    for column in df.columns:
        normalized = column.lower()
        if "date" in normalized or "time" in normalized:
            datetime_columns.append(column)
    return datetime_columns

def parse_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df.copy()
    for column in detect_datetime_columns(parsed):
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce", utc=True)
    for column in parsed.columns:
        if column in detect_datetime_columns(parsed):
            continue
        if parsed[column].dtype == object:
            numeric_version = pd.to_numeric(parsed[column], errors="coerce")
            original_non_null = parsed[column].notna().sum()
            numeric_non_null = numeric_version.notna().sum()
            if original_non_null > 0 and numeric_non_null / original_non_null >= 0.8:
                parsed[column] = numeric_version
    return parsed

def mode_or_unknown(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "Unknown"
    mode = non_null.mode()
    return non_null.iloc[0] if mode.empty else mode.iloc[0]

def aggregate_hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.copy()
    datetime_columns = detect_datetime_columns(daily)
    if not datetime_columns:
        return daily

    time_column = datetime_columns[0]
    daily["date"] = daily[time_column].dt.floor("D")
    if "Data estimated" in daily.columns:
        daily["estimated_hour_flag"] = daily["Data estimated"].astype(int)
    daily["hourly_row_count"] = 1

    group_columns = [column for column in ["Country", "Zone name", "Zone id", "date"] if column in daily.columns]
    aggregations = {}

    for column in daily.columns:
        if column in group_columns or column == time_column:
            continue
        if column == "hourly_row_count":
            aggregations[column] = "sum"
            continue
        if column == "estimated_hour_flag":
            aggregations[column] = "sum"
            continue
        if column in {"source_file"}:
            aggregations[column] = mode_or_unknown
            continue
        if pd.api.types.is_numeric_dtype(daily[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = mode_or_unknown

    aggregated = daily.groupby(group_columns, dropna=False).agg(aggregations).reset_index()
    aggregated["Datetime (UTC)"] = aggregated["date"]
    if "estimated_hour_flag" in aggregated.columns:
        aggregated["estimated_hour_share"] = aggregated["estimated_hour_flag"] / aggregated["hourly_row_count"].replace(0, np.nan)
    return aggregated

def schema_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    row_count = max(len(df), 1)
    for column in df.columns:
        null_count = int(df[column].isna().sum())
        unique_count = int(df[column].nunique(dropna=True))
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "null_count": null_count,
                "null_pct": round((null_count / row_count) * 100, 2),
                "unique_count": unique_count,
                "unique_ratio": round(unique_count / row_count, 4),
                "sample_value": "" if df[column].dropna().empty else str(df[column].dropna().iloc[0]),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["null_pct", "unique_ratio"], ascending=[False, False])
    audit["high_cardinality_flag"] = audit["unique_ratio"] >= 0.9
    return audit

def main():
    paths = sorted(ROOT.glob(INPUT_PATTERN))
    if not paths:
        fallback = ROOT / "initial_dataset.csv"
        if fallback.exists():
            paths = [fallback]
        else:
            print("Error: No input files found.")
            return

    print(f"Loading {len(paths)} files...")
    raw_df = load_dataset(paths)
    raw_df.to_csv(OUTPUT_DIR / "merged_hourly_dataset.csv", index=False)
    
    print("Parsing and aggregating to daily...")
    parsed_df = parse_columns(raw_df)
    daily_df = aggregate_hourly_to_daily(parsed_df)
    daily_df.to_csv(OUTPUT_DIR / "merged_daily_dataset.csv", index=False)
    
    print("Performing schema audit...")
    audit_df = schema_audit(daily_df)
    audit_df.to_csv(OUTPUT_DIR / "schema_audit.csv", index=False)
    
    print(f"Step 1 completed. Daily shape: {daily_df.shape}")

if __name__ == "__main__":
    main()
