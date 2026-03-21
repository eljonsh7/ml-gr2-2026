from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
INPUT_PATTERN = "hourly-20*.csv"
STEP_DIRS = {step: ROOT / f"Step {step}" for step in range(1, 9)}


@dataclass
class StepLog:
    step: str
    detail: str
    rows_before: int
    rows_after: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    for step_dir in STEP_DIRS.values():
        step_dir.mkdir(exist_ok=True)


def write_step_csv(step: int, filename: str, df: pd.DataFrame) -> None:
    df.to_csv(STEP_DIRS[step] / filename, index=False)


def write_step_text(step: int, filename: str, content: str) -> None:
    (STEP_DIRS[step] / filename).write_text(content, encoding="utf-8")


def write_step_json(step: int, filename: str, data: Any) -> None:
    (STEP_DIRS[step] / filename).write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")


def discover_input_paths() -> list[Path]:
    paths = sorted(ROOT.glob(INPUT_PATTERN))
    if paths:
        return paths
    fallback = ROOT / "initial_dataset.csv"
    if fallback.exists():
        return [fallback]
    raise FileNotFoundError("No input dataset files were found in Faza I.")


def load_dataset(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    datetime_columns: list[str] = []
    for column in df.columns:
        normalized = column.lower()
        if "date" in normalized or "time" in normalized:
            datetime_columns.append(column)
    return datetime_columns


def detect_numeric_and_categorical(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    return numeric_columns, categorical_columns


def build_logical_bounds(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    bounds: dict[str, dict[str, Any]] = {}
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


def mode_or_unknown(series: pd.Series) -> Any:
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
    aggregations: dict[str, Any] = {}

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


def apply_constraint_filtering(df: pd.DataFrame, bounds: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, list[StepLog]]:
    filtered = df.copy()
    logs: list[StepLog] = []
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
            logs.append(
                StepLog(
                    step="constraint_filter",
                    detail=f"{column}: {rule['rule']}",
                    rows_before=before,
                    rows_after=after,
                )
            )
    return filtered, logs


def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filled = df.copy()
    actions: list[dict[str, Any]] = []
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
    return filled, pd.DataFrame(actions)


def select_target_column(df: pd.DataFrame) -> str:
    preferred_targets = [
        "Carbon intensity gCO₂eq/kWh (direct)",
        "Carbon intensity gCO₂eq/kWh (Life cycle)",
    ]
    for column in preferred_targets:
        if column in df.columns:
            return column
    numeric_columns, _ = detect_numeric_and_categorical(df)
    if not numeric_columns:
        raise ValueError("No numeric columns available for target creation.")
    return numeric_columns[0]


def create_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    output = df.copy()
    ranked_target = output[target_column].rank(method="first")
    output["target_quantile_class"] = pd.qcut(
        ranked_target,
        q=3,
        labels=["low", "medium", "high"],
    )
    return output


def audit_class_balance(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    counts = df[target_column].value_counts(dropna=False).rename_axis(target_column).reset_index(name="count")
    counts["pct"] = (counts["count"] / len(df) * 100).round(2)
    counts["minority_flag"] = counts["pct"] < 20.0
    return counts


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    datetime_columns = detect_datetime_columns(enriched)
    if not datetime_columns:
        return enriched
    primary_dt = datetime_columns[0]
    enriched["hour"] = enriched[primary_dt].dt.hour
    enriched["day_of_week"] = enriched[primary_dt].dt.dayofweek
    enriched["is_weekend"] = enriched["day_of_week"].isin([5, 6]).astype(int)
    enriched["month"] = enriched[primary_dt].dt.month
    enriched["day"] = enriched[primary_dt].dt.day
    return enriched


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if {
        "Carbon intensity gCO₂eq/kWh (direct)",
        "Carbon intensity gCO₂eq/kWh (Life cycle)",
    }.issubset(enriched.columns):
        enriched["life_cycle_gap"] = (
            enriched["Carbon intensity gCO₂eq/kWh (Life cycle)"]
            - enriched["Carbon intensity gCO₂eq/kWh (direct)"]
        )
    if {
        "Carbon-free energy percentage (CFE%)",
        "Renewable energy percentage (RE%)",
    }.issubset(enriched.columns):
        enriched["non_renewable_clean_gap"] = (
            enriched["Carbon-free energy percentage (CFE%)"]
            - enriched["Renewable energy percentage (RE%)"]
        )
        denominator = enriched["Carbon-free energy percentage (CFE%)"].replace(0, np.nan)
        enriched["renewable_share_within_cfe"] = (
            enriched["Renewable energy percentage (RE%)"] / denominator
        ).fillna(0.0)
    return enriched


def aggregate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    aggregating = df.copy()
    datetime_columns = detect_datetime_columns(aggregating)
    group_columns = [column for column in ["Country", "Zone name", "Zone id"] if column in aggregating.columns]
    if datetime_columns:
        time_column = datetime_columns[0]
        aggregating["year"] = aggregating[time_column].dt.year
        aggregating["month"] = aggregating[time_column].dt.month
        group_columns.extend(["year", "month"])
    numeric_columns, _ = detect_numeric_and_categorical(aggregating)
    aggregation_map = {}
    for column in numeric_columns:
        if column == "target_quantile_class":
            continue
        aggregation_map[column] = ["mean", "sum"]
    aggregation_map["row_count"] = ["count"]
    aggregating["row_count"] = 1
    grouped = aggregating.groupby(group_columns, dropna=False).agg(aggregation_map)
    grouped.columns = ["_".join(part for part in column if part) for column in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns, _ = detect_numeric_and_categorical(df)
    candidates = [
        column
        for column in numeric_columns
        if column != "is_weekend" and not str(column).startswith("target_")
    ]
    numeric_frame = df[candidates].copy()
    outliers = pd.DataFrame(index=df.index)
    if numeric_frame.empty:
        outliers["iqr_outlier_count"] = 0
        outliers["zscore_outlier_count"] = 0
        outliers["isolation_forest_flag"] = 0
        outliers["outlier_consensus_count"] = 0
        outliers["outlier_consensus_flag"] = 0
        return outliers

    q1 = numeric_frame.quantile(0.25)
    q3 = numeric_frame.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_flags = ((numeric_frame.lt(lower)) | (numeric_frame.gt(upper))).astype(int)

    std = numeric_frame.std(ddof=0).replace(0, np.nan)
    zscores = ((numeric_frame - numeric_frame.mean()) / std).abs()
    z_flags = (zscores > 3.0).fillna(False).astype(int)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_frame)
    isolation = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    isolation_raw = isolation.fit_predict(scaled)
    isolation_flag = pd.Series((isolation_raw == -1).astype(int), index=df.index)

    outliers["iqr_outlier_count"] = iqr_flags.sum(axis=1)
    outliers["zscore_outlier_count"] = z_flags.sum(axis=1)
    outliers["isolation_forest_flag"] = isolation_flag
    outliers["outlier_consensus_count"] = (
        (outliers["iqr_outlier_count"] > 0).astype(int)
        + (outliers["zscore_outlier_count"] > 0).astype(int)
        + outliers["isolation_forest_flag"]
    )
    outliers["outlier_consensus_flag"] = (outliers["outlier_consensus_count"] >= 2).astype(int)
    return outliers


def build_model_matrix(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    modelling = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    numeric_columns, categorical_columns = detect_numeric_and_categorical(modelling)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )
    transformed = preprocessor.fit_transform(modelling)
    feature_names = preprocessor.get_feature_names_out()
    X = pd.DataFrame(transformed, columns=feature_names, index=modelling.index)
    return X, y, preprocessor


def balance_training_split(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series, str]:
    class_counts = y_train.value_counts()
    class_share = class_counts / class_counts.sum()
    if class_share.min() >= 0.2:
        return X_train.copy(), y_train.copy(), "Skipped (already balanced)"
    if class_counts.min() < 6:
        sampler = ADASYN(random_state=42, n_neighbors=max(1, class_counts.min() - 1))
        sampler_name = "ADASYN"
    else:
        sampler = SMOTE(random_state=42)
        sampler_name = "SMOTE"
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_resampled, columns=X_train.columns),
        pd.Series(y_resampled, name=y_train.name),
        sampler_name,
    )


def plot_null_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    sample = df.sample(n=min(len(df), 500), random_state=42) if len(df) > 500 else df
    plt.figure(figsize=(12, 6))
    sns.heatmap(sample.isna(), cbar=False, yticklabels=False, cmap="mako")
    plt.title("Missing Values Heatmap (sampled rows)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_class_balance(before: pd.DataFrame, after: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.barplot(data=before, x="target_quantile_class", y="count", ax=axes[0], palette="crest")
    axes[0].set_title("Class Balance Before Resampling")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    sns.barplot(data=after, x="target_quantile_class", y="count", ax=axes[1], palette="flare")
    axes[1].set_title("Class Balance After Resampling")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_report(
    input_files: list[Path],
    raw_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    imputation_log: pd.DataFrame,
    class_distribution_before: pd.DataFrame,
    class_distribution_after: pd.DataFrame,
    aggregation_df: pd.DataFrame,
    sampler_name: str,
    target_source_column: str,
    markdown_path: Path,
) -> None:
    before_table = class_distribution_before.to_string(index=False)
    after_table = class_distribution_after.to_string(index=False)
    report_lines = [
        "# Faza I Data Preparation Report",
        "",
        "## Dataset Overview",
        f"- Source files: `{', '.join(path.name for path in input_files)}`",
        f"- Merged hourly shape: `{raw_df.shape[0]} rows x {raw_df.shape[1]} columns`",
        f"- Daily merged shape: `{daily_df.shape[0]} rows x {daily_df.shape[1]} columns`",
        f"- Cleaned shape: `{cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns`",
        f"- Target source column: `{target_source_column}`",
        "",
        "## Step 1: Initial Data Overview",
        f"- Columns audited: `{audit_df.shape[0]}`",
        f"- High-cardinality columns flagged: `{int(audit_df['high_cardinality_flag'].sum())}`",
        "",
        "## Step 2: Core Data Cleaning",
        f"- Constraint filtering actions: `{cleaning_log.shape[0]}`",
        f"- Imputation actions: `{imputation_log.shape[0]}`",
        "",
        "## Step 3: Class Definition & Imbalance Metrics",
        "```text",
        before_table,
        "```",
        "",
        "## Step 4: Sampling & Balancing",
        f"- Holdout strategy: `train_test_split(stratify=y, test_size=0.2, random_state=42)`",
        f"- Resampler used: `{sampler_name}`",
        "```text",
        after_table,
        "```",
        "",
        "## Step 5: Data Aggregation",
        f"- Aggregated rows: `{aggregation_df.shape[0]}`",
        f"- Aggregated columns: `{aggregation_df.shape[1]}`",
        "",
        "## Step 6: Subsets & Transformations",
        "- Added temporal features: `hour`, `day_of_week`, `is_weekend`, `month`, `day`",
        "- Added derived features when supported by the dataset schema.",
        "- Standardization applied inside the modelling matrix builder.",
        "",
        "## Step 7: Multi-Method Outlier Detection",
        "- Outlier columns saved in the prepared dataset:",
        "- `iqr_outlier_count`",
        "- `zscore_outlier_count`",
        "- `isolation_forest_flag`",
        "- `outlier_consensus_count`",
        "- `outlier_consensus_flag`",
        "",
        "## Step 8: Presentation & Reporting",
        "- Null heatmap and class-balance plots were exported to `output/plots`.",
        "- Structured logs and prepared datasets were exported to `output/`.",
        "",
    ]
    markdown_path.write_text("\n".join(report_lines), encoding="utf-8")


def export_pdf(markdown_path: Path, pdf_path: Path) -> None:
    content = markdown_path.read_text(encoding="utf-8").splitlines()
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        page_text = "\n".join(content[:60])
        ax.text(0.01, 0.99, page_text, va="top", ha="left", family="monospace", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_step_summary(step: int, title: str, lines: list[str]) -> None:
    summary_lines = [f"# Step {step}: {title}", ""] + lines + [""]
    write_step_text(step, "summary.md", "\n".join(summary_lines))


def main() -> None:
    ensure_output_dirs()

    input_files = discover_input_paths()
    raw_df = load_dataset(input_files)
    parsed_hourly_df = parse_columns(raw_df)
    daily_df = aggregate_hourly_to_daily(parsed_hourly_df)

    write_step_csv(1, "merged_hourly_dataset.csv", raw_df)
    write_step_csv(1, "merged_daily_dataset.csv", daily_df)
    audit_df = schema_audit(daily_df)
    write_step_csv(1, "schema_audit.csv", audit_df)
    build_step_summary(
        1,
        "Initial Data Overview",
        [
            f"- Source files: `{', '.join(path.name for path in input_files)}`",
            f"- Merged hourly shape: `{raw_df.shape[0]} rows x {raw_df.shape[1]} columns`",
            f"- Daily merged shape: `{daily_df.shape[0]} rows x {daily_df.shape[1]} columns`",
            f"- High-cardinality columns flagged: `{int(audit_df['high_cardinality_flag'].sum())}`",
        ],
    )

    bounds = build_logical_bounds(daily_df)
    cleaned_df, filter_logs = apply_constraint_filtering(daily_df, bounds)
    cleaned_df, imputation_log = impute_missing_values(cleaned_df)
    write_step_csv(2, "cleaned_dataset.csv", cleaned_df)
    cleaning_log = pd.DataFrame(
        [
            {
                "step": log.step,
                "detail": log.detail,
                "rows_before": log.rows_before,
                "rows_after": log.rows_after,
                "rows_removed": log.rows_removed,
            }
            for log in filter_logs
        ]
    )
    cleaning_log.to_csv(STEP_DIRS[2] / "cleaning_log.csv", index=False)
    imputation_log.to_csv(STEP_DIRS[2] / "imputation_log.csv", index=False)
    write_step_json(2, "logical_bounds.json", bounds)
    build_step_summary(
        2,
        "Core Data Cleaning",
        [
            f"- Constraint filtering actions: `{cleaning_log.shape[0]}`",
            f"- Imputation actions: `{imputation_log.shape[0]}`",
            f"- Output shape: `{cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns`",
        ],
    )

    cleaned_df = add_temporal_features(cleaned_df)
    cleaned_df = add_derived_features(cleaned_df)

    target_source_column = select_target_column(cleaned_df)
    prepared_df = create_target(cleaned_df, target_source_column)
    class_distribution_before = audit_class_balance(prepared_df, "target_quantile_class")
    write_step_csv(3, "dataset_with_target.csv", prepared_df)
    write_step_csv(3, "class_distribution.csv", class_distribution_before)
    build_step_summary(
        3,
        "Class Definition & Imbalance Metrics",
        [
            f"- Target source column: `{target_source_column}`",
            f"- Classes created: `{', '.join(class_distribution_before['target_quantile_class'].astype(str))}`",
            f"- Minority classes under 20%: `{int(class_distribution_before['minority_flag'].sum())}`",
        ],
    )

    outlier_flags = detect_outliers(prepared_df)
    prepared_df = pd.concat([prepared_df, outlier_flags], axis=1)

    aggregation_df = aggregate_dataset(prepared_df)

    X, y, _ = build_model_matrix(prepared_df, "target_quantile_class")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train_balanced, y_train_balanced, sampler_name = balance_training_split(X_train, y_train)
    class_distribution_after = audit_class_balance(
        pd.DataFrame({"target_quantile_class": y_train_balanced}),
        "target_quantile_class",
    )
    pd.DataFrame(X_train_balanced, columns=X_train.columns).to_csv(
        STEP_DIRS[4] / "train_balanced_features.csv",
        index=False,
    )
    pd.DataFrame({"target_quantile_class": y_train_balanced}).to_csv(
        STEP_DIRS[4] / "train_balanced_target.csv",
        index=False,
    )
    X_test.to_csv(STEP_DIRS[4] / "test_features.csv", index=False)
    pd.DataFrame({"target_quantile_class": y_test}).to_csv(STEP_DIRS[4] / "test_target.csv", index=False)
    write_step_csv(4, "class_distribution_after_balancing.csv", class_distribution_after)
    build_step_summary(
        4,
        "Sampling & Balancing",
        [
            f"- Holdout test size: `20%`",
            f"- Resampler used: `{sampler_name}`",
            f"- Balanced train rows: `{len(y_train_balanced)}`",
        ],
    )

    write_step_csv(5, "aggregated_dataset.csv", aggregation_df)
    build_step_summary(
        5,
        "Data Aggregation",
        [
            f"- Aggregated rows: `{aggregation_df.shape[0]}`",
            f"- Aggregated columns: `{aggregation_df.shape[1]}`",
            "- Aggregation level: monthly summaries by country/zone",
        ],
    )

    write_step_csv(6, "feature_engineered_dataset.csv", prepared_df.drop(columns=outlier_flags.columns))
    build_step_summary(
        6,
        "Subsets & Transformations",
        [
            "- Added temporal features: `hour`, `day_of_week`, `is_weekend`, `month`, `day`",
            "- Added derived carbon/renewable gap features",
            "- Standardized model matrix exported in Step 4 train/test splits",
        ],
    )

    write_step_csv(7, "outlier_flags_dataset.csv", prepared_df)
    build_step_summary(
        7,
        "Multi-Method Outlier Detection",
        [
            f"- Consensus outliers flagged: `{int(prepared_df['outlier_consensus_flag'].sum())}`",
            "- Detectors used: IQR, Z-score, Isolation Forest",
            "- Consensus rule: at least 2 detectors agree",
        ],
    )

    plot_null_heatmap(daily_df, PLOTS_DIR / "null_heatmap.png")
    plot_class_balance(
        class_distribution_before,
        class_distribution_after,
        PLOTS_DIR / "class_balance_comparison.png",
    )
    plot_null_heatmap(daily_df, STEP_DIRS[8] / "null_heatmap.png")
    plot_class_balance(
        class_distribution_before,
        class_distribution_after,
        STEP_DIRS[8] / "class_balance_comparison.png",
    )

    raw_df.to_csv(OUTPUT_DIR / "merged_hourly_dataset.csv", index=False)
    daily_df.to_csv(OUTPUT_DIR / "merged_daily_dataset.csv", index=False)
    prepared_df.to_csv(OUTPUT_DIR / "prepared_dataset.csv", index=False)
    aggregation_df.to_csv(OUTPUT_DIR / "aggregated_dataset.csv", index=False)
    cleaning_log.to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False)
    imputation_log.to_csv(OUTPUT_DIR / "imputation_log.csv", index=False)
    audit_df.to_csv(OUTPUT_DIR / "schema_audit.csv", index=False)
    pd.DataFrame(X_train_balanced, columns=X_train.columns).to_csv(
        OUTPUT_DIR / "train_balanced_features.csv",
        index=False,
    )
    pd.DataFrame({"target_quantile_class": y_train_balanced}).to_csv(
        OUTPUT_DIR / "train_balanced_target.csv",
        index=False,
    )
    X_test.to_csv(OUTPUT_DIR / "test_features.csv", index=False)
    pd.DataFrame({"target_quantile_class": y_test}).to_csv(OUTPUT_DIR / "test_target.csv", index=False)
    (OUTPUT_DIR / "logical_bounds.json").write_text(json.dumps(bounds, default=str, indent=2), encoding="utf-8")

    report_path = OUTPUT_DIR / "phase1_report.md"
    build_report(
        input_files=input_files,
        raw_df=raw_df,
        daily_df=daily_df,
        cleaned_df=prepared_df,
        audit_df=audit_df,
        cleaning_log=cleaning_log,
        imputation_log=imputation_log,
        class_distribution_before=class_distribution_before,
        class_distribution_after=class_distribution_after,
        aggregation_df=aggregation_df,
        sampler_name=sampler_name,
        target_source_column=target_source_column,
        markdown_path=report_path,
    )
    export_pdf(report_path, OUTPUT_DIR / "phase1_report.pdf")
    build_report(
        input_files=input_files,
        raw_df=raw_df,
        daily_df=daily_df,
        cleaned_df=prepared_df,
        audit_df=audit_df,
        cleaning_log=cleaning_log,
        imputation_log=imputation_log,
        class_distribution_before=class_distribution_before,
        class_distribution_after=class_distribution_after,
        aggregation_df=aggregation_df,
        sampler_name=sampler_name,
        target_source_column=target_source_column,
        markdown_path=STEP_DIRS[8] / "phase1_report.md",
    )
    export_pdf(STEP_DIRS[8] / "phase1_report.md", STEP_DIRS[8] / "phase1_report.pdf")
    build_step_summary(
        8,
        "Presentation & Reporting",
        [
            "- Exported markdown and PDF report",
            "- Exported null heatmap and class-balance comparison chart",
            f"- Final prepared dataset rows: `{prepared_df.shape[0]}`",
        ],
    )

    print("Phase 1 data preparation completed.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
