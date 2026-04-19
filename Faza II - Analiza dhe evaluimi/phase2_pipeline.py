"""
Faza II: Trajnimi i Modelit — Comprehensive ML Pipeline
========================================================
Starts from the final processed dataset produced in Faza I and performs:
- train/test split
- preprocessing (standardization + one-hot encoding)
- optional class balancing on the training split only
- supervised learning experiments
- learning theory visualizations
- unsupervised clustering on processed features
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
PHASE1_FINAL_DATASET = (
    ROOT.parent
    / "Faza I - Përgatitja e Modelit"
    / "Hapi 7 - Finalizimi i Datasetit"
    / "feature_engineered_dataset.csv"
)
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

PLOT_DPI = 180
RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="deep")


def load_phase1_dataset() -> pd.DataFrame:
    if not PHASE1_FINAL_DATASET.exists():
        raise FileNotFoundError(f"Phase I final dataset not found: {PHASE1_FINAL_DATASET}")
    return pd.read_csv(PHASE1_FINAL_DATASET)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_column = "target_quantile_class"
    if target_column not in df.columns:
        raise ValueError(f"Target column `{target_column}` not found in Phase I final dataset.")
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )


def preprocess_splits(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test, columns=feature_names, index=X_test_raw.index)
    return X_train_df, X_test_df, preprocessor


def balance_training_split(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series, str]:
    class_counts = y_train.value_counts()
    class_share = class_counts / class_counts.sum()
    if class_share.min() >= 0.2:
        return X_train.copy(), y_train.copy(), "Skipped (already balanced)"
    if class_counts.min() < 6:
        sampler = ADASYN(random_state=RANDOM_STATE, n_neighbors=max(1, class_counts.min() - 1))
        sampler_name = "ADASYN"
    else:
        sampler = SMOTE(random_state=RANDOM_STATE)
        sampler_name = "SMOTE"
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_resampled, columns=X_train.columns),
        pd.Series(y_resampled, name=y_train.name),
        sampler_name,
    )


def save_phase2_inputs(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_processed: pd.DataFrame,
    X_test_processed: pd.DataFrame,
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
) -> None:
    X_train_raw.to_csv(OUTPUT / "phase2_train_raw_features.csv", index=False)
    X_test_raw.to_csv(OUTPUT / "phase2_test_raw_features.csv", index=False)
    pd.DataFrame({"target_quantile_class": y_train}).to_csv(OUTPUT / "phase2_train_raw_target.csv", index=False)
    pd.DataFrame({"target_quantile_class": y_test}).to_csv(OUTPUT / "phase2_test_target.csv", index=False)
    X_train_processed.to_csv(OUTPUT / "phase2_train_processed_features.csv", index=False)
    X_test_processed.to_csv(OUTPUT / "phase2_test_processed_features.csv", index=False)
    X_train_balanced.to_csv(OUTPUT / "train_balanced_features.csv", index=False)
    pd.DataFrame({"target_quantile_class": y_train_balanced}).to_csv(OUTPUT / "train_balanced_target.csv", index=False)


def define_models_and_params() -> dict[str, dict[str, object]]:
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"),
            "params": {"C": [0.01, 0.1, 1, 10]},
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            "params": {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        },
        "SVM (Linear)": {
            "estimator": SVC(kernel="linear", random_state=RANDOM_STATE),
            "params": {"C": [0.1, 1, 10]},
        },
        "SVM (RBF)": {
            "estimator": SVC(kernel="rbf", random_state=RANDOM_STATE),
            "params": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        },
        "Neural Network (MLP)": {
            "estimator": MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
            "params": {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [0.0001, 0.001]},
        },
    }


def train_all_supervised(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    models = define_models_and_params()
    results = []
    fitted_models: dict[str, object] = {}
    reports_text = []

    for name, spec in models.items():
        print(f"  Training {name} ...")
        grid = GridSearchCV(
            spec["estimator"],
            spec["params"],
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        fitted_models[name] = best

        y_pred = best.predict(X_test)
        results.append(
            {
                "Model": name,
                "Best Params": json.dumps(grid.best_params_),
                "CV F1 (macro)": round(grid.best_score_, 4),
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision (macro)": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "Recall (macro)": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "F1 (macro)": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            }
        )

        cm = confusion_matrix(y_test, y_pred, labels=best.classes_)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best.classes_, yticklabels=best.classes_, ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT / f"confusion_matrix_{safe_name}.png", dpi=PLOT_DPI)
        plt.close(fig)

        report = classification_report(y_test, y_pred, target_names=[str(c) for c in best.classes_])
        reports_text.append(f"{'=' * 60}\n{name}\nBest params: {grid.best_params_}\n{'=' * 60}\n{report}\n")

    (OUTPUT / "classification_reports.txt").write_text("\n".join(reports_text), encoding="utf-8")
    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(OUTPUT / "model_results.csv")
    return results_df, fitted_models


def plot_algorithm_comparison(results_df: pd.DataFrame) -> None:
    metric_cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    ax = results_df[metric_cols].plot(kind="bar", figsize=(12, 6), colormap="tab10", edgecolor="black", linewidth=0.5)
    ax.set_title("Algorithm Comparison — Supervised Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "algorithm_comparison.png", dpi=PLOT_DPI)
    plt.close()


def plot_feature_importance(rf_model: RandomForestClassifier, X_train: pd.DataFrame) -> None:
    importances = rf_model.feature_importances_
    feature_names = [c.replace("numeric__", "").replace("categorical__", "") for c in X_train.columns]
    indices = np.argsort(importances)[-15:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(indices)), importances[indices], color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT / "feature_importance_rf.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_learning_curves(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    estimator = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=1)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        cv=3,
        scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation Score")
    ax.set_title("Learning Curves — Random Forest (Overfitting Analysis)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score (macro)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / "learning_curves.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_regularization_effect(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    c_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
    train_scores = []
    test_scores = []
    for c in c_values:
        lr = LogisticRegression(C=c, max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
        lr.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, lr.predict(X_train), average="macro", zero_division=0))
        test_scores.append(f1_score(y_test, lr.predict(X_test), average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(c_values, train_scores, "o-", color="blue", label="Train F1")
    ax.plot(c_values, test_scores, "s-", color="red", label="Test F1")
    ax.set_xscale("log")
    ax.set_xlabel("Regularization Parameter C (log scale)")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Regularization Effect — Logistic Regression", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvspan(c_values[0], 0.05, alpha=0.08, color="blue")
    ax.axvspan(20, c_values[-1], alpha=0.08, color="red")
    ax.text(0.005, min(train_scores) * 0.98, "← Strong regularization\n    (Underfitting)", fontsize=8, color="blue")
    ax.text(30, max(test_scores) * 0.98, "Weak regularization →\n    (Overfitting risk)", fontsize=8, color="red")
    plt.tight_layout()
    fig.savefig(OUTPUT / "regularization_effect.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_mlp_loss_curve(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        alpha=0.001,
    )
    mlp.fit(X_train, y_encoded)

    if getattr(mlp, "loss_curve_", None):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(mlp.loss_curve_, color="purple", linewidth=2)
        ax.set_title("Neural Network Training Loss — Gradient Descent Convergence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch (Iteration)")
        ax.set_ylabel("Loss (Cost Function)")
        ax.grid(True, alpha=0.3)
        ax.annotate(
            "Convergence",
            xy=(len(mlp.loss_curve_) - 1, mlp.loss_curve_[-1]),
            xytext=(-80, 30),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
            color="purple",
        )
        plt.tight_layout()
        fig.savefig(OUTPUT / "mlp_loss_curve.png", dpi=PLOT_DPI)
        plt.close(fig)

    if getattr(mlp, "validation_scores_", None):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(mlp.validation_scores_, color="green", linewidth=2)
        ax.set_title("Neural Network Validation Score per Epoch", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUTPUT / "mlp_validation_curve.png", dpi=PLOT_DPI)
        plt.close(fig)


def run_unsupervised(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series):
    print("  Running Unsupervised Learning ...")
    inertias = []
    sil_scores = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_train)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_train, labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), inertias, "o-", color="steelblue", linewidth=2)
    axes[0].set_title("Elbow Method — K-Means", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(list(k_range), sil_scores, "s-", color="coral", linewidth=2)
    axes[1].set_title("Silhouette Scores — K-Means", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / "elbow_and_silhouette.png", dpi=PLOT_DPI)
    plt.close(fig)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    km_labels_test = kmeans.fit_predict(X_test)
    km_sil = silhouette_score(X_test, km_labels_test)

    agglo = AgglomerativeClustering(n_clusters=3)
    ag_labels_test = agglo.fit_predict(X_test)
    ag_sil = silhouette_score(X_test, ag_labels_test)

    cluster_results = pd.DataFrame(
        {
            "Algorithm": ["K-Means (k=3)", "Agglomerative (k=3)"],
            "Silhouette Score": [round(km_sil, 4), round(ag_sil, 4)],
        }
    )
    cluster_results.to_csv(OUTPUT / "clustering_results.csv", index=False)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_test_pca = pca.fit_transform(X_test)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    scatter1 = axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=km_labels_test, cmap="viridis", alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
    axes[0].set_title("PCA — K-Means Clusters", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    scatter2 = axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=ag_labels_test, cmap="magma", alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
    axes[1].set_title("PCA — Agglomerative Clusters", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.colorbar(scatter2, ax=axes[1], label="Cluster")

    ordered_labels = list(pd.Series(y_test).astype(str).sort_values().unique())
    label_map = {label: idx for idx, label in enumerate(ordered_labels)}
    y_numeric = y_test.astype(str).map(label_map).values
    scatter3 = axes[2].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_numeric, cmap="coolwarm", alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
    axes[2].set_title("PCA — True Labels", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    axes[2].set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    cbar = plt.colorbar(scatter3, ax=axes[2], ticks=list(label_map.values()))
    cbar.ax.set_yticklabels(ordered_labels)

    plt.tight_layout()
    fig.savefig(OUTPUT / "pca_clusters_comparison.png", dpi=PLOT_DPI)
    plt.close(fig)
    return cluster_results, explained


def write_results_summary(
    raw_df: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_train_processed: pd.DataFrame,
    sampler_name: str,
    results_df: pd.DataFrame,
    cluster_results: pd.DataFrame,
    pca_variance: np.ndarray,
) -> None:
    top_model = results_df["F1 (macro)"].idxmax()
    supervised_table = results_df.to_string()
    unsupervised_table = cluster_results.to_string(index=False)
    summary = [
        "# Phase II Results Summary",
        "",
        "## Dataset Flow",
        f"- Input dataset: `{PHASE1_FINAL_DATASET}`",
        f"- Raw Phase I final shape: `{raw_df.shape[0]} rows x {raw_df.shape[1]} columns`",
        f"- Train split: `{X_train_raw.shape[0]} rows`",
        f"- Test split: `{X_test_raw.shape[0]} rows`",
        f"- Processed feature count after encoding/scaling: `{X_train_processed.shape[1]}`",
        f"- Training balance strategy: `{sampler_name}`",
        "",
        "## Supervised Results",
        "```text",
        supervised_table,
        "```",
        "",
        f"- Best model by F1 (macro): `{top_model}`",
        "",
        "## Unsupervised Results",
        "```text",
        unsupervised_table,
        "```",
        "",
        f"- PCA explained variance (PC1 + PC2): `{(pca_variance.sum() * 100):.2f}%`",
        "",
    ]
    (OUTPUT / "results_summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("FAZA II — TRAJNIMI I MODELIT")
    print("=" * 60)

    print("\n[0] Loading final processed dataset from Faza I ...")
    raw_df = load_phase1_dataset()
    X_raw, y = split_features_and_target(raw_df)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_processed, X_test_processed, _ = preprocess_splits(X_train_raw, X_test_raw)
    X_train_balanced, y_train_balanced, sampler_name = balance_training_split(X_train_processed, y_train)
    save_phase2_inputs(
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        X_train_balanced,
        y_train_balanced,
    )
    print(f"    Train raw: {X_train_raw.shape[0]} rows x {X_train_raw.shape[1]} columns")
    print(f"    Test raw:  {X_test_raw.shape[0]} rows x {X_test_raw.shape[1]} columns")
    print(f"    Processed train: {X_train_processed.shape[0]} rows x {X_train_processed.shape[1]} features")
    print(f"    Balance strategy: {sampler_name}")
    print(f"    Classes: {sorted(y_train.astype(str).unique())}")

    print("\n[1] SUPERVISED LEARNING — Training 6 algorithms with GridSearchCV ...")
    results_df, fitted = train_all_supervised(X_train_balanced, y_train_balanced, X_test_processed, y_test)
    print("\n    Results:")
    print(results_df[["Accuracy", "F1 (macro)", "F1 (weighted)"]].to_string())

    print("\n[2] Generating algorithm comparison chart ...")
    plot_algorithm_comparison(results_df)

    print("\n[3] Generating feature importance plot (Random Forest) ...")
    plot_feature_importance(fitted["Random Forest"], X_train_balanced)

    print("\n[4] Generating learning curves ...")
    plot_learning_curves(X_train_balanced, y_train_balanced)

    print("\n[5] Generating regularization effect plot ...")
    plot_regularization_effect(X_train_balanced, y_train_balanced, X_test_processed, y_test)

    print("\n[6] Generating neural network loss curve ...")
    plot_mlp_loss_curve(X_train_balanced, y_train_balanced)

    print("\n[7] Running unsupervised learning ...")
    cluster_results, pca_variance = run_unsupervised(X_train_balanced, X_test_processed, y_test)
    print(f"    {cluster_results.to_string(index=False)}")

    write_results_summary(
        raw_df,
        X_train_raw,
        X_test_raw,
        X_train_processed,
        sampler_name,
        results_df,
        cluster_results,
        pca_variance,
    )

    print("\n" + "=" * 60)
    print("FAZA II COMPLETE — All outputs saved to Faza II/output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
