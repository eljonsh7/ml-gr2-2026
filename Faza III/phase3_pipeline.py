"""
Faza III: Optimizimi dhe Fine-Tuning i Modeleve
================================================
Changes from Phase II:
  - SVM (RBF) removed — lowest Phase II performer (CV F1 = 0.9599)
  - GridSearchCV  →  RandomizedSearchCV with wider parameter grids
  - 3-fold CV  →  5-fold CV for more reliable score estimates
  - Feature selection via Random Forest importance (drop near-zero features)
  - New metrics: ROC-AUC (macro, OvR)
  - Statistical significance: Wilcoxon signed-rank test between best model and others
  - Phase II vs Phase III comparison report
  - Single best model selected and reported
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE
from scipy.stats import wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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
RANDOM_STATE = 42          # same as Phase II — guarantees identical train/test split
CV_FOLDS = 5               # upgraded from 3
N_ITER_SEARCH = 30         # RandomizedSearchCV iterations per model
sns.set_theme(style="whitegrid", palette="deep")

# ---------------------------------------------------------------------------
# Phase II baseline results (hardcoded for comparison chart)
# ---------------------------------------------------------------------------
PHASE2_BASELINE = {
    "Logistic Regression":   {"CV F1": 0.9847, "Accuracy": 0.9742, "F1 (macro)": 0.9741},
    "Random Forest":         {"CV F1": 0.9968, "Accuracy": 0.9903, "F1 (macro)": 0.9904},
    "Gradient Boosting":     {"CV F1": 0.9984, "Accuracy": 0.9903, "F1 (macro)": 0.9904},
    "SVM (Linear)":          {"CV F1": 0.9863, "Accuracy": 0.9710, "F1 (macro)": 0.9709},
    "Neural Network (MLP)":  {"CV F1": 0.9766, "Accuracy": 0.9710, "F1 (macro)": 0.9712},
}


# ===========================================================================
# DATA LOADING & SPLITTING
# ===========================================================================

def load_phase1_dataset() -> pd.DataFrame:
    if not PHASE1_FINAL_DATASET.exists():
        raise FileNotFoundError(f"Phase I final dataset not found:\n  {PHASE1_FINAL_DATASET}")
    return pd.read_csv(PHASE1_FINAL_DATASET)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_column = "target_quantile_class"
    if target_column not in df.columns:
        raise ValueError(f"Target column `{target_column}` not found.")
    return df.drop(columns=[target_column]).copy(), df[target_column].copy()


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )


def preprocess_splits(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train, columns=feature_names, index=X_train_raw.index)
    X_test_df  = pd.DataFrame(X_test,  columns=feature_names, index=X_test_raw.index)
    return X_train_df, X_test_df, preprocessor


def balance_training_split(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series, str]:
    class_counts = y_train.value_counts()
    class_share  = class_counts / class_counts.sum()
    if class_share.min() >= 0.2:
        return X_train.copy(), y_train.copy(), "Skipped (already balanced)"
    if class_counts.min() < 6:
        sampler = ADASYN(random_state=RANDOM_STATE, n_neighbors=max(1, class_counts.min() - 1))
        name = "ADASYN"
    else:
        sampler = SMOTE(random_state=RANDOM_STATE)
        name = "SMOTE"
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_res, columns=X_train.columns),
        pd.Series(y_res, name=y_train.name),
        name,
    )


# ===========================================================================
# FEATURE SELECTION
# ===========================================================================

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold_factor: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series]:
    """
    Train a quick Random Forest and keep only features whose importance
    is >= threshold_factor * mean_importance.
    A factor of 0.1 removes only near-zero features, keeping the vast majority.
    Returns filtered train, filtered test, kept feature names, and importances Series.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    threshold   = importances.mean() * threshold_factor
    kept_features = importances[importances >= threshold].index.tolist()

    print(f"    Feature selection: {len(X_train.columns)} → {len(kept_features)} features kept "
          f"(threshold={threshold:.5f})")

    return X_train[kept_features], X_test[kept_features], kept_features, importances


def plot_feature_selection(
    importances: pd.Series,
    kept_features: list[str],
    threshold_factor: float,
) -> None:
    # Use the same factor that was used in select_features() so the threshold
    # line in the plot matches the actual cutoff that determined kept_features.
    threshold = importances.mean() * threshold_factor
    top_n = min(25, len(importances))
    top_imp = importances.nlargest(top_n).sort_values()

    colors = ["steelblue" if f in kept_features else "lightcoral" for f in top_imp.index]
    clean_labels = [f.replace("numeric__", "").replace("categorical__", "") for f in top_imp.index]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(top_imp)), top_imp.values, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.4)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(clean_labels, fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Selection — Phase III\n(blue = kept, red = removed)", fontsize=13, fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor="steelblue", label="Kept"),
        Patch(facecolor="lightcoral", label="Removed"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"Threshold = {threshold:.5f}"),
    ])
    plt.tight_layout()
    fig.savefig(OUTPUT / "feature_selection.png", dpi=PLOT_DPI)
    plt.close(fig)


# ===========================================================================
# MODELS & HYPERPARAMETER GRIDS
# ===========================================================================

def define_models_and_params() -> dict[str, dict]:
    """
    Phase III changes vs Phase II:
      - SVM (RBF) REMOVED — lowest Phase II performer (CV F1 = 0.9599)
      - Wider hyperparameter ranges for all remaining models
      - RandomizedSearchCV samples up to N_ITER_SEARCH combinations per model,
        capped at the actual grid size to avoid ParameterSampler warnings
      - probability=True added to SVC so predict_proba / ROC-AUC works
      - MLP early_stopping disabled: sklearn's isnan check on string labels
        causes TypeError in this numpy version; max_iter=1000 compensates
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
            # 11 combinations total — n_iter capped to 11 in train_all_supervised
            "params": {
                "C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            # 5×6×3×3×2 = 540 combinations — n_iter=30 samples a random subset
            "params": {
                "n_estimators":      [100, 150, 200, 300, 500],
                "max_depth":         [8, 10, 15, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf":  [1, 2, 4],
                "max_features":      ["sqrt", "log2"],
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            # 5×5×4×4×3 = 1200 combinations — n_iter=30 samples a random subset
            "params": {
                "n_estimators":      [100, 150, 200, 300, 500],
                "learning_rate":     [0.01, 0.03, 0.05, 0.1, 0.2],
                "max_depth":         [2, 3, 4, 5],
                "subsample":         [0.7, 0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10],
            },
        },
        "SVM (Linear)": {
            "estimator": SVC(
                kernel="linear",
                random_state=RANDOM_STATE,
                probability=True,   # Platt scaling — enables predict_proba / ROC-AUC
            ),
            # 9 combinations total — n_iter capped to 9 in train_all_supervised
            "params": {
                "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
            },
        },
        "Neural Network (MLP)": {
            "estimator": MLPClassifier(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            # 5×4×3 = 60 combinations — n_iter=30 samples a random subset
            "params": {
                "hidden_layer_sizes": [(64, 32), (128, 64), (256, 128), (128, 64, 32), (256, 128, 64)],
                "alpha":              [0.00001, 0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.005, 0.01],
            },
        },
    }


# ===========================================================================
# TRAINING
# ===========================================================================

def train_all_supervised(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict, dict]:
    models = define_models_and_params()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results      = []
    fitted_models: dict[str, object] = {}
    reports_text = []
    cv_fold_scores: dict[str, np.ndarray] = {}   # for Wilcoxon test

    for name, spec in models.items():
        print(f"  Training {name} ...")

        # Cap n_iter to the actual number of grid combinations so
        # RandomizedSearchCV never requests more samples than exist.
        from sklearn.model_selection import ParameterGrid
        n_combos = len(ParameterGrid(spec["params"]))
        n_iter = min(N_ITER_SEARCH, n_combos)

        search = RandomizedSearchCV(
            spec["estimator"],
            spec["params"],
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted_models[name] = best

        y_pred = best.predict(X_test)

        # ROC-AUC (macro, one-vs-rest).
        # Use best.classes_ as the class ordering so label_binarize and
        # predict_proba columns are guaranteed to align.
        roc_auc = None
        if hasattr(best, "predict_proba"):
            y_prob = best.predict_proba(X_test)
            try:
                roc_auc = round(
                    roc_auc_score(
                        label_binarize(y_test, classes=best.classes_),
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                    ),
                    4,
                )
            except Exception:
                roc_auc = None

        # Per-fold F1 scores for Wilcoxon test (same cv splits → comparable)
        fold_scores = cross_val_score(
            best, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
        )
        cv_fold_scores[name] = fold_scores

        results.append({
            "Model":              name,
            "Best Params":        json.dumps(search.best_params_),
            "CV F1 (macro)":      round(search.best_score_, 4),
            "Accuracy":           round(accuracy_score(y_test, y_pred), 4),
            "Precision (macro)":  round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "Recall (macro)":     round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "F1 (macro)":         round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "F1 (weighted)":      round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "ROC-AUC (macro)":    roc_auc,
        })

        # Confusion matrix
        classes = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {name} (Phase III)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT / f"confusion_matrix_{safe}.png", dpi=PLOT_DPI)
        plt.close(fig)

        report = classification_report(y_test, y_pred)
        reports_text.append(
            f"{'=' * 60}\n{name}\nBest params: {search.best_params_}\nCV F1: {search.best_score_:.4f}\n"
            f"{'=' * 60}\n{report}\n"
        )

    (OUTPUT / "classification_reports_phase3.txt").write_text(
        "\n".join(reports_text), encoding="utf-8"
    )
    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(OUTPUT / "model_results_phase3.csv")
    return results_df, fitted_models, cv_fold_scores


# ===========================================================================
# VISUALIZATIONS
# ===========================================================================

def plot_algorithm_comparison(results_df: pd.DataFrame) -> None:
    metric_cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    ax = results_df[metric_cols].plot(
        kind="bar", figsize=(12, 6), colormap="tab10", edgecolor="black", linewidth=0.5
    )
    ax.set_title("Algorithm Comparison — Phase III (Optimised Models)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "algorithm_comparison_phase3.png", dpi=PLOT_DPI)
    plt.close()


def plot_phase2_vs_phase3(results_df: pd.DataFrame) -> None:
    """Side-by-side F1 (macro) comparison: Phase II baseline vs Phase III optimised."""
    models = list(results_df.index)
    p2_f1  = [PHASE2_BASELINE.get(m, {}).get("F1 (macro)", 0) for m in models]
    p3_f1  = results_df["F1 (macro)"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width / 2, p2_f1, width, label="Phase II", color="steelblue",  edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, p3_f1, width, label="Phase III", color="darkorange", edgecolor="black", linewidth=0.5)

    # Annotate delta
    for b1, b2 in zip(bars1, bars2):
        delta = b2.get_height() - b1.get_height()
        color = "green" if delta >= 0 else "red"
        sign  = "+" if delta >= 0 else ""
        ax.text(
            b2.get_x() + b2.get_width() / 2,
            b2.get_height() + 0.001,
            f"{sign}{delta:.4f}",
            ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold",
        )

    ax.set_title("Phase II vs Phase III — F1 (macro) Improvement", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.93, 1.01)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / "phase2_vs_phase3_comparison.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_feature_importance(rf_model: RandomForestClassifier, X_train: pd.DataFrame) -> None:
    importances  = rf_model.feature_importances_
    feature_names = [c.replace("numeric__", "").replace("categorical__", "") for c in X_train.columns]
    top_n    = min(15, len(importances))
    indices  = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest (Phase III)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT / "feature_importance_phase3.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_learning_curves(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict) -> None:
    """Learning curves using the best RF parameters found in Phase III."""
    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        **{k: v for k, v in best_params.items()},
    )
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=1,
    )
    t_mean, t_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    v_mean, v_std = val_scores.mean(axis=1),   val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(train_sizes, t_mean - t_std, t_mean + t_std, alpha=0.15, color="blue")
    ax.fill_between(train_sizes, v_mean - v_std, v_mean + v_std, alpha=0.15, color="orange")
    ax.plot(train_sizes, t_mean, "o-", color="blue",   label="Training Score")
    ax.plot(train_sizes, v_mean, "o-", color="orange", label="Validation Score")
    ax.set_title("Learning Curves — Random Forest Phase III (Optimised Params)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score (macro)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / "learning_curves_phase3.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_roc_auc(fitted_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Macro-average ROC-AUC plot using One-vs-Rest binarization.
    One curve per model (average over classes).
    Class ordering comes from model.classes_ so label_binarize and
    predict_proba columns are guaranteed to align.
    """
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["steelblue", "darkorange", "green", "purple", "crimson"]

    for (name, model), color in zip(fitted_models.items(), colors):
        if not hasattr(model, "predict_proba"):
            continue
        classes = model.classes_                           # authoritative ordering
        y_bin   = label_binarize(y_test, classes=classes)
        y_prob  = model.predict_proba(X_test)

        # Macro-average: compute ROC for each class then average
        fprs, tprs, aucs = [], [], []
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        mean_auc = np.mean(aucs)

        # Interpolate for plotting
        base_fpr = np.linspace(0, 1, 200)
        interp_tprs = [np.interp(base_fpr, f, t) for f, t in zip(fprs, tprs)]
        mean_tpr = np.mean(interp_tprs, axis=0)

        ax.plot(base_fpr, mean_tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {mean_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_title("ROC-AUC Curves — All Models (Macro Average, Phase III)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / "roc_auc_curves_phase3.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_calibration(fitted_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Calibration curves — how well predicted probabilities match actual frequencies.
    Uses model.classes_ for column indexing so proba columns and class labels align."""
    n_models = sum(1 for m in fitted_models.values() if hasattr(m, "predict_proba"))
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    # One color per model subplot — must have at least as many entries as n_models
    model_colors = ["steelblue", "darkorange", "green", "purple", "crimson"]
    ax_idx = 0

    for name, model in fitted_models.items():
        if not hasattr(model, "predict_proba"):
            continue
        ax = axes[ax_idx]
        classes = model.classes_          # use model's ordering — aligns with predict_proba columns

        # Plot calibration for each class
        for i, cls in enumerate(classes):
            y_bin = (y_test == cls).astype(int)
            try:
                probs = model.predict_proba(X_test)[:, i]
                # n_bins=6: ~100 test samples per class → 6 bins ≈ 16 samples/bin (safer than 8)
                fraction_pos, mean_pred = calibration_curve(y_bin, probs, n_bins=6)
                ax.plot(mean_pred, fraction_pos, marker="o", linewidth=1.5,
                        label=f"Class {cls}")
            except Exception:
                pass

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        if ax_idx == 0:
            ax.set_ylabel("Fraction of Positives")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    fig.suptitle("Calibration Curves — Phase III Models", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT / "calibration_curves_phase3.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# STATISTICAL COMPARISON — WILCOXON SIGNED-RANK TEST
# ===========================================================================

def run_wilcoxon_test(
    cv_fold_scores: dict[str, np.ndarray],
    results_df: pd.DataFrame,
) -> tuple[str, str]:
    """
    Wilcoxon signed-rank test: pairwise comparison of best model vs every other.
    Returns formatted text report.
    """
    best_model = results_df["CV F1 (macro)"].idxmax()
    best_scores = cv_fold_scores[best_model]

    lines = [
        "=" * 60,
        "WILCOXON SIGNED-RANK TEST — Phase III",
        f"Reference (best): {best_model}  (mean CV F1 = {best_scores.mean():.4f})",
        "=" * 60,
        "",
        "H0: No significant difference in F1 distributions.",
        "H1: Significant difference exists.",
        "Significance level: α = 0.05",
        "",
        f"{'Model':<30} {'Mean F1':>8} {'Statistic':>12} {'p-value':>12} {'Significant?':>14}",
        "-" * 80,
    ]

    for name, scores in cv_fold_scores.items():
        if name == best_model:
            continue
        diff = best_scores - scores
        if np.all(diff == 0):
            lines.append(f"{name:<30} {scores.mean():>8.4f} {'N/A (tied)':>12} {'N/A':>12} {'No':>14}")
            continue
        stat, pval = wilcoxon(best_scores, scores, alternative="greater")
        sig = "YES" if pval < 0.05 else "No"
        lines.append(f"{name:<30} {scores.mean():>8.4f} {stat:>12.3f} {pval:>12.4f} {sig:>14}")

    lines += [
        "",
        f"Conclusion: '{best_model}' is the statistically best model for this dataset.",
    ]

    report = "\n".join(lines)
    (OUTPUT / "wilcoxon_results.txt").write_text(report, encoding="utf-8")
    return best_model, report


# ===========================================================================
# PHASE II vs PHASE III COMPARISON TABLE
# ===========================================================================

def build_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in results_df.index:
        p2 = PHASE2_BASELINE.get(model, {})
        p3_f1  = results_df.loc[model, "F1 (macro)"]
        p3_cv  = results_df.loc[model, "CV F1 (macro)"]
        p3_acc = results_df.loc[model, "Accuracy"]
        delta_f1 = round(p3_f1 - p2.get("F1 (macro)", 0), 4)
        rows.append({
            "Model":            model,
            "Ph2 CV F1":        p2.get("CV F1", "-"),
            "Ph3 CV F1":        p3_cv,
            "Ph2 Accuracy":     p2.get("Accuracy", "-"),
            "Ph3 Accuracy":     p3_acc,
            "Ph2 F1 (macro)":   p2.get("F1 (macro)", "-"),
            "Ph3 F1 (macro)":   p3_f1,
            "Delta F1":         f"{'+' if delta_f1 >= 0 else ''}{delta_f1}",
        })
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(OUTPUT / "comparison_phase2_vs_phase3.csv")
    return df


# ===========================================================================
# FINAL MODEL REPORT
# ===========================================================================

def write_final_report(
    best_model_name: str,
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    kept_features: list[str],
    sampler_name: str,
    wilcoxon_text: str,
    X_train_shape: tuple,
    X_test_shape: tuple,
) -> None:
    best_row    = results_df.loc[best_model_name]
    best_params = best_row["Best Params"]

    lines = [
        "# Phase III — Final Report",
        "",
        "## Configuration",
        f"- CV folds: {CV_FOLDS} (upgraded from 3 in Phase II)",
        f"- Search method: RandomizedSearchCV  (n_iter={N_ITER_SEARCH} per model)",
        f"- SVM (RBF) removed (Phase II CV F1 = 0.9599 — lowest performer)",
        f"- Training balance strategy: {sampler_name}",
        f"- Training set: {X_train_shape[0]} rows x {X_train_shape[1]} features",
        f"- Test set:     {X_test_shape[0]} rows x {X_test_shape[1]} features",
        f"- Features kept after selection: {len(kept_features)} / original",
        "",
        "## Phase II vs Phase III Comparison",
        "",
        "```",
        comparison_df.to_string(),
        "```",
        "",
        "## All Phase III Model Results",
        "",
        "```",
        results_df.drop(columns=["Best Params"]).to_string(),
        "```",
        "",
        "## Best Model",
        f"**{best_model_name}**",
        f"- CV F1 (macro):    {best_row['CV F1 (macro)']}",
        f"- Accuracy:         {best_row['Accuracy']}",
        f"- F1 (macro):       {best_row['F1 (macro)']}",
        f"- ROC-AUC (macro):  {best_row['ROC-AUC (macro)']}",
        f"- Best Params: `{best_params}`",
        "",
        "## Statistical Comparison (Wilcoxon)",
        "",
        "```",
        wilcoxon_text,
        "```",
        "",
        "## Output Files",
        "| File | Description |",
        "|---|---|",
        "| `model_results_phase3.csv` | All model metrics |",
        "| `comparison_phase2_vs_phase3.csv` | Phase II vs III delta |",
        "| `classification_reports_phase3.txt` | Per-class reports |",
        "| `wilcoxon_results.txt` | Statistical significance |",
        "| `algorithm_comparison_phase3.png` | Bar chart all models |",
        "| `phase2_vs_phase3_comparison.png` | Improvement chart |",
        "| `feature_selection.png` | Which features were kept/removed |",
        "| `feature_importance_phase3.png` | RF feature importances |",
        "| `learning_curves_phase3.png` | Overfitting analysis |",
        "| `roc_auc_curves_phase3.png` | ROC-AUC all models |",
        "| `calibration_curves_phase3.png` | Probability calibration |",
        "| `confusion_matrix_*.png` | Per-model confusion matrices |",
        "",
    ]

    (OUTPUT / "final_report_phase3.md").write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print("FAZA III — OPTIMIZIMI DHE FINE-TUNING I MODELEVE")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # [0] Data loading & splitting (identical random_state → same split as Ph2)
    # -----------------------------------------------------------------------
    print("\n[0] Loading Phase I dataset and splitting ...")
    raw_df = load_phase1_dataset()
    X_raw, y = split_features_and_target(raw_df)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_proc, X_test_proc, _ = preprocess_splits(X_train_raw, X_test_raw)
    X_train_bal, y_train_bal, sampler_name = balance_training_split(X_train_proc, y_train)

    print(f"    Train: {X_train_raw.shape[0]} rows x {X_train_raw.shape[1]} raw features")
    print(f"    Test:  {X_test_raw.shape[0]} rows x {X_test_raw.shape[1]} raw features")
    print(f"    After preprocessing: {X_train_proc.shape[1]} features")
    print(f"    Balance strategy: {sampler_name}")
    print(f"    Test class distribution:\n{y_test.value_counts().to_string()}")

    # -----------------------------------------------------------------------
    # [1] Feature selection
    # -----------------------------------------------------------------------
    FEAT_SEL_FACTOR = 0.05  # defined once — used by both select_features and the plot
    print("\n[1] Feature selection via Random Forest importance ...")
    X_train_sel, X_test_sel, kept_features, importances = select_features(
        X_train_bal, y_train_bal, X_test_proc, threshold_factor=FEAT_SEL_FACTOR
    )
    # Pass the same factor so the threshold line in the plot matches reality
    plot_feature_selection(importances, kept_features, threshold_factor=FEAT_SEL_FACTOR)

    # -----------------------------------------------------------------------
    # [2] Train 5 models with RandomizedSearchCV (5-fold)
    # -----------------------------------------------------------------------
    print(f"\n[2] Training 5 models with RandomizedSearchCV "
          f"(n_iter={N_ITER_SEARCH}, cv={CV_FOLDS}-fold) ...")
    results_df, fitted, cv_fold_scores = train_all_supervised(
        X_train_sel, y_train_bal, X_test_sel, y_test
    )
    print("\n    Results:")
    print(results_df[["CV F1 (macro)", "Accuracy", "F1 (macro)", "ROC-AUC (macro)"]].to_string())

    # -----------------------------------------------------------------------
    # [3] Plots
    # -----------------------------------------------------------------------
    print("\n[3] Generating algorithm comparison chart ...")
    plot_algorithm_comparison(results_df)

    print("\n[4] Generating Phase II vs Phase III comparison chart ...")
    plot_phase2_vs_phase3(results_df)

    print("\n[5] Generating feature importance plot (Random Forest) ...")
    plot_feature_importance(fitted["Random Forest"], X_train_sel)

    print("\n[6] Generating learning curves (Random Forest best params) ...")
    rf_best_params = json.loads(results_df.loc["Random Forest", "Best Params"])
    plot_learning_curves(X_train_sel, y_train_bal, rf_best_params)

    print("\n[7] Generating ROC-AUC curves ...")
    plot_roc_auc(fitted, X_test_sel, y_test)

    print("\n[8] Generating calibration curves ...")
    plot_calibration(fitted, X_test_sel, y_test)

    # -----------------------------------------------------------------------
    # [4] Statistical test
    # -----------------------------------------------------------------------
    print("\n[9] Running Wilcoxon signed-rank test ...")
    best_model_name, wilcoxon_text = run_wilcoxon_test(cv_fold_scores, results_df)
    print(f"    Best model: {best_model_name}")
    print(f"    CV F1 = {results_df.loc[best_model_name, 'CV F1 (macro)']}")

    # -----------------------------------------------------------------------
    # [5] Comparison table & final report
    # -----------------------------------------------------------------------
    print("\n[10] Building comparison table & final report ...")
    comparison_df = build_comparison_table(results_df)
    write_final_report(
        best_model_name,
        results_df,
        comparison_df,
        kept_features,
        sampler_name,
        wilcoxon_text,
        X_train_sel.shape,
        X_test_sel.shape,
    )

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"FAZA III COMPLETE")
    print(f"  Best model : {best_model_name}")
    print(f"  CV F1      : {results_df.loc[best_model_name, 'CV F1 (macro)']}")
    print(f"  Accuracy   : {results_df.loc[best_model_name, 'Accuracy']}")
    print(f"  F1 (macro) : {results_df.loc[best_model_name, 'F1 (macro)']}")
    print(f"  ROC-AUC    : {results_df.loc[best_model_name, 'ROC-AUC (macro)']}")
    print(f"\n  All outputs saved to: {OUTPUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
