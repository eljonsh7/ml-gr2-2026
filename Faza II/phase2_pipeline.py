"""
Faza II: Trajnimi i Modelit — Comprehensive ML Pipeline
========================================================
Covers syllabus topics: Supervised (LR, RF, GB, SVM, MLP/Neural Network),
Unsupervised (K-Means, Agglomerative + PCA), Learning Theory (learning curves,
regularization, cost convergence, hyperparameter tuning, confusion matrices).
"""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from textwrap import dedent

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FAZA1_OUTPUT = ROOT.parent / "Faza I" / "output"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

PLOT_DPI = 180
sns.set_theme(style="whitegrid", palette="deep")


# ══════════════════════════════════════════════
# 0. DATA LOADING
# ══════════════════════════════════════════════
def load_data():
    X_train = pd.read_csv(FAZA1_OUTPUT / "train_balanced_features.csv")
    y_train = pd.read_csv(FAZA1_OUTPUT / "train_balanced_target.csv")["target_quantile_class"]
    X_test = pd.read_csv(FAZA1_OUTPUT / "test_features.csv")
    y_test = pd.read_csv(FAZA1_OUTPUT / "test_target.csv")["target_quantile_class"]
    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════
# 1. SUPERVISED LEARNING — TRAINING + TUNING
# ══════════════════════════════════════════════
def define_models_and_params():
    """Define each supervised model with its hyperparameter grid for GridSearchCV."""
    models = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs"),
            "params": {"C": [0.01, 0.1, 1, 10]},
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        },
        "SVM (Linear)": {
            "estimator": SVC(kernel="linear", random_state=42),
            "params": {"C": [0.1, 1, 10]},
        },
        "SVM (RBF)": {
            "estimator": SVC(kernel="rbf", random_state=42),
            "params": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        },
        "Neural Network (MLP)": {
            "estimator": MLPClassifier(max_iter=500, random_state=42),
            "params": {"hidden_layer_sizes": [(64, 32), (128, 64)], "alpha": [0.0001, 0.001]},
        },
    }
    return models


def train_all_supervised(X_train, y_train, X_test, y_test):
    """Train every supervised model with GridSearchCV and return results + fitted models."""
    models = define_models_and_params()
    results = []
    fitted_models = {}
    reports_text = []

    for name, spec in models.items():
        print(f"  Training {name} ...")
        grid = GridSearchCV(
            spec["estimator"],
            spec["params"],
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        fitted_models[name] = best

        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Best Params": json.dumps(grid.best_params_),
            "CV F1 (macro)": round(grid.best_score_, 4),
            "Accuracy": round(acc, 4),
            "Precision (macro)": round(prec, 4),
            "Recall (macro)": round(rec, 4),
            "F1 (macro)": round(f1, 4),
            "F1 (weighted)": round(f1_w, 4),
        })

        # ── Confusion matrix plot ──
        cm = confusion_matrix(y_test, y_pred, labels=best.classes_)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=best.classes_, yticklabels=best.classes_, ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(OUTPUT / f"confusion_matrix_{safe_name}.png", dpi=PLOT_DPI)
        plt.close(fig)

        # ── Classification report text ──
        report = classification_report(y_test, y_pred, target_names=[str(c) for c in best.classes_])
        reports_text.append(f"{'='*60}\n{name}\nBest params: {grid.best_params_}\n{'='*60}\n{report}\n")

    # Save classification reports
    (OUTPUT / "classification_reports.txt").write_text("\n".join(reports_text), encoding="utf-8")

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(OUTPUT / "model_results.csv")
    return results_df, fitted_models


# ══════════════════════════════════════════════
# 2. ALGORITHM COMPARISON BAR CHART
# ══════════════════════════════════════════════
def plot_algorithm_comparison(results_df):
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


# ══════════════════════════════════════════════
# 3. FEATURE IMPORTANCE (Random Forest)
# ══════════════════════════════════════════════
def plot_feature_importance(rf_model, X_train):
    importances = rf_model.feature_importances_
    feature_names = [c.replace("numeric__", "").replace("categorical__", "") for c in X_train.columns]
    indices = np.argsort(importances)[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(indices)), importances[indices], color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT / "feature_importance_rf.png", dpi=PLOT_DPI)
    plt.close(fig)


# ══════════════════════════════════════════════
# 4. LEARNING CURVES (Overfitting / Underfitting)
# ══════════════════════════════════════════════
def plot_learning_curves(X_train, y_train):
    """Show how train vs validation score changes with training set size — demonstrates overfitting/underfitting."""
    estimator = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train,
        cv=3, scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1, random_state=42,
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


# ══════════════════════════════════════════════
# 5. REGULARIZATION EFFECT (Logistic Regression)
# ══════════════════════════════════════════════
def plot_regularization_effect(X_train, y_train, X_test, y_test):
    """Show how the regularization parameter C affects model performance — demonstrates regularization concept."""
    C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
    train_scores = []
    test_scores = []
    for c in C_values:
        lr = LogisticRegression(C=c, max_iter=2000, random_state=42, solver="lbfgs")
        lr.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, lr.predict(X_train), average="macro", zero_division=0))
        test_scores.append(f1_score(y_test, lr.predict(X_test), average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(C_values, train_scores, "o-", color="blue", label="Train F1")
    ax.plot(C_values, test_scores, "s-", color="red", label="Test F1")
    ax.set_xscale("log")
    ax.set_xlabel("Regularization Parameter C (log scale)")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Regularization Effect — Logistic Regression", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate overfitting/underfitting zones
    ax.axvspan(C_values[0], 0.05, alpha=0.08, color="blue", label="Underfitting zone")
    ax.axvspan(20, C_values[-1], alpha=0.08, color="red", label="Overfitting risk")
    ax.text(0.005, min(train_scores) * 0.98, "← Strong regularization\n    (Underfitting)", fontsize=8, color="blue")
    ax.text(30, max(test_scores) * 0.98, "Weak regularization →\n    (Overfitting risk)", fontsize=8, color="red")

    plt.tight_layout()
    fig.savefig(OUTPUT / "regularization_effect.png", dpi=PLOT_DPI)
    plt.close(fig)


# ══════════════════════════════════════════════
# 6. NEURAL NETWORK LOSS CURVE (Gradient Descent)
# ══════════════════════════════════════════════
def plot_mlp_loss_curve(X_train, y_train):
    """Train a dedicated MLP with early_stopping to extract loss curve — demonstrates cost function and gradient descent."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.15, alpha=0.001,
    )
    mlp.fit(X_train, y_encoded)

    if not hasattr(mlp, "loss_curve_") or not mlp.loss_curve_:
        print("  (MLP loss curve not available)")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(mlp.loss_curve_, color="purple", linewidth=2)
    ax.set_title("Neural Network Training Loss — Gradient Descent Convergence", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch (Iteration)")
    ax.set_ylabel("Loss (Cost Function)")
    ax.grid(True, alpha=0.3)

    ax.annotate("Convergence", xy=(len(mlp.loss_curve_) - 1, mlp.loss_curve_[-1]),
                xytext=(-80, 30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=10, color="purple")

    plt.tight_layout()
    fig.savefig(OUTPUT / "mlp_loss_curve.png", dpi=PLOT_DPI)
    plt.close(fig)

    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(mlp.validation_scores_, color="green", linewidth=2)
        ax.set_title("Neural Network Validation Score per Epoch", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUTPUT / "mlp_validation_curve.png", dpi=PLOT_DPI)
        plt.close(fig)


# ══════════════════════════════════════════════
# 7. UNSUPERVISED LEARNING — K-MEANS + AGGLO
# ══════════════════════════════════════════════
def run_unsupervised(X_train, y_train, X_test, y_test):
    """Run K-Means (with elbow + silhouette) and Agglomerative Clustering, then PCA visualization."""
    print("  Running Unsupervised Learning ...")

    # ── Elbow Method ──
    inertias = []
    sil_scores = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_train, labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(K_range), inertias, "o-", color="steelblue", linewidth=2)
    axes[0].set_title("Elbow Method — K-Means", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(K_range), sil_scores, "s-", color="coral", linewidth=2)
    axes[1].set_title("Silhouette Scores — K-Means", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT / "elbow_and_silhouette.png", dpi=PLOT_DPI)
    plt.close(fig)

    # ── K-Means with k=3 ──
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels_test = kmeans.fit_predict(X_test)
    km_sil = silhouette_score(X_test, km_labels_test)

    # ── Agglomerative with k=3 ──
    agglo = AgglomerativeClustering(n_clusters=3)
    ag_labels_test = agglo.fit_predict(X_test)
    ag_sil = silhouette_score(X_test, ag_labels_test)

    cluster_results = pd.DataFrame({
        "Algorithm": ["K-Means (k=3)", "Agglomerative (k=3)"],
        "Silhouette Score": [round(km_sil, 4), round(ag_sil, 4)],
    })
    cluster_results.to_csv(OUTPUT / "clustering_results.csv", index=False)

    # ── PCA Dimensionality Reduction + Visualization ──
    pca = PCA(n_components=2, random_state=42)
    X_test_pca = pca.fit_transform(X_test)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # PCA colored by K-Means clusters
    scatter1 = axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=km_labels_test,
                               cmap="viridis", alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
    axes[0].set_title("PCA — K-Means Clusters", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    # PCA colored by Agglomerative clusters
    scatter2 = axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=ag_labels_test,
                               cmap="magma", alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
    axes[1].set_title("PCA — Agglomerative Clusters", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.colorbar(scatter2, ax=axes[1], label="Cluster")

    # PCA colored by TRUE labels
    label_map = {"low": 0, "medium": 1, "high": 2}
    y_numeric = y_test.map(label_map).values
    scatter3 = axes[2].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_numeric,
                               cmap="coolwarm", alpha=0.6, s=20, edgecolors="k", linewidths=0.3)
    axes[2].set_title("PCA — True Labels (low/medium/high)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    axes[2].set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    cbar = plt.colorbar(scatter3, ax=axes[2], ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["low", "medium", "high"])

    plt.tight_layout()
    fig.savefig(OUTPUT / "pca_clusters_comparison.png", dpi=PLOT_DPI)
    plt.close(fig)

    return cluster_results, explained


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("FAZA II — TRAJNIMI I MODELIT")
    print("=" * 60)

    # 0. Load data
    print("\n[0] Loading data from Faza I ...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"    Train: {X_train.shape[0]} rows x {X_train.shape[1]} features")
    print(f"    Test:  {X_test.shape[0]} rows x {X_test.shape[1]} features")
    print(f"    Classes: {sorted(y_train.unique())}")

    # 1. Supervised models
    print("\n[1] SUPERVISED LEARNING — Training 6 algorithms with GridSearchCV ...")
    results_df, fitted = train_all_supervised(X_train, y_train, X_test, y_test)
    print("\n    Results:")
    print(results_df[["Accuracy", "F1 (macro)", "F1 (weighted)"]].to_string())

    # 2. Comparison chart
    print("\n[2] Generating algorithm comparison chart ...")
    plot_algorithm_comparison(results_df)

    # 3. Feature importance
    print("\n[3] Generating feature importance plot (Random Forest) ...")
    plot_feature_importance(fitted["Random Forest"], X_train)

    # 4. Learning curves
    print("\n[4] Generating learning curves (Overfitting/Underfitting analysis) ...")
    plot_learning_curves(X_train, y_train)

    # 5. Regularization
    print("\n[5] Generating regularization effect plot (Logistic Regression) ...")
    plot_regularization_effect(X_train, y_train, X_test, y_test)

    # 6. MLP loss curve
    print("\n[6] Generating neural network loss curve (Gradient Descent) ...")
    plot_mlp_loss_curve(X_train, y_train)

    # 7. Unsupervised
    print("\n[7] UNSUPERVISED LEARNING — K-Means, Agglomerative, PCA ...")
    cluster_results, pca_variance = run_unsupervised(X_train, y_train, X_test, y_test)
    print(f"    {cluster_results.to_string(index=False)}")

    # Done
    print("\n" + "=" * 60)
    print("FAZA II COMPLETE — All outputs saved to Faza II/output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
