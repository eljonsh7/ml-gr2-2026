# Faza III — Optimizimi dhe Fine-Tuning i Modeleve

## Table of Contents
1. [Overview](#1-overview)
2. [What Changed from Phase II](#2-what-changed-from-phase-ii)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Step 0 — Data Loading & Splitting](#4-step-0--data-loading--splitting)
5. [Step 1 — Preprocessing](#5-step-1--preprocessing)
6. [Step 2 — Class Balance Check](#6-step-2--class-balance-check)
7. [Step 3 — Feature Selection](#7-step-3--feature-selection)
8. [Step 4 — Hyperparameter Search Strategy](#8-step-4--hyperparameter-search-strategy)
9. [Algorithms — Full Mathematical Treatment](#9-algorithms--full-mathematical-treatment)
   - [9.1 Logistic Regression](#91-logistic-regression)
   - [9.2 Random Forest](#92-random-forest)
   - [9.3 Gradient Boosting](#93-gradient-boosting)
   - [9.4 SVM Linear](#94-support-vector-machine--linear-kernel)
   - [9.5 Neural Network (MLP)](#95-neural-network--multi-layer-perceptron)
10. [Evaluation Metrics — Full Formulas](#10-evaluation-metrics--full-formulas)
11. [Statistical Significance — Wilcoxon Test](#11-statistical-significance--wilcoxon-signed-rank-test)
12. [Results](#12-results)
13. [Phase II vs Phase III Comparison](#13-phase-ii-vs-phase-iii-comparison)
14. [Output Files](#14-output-files)

---

## 1. Overview

**Goal:** Take the five best-performing supervised algorithms from Phase II, widen their hyperparameter search space, and identify a single statistically superior model for predicting the daily carbon intensity class of Kosovo's power grid.

**Target variable:** `target_quantile_class` — three classes: `High`, `Medium`, `Low`

**Input:** `feature_engineered_dataset.csv` from Phase I (1,550 rows × 20 columns)

**Decision removed from Phase II:** SVM (RBF) — Phase II CV F1 = 0.9599, lowest of all six models. Its additional flexibility brought no gain over the linear kernel, confirming that the dominant structure in this dataset is not well-captured by a radial basis function.

---

## 2. What Changed from Phase II

| Aspect | Phase II | Phase III |
|---|---|---|
| Models | 6 (incl. SVM RBF) | 5 (SVM RBF removed) |
| Search method | `GridSearchCV` | `RandomizedSearchCV` |
| CV folds | 3 | 5 |
| Parameter ranges | Narrow | Wide (3–5× more values) |
| Feature selection | None | RF-importance threshold |
| Metrics | Accuracy, Precision, Recall, F1 | + ROC-AUC (macro, OvR) |
| Statistical test | None | Wilcoxon signed-rank |
| Final report | Both RF & GB tied | Single winner declared |

---

## 3. Pipeline Architecture

```
Phase I dataset (1,550 × 20)
        │
        ▼
  train_test_split (stratified, 80/20)
        │
        ├──── X_train_raw (1,240 × 19) ──── StandardScaler + OneHotEncoder ──▶ X_train_proc (1,240 × 25)
        │                                                                                │
        │                                                                    balance_training_split
        │                                                                       (already balanced)
        │                                                                                │
        │                                                                    Feature Selection (RF)
        │                                                                       25 → 9 features
        │                                                                                │
        │                                                            ┌───────────────────┤
        │                                                            │  RandomizedSearchCV│
        │                                                            │  (5-fold Stratified│
        │                                                            │   KFold, F1 macro) │
        │                                                            ├────────────────────┤
        │                                                            │ Logistic Regression│
        │                                                            │ Random Forest      │
        │                                                            │ Gradient Boosting  │
        │                                                            │ SVM (Linear)       │
        │                                                            │ MLP                │
        │                                                            └─────────┬──────────┘
        │                                                                      │
        └──── X_test_raw (310 × 19) ──────── transform ──────── X_test_sel ───▶ Evaluate
                                                                                │
                                                         ┌──────────────────────┤
                                                         │  Metrics             │
                                                         │  Confusion Matrices  │
                                                         │  ROC-AUC             │
                                                         │  Calibration Curves  │
                                                         │  Learning Curves     │
                                                         │  Wilcoxon Test       │
                                                         └──────────────────────┘
```

---

## 4. Step 0 — Data Loading & Splitting

### Stratified Train/Test Split

The dataset is split 80% train / 20% test using `stratify=y`:

```
RANDOM_STATE = 42   (guarantees reproducibility and identical split to Phase II)
test_size    = 0.2

Train: 1,240 rows
Test:    310 rows
```

**Why stratification?**
Without stratification, a random split might by chance put more `High` samples in the test set and fewer in training. Stratification enforces that the class proportions in both sets match the original dataset. Formally, if class $k$ has proportion $p_k$ in the full dataset, stratification guarantees:

$$\frac{n_k^{\text{train}}}{n^{\text{train}}} \approx \frac{n_k^{\text{test}}}{n^{\text{test}}} \approx p_k$$

**Test set class distribution (actual):**
| Class | Count | Share |
|---|---|---|
| Medium | 106 | 34.2% |
| High | 105 | 33.9% |
| Low | 99 | 31.9% |

---

## 5. Step 1 — Preprocessing

Preprocessing is fit **only on the training set** and then applied to the test set. Fitting on the full dataset before splitting is data leakage — the model would have indirect knowledge of test statistics.

### 5.1 StandardScaler (Numeric Columns)

Each numeric feature $x_j$ is transformed to zero mean and unit variance:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

Where:
- $\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$ — mean of feature $j$ over the training set
- $\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2}$ — standard deviation

**Why?** Algorithms like Logistic Regression, SVM, and MLP optimize a loss function using gradient-based methods. A feature with values in [0, 10,000] would produce gradients 1,000× larger than a feature with values in [0, 10], causing the optimizer to move disproportionately in that direction. Standardization removes this scale dependence.

Tree-based models (Random Forest, Gradient Boosting) are scale-invariant — they split on thresholds and only care about feature ordering, not magnitude. StandardScaler does not hurt them either.

### 5.2 OneHotEncoder (Categorical Columns)

Each categorical feature with $K$ unique categories is replaced by $K$ binary columns:

$$x_j \in \{c_1, c_2, \ldots, c_K\} \longrightarrow [0, 0, \ldots, 1, \ldots, 0] \in \{0,1\}^K$$

**Parameters used:**
- `handle_unknown="ignore"` — if the test set contains a category not seen during training, all its indicator columns are set to 0 rather than raising an error
- `sparse_output=False` — returns a dense NumPy array for compatibility with all downstream estimators

**Why not ordinal encoding?** Ordinal encoding (e.g., `Low=0, Medium=1, High=2`) implies an ordering that does not exist for arbitrary categorical variables. One-hot encoding makes no such assumption.

**Result after preprocessing:** 25 features (from 19 raw; extra columns from one-hot expanding categorical variables).

---

## 6. Step 2 — Class Balance Check

Evaluated **only on the training split** (checking the test set would be leakage — models are evaluated on the test set as-is).

**Threshold rule:**
$$\text{If } \min_k \left(\frac{n_k}{n}\right) \geq 0.20 \Rightarrow \text{balanced, skip resampling}$$

**Result in this run:** Training split was naturally balanced → `"Skipped (already balanced)"`

If imbalanced, the pipeline applies:

### SMOTE — Synthetic Minority Over-sampling Technique

For each minority sample $\mathbf{x}_i$, select one of its $k$ nearest neighbors $\mathbf{x}_{nn}$ uniformly at random and create:

$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim \text{Uniform}(0, 1)$$

This generates synthetic samples **along the line segment** between real minority class points.

**Condition:** Used when $\min(n_k) \geq 6$ (enough neighbors for stable interpolation).

### ADASYN — Adaptive Synthetic Sampling

ADASYN improves on SMOTE by generating **more synthetic samples near the decision boundary** — i.e., around minority class points that are surrounded by majority class points and therefore harder to classify correctly.

For each minority sample $\mathbf{x}_i$, compute:

$$r_i = \frac{\Delta_i}{k}, \quad \Delta_i = \text{number of majority-class samples among } k \text{ nearest neighbors of } \mathbf{x}_i$$

Then normalize: $\hat{r}_i = r_i / \sum_j r_j$

The number of synthetic samples generated for $\mathbf{x}_i$ is $G \cdot \hat{r}_i$, where $G$ is the total synthetic samples needed. Samples near the boundary (high $r_i$) get more synthetic neighbors.

**Condition:** Used when $\min(n_k) < 6$ (SMOTE needs at least 6 neighbors).

---

## 7. Step 3 — Feature Selection

A preliminary Random Forest (100 trees) is trained on the balanced training set to compute **Gini-based feature importances**.

### Feature Importance Formula (Mean Decrease in Impurity)

For a single decision tree, the importance of feature $j$ is:

$$I(j) = \sum_{t \in \text{nodes where } j \text{ is used}} \frac{n_t}{n} \cdot \Delta \text{Gini}(t)$$

Where:
- $n_t$ = number of training samples reaching node $t$
- $n$ = total training samples
- $\Delta \text{Gini}(t) = \text{Gini}(t) - \frac{n_L}{n_t}\text{Gini}(t_L) - \frac{n_R}{n_t}\text{Gini}(t_R)$ — impurity decrease at split $t$
- $\text{Gini}(t) = 1 - \sum_{k} p_k^2$ — Gini impurity at node $t$

For a Random Forest, the importance is averaged over all trees and normalized to sum to 1.

### Threshold

$$\text{threshold} = \bar{I} \times 0.05, \quad \bar{I} = \frac{1}{p} \sum_{j=1}^{p} I(j)$$

A factor of 0.05 removes only features whose importance is below 5% of the mean — this is a conservative cutoff that drops near-zero features while keeping the vast majority.

**Result:** 25 features → **9 features kept**

This reduces noise from irrelevant features, speeds up training, and can improve generalization. The same mask is applied identically to the test set (no re-fitting on test data).

---

## 8. Step 4 — Hyperparameter Search Strategy

### RandomizedSearchCV

Instead of exhaustively testing every combination (GridSearchCV), RandomizedSearchCV samples $n\_iter$ combinations uniformly at random from the parameter distributions:

$$\theta^* = \arg\max_{\theta \in S} \mathbb{E}[\text{CV Score}(\theta)], \quad S \subset \Theta, \; |S| = n\_iter$$

**Why RandomizedSearchCV over GridSearchCV?**
- GridSearchCV with wide grids would require thousands of model fits (e.g., Gradient Boosting has $5 \times 5 \times 4 \times 4 \times 3 = 1200$ combinations — at 5 folds each, that's 6,000 fits per model)
- RandomizedSearchCV samples 30 combinations (150 fits per model) and still explores the full range
- Empirically, random search finds equally good hyperparameters as grid search in a fraction of the time *(Bergstra & Bengio, 2012)*

**n_iter is capped per model:**
```
Logistic Regression : min(30, 11)  = 11   ← exhaustive (only 11 combinations)
Random Forest       : min(30, 540) = 30   ← random subset
Gradient Boosting   : min(30,1200) = 30   ← random subset
SVM (Linear)        : min(30,   9) =  9   ← exhaustive (only 9 combinations)
MLP                 : min(30,  60) = 30   ← random subset
```

### StratifiedKFold Cross-Validation (5 folds)

The training set is partitioned into 5 non-overlapping, stratified folds:

```
Fold 1: [████░░░░░░░░░░░░░░░░]  validate / train on remaining 4
Fold 2: [░░░░████░░░░░░░░░░░░]
Fold 3: [░░░░░░░░████░░░░░░░░]
Fold 4: [░░░░░░░░░░░░████░░░░]
Fold 5: [░░░░░░░░░░░░░░░░████]
```

CV score for a parameter configuration $\theta$:

$$\widehat{\text{CV}}(\theta) = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_{\text{macro}}(M_\theta^{(-k)}, D_k)$$

Where $M_\theta^{(-k)}$ is the model trained on all folds except $k$, evaluated on fold $k$.

**Why 5-fold and not 3-fold (Phase II)?**
More folds → more training data per fold → lower bias in the score estimate, and averaging over 5 held-out sets reduces variance. 5-fold is the standard in academic ML evaluation.

**Scoring metric:** `f1_macro` — macro-averaged F1 treats all classes equally regardless of support, which is appropriate for a 3-class problem where no single class should dominate the optimization.

---

## 9. Algorithms — Full Mathematical Treatment

### 9.1 Logistic Regression

**Type:** Probabilistic linear classifier (generalized linear model)

**Best hyperparameters found:** `C = 100`

#### Multinomial (Softmax) Formulation

For $K = 3$ classes, the probability of class $k$ given input $\mathbf{x} \in \mathbb{R}^p$ is:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}$$

This is the **softmax function** — it produces a proper probability distribution over $K$ classes (all values positive, sum to 1).

#### Loss Function — Multinomial Cross-Entropy + L2 Regularization

The model is trained by minimizing:

$$\mathcal{L}(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbf{1}[y_i = k] \log P(y_i = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^{K} \|\mathbf{w}_k\|_2^2$$

Where:
- First term: **cross-entropy loss** — penalizes low predicted probability for the correct class
- Second term: **L2 regularization** — shrinks weights toward zero to prevent overfitting
- $C$ = **inverse regularization strength**: large $C$ → weak regularization → more flexible; small $C$ → strong regularization → simpler model

#### Regularization Effect

$$C \to 0 \Rightarrow \mathbf{W} \to \mathbf{0} \quad \text{(underfitting)}$$
$$C \to \infty \Rightarrow \text{no penalty, pure MLE} \quad \text{(overfitting risk)}$$

**Best C = 100** (weak regularization) indicates the dataset has low noise and the model benefits from fitting closely to the training distribution.

#### Decision Rule

$$\hat{y} = \arg\max_{k} \; P(y = k \mid \mathbf{x})$$

#### Phase III Results

| Metric | Value |
|---|---|
| CV F1 (macro) | 0.9872 |
| Accuracy | 0.9806 |
| F1 (macro) | 0.9806 |
| ROC-AUC (macro) | 0.9993 |

**Per-class:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 1.00 | 0.99 | 1.00 | 105 |
| Low | 0.97 | 0.98 | 0.97 | 99 |
| Medium | 0.97 | 0.97 | 0.97 | 106 |

---

### 9.2 Random Forest

**Type:** Ensemble of decision trees using Bootstrap Aggregation (Bagging)

**Best hyperparameters found:** `n_estimators=100, max_depth=8, min_samples_split=2, min_samples_leaf=4, max_features="log2"`

#### Decision Tree — Splitting Criterion (Gini Impurity)

At each node $t$, the tree selects the feature $j^*$ and threshold $\tau^*$ that maximizes the impurity decrease:

$$j^*, \tau^* = \arg\max_{j, \tau} \; \Delta\text{Gini}(t, j, \tau)$$

$$\Delta\text{Gini}(t, j, \tau) = \text{Gini}(t) - \frac{n_L}{n_t}\text{Gini}(t_L) - \frac{n_R}{n_t}\text{Gini}(t_R)$$

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} \hat{p}_{tk}^2, \quad \hat{p}_{tk} = \frac{\text{samples of class } k \text{ at node } t}{n_t}$$

A pure node ($\hat{p}_{tk} = 1$ for one $k$) has $\text{Gini} = 0$ (minimum). A fully mixed node has $\text{Gini} = 1 - \frac{1}{K}$ (maximum).

#### Bagging (Bootstrap Aggregation)

For each tree $b = 1, \ldots, B$:
1. Draw a bootstrap sample $D_b$ of size $n$ with replacement from the training set
2. At each split, consider only a random subset of $m = \lfloor\log_2(p)\rfloor$ features (`max_features="log2"`)
3. Grow the tree to maximum depth or until leaf purity conditions are met

**Why random feature subsets?** If one feature is very predictive, all trees in a standard bagging ensemble would use it at the root, making the trees highly correlated. Random subsets decorrelate the trees, reducing ensemble variance without increasing bias.

#### Ensemble Prediction (Majority Vote)

$$\hat{y} = \text{mode}\left(\hat{y}_1(\mathbf{x}), \hat{y}_2(\mathbf{x}), \ldots, \hat{y}_B(\mathbf{x})\right)$$

#### Bias-Variance Decomposition

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

Individual deep trees have low bias but high variance (overfit to their bootstrap sample). Averaging $B$ decorrelated trees reduces variance by approximately $\frac{1}{B}$ while keeping bias constant.

#### Hyperparameter Meanings

| Param | Value | Effect |
|---|---|---|
| `n_estimators` | 100 | Number of trees — more trees → lower variance but diminishing returns |
| `max_depth` | 8 | Maximum tree depth — prevents individual trees from overfitting |
| `min_samples_leaf` | 4 | Leaf must have ≥4 samples — regularizes the tree, reduces noise |
| `min_samples_split` | 2 | Node must have ≥2 samples to be split |
| `max_features` | log2 | $m = \lfloor\log_2(9)\rfloor = 3$ features considered per split |

#### Phase III Results

| Metric | Value |
|---|---|
| CV F1 (macro) | 0.9984 |
| Accuracy | 0.9903 |
| F1 (macro) | 0.9904 |
| ROC-AUC (macro) | 0.9999 |

**Per-class:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 1.00 | 0.98 | 0.99 | 105 |
| Low | 0.99 | 1.00 | 0.99 | 99 |
| Medium | 0.98 | 0.99 | 0.99 | 106 |

---

### 9.3 Gradient Boosting

**Type:** Additive ensemble of shallow decision trees built sequentially using gradient descent in function space

**Best hyperparameters found:** `n_estimators=500, learning_rate=0.2, max_depth=5, subsample=0.9, min_samples_split=5`

#### Additive Model

The prediction is a sum of $M$ weak learners (trees):

$$F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^{M} \eta \cdot h_m(\mathbf{x})$$

Where:
- $F_0(\mathbf{x})$ = initial prediction (e.g., log-odds of most frequent class)
- $h_m(\mathbf{x})$ = the $m$-th tree, trained to predict the **negative gradient** (pseudo-residuals) of the loss with respect to the current model
- $\eta$ = `learning_rate` (shrinkage parameter)

#### Gradient Descent in Function Space

At each boosting round $m$, compute the pseudo-residuals:

$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}$$

For multinomial cross-entropy loss across $K$ classes:

$$\mathcal{L} = -\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbf{1}[y_i = k] \log p_{ik}$$

The pseudo-residual for class $k$ at sample $i$ is:

$$r_{imk} = \mathbf{1}[y_i = k] - p_{ik,m-1}$$

This is simply the difference between the true one-hot label and the current probability prediction — the model learns to reduce this residual at each step.

#### Shrinkage (learning_rate)

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x}), \quad \eta = 0.2$$

Small $\eta$ → many small steps (more trees needed, better regularization).
Large $\eta$ → fewer, larger steps (faster but may overshoot).

#### Stochastic Gradient Boosting (subsample)

With `subsample=0.9`, each tree $h_m$ is trained on a random 90% subsample of the training data (without replacement), drawn fresh each round. This introduces randomness that:
- Reduces correlation between consecutive trees
- Acts as implicit regularization
- Often improves generalization (Friedman, 1999)

#### Difference from Random Forest

| | Random Forest | Gradient Boosting |
|---|---|---|
| Tree construction | **Parallel** (independent) | **Sequential** (each corrects previous) |
| Target | Original labels | Pseudo-residuals (gradient of loss) |
| Randomness | Bootstrap + feature subsets | Subsampling (stochastic) |
| Regularization | Depth, leaf size | Learning rate, depth, subsampling |
| Typical trees | Deep | **Shallow** (max_depth=5) |

#### Hyperparameter Meanings

| Param | Value | Effect |
|---|---|---|
| `n_estimators` | 500 | Boosting rounds — more rounds fit residuals more precisely |
| `learning_rate` | 0.2 | Shrinkage — scales each tree's contribution |
| `max_depth` | 5 | Shallow trees are weak learners — depth 5 allows moderate interactions |
| `subsample` | 0.9 | 90% of training data per tree — stochastic regularization |
| `min_samples_split` | 5 | Node must have ≥5 samples to split |

#### Phase III Results (Best Model)

| Metric | Value |
|---|---|
| CV F1 (macro) | **0.9992** ← highest |
| Accuracy | 0.9903 |
| F1 (macro) | 0.9904 |
| ROC-AUC (macro) | **0.9999** |

**Per-class:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 1.00 | 0.98 | 0.99 | 105 |
| Low | 0.99 | 1.00 | 0.99 | 99 |
| Medium | 0.98 | 0.99 | 0.99 | 106 |

---

### 9.4 Support Vector Machine — Linear Kernel

**Type:** Maximum-margin linear classifier

**Best hyperparameters found:** `C = 50`

**Note:** `probability=True` enables Platt scaling so `predict_proba` is available for ROC-AUC computation.

#### Binary SVM Formulation (extended to multi-class via One-vs-Rest)

For a binary problem with labels $y \in \{-1, +1\}$, the SVM finds the hyperplane $\mathbf{w}^\top \mathbf{x} + b = 0$ that maximizes the margin:

$$\max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|} \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \; \forall i$$

Equivalently (primal form with slack variables):

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$
$$\text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Where:
- $\frac{1}{\|\mathbf{w}\|}$ = **margin width** — maximizing this minimizes $\|\mathbf{w}\|^2$
- $\xi_i \geq 0$ = **slack variables** — allow misclassifications (soft margin)
- $C$ = **regularization parameter**: large $C$ → penalize misclassifications heavily → small margin; small $C$ → allow more slack → larger margin

#### Hinge Loss Interpretation

The SVM objective is equivalent to:

$$\min_{\mathbf{w}} \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^{n} \max(0, 1 - y_i \mathbf{w}^\top \mathbf{x}_i)$$

The term $\max(0, 1 - y_i f(\mathbf{x}_i))$ is the **hinge loss** — zero for correctly classified points outside the margin, linear for points inside or beyond the margin.

#### Multi-class Extension (One-vs-Rest)

For $K = 3$ classes, three binary SVMs are trained:
- $f_1$: High vs. {Low, Medium}
- $f_2$: Low vs. {High, Medium}
- $f_3$: Medium vs. {High, Low}

$$\hat{y} = \arg\max_{k} \; f_k(\mathbf{x})$$

#### Platt Scaling (probability=True)

The decision function $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ is converted to a probability via sigmoid:

$$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{Af(\mathbf{x}) + B}}$$

Parameters $A$ and $B$ are fit by maximum likelihood on a held-out validation set (5-fold cross-validation internally). This enables ROC-AUC computation.

#### Hyperparameter Meanings

| Param | Value | Effect |
|---|---|---|
| `C` | 50 | High C → small margin, penalize misclassifications heavily |
| `kernel` | linear | Decision boundary is a hyperplane in the original feature space |

#### Phase III Results

| Metric | Value |
|---|---|
| CV F1 (macro) | 0.9904 |
| Accuracy | 0.9839 |
| F1 (macro) | 0.9838 |
| ROC-AUC (macro) | 0.9995 |

**Per-class:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 1.00 | 0.99 | 1.00 | 105 |
| Low | 0.97 | 0.99 | 0.98 | 99 |
| Medium | 0.98 | 0.97 | 0.98 | 106 |

---

### 9.5 Neural Network — Multi-Layer Perceptron

**Type:** Fully-connected feedforward neural network

**Best hyperparameters found:** `hidden_layer_sizes=(64, 32), alpha=0.001, learning_rate_init=0.005`

**Architecture:**

```
Input Layer:    9 neurons  (one per selected feature)
Hidden Layer 1: 64 neurons + ReLU activation
Hidden Layer 2: 32 neurons + ReLU activation
Output Layer:   3 neurons  + Softmax (one per class)
```

#### Forward Pass

For a network with $L$ layers, the output of layer $l$ is:

$$\mathbf{a}^{(l)} = g\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

Where:
- $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ — weight matrix of layer $l$
- $\mathbf{b}^{(l)} \in \mathbb{R}^{d_l}$ — bias vector
- $g(\cdot)$ — activation function

**Hidden layers — ReLU activation:**

$$g(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

ReLU is preferred over sigmoid/tanh because:
- No vanishing gradient problem for positive activations ($g'(z) = 1$ for $z > 0$)
- Sparse activations (many neurons output 0) → implicit regularization
- Computationally cheap

**Output layer — Softmax:**

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

#### Loss Function — Cross-Entropy + L2 Regularization

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} \mathbf{1}[y_i = k] \log P(y_i = k \mid \mathbf{x}_i) + \frac{\alpha}{2} \sum_{l} \|\mathbf{W}^{(l)}\|_F^2$$

Where:
- $\alpha = 0.001$ — L2 regularization strength (penalizes large weights, reduces overfitting)
- $\|\mathbf{W}\|_F^2 = \sum_{i,j} W_{ij}^2$ — Frobenius norm (sum of squared weights)

#### Backpropagation

Gradients are computed via the chain rule backward through the network:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(l)}} \cdot \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$$

For ReLU: $\frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} = \mathbf{1}[z_j^{(l)} > 0]$

#### Optimizer — Adam (Adaptive Moment Estimation)

The MLP uses Adam, which maintains exponential moving averages of the gradient $m_t$ and squared gradient $v_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

Bias-corrected estimates: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$

Weight update:
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t, \quad \eta = \text{learning\_rate\_init} = 0.005$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

#### Hyperparameter Meanings

| Param | Value | Effect |
|---|---|---|
| `hidden_layer_sizes` | (64, 32) | Network width — 64 neurons then 32; deeper but narrowing |
| `alpha` | 0.001 | L2 penalty strength — moderate regularization |
| `learning_rate_init` | 0.005 | Initial Adam step size |
| `max_iter` | 1000 | Maximum training epochs |

**Note on early_stopping:** Disabled due to a `numpy.isnan` incompatibility with string class labels in this sklearn/numpy version. `max_iter=1000` compensates by allowing sufficient training epochs.

#### Phase III Results

| Metric | Value |
|---|---|
| CV F1 (macro) | 0.9871 |
| Accuracy | 0.9742 |
| F1 (macro) | 0.9743 |
| ROC-AUC (macro) | 0.9989 |

**Per-class:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 0.96 | 0.99 | 0.98 | 105 |
| Low | 0.98 | 0.99 | 0.98 | 99 |
| Medium | 0.98 | 0.94 | 0.96 | 106 |

---

## 10. Evaluation Metrics — Full Formulas

For each class $k$, define:
- $TP_k$ = true positives for class $k$
- $FP_k$ = false positives for class $k$ (other classes predicted as $k$)
- $FN_k$ = false negatives for class $k$ ($k$ predicted as another class)

### Accuracy

$$\text{Accuracy} = \frac{\sum_k TP_k}{n} = \frac{\text{correctly classified}}{\text{total samples}}$$

### Precision (per class)

$$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}$$

"Of all samples predicted as class $k$, what fraction truly is class $k$?"

### Recall (per class)

$$\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

"Of all true class $k$ samples, what fraction did we correctly predict?"

### F1 Score (per class)

$$\text{F1}_k = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} = \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}$$

Harmonic mean of precision and recall — penalizes models that sacrifice one for the other.

### Macro-averaged F1

$$\text{F1}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_k$$

Each class contributes equally, regardless of its sample count. This is the primary ranking metric — it ensures the model is not judged only on the majority class.

### Weighted F1

$$\text{F1}_{\text{weighted}} = \frac{\sum_{k=1}^{K} n_k \cdot \text{F1}_k}{\sum_{k=1}^{K} n_k}$$

Weights by class support — more informative when classes are imbalanced.

### ROC-AUC (Macro, One-vs-Rest)

For each class $k$, train a binary classifier (class $k$ vs. all others). The ROC curve plots:

$$\text{TPR}_k(\tau) = \frac{TP_k(\tau)}{TP_k(\tau) + FN_k(\tau)}, \quad \text{FPR}_k(\tau) = \frac{FP_k(\tau)}{FP_k(\tau) + TN_k(\tau)}$$

as the decision threshold $\tau$ varies from 0 to 1.

$$\text{AUC}_k = \int_0^1 \text{TPR}_k(\text{FPR}) \; d(\text{FPR})$$

Macro-averaged:

$$\text{ROC-AUC}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{AUC}_k$$

AUC = 1.0 means perfect separation; AUC = 0.5 means random classifier.

**All Phase III models achieved ROC-AUC > 0.998** — near-perfect probability calibration and class separation.

---

## 11. Statistical Significance — Wilcoxon Signed-Rank Test

To confirm that Gradient Boosting is **statistically** better (not just numerically), a Wilcoxon signed-rank test is performed on the per-fold CV F1 scores.

### Setup

- Same `StratifiedKFold(n_splits=5)` object used for all models → identical fold assignments → **paired comparison**
- Reference: **Gradient Boosting** (highest CV F1 = 0.9992)
- Test: one-sided, $H_1$: Gradient Boosting F1 > other model's F1

### Wilcoxon Signed-Rank Procedure

For $n = 5$ paired differences $d_i = \text{GB}_i - \text{Model}_i$:

1. Compute $|d_i|$ and rank them from smallest to largest
2. Assign each rank the sign of $d_i$
3. Compute $W^+ = \sum_{\{i: d_i > 0\}} R_i$ (sum of positive-difference ranks)
4. Under $H_0$, $W^+$ follows a known discrete distribution; compute $p$-value

$$W^+ \geq W^+_{\text{critical}} \Rightarrow \text{reject } H_0$$

**Why Wilcoxon and not a t-test?**
The t-test assumes normally distributed differences. With $n=5$ paired observations, normality cannot be verified. Wilcoxon is a non-parametric test that only requires the differences to be symmetric — a weaker, more defensible assumption.

**Minimum achievable p-value with n=5 (one-sided):**

$$P(W^+ = 15) = \frac{1}{2^5} = \frac{1}{32} = 0.03125 < 0.05$$

(achieved when all 5 differences are positive and in the expected direction)

### Actual Results

| Model | Mean F1 | Statistic (W+) | p-value | Significant? |
|---|---|---|---|---|
| **Gradient Boosting** | 0.9992 | — (reference) | — | — |
| Random Forest | 0.9984 | 1.000 | 0.5000 | No |
| SVM (Linear) | 0.9904 | 10.000 | 0.0625 | No |
| Logistic Regression | 0.9872 | 15.000 | 0.0312 | **YES** |
| Neural Network (MLP) | 0.9871 | 15.000 | 0.0312 | **YES** |

**Interpretation:**
- Gradient Boosting is **statistically significantly better** than Logistic Regression and MLP (p < 0.05)
- The difference between Gradient Boosting and Random Forest is **not statistically significant** (p = 0.50) — they perform essentially identically on all 5 folds
- The difference between Gradient Boosting and SVM Linear is borderline (p = 0.0625, just above α = 0.05)

**Conclusion:** Gradient Boosting is declared the winner based on the highest CV F1 (0.9992) and statistically confirmed superiority over the weaker models.

---

## 12. Results

### Full Phase III Results Table

| Model | CV F1 | Accuracy | Precision | Recall | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.9992** | 0.9903 | 0.9904 | 0.9905 | 0.9904 | **0.9999** |
| Random Forest | 0.9984 | 0.9903 | 0.9904 | 0.9905 | 0.9904 | 0.9999 |
| SVM (Linear) | 0.9904 | 0.9839 | 0.9837 | 0.9840 | 0.9838 | 0.9995 |
| Logistic Regression | 0.9872 | 0.9806 | 0.9806 | 0.9807 | 0.9806 | 0.9993 |
| Neural Network (MLP) | 0.9871 | 0.9742 | 0.9745 | 0.9746 | 0.9743 | 0.9989 |

### Winner: Gradient Boosting

```
Best Parameters : n_estimators=500, learning_rate=0.2, max_depth=5,
                  subsample=0.9, min_samples_split=5
CV F1 (macro)   : 0.9992
Accuracy        : 0.9903  (307/310 correct)
F1 (macro)      : 0.9904
ROC-AUC (macro) : 0.9999
```

---

## 13. Phase II vs Phase III Comparison

| Model | Ph2 F1 | Ph3 F1 | Delta F1 | Ph2 CV F1 | Ph3 CV F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9741 | 0.9806 | **+0.0065** | 0.9847 | 0.9872 |
| Random Forest | 0.9904 | 0.9904 | +0.0000 | 0.9968 | 0.9984 |
| Gradient Boosting | 0.9904 | 0.9904 | +0.0000 | 0.9984 | **0.9992** |
| SVM (Linear) | 0.9709 | 0.9838 | **+0.0129** | 0.9863 | 0.9904 |
| Neural Network (MLP) | 0.9712 | 0.9743 | **+0.0031** | 0.9766 | 0.9871 |

**Key observations:**
1. **SVM Linear had the largest test F1 improvement (+0.0129)** — wider C range (up to 100) found C=50 which Phase II's grid (max C=10) missed entirely
2. **Logistic Regression improved by +0.0065** — C=100 was outside Phase II's grid
3. **RF and GB test F1 unchanged** — they were already near-optimal; Phase III confirmed it with higher CV confidence
4. **GB CV F1 improved from 0.9984 → 0.9992** — the larger parameter space (500 trees, subsample=0.9) squeezed out remaining variance
5. **All models improved or held** — no model degraded, confirming the wider search was beneficial

---

## 14. Output Files

| File | Description |
|---|---|
| `model_results_phase3.csv` | All 5 model metrics + best hyperparameters |
| `comparison_phase2_vs_phase3.csv` | Phase II vs III delta table |
| `classification_reports_phase3.txt` | Full per-class precision/recall/F1 for all models |
| `wilcoxon_results.txt` | Statistical significance test results |
| `final_report_phase3.md` | Executive summary report |
| `algorithm_comparison_phase3.png` | Grouped bar chart: Accuracy, Precision, Recall, F1 |
| `phase2_vs_phase3_comparison.png` | Side-by-side F1 with delta annotations |
| `feature_selection.png` | Importance bars (blue=kept, red=removed) with threshold line |
| `feature_importance_phase3.png` | Top features by RF importance (post-selection) |
| `learning_curves_phase3.png` | Train vs. validation F1 across training set sizes |
| `roc_auc_curves_phase3.png` | Macro-average ROC curves for all models |
| `calibration_curves_phase3.png` | Predicted probability vs. actual fraction per class |
| `confusion_matrix_gradient_boosting.png` | 3×3 heatmap — best model |
| `confusion_matrix_random_forest.png` | 3×3 heatmap |
| `confusion_matrix_logistic_regression.png` | 3×3 heatmap |
| `confusion_matrix_svm_linear.png` | 3×3 heatmap |
| `confusion_matrix_neural_network_mlp.png` | 3×3 heatmap |

---

## References

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189–1232.
- Friedman, J. H. (1999). *Stochastic Gradient Boosting*. Computational Statistics & Data Analysis, 38(4), 367–378.
- Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR, 13, 281–305.
- Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273–297.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines*. Advances in Large Margin Classifiers.
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. ICLR 2015.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16, 321–357.
- He, H. et al. (2008). *ADASYN: Adaptive Synthetic Sampling Approach*. IJCNN 2008.
- Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods*. Biometrics Bulletin, 1(6), 80–83.
