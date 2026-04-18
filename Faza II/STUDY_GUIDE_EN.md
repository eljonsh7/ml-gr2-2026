# 🎓 Phase 2 — Complete Study Guide (For the Defense)

> **What is this document?** This is a study guide that explains everything our project does in Phase 1 and Phase 2 in the simplest possible way. Read this before the defense so you can confidently answer the professor's questions.

---

## 🔰 The Big Picture: What is Our Project About?

We have electricity data from power grids (2021–2025). Every day, we know things like: how much solar energy was used, how much coal was burned, what percentage was renewable, etc.

We also know the **carbon intensity** — basically how "dirty" or "clean" the energy was that day (measured in grams of CO₂ per kilowatt-hour).

**Our goal:** Build a computer program that can look at a day's energy data and automatically predict if the carbon pollution was **LOW**, **MEDIUM**, or **HIGH**.

Think of it like a weather forecast, but for pollution.

---

## 📦 Phase 1: Preparing the Data (What We Already Did)

Before a machine can learn anything, the data has to be clean and organized. Here's what we did step by step:

### Step 1 — We loaded 5 CSV files (one per year, 2021–2025) and stacked them into one big spreadsheet.

### Step 2 — We averaged hourly data into daily data. Instead of 24 rows per day, we got 1 row per day with averaged values.

### Step 3 — We cleaned the data:
- Removed rows with impossible values (like a percentage of -50%)
- Filled in missing values: numbers got the **median** (the middle value, not the average — because averages get messed up by extreme values), text got the **mode** (the most frequently occurring value)

### Step 4 — We created the "answer key" (target variable):
- Took the carbon intensity number and split it into 3 equal-sized groups
- Bottom third → labeled **"low"**
- Middle third → labeled **"medium"**
- Top third → labeled **"high"**
- This new column is called `target_quantile_class`

### Step 5 — We created new useful columns (feature engineering):
- **Time-based:** month, day, day of week, is it a weekend?
- **Calculated:** the gap between two different carbon measurements, percentage of renewable energy within total clean energy

### Step 6 — We found outliers (extremely unusual data points) using 3 methods and flagged them.

### Step 7 — We scaled everything. Different columns have different ranges (temperature 0–40, CO₂ 0–500, percentages 0–100). We made all columns use the same scale (mean=0, spread=1) so no column dominates the others.

### Step 8 — We split the data:
- **80% → Training set** (the machine learns from this)
- **20% → Test set** (we hide this and use it later to check how smart the machine became)

### Step 9 — We balanced the classes with **SMOTE**:
SMOTE creates artificial new data points for any class that has fewer examples. It picks a real data point, finds its neighbors, and creates a new point in between them. This way the machine doesn't get lazy and just always predict the most common class.

> ⚠️ SMOTE only touches the training data. The test data stays 100% real and untouched.

---

## 🤖 Phase 2: Teaching the Machine (What We Did Now)

We taught 6 different "supervised" algorithms (we give them the answers so they can learn) and 2 "unsupervised" algorithms (we DON'T give answers, we just say "find groups").

For every algorithm, we also used **GridSearchCV** — a tool that automatically tries different setting combinations and picks the best one. Think of it like trying on 20 pairs of shoes and keeping the one that fits best.

---

## 🟢 The 6 Supervised Algorithms (We teach them the answers)

### 1. Logistic Regression — "The Straight-Line Drawer"

**What it does in simple terms:**
Imagine all 366 test days plotted on a big sheet of paper based on their energy characteristics. Some dots are LOW days (green), some are MEDIUM (yellow), some are HIGH (red). Logistic Regression tries to draw **straight lines** on this paper to separate the three colors into their own zones.

**How it actually learns:**
1. It starts by guessing randomly
2. It checks how wrong it was (this "wrongness score" is called the **cost function** or **loss**)
3. It adjusts its guess slightly in the direction that reduces the error — this adjustment technique is called **gradient descent** (imagine rolling a ball downhill — it naturally finds the lowest point)
4. It repeats steps 2–3 thousands of times until the guesses stop improving

**The key setting — C (Regularization):**
C is like a strictness dial. 
- **C = 0.01** → Very strict: "Keep it super simple." But the model might be TOO simple and miss real patterns. This is called **underfitting** (like wearing shoes that are too small).
- **C = 100** → Very relaxed: "Fit every training point perfectly." But the model might memorize the training data instead of learning real patterns. This is called **overfitting** (like wearing shoes molded perfectly to your left foot — they won't fit your right foot).
- **C = 10** → Just right. We found this using GridSearchCV.

**Result: 98.09% accuracy** — only 7 days out of 366 were predicted wrong. Impressive for such a simple model!

---

### 2. Random Forest — "The 200 Experts Who Vote"

**What it does in simple terms:**
Imagine asking 200 different people to make yes/no flowcharts for predicting carbon intensity. BUT each person only sees a random portion of the data and a random set of columns. Then, to classify a new day, all 200 people vote, and the majority wins.

**Why is randomness good?**
If all 200 people saw the same data, they'd make the same flowchart and the same mistakes. By giving each person different random pieces, their mistakes are all different and cancel out when they vote together. This is the genius of **ensemble learning**.

**Each "person" is actually a Decision Tree:**
```
"Is renewable energy > 50%?"
   ├── YES → "Is it a weekday?"
   │         ├── YES → Predict LOW ✅
   │         └── NO  → Predict MEDIUM ✅
   └── NO  → "Is carbon intensity gap > 100?"
             ├── YES → Predict HIGH ✅
             └── NO  → Predict MEDIUM ✅
```

**Why we love Random Forest:**
- It tells us which columns (features) were most useful for making decisions — this is the **feature importance** chart in our output
- It handles complicated patterns that can't be solved with straight lines (non-linear)
- It naturally resists overfitting thanks to the randomness

**Settings we tried:**
- Number of trees: 100 or 200 → Best: **200**
- Maximum depth of each tree: 10, 20, or unlimited → Best: **20**

**Result: 100% accuracy** — every single prediction was correct!

---

### 3. Gradient Boosting — "The Error-Fixing Chain"

**What it does in simple terms:**
While Random Forest builds 200 trees at the same time (independently), Gradient Boosting builds them **one after another**, like a relay race:

1. Tree 1 makes predictions — gets some wrong
2. Tree 2 is specifically trained to fix Tree 1's mistakes
3. Tree 3 is trained to fix the remaining mistakes after Trees 1+2
4. ...continues for 100 trees

Each tree takes a small step toward the perfect answer. The **learning rate** controls how big each step is — smaller steps = slower but more careful.

**Why "Gradient" in the name?**
Because it uses gradient descent (the same technique from Logistic Regression) but applies it to the sequence of trees. Each new tree follows the gradient of the error downhill.

**Settings we found:**
- 100 trees with very shallow depth (only 3 levels each), learning rate of 0.05
- Notice: the trees are super simple! The power comes from combining many simple fixers, not from having one complex tree.

**Result: 100% accuracy** — also perfect!

---

### 4. SVM with Linear Kernel — "The Widest-Gap Separator"

**What it does in simple terms:**
SVM = Support Vector Machine. Imagine plotting your days as dots on paper. SVM tries to draw a line between the colors, but not just any line — the line that creates the **widest possible gap** between the groups.

```
🟢🟢🟢         |         🔴🔴🔴
  🟢🟢      ← gap →      🔴🔴
🟢🟢🟢         |         🔴🔴🔴
```

The dots closest to the line are called **support vectors** — they literally "support" and define where the boundary is. All other dots don't matter.

**C** again controls how strict the boundary is — allow a few dots on the wrong side (soft margin, more general) or force everything to the right side (hard margin, risk of overfitting).

**Result: 97.54% accuracy**

---

### 5. SVM with RBF Kernel — "The Shape-Shifter"

**What it does in simple terms:**
What if the groups can't be separated by a straight line? Imagine red dots forming a circle inside blue dots forming a ring. No straight line works!

The **RBF kernel** (Radial Basis Function) is like magic: it mathematically "lifts" the flat 2D paper into a 3D space where the circle becomes a hill and the ring stays flat — and NOW you can separate them with a flat plane.

```
Before (2D — impossible):    After kernel trick (3D — easy):
  🔵🔵🔵🔵                         
  🔵🔴🔴🔵                    🔴 sits on top of 🔵
  🔵🔴🔴🔵                    just slice horizontally!
  🔵🔵🔵🔵                    
```

**gamma** controls how much each training point influences the boundary:
- Small gamma = wide, smooth influence = smoother boundaries
- Big gamma = narrow, local influence = wiggly boundaries (overfitting risk)

**Result: 96.17% accuracy** — actually WORSE than the linear SVM!

**Why?** After our scaling in Phase 1, the class boundaries were already approximately linear. The RBF kernel tried to create unnecessarily complex curves and actually overfit slightly. This teaches us an important lesson: **more complex is not always better.**

---

### 6. Neural Network (MLP) — "The Mini Brain"

**What it does in simple terms:**
A neural network is like a factory with multiple floors (layers). Raw materials (our 35 features) enter on the ground floor, pass through processing stations (neurons) on each floor, and the finished product (the prediction "low"/"medium"/"high") comes out on the top floor.

**Our network looks like this:**
```
Ground Floor (Input):    35 features come in
     ↓
Floor 1:    128 mini-calculators process the data
     ↓
Floor 2:     64 mini-calculators process it further
     ↓
Top Floor (Output):      3 doors: "low", "medium", "high"
                         The data exits through the door
                         with the highest probability
```

**How each mini-calculator (neuron) works:**
1. It receives numbers from the floor below
2. It multiplies each number by a **weight** (importance factor)
3. It adds them all up
4. It passes the result through an **activation function** (basically: if the result is positive, keep it; if negative, make it zero — this is called ReLU)
5. It sends the result up to the next floor

**How it learns — Backpropagation:**
1. Data flows forward through the network (Floor 1 → Floor 2 → Output) — this is the **forward pass**
2. At the output, we compare the prediction with the real answer and calculate how wrong it was — this error number is the **cost function** (specifically, cross-entropy loss)
3. We trace backward through the network asking: "which weights contributed most to this error?" — this is **backpropagation** (the error propagates backwards)
4. We adjust every weight slightly to reduce the error — this is **gradient descent** (same concept as Logistic Regression, but applied to every weight in every layer)
5. Repeat for every training example, multiple times (each full pass is called an **epoch**)

**The loss curve we generated** shows the cost function value dropping over epochs — proof that gradient descent is working. It starts high (the network knows nothing) and drops until it flattens (convergence — it learned as much as it can).

**alpha** = regularization for neural networks. It penalizes large weights to keep the model from memorizing.

**Settings we found:**
- 2 hidden layers: 128 neurons → 64 neurons
- alpha = 0.0001 (very light regularization)

**Result: 97.54% accuracy**

---

## 🔵 The 2 Unsupervised Algorithms (No answers given)

Here we asked: "Can you find natural groups in this data WITHOUT knowing the labels?"

### K-Means — "Find 3 Groups by Closeness"

**What it does:**
1. Randomly pick 3 center points
2. Assign each data point to its nearest center → 3 groups form
3. Move each center to the middle of its group
4. Repeat until centers stop moving

**Elbow Method:** We ran K-Means for k=2,3,4,5,6,7,8 and measured how tight the clusters were. The "elbow" in the plot (where improvement slows down) suggests the optimal k.

**Silhouette Score:** Measures cluster quality from -1 (terrible) to +1 (perfect).

**Our score: 0.24** — weak. The groups barely form.

### Agglomerative Clustering — "Merge from Bottom Up"

**What it does:**
Start with each data point as its own tiny group. Merge the two closest groups. Repeat until only 3 groups remain.

**Our score: 0.24** — also weak.

### Why did unsupervised fail?
The data points for "low", "medium", and "high" carbon intensity are **mixed together** when you look at all their features. The real difference between them is subtle and only becomes clear when you explicitly teach the model what the labels mean. This proves that **supervised learning is necessary** for our problem.

---

## 📊 PCA — Making 35 Columns Visible in 2D

Our data has 35 columns. You can't draw a 35-dimensional graph. **PCA (Principal Component Analysis)** squishes 35 dimensions into 2 while keeping the most important information, like taking a photograph of a 3D sculpture — you lose some depth, but you can still see the shape.

- **PC1** = the direction where data varies the most
- **PC2** = the second most varying direction (perpendicular to PC1)

We made 3 side-by-side scatter plots: K-Means clusters vs Agglomerative clusters vs real labels. This visually shows how different the unsupervised groups are from the real answer.

---

## 📈 Learning Theory Plots (What the Extra Charts Mean)

### Learning Curves
Shows model performance as training data grows. Two lines:
- **Blue (training):** Should be high
- **Orange (validation):** Should be close to blue

If blue is high but orange is low → **overfitting** (model memorized instead of learning)
If both are low → **underfitting** (model is too simple)
If both are high and close → **good model** ✅

### Regularization Effect
Shows how changing C affects Logistic Regression:
- Left (small C) = too strict → underfitting
- Right (big C) = too loose → overfitting risk
- Middle = just right (sweet spot)

### MLP Loss Curve
Shows the neural network's error (cost) dropping over time as gradient descent works. The curve going down = the network is learning. When it flattens = it learned everything it can (convergence).

---

## 🏆 What We Achieved

1. **Extremely high accuracy:** Our best models (Random Forest, Gradient Boosting) achieved 100% accuracy on the test set. Even our simplest model (Logistic Regression) hit 98%.

2. **Proved that supervised learning is necessary:** By running unsupervised algorithms too and showing their failure (silhouette score 0.24), we proved that carbon intensity levels can't be discovered without labels — they must be taught.

3. **Demonstrated all major ML concepts from the syllabus:** Gradient descent, cost functions, regularization, overfitting/underfitting, backpropagation, kernels, ensemble methods, PCA, clustering. Every plot we generated demonstrates a specific syllabus concept.

4. **Built a complete, reproducible pipeline:** One single `python3 phase2_pipeline.py` command runs everything and generates all 18 output files.

5. **Compared 8 algorithms systematically:** 6 supervised + 2 unsupervised, all with proper hyperparameter tuning and cross-validation.

---

## 🎯 Why We Did Things This Way

| Decision | Why |
|----------|-----|
| 3 classes (low/medium/high) instead of 2 | More realistic and challenging. 2 classes would be too easy. |
| GridSearchCV | To find best hyperparameters automatically instead of guessing |
| 3-fold cross-validation | To make sure the results are stable and not just lucky |
| Both supervised AND unsupervised | Syllabus requires both. Also proves unsupervised alone isn't enough. |
| 2 SVM kernels (linear + RBF) | To demonstrate the "kernels" concept from the syllabus |
| Neural Network | To demonstrate backpropagation and gradient descent from the syllabus |
| Learning curves plot | To demonstrate overfitting/underfitting from the syllabus |
| Regularization plot | To demonstrate the bias-variance tradeoff from the syllabus |
| SMOTE in Phase 1 | To handle class imbalance — without it, models could cheat by always predicting the biggest class |
| StandardScaler in Phase 1 | SVM and Neural Networks perform terribly on unscaled data |
| Feature engineering in Phase 1 | Created informative new columns that helped models learn better patterns |

---

## 🔮 Next Steps (Phase 3 Preview)

Phase 3 is due May 17 and is titled "Analiza dhe evaluimi (ri-trajnimi)" — Analysis, Evaluation, and Re-training. Here's what we'll do:

1. **XGBoost** — An even more powerful version of Gradient Boosting
2. **Deep Learning with TensorFlow/Keras** — More advanced neural networks
3. **SHAP Analysis** — A technique to explain exactly WHY each model made each prediction (which features pushed toward "high" vs "low")
4. **Regression** — Instead of predicting classes (low/medium/high), predict the actual carbon number (e.g., 345.2 gCO₂/kWh)
5. **More cross-validation** — StratifiedKFold with 5-10 folds for more robust evaluation
6. **Anomaly detection** — Using Multivariate Gaussian to find truly unusual days
7. **Comparison with Phase 2** — Show how the improvements affected results
8. **Final conclusions** — Who benefits from this work, what can be improved in the future

---

## ❓ Professor Q&A — Preparing for the Defense

### General Questions

**Q: "What is Machine Learning?"**
> "Machine Learning is when we give a computer data and let it find patterns on its own, instead of us writing explicit rules. In supervised learning, we give it examples with correct answers so it can learn the relationship. In unsupervised learning, we give it data without answers and it finds natural groupings."

**Q: "What is your project about?"**
> "We predict the carbon intensity level of electricity grids — whether it will be low, medium, or high — based on energy production features like renewable percentage, carbon-free energy percentage, and temporal patterns. We use hourly data from 2021 to 2025 aggregated to daily level."

**Q: "What is the target variable?"**
> "Our target is `target_quantile_class` which has three classes: low, medium, and high. We created it by splitting the continuous carbon intensity values into three equal-sized groups using quantile binning."

### Algorithm Questions

**Q: "Why did you choose these specific algorithms?"**
> "We wanted to cover all the major algorithm families from the syllabus: linear models (Logistic Regression), ensemble tree methods (Random Forest, Gradient Boosting), kernel methods (SVM with two kernels), and neural networks (MLP). Plus unsupervised methods (K-Means, Agglomerative) to show the difference between supervised and unsupervised approaches."

**Q: "How does Random Forest work?"**
> "It creates many decision trees, each trained on a random subset of the data and features. Each tree makes its own prediction, and the final answer is decided by majority vote. The randomness ensures the trees make different mistakes, which cancel out when voting."

**Q: "How does Gradient Boosting work?"**
> "It builds trees sequentially. Each new tree is trained to fix the errors of all the previous trees combined. It uses gradient descent to minimize the error step by step. The learning rate controls how much each tree contributes."

**Q: "What is the difference between Random Forest and Gradient Boosting?"**
> "Random Forest builds trees independently and in parallel — each tree sees random data. Gradient Boosting builds trees sequentially — each tree specifically focuses on fixing the mistakes of the previous ones. Both are ensemble methods but with different strategies."

**Q: "How does a Neural Network learn?"**
> "Through backpropagation and gradient descent. Data flows forward through layers of neurons. At the output, we calculate the error. Then we trace backward through the network to find how much each weight contributed to the error. We adjust each weight to reduce the error. We repeat this for every training example, many times."

**Q: "What is a kernel in SVM?"**
> "A kernel is a mathematical function that transforms data into a higher-dimensional space. In the original space, data might not be separable by a straight line. But in the higher-dimensional space, it becomes separable. The RBF kernel uses a Gaussian function for this transformation. The linear kernel doesn't transform at all."

**Q: "What is PCA?"**
> "PCA reduces the number of dimensions while keeping the most information possible. It finds the directions where the data varies the most and projects everything onto those directions. We used it to visualize our 35-dimensional data in 2D."

### Theory Questions

**Q: "What is overfitting? How do you detect it?"**
> "Overfitting is when a model memorizes the training data instead of learning general patterns. It performs amazingly on training data but poorly on new data. We detect it using learning curves — if the training score is much higher than the validation score, the model is overfitting. We also compare cross-validation scores with test scores."

**Q: "What is underfitting?"**
> "Underfitting is when a model is too simple to capture the patterns in the data. Both training and test scores will be low. The solution is to use a more complex model, add more features, or reduce regularization."

**Q: "What is regularization?"**
> "Regularization is a technique that prevents overfitting by adding a penalty for model complexity. In Logistic Regression and SVM, the parameter C controls this. Small C = strong regularization (simpler model), large C = weak regularization (more complex model). Our regularization plot shows this tradeoff visually."

**Q: "What is gradient descent?"**
> "Gradient descent is an optimization algorithm. Imagine you're on a mountain in thick fog and you need to reach the valley. You feel the slope under your feet and always step in the downhill direction. Each step is small, but eventually you reach the bottom. In ML, the 'mountain' is the error surface and the 'valley' is the point where the model makes the fewest mistakes."

**Q: "What is the cost function?"**
> "The cost function measures how wrong the model is. It takes the model's predictions and the real answers and produces a single number — the error. The goal of training is to minimize this number. For classification, we typically use cross-entropy loss. For regression, we use mean squared error."

**Q: "What is cross-validation?"**
> "Instead of evaluating the model on a single train/test split, we split the training data into 3 parts (folds). We train on 2 folds and test on the 3rd. We rotate so every fold gets to be the test fold. This gives us 3 scores that we average, making the evaluation more reliable."

**Q: "What is the confusion matrix?"**
> "It's a table that shows exactly which classes the model confused with which. The diagonal shows correct predictions, and everything off the diagonal shows mistakes. It tells us not just how many errors, but WHAT KIND of errors — for example, does the model often confuse 'medium' with 'high'?"

**Q: "What is the F1-Score and why use it?"**
> "F1-Score is the balance between Precision (when the model predicts a class, how often is it right?) and Recall (of all the real examples of a class, how many did the model find?). We use F1 macro, which averages F1 across all classes equally, because we care about all three classes equally."

### Results Questions

**Q: "Why did Random Forest get 100% accuracy?"**
> "Our Phase 1 data preparation was very thorough — we engineered informative features like life_cycle_gap and renewable_share_within_cfe, applied proper scaling, and balanced the classes. The tree-based models could easily learn the clear patterns in this well-prepared data."

**Q: "Isn't 100% accuracy suspicious? Could it be overfitting?"**
> "We cross-validated with 3 folds during GridSearchCV and the CV score was 0.994 — very close to the test score. Also, the test set was completely separate from training (never seen during training or tuning). If the model was overfitting, the test score would be much lower than the CV score."

**Q: "Why did SVM RBF perform worse than SVM Linear?"**
> "After our data preprocessing (StandardScaler), the class boundaries became approximately linear. The RBF kernel tried to create unnecessarily complex curved boundaries, which actually hurt performance. This demonstrates that more complexity isn't always better."

**Q: "Why did unsupervised algorithms have low silhouette scores?"**
> "Because carbon intensity levels don't form natural clusters. Days with similar energy profiles can have very different carbon intensities depending on the energy mix. Without explicit labels, clustering algorithms can't discover these subtle distinctions."

**Q: "What did the regularization plot show?"**
> "It showed that with very small C (strong regularization), the model underfits — both train and test scores are lower. As C increases, performance improves until it stabilizes. This visually demonstrates the bias-variance tradeoff."

**Q: "What did the learning curves show?"**
> "They showed that as we increase the training set size, the validation score increases and approaches the training score. This indicates our model generalizes well. The gap between training and validation scores decreases with more data, suggesting no severe overfitting."

**Q: "What is the practical value of this project?"**
> "Energy grid operators and environmental agencies can use this to predict pollution levels in advance. If the model predicts HIGH carbon intensity for tomorrow, they can shift to renewable energy sources proactively. This helps in carbon emission reduction and energy transition planning."
