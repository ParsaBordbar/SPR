# Complete Assignment Summary: Linear & Logistic Regression

## Overview

This assignment covers three fundamental machine learning techniques:

| Question | Topic | Task | Dataset |
|----------|-------|------|---------|
| **Q1** | Linear Regression | Predict MPG from weight | auto_mpg.csv |
| **Q2** | Logistic Regression | Predict bank subscription | bank-full.csv |
| **Q3** | Multi-class Classification | Classify wine varieties | wine_dataset.csv |

---

## Question 1: Linear Regression

### Goal
Predict fuel efficiency (MPG) from vehicle weight using three different methods.

### Key Concepts
- **Model:** y = θ₀ + θ₁*x (a line)
- **Cost Function:** MSE = (1/2m) * Σ(y_pred - y_true)²
- **Bias Term:** θ₀ (the intercept, where line crosses y-axis)
- **Normalization:** (X - mean) / std for faster convergence

### Three Methods Implemented

#### A) Batch Gradient Descent (BGD)
- Updates parameters using ALL samples
- Cost function decreases smoothly
- **Pros:** Stable, smooth convergence
- **Cons:** Slower per iteration

#### B) Stochastic Gradient Descent (SGD)
- Updates parameters after EACH sample
- Cost function fluctuates
- **Pros:** Faster per iteration
- **Cons:** Noisier convergence

#### C) Closed-Form Solution (Normal Equation)
- Direct mathematical solution: θ = (X^T X)^-1 X^T y
- **Pros:** Instant, no iterations needed
- **Cons:** Computationally expensive for large data

### Results
All three methods should converge to similar solution:
```
θ₀ ≈ 23.5 (intercept/bias)
θ₁ ≈ -0.003 (slope)
Test MSE ≈ 11.5
```

### Implementation Checklist
- ✅ Exploratory Data Analysis (shape, stats, correlation, plots)
- ✅ 80-20 train-test split with random_state=42
- ✅ Normalize X using training set statistics
- ✅ Implement BGD with multiple learning rates
- ✅ Implement SGD with multiple learning rates
- ✅ Implement closed-form solution
- ✅ Compare all three methods
- ✅ Visualize cost functions and regression lines

---

## Question 2: Logistic Regression

### Goal
Predict binary target (Subscribe: yes/no) using customer features.

### Key Concepts
- **Model:** y = sigmoid(z) where z = θ₀ + θ₁*x₁ + θ₂*x₂ + ...
- **Sigmoid Function:** 1 / (1 + e^(-z)) → outputs probability [0, 1]
- **Cost Function:** Cross-entropy loss = -1/m * Σ(y*log(y_pred) + (1-y)*log(1-y_pred))
- **Decision Boundary:** 0.5 probability

### Data Preparation (Task 2.3)

1. **Encode Categorical Features**
   - **Ordinal Encoding:** For ordered features (e.g., education: primary < secondary < tertiary)
   - **One-Hot Encoding:** For unordered features (e.g., job type)

2. **Check Class Balance**
   - If imbalanced (< 30% minority), use F1-Score instead of accuracy

3. **Select Features by Correlation**
   - Compute |correlation| with target
   - Keep top 7 features

4. **Remove Duplicates**
   - Only from training set
   - Duplicates provide no new information

5. **Stratified Train-Test Split (80-20)**
   - Preserve class proportions in both sets
   - Prevents misleading results

6. **Standardize Numerical Features**
   - Only for numerical features (not one-hot encoded)
   - (X - mean) / std using training set statistics
   - Apply to both train and test with training stats

### Evaluation Metrics (ALL FOUR REQUIRED)

```
Confusion Matrix:
                 Predicted
            Negative    Positive
Actual   No    TN        FP
        Yes    FN        TP
```

1. **Accuracy** = (TP + TN) / Total
   - Overall correctness
   - ⚠️ Misleading on imbalanced data

2. **Precision** = TP / (TP + FP)
   - When I say YES, am I right?
   - Use when false positives are expensive

3. **Recall** = TP / (TP + FN)
   - Do I find all YES cases?
   - Use when false negatives are expensive

4. **F1-Score** = 2 * (Precision * Recall) / (Precision + Recall)
   - Balanced metric
   - Best for imbalanced datasets

### Expected Results
```
Train Accuracy: 85-95%
Test Accuracy: 85-92%
Precision: 40-70% (many false positives ok)
Recall: 50-80% (want to find subscribers)
F1-Score: 50-75%
```

### Implementation Checklist
- ✅ Explore dataset
- ✅ Encode categorical features
- ✅ Check class balance
- ✅ Select top 7 features
- ✅ Remove duplicates
- ✅ Stratified train-test split
- ✅ Standardize numerical features
- ✅ Plot feature distributions
- ✅ Implement logistic regression
- ✅ Train with multiple learning rates
- ✅ Compute confusion matrix
- ✅ Calculate all 4 metrics
- ✅ Visualize cost function

---

## Question 3: Multi-class Classification

### Goal
Classify wine into one of K classes (e.g., 3 varieties).

### Data Preparation (Task 3.2)

1. **Handle Missing Values**
   - Compute mean from TRAINING set
   - Fill using training mean in both sets

2. **Remove Duplicates**
   - Only from training set

3. **Detect & Remove Outliers (Z-score)**
   - Threshold: |Z| > 2.75
   - Z = |X - mean| / std
   - Remove only from training set

4. **Normalize with Min-Max**
   - Formula: (X - min) / (max - min)
   - Range: [0, 1]
   - Use training min/max for test set

### Three Methods

#### A) One-vs-All (OvA)

Train K binary classifiers (one per class):
```
Class 1 vs. Rest
Class 2 vs. Rest
Class 3 vs. Rest
```

**Prediction:** Take highest probability

**Pros:** Fast, simple
**Cons:** Class imbalance issues

#### B) One-vs-One (OvO)

Train K(K-1)/2 binary classifiers (all pairs):
```
Class 1 vs. Class 2
Class 1 vs. Class 3
Class 2 vs. Class 3
```

**Prediction:** Voting scheme (majority wins)

**Pros:** Better for imbalanced data, often best accuracy
**Cons:** More classifiers, slower

#### C) Softmax Regression

Single model that directly outputs probability for each class:
```
z = X @ Theta  (shape: m × K)
y_pred = softmax(z)  (probabilities sum to 1)

softmax(z)_i = e^(z_i) / Σ(e^(z_j))
```

**Prediction:** Highest probability

**Pros:** Elegant, fast, efficient
**Cons:** Slightly harder to understand

### Comparison

| Metric | OvA | OvO | Softmax |
|--------|-----|-----|---------|
| # Models | K | K(K-1)/2 | 1 |
| Training | Fast | Slower | Fast |
| Inference | Medium | Slow | Fastest |
| Memory | Low | High | Lowest |
| Accuracy | Good | Often best | Good |

### Expected Results
```
OvA Test Accuracy:     85-92%
OvO Test Accuracy:     87-94% (often highest)
Softmax Test Accuracy: 86-93%
```

### Implementation Checklist
- ✅ Load and explore wine dataset
- ✅ Handle categorical features
- ✅ Handle missing values (train only)
- ✅ Remove duplicates (train only)
- ✅ Detect outliers (Z-score, threshold=2.75)
- ✅ Remove outliers (train only)
- ✅ Normalize with Min-Max
- ✅ Plot feature distributions
- ✅ Implement One-vs-All
- ✅ Implement One-vs-One
- ✅ Implement Softmax
- ✅ Train all three methods
- ✅ Compare accuracies
- ✅ Visualize results
- ✅ Identify best method

---

## Key Principles Across All Questions

### 1. Data Leakage Prevention
```
✅ CORRECT:
   mean = compute(X_train)
   X_train_norm = (X_train - mean) / std
   X_test_norm = (X_test - mean) / std  # Use training stats!

❌ WRONG:
   mean = compute(X)  # Uses test data!
   X_train_norm = (X_train - mean) / std
```

### 2. Normalization vs Standardization

| Method | Formula | Range | Use When |
|--------|---------|-------|----------|
| Standardization | (X - mean) / std | (-∞, +∞) | Linear regression |
| Min-Max | (X - min) / (max - min) | [0, 1] | Multi-class, neural networks |

### 3. Train-Test Split
- **Stratified:** For classification (preserve class proportions)
- **Random:** For regression

### 4. Bias Term (θ₀)
- The intercept in the model
- Always add column of 1s to X
- X_with_bias = np.column_stack([np.ones(m), X])

### 5. Cost Functions
- **Linear:** MSE = (1/2m) * Σ(error²)
- **Logistic:** Cross-entropy = -1/m * Σ(y*log(pred) + (1-y)*log(1-pred))
- **Multi-class:** Cross-entropy = -1/m * Σ(y_one_hot * log(y_pred))

### 6. Gradient Descent
Same formula for all:
```
gradients = (1/m) * X^T @ (y_pred - y_true)
theta -= learning_rate * gradients
```

---

## Files to Submit

```
YourNames.zip
├── question1_linear_regression.py
├── question2_logistic_regression.py
├── question3_multiclass_classification.py
├── report.pdf or report.md
├── task1_exploratory_analysis.png
├── linear_regression_results.png
├── task2_numerical_distributions.png
├── task2_logistic_regression.png
├── task3_multiclass_classification.png
└── README.md (optional but helpful)
```

---

## Summary Statistics

### Question 1
- Dataset: auto_mpg.csv (392 samples, 4 features)
- Target: MPG (continuous)
- Model: Linear regression
- Feature: Weight (normalized)
- Methods: BGD, SGD, Closed-form

### Question 2
- Dataset: bank-full.csv (45211 samples, 20 features)
- Target: Subscription (binary: yes/no)
- Model: Logistic regression
- Features: 7 selected by correlation
- Metrics: Accuracy, Precision, Recall, F1-Score

### Question 3
- Dataset: wine_dataset.csv (178 samples, 13 features)
- Target: Wine class (multi-class: 1, 2, 3)
- Models: OvA (3), OvO (3), Softmax (1)
- Preprocessing: Outlier removal, Min-Max normalization

---

## Common Mistakes & Fixes

### Mistake 1: Broadcasting Error
```
❌ WRONG: y shape is (m, 1) instead of (m,)
✅ FIX: y = y.flatten()
```

### Mistake 2: Forgetting Bias Term
```
❌ WRONG: y_pred = X @ theta  # theta is (n,)
✅ FIX: X_with_bias = np.column_stack([np.ones(m), X])
        y_pred = X_with_bias @ theta
```

### Mistake 3: Data Leakage
```
❌ WRONG: mean = np.mean(X)  # Full data!
✅ FIX: mean = np.mean(X_train)
```

### Mistake 4: Wrong Split
```
❌ WRONG: train_indices = random_indices
         test_indices = random_indices  # Different!
✅ FIX: Use same indices from shuffled array
```

### Mistake 5: Not Standardizing
```
❌ WRONG: X_train = X[train_indices]
         X_test = X[test_indices]
✅ FIX: Standardize before splitting (on full data for train params)
```

---

## Formulas Quick Reference

### Linear Regression
```
Hypothesis: y = θ₀ + θ₁*x
Cost: J = (1/2m) * Σ(y_pred - y)²
Gradient: dJ/dθ = (1/m) * Σ(y_pred - y) * x
```

### Logistic Regression
```
Hypothesis: y = sigmoid(θ₀ + θ₁*x)
Sigmoid: σ(z) = 1 / (1 + e^(-z))
Cost: J = -1/m * Σ(y*log(y_pred) + (1-y)*log(1-y_pred))
Gradient: dJ/dθ = (1/m) * X^T @ (y_pred - y)
```

### Softmax
```
softmax(z)_i = e^(z_i) / Σ(e^(z_j))
Cost: J = -1/m * Σ(y_one_hot * log(y_pred))
```

---

## Learning Rates to Try

```
Linear Regression:
  BGD: 0.001, 0.01, 0.1
  SGD: 0.001, 0.01, 0.1

Logistic Regression:
  0.01, 0.1, 1.0

Multi-class:
  0.01, 0.1, 1.0
```

---

## Final Checklist Before Submission

- ✅ All three questions implemented
- ✅ Code runs without errors
- ✅ All metrics computed and reported
- ✅ Visualizations created and saved
- ✅ Report includes all results
- ✅ Code is commented and readable
- ✅ No data leakage
- ✅ Proper train-test split used
- ✅ All files packaged in ZIP
- ✅ ZIP file named: YourNames.zip

---

## You're Ready! 🚀

You now have:
- ✅ Complete implementations for all 3 questions
- ✅ Comprehensive explanations of all concepts
- ✅ Visual guides and examples
- ✅ Quick reference cards
- ✅ Bug fixes and common mistakes guide
- ✅ All formulas and code snippets

**Time to submit!**

Good luck! You've mastered Linear Regression, Logistic Regression, and Multi-class Classification! 🎉
