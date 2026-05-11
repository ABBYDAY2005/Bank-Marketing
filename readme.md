# Bank Marketing Campaign — Subscription Prediction

A machine learning classification project that predicts whether a bank customer will subscribe to a term deposit, based on the UCI Bank Marketing dataset.

---

## Problem Statement

Banks run telemarketing campaigns to sell term deposit subscriptions. This project builds and compares multiple classifiers to identify which customers are likely to say **yes**, enabling more targeted and cost-effective outreach.

---

## Dataset

**Source:** [Kaggle — Bank Marketing Dataset](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)  
**File used:** `bank-additional-full.csv` (semicolon-separated)  
**Target column:** `y` — whether the client subscribed (`yes` / `no`)

The dataset is highly **imbalanced** (majority `no` class), which is addressed using SMOTE oversampling.

---

## Project Structure

```
Bank_Marketing.ipynb     ← Main notebook (all steps end-to-end)
README.md
```

---

## Workflow

### 1. Data Loading
- Dataset downloaded from Kaggle via the Kaggle API inside Google Colab.
- Loaded with `pandas` using `;` as the separator.

### 2. Exploratory Analysis
- `df.info()` for schema and null checks.
- Class distribution check on the target `y`.

### 3. Preprocessing
- **Feature/target split:** `X` = all columns except `y`; `y` mapped to binary (1 = yes, 0 = no).
- **Numerical pipeline:** Median imputation → StandardScaler.
- **Categorical pipeline:** Mode imputation → OneHotEncoder (unknown categories ignored).
- Combined using `ColumnTransformer`.

### 4. Train-Test Split
- 80/20 stratified split (`random_state=42`) to preserve class ratios.

### 5. Class Imbalance Handling
- **SMOTE** (Synthetic Minority Oversampling Technique) applied on the training set only, after preprocessing.

### 6. Models Trained

| Model | Key Config |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | Default |
| Random Forest | `n_estimators=200`, `n_jobs=-1` |
| K-Nearest Neighbors | `k=5` |
| Support Vector Classifier | `probability=True` |
| XGBoost | `eval_metric='logloss'` |

### 7. Evaluation Metrics
Each model is evaluated on the held-out test set:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC

Results are compiled into a DataFrame sorted by **F1-Score**.

### 8. Visualizations
- **Bar plots:** Side-by-side comparison of all metrics across models.
- **ROC Curves:** All models on a single plot with AUC scores in the legend.
- **Feature Importance:** Top 15 features for Random Forest and XGBoost.

---

## Requirements

```
pandas
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
kaggle
```

Install via:
```bash
pip install pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn kaggle
```

---

## How to Run

1. Upload your `kaggle.json` API key when prompted in Colab.
2. Run all cells top to bottom.
3. The notebook will download the dataset, preprocess it, train all models, and display evaluation results and plots automatically.

---

## Key Design Decisions

- **SMOTE after split** — oversampling is applied only to training data to prevent data leakage.
- **Stratified split** — preserves the original class ratio in both train and test sets.
- **Probability calibration** — `SVC(probability=True)` enables ROC-AUC computation for SVM.
- **Feature importance** — extracted post-training from Random Forest and XGBoost to identify the most predictive client attributes.