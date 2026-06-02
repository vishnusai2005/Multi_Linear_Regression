# Multi_Linear_Regression
# 🏥 Medical Insurance Premium Prediction
### Multiple Linear Regression | EDA | Log Transformation | Feature Importance
## 📌 Problem Statement

Medical insurance premiums vary significantly across individuals based on personal and lifestyle attributes. The goal of this project is to **predict the annual medical insurance charges** for an individual using demographic and lifestyle features via **Multiple Linear Regression (MLR)**.

This is a supervised regression problem where the target variable is `charges` — the dollar amount billed to the policyholder.

---

## 📁 Project Structure

```
medical-insurance-premium/
├── Medical_insurence.ipynb   # Main notebook (EDA + Modeling)
├── insurance.csv             # Dataset (1338 rows × 7 columns)
└── README.md
```

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| **Source** | IBM Watson / Kaggle (Synthetic but realistic) |
| **Rows** | 1,338 |
| **Features** | 6 (5 input + 1 target) |
| **Missing Values** | 0 |
| **Duplicates** | 1 |

### Feature Descriptions

| Feature | Type | Description | Range / Values |
|---|---|---|---|
| `age` | Numerical | Age of the primary beneficiary | 18 – 64 years |
| `sex` | Categorical | Policyholder's gender | male, female |
| `bmi` | Numerical | Body Mass Index (kg/m²) | 15.96 – 53.13 |
| `children` | Numerical | Number of dependents covered | 0 – 5 |
| `smoker` | Categorical | Whether the individual smokes | yes, no |
| `region` | Categorical | Residential area in the US | northeast, northwest, southeast, southwest |
| `charges` *(target)* | Numerical | Medical insurance billed annually (USD) | $1,122 – $63,770 |

### Descriptive Statistics

| Stat | age | bmi | children | charges |
|---|---|---|---|---|
| **Mean** | 39.21 | 30.66 | 1.09 | $13,270 |
| **Std Dev** | 14.05 | 6.10 | 1.21 | $12,110 |
| **Min** | 18 | 15.96 | 0 | $1,122 |
| **Median (50%)** | 39 | 30.40 | 1 | $9,382 |
| **Max** | 64 | 53.13 | 5 | $63,770 |

> 📌 High standard deviation in `charges` ($12,110) relative to its mean ($13,270) signals strong **right-skewness** — a key insight driving the log-transformation in the improved model.

---

## 🔍 Exploratory Data Analysis (EDA)

### 1. Numerical Feature Distributions

- **`age`**: Roughly uniform distribution across 18–64. No obvious outliers.
- **`bmi`**: Near-normal distribution centered around 30.66. A few outliers on the high end (>50).
- **`charges`**: Heavily **right-skewed** with a long tail toward high values. Most policies cluster under $20,000, but a subset exceeds $40,000–$63,770. This skewness violates the normality assumption of OLS regression.

### 2. Categorical Feature Distributions

| Feature | Distribution |
|---|---|
| `sex` | Balanced — ~50% male, ~50% female |
| `smoker` | Imbalanced — **79.5% non-smokers (1064)** vs 20.5% smokers (274) |
| `region` | Nearly uniform — SE: 364, SW: 325, NW: 325, NE: 324 |
| `children` | Majority have 0–2 children; 4–5 children are rare |

### 3. Key Bivariate Insights

#### 🚬 Smoking is the Single Biggest Cost Driver

| Smoker Status | Avg. Annual Charges |
|---|---|
| **No** | $8,434 |
| **Yes** | $32,050 |

> Smokers pay **~3.8× more** in insurance premiums than non-smokers on average.
> Scatter plots of age vs. charges and BMI vs. charges clearly revealed two distinct clusters — smokers forming a high-cost band, non-smokers forming a low-cost band.

#### 📅 Age Has a Positive Linear Relationship with Charges
Charges generally increase with age, but the relationship is **not perfectly linear** — there are three visible clusters in the scatter plot, which are explained by the smoking status overlay.

#### ⚖️ BMI Alone Has Low Impact, But High BMI + Smoking Is Explosive
BMI has a modest correlation with charges (r = 0.20) overall, but when combined with smoker status, obese smokers have disproportionately high charges.

#### 🌍 Regional Differences Are Minor

| Region | Avg. Charges |
|---|---|
| Southeast | $14,735 |
| Northeast | $13,406 |
| Northwest | $12,418 |
| Southwest | $12,347 |

Southeast has the highest average charges — partly explained by a higher BMI distribution in that region.

---

## ⚙️ Data Preprocessing

### One-Hot Encoding (with `drop_first=True`)

Categorical variables `sex`, `smoker`, and `region` were encoded using `pd.get_dummies()`.

| Original Column | Encoded Columns |
|---|---|
| `sex` | `sex_male` (female is dropped as reference) |
| `smoker` | `smoker_yes` (no is dropped as reference) |
| `region` | `region_northwest`, `region_southeast`, `region_southwest` (northeast dropped) |

All dummy columns were cast to `int` to ensure compatibility with sklearn.

### Final Feature Set (after encoding)

```
age | bmi | children | sex_male | smoker_yes | region_northwest | region_southeast | region_southwest
```

### Train-Test Split

```python
test_size = 0.2  → 268 test samples
train_size = 0.8 → 1070 train samples
random_state = 42
```

---

## 🤖 Model 1: Baseline Multiple Linear Regression

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
```

The model fits a hyperplane: **ŷ = β₀ + β₁·age + β₂·bmi + β₃·children + ... + βₙ·xₙ**

### Baseline Performance (raw `charges`)

| Metric | Value |
|---|---|
| **R² Score** | 0.7838 |
| **MSE** | $33,566,439 |
| **RMSE** | **$5,793.66** |
| **MAE** | **$4,176.27** |

> R² of 0.7838 means the model explains ~78% of the variance in insurance charges. The high RMSE ($5,793) indicates predictions can deviate significantly on high-charge individuals.

---

## 🚀 Model 2: Improved Model with Log-Transformed Target

### Why Log Transformation?

`charges` is **right-skewed**. OLS regression assumptions require:
- **Normality of residuals** — violated with raw charges
- **Homoscedasticity** (constant error variance) — variance increases with charge magnitude

Applying `np.log(charges)` compresses the heavy tail, making the distribution more Gaussian and stabilizing residual variance.

```python
encoded_df['charges'] = np.log(encoded_df['charges'])
```

After transformation, the histogram of `log(charges)` shows a **near-normal bell curve** — confirming the transformation was appropriate.

### Improved Model Performance

| Metric | Log Scale | Back-Transformed |
|---|---|---|
| **R² Score** | **0.8050** | 0.6083 |
| **RMSE** | 0.4188 (log units) | $7,797 |
| **MAE** | 0.2696 (log units) | $3,884 |

> **R² improved from 0.7838 → 0.8050** in log space — meaning the model better captures the underlying relationships once the scale effect is removed. The MAE of $3,884 (back-transformed) is meaningful — most predictions are within ~$3,900 of actual charges.

---

## 📈 Feature Importance: Regression Coefficients (Log Model)

In the log-transformed model, coefficients represent **percentage change in charges** per unit increase in a feature.

| Feature | Coefficient | Interpretation |
|---|---|---|
| `smoker_yes` | **+1.5520** | Smoking multiplies expected charges by **e¹·⁵⁵ ≈ 4.7×** — far the most dominant driver |
| `age` | **+0.0343** | Each additional year of age increases charges by ~**3.5%** |
| `children` | **+0.0926** | Each additional dependent increases charges by ~**9.7%** |
| `bmi` | **+0.0136** | Each BMI unit increase raises charges by ~**1.4%** |
| `sex_male` | **−0.0743** | Males predicted slightly lower charges (−7.2%) than females in this model |
| `region_southeast` | **−0.1367** | Southeast is ~12.8% cheaper than Northeast (reference) |
| `region_southwest` | **−0.1231** | Southwest is ~11.6% cheaper than Northeast |
| `region_northwest` | **−0.0562** | Northwest is ~5.5% cheaper than Northeast |
| `intercept` | **7.0527** | Baseline log-charge when all features are 0 (e^7.05 ≈ $1,152) |

### Correlation with Target (Encoded Features)

| Feature | Corr. with `charges` | Corr. with `log(charges)` |
|---|---|---|
| `smoker_yes` | **0.787** | **0.665** |
| `age` | 0.299 | **0.528** |
| `bmi` | 0.196 | 0.131 |
| `children` | 0.068 | 0.161 |
| `sex_male` | 0.057 | 0.006 |
| `region_*` | < 0.08 | < 0.02 |

> Notice that `age`'s correlation with the target **jumps from 0.30 → 0.53** after log-transformation — confirming that log space better captures the multiplicative aging effect on insurance costs.

---

## 🧠 Key Findings Summary

1. **Smoking is overwhelmingly the most important predictor.** A smoker's predicted premium is ~4.7× that of a non-smoker after all other factors are controlled. This single feature has a correlation of 0.787 with raw charges.

2. **Age has a multiplicative, not purely additive, effect on charges.** The improved correlation (0.30 → 0.53) in log space confirms premiums compound with age rather than growing linearly.

3. **BMI's individual contribution is moderate**, but its interaction with smoking (not explicitly modeled here) is known to be extreme — obese smokers are among the highest-cost individuals in the dataset.

4. **Geographic region has minimal predictive power.** All region dummies have near-zero correlations with log(charges), suggesting regional pricing differences are marginal compared to individual health behaviors.

5. **Gender has negligible impact.** The `sex_male` coefficient is small and likely not statistically significant — consistent with insurance regulations in the US that restrict sex-based premium pricing.

6. **Log-transformation is essential** — it reduces MSE in log space from an incoherent raw-scale metric to a meaningful 0.4188 RMSE (predicting within ~42% of the log charge), and raises R² by 2+ percentage points.

---

## 🔧 Technology Stack

| Tool | Purpose |
|---|---|
| `Python 3.8+` | Core language |
| `pandas` | Data loading, manipulation, encoding |
| `numpy` | Numerical operations, log transformation |
| `matplotlib` | Base plotting framework |
| `seaborn` | Statistical visualization (histplots, boxplots, scatterplots, countplots) |
| `sklearn.linear_model.LinearRegression` | MLR model |
| `sklearn.model_selection.train_test_split` | Data splitting |
| `sklearn.metrics` | MSE, R² evaluation |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medical-insurance-premium.git
cd medical-insurance-premium
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook Medical_insurence.ipynb
```

---

## 🔮 Limitations & Future Improvements

| Limitation | Suggested Fix |
|---|---|
| Linear model may miss interaction terms (e.g., BMI × smoker) | Add `PolynomialFeatures` or explicit interaction columns |
| LOFO feature importance incomplete | Implementing Leave-One-Feature-Out loop to measure ΔR² per feature |
| No cross-validation | Adding K-Fold CV for more robust R² estimates |
| Potential heteroscedasticity remains | Trying Ridge/Lasso regularization or Quantile Regression |
| Outlier charges (>$50,000) may distort predictions | Applying IQR-based outlier capping before modeling |
| Region has near-zero impact | Consider dropping region to simplify the model |
| Tree-based models not explored | Try Decision Tree, Random Forest, XGBoost for non-linear capture |

