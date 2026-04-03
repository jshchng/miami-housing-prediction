<div align="center">
  <h1>Miami Housing Price Prediction</h1>
  <p>
    <img src="https://img.shields.io/badge/R-4.2+-276DC3?logo=r&logoColor=white" />
    <img src="https://img.shields.io/badge/tidymodels-1.0-4B9CD3" />
    <img src="https://img.shields.io/badge/XGBoost-Final%20Model-FF6600" />
    <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
  </p>
  <p align="center"><a href="https://jshchng.github.io/miami-housing-prediction/">📊 View Full Analysis Reports</a></p>
</div>

Miami homeowners receiving offers have no fast, affordable way to know if the price is fair. This project builds a machine learning pipeline that predicts residential sale prices from observable property features, trained on 13,932 Miami transactions, tuned across 8 model families, and validated on a held-out test set. The final XGBoost model predicts within $42,294 at the median and explains 93.6% of variance in sale price.

---

## Key Results

| Metric | Value |
|---|---|
| Test RMSE (log scale) | 0.1415 |
| Test R-squared | 0.9363 |
| Test RMSE (dollars) | $89,864 |
| Test MAE (dollars) | $42,294 |
| Improvement over null model | 75% reduction in RMSE |

The null model (predicting the mean sale price for every property) produces an RMSE of 0.567 on the log scale. The final XGBoost model reduces this to 0.1415, a 75% improvement.

---

## Motivation

When a homeowner receives an offer, they face a hard question: is this fair? Professional appraisals are slow and expensive. This project builds a model that estimates Miami residential sale prices from observable property features, giving homeowners a fast, data-driven reference point for evaluating offers.

---

## Tech Stack

`R` · `tidymodels` · `XGBoost` · `ranger` · `glmnet` · `kernlab` · `earth` · `Quarto` · `ggplot2` · `doParallel`

---

## Repository Structure
```
miami-housing-prediction/
├── R/
│   ├── 01_cleaning.R             # Raw data cleaning, type corrections, log transform
│   ├── 02_split_and_folds.R      # 80/20 train-test split, 5-fold CV with 3 repeats
│   ├── 03_recipes.R              # Four preprocessing recipes (Basic, YJ, EDA, Interactions)
│   └── 04_helpers.R              # Shared metrics, parallel backend, save utilities
├── tuning/
│   ├── tune_lm.R                 # Linear regression baseline
│   ├── tune_en.R                 # Elastic net (penalty + mixture)
│   ├── tune_lasso.R              # Lasso regression
│   ├── tune_knn.R                # K-nearest neighbors
│   ├── tune_rf.R                 # Random forest (mtry + min_n)
│   ├── tune_xgboost.R            # XGBoost (trees, depth, learn rate, regularization)
│   ├── tune_svmrbf.R             # SVM with radial basis function kernel
│   └── tune_mars.R               # Multivariate adaptive regression splines
├── reports/
│   ├── 01_data_cleaning.qmd      # Cleaning decisions, type corrections, missingness
│   ├── 02_eda.qmd                # Exploratory analysis motivating feature engineering
│   ├── 03_model_results.qmd      # Full model comparison, test evaluation, residuals, VIP
│   └── 04_executive_summary.qmd  # Non-technical summary of findings
├── docs/                         # Rendered HTML reports served via GitHub Pages
├── data/
│   ├── raw/                      # Original CSV (gitignored)
│   └── processed/                # Cleaned RDS files (gitignored)
├── results/
│   └── tuning/                   # Tuning result RDS files (gitignored)
└── _quarto.yml                   # Quarto project config
```

---

## Methodology

### Exploratory Data Analysis
Rather than jumping straight to modeling, I spent time understanding the shape of each predictor relationship with log sale price. This revealed that latitude and ocean distance have nonlinear curves that cannot be linearized, longitude follows a U-shape, and highway distance and special features value are heavily right-skewed. These specific findings, not a generic automated pipeline, drove the feature engineering decisions that ultimately produced the winning model.

### Feature Engineering
Four distinct preprocessing strategies were built and systematically evaluated. The key insight: manually crafted EDA-driven transformations outperformed automated alternatives (Yeo-Johnson, PCA interactions) across every flexible model family. This demonstrates that domain judgment adds value even when powerful models like XGBoost are available, a finding that runs counter to the "just throw it at the model" approach common in practice.

### Cross-Validation Strategy
Models were evaluated using 5-fold cross-validation with 3 repeats, stratified on the outcome variable, giving 15 estimates of generalization performance per candidate model. The test set was held out completely until a single final evaluation, preventing any form of data leakage or optimistic bias in the reported results.

### Model Selection
Eight model families were compared across four recipes, producing a comprehensive sweep of parametric, nonparametric, tree-based, kernel, and spline-based approaches. The clear finding: nonlinear tree-based models substantially outperformed linear methods, and the performance gap between XGBoost (0.144) and the best linear model (0.217) confirms the data has complex structure that coefficients cannot capture.

---

## Model Comparison

| Model | Best Recipe | CV RMSE |
|---|---|---|
| **XGBoost** | **EDA Transformations** | **0.1440** |
| Random Forest | EDA Transformations | 0.1540 |
| SVM-RBF | EDA Transformations | 0.1590 |
| MARS | EDA Transformations | 0.1640 |
| Linear Regression | Interactions + PCA | 0.2170 |
| Elastic Net | Interactions + PCA | 0.2170 |
| Lasso | Interactions + PCA | 0.2170 |
| KNN | EDA Transformations | 0.2330 |

---

## Skills Demonstrated

| Area | Tools and Techniques |
|---|---|
| Machine Learning | XGBoost, Random Forest, SVM, MARS, KNN, Elastic Net |
| Feature Engineering | Power transforms, splines, PCA, interaction terms |
| Validation | k-fold CV, stratified splitting, train/test discipline |
| Programming | R, tidymodels, tidyverse, ggplot2, Quarto |
| Engineering | Parallel computing, modular script design, reproducible pipeline |
| Communication | Executive summary, residual analysis, variable importance |

---

## What I Would Do Next

- [ ]  **More recent data:** retrain on post-2020 sales to capture current market conditions and the pandemic-era price surge in Miami
- [ ]  **Ensemble stacking:** combine XGBoost and random forest predictions using a meta-learner, which often produces another 5-10% improvement
- [ ]  **Shiny app:** deploy an interactive prediction tool where a homeowner enters their property details and gets an instant price estimate with a confidence interval
- [ ]  **Neighborhood features:** enrich the dataset with school ratings, walkability scores, and crime statistics to improve predictions in areas where the current features are weakest
- [ ]  **Residual analysis by geography:** map prediction errors spatially to identify neighborhoods where the model systematically over or underpredicts

---

## Reproducing the Analysis

### Requirements
- R 4.2+
- Quarto

### Package Installation
```r
install.packages(c(
  "tidyverse", "tidymodels", "janitor", "naniar", "skimr",
  "doParallel", "ranger", "xgboost", "kernlab", "kknn", "earth",
  "glmnet", "gt", "vip", "patchwork"
))
```

### Pipeline
```r
# 1. Clean data
source("R/01_cleaning.R")

# 2. Create splits and folds
source("R/02_split_and_folds.R")

# 3. Tune models (computationally intensive)
source("tuning/tune_lm.R")
source("tuning/tune_en.R")
source("tuning/tune_lasso.R")
source("tuning/tune_knn.R")
source("tuning/tune_rf.R")
source("tuning/tune_xgboost.R")
source("tuning/tune_svmrbf.R")
source("tuning/tune_mars.R")

# 4. Render reports
# quarto render reports/01_data_cleaning.qmd
# quarto render reports/02_eda.qmd
# quarto render reports/03_model_results.qmd
# quarto render reports/04_executive_summary.qmd
```

### Data
The raw data file `miami-housing.csv` is not included due to file size. The dataset is publicly available from the UCI Machine Learning Repository.

---

## License
MIT