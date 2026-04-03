# R/03_recipes.R
# Defines all four preprocessing recipes used across tuning scripts.
# Source this file; do not run it directly.

library(tidyverse)
library(tidymodels)

splits <- read_rds("data/processed/miami_split.rds")
miami_train <- splits$train

# ── Base formula ───────────────────────────────────────────────────────────────
base_formula <- sale_prc_log ~ .

# ── Recipe 1: Basic ────────────────────────────────────────────────────────────
recipe_basic <- recipe(base_formula, data = miami_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ── Recipe 2: Yeo-Johnson ──────────────────────────────────────────────────────
# Automatically finds the optimal power transformation for each numeric predictor
recipe_yj <- recipe(base_formula, data = miami_train) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ── Recipe 3: EDA-driven transformations ───────────────────────────────────────
# Manually specified based on relationships observed during EDA:
#   - log(distance to highway): strong right skew
#   - sqrt(special features value): moderate right skew
#   - longitude^2: U-shaped relationship with price
#   - natural splines for latitude and ocean distance: nonlinear curves
recipe_eda <- recipe(base_formula, data = miami_train) %>%
  step_log(hwy_dist, offset = 1) %>%
  step_sqrt(spec_feat_val) %>%
  step_mutate(longitude_sq = longitude^2) %>%
  step_ns(latitude, deg_free = 4) %>%
  step_ns(ocean_dist, deg_free = 4) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ── Recipe 4: Interactions + PCA ──────────────────────────────────────────────
# Builds on Yeo-Johnson, adds all pairwise interactions, then reduces
# dimensionality with PCA to prevent multicollinearity and overfitting
recipe_interactions <- recipe(base_formula, data = miami_train) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_interact(terms = ~ all_predictors():all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold = 0.95)

message("✓ All four recipes defined and ready.")