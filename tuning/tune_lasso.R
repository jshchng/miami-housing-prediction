# tuning/tune_lasso.R
# Lasso — elastic net with mixture fixed at 1 (pure L1 regularization).
# Useful for feature selection: drives irrelevant coefficients to exactly zero.
# Input:  data/processed/miami_split.rds
# Output: results/tuning/lasso_basic.rds, lasso_yj.rds, lasso_eda.rds, lasso_interactions.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

lasso_spec <- linear_reg(
  penalty = tune(),
  mixture = 1
) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

lasso_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  levels = 20
)

tune_lasso <- function(recipe, name) {
  result <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(lasso_spec) %>%
    tune_grid(
      resamples = folds,
      grid      = lasso_grid,
      metrics   = model_metrics,
      control   = grid_control
    )
  save_tuning_result(result, name)
  result
}

lasso_basic        <- tune_lasso(recipe_basic,        "lasso_basic")
lasso_yj           <- tune_lasso(recipe_yj,           "lasso_yj")
lasso_eda          <- tune_lasso(recipe_eda,          "lasso_eda")
lasso_interactions <- tune_lasso(recipe_interactions, "lasso_interactions")

# Best RMSE per recipe
list(
  basic        = lasso_basic,
  `yeo-johnson` = lasso_yj,
  eda          = lasso_eda,
  interactions = lasso_interactions
) %>%
  map_dfr(~ show_best(.x, metric = "rmse", n = 1), .id = "recipe") %>%
  select(recipe, mean, std_err, penalty) %>%
  arrange(mean) %>%
  print()