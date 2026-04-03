# tuning/tune_xgboost.R
# XGBoost — tunes trees, tree depth, learning rate, and regularization.
# Most hyperparameters of any model in this project.
# Input:  data/processed/miami_split.rds
# Output: results/tuning/xgb_basic.rds, xgb_yj.rds, xgb_eda.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  min_n          = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Latin hypercube samples the parameter space more efficiently than a regular grid
# at this many dimensions
xgb_grid <- grid_latin_hypercube(
  trees(range          = c(500, 2000)),
  tree_depth(range     = c(3, 10)),
  learn_rate(range     = c(-3, -1)),
  loss_reduction(range = c(-5, 0)),
  min_n(range          = c(2, 20)),
  size = 30
)

tune_xgb <- function(recipe, name) {
  result <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(xgb_spec) %>%
    tune_grid(
      resamples = folds,
      grid      = xgb_grid,
      metrics   = model_metrics,
      control   = grid_control
    )
  save_tuning_result(result, name)
  result
}

xgb_basic <- tune_xgb(recipe_basic, "xgb_basic")
xgb_yj    <- tune_xgb(recipe_yj,   "xgb_yj")
xgb_eda   <- tune_xgb(recipe_eda,  "xgb_eda")

# Best RMSE per recipe
list(
  basic        = xgb_basic,
  `yeo-johnson` = xgb_yj,
  eda          = xgb_eda
) %>%
  map_dfr(~ show_best(.x, metric = "rmse", n = 1), .id = "recipe") %>%
  select(recipe, mean, std_err, trees, tree_depth, learn_rate, min_n) %>%
  arrange(mean) %>%
  print()