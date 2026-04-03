# tuning/tune_rf.R
# Random forest — tunes mtry (predictors per split) and min_n (minimum node size).
# Trees is set high and fixed; the ensemble handles variance without tuning it.
# Input:  data/processed/miami_split.rds
# Output: results/tuning/rf_basic.rds, rf_yj.rds, rf_eda.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

rf_grid <- grid_regular(
  mtry(range  = c(2, 15)),
  min_n(range = c(2, 20)),
  levels = 6
)

tune_rf <- function(recipe, name) {
  result <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(rf_spec) %>%
    tune_grid(
      resamples = folds,
      grid      = rf_grid,
      metrics   = model_metrics,
      control   = grid_control
    )
  save_tuning_result(result, name)
  result
}

# Interactions recipe excluded — PCA destroys the tree splitting interpretability
# and ranger cannot handle the high column count efficiently
rf_basic <- tune_rf(recipe_basic, "rf_basic")
rf_yj    <- tune_rf(recipe_yj,   "rf_yj")
rf_eda   <- tune_rf(recipe_eda,  "rf_eda")

# Best RMSE per recipe
list(
  basic        = rf_basic,
  `yeo-johnson` = rf_yj,
  eda          = rf_eda
) %>%
  map_dfr(~ show_best(.x, metric = "rmse", n = 1), .id = "recipe") %>%
  select(recipe, mean, std_err, mtry, min_n) %>%
  arrange(mean) %>%
  print()