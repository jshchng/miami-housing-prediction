# tuning/tune_svmrbf.R
# Support vector machine with radial basis function kernel.
# Tunes cost (margin penalty) and rbf_sigma (kernel bandwidth).
# Input:  data/processed/miami_split.rds
# Output: results/tuning/svm_basic.rds, svm_yj.rds, svm_eda.rds, svm_interactions.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

svm_spec <- svm_rbf(
  cost      = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("regression")

svm_grid <- grid_regular(
  cost(range      = c(-2, 4)),
  rbf_sigma(range = c(-4, 0)),
  levels = 8
)

tune_svm <- function(recipe, name) {
  result <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(svm_spec) %>%
    tune_grid(
      resamples = folds,
      grid      = svm_grid,
      metrics   = model_metrics,
      control   = grid_control
    )
  save_tuning_result(result, name)
  result
}

svm_basic        <- tune_svm(recipe_basic,        "svm_basic")
svm_yj           <- tune_svm(recipe_yj,           "svm_yj")
svm_eda          <- tune_svm(recipe_eda,          "svm_eda")
svm_interactions <- tune_svm(recipe_interactions, "svm_interactions")

# Best RMSE per recipe
list(
  basic        = svm_basic,
  `yeo-johnson` = svm_yj,
  eda          = svm_eda,
  interactions = svm_interactions
) %>%
  map_dfr(~ show_best(.x, metric = "rmse", n = 1), .id = "recipe") %>%
  select(recipe, mean, std_err, cost, rbf_sigma) %>%
  arrange(mean) %>%
  print()