# tuning/tune_knn.R
# K-nearest neighbors — tunes the number of neighbors.
# Nonparametric: makes no assumptions about functional form.
# Input:  data/processed/miami_split.rds
# Output: results/tuning/knn_basic.rds, knn_yj.rds, knn_eda.rds, knn_interactions.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

knn_spec <- nearest_neighbor(
  neighbors = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("regression")

knn_grid <- grid_regular(
  neighbors(range = c(1, 50)),
  levels = 15
)

tune_knn <- function(recipe, name) {
  result <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(knn_spec) %>%
    tune_grid(
      resamples = folds,
      grid      = knn_grid,
      metrics   = model_metrics,
      control   = grid_control
    )
  save_tuning_result(result, name)
  result
}

knn_basic        <- tune_knn(recipe_basic,        "knn_basic")
knn_yj           <- tune_knn(recipe_yj,           "knn_yj")
knn_eda          <- tune_knn(recipe_eda,          "knn_eda")
knn_interactions <- tune_knn(recipe_interactions, "knn_interactions")

# Best RMSE per recipe
list(
  basic        = knn_basic,
  `yeo-johnson` = knn_yj,
  eda          = knn_eda,
  interactions = knn_interactions
) %>%
  map_dfr(~ show_best(.x, metric = "rmse", n = 1), .id = "recipe") %>%
  select(recipe, mean, std_err, neighbors) %>%
  arrange(mean) %>%
  print()