# tuning/tune_en.R
# Elastic net — tunes penalty (regularization strength) and mixture (L1/L2 ratio).
# mixture = 1 is pure lasso, mixture = 0 is pure ridge.
# Input:  data/processed/miami_split.rds
# Output: results/tuning/en_basic.rds, en_yj.rds, en_eda.rds, en_interactions.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

en_spec <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

en_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(range = c(0, 1)),
  levels = 10
)

tune_en <- function(recipe, name) {
  result <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(en_spec) %>%
    tune_grid(
      resamples = folds,
      grid      = en_grid,
      metrics   = model_metrics,
      control   = grid_control
    )
  save_tuning_result(result, name)
  result
}

en_basic        <- tune_en(recipe_basic,        "en_basic")
en_yj           <- tune_en(recipe_yj,           "en_yj")
en_eda          <- tune_en(recipe_eda,          "en_eda")
en_interactions <- tune_en(recipe_interactions, "en_interactions")

# Best RMSE per recipe
list(
  basic        = en_basic,
  `yeo-johnson` = en_yj,
  eda          = en_eda,
  interactions = en_interactions
) %>%
  map_dfr(~ show_best(.x, metric = "rmse", n = 1), .id = "recipe") %>%
  select(recipe, mean, std_err, penalty, mixture) %>%
  arrange(mean) %>%
  print()