# tuning/tune_mars.R
# Multivariate adaptive regression splines — tunes the number of terms and
# degree of interaction. Automatically detects nonlinear relationships and
# breakpoints without manual feature engineering.
# Input:  data/processed/miami_split.rds
# Output: results/tuning/mars_basic.rds, mars_yj.rds, mars_eda.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

mars_spec <- mars(
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth") %>%
  set_mode("regression")

mars_grid <- grid_regular(
  num_terms(range   = c(5, 50)),
  prod_degree(range = c(1, 3)),
  levels = 10
)

tune_mars <- function(recipe, name) {
  result <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(mars_spec) %>%
    tune_grid(
      resamples = folds,
      grid      = mars_grid,
      metrics   = model_metrics,
      control   = grid_control
    )
  save_tuning_result(result, name)
  result
}

# Interactions recipe excluded — MARS finds its own interactions internally
# via prod_degree, making the PCA-compressed recipe redundant
mars_basic <- tune_mars(recipe_basic, "mars_basic")
mars_yj    <- tune_mars(recipe_yj,   "mars_yj")
mars_eda   <- tune_mars(recipe_eda,  "mars_eda")

# Best RMSE per recipe
list(
  basic        = mars_basic,
  `yeo-johnson` = mars_yj,
  eda          = mars_eda
) %>%
  map_dfr(~ show_best(.x, metric = "rmse", n = 1), .id = "recipe") %>%
  select(recipe, mean, std_err, num_terms, prod_degree) %>%
  arrange(mean) %>%
  print()