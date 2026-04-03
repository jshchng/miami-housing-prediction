# tuning/tune_lm.R
# Linear regression — no hyperparameters to tune, so we use fit_resamples().
# Evaluated against all four recipes as a baseline comparison.
# Input:  data/processed/miami_split.rds
# Output: results/tuning/lm_basic.rds, lm_yj.rds, lm_eda.rds, lm_interactions.rds

library(tidyverse)
library(tidymodels)

source("R/03_recipes.R")
source("R/04_helpers.R")

splits <- read_rds("data/processed/miami_split.rds")
folds  <- splits$folds

lm_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Fit across all four recipes
lm_basic <- workflow() %>%
  add_recipe(recipe_basic) %>%
  add_model(lm_spec) %>%
  fit_resamples(folds, metrics = model_metrics, control = tune_control)

lm_yj <- workflow() %>%
  add_recipe(recipe_yj) %>%
  add_model(lm_spec) %>%
  fit_resamples(folds, metrics = model_metrics, control = tune_control)

lm_eda <- workflow() %>%
  add_recipe(recipe_eda) %>%
  add_model(lm_spec) %>%
  fit_resamples(folds, metrics = model_metrics, control = tune_control)

lm_interactions <- workflow() %>%
  add_recipe(recipe_interactions) %>%
  add_model(lm_spec) %>%
  fit_resamples(folds, metrics = model_metrics, control = tune_control)

save_tuning_result(lm_basic,        "lm_basic")
save_tuning_result(lm_yj,           "lm_yj")
save_tuning_result(lm_eda,          "lm_eda")
save_tuning_result(lm_interactions, "lm_interactions")

# Quick summary
bind_rows(
  collect_metrics(lm_basic)        %>% mutate(recipe = "basic"),
  collect_metrics(lm_yj)           %>% mutate(recipe = "yeo-johnson"),
  collect_metrics(lm_eda)          %>% mutate(recipe = "eda"),
  collect_metrics(lm_interactions) %>% mutate(recipe = "interactions")
) %>%
  filter(.metric == "rmse") %>%
  select(recipe, mean, std_err) %>%
  arrange(mean) %>%
  print()