# R/04_helpers.R
# Shared utilities sourced by tuning scripts.
# Provides: metric set, control settings, and a save helper.

library(tidymodels)
library(doParallel)
library(readr)

# Metric used consistently across all model evaluations
model_metrics <- metric_set(rmse, rsq, mae)

# Parallel backend — uses all available cores minus one to keep system responsive
registerDoParallel(cores = parallel::detectCores() - 1)

# Tuning control: save predictions for residual analysis, enable parallelism
tune_control <- control_resamples(
  save_pred    = TRUE,
  parallel_over = "everything"
)

grid_control <- control_grid(
  save_pred     = TRUE,
  parallel_over = "everything",
  verbose       = TRUE
)

# Saves a tuning result with a consistent naming convention
save_tuning_result <- function(result, name) {
  path <- paste0("results/tuning/", name, ".rds")
  write_rds(result, path)
  message("✓ Saved: ", path)
}