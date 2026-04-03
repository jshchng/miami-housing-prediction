# R/02_split_and_folds.R
# Input:  data/processed/miami_clean.rds
# Output: data/processed/splits.rds

library(tidyverse)
library(tidymodels)

set.seed(123)

# Load cleaned data
miami <- read_rds("data/processed/miami_clean.rds")

# 80/20 train-test split, stratified on log sale price
miami_split <- initial_split(miami, prop = 0.80, strata = sale_prc_log)
miami_train <- training(miami_split)
miami_test  <- testing(miami_split)

# 5-fold CV with 3 repeats, stratified — used for all tuning
miami_folds <- vfold_cv(miami_train, v = 5, repeats = 3, strata = sale_prc_log)

write_rds(
  list(split = miami_split, train = miami_train, test = miami_test, folds = miami_folds),
  "data/processed/miami_split.rds"
)

message("✓ Split and folds written to data/processed/miami_split.rds")
message("  Training rows: ", nrow(miami_train))
message("  Testing rows:  ", nrow(miami_test))