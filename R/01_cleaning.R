# R/01_cleaning.R
# Input:  data/raw/miami-housing.csv
# Output: data/processed/miami_clean.rds

library(tidyverse)
library(janitor)
library(naniar)
library(skimr)

# Load and standardize column names
miami_raw <- read_csv("data/raw/miami-housing.csv")

miami <- miami_raw %>%
  clean_names() %>%
  rename(plane_noise = avno60plus) %>%
  select(-parcelno)

# Convert numeric-coded categoricals to factors
miami <- miami %>%
  mutate(
    plane_noise       = factor(plane_noise),
    month_sold        = factor(month_sold),
    structure_quality = factor(structure_quality)
  )

# Confirm no missing values — no imputation needed
miss_var_summary(miami)

# sale_prc is right-skewed; log transformation normalizes the distribution
ggplot(miami, aes(x = sale_prc)) +
  geom_freqpoly(bins = 60) +
  labs(title = "Sale price — original scale", x = "Sale price (USD)", y = "Count")

ggplot(miami, aes(x = log(sale_prc))) +
  geom_freqpoly(bins = 60) +
  labs(title = "Sale price — log scale", x = "log(Sale price)", y = "Count")

miami <- miami %>%
  mutate(sale_prc_log = log(sale_prc)) %>%
  select(-sale_prc)

# Final check: expect 13,932 rows, 16 cols, 3 factors, 0 missing
skim(miami)

write_rds(miami, "data/processed/miami_clean.rds")
message("✓ miami_clean.rds written to data/processed/")