library(data.table)
library(ggplot2)
library(dplyr)
library(purrr)

dataset <- readRDS("icml_other_datasets/openml__APSFailure__168868.rds")
dataset <- dataset %>% select(-target)
M <- as.data.table(is.na(dataset))
M_patterns_only <- apply(M, 1, function(x) paste(as.integer(x), collapse = ""))

pattern_counts <- table(M_patterns_only)


df_counts <- as.data.frame(pattern_counts)
colnames(df_counts) <- c("pattern", "group_size")
df_counts$group_size <- as.numeric(df_counts$group_size)

N_total <- sum(df_counts$group_size)

size_dist <- df_counts %>%
  group_by(group_size) %>%
  summarise(rows_in_this_size = sum(group_size)) %>%
  arrange(desc(group_size))

size_dist <- size_dist %>%
  mutate(
    k = group_size -1,
    # Cumulative sum of rows from largest groups down to smallest
    cumulative_rows = cumsum(rows_in_this_size),
    proportion = cumulative_rows / N_total
  )

ggplot(size_dist, aes(x = k, y = proportion)) +
  geom_step(size = 1, color = "firebrick") +
  # Using log scale for x because k usually varies from 0 to thousands
  scale_x_log10(labels = scales::comma) + 
  theme_minimal() +
  labs(
    title = "Missingness Pattern Redundancy",
    x = "Number of other rows sharing the pattern (k)",
    y = "Proportion of rows",
    subtitle = "Higher values indicate more redundant/repetitive missing data"
  )






# Assuming you have already run thes code to generate 'size_dist' and 'N_total'
# from your previous snippet...

# 1. Define the coverage thresholds we are looking for
thresholds <- seq(1, 0.45, by = -0.05)
thresholds <- c(1, 0.95, 0.9275,0.90, 0.85, 0.80, 0.75, 0.7,0.65, 0.6)

# 2. Extract the required k for each threshold
# Note: we find the smallest k in your size_dist where the proportion is AT LEAST the threshold.
# Since size_dist is sorted by descending group_size (largest groups first), 
# the first record that hits the threshold represents the boundary k.

coverage_table <- map_df(thresholds, function(t) {
  # Find the first entry where proportion >= threshold
  match <- size_dist %>%
    filter(proportion >= t) %>%
    slice(1)
  
  # If no match found, k is effectively 0 (no patterns large enough)
  k_val <- ifelse(nrow(match) > 0, match$k, 0)
  
  data.frame(
    Coverage_Threshold = t,
    Required_k = k_val
  )
})

# 3. Print the resulting table
print(coverage_table)
