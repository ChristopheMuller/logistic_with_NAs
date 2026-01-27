library(dplyr)
library(tidyr)

# --- 1. Prepare Data ---

final_df <- read.csv("real_datasets_results/real_datasets_metrics_detailed.csv")

#in lower case, and only two first letters for dataset names
final_df$Dataset <- tolower(final_df$Dataset)
final_df$Dataset <- substr(final_df$Dataset, 1, 3)

# Filter for AUC metric only
auc_df <- final_df %>%
  filter(Metric == "AUC") %>%
  # Ensure values are numeric
  mutate(Value = as.numeric(Value))

# Calculate Average AUC per Method per Dataset
auc_means <- auc_df %>%
  group_by(Dataset, Method) %>%
  summarise(Mean_AUC = mean(Value, na.rm = TRUE), .groups = "drop")

# --- 2. Statistical Testing (Paired t-test) ---

# Function to check significance against the best method
check_significance <- function(curr_dataset, summary_data, full_data) {
  
  # 1. Identify the Best Method (Highest Mean AUC) for this dataset
  best_method <- summary_data %>%
    filter(Dataset == curr_dataset) %>%
    slice_max(Mean_AUC, n = 1, with_ties = FALSE) %>%
    pull(Method)
  
  # 2. Get the fold-wise results for the best method
  best_folds <- full_data %>%
    filter(Dataset == curr_dataset, Method == best_method) %>%
    arrange(Fold) %>%
    select(Fold, Best_Value = Value)
  
  # 3. Compare every method to the best method
  dataset_methods <- unique(summary_data$Method[summary_data$Dataset == curr_dataset])
  
  significance_results <- list()
  
  for (m in dataset_methods) {
    # If it's the best method, it's automatically "not worse"
    if (m == best_method) {
      significance_results[[m]] <- TRUE
      next
    }
    
    # Get fold results for the current method
    curr_folds <- full_data %>%
      filter(Dataset == curr_dataset, Method == m) %>%
      arrange(Fold) %>%
      select(Fold, Curr_Value = Value)
    
    # Join to ensure paired folds align (handle missing folds if any)
    paired_data <- inner_join(best_folds, curr_folds, by = "Fold")
    
    # Need at least 2 pairs for a t-test
    if (nrow(paired_data) < 2) {
      # Fallback: if not enough data, don't bold unless it's strictly equal
      significance_results[[m]] <- FALSE 
    } else {
      # Paired t-test
      # H0: Difference is 0 (Methods are equal)
      # Ha: Best > Current (Current is worse)
      # If p < 0.05, we reject H0 -> Best is significantly better -> Current is significantly worse (No Bold)
      # If p >= 0.05, we fail to reject -> Current is comparable (Bold)
      tryCatch({
        t_res <- t.test(paired_data$Best_Value, paired_data$Curr_Value, 
                        paired = TRUE, alternative = "greater")
        significance_results[[m]] <- (t_res$p.value >= 0.05)
      }, error = function(e) {
        # If t-test fails (e.g. constant data), fallback to FALSE
        significance_results[[m]] <- FALSE
      })
    }
  }
  
  # Return data frame of flags
  data.frame(
    Dataset = curr_dataset,
    Method = names(significance_results),
    Is_Bold = unlist(significance_results),
    row.names = NULL
  )
}

# Apply testing across all datasets
bold_flags <- lapply(unique(auc_means$Dataset), function(d) {
  check_significance(d, auc_means, auc_df)
}) %>% bind_rows()

# --- 3. Construct the Table ---

# Merge means with bold flags
final_table_data <- auc_means %>%
  left_join(bold_flags, by = c("Dataset", "Method")) %>%
  mutate(
    # Format number to 2 decimal places
    Display_Str = sprintf("%.2f", Mean_AUC),
    # Apply Bold Wrapper if flag is TRUE
    Display_Str = ifelse(Is_Bold, paste0("\\textbf{", Display_Str, "}"), Display_Str)
  ) %>%
  arrange(Dataset) %>%
  select(Dataset, Method, Display_Str)

# Pivot to Wide Format (One row per Method, One col per Dataset)
wide_table <- final_table_data %>%
  pivot_wider(names_from = Dataset, values_from = Display_Str)

# --- 4. View/Save ---

# Print to console
print(wide_table)

# Save as CSV (Note: CSVs don't render LaTeX bolding, but the text will contain the tags)
write.csv(wide_table, file.path(results_path, "AUC_summary_table.csv"), row.names = FALSE)

# If you want a LaTeX table printed to console
if (requireNamespace("xtable", quietly = TRUE)) {
  print(xtable::xtable(wide_table[,c(1,11:20)]), include.rownames = FALSE, sanitize.text.function = identity)
}
