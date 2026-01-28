library(dplyr)
library(purrr)

analyze_rds_file <- function(file_path) {
  data <- readRDS(file_path)
  
  if (is.null(dim(data))) {
    return(data.frame(
      file = file_path,
      n_rows = length(data),
      n_cols = 1,
      n_missing = sum(is.na(data)),
      n_unique_masks = NA,
      top_1_mask_count = NA,
      top_3_mask_count = NA,
      top_5_mask_count = NA,
      top_10_mask_count = NA,
      top_15_mask_count = NA
    ))
  }
  
  n_rows <- nrow(data)
  n_cols <- ncol(data)
  n_missing <- sum(is.na(data))
  
  na_matrix <- is.na(data)
  mode(na_matrix) <- "integer"
  masks <- apply(na_matrix, 1, paste, collapse = "")
  
  mask_counts <- sort(table(masks), decreasing = TRUE)
  n_unique_masks <- length(mask_counts)
  
  get_top_k <- function(k, counts) {
    if (k <= length(counts)) {
      sum(counts[1:k])
    } else {
      sum(counts)
    }
  }
  
  k_values <- c(1, 3, 5, 10, 15)
  top_k_stats <- setNames(
    lapply(k_values, get_top_k, counts = mask_counts),
    paste0("top_", k_values, "_mask_count")
  )
  
  result <- data.frame(
    file = file_path,
    n_rows = n_rows,
    n_cols = n_cols,
    n_missing = n_missing,
    n_unique_masks = n_unique_masks,
    stringsAsFactors = FALSE
  )
  
  bind_cols(result, as.data.frame(top_k_stats))
}

file_list <- list.files(path = "real_datasets/", pattern = "\\.RDS$", recursive = TRUE, full.names = TRUE, ignore.case = TRUE)

results <- map_dfr(file_list, analyze_rds_file)

print(results)

saveRDS(results, file = "real_datasets_results/coverage_results.rds")


resul_loaded <- readRDS("real_datasets_results/coverage_results.rds")

# Cols of interest
# - file name (without rds, and in lower case)
# - n_rows
# - n_cols
# - n_missing (% of cells)
# - top 3 mask coverage (% of rows having top 3 masks)
# - top 10 mask coverage (% of rows having top 10 masks)

final_results <- resul_loaded %>%
  mutate(
    file = tools::file_path_sans_ext(basename(tolower(file))),
    pct_missing = n_missing / (n_rows * n_cols) * 100,
    top_3_mask_coverage = top_3_mask_count / n_rows * 100,
    top_10_mask_coverage = top_10_mask_count / n_rows * 100
  ) %>%
  select(
    file,
    n_rows,
    n_cols,
    n_unique_masks,
    pct_missing,
    top_3_mask_coverage,
    top_10_mask_coverage
  )
print(final_results)


library(xtable)
latex_table <- xtable(final_results, 
                      caption = "Summary of missing data statistics per dataset.",
                      label = "tab:real_data_sets",
                      digits = c(0, 0, 0, 0, 0, 2, 2, 2))

print(latex_table, 
      file = "tables_and_figures/missing_data_table.tex", 
      include.rownames = FALSE,  # Usually cleaner without row numbers
      booktabs = TRUE,           # Use booktabs style (requires \usepackage{booktabs} in LaTeX)
      sanitize.text.function = function(x) { x }) # Optional: Use NULL or default if you want standard escaping
