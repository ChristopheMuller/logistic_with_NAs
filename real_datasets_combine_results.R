# Setup paths
base_path <- "icml_other_datasets_results"
# List your subdirectories here
sub_dirs <- c("preds_0" ,"preds_170", "preds_210_to_300", "preds_coverages")
output_dir <- "icml_other_datasets_results/preds"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Identify all unique datasets processed across all folders
all_files <- list.files(file.path(base_path, sub_dirs), pattern = "_mc_results.RDS")
dataset_names <- unique(gsub("_mc_results.RDS", "", all_files))

cat("Found", length(dataset_names), "datasets to merge.\n")

for (ds in dataset_names) {
  cat("Merging results for:", ds, "...\n")
  
  combined_results <- list()
  y_file_saved <- FALSE
  
  for (sd in sub_dirs) {
    results_path <- file.path(base_path, sd, paste0(ds, "_mc_results.RDS"))
    y_path <- file.path(base_path, sd, paste0(ds, "_Y.RDS"))
    
    # Merge the Method Results
    if (file.exists(results_path)) {
      part_results <- readRDS(results_path)
      combined_results <- c(combined_results, part_results)
    }
    
    # Copy the Y truth file (only needs to be done once per dataset)
    if (!y_file_saved && file.exists(y_path)) {
      file.copy(y_path, file.path(output_dir, paste0(ds, "_Y.RDS")), overwrite = TRUE)
      y_file_saved <- TRUE
    }
  }
  
  # 2. Save the combined results
  if (length(combined_results) > 0) {
    saveRDS(combined_results, file = file.path(output_dir, paste0(ds, "_mc_results.RDS")))
    cat("  Done. Total method-fold entries:", length(combined_results), "\n")
  } else {
    cat("  Warning: No results found for", ds, "\n")
  }
}

cat("\nAll results have been merged into:", output_dir, "\n")