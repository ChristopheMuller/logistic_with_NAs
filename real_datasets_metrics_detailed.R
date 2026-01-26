library(dplyr)
library(tidyr)
library(pROC)
library(reliabilitydiag)

# Datasets definition
datasets <- c("boys", "colic", "debt", "diabetes", "globwarm", "housevotes84", 
              "oceanbuoys", "popmis", "pulplignin", "selfreport", "soybean", 
              "tbc", "vnf", "walking",
              "airquality", "chorizonDL", "NHANES", "Ozone", "pedestrian", 
              "riskfactors", "SBS5242", "sleep")

preds_path <- "real_datasets_results/preds/"
results_path <- "real_datasets_results/"
results_list <- list()

cat("Starting metric computation for MCCV results...\n")

for (d_name in datasets) {
  
  res_file <- file.path(preds_path, paste0(d_name, "_mc_results.RDS"))
  
  if (!file.exists(res_file)) {
    cat("Skipping", d_name, "- results not found.\n")
    next
  }
  
  cat("Processing", d_name, "...\n")
  
  mc_results <- readRDS(res_file)
  
  for (i in seq_along(mc_results)) {
    result <- mc_results[[i]]
    
    method_name <- result$method
    fold_id <- result$fold
    probs <- result$preds
    y_truth <- result$truth
    
    # Ensure Numeric Y (0/1)
    y_n <- as.numeric(y_truth)
    
    # Handle NAs in predictions (fallback to mean of current test set)
    if (any(is.na(probs))) {
      probs[is.na(probs)] <- mean(y_n, na.rm = TRUE)
    }
    
    # --- Metric Computation ---
    
    # 1. AUC
    # We use tryCatch because if a fold has only one class (all 0s or all 1s), AUC is undefined
    auc_val <- tryCatch({
      # quiet=TRUE suppresses messages about direction
      as.numeric(pROC::roc(y_n, probs, quiet = TRUE)$auc)
    }, error = function(e) NA)
    
    # 2. Brier Score
    brier_val <- mean((probs - y_n)^2)
    
    # 3. Misclassification Rate (Threshold 0.5)
    acc_val <- mean((probs > 0.5) == (y_n == 1))
    misclass_val <- 1 - acc_val
    
    # 4. Calibration Score (with Zero Variance Fix)
    calib_val <- tryCatch({
      # If predictions are constant (variance is 0), reliability diagram fails.
      if (var(probs) == 0) {
        NA
      } else {
        # Using reliabilitydiag
        diag <- reliabilitydiag(preds = probs, y = as.integer(y_n))
        # Extract miscalibration score (index 1)
        summary(diag)$miscalibration[1]
      }
    }, error = function(e) {
      NA
    })
    
    # --- Storage ---
    
    # Helper to add row
    add_res <- function(metric, value) {
      data.frame(
        Dataset = d_name,
        Method = method_name,
        Fold = fold_id,
        Metric = metric,
        Value = value
      )
    }
    
    results_list[[length(results_list) + 1]] <- add_res("AUC", auc_val)
    results_list[[length(results_list) + 1]] <- add_res("Brier", brier_val)
    results_list[[length(results_list) + 1]] <- add_res("Misclassification", misclass_val)
    results_list[[length(results_list) + 1]] <- add_res("Calibration", calib_val)
  }
}


final_df <- do.call(rbind, results_list)
output_file <- file.path(results_path, "real_datasets_metrics_detailed.csv")
write.csv(final_df, output_file, row.names = FALSE)

cat("Processing complete. Saved to:", output_file, "\n")