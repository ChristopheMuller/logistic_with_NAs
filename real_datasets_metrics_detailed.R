library(dplyr)
library(tidyr)
library(pROC)
library(reliabilitydiag)

# Datasets definition (matching your evaluation script)
datasets <- c("boys", "colic", "debt", "diabetes", "globwarm", "housevotes84", 
              "oceanbuoys", "popmis", "pulplignin", "selfreport", "soybean", 
              "tbc", "vnf", "walking",
              "airquality", "chorizonDL", "NHANES", "Ozone", "pedestrian", 
              "riskfactors", "SBS5242", "sleep")

preds_path <- "real_datasets_results/preds/"
results_path <- "real_datasets_results/"
results_list <- list()

cat("Starting metric computation...\n")

for (d_name in datasets) {
  
  preds_file <- file.path(preds_path, paste0(d_name, "_preds.RDS"))
  y_file <- file.path(preds_path, paste0(d_name, "_Y.RDS"))
  folds_file <- file.path(preds_path, paste0(d_name, "_folds.RDS"))
  
  # Check if results exist for this dataset
  if (!file.exists(preds_file)) {
    cat("Skipping", d_name, "- results not found.\n")
    next
  }
  
  cat("Processing", d_name, "...\n")
  
  all_preds <- readRDS(preds_file)
  Y_truth <- readRDS(y_file) # This is likely logical or numeric
  fold_ids <- readRDS(folds_file)
  
  # Ensure Y is numeric (0/1) for calculations
  Y_num <- as.numeric(Y_truth)
  
  for (method_name in names(all_preds)) {
    probs <- all_preds[[method_name]]
    
    # Handle NA predictions (fallback to mean, consistent with evaluation script)
    if (any(is.na(probs))) {
      probs[is.na(probs)] <- mean(Y_num, na.rm=TRUE)
    }
    
    # Create a temporary dataframe for fold-wise processing
    df_method <- data.frame(
      y_n = Y_num,
      prob = probs,
      fold = fold_ids
    )
    
    # Loop over each fold to compute metrics individually
    unique_folds <- sort(unique(fold_ids))
    
    for (k in unique_folds) {
      # Subset data for the current fold
      fold_data <- df_method %>% filter(fold == k)
      
      # 1. AUC
      auc_val <- tryCatch({
        as.numeric(pROC::roc(fold_data$y_n, fold_data$prob, quiet=TRUE)$auc)
      }, error = function(e) NA)
      
      # 2. Brier Score (Mean Squared Error between prob and outcome)
      brier_val <- mean((fold_data$prob - fold_data$y_n)^2)
      
      # 3. Misclassification Rate (1 - Accuracy)
      # Using 0.5 threshold
      acc_val <- mean((fold_data$prob > 0.5) == (fold_data$y_n == 1))
      misclass_val <- 1 - acc_val
      
      # 4. Calibration Score
      # Using reliabilitydiag as in your 6B script.
      # Since we don't have Bayes optimal probs (real data), we only provide preds and y.
      calib_val <- tryCatch({
        diag <- reliabilitydiag(preds = fold_data$prob, y = as.integer(fold_data$y_n))
        # Extracting the miscalibration score (usually index 1 is the main score)
        summary(diag)$miscalibration[1]
      }, error = function(e) {
        NA
      })
      
      # Store metrics in a long format
      # Format: Dataset, Method, Fold, Metric, Value
      
      results_list[[length(results_list) + 1]] <- data.frame(
        Dataset = d_name,
        Method = method_name,
        Fold = k,
        Metric = "AUC",
        Value = auc_val
      )
      
      results_list[[length(results_list) + 1]] <- data.frame(
        Dataset = d_name,
        Method = method_name,
        Fold = k,
        Metric = "Brier",
        Value = brier_val
      )
      
      results_list[[length(results_list) + 1]] <- data.frame(
        Dataset = d_name,
        Method = method_name,
        Fold = k,
        Metric = "Misclassification",
        Value = misclass_val
      )
      
      results_list[[length(results_list) + 1]] <- data.frame(
        Dataset = d_name,
        Method = method_name,
        Fold = k,
        Metric = "Calibration",
        Value = calib_val
      )
    }
  }
}

# Combine all results into one dataframe
final_df <- do.call(rbind, results_list)

# Save to CSV
output_file <- file.path(results_path, "real_datasets_metrics_detailed.csv")
write.csv(final_df, output_file, row.names = FALSE)
