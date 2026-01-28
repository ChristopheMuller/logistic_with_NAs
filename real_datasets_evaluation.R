library(pROC, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(tidyr, quietly=TRUE)
library(ggplot2, quietly=TRUE)

datasets <- c("boys", "colic", "debt", "diabetes", "globwarm", "housevotes84", 
              "oceanbuoys", "popmis", "pulplignin", "selfreport", "soybean", 
              "tbc", "vnf", "walking",
              
              "airquality", "chorizonDL", "NHANES", "Ozone", "pedestrian", 
              "riskfactors", "SBS5242", "sleep")

results_path <- "real_datasets_results/"
results_list <- list()

for (d_name in datasets) {
  
  preds_file <- paste0(results_path, d_name, "_preds.RDS")
  y_file <- paste0(results_path, d_name, "_Y.RDS")
  folds_file <- paste0(results_path, d_name, "_folds.RDS")
  
  if (!file.exists(preds_file)) next
  
  all_preds <- readRDS(preds_file)
  Y_truth <- readRDS(y_file)
  fold_ids <- readRDS(folds_file)
  
  Y_num <- as.numeric(Y_truth)
  
  for (method_name in names(all_preds)) {
    probs <- all_preds[[method_name]]
    
    if (any(is.na(probs))) {
      probs[is.na(probs)] <- mean(Y_num, na.rm=TRUE)
    }
    
    # Create a temporary dataframe for easy grouping
    tmp_df <- data.frame(
      y = Y_truth, 
      y_n = Y_num, 
      prob = probs, 
      fold = fold_ids
    )
    
    # Compute metrics PER FOLD
    fold_metrics <- tmp_df %>%
      group_by(fold) %>%
      summarise(
        auc = as.numeric(pROC::roc(y, prob, quiet=TRUE)$auc),
        brier = mean((prob - y_n)^2),
        acc = mean((prob > 0.5) == y),
        .groups = 'drop'
      )
    
    # Aggregate results
    results_list[[length(results_list) + 1]] <- data.frame(
      Dataset = d_name,
      Method = method_name,
      AUC_Mean = mean(fold_metrics$auc),
      AUC_SD = sd(fold_metrics$auc),
      Brier_Mean = mean(fold_metrics$brier),
      Brier_SD = sd(fold_metrics$brier),
      Accuracy_Mean = mean(fold_metrics$acc),
      Accuracy_SD = sd(fold_metrics$acc)
    )
  }
}

results_df <- do.call(rbind, results_list)

print(results_df %>% 
        select(Dataset, Method, AUC_Mean, AUC_SD) %>% 
        arrange(Dataset, desc(AUC_Mean)))

# Visualization with Error Bars (Mean +/- SD)
max_SD <- (results_df$AUC_Mean + results_df$AUC_SD)
max_SD [max_SD > 1] <- 1
min_SD <- (results_df$AUC_Mean - results_df$AUC_SD)
min_SD [min_SD < 0.5] <- 0.5
plot_auc <- ggplot(results_df, aes(x = reorder(Method, AUC_Mean), y = AUC_Mean, color = Method)) +
  geom_point(size = 1) +
  geom_errorbar(aes(ymin = min_SD, ymax = max_SD), width = 0.2) +
  facet_wrap(~Dataset, scales = "free_y") +
  coord_flip() +
  # set x limit: 0.5 to 1
  ylim(0.5, 1) +
  theme_minimal() +
  labs(title = "Method Performance (AUC) with Uncertainty (SD)", 
       x = "Method", y = "AUC (Mean +/- SD)") +
  theme(legend.position = "none")

print(plot_auc)
