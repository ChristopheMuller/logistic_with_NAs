library(tidyr)
library(dplyr)
library(pROC)
library(knitr)

folders <- c("real_datasets_results")

dataset_names <- c("boys", "colic", "debt", "diabetes", "globwarm", "housevotes84", 
              "oceanbuoys", "popmis", "pulplignin", "selfreport", "soybean", 
              "tbc", "vnf", "walking",
              
              "airquality", "chorizonDL", "NHANES", "Ozone", "pedestrian", 
              "riskfactors", "SBS5242", "sleep")

results_list <- list()

for (fldr in folders) {
  for (dname in dataset_names) {
    
    path_base <- file.path(fldr, "real_datasets_results")
    if (!dir.exists(path_base)) path_base <- fldr
    
    pred_file <- file.path(path_base, paste0(dname, "_preds.RDS"))
    y_file <- file.path(path_base, paste0(dname, "_Y.RDS"))
    fold_file <- file.path(path_base, paste0(dname, "_folds.RDS"))
    
    if (file.exists(pred_file) && file.exists(y_file) && file.exists(fold_file)) {
      
      preds_list <- readRDS(pred_file)
      Y <- readRDS(y_file)
      fold_ids <- readRDS(fold_file)
      Y_num <- as.numeric(Y)
      
      for (method_name in names(preds_list)) {
        probs <- preds_list[[method_name]]
        
        if (any(is.na(probs))) {
          probs[is.na(probs)] <- mean(Y_num, na.rm=TRUE)
        }
        
        tmp_df <- data.frame(
          y = Y,
          prob = probs,
          fold = fold_ids
        )
        
        fold_metrics <- tmp_df %>%
          group_by(fold) %>%
          summarise(
            auc = tryCatch(as.numeric(pROC::roc(y, prob, quiet=TRUE)$auc), error = function(e) NA),
            .groups = 'drop'
          )
        
        fold_metrics$Dataset <- dname
        fold_metrics$Method <- method_name
        fold_metrics$Folder <- fldr
        
        results_list[[length(results_list) + 1]] <- fold_metrics
      }
    }
  }
}

all_folds <- bind_rows(results_list)

method_stats <- all_folds %>%
  group_by(Dataset, Method) %>%
  summarise(
    Mean_AUC = mean(auc, na.rm = TRUE),
    SE_AUC = sd(auc, na.rm = TRUE) / sqrt(n()),
    Fold_Data = list(auc),
    .groups = 'drop'
  )

best_methods <- method_stats %>%
  group_by(Dataset) %>%
  summarise(
    Best_Mean = max(Mean_AUC),
    Best_Method = Method[which.max(Mean_AUC)],
    Best_Folds = Fold_Data[which.max(Mean_AUC)],
    .groups = 'drop'
  )

final_table <- method_stats %>%
  left_join(best_methods, by = "Dataset") %>%
  rowwise() %>%
  mutate(
    p_val = if(Method == Best_Method) {
      1.0
    } else {
      tryCatch(
        t.test(unlist(Best_Folds), unlist(Fold_Data), paired = TRUE, alternative = "greater")$p.value,
        error = function(e) 0
      )
    },
    is_bold = p_val >= 0.05,
    formatted_auc = sprintf("%.2f", Mean_AUC),
    formatted_cell = ifelse(is_bold, paste0("\\textbf{", formatted_auc, "}"), formatted_auc)
  ) %>%
  ungroup()

wide_auc <- final_table %>%
  select(Dataset, Method, formatted_cell) %>%
  pivot_wider(names_from = Dataset, values_from = formatted_cell) %>%
  arrange(Method)

print(wide_auc)
kable(wide_auc, format = "latex", booktabs = TRUE, escape = FALSE)

