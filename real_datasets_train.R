# Packages
library(tidyr)
library(dplyr)
library(furrr)
library(future)
library(reticulate)
library(stringr)
library(ggplot2)
source("methods_in_R.R")

# Setup parallel processing
plan(multisession, workers = parallel::detectCores() - 4)

# Input
k_fold_mc <- 15

get_fresh_methods <- function() {
  list(    
    # SAEMLogisticRegression$new(name="SAEM", lambda=0, alpha=0),
    
    # MICELogisticRegression$new(name="MICE.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    # MICELogisticRegression$new(name="MICE.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),
    
    # MICERFLogisticRegression$new(name="MICE.RF.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    # MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),
    
    # MICELogisticRegression$new(name="MICE.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    # MICELogisticRegression$new(name="MICE.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),
    
    # MICERFLogisticRegression$new(name="MICE.RF.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    # MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),
    
    # MeanImputationLogisticRegression$new(name="Mean.IMP", mask=FALSE),
    # MeanImputationLogisticRegression$new(name="Mean.IMP.M", mask=TRUE),
    
    # ConstantImputationLogisticRegression$new(name="05.IMP", fill_value=0.5, mask=FALSE),
    # ConstantImputationLogisticRegression$new(name="05.IMP.M", fill_value=0.5, mask=TRUE),
    
    # RegLogPatByPat$new(name="PbP"),
    RegLogPatByPatMinObservationPreDefined$new(name="PbP.1596", k=1596),
    RegLogPatByPatMinObservationPreDefined$new(name="PbP.1806", k=1806),
    RegLogPatByPatMinObservationPreDefined$new(name="PbP.3149", k=3149),
    RegLogPatByPatMinObservationPreDefined$new(name="PbP.74", k=74)
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k10", k=10),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k20", k=20),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k30", k=30),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k40", k=40),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k50", k=50),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k60", k=60),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k70", k=70),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k80", k=80),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k90", k=90),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.k100", k=100),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.110", k=110),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.120", k=120),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.130", k=130),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.140", k=140),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.150", k=150),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.160", k=160),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.170", k=170),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.180", k=180),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.190", k=190),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP.200", k=200),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP602", k=602),
    # RegLogPatByPatMinObservationPreDefined$new(name="Pbp837", k=837),
    # RegLogPatByPatMinObservationPreDefined$new(name="PbP1383", k=1383)
  )
}

methods_cannot_deal_with_categorical <- c(
  "SAEM",
  "05.IMP",
  "05.IMP.M"
)

# data_info <- list(
#   airquality = list(
#     file = "airquality",
#     var = "Wind",
#     value = 9.7
#   ),
#   boys = list(
#     file = "boys",
#     var = "age",
#     value = 10.5045
#   ),
#   # chorizonDL = list(
#   #   file = "chorizonDL",
#   #   var = "Ti_XRF",
#   #   value = 0.347
#   # ),
#   colic = list(
#     file = "colic",
#     var = "outcome",
#     value = 2
#   ),
#   debt = list(
#     file = "debt",
#     var = "prodebt",
#     value = 3.24
#   ),
#   diabetes = list(
#     file = "diabetes",
#     var = "Class",
#     value = 0
#   ),
#   globwarn = list(
#     file = "globwarm",
#     var = "chesapeake",
#     value = -0.48
#   ),
#   housevotes84 = list(
#     file = "housevotes84",
#     var = "Class",
#     value = 1
#   ),
#   NHANES = list(
#     file = "NHANES",
#     var = "Age",
#     value = 36
#   ),
#   oceanbuoys = list(
#     file = "oceanbuoys",
#     var = "wind_ns",
#     value = 2.9
#   ),
#   Ozone = list(
#     file = "Ozone",
#     var = "V13",
#     value = 110
#   ),
#   pedestrian = list(
#     file = "pedestrian",
#     var = "sensor_id",
#     value = 10
#   ),
#   popmis = list(
#     file = "popmis",
#     var = "teachpop",
#     value = 4
#   ),
#   pulplignin = list(
#     file = "pulplignin",
#     var = "Y.Kappa",
#     value = 20.74
#   ),
#   ## riskfactors = list(
#   ##   file = "riskfactors",
#   ##   var = "health_general",
#   ##   value = 2
#   ## ),
#   SBS5242 = list(
#     file = "SBS5242",
#     var = "USB",
#     value = 4.779415
#   ),
#   selfreport = list(
#     file = "selfreport",
#     var = "sex",
#     value = 1.5
#   ),
#   sleep = list(
#     file = "sleep",
#     var = "Danger",
#     value = 2
#   ),
#   soybean = list(
#     file = "soybean",
#     var = "Class",
#     value = 7
#   ),
#   tbc = list(
#     file = "tbc",
#     var = "sex",
#     value = 1
#   ),
#   vnf = list(
#     file = "vnf",
#     var = "Q8.1",
#     value = 1
#   ),
#   walking = list(
#     file = "walking",
#     var = "sex",
#     value = 1
#   )
# )


rds_dir <- "icml_other_datasets"
file_names <- list.files(path = rds_dir, pattern = "\\.rds$", full.names = FALSE)[2]
dataset_names <- tools::file_path_sans_ext(file_names)
data_info <- lapply(dataset_names, function(name) {
  list(
    file = name,
    var = "target",
    value = 0  
  )
})
names(data_info) <- dataset_names

# Verify the structure
str(data_info)


# Function to process a single fold
process_fold <- function(fold, X, Y, M, cat_var, methods_cannot_deal_with_categorical, seed_base) {
  source("methods_in_R.R")
  
  set.seed(seed_base + fold)
  
  n <- nrow(X)
  prop_test <- 0.2
  
  # MCCV Sampling
  test_indices <- sample(1:n, size = floor(prop_test * n), replace = FALSE)
  train_indices <- setdiff(1:n, test_indices)
  
  X_train <- X[train_indices, ]
  Y_train <- Y[train_indices]
  M_train <- M[train_indices, ]
  
  X_test <- X[test_indices, ]
  Y_test <- Y[test_indices]
  M_test <- M[test_indices, ]
  
  # Handle unseen factor levels
  factor_cols <- which(cat_var)
  for(j in factor_cols){
    train_levs <- unique(X_train[[j]])
    is_unseen <- !(X_test[[j]] %in% train_levs) & !is.na(X_test[[j]])
    
    if(any(is_unseen)){
      X_test[is_unseen, j] <- NA
      M_test[is_unseen, j] <- TRUE
    }
  }
  
  # Get fresh methods for this fold (important for parallel processing)
  methods <- get_fresh_methods()
  
  # Store results for this fold
  fold_method_results <- list()
  
  for(method in methods){
    
    if(method$name %in% methods_cannot_deal_with_categorical){
      X_train_met <- X_train[, !cat_var, drop=FALSE]
      X_test_met <- X_test[, !cat_var, drop=FALSE]
      M_train_met <- M_train[, !cat_var, drop=FALSE]
      M_test_met <- M_test[, !cat_var, drop=FALSE]
    } else {
      X_train_met <- X_train
      X_test_met <- X_test
      M_train_met <- M_train
      M_test_met <- M_test
    }
    
    if (ncol(X_test_met) == 0) {
      is_fully_missing <- rep(TRUE, nrow(X_test_met))
    } else {
      is_fully_missing <- rowSums(M_test_met) == ncol(M_test_met)
    }
    
    baseline_prob <- mean(Y_train, na.rm = TRUE)
    final_preds <- rep(baseline_prob, nrow(X_test_met))
    
    if (any(!is_fully_missing)) {
      valid_idx <- which(!is_fully_missing)
      
      X_test_valid <- X_test_met[valid_idx, , drop=FALSE]
      M_test_valid <- M_test_met[valid_idx, , drop=FALSE]
      
      valid_preds <- tryCatch({
        method$fit(X_train_met, M_train_met, Y_train, X_test_valid, M_test_valid)
        method$predict_probs(X_test_valid, M_test_valid)
      }, error = function(e) {
        return(rep(mean(Y_train, na.rm = TRUE), nrow(X_test_valid)))
      })        
      final_preds[valid_idx] <- valid_preds
    }
    
    # Store result for this method
    fold_method_results[[length(fold_method_results) + 1]] <- list(
      fold = fold,
      method = method$name,
      indices = test_indices,
      preds = final_preds,
      truth = Y_test 
    )
  }
  
  return(fold_method_results)
}

# Training
dir.create("icml_real_datasets_results/preds", showWarnings = FALSE, recursive = TRUE)

for(datas in data_info){
  file <- datas$file
  var <- datas$var
  value <- datas$value
  
  cat("\n========================================\n")
  cat("Processing dataset:", file, "\n")
  cat("========================================\n")
  
  dataset <- readRDS(paste0("icml_other_datasets/", file, ".RDS"))
  remove_NAs <- is.na(dataset[[var]])
  dataset <- dataset[!remove_NAs, ]
  
  Y.values <- dataset[[var]]
  Y <- as.numeric(Y.values) <= value
  X <- dataset %>% select(-all_of(var))
  cat_var <- sapply(X, is.factor)
  
  n <- nrow(X)
  d <- ncol(X)
  
  # Set base seed for this dataset
  seed_base <- sum(utf8ToInt(file))
  set.seed(seed_base)
  
  # Randomize the indices
  new_idx <- sample(1:n, n, replace=FALSE)
  X <- X[new_idx, ]
  Y <- Y[new_idx]
  M <- is.na(X)
  
  cat("Starting parallel processing of", k_fold_mc, "folds...\n")
  start_time <- Sys.time()
  
  # Parallel execution across folds
  fold_results_nested <- future_map(
    1:k_fold_mc, 
    ~process_fold(.x, X, Y, M, cat_var, methods_cannot_deal_with_categorical, seed_base),
    .options = furrr_options(seed = TRUE),
    .progress = TRUE
  )
  
  # Flatten the nested list structure
  fold_results <- unlist(fold_results_nested, recursive = FALSE)
  
  end_time <- Sys.time()
  cat("Completed in", round(difftime(end_time, start_time, units = "mins"), 2), "minutes\n")
  
  # Save results
  
  saveRDS(fold_results, file=paste0("icml_other_datasets_results/preds/", file, "_mc_results.RDS"))
  saveRDS(Y, file=paste0("icml_other_datasets_results/preds/", file, "_Y.RDS"))
  
  cat("Results saved for", file, "\n")
}

# Clean up parallel workers
plan(sequential)
cat("\nAll datasets processed!\n")
