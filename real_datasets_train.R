

# Packages
library(tidyr)
library(dplyr)
library(furrr)
library(future)
library(reticulate)
library(stringr)
library(ggplot2)
source("methods_in_R.R")


# Input
k_fold <- 5
get_fresh_methods <- function() {
  list(
    # SAEMLogisticRegression$new(name="SAEM", lambda=0, alpha=0),
    # 
    # MICELogisticRegression$new(name="MICE.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    # MICELogisticRegression$new(name="MICE.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),
    # 
    # MICERFLogisticRegression$new(name="MICE.RF.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
    # MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),
    # 
    # MICELogisticRegression$new(name="MICE.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    # MICELogisticRegression$new(name="MICE.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),
    # 
    # MICERFLogisticRegression$new(name="MICE.RF.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
    # MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),
    
    MeanImputationLogisticRegression$new(name="Mean.IMP", mask=FALSE),
    MeanImputationLogisticRegression$new(name="Mean.IMP.M", mask=TRUE),
    
    ConstantImputationLogisticRegression$new(name="05.IMP", fill_value=0.5, mask=FALSE),
    ConstantImputationLogisticRegression$new(name="05.IMP.M", fill_value=0.5, mask=TRUE),
    
    RegLogPatByPat$new(name="PbP"),
    RegLogPatByPatMinObservation$new(name="PbP.MinObs")
  )
}

methods_cannot_deal_with_categorical <- c(
  "SAEM",
  "05.IMP",
  "05.IMP.M"
)

data_info <- list(
  airquality = list(
    file = "airquality",
    var = "Wind",
    value = 9.7
  ),
  boys = list(
    file = "boys",
    var = "age",
    value = 10.5045
  ),
  ## chorizonDL = list(
  ##   file = "chorizonDL",
  ##   var = "Ti_XRF",
  ##   value = 0.347
  ## ),
  colic = list(
    file = "colic",
    var = "outcome",
    value = 2
  ),
  debt = list(
    file = "debt",
    var = "prodebt",
    value = 3.24
  ),
  diabetes = list(
    file = "diabetes",
    var = "Class",
    value = 0
  ),
  globwarn = list(
    file = "globwarm",
    var = "chesapeake",
    value = -0.48
  ),
  housevotes84 = list(
    file = "housevotes84",
    var = "Class",
    value = 1
  ),
  NHANES = list(
    file = "NHANES",
    var = "Age",
    value = 36
  ),
  oceanbuoys = list(
    file = "oceanbuoys",
    var = "wind_ns",
    value = 2.9
  ),
  Ozone = list(
    file = "Ozone",
    var = "V13",
    value = 110
  ),
  pedestrian = list(
    file = "pedestrian",
    var = "sensor_id",
    value = 10
  ),
  popmis = list(
    file = "popmis",
    var = "teachpop",
    value = 4
  ),
  pulplignin = list(
    file = "pulplignin",
    var = "Y.Kappa",
    value = 20.74
  ),
  ## riskfactors = list(
  ##   file = "riskfactors",
  ##   var = "health_general",
  ##   value = 2
  ## ),
  SBS5242 = list(
    file = "SBS5242",
    var = "USB",
    value = 4.779415
  ),
  selfreport = list(
    file = "selfreport",
    var = "sex",
    value = 1.5
  ),
  sleep = list(
    file = "sleep",
    var = "Danger",
    value = 2
  ),
  soybean = list(
    file = "soybean",
    var = "Class",
    value = 7
  ),
  tbc = list(
    file = "tbc",
    var = "sex",
    value = 1
  ),
  vnf = list(
    file = "vnf",
    var = "Q8.1",
    value = 1
  ),
  walking = list(
    file = "walking",
    var = "sex",
    value = 1
  )
)


# Training

for(datas in data_info){
  file <- datas$file
  var <- datas$var
  value <- datas$value
  
  dataset <- readRDS(paste0("real_datasets/", file, ".RDS"))
  remove_NAs <- is.na(dataset[[var]])
  dataset <- dataset[!remove_NAs, ]
  
  Y.values <- dataset[[var]]
  Y <- as.numeric(Y.values) <= value
  X <- dataset %>% select(-all_of(var))
  cat_var <- sapply(X, is.factor)
  
  n <- nrow(X)
  d <- ncol(X)
  
  # randomize the indices
  new_idx <- sample(1:n, n, replace=FALSE)
  X <- X[new_idx, ]
  Y <- Y[new_idx]
  M <- is.na(X)
  
  methods <- get_fresh_methods()
  
  all_preds <- list()
  for(met in methods){
    all_preds[[met$name]] <- rep(NA, n)
  }
  
  fold_ids <- numeric(n)
  
  for(fold in 1:k_fold){
    cat("Dataset:", file, "- Fold:", fold, "\n")
    test_indices <- seq(fold, n, by=k_fold)
    train_indices <- setdiff(1:n, test_indices)
    fold_ids[test_indices] <- fold
    
    X_train <- X[train_indices, ]
    Y_train <- Y[train_indices]
    M_train <- M[train_indices, ]
    
    X_test <- X[test_indices, ]
    Y_test <- Y[test_indices]
    M_test <- M[test_indices, ]
    
    factor_cols <- which(cat_var)
    for(j in factor_cols){
      train_levs <- unique(X_train[[j]])
      
      is_unseen <- !(X_test[[j]] %in% train_levs) & !is.na(X_test[[j]])
      
      if(any(is_unseen)){
        X_test[is_unseen, j] <- NA
        M_test[is_unseen, j] <- TRUE
      }
      
    }
    
    for(method in methods){
      cat("   Method:", method$name, "\n")
      
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
          cat(paste0("   Warning: Method '", method$name, "' failed (", e$message, "). Using global mean.\n"))
          return(rep(mean(Y_train, na.rm = TRUE), nrow(X_test_valid)))
        })        
        final_preds[valid_idx] <- valid_preds
      }
      all_preds[[method$name]][test_indices] <- final_preds
      
    }
  }
  
  dir.create("real_datasets_results/preds", showWarnings = FALSE)
  saveRDS(all_preds, file=paste0("real_datasets_results/preds/", file, "_preds.RDS"))
  saveRDS(Y, file=paste0("real_datasets_results/preds/", file, "_Y.RDS"))
  saveRDS(fold_ids, file=paste0("real_datasets_results/preds/", file, "_folds.RDS"))
}

