

library(reticulate)
library(dplyr)
library(stringr)


source("methods_in_R.R")
reticulate::use_python(Sys.which("python3"))
# reticulate::use_python("C:\\Users\\Chris\\Anaconda3\\envs\\logistic\\python.exe")


exp <- "ExpC"
training_sizes <- c(100, 500, 1000, 5000, 10000, 50000, 100000, 200000, 300000)
test_size <- 15000


np <- import("numpy")
data <- np$load(file.path("data", exp, "original_data", "LOG_n315000_d3_corr065_prcNA035_prop105_rep0.npz"))
X_obs <- data$f[["X_obs"]]
M <- data$f[["M"]]
y <- data$f[["y"]]

data_test <- np$load(file.path("data", exp, "test_data", "LOG_n315000_d3_corr065_prcNA035_prop105_rep0.npz"))
X_test <- data_test$f[["X_obs"]]
M_test <- data_test$f[["M"]]
y_probs_test <- data_test$f[["y_probs"]]
y_test <- data_test$f[["y"]]


results = list()
results$runnint_time <- list()
results$y_probs_pred <- list()
results$beta_estimated <- list()
results$mu_estimated <- list()
results$sigma_estimated <- list()

for (n_train in training_sizes){
  
  print(paste0("Training size: ", n_train))
  
  X_train <- X_obs[1:n_train, ]
  M_train <- M[1:n_train, ]
  y_train <- y[1:n_train]
  
  method <- SAEMLogisticRegression$new(name="SAEM")
  
  tic <- Sys.time()
  method$fit(X_train, M_train, y_train, X_test, M_test)
  toc <- Sys.time()
  running_time <- as.numeric(difftime(toc, tic, units = "secs"))
  
  y_probs_pred <- method$predict_probs(X_test, M_test)
  beta_estimated <- method$model$coefficients
  mu_estimated <- method$model$mu.X
  sigma_estimated <- method$model$Sig.X
  
  results$runnint_time[[as.character(n_train)]] <- running_time
  results$y_probs_pred[[as.character(n_train)]] <- y_probs_pred
  results$beta_estimated[[as.character(n_train)]] <- beta_estimated
  results$mu_estimated[[as.character(n_train)]] <- mu_estimated
  
  
}


current_direcotry <- getwd()
results_path <- file.path(current_direcotry, "/data/ExpC/SAEM_deepdive.RDS")
saveRDS(results, results_path)


