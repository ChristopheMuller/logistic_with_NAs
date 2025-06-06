library(reticulate)
library(dplyr)
library(stringr)

source("methods_in_R.R")
# reticulate::use_python(Sys.which("python3"))
# reticulate::use_python("C:\\Users\\Chris\\Anaconda3\\envs\\logistic\\python.exe")


# Configuration
exp <- "SimulationD"
# training_sizes <- c(100)
training_sizes <- c(100, 500, 1000, 5000, 10000, 50000)
test_size <- 15000

# Initialize methods list
methods_list <- list(
  # SAEMLogisticRegression$new(name="SAEM"),

  # MICELogisticRegression$new(name="MICE.M.IMP", n_imputations=1, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.M.IMP.M", n_imputations=1, add.y=FALSE, mask.after=TRUE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.Y.M.IMP", n_imputations=1, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.Y.M.IMP.M", n_imputations=1, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.IMP.M", n_imputations=1, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
  # MICELogisticRegression$new(name="MICE.Y.IMP.M", n_imputations=1, add.y=TRUE, mask.after=TRUE, mask.before=FALSE),
  # MICELogisticRegression$new(name="MICE.IMP", n_imputations=1, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
  # MICELogisticRegression$new(name="MICE.Y.IMP", n_imputations=1, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),

  # MICELogisticRegression$new(name="MICE.10.M.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.10.M.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE),
  # MICELogisticRegression$new(name="MICE.10.Y.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),
  # MICELogisticRegression$new(name="MICE.10.Y.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=FALSE),
  # MICELogisticRegression$new(name="MICE.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
  # MICELogisticRegression$new(name="MICE.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE)

  MICELogisticRegression$new(name="MICE.1000.IMP", n_imputations=1000, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
  MICELogisticRegression$new(name="MICE.1000.Y.IMP", n_imputations=1000, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),
  MICELogisticRegression$new(name="MICE.1000.M.IMP", n_imputations=1000, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
  MICELogisticRegression$new(name="MICE.1000.Y.M.IMP", n_imputations=1000, add.y=TRUE, mask.after=FALSE, mask.before=TRUE)

  # MICERFLogisticRegression$new(name="MICE.RF.10.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=FALSE),
  # MICERFLogisticRegression$new(name="MICE.RF.10.Y.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=FALSE),
  # MICERFLogisticRegression$new(name="MICE.RF.10.M.IMP", n_imputations=10, add.y=FALSE, mask.after=FALSE, mask.before=TRUE),
  # MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP", n_imputations=10, add.y=TRUE, mask.after=FALSE, mask.before=TRUE)
  # MICERFLogisticRegression$new(name="MICE.RF.10.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=FALSE),
  # MICERFLogisticRegression$new(name="MICE.RF.10.Y.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=FALSE),
  # MICERFLogisticRegression$new(name="MICE.RF.10.M.IMP.M", n_imputations=10, add.y=FALSE, mask.after=TRUE, mask.before=TRUE),
  # MICERFLogisticRegression$new(name="MICE.RF.10.Y.M.IMP.M", n_imputations=10, add.y=TRUE, mask.after=TRUE, mask.before=TRUE)

)

print("TRAINING IN R:")
print("Methods:")
for (method in methods_list) {
  print(method$name)
}
print("Training sizes:")
print(training_sizes)

# Read setup data
df_set_up <- read.csv(file.path("data", exp, "set_up.csv"))

# Check if simulation file exists and create/read it
simulation_file <- file.path("data", exp, "simulation.csv")
if (file.exists(simulation_file)) {
  simulations_df <- read.csv(simulation_file)
  # if the file does not contain the column "running_time_pred", add it
  if (!"running_time_pred" %in% colnames(simulations_df)) {
    simulations_df$running_time_pred <- NA
  }
} else {
  simulations_df <- data.frame(
    set_up = character(),
    method = character(),
    n_train = numeric(),
    estimated_beta = character(),
    file_name = character(),
    running_time = numeric(),
    running_time_pred = numeric(),
    stringsAsFactors = FALSE
  )
}


np <- import("numpy")
for (i in 1:nrow(df_set_up)) {
  cat(sprintf("Running set up %d out of %d: %s\n", i, nrow(df_set_up), df_set_up$set_up[i]))
  
  data <- np$load(file.path("data", exp, "original_data", paste0(df_set_up$set_up[i], ".npz")))
  X_obs <- data$f[["X_obs"]]
  M <- data$f[["M"]]
  y <- data$f[["y"]]
  
  data_test <- np$load(file.path("data", exp, "test_data", paste0(df_set_up$set_up[i], ".npz")))
  X_test <- data_test$f[["X_obs"]]
  M_test <- data_test$f[["M"]]
  y_probs_test <- data_test$f[["y_probs"]]
  y_test <- data_test$f[["y"]]
  
  for (n_train in training_sizes) {
    cat(sprintf("\tTraining size: %d\n", n_train))
    
    X_train <- X_obs[1:n_train, ]
    M_train <- M[1:n_train, ]
    y_train <- y[1:n_train]
    
    for (method in methods_list) {

      fit_success <- tryCatch({
        tic <- Sys.time()
        method$fit(X_train, M_train, y_train, X_test, M_test)
        toc <- Sys.time()
        running_time <- as.numeric(difftime(toc, tic, units = "secs"))
        TRUE
      }, error = function(e) {
        cat(sprintf("Error in fit for method %s: %s\n", method$name, e$message))
        FALSE
      })
      
      if (fit_success && method$can_predict) {

        pred_success <- tryCatch({
          tic.pred <- Sys.time()
          y_probs_pred <- method$predict_probs(X_test, M_test)
          toc.pred <- Sys.time()
          running_time.pred <- as.numeric(difftime(toc.pred, tic.pred, units = "secs"))
          
          save_name <- paste0(df_set_up$set_up[i], "_", method$name, "_", n_train)
          
          np$savez(
            file.path("data", exp, "pred_data", paste0(save_name, ".npz")),
            y_probs_pred = y_probs_pred
          )
          TRUE
        }, error = function(e) {
          cat(sprintf("Error in prediction for method %s: %s\n", method$name, e$message))
          FALSE
        })
        
        if (fit_success && pred_success) {

          if (method$return_beta) {
            estimated_beta <- method$return_params()
          } else {
            estimated_beta <- NULL
          }
          
          new_row <- data.frame(
            set_up = df_set_up$set_up[i],
            method = method$name,
            n_train = n_train,
            estimated_beta = toString(estimated_beta),
            file_name = save_name,
            running_time = running_time,
            running_time_pred = running_time.pred,
            stringsAsFactors = FALSE
          )
          simulations_df <- rbind(simulations_df, new_row)
          
          write.csv(simulations_df, simulation_file, row.names = FALSE)
        }
      }
    }
  }
}
