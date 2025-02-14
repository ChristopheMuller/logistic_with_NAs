library(reticulate)
library(dplyr)
library(stringr)

source("methods_in_R.R")
reticulate::use_python(Sys.which("python3"))
# reticulate::use_python("C:\\Users\\Chris\\Anaconda3\\envs\\logistic\\python.exe")


# Configuration
exp <- "ExpA"
training_sizes <- c(500, 1000, 5000, 10000, 50000)
test_size <- 15000

# Initialize methods list
methods_list <- list(
  # MICELogisticRegression$new(name="MICE.IMP", n_imputations = 1),
  # MICELogisticRegression$new(name="MICE.5.IMP", n_imputations = 5),
  # MICELogisticRegression$new(name="MICE.M.IMP", n_imputations = 1, mask=TRUE),
  # MICELogisticRegression$new(name="MICE.Y.IMP", n_imputations = 1, add.y=TRUE),
  # MICELogisticRegression$new(name="MICE.Y.M.IMP", n_imputations = 1, add.y=TRUE, mask=TRUE),
  # SAEMLogisticRegression$new(name="SAEM")
  # MICELogisticRegression$new(name="MICE.10.IMP", n_imputations = 10),
  # MICELogisticRegression$new(name="MICE.20.IMP", n_imputations = 20),
  MICELogisticRegression$new(name="MICE.100.Y.IMP", n_imputations = 100, add.y=TRUE)
)

# Read setup data
df_set_up <- read.csv(file.path("data", exp, "set_up.csv"))

# Check if simulation file exists and create/read it
simulation_file <- file.path("data", exp, "simulation.csv")
if (file.exists(simulation_file)) {
  simulations_df <- read.csv(simulation_file)
} else {
  simulations_df <- data.frame(
    set_up = character(),
    method = character(),
    n_train = numeric(),
    estimated_beta = character(),
    file_name = character(),
    stringsAsFactors = FALSE
  )
}


# Main loop
for (i in 1:nrow(df_set_up)) {
  cat(sprintf("Running set up %d out of %d: %s\n", i, nrow(df_set_up), df_set_up$set_up[i]))
  
  # Load NPZ files using reticulate
  np <- import("numpy")
  data <- np$load(file.path("data", exp, "original_data", paste0(df_set_up$set_up[i], ".npz")))
  X_obs <- data$f[["X_obs"]]
  M <- data$f[["M"]]
  y <- data$f[["y"]]
  y_probs <- data$f[["y_probs"]]
  X_full <- data$f[["X_full"]]
  
  # Parse true beta from setup
  true_beta <- as.numeric(str_split(gsub("\\[|\\]", "", df_set_up$true_beta[i]), " ")[[1]])
  
  # Load test data
  data_test <- np$load(file.path("data", exp, "test_data", paste0(df_set_up$set_up[i], ".npz")))
  X_test <- data_test$f[["X_obs"]]
  M_test <- data_test$f[["M"]]
  y_probs_test <- data_test$f[["y_probs"]]
  y_test <- data_test$f[["y"]]
  
  for (n_train in training_sizes) {
    cat(sprintf("\tTraining size: %d\n", n_train))
    
    # Subset training data
    X_train <- X_obs[1:n_train, ]
    M_train <- M[1:n_train, ]
    y_train <- y[1:n_train]
    
    for (method in methods_list) {
      # Wrap fit method in try-catch
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
      
      # Only proceed if fit was successful
      if (fit_success && method$can_predict) {
        # Wrap prediction in try-catch
        pred_success <- tryCatch({
          y_probs_pred <- method$predict_probs(X_test, M_test)
          
          save_name <- paste0(df_set_up$set_up[i], "_", method$name, "_", n_train)
          
          # Save predictions using numpy
          np$savez(
            file.path("data", exp, "pred_data", paste0(save_name, ".npz")),
            y_probs_pred = y_probs_pred
          )
          TRUE
        }, error = function(e) {
          cat(sprintf("Error in prediction for method %s: %s\n", method$name, e$message))
          FALSE
        })
        
        # Only add row to simulations if both fit and prediction were successful
        if (fit_success && pred_success) {
          # Get estimated parameters
          if (method$return_beta) {
            estimated_beta <- method$return_params()
          } else {
            estimated_beta <- NULL
          }
          
          # Update simulations dataframe
          new_row <- data.frame(
            set_up = df_set_up$set_up[i],
            method = method$name,
            n_train = n_train,
            estimated_beta = toString(estimated_beta),
            file_name = save_name,
            running_time = running_time,
            stringsAsFactors = FALSE
          )
          simulations_df <- rbind(simulations_df, new_row)
          
          # Save updated simulations
          write.csv(simulations_df, simulation_file, row.names = FALSE)
        }
      }
    }
  }
}
