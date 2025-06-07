library(mice, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(stringr, quietly=TRUE)
library(misaem, quietly=TRUE)

# Base class for imputation methods
ImputationMethod <- R6::R6Class("ImputationMethod",
  public = list(
      name = NULL,
      can_predict = TRUE,
      return_beta = TRUE,
      model = NULL,
    imputation_model = NULL,
      
      initialize = function(name) {
        self$name <- name
      },
      
      fit = function(X, M, y) {
        stop("Method not implemented")
      },
      
      predict_probs = function(X_new, M_new) {
        stop("Method not implemented")
      },
      
      return_params = function() {
        if (!self$return_beta) return(NULL)
        return(coef(self$model))
      }
    )
)

MICELogisticRegression <- R6::R6Class("MICELogisticRegression",
  inherit = ImputationMethod,
  public = list(
    n_imputations = 5,
    maxit = 5,
    mask.after = FALSE,
    add.y = FALSE,
    mask.before = FALSE,
    
    initialize = function(name, n_imputations = 5, maxit = 5, mask.before = FALSE, add.y = FALSE, mask.after = FALSE) {
      super$initialize(name)
      self$n_imputations <- n_imputations
      self$maxit <- maxit
      self$add.y <- add.y
      self$mask.before <- mask.before
      self$mask.after <- mask.after
    },
    
    fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
      
      # change the col names of M-train: M1, .., Md
      colnames(M_train) <- paste0("M", seq_len(ncol(M_train)))
      M_train <- as.data.frame(M_train)

      # Combine training data for imputation
      data_train <- as.data.frame(X_train)
      
      # Add y to data if add.y is TRUE
      if (self$add.y) {
        data_train$y <- y_train
      }
      
      # Add mask before imputation if mask.before is TRUE
      if (self$mask.before) {
        data_train <- cbind(data_train, as.data.frame(M_train))
      }
      
      # Create ignore vector for MICE
      ignore_vec <- rep(FALSE, nrow(data_train))
      
      # If test set is provided
      if (!is.null(X_test)) {
        colnames(M_test) <- paste0("M", seq_len(ncol(M_test)))
        data_test <- as.data.frame(X_test)
        
        # Add y (as NA) to test data if add.y is TRUE
        if (self$add.y) {
          data_test$y <- NA
        }

        # Add mask before imputation if mask.before is TRUE
        if (self$mask.before) {
          data_test <- cbind(data_test, as.data.frame(M_test))
        }
        
        # Combine train and test data
        data_full <- rbind(data_train, data_test)
        
        # Update ignore vector
        ignore_vec <- c(ignore_vec, rep(TRUE, nrow(data_test)))
        
        # Run MICE on full dataset
        self$imputation_model <- mice(data_full, m = self$n_imputations, 
                                      maxit = self$maxit, 
                                      printFlag = FALSE, 
                                      ignore = ignore_vec)
      } else {
        # Run MICE only on training data
        self$imputation_model <- mice(data_train, m = self$n_imputations, 
                                      maxit = self$maxit, 
                                      printFlag = FALSE)
      }
      
      print(self$imputation_model$loggedEvents)
      
      # Fit logistic regression on each imputed training dataset
      models <- list()
      for(i in 1:self$n_imputations) {
        # Complete only the training data
        imp_train_data <- complete(self$imputation_model, i)[!ignore_vec, ]

        # Remove y if it was added during imputation
        if (self$add.y) {
          imp_train_data <- imp_train_data[, !names(imp_train_data) %in% "y"]
        }

        if (self$mask.before) {
          imp_train_data <- imp_train_data[, !(names(imp_train_data) %in% names(M_train))]
        }

        # Add mask before logistic regression if mask is TRUE
        if (self$mask.after) {
          imp_train_data <- cbind(imp_train_data, as.data.frame(M_train))
        }

        # Fit logistic regression
        formula <- as.formula(paste("y ~", paste(names(imp_train_data)[names(imp_train_data) != "y"], 
                                                 collapse = " + ")))
        models[[i]] <- glm(formula, family = binomial(), data = cbind(imp_train_data, y = y_train))
      }
      
      # Store models
      self$model <- models
      TRUE
    },
    
    predict_probs = function(X_new, M_new) {

      # change the col names of M_new: M1, .., Md
      colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))

      # Predict for each imputed model
      pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
      
      for(i in 1:self$n_imputations) {
        # Complete the test data using the same imputation model
        imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
        
        # Remove y if it was added during imputation
        if (self$add.y) {
          imp_test <- imp_test[, !names(imp_test) %in% "y"]
        }

        if (self$mask.before) {
          imp_test <- imp_test[, !names(imp_test) %in% names(M_new)]
        }
        
        # Add mask before prediction if mask is TRUE
        if (self$mask.after) {
          imp_test <- cbind(imp_test, as.data.frame(M_new))
        }
        
        # Predict probabilities
        pred_probs[,i] <- predict(self$model[[i]], newdata = imp_test, type = "response")
      }
      
      # Average predictions across imputations
      return(rowMeans(pred_probs))
    },
                                        
    return_params = function() {
      if (!self$return_beta) return(NULL)
      
      # Pool coefficients using Rubin's rules
      coef_list <- lapply(self$model, coef)
      pooled_coef <- Reduce('+', coef_list) / length(coef_list)
      
      # Separate intercept and coefficients
      intercept <- pooled_coef[1]  # First coefficient is intercept in R
      coefficients <- pooled_coef[-1]  # All other coefficients
      
      # Remove names from the vectors
      names(intercept) <- NULL
      names(coefficients) <- NULL
      
      # Create the exact string format to match Python output
      coef_str <- paste(coefficients, collapse = ", ")
      int_str <- as.character(intercept)
      
      return(sprintf("[[%s], [%s]]", coef_str, int_str))
    }
  )                     
)


MICERFLogisticRegression <- R6::R6Class("MICERFLogisticRegression",
                                          inherit = ImputationMethod,
                                          public = list(
                                            n_imputations = 5,
                                            maxit = 5,
                                            mask.after = FALSE,
                                            add.y = FALSE,
                                            mask.before = FALSE,
                                            
                                            initialize = function(name, n_imputations = 5, maxit = 5, mask.before = FALSE, add.y = FALSE, mask.after = FALSE) {
                                              super$initialize(name)
                                              self$n_imputations <- n_imputations
                                              self$maxit <- maxit
                                              self$add.y <- add.y
                                              self$mask.before <- mask.before
                                              self$mask.after <- mask.after
                                            },
                                            
                                            fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
                                              
                                              # change the col names of M-train: M1, .., Md
                                              colnames(M_train) <- paste0("M", seq_len(ncol(M_train)))
                                              M_train <- as.data.frame(M_train)

                                              # Combine training data for imputation
                                              data_train <- as.data.frame(X_train)
                                              
                                              # Add y to data if add.y is TRUE
                                              if (self$add.y) {
                                                data_train$y <- y_train
                                              }
                                              
                                              # Add mask before imputation if mask.before is TRUE
                                              if (self$mask.before) {
                                                data_train <- cbind(data_train, as.data.frame(M_train))
                                              }
                                              
                                              # Create ignore vector for MICE
                                              ignore_vec <- rep(FALSE, nrow(data_train))
                                              
                                              # If test set is provided
                                              if (!is.null(X_test)) {
                                                colnames(M_test) <- paste0("M", seq_len(ncol(M_test)))
                                                data_test <- as.data.frame(X_test)
                                                
                                                # Add y (as NA) to test data if add.y is TRUE
                                                if (self$add.y) {
                                                  data_test$y <- NA
                                                }

                                                # Add mask before imputation if mask.before is TRUE
                                                if (self$mask.before) {
                                                  data_test <- cbind(data_test, as.data.frame(M_test))
                                                }
                                                
                                                # Combine train and test data
                                                data_full <- rbind(data_train, data_test)
                                                
                                                # Update ignore vector
                                                ignore_vec <- c(ignore_vec, rep(TRUE, nrow(data_test)))
                                                
                                                # Run MICE on full dataset
                                                self$imputation_model <- mice(data_full, m = self$n_imputations, 
                                                                              maxit = self$maxit, 
                                                                              printFlag = FALSE, 
                                                                              ignore = ignore_vec, method="rf")
                                              } else {
                                                # Run MICE only on training data
                                                self$imputation_model <- mice(data_train, m = self$n_imputations, 
                                                                              maxit = self$maxit, 
                                                                              printFlag = FALSE, method="rf")
                                              }
                                              
                                              # Fit logistic regression on each imputed training dataset
                                              models <- list()
                                              for(i in 1:self$n_imputations) {
                                                # Complete only the training data
                                                imp_train_data <- complete(self$imputation_model, i)[!ignore_vec, ]

                                                # Remove y if it was added during imputation
                                                if (self$add.y) {
                                                  imp_train_data <- imp_train_data[, !names(imp_train_data) %in% "y"]
                                                }

                                                if (self$mask.before) {
                                                  imp_train_data <- imp_train_data[, !(names(imp_train_data) %in% names(M_train))]
                                                }

                                                # Add mask before logistic regression if mask is TRUE
                                                if (self$mask.after) {
                                                  imp_train_data <- cbind(imp_train_data, as.data.frame(M_train))
                                                }

                                                # Fit logistic regression
                                                formula <- as.formula(paste("y ~", paste(names(imp_train_data)[names(imp_train_data) != "y"], 
                                                                                        collapse = " + ")))
                                                models[[i]] <- glm(formula, family = binomial(), data = cbind(imp_train_data, y = y_train))
                                              }
                                              
                                              # Store models
                                              self$model <- models
                                              TRUE
                                            },
                                            
                                            predict_probs = function(X_new, M_new) {

                                              # change the col names of M_new: M1, .., Md
                                              colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))

                                              # Predict for each imputed model
                                              pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
                                              
                                              for(i in 1:self$n_imputations) {
                                                # Complete the test data using the same imputation model
                                                imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
                                                
                                                # Remove y if it was added during imputation
                                                if (self$add.y) {
                                                  imp_test <- imp_test[, !names(imp_test) %in% "y"]
                                                }

                                                if (self$mask.before) {
                                                  imp_test <- imp_test[, !names(imp_test) %in% names(M_new)]
                                                }
                                                
                                                # Add mask before prediction if mask is TRUE
                                                if (self$mask.after) {
                                                  imp_test <- cbind(imp_test, as.data.frame(M_new))
                                                }
                                                
                                                # Predict probabilities
                                                pred_probs[,i] <- predict(self$model[[i]], newdata = imp_test, type = "response")
                                              }
                                              
                                              # Average predictions across imputations
                                              return(rowMeans(pred_probs))
                                            },
                                                                                
                                            return_params = function() {
                                              if (!self$return_beta) return(NULL)
                                              
                                              # Pool coefficients using Rubin's rules
                                              coef_list <- lapply(self$model, coef)
                                              pooled_coef <- Reduce('+', coef_list) / length(coef_list)
                                              
                                              # Separate intercept and coefficients
                                              intercept <- pooled_coef[1]  # First coefficient is intercept in R
                                              coefficients <- pooled_coef[-1]  # All other coefficients
                                              
                                              # Remove names from the vectors
                                              names(intercept) <- NULL
                                              names(coefficients) <- NULL
                                              
                                              # Create the exact string format to match Python output
                                              coef_str <- paste(coefficients, collapse = ", ")
                                              int_str <- as.character(intercept)
                                              
                                              return(sprintf("[[%s], [%s]]", coef_str, int_str))
                                            }
                                          )                     
)


SAEMLogisticRegression <- R6::R6Class("SAEMLogisticRegression",
                                      inherit = ImputationMethod,
                                      public = list(
                                        initialize = function(name) {
                                          super$initialize(name)
                                        },
                                        
                                        fit = function(X, M, y, X_test = NULL, M_test = NULL) {
                                          # Convert data to required format
                                          data <- as.data.frame(X)
                                          colnames(data) <- paste0("X", 1:ncol(X))
                                          data$y <- y
                                          
                                          # Fit SAEM model
                                          formula <- as.formula(paste("y ~", paste(colnames(data)[1:(ncol(data)-1)], collapse = " + ")))
                                          self$model <- miss.glm(formula, data = data, print_iter = FALSE)
                                          
                                          TRUE
                                        },
                                        
                                        predict_probs = function(X_new, M_new) {
                                          # Prepare test data
                                          X_test <- as.data.frame(X_new)
                                          colnames(X_test) <- paste0("X", 1:ncol(X_new))
                                          
                                          # Get predictions
                                          pred_probs <- predict(self$model, newdata = X_test, type = "response", mcmc_map=500)

                                          return(pred_probs)
                                        },
                                        
                                        return_params = function() {
                                          if (!self$return_beta) return(NULL)
                                          
                                          # Extract coefficients
                                          coef_summary <- summary(self$model)$coef
                                          coefficients <- coef_summary[-1, "Estimate"]  # All except intercept
                                          intercept <- coef_summary[1, "Estimate"]     # Intercept only
                                          
                                          # Remove names from the vectors
                                          names(coefficients) <- NULL
                                          names(intercept) <- NULL
                                          
                                          # Create the exact string format to match Python output
                                          coef_str <- paste(coefficients, collapse = ", ")
                                          int_str <- as.character(intercept)
                                          
                                          return(sprintf("[[%s], [%s]]", coef_str, int_str))
                                        }
                                      )
)

MeanImputationLogisticRegression <- R6::R6Class("MeanImputationLogisticRegression",
  inherit = ImputationMethod,
  public = list(
    column_means = NULL,
    mask = FALSE,

    initialize = function(name = "Mean", mask = FALSE) {
      super$initialize(name)
      self$mask <- mask
    },

    fit = function(X, M, y, X_test = NULL, M_test = NULL) {
      data_train <- as.data.frame(X)

      self$column_means <- list()
      for (col_name in names(data_train)) {
        self$column_means[[col_name]] <- mean(data_train[[col_name]], na.rm = TRUE)
        if (any(is.na(data_train[[col_name]]))) {
          data_train[[col_name]][is.na(data_train[[col_name]])] <- self$column_means[[col_name]]
        }
      }

      if (self$mask) {
        colnames(M) <- paste0("M", seq_len(ncol(M)))
        data_train <- cbind(data_train, as.data.frame(M))
      }

      formula <- as.formula(paste("y ~", paste(names(data_train), collapse = " + ")))
      self$model <- glm(formula, family = binomial(), data = cbind(data_train, y = y))
      TRUE
    },

    predict_probs = function(X_new, M_new) {
      data_new <- as.data.frame(X_new)

      for (col_name in names(data_new)) {
        if (any(is.na(data_new[[col_name]]))) {
          if (!is.null(self$column_means[[col_name]])) {
            data_new[[col_name]][is.na(data_new[[col_name]])] <- self$column_means[[col_name]]
          } else {
            data_new[[col_name]][is.na(data_new[[col_name]])] <- mean(data_new[[col_name]], na.rm = TRUE)
            cat("Warning: No mean stored for column", col_name, "- using mean of new data.\n")
          }
        }
      }

      if (self$mask) {
        colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))
        data_new <- cbind(data_new, as.data.frame(M_new))
      }

      return(predict(self$model, newdata = data_new, type = "response"))
    },

    return_params = function() {
      if (!self$return_beta) return(NULL)

      # Get coefficients from the fitted glm model
      # For glm, coef(self$model) directly gives the coefficients
      # The first element is the intercept, followed by other coefficients
      model_coef <- coef(self$model)

      # Separate intercept and coefficients
      intercept <- model_coef[1]  # First coefficient is intercept in R
      coefficients <- model_coef[-1] # All other coefficients

      # Remove names from the vectors
      names(intercept) <- NULL
      names(coefficients) <- NULL

      # Create the exact string format to match Python output
      coef_str <- paste(coefficients, collapse = ", ")
      int_str <- as.character(intercept)

      return(sprintf("[[%s], [%s]]", coef_str, int_str))
    }
  )
)

ConstantImputationLogisticRegression <- R6::R6Class("ConstantImputationLogisticRegression",
  inherit = ImputationMethod,
  public = list(
    fill_value = 0, 
    mask = FALSE,

    initialize = function(name = "0.IMP", fill_value = 0, mask = FALSE) {
      super$initialize(name)
      self$fill_value <- fill_value
      self$mask <- mask
    },

    fit = function(X, M, y, X_test = NULL, M_test = NULL) {
      data_train <- as.data.frame(X)

      # Impute missing values with the constant fill_value
      for (col_name in names(data_train)) {
        if (any(is.na(data_train[[col_name]]))) {
          data_train[[col_name]][is.na(data_train[[col_name]])] <- self$fill_value
        }
      }

      # Add mask M to covariates if 'mask' is TRUE
      if (self$mask) {
        colnames(M) <- paste0("M", seq_len(ncol(M)))
        data_train <- cbind(data_train, as.data.frame(M))
      }

      formula <- as.formula(paste("y ~", paste(names(data_train), collapse = " + ")))
      self$model <- glm(formula, family = binomial(), data = cbind(data_train, y = y))
      TRUE
    },

    predict_probs = function(X_new, M_new) {
      data_new <- as.data.frame(X_new)

      for (col_name in names(data_new)) {
        if (any(is.na(data_new[[col_name]]))) {
          data_new[[col_name]][is.na(data_new[[col_name]])] <- self$fill_value
        }
      }

      if (self$mask) {
        colnames(M_new) <- paste0("M", seq_len(ncol(M_new)))
        data_new <- cbind(data_new, as.data.frame(M_new))
      }

      return(predict(self$model, newdata = data_new, type = "response"))
    },

    return_params = function() {
      if (!self$return_beta) return(NULL)

      model_coef <- coef(self$model)
      intercept <- model_coef[1]
      coefficients <- model_coef[-1]

      names(intercept) <- NULL
      names(coefficients) <- NULL

      coef_str <- paste(coefficients, collapse = ", ")
      int_str <- as.character(intercept)

      return(sprintf("[[%s], [%s]]", coef_str, int_str))
    }
  )
)