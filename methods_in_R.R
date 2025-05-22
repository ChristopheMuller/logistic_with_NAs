library(mice)
library(dplyr)
library(stringr)
# library(misaem.fork)
library(CALIBERrfimpute)
library(misaem)

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
    mask = FALSE,
    add.y = FALSE,
    
    initialize = function(name, n_imputations = 5, maxit = 5, mask = FALSE, add.y = FALSE) {
      super$initialize(name)
      self$n_imputations <- n_imputations
      self$maxit <- maxit
      self$mask <- mask
      self$add.y <- add.y
    },
    
    fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
      # Combine training data for imputation
      data_train <- as.data.frame(X_train)
      
      # Add y to data if add.y is TRUE
      if (self$add.y) {
        data_train$y <- y_train
      }
      
      # Create ignore vector for MICE
      ignore_vec <- rep(FALSE, nrow(data_train))
      
      # If test set is provided
      if (!is.null(X_test)) {
        data_test <- as.data.frame(X_test)
        
        # Add y (as NA) to test data if add.y is TRUE
        if (self$add.y) {
          data_test$y <- NA
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
      
      # Fit logistic regression on each imputed training dataset
      models <- list()
      for(i in 1:self$n_imputations) {
        # Complete only the training data
        imp_train_data <- complete(self$imputation_model, i)[!ignore_vec, ]

        # Remove y if it was added during imputation
        if (self$add.y) {
          imp_train_data <- imp_train_data[, !names(imp_train_data) %in% "y"]
        }

        # Add mask before logistic regression if mask is TRUE
        if (self$mask) {
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
      # Predict for each imputed model
      pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
      
      for(i in 1:self$n_imputations) {
        # Complete the test data using the same imputation model
        imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
        
        # Remove y if it was added during imputation
        if (self$add.y) {
          imp_test <- imp_test[, !names(imp_test) %in% "y"]
        }
        
        # Add mask before prediction if mask is TRUE
        if (self$mask) {
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


MICECartLogisticRegression <- R6::R6Class("MICECartLogisticRegression",
                                      inherit = ImputationMethod,
                                      public = list(
                                        n_imputations = 5,
                                        maxit = 5,
                                        mask = FALSE,
                                        add.y = FALSE,
                                        
                                        initialize = function(name, n_imputations = 5, maxit = 5, mask = FALSE, add.y = FALSE) {
                                          super$initialize(name)
                                          self$n_imputations <- n_imputations
                                          self$maxit <- maxit
                                          self$mask <- mask
                                          self$add.y <- add.y
                                        },
                                        
                                        fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
                                          # Combine training data for imputation
                                          data_train <- as.data.frame(X_train)
                                          
                                          # Add y to data if add.y is TRUE
                                          if (self$add.y) {
                                            data_train$y <- y_train
                                          }
                                          
                                          # Create ignore vector for MICE
                                          ignore_vec <- rep(FALSE, nrow(data_train))
                                          
                                          # If test set is provided
                                          if (!is.null(X_test)) {
                                            data_test <- as.data.frame(X_test)
                                            
                                            # Add y (as NA) to test data if add.y is TRUE
                                            if (self$add.y) {
                                              data_test$y <- NA
                                            }
                                            
                                            # Combine train and test data
                                            data_full <- rbind(data_train, data_test)
                                            
                                            # Update ignore vector
                                            ignore_vec <- c(ignore_vec, rep(TRUE, nrow(data_test)))
                                            
                                            # Run MICE on full dataset
                                            self$imputation_model <- mice(data_full, m = self$n_imputations, 
                                                                          maxit = self$maxit, 
                                                                          printFlag = FALSE, 
                                                                          ignore = ignore_vec, method="cart")
                                          } else {
                                            # Run MICE only on training data
                                            self$imputation_model <- mice(data_train, m = self$n_imputations, 
                                                                          maxit = self$maxit, 
                                                                          printFlag = FALSE, method="cart")
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
                                            
                                            # Add mask before logistic regression if mask is TRUE
                                            if (self$mask) {
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
                                          # Predict for each imputed model
                                          pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
                                          
                                          for(i in 1:self$n_imputations) {
                                            # Complete the test data using the same imputation model
                                            imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
                                            
                                            # Remove y if it was added during imputation
                                            if (self$add.y) {
                                              imp_test <- imp_test[, !names(imp_test) %in% "y"]
                                            }
                                            
                                            # Add mask before prediction if mask is TRUE
                                            if (self$mask) {
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
                                            mask = FALSE,
                                            add.y = FALSE,
                                            
                                            initialize = function(name, n_imputations = 5, maxit = 5, mask = FALSE, add.y = FALSE) {
                                              super$initialize(name)
                                              self$n_imputations <- n_imputations
                                              self$maxit <- maxit
                                              self$mask <- mask
                                              self$add.y <- add.y
                                            },
                                            
                                            fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
                                              # Combine training data for imputation
                                              data_train <- as.data.frame(X_train)
                                              
                                              # Add y to data if add.y is TRUE
                                              if (self$add.y) {
                                                data_train$y <- y_train
                                              }
                                              
                                              # Create ignore vector for MICE
                                              ignore_vec <- rep(FALSE, nrow(data_train))
                                              
                                              # If test set is provided
                                              if (!is.null(X_test)) {
                                                data_test <- as.data.frame(X_test)
                                                
                                                # Add y (as NA) to test data if add.y is TRUE
                                                if (self$add.y) {
                                                  data_test$y <- NA
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
                                                
                                                # Add mask before logistic regression if mask is TRUE
                                                if (self$mask) {
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
                                              # Predict for each imputed model
                                              pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
                                              
                                              for(i in 1:self$n_imputations) {
                                                # Complete the test data using the same imputation model
                                                imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
                                                
                                                # Remove y if it was added during imputation
                                                if (self$add.y) {
                                                  imp_test <- imp_test[, !names(imp_test) %in% "y"]
                                                }
                                                
                                                # Add mask before prediction if mask is TRUE
                                                if (self$mask) {
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


MICECaliberLogisticRegression <- R6::R6Class("MICERFLogisticRegression",
                                        inherit = ImputationMethod,
                                        public = list(
                                          n_imputations = 5,
                                          maxit = 5,
                                          mask = FALSE,
                                          add.y = FALSE,
                                          
                                          initialize = function(name, n_imputations = 5, maxit = 5, mask = FALSE, add.y = FALSE) {
                                            super$initialize(name)
                                            self$n_imputations <- n_imputations
                                            self$maxit <- maxit
                                            self$mask <- mask
                                            self$add.y <- add.y
                                          },
                                          
                                          fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
                                            # Combine training data for imputation
                                            data_train <- as.data.frame(X_train)
                                            
                                            # Add y to data if add.y is TRUE
                                            if (self$add.y) {
                                              data_train$y <- y_train
                                            }
                                            
                                            # Create ignore vector for MICE
                                            ignore_vec <- rep(FALSE, nrow(data_train))
                                            
                                            # If test set is provided
                                            if (!is.null(X_test)) {
                                              data_test <- as.data.frame(X_test)
                                              
                                              # Add y (as NA) to test data if add.y is TRUE
                                              if (self$add.y) {
                                                data_test$y <- NA
                                              }
                                              
                                              # Combine train and test data
                                              data_full <- rbind(data_train, data_test)
                                              
                                              # Update ignore vector
                                              ignore_vec <- c(ignore_vec, rep(TRUE, nrow(data_test)))
                                              
                                              # Run MICE on full dataset
                                              self$imputation_model <- mice(data_full, m = self$n_imputations, 
                                                                            maxit = self$maxit, 
                                                                            printFlag = FALSE, 
                                                                            ignore = ignore_vec, method="rfcont")
                                            } else {
                                              # Run MICE only on training data
                                              self$imputation_model <- mice(data_train, m = self$n_imputations, 
                                                                            maxit = self$maxit, 
                                                                            printFlag = FALSE, method="rfcont")
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
                                              
                                              # Add mask before logistic regression if mask is TRUE
                                              if (self$mask) {
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
                                            # Predict for each imputed model
                                            pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
                                            
                                            for(i in 1:self$n_imputations) {
                                              # Complete the test data using the same imputation model
                                              imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
                                              
                                              # Remove y if it was added during imputation
                                              if (self$add.y) {
                                                imp_test <- imp_test[, !names(imp_test) %in% "y"]
                                              }
                                              
                                              # Add mask before prediction if mask is TRUE
                                              if (self$mask) {
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




MICERandomLogisticRegression <- R6::R6Class("MICERandomLogisticRegression",
                                          inherit = ImputationMethod,
                                          public = list(
                                            n_imputations = 5,
                                            maxit = 5,
                                            mask = FALSE,
                                            add.y = FALSE,
                                            
                                            initialize = function(name, n_imputations = 5, maxit = 5, mask = FALSE, add.y = FALSE) {
                                              super$initialize(name)
                                              self$n_imputations <- n_imputations
                                              self$maxit <- maxit
                                              self$mask <- mask
                                              self$add.y <- add.y
                                            },
                                            
                                            fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
                                              # Combine training data for imputation
                                              data_train <- as.data.frame(X_train)
                                              
                                              # Add y to data if add.y is TRUE
                                              if (self$add.y) {
                                                data_train$y <- y_train
                                              }
                                              
                                              # Create ignore vector for MICE
                                              ignore_vec <- rep(FALSE, nrow(data_train))
                                              
                                              # If test set is provided
                                              if (!is.null(X_test)) {
                                                data_test <- as.data.frame(X_test)
                                                
                                                # Add y (as NA) to test data if add.y is TRUE
                                                if (self$add.y) {
                                                  data_test$y <- NA
                                                }
                                                
                                                # Combine train and test data
                                                data_full <- rbind(data_train, data_test)
                                                
                                                # Update ignore vector
                                                ignore_vec <- c(ignore_vec, rep(TRUE, nrow(data_test)))
                                                
                                                # Run MICE on full dataset
                                                self$imputation_model <- mice(data_full, m = self$n_imputations, 
                                                                              maxit = self$maxit, 
                                                                              printFlag = FALSE, 
                                                                              ignore = ignore_vec, method="sample")
                                              } else {
                                                # Run MICE only on training data
                                                self$imputation_model <- mice(data_train, m = self$n_imputations, 
                                                                              maxit = self$maxit, 
                                                                              printFlag = FALSE, method="sample")
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
                                                
                                                # Add mask before logistic regression if mask is TRUE
                                                if (self$mask) {
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
                                              # Predict for each imputed model
                                              pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
                                              
                                              for(i in 1:self$n_imputations) {
                                                # Complete the test data using the same imputation model
                                                imp_test <- complete(self$imputation_model, i)[nrow(self$imputation_model$data) - nrow(X_new) + 1:nrow(X_new), ]
                                                
                                                # Remove y if it was added during imputation
                                                if (self$add.y) {
                                                  imp_test <- imp_test[, !names(imp_test) %in% "y"]
                                                }
                                                
                                                # Add mask before prediction if mask is TRUE
                                                if (self$mask) {
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





DRFLogisticRegression <- R6::R6Class("DRF",
                                          inherit = ImputationMethod,
                                          public = list(
                                            n_imputations = 5,
                                            fit_1 = NULL,
                                            fit_2 = NULL,

                                            initialize = function(name, n_imputations = 5, maxit = 5, mask = FALSE, add.y = FALSE) {
                                              super$initialize(name)
                                              self$n_imputations <- n_imputations
                                            },
                                            
                                            fit = function(X_train, M_train, y_train, X_test = NULL, M_test = NULL) {
                                              # Combine training data for imputation
                                              data_train <- as.data.frame(X_train)
                                              
                                              ### Fit DRF for each pattern

                                              idx_1_mis <- which(M_train[,1] == 1)
                                              idx_1_obs <- which(M_train[,1] == 0)
                                              idx_2_mis <- which(M_train[,2] == 1)
                                              idx_2_obs <- which(M_train[,2] == 0)
                                                                                            
                                              # fill missing values by mean
                                              data_train[idx_1_mis, 1] <- mean(data_train[, 1], na.rm = TRUE)
                                              data_train[idx_2_mis, 2] <- mean(data_train[, 2], na.rm = TRUE)
                                              
                                              data_train <- as.matrix(data_train)

                                              self$fit_1 <- drf::drf(Y = data_train[idx_1_obs,1],
                                                                X = matrix(data_train[idx_1_obs, -1]))
                                              
                                              self$fit_2 <- drf::drf(Y = data_train[idx_2_obs,2],
                                                                X = matrix(data_train[idx_2_obs, -2]))
                                              

                                              # samples n_imputations from each model
                                              xmis1 <- matrix(data_train[idx_1_mis, -1])
                                              yobs1 <- matrix(data_train[idx_1_obs, -1])
                                              imputed_1_weights <- predict(self$fit_1, newdata = xmis1)
                                              imputed_1 <- vapply(1:nrow(xmis1), function(s) {
                                                yobs1[sample(1:nrow(yobs1), size = self$n_imputations, replace = T, prob = imputed_1_weights$weights[s, ]), ]
                                              }, numeric(self$n_imputations))  # dim = (n_imptutations, nrow(xmis1))
                                              
                                              xmis2 <- matrix(data_train[idx_2_mis, -2])
                                              yobs2 <- matrix(data_train[idx_2_obs, -2])
                                              imputed_2_weights <- predict(self$fit_2, newdata = xmis2)
                                              imputed_2 <- vapply(1:nrow(xmis2), function(s) {
                                                yobs2[sample(1:nrow(yobs2), size = self$n_imputations, replace = T, prob = imputed_2_weights$weights[s, ]), ]
                                              }, numeric(self$n_imputations))  # dim = (n_imptutations, nrow(xmis2))
                                              
                                              
                                              
                                              # Fit logistic regression on each imputed training dataset
                                              models <- list()
                                              for(i in 1:self$n_imputations) {
                                                
                                                imp_train_data <- data_train
                                                imp_train_data[idx_1_mis, 1] <- imputed_1[i, ]
                                                imp_train_data[idx_2_mis, 2] <- imputed_2[i, ]
                                                
                                                imp_train_data <- as.data.frame(imp_train_data)
                                                names(imp_train_data) <- paste0("X", 1:ncol(imp_train_data))
                                                
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
                                              X_new <- as.data.frame(X_new)
                                              n <- nrow(X_new)
                                              
                                              # Determine missing indices
                                              idx_1_mis <- which(M_new[, 1] == 1)
                                              idx_2_mis <- which(M_new[, 2] == 1)
                                              
                                              # Fill in missing with means first, as in training
                                              X_new[idx_1_mis, 1] <- mean(X_new[, 1], na.rm = TRUE)
                                              X_new[idx_2_mis, 2] <- mean(X_new[, 2], na.rm = TRUE)
                                              
                                              X_new_mat <- as.matrix(X_new)
                                              
                                              # Use stored DRF fits to generate imputations
                                              xmis1 <- matrix(X_new_mat[idx_1_mis, -1])
                                              yobs1 <- matrix(self$fit_1$Y)
                                              weights1 <- predict(self$fit_1, newdata = xmis1)$weights
                                              imputed_1 <- vapply(1:nrow(xmis1), function(s) {
                                                yobs1[sample(1:nrow(yobs1), self$n_imputations, replace = TRUE, prob = weights1[s, ]), ]
                                              }, numeric(self$n_imputations))
                                              
                                              xmis2 <- matrix(X_new_mat[idx_2_mis, -2])
                                              yobs2 <- matrix(self$fit_2$Y)
                                              weights2 <- predict(self$fit_2, newdata = xmis2)$weights
                                              imputed_2 <- vapply(1:nrow(xmis2), function(s) {
                                                yobs2[sample(1:nrow(yobs2), self$n_imputations, replace = TRUE, prob = weights2[s, ]), ]
                                              }, numeric(self$n_imputations))
                                              
                                              # Predict using each model on the respective imputed dataset
                                              pred_probs <- matrix(0, nrow = n, ncol = self$n_imputations)
                                              
                                              for(i in 1:self$n_imputations) {
                                                imp_data <- X_new_mat
                                                if (length(idx_1_mis) > 0) imp_data[idx_1_mis, 1] <- imputed_1[i, ]
                                                if (length(idx_2_mis) > 0) imp_data[idx_2_mis, 2] <- imputed_2[i, ]
                                                
                                                imp_df <- as.data.frame(imp_data)
                                                names(imp_df) <- paste0("X", 1:ncol(imp_df))
                                                
                                                pred_probs[, i] <- predict(self$model[[i]], newdata = imp_df, type = "response")
                                              }
                                              
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

SAEM_MI <- R6::R6Class("SAEM_MI",
                       inherit = ImputationMethod,
                       public = list(
                         n_imputations = 5,
                         models = NULL,
                         
                         initialize = function(name, n_imputations = 5) {
                           super$initialize(name)
                           self$n_imputations <- n_imputations
                         },
                         
                         fit = function(X, M, y, X_test = NULL, M_test = NULL) {
                           # Convert data to required format
                           data <- as.data.frame(X)
                           colnames(data) <- paste0("X", 1:ncol(X))
                           data$y <- y
                           
                           # Create formula for the model
                           formula <- as.formula(paste("y ~", paste(colnames(data)[1:(ncol(data)-1)], collapse = " + ")))
                           
                           # Fit multiple SAEM models
                           self$models <- list()
                           for (i in 1:self$n_imputations) {
                             # Print progress
                             cat(sprintf("Fitting SAEM model %d of %d\n", i, self$n_imputations))
                             
                             # Fit SAEM model with different random seed for each iteration
                             set.seed(i)  # Different seed for each model
                             self$models[[i]] <- miss.glm(formula, data = data, print_iter = FALSE, 
                                                          control = list(tau = 1, maxruns = 750))
                           }
                           
                           # Store the first model as the main model for parameter extraction
                           self$model <- self$models[[1]]
                           
                           TRUE
                         },
                         
                         predict_probs = function(X_new, M_new) {
                           # Prepare test data
                           X_test <- as.data.frame(X_new)
                           colnames(X_test) <- paste0("X", 1:ncol(X_new))
                           
                           # Predict for each model and store in matrix
                           pred_matrix <- matrix(0, nrow = nrow(X_test), ncol = self$n_imputations)
                           
                           for (i in 1:self$n_imputations) {
                             pred_matrix[, i] <- predict(self$models[[i]], newdata = X_test, type = "response")
                           }
                           
                           # Return average predictions across all models
                           return(rowMeans(pred_matrix))
                         },
                         
                         return_params = function() {
                           if (!self$return_beta) return(NULL)
                           
                           # Extract coefficients from all models
                           all_coefs <- lapply(self$models, function(model) {
                             coef_summary <- summary(model)$coef
                             list(
                               coefficients = coef_summary[-1, "Estimate"],  # All except intercept
                               intercept = coef_summary[1, "Estimate"]      # Intercept only
                             )
                           })
                           
                           # Average coefficients across all models (Rubin's rules - simple average)
                           avg_coefs <- list(
                             coefficients = Reduce(`+`, lapply(all_coefs, function(x) x$coefficients)) / self$n_imputations,
                             intercept = sum(sapply(all_coefs, function(x) x$intercept)) / self$n_imputations
                           )
                           
                           # Remove names from the vectors
                           names(avg_coefs$coefficients) <- NULL
                           names(avg_coefs$intercept) <- NULL
                           
                           # Create the exact string format to match Python output
                           coef_str <- paste(avg_coefs$coefficients, collapse = ", ")
                           int_str <- as.character(avg_coefs$intercept)
                           
                           return(sprintf("[[%s], [%s]]", coef_str, int_str))
                         }
                       )
)
