library(mice)
library(dplyr)
library(stringr)
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

# MICE implementation with logistic regression
MICELogisticRegression <- R6::R6Class("MICELogisticRegression",
                                      inherit = ImputationMethod,
                                      public = list(
                                        n_imputations = 5,
                                        maxit = 5,
                                        
                                        initialize = function(name, n_imputations = 5, maxit = 5) {
                                          super$initialize(name)
                                          self$n_imputations <- n_imputations
                                          self$maxit <- maxit
                                        },
                                        
                                        fit = function(X, M, y) {
                                          # Combine data for imputation
                                          data <- as.data.frame(X)
                                          data$y <- y
                                          
                                          # Run MICE
                                          self$imputation_model <- mice(data, m = self$n_imputations, maxit = self$maxit, printFlag = FALSE)
                                          
                                          # Fit logistic regression on each imputed dataset
                                          models <- list()
                                          for(i in 1:self$n_imputations) {
                                            imp_data <- complete(self$imputation_model, i)
                                            X_imp <- imp_data[, 1:(ncol(imp_data)-1)]
                                            y_imp <- imp_data$y
                                            
                                            # Fit logistic regression
                                            formula <- as.formula(paste("y ~", paste(colnames(X_imp), collapse = " + ")))
                                            models[[i]] <- glm(formula, family = binomial(), data = imp_data)
                                          }
                                          
                                          # Pool coefficients using Rubin's rules
                                          self$model <- models
                                          TRUE
                                        },
                                        
                                        predict_probs = function(X_new, M_new) {
                                          X_test <- as.data.frame(X_new)

                                          # Predict for each imputed model
                                          pred_probs <- matrix(0, nrow = nrow(X_new), ncol = self$n_imputations)
                                          
                                          for(i in 1:self$n_imputations) {
                                            # Use the same imputation model from training
                                            imp_test <- complete(self$imputation_model, i)
                                            pred_probs[,i] <- predict(self$model[[i]], newdata = X_test, type = "response")
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
                                        
                                        fit = function(X, M, y) {
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
                                          pred_probs <- predict(self$model, newdata = X_test, type = "response")
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
