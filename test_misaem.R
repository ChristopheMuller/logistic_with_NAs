
exp = "ExpC"

library(reticulate)
library(dplyr)
library(stringr)



df_set_up <- read.csv(file.path("data", exp, "set_up.csv"))


i = 1

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
true_intercept <- 0

# Load test data
data_test <- np$load(file.path("data", exp, "test_data", paste0(df_set_up$set_up[i], ".npz")))
X_test <- data_test$f[["X_obs"]]
M_test <- data_test$f[["M"]]
y_probs_test <- data_test$f[["y_probs"]]
y_test <- data_test$f[["y"]]

n_train <- 20000
X_train <- X_obs[1:n_train, ]
M_train <- M[1:n_train, ]
y_train <- y[1:n_train]

data <- as.data.frame(X_train)
colnames(data) <- paste0("X", 1:ncol(X_train))
data$y <- y_train

model <- misaem::miss.glm("y ~ .", data)

model$mu.X
df_set_up[i,5]
model$Sig.X
toeplitz(c(1,df_set_up[i,8], df_set_up[i,8]^2))
model$coefficients
df_set_up[i,4]

TrueBeta <- df_set_up[i,4]
vec <- as.numeric(unlist(strsplit(gsub("\\[|\\]", "", TrueBeta), " ")))
TrueBeta <- vec[!is.na(vec)]
TrueBeta <- c(true_intercept, TrueBeta)

BetaEst <- model$coefficients
BetaEstStdErr <- model$s.err

library(ggplot2)

# Create a data frame for plotting
df_beta <- data.frame(
  Beta = BetaEst,
  StdErr = BetaEstStdErr,
  TrueBeta = TrueBeta,
  Variable = names(BetaEst)  # Extract variable names
)

# Compute 95% Confidence Interval
df_beta$Lower <- df_beta$Beta - 1.96 * df_beta$StdErr
df_beta$Upper <- df_beta$Beta + 1.96 * df_beta$StdErr

# Plot
ggplot(df_beta, aes(x = Variable, y = Beta)) +
  geom_point(color = "blue", size = 3) +  # Estimated Beta
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "blue") +  # CI
  geom_point(aes(y = TrueBeta), color = "red", shape = 17, size = 3) +  # True Beta
  labs(title = "Beta Coefficients with 95% Confidence Intervals",
       x = "Variable",
       y = "Coefficient Estimate") +
  theme_minimal() +
  coord_flip()

(mae <- mean(abs(BetaEst - TrueBeta)))


