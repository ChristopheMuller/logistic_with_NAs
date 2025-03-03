

library(reticulate)
library(dplyr)
library(stringr)


# remotes::install_local("~/INRIA/R_scripts/misaem_fork/", force=TRUE)


source("methods_in_R.R")
# reticulate::use_python(Sys.which("python3"))
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

data_bayes <- np$load(file.path("data", exp, "bayes_data", "LOG_n315000_d3_corr065_prcNA035_prop105_rep0.npz"))
y_bayes <- data_bayes$f[["y_probs_bayes"]]

# Load the true beta

true_beta <- c(0, 1.62434536, -0.61175641, -0.52817175)
true_sigma <- toeplitz(c(1, 0.65, 0.65^2))
true_mu <- c(0, 0, 0)


n_train <- 1000

X_train <- X_obs[1:n_train,]
M_train <- M[1:n_train,]
print(sum(M_train)/length(M_train))
y_train <- y[1:n_train]


data_train <- as.data.frame(X_train)
data_train$y <- y_train


all_predictions <- list()
for (i in 1:1){

  model <- misaem.fork::miss.glm("y ~ .", data=data_train, control=list(nmcmc=5, maxruns=20, k1=1, tau=9999),
                                 init_params = list(beta=true_beta, 
                                                    Sigma=true_sigma, 
                                                    mu=true_mu))
  y.hat <- predict(model, newdata = as.data.frame(X_test)[1:100,], mcmc_map=500)
  all_predictions[[i]] <- y.hat
  print(i)
}

y.hat <- do.call(cbind, all_predictions)
y.hat <- rowMeans(y.hat)
mean(abs(y.hat - y_bayes[1:100]))

model$trace$mu
mean(abs(y.hat[,1] - y_bayes[1:100]))




library(ggplot2)
library(dplyr)
library(tidyr)

# Number of iterations
num_iter <- length(model$trace$X.iter)

# Compute mean at each iteration
mean_evolution <- sapply(1:num_iter, function(k) {
  colMeans(model$trace$X.iter[[k]])  # Compute column means
})

# Convert to a tidy dataframe for ggplot
df_plot <- data.frame(t(mean_evolution)) %>%
  mutate(iteration = 1:num_iter) %>%
  pivot_longer(cols = -iteration, names_to = "variable", values_to = "mean")

# Plot the evolution
ggplot(df_plot, aes(x = iteration, y = mean, color = variable)) +
  geom_line() +
  labs(title = "Evolution of Column Means Over Iterations",
       x = "Iteration",
       y = "Mean Value") +
  theme_minimal()


library(ggplot2)
library(dplyr)
library(tidyr)

# Number of iterations
num_iter <- length(model$trace$X.iter)

# Compute covariance at each iteration
cov_evolution <- sapply(1:num_iter, function(k) {
  Sigma <- cov(model$trace$X.iter[[k]])  # Compute covariance matrix
  c(Sigma[1,1], Sigma[2,2], Sigma[3,3], Sigma[1,2], Sigma[2,3], Sigma[1,3])  # Extract elements
})

# Convert to a tidy dataframe for ggplot
df_cov <- data.frame(t(cov_evolution)) %>%
  mutate(iteration = 1:num_iter) %>%
  pivot_longer(cols = -iteration, names_to = "cov_element", values_to = "value")

# Rename elements for readability
df_cov$cov_element <- factor(df_cov$cov_element, 
                             labels = c("Sigma11", "Sigma22", "Sigma33", "Sigma12", "Sigma23", "Sigma13"))

# Plot the evolution
ggplot(df_cov, aes(x = iteration, y = value, color = cov_element)) +
  geom_line() +
  labs(title = "Evolution of Covariance Matrix Elements",
       x = "Iteration",
       y = "Value") +
  theme_minimal()



