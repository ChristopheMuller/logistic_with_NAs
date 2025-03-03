

results <- readRDS("~/INRIA/R_scripts/logistic_with_NAs/data/ExpC/SAEM_deepdive_tau08_500mcmc.RDS")

true_beta <- c(0, 1.62434536, -0.61175641, -0.52817175)
true_sigma <- toeplitz(c(1, 0.65, 0.65^2))
true_mu <- c(0, 0, 0)

## Plot the results

# Plot the running time

running_time <- unlist(results$runnint_time)
training_sizes <- as.numeric(names(running_time))

plot(training_sizes, running_time, type="b", xlab="Training size", ylab="Running time (s)", main="Running time vs Training size", log="x")

(max_time <- max(running_time) / 60)

# Plot the beta estimation

beta_estimated <- results$beta_estimated
beta_estimated <- do.call(rbind, beta_estimated)
beta_estimated <- t(beta_estimated)

plot(true_beta, col="red", type="b", xlab="Beta index", ylab="Beta value", main="Beta estimation")
lines(beta_estimated[,1], col="blue", type="b")
lines(beta_estimated[,2], col="blue", type="b")
lines(beta_estimated[,3], col="blue", type="b")
lines(beta_estimated[,4], col="blue", type="b")
lines(beta_estimated[,5], col="blue", type="b")
lines(beta_estimated[,6], col="blue", type="b")
lines(beta_estimated[,7], col="blue", type="b")
lines(beta_estimated[,8], col="blue", type="b")
lines(beta_estimated[,9], col="blue", type="b")

mean_squared_errors <- apply(beta_estimated, 2, function(x) mean((x - true_beta)^2))
plot(training_sizes, mean_squared_errors, type="b", xlab="Beta index", ylab="Mean squared error", main="Mean squared error for beta estimation", log="x")
lines(x=training_sizes, rep(0, length(training_sizes)), col="red")

mean_abs_errors <- apply(beta_estimated, 2, function(x) mean(abs(x - true_beta)))
plot(training_sizes, mean_abs_errors, type="b", xlab="Beta index", ylab="Mean absolute error", main="Mean absolute error for beta estimation", log="x")
lines(x=training_sizes, rep(0, length(training_sizes)), col="red")

# Plot the mu estimation

mu_estimated <- results$mu_estimated
mu_estimated <- do.call(rbind, mu_estimated)
mu_estimated <- t(mu_estimated)

plot(true_mu, col="red", type="b", xlab="Mu index", ylab="Mu value", main="Mu estimation")
lines(mu_estimated[,1], col="blue", type="b")
lines(mu_estimated[,2], col="blue", type="b")
lines(mu_estimated[,3], col="blue", type="b")
lines(mu_estimated[,4], col="blue", type="b")
lines(mu_estimated[,5], col="blue", type="b")
lines(mu_estimated[,6], col="blue", type="b")
lines(mu_estimated[,7], col="blue", type="b")
lines(mu_estimated[,8], col="blue", type="b")
lines(mu_estimated[,9], col="blue", type="b")

mean_squared_errors <- apply(mu_estimated, 2, function(x) mean((x - true_mu)^2))
plot(training_sizes, mean_squared_errors, type="b", xlab="Mu index", ylab="Mean squared error", main="Mean squared error for mu estimation", log="x")
lines(x=training_sizes, rep(0, length(training_sizes)), col="red")

mean_abs_errors <- apply(mu_estimated, 2, function(x) mean(abs(x - true_mu)))
plot(training_sizes, mean_abs_errors, type="b", xlab="Mu index", ylab="Mean absolute error", main="Mean absolute error for mu estimation", log="x")
lines(x=training_sizes, rep(0, length(training_sizes)), col="red")

# Plot the sigma estimation

sigma_estimated_all <-results$sigma_estimated
# restructure to get vector of S11, S22, S33, S12, S23, S13
sigma_estimated <- lapply(sigma_estimated_all, function(x) c(x[1,1], x[2,2], x[3,3], x[1,2], x[2,3], x[1,3]))
sigma_estimated <- do.call(rbind, sigma_estimated)
sigma_estimated <- t(sigma_estimated)

true_sigma_vec <- c(1, 1, 1, 0.65, 0.65, 0.65**2)

plot(true_sigma_vec, col="red", type="b", xlab="Sigma index", ylab="Sigma value", main="Sigma estimation")
lines(sigma_estimated[,1], col="blue", type="b")
lines(sigma_estimated[,2], col="blue", type="b")
lines(sigma_estimated[,3], col="blue", type="b")
lines(sigma_estimated[,4], col="blue", type="b")
lines(sigma_estimated[,5], col="blue", type="b")
lines(sigma_estimated[,6], col="blue", type="b")
lines(sigma_estimated[,7], col="blue", type="b")
lines(sigma_estimated[,8], col="blue", type="b")
lines(sigma_estimated[,9], col="blue", type="b")

mean_squared_errors <- apply(sigma_estimated, 2, function(x) mean((x - true_sigma_vec)^2))
plot(training_sizes, mean_squared_errors, type="b", xlab="Sigma index", ylab="Mean squared error", main="Mean squared error for sigma estimation", log="x")
lines(x=training_sizes, rep(0, length(training_sizes)), col="red")

mean_abs_errors <- apply(sigma_estimated, 2, function(x) mean(abs(x - true_sigma_vec)))
plot(training_sizes, mean_abs_errors, type="b", xlab="Sigma index", ylab="Mean absolute error", main="Mean absolute error for sigma estimation", log="x")
lines(x=training_sizes, rep(0, length(training_sizes)), col="red")

Frobenius_norm_fun <- function(x, y) sqrt(sum((x - y)^2))
frobenius_norms <- c()
for (train_size in training_sizes){
  frobenius_norms <- c(frobenius_norms, Frobenius_norm_fun(sigma_estimated_all[[as.character(train_size)]], true_sigma))
}
plot(training_sizes, frobenius_norms, type="b", xlab="Training size", ylab="Frobenius norm", main="Frobenius norm for sigma estimation", log="x")
lines(x=training_sizes, rep(0, length(training_sizes)), col="red")

