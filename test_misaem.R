
#remotes::install_local("~/INRIA/R_scripts/misaem_fork", force=TRUE)
#remotes::install_github("wjiang94/misaem", force=TRUE)

# generate data

mu <- c(3,2,1)
sigma <- matrix(c(1,0.5,0.1,0.5,1,0.5,0.1,0.5,1), nrow=3, ncol=3)
beta <- c(-2,1,1)

n <- 100

X <- MASS::mvrnorm(n, mu, sigma)
y.logits <- X %*% beta
y.probs <- 1/(1+exp(-y.logits))
y <- rbinom(n, 1, y.probs)

# add some NAs in X, mcar with p = 0.20
M <- matrix(rbinom(n*3, 1, 0.2), nrow=n, ncol=3)
X[M == 1] <- NA
X_old <- X

# remove rows with full NAs
X <- X[!apply(is.na(X_old), 1, all),]
y <- y[!apply(is.na(X_old), 1, all)]


data <- data.frame(y, X)
formula <- as.formula(paste("y ~", paste0("X", 1:3, collapse = " + ")))


tic <- Sys.time()
misaem.fork.model <- misaem.fork::miss.glm(formula, data)
toc <- Sys.time()

(time.misaem.fork <- toc - tic)

tic <- Sys.time()
misaem.model <- misaem::miss.glm(formula, data)
toc <- Sys.time()

(time.misaem <- toc - tic)


X.new <- X[1:50,]
predict(misaem.model, X.new)

X.new <- X[1:50,]
library(misaem.fork)
predict(misaem.fork.model, X.new, method="map")
