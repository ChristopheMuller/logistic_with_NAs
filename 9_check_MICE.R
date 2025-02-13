

source("methods_in_R.R")

library(dplyr)

set.seed(123)

d <- 5
n <- 65000
rho <- 0.95
cov <- toeplitz(rho^(0:(d-1)))
X1<- MASS::mvrnorm(n = n, mu = rep(0, d), Sigma = cov)
beta1 <- rnorm(d, 0, 1)

d <- 10
n <- 65000
rho <- 0.5
cov <- toeplitz(rho^(0:(d-1)))
X2<- MASS::mvrnorm(n = n, mu = rep(0, d), Sigma = cov)
beta2 <- c(beta1, rep(0, 5))

d <- 30
n <- 65000
rho <- 0.5
cov <- toeplitz(rho^(0:(d-1)))
X3<- MASS::mvrnorm(n = n, mu = rep(0, d), Sigma = cov)
beta3 <- c(beta1, rep(0, 25))


Y1 <- X1 %*% beta1
Y1 <- 1/(1 + exp(-Y1))
Y1 <- rbinom(n = 65000, size = 1, prob = Y1)
Y2 <- X2 %*% beta2
Y2 <- 1/(1 + exp(-Y2))
Y2 <- rbinom(n = 65000, size = 1, prob = Y2)
Y3 <- X3 %*% beta3
Y3 <- 1/(1 + exp(-Y3))
Y3 <- rbinom(n = 65000, size = 1, prob = Y3)

#######################

n_train <- c(100, 500, 1000, 5000)

X1_test <- X1[50001:65000,]
Y1_test <- Y1[50001:65000]

X2_test <- X2[50001:65000,]
Y2_test <- Y2[50001:65000]

X3_test <- X3[50001:65000,]
Y3_test <- Y3[50001:65000]

M_test <- matrix(0, nrow = 15000, ncol = 30)
for (i in 1:15000){
  M_test[i,] <- sample(c(0,1), 30, replace = TRUE, prob = c(0.7, 0.3))
}

X1_test_m <- X1_test
X1_test_m[M_test[,1:5] == 1] <- NA

X2_test_m <- X2_test
X2_test_m[M_test[,1:10] == 1] <- NA

X3_test_m <- X3_test
X3_test_m[M_test == 1] <- NA


scores <- data.frame(matrix(NA, nrow = 4, ncol = 3))
estimated_betas1 <- c()
estimated_betas2 <- c()
estimated_betas3 <- c()

for (i in 1:4){
  
  n_t <- n_train[i]
  
  X1_train <- X1[1:n_t,]
  Y1_train <- Y1[1:n_t]
  
  X2_train <- X2[1:n_t,]
  Y2_train <- Y2[1:n_t]
  
  X3_train <- X3[1:n_t,]
  Y3_train <- Y3[1:n_t]
  
  # create mask of missing values M, mcar 30%
  M <- matrix(0, nrow = n_t, ncol = 30)
  for (j in 1:n_t){
    M[j,] <- sample(c(0,1), 30, replace = TRUE, prob = c(0.7, 0.3))
  }
  M1 <- M[1:n_t,1:5]
  M2 <- M[1:n_t,6:10]
  M3 <- M[1:n_t,1:30]
  
  
  # create missing values
  X1_train_m <- X1_train
  X1_train_m[M1 == 1] <- NA
  
  X2_train_m <- X2_train
  X2_train_m[M2 == 1] <- NA
  
  X3_train_m <- X3_train
  X3_train_m[M3 == 1] <- NA
  
  
  # MICE METHOD 
  
  #1
  method <- MICELogisticRegression$new(name="MICE.IMP", n_imputations = 1)
  method$fit(X1_train_m, M1, Y1_train, X1_test_m, M_test[,1:5])
  y1_pred <- method$predict_probs(X1_test_m, M_test[,1:5])
  y1_pred <- ifelse(y1_pred > 0.5, 1, 0)
  acc1 <- sum(y1_pred == Y1_test)/15000
  estimated_beta1 <- method$return_params()
  
  
  #2
  method <- MICELogisticRegression$new(name="MICE.IMP", n_imputations = 1)
  method$fit(X2_train_m, M2, Y2_train, X2_test_m, M_test[,1:10])
  y2_pred <- method$predict_probs(X2_test_m, M_test[,1:10])
  y2_pred <- ifelse(y2_pred > 0.5, 1, 0)
  acc2 <- sum(y2_pred == Y2_test)/15000
  estimated_beta2 <- method$return_params()
  
  #3
  method <- MICELogisticRegression$new(name="MICE.IMP", n_imputations = 1)
  method$fit(X3_train_m, M3, Y3_train, X3_test_m, M_test)
  y3_pred <- method$predict_probs(X3_test_m, M_test)
  y3_pred <- ifelse(y3_pred > 0.5, 1, 0)
  acc3 <- sum(y3_pred == Y3_test)/15000
  estimated_beta3 <- method$return_params()
  
  scores[i,1] <- acc1
  scores[i,2] <- acc2
  scores[i,3] <- acc3
  
  estimated_betas1 <- c(estimated_betas1, estimated_beta1)
  estimated_betas2 <- c(estimated_betas2, estimated_beta2)
  estimated_betas3 <- c(estimated_betas3, estimated_beta3)
  
  print(i)
  
}


print(beta1)
print(estimated_betas1[4])

print(beta2)
print(estimated_betas2[4])

print(beta3)
print(estimated_betas3[4])

print(scores)


