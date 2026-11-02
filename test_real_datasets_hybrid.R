source("real_datasets_hybrid_helpers.R")
stopifnot(auc_fixed(c(0,0,1,1), c(.1,.2,.8,.9)) == 1)
stopifnot(auc_fixed(c(0,0,1,1), c(.9,.8,.2,.1)) == 0)
stopifnot(auc_fixed(c(0,1), c(.5,.5)) == .5)
stopifnot(is.na(auc_fixed(c(1,1), c(.4,.8))))
pooled <- c(.3,.4,.6); specialist <- c(.1,NA,.9); ratio <- c(1,0,10)
h <- hybrid_predictions(pooled, specialist, ratio, Inf)
stopifnot(identical(h$prob, pooled), !any(h$use))
h <- hybrid_predictions(pooled, specialist, ratio, 2)
stopifnot(identical(h$prob,c(.3,.4,.9)), identical(h$use,c(FALSE,FALSE,TRUE)))
x <- data.frame(a=seq_len(18), b=factor(rep(c("a","b","c"),6)), missing=rep(NA_real_,18))
y <- rep(c(0,1,1,0,1,0),3)
prepared <- prepare_covariates(x,x)
p <- fit_pooled(prepared$train,prepared$test,y)
s <- fit_specialists(prepared$train,prepared$test,y)
stopifnot(all(is.finite(p$prob)), s$audit$Coefficients == 4L)
covers <- sapply(c(0,1,2,3,5,10,Inf),function(k)
  mean(hybrid_predictions(p$prob,s$prob,s$ratio,k)$use))
stopifnot(all(diff(covers)<=0),tail(covers,1)==0)
# Fully missing observations receive valid pooled probabilities.
blank <- data.frame(a=c(NA_real_,NA_real_))
sp <- fit_specialists(blank,blank,c(0,1)); pl <- fit_pooled(blank,blank,c(0,1))
stopifnot(all(is.na(sp$prob)), all(is.finite(pl$prob)))
# Global unseen level is train-only-imputed, not learned from test frequencies.
tr <- data.frame(a=rep(c("red","blue"),10), b=seq_len(20))
te <- data.frame(a=c("green","red"), b=c(5,6))
prepared <- prepare_covariates(tr,te)
stopifnot(is.na(prepared$test$a[1]))
p <- fit_pooled(prepared$train,prepared$test,rep(c(0,1,1,0),5))
stopifnot(length(p$prob)==2L, all(is.finite(p$prob)))
# A factor level seen globally but absent in this pattern uses the pooled model.
tr <- data.frame(a=factor(c("red","red","blue","blue","green","green")),
                 b=c(NA,NA,NA,NA,1,2))
te <- data.frame(a=factor("green",levels=levels(tr$a)), b=NA_real_)
sp <- fit_specialists(tr,te,c(0,1,0,1,0,1))
stopifnot(is.na(sp$prob[1]))
cat("Hybrid checks passed: AUC orientation/ties, endpoints, routing, coefficient counts, monotonic coverage and categorical edge cases.\n")
if (requireNamespace("glmnet", quietly=TRUE)) {
  # Ridge remains finite under complete separation; high penalties shrink slopes.
  x <- data.frame(z=seq(-2, 2, length.out=40))
  y <- as.numeric(x$z > 0)
  fit <- capture_ridge(x, y, seed=7L)
  stopifnot(!inherits(fit$fit, "error"), fit$fit$selection == "training_cv")
  prob <- capture_prediction(fit$fit, x)$prob
  stopifnot(all(is.finite(prob)), all(prob > 0 & prob < 1), auc_fixed(y,prob)==1)
  # Very small patterns use an explicit fixed penalty, without test tuning.
  small <- capture_ridge(x[1:4,,drop=FALSE], c(0,1,0,1))
  stopifnot(small$fit$selection == "fixed_small_pattern", small$fit$lambda == 1)
  tiny <- capture_ridge(data.frame(z=c(-1,1)), c(0,1))
  tiny_prob <- capture_prediction(tiny$fit, data.frame(z=c(-1,1)))$prob
  stopifnot(all(is.finite(tiny_prob)), tiny_prob[1] < .5, tiny_prob[2] > .5)
  # Check the direct small-pattern solver agrees with glmnet's ridge objective.
  xx <- data.frame(z=seq(-2,2,length.out=10))
  yy <- rep(c(0,1),5)
  small <- capture_ridge(xx, yy)
  reference <- glmnet::glmnet(cbind(z=xx$z, padding=0), yy, family="binomial",
                              alpha=0, lambda=1, standardize=TRUE, thresh=1e-12)
  reference_prob <- as.numeric(predict(reference,newx=cbind(z=xx$z,padding=0),s=1,type="response"))
  stopifnot(max(abs(capture_prediction(small$fit,xx)$prob-reference_prob)) < 1e-4)
  # Intercept-only specialists require no penalised slope or glmnet matrix.
  xx <- data.frame(z=rep(1,4))
  special <- fit_specialists(xx,xx,c(0,1,0,1),method="ridge")
  stopifnot(all(special$prob==.5), special$audit$Coefficients==1L)
  # Unsupported within-pattern categorical levels still retain pooled fallback.
  sp <- fit_specialists(tr,te,c(0,1,0,1,0,1),method="ridge")
  stopifnot(is.na(sp$prob[1]))
  cat("Ridge checks passed: separation, one-feature padding, training CV, tiny-pattern default, intercept-only and unseen-level handling.\n")
}
