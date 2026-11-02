# Base-R helpers for the PbP/pooled threshold experiment.
pattern_keys <- function(x) apply(is.na(x), 1L, paste, collapse="_")

auc_fixed <- function(y, prob) {
  stopifnot(length(y) == length(prob), all(is.finite(prob)))
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (!n1 || !n0) return(NA_real_)
  # Mann-Whitney form of AUC: higher probability always means class 1.
  (sum(rank(prob, ties.method="average")[y == 1]) - n1*(n1+1)/2) / (n1*n0)
}

capture_glm <- function(x, y) {
  warnings <- character()
  fit <- tryCatch(withCallingHandlers({
    if (ncol(x)) {
      glm(.outcome ~ ., data=cbind(x, .outcome=as.numeric(y)), family=binomial())
    } else {
      glm(.outcome ~ 1, data=data.frame(.outcome=as.numeric(y)), family=binomial())
    }
  }, warning=function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error=function(e) e)
  list(fit=fit, warnings=paste(unique(warnings), collapse=" | "))
}

capture_prediction <- function(fit, x) {
  warnings <- character()
  prob <- tryCatch(withCallingHandlers(
    if (inherits(fit, "pattern_ridge")) predict_pattern_ridge(fit, x) else
      predict(fit, newdata=x, type="response"),
    warning=function(w) {
      warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
    }), error=function(e) e)
  list(prob=prob, warnings=paste(unique(warnings), collapse=" | "))
}

capture_ridge <- function(x, y, seed=1L) {
  warnings <- character()
  fit <- tryCatch(withCallingHandlers({
    stopifnot(requireNamespace("glmnet", quietly=TRUE))
    tt <- if (ncol(x)) terms(~ ., data=x) else terms(~ 1)
    design <- if (ncol(x)) model.matrix(tt, x) else matrix(1, length(y), 1L,
                                                         dimnames=list(NULL,"(Intercept)"))
    contrasts <- attr(design, "contrasts")
    design <- design[, colnames(design) != "(Intercept)", drop=FALSE]
    xlevels <- lapply(x[vapply(x, is.factor, logical(1))], levels)
    if (!ncol(design)) {
      structure(list(model=NULL, constant=mean(y), lambda=NA_real_,
                     selection="intercept_only", converged=TRUE, xlevels=xlevels,
                     terms=tt, columns=character(), contrasts=contrasts, padded=FALSE),
                class="pattern_ridge")
    } else {
      columns <- colnames(design)
      padded <- ncol(design) == 1L
      if (padded) design <- cbind(design, .padding=0)
      lambda_grid <- 10^seq(2, -4, length.out=30)
      # CV is deliberately withheld for tiny patterns. Fixed lambda=1 is a
      # prespecified default on glmnet's average-loss scale, not test-tuned.
      can_cv <- length(y) >= 12L && min(table(y)) >= 3L
      if (can_cv) {
        nfolds <- min(5L, min(table(y)))
        set.seed(seed)
        foldid <- integer(length(y))
        for (label in unique(y)) {
          idx <- which(y == label)
          foldid[idx] <- sample(rep(seq_len(nfolds), length.out=length(idx)))
        }
        cv <- glmnet::cv.glmnet(design, y, family="binomial", alpha=0,
                               lambda=lambda_grid, standardize=TRUE, intercept=TRUE,
                               foldid=foldid, type.measure="deviance", maxit=100000)
        model <- cv$glmnet.fit; lambda <- cv$lambda.min; selection <- "training_cv"
      } else {
        lambda <- 1
        # glmnet refuses a class containing only one observation. Solve the
        # same mean-log-loss + ridge objective directly for small patterns.
        center <- colMeans(design)
        scale <- sqrt(colMeans(sweep(design, 2L, center)^2))
        scale[scale == 0] <- 1
        z <- sweep(sweep(design, 2L, center), 2L, scale, "/")
        z <- cbind(1, z)
        objective <- function(beta) {
          eta <- drop(z %*% beta)
          mean(pmax(eta, 0) + log1p(exp(-abs(eta))) - y*eta) +
            lambda*sum(beta[-1]^2)/2
        }
        gradient <- function(beta) {
          drop(crossprod(z, plogis(drop(z %*% beta))-y))/length(y) +
            c(0, lambda*beta[-1])
        }
        opt <- optim(c(qlogis(mean(y)), rep(0, ncol(design))), objective,
                     gradient, method="BFGS", control=list(maxit=10000, reltol=1e-10))
        model <- structure(list(beta=opt$par, center=center, scale=scale,
                                jerr=opt$convergence), class="small_pattern_ridge")
        selection <- "fixed_small_pattern"
      }
      structure(list(model=model, constant=NULL, lambda=lambda, selection=selection,
                     converged=model$jerr == 0L, xlevels=xlevels, terms=tt,
                     columns=columns, contrasts=contrasts, padded=padded), class="pattern_ridge")
    }
  }, warning=function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error=function(e) e)
  list(fit=fit, warnings=paste(unique(warnings), collapse=" | "))
}

predict_pattern_ridge <- function(fit, x) {
  if (!is.null(fit$constant)) return(rep(fit$constant, nrow(x)))
  frame <- model.frame(fit$terms, data=x, na.action=na.pass, xlev=fit$xlevels)
  design <- model.matrix(fit$terms, frame, contrasts.arg=fit$contrasts)
  design <- design[, fit$columns, drop=FALSE]
  if (fit$padded) design <- cbind(design, .padding=0)
  valid <- apply(design, 1L, function(row) all(is.finite(row)))
  prob <- rep(NA_real_, nrow(x))
  if (any(valid)) {
    if (inherits(fit$model, "small_pattern_ridge")) {
      z <- sweep(sweep(design[valid,,drop=FALSE], 2L, fit$model$center),
                 2L, fit$model$scale, "/")
      prob[valid] <- plogis(drop(cbind(1, z) %*% fit$model$beta))
    } else {
      prob[valid] <- as.numeric(predict(fit$model, newx=design[valid,,drop=FALSE],
                                       s=fit$lambda, type="response"))
    }
  }
  prob
}

prepare_covariates <- function(x_train, x_test) {
  # GLM encodes character covariates categorically too. Make this explicit.
  for (j in seq_along(x_train)) {
    if (is.character(x_train[[j]]) || is.factor(x_train[[j]])) {
      train_values <- as.character(x_train[[j]])
      test_values <- as.character(x_test[[j]])
      levels <- unique(train_values[!is.na(train_values)])
      x_train[[j]] <- factor(train_values, levels=levels)
      x_test[[j]] <- factor(test_values, levels=levels)
    }
  }
  # Internal safe names avoid formula parsing problems in source column names.
  names(x_train) <- names(x_test) <- paste0("feature_", seq_along(x_train))
  list(train=x_train, test=x_test)
}

fit_pooled <- function(x_train, x_test, y) {
  dropped <- character()
  for (name in names(x_train)) {
    col <- x_train[[name]]
    known <- col[!is.na(col)]
    if (!length(known)) { dropped <- c(dropped, name); next }
    if (is.numeric(col)) {
      value <- mean(known)
    } else {
      values <- unique(known)
      value <- values[which.max(tabulate(match(known, values)))]
    }
    x_train[[name]][is.na(col)] <- value
    x_test[[name]][is.na(x_test[[name]])] <- value
    if (length(unique(x_train[[name]])) <= 1L) dropped <- c(dropped, name)
  }
  keep <- setdiff(names(x_train), dropped)
  train <- droplevels(x_train[, keep, drop=FALSE])
  test <- x_test[, keep, drop=FALSE]
  result <- capture_glm(train, y)
  if (inherits(result$fit, "error")) stop(conditionMessage(result$fit))
  predicted <- capture_prediction(result$fit, test)
  if (inherits(predicted$prob, "error")) stop(conditionMessage(predicted$prob))
  if (length(predicted$prob) != nrow(test) || any(!is.finite(predicted$prob)))
    stop("Pooled model returned invalid probabilities; no constant replacement used")
  list(prob=as.numeric(predicted$prob), warnings=paste(result$warnings, predicted$warnings),
       converged=isTRUE(result$fit$converged), dropped=paste(dropped, collapse=";"))
}

fit_specialists <- function(x_train, x_test, y, method="glm", seed=1L) {
  stopifnot(method %in% c("glm", "ridge"))
  train_keys <- pattern_keys(x_train); test_keys <- pattern_keys(x_test)
  prob <- rep(NA_real_, nrow(x_test)); ratio <- rep(NA_real_, nrow(x_test))
  audit <- list()
  for (key in unique(train_keys)) {
    tr <- which(train_keys == key); te <- which(test_keys == key)
    observed <- !is.na(x_train[tr[1], , drop=FALSE])
    xp <- x_train[tr, as.logical(observed), drop=FALSE]
    keep <- vapply(xp, function(z) length(unique(z)) > 1L, logical(1))
    xp <- droplevels(xp[, keep, drop=FALSE])
    p <- tryCatch(if (ncol(xp)) ncol(model.matrix(~ ., xp)) else 1L,
                  error=function(e) NA_integer_)
    status <- "fitted"; message <- ""; warnings <- ""; converged <- NA
    lambda <- NA_real_; selection <- "none"
    if (!any(observed)) {
      status <- "no_observed_covariates"
    } else if (length(unique(y[tr])) < 2L) {
      status <- "single_class"
    } else if (is.na(p)) {
      status <- "design_error"
    } else {
      result <- if (method == "ridge") capture_ridge(xp, y[tr], seed=seed) else capture_glm(xp, y[tr])
      warnings <- result$warnings
      if (inherits(result$fit, "error")) {
        status <- "fit_error"; message <- conditionMessage(result$fit)
      } else {
        converged <- isTRUE(result$fit$converged)
        if (method == "ridge") {
          lambda <- result$fit$lambda; selection <- result$fit$selection
        }
        if (length(te)) {
          xt <- x_test[te, names(xp), drop=FALSE]
          # A categorical level can be seen globally but absent in this pattern.
          for (name in names(result$fit$xlevels)) {
            xt[[name]] <- factor(as.character(xt[[name]]), levels=result$fit$xlevels[[name]])
          }
          predicted <- capture_prediction(result$fit, xt)
          warnings <- paste(warnings, predicted$warnings)
          if (inherits(predicted$prob, "error")) {
            status <- "prediction_error"; message <- conditionMessage(predicted$prob)
          } else {
            available <- is.finite(predicted$prob) & predicted$prob >= 0 & predicted$prob <= 1
            prob[te[available]] <- predicted$prob[available]
            if (any(!available)) status <- "partial_prediction"
          }
        }
      }
    }
    if (length(te) && !is.na(p)) ratio[te] <- length(tr)/p
    audit[[length(audit)+1L]] <- data.frame(
      Pattern=key, TrainN=length(tr), TestN=length(te), Coefficients=p,
      SupportRatio=length(tr)/p, Status=status, Converged=converged,
      PredictionsAvailable=sum(is.finite(prob[te])), Warnings=warnings, Message=message,
      Lambda=lambda, PenaltySelection=selection)
  }
  unseen <- setdiff(unique(test_keys), unique(train_keys))
  for (key in unseen) {
    audit[[length(audit)+1L]] <- data.frame(
      Pattern=key, TrainN=0L, TestN=sum(test_keys == key), Coefficients=NA_integer_,
      SupportRatio=0, Status="unseen_pattern", Converged=NA,
      PredictionsAvailable=0L, Warnings="", Message="", Lambda=NA_real_, PenaltySelection="none")
  }
  list(prob=prob, ratio=ratio, audit=do.call(rbind, audit))
}

hybrid_predictions <- function(pooled, specialist, ratio, k) {
  use <- is.finite(specialist) & is.finite(ratio) & ratio >= k
  # Inf is the exact pooled endpoint even if support metadata are malformed.
  if (is.infinite(k)) use[] <- FALSE
  prob <- pooled; prob[use] <- specialist[use]
  list(prob=prob, use=use)
}
