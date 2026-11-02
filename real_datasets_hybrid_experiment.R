# Standalone base-R runner; does not modify the existing benchmark outputs.
# Rscript --vanilla real_datasets_hybrid_experiment.R
# Optional: --datasets=boys,airquality --folds=2 --out=real_datasets_results/hybrid_smoke
source("real_datasets_hybrid_helpers.R")
args <- commandArgs(trailingOnly=TRUE)
option <- function(name, default) {
  match <- args[startsWith(args, paste0("--", name, "="))]
  if (length(match)) sub(paste0("^--", name, "="), "", tail(match, 1)) else default
}
n_folds <- as.integer(option("folds", "15"))
stopifnot(n_folds >= 1L, n_folds <= 15L)
with_ridge <- identical(option("ridge", "false"), "true")
if (with_ridge && !requireNamespace("glmnet", quietly=TRUE)) stop("Ridge experiment requires glmnet")
out <- option("out", if (with_ridge) "real_datasets_results/hybrid_ridge_threshold" else
                "real_datasets_results/hybrid_threshold")
dir.create(file.path(out, "datasets"), recursive=TRUE, showWarnings=FALSE)
k_values <- c(0, 0.5, 1, 2, 3, 4, 5, 7, 10, 30, Inf)
targets <- c(airquality="Wind", boys="age", colic="outcome", debt="prodebt",
             diabetes="Class", globwarm="chesapeake", housevotes84="Class",
             NHANES="Age", oceanbuoys="wind_ns", Ozone="V13",
             pedestrian="sensor_id", popmis="teachpop", pulplignin="Y.Kappa",
             SBS5242="USB", selfreport="sex", sleep="Danger", soybean="Class",
             tbc="sex", vnf="Q8.1", walking="sex")
cutoffs <- c(airquality=9.7, boys=10.5045, colic=2, debt=3.24, diabetes=0,
             globwarm=-0.48, housevotes84=1, NHANES=36, oceanbuoys=2.9,
             Ozone=110, pedestrian=10, popmis=4, pulplignin=20.74,
             SBS5242=4.779415, selfreport=1.5, sleep=2, soybean=7, tbc=1,
             vnf=1, walking=1)
sources <- list(
  read.csv("real_datasets_results/real_datasets_metrics_detailed.csv"),
  read.csv("tabzilla/tabzilla_real_datasets_results/real_datasets_metrics_detailed.csv"))
manifest <- rbind(data.frame(Dataset=unique(sources[[1]]$Dataset), Source=1L),
                  data.frame(Dataset=setdiff(unique(sources[[2]]$Dataset), "colic"), Source=2L))
selected <- option("datasets", "all")
if (selected != "all") {
  requested <- strsplit(selected, ",", fixed=TRUE)[[1]]
  stopifnot(all(requested %in% manifest$Dataset))
  manifest <- manifest[manifest$Dataset %in% requested, , drop=FALSE]
}
all_metrics <- all_audit <- all_status <- list()
for (i in seq_len(nrow(manifest))) {
  name <- manifest$Dataset[i]; source_id <- manifest$Source[i]
  cat(sprintf("[%d/%d] %s\n", i, nrow(manifest), name)); flush.console()
  dataset_result <- tryCatch({
    if (source_id == 1L) {
      stem <- name; raw_path <- paste0("real_datasets/", name, ".RDS")
      pred_dir <- "real_datasets_results/preds"
      target <- targets[[name]]; cutoff <- cutoffs[[name]]
    } else {
      files <- list.files("tabzilla/tabzilla_real_datasets", full.names=TRUE)
      raw_path <- if (name == "Wisconsin") files[grepl("Wisconsin-breast", files, fixed=TRUE)] else
        files[startsWith(basename(files), paste0("openml__", name, "__"))]
      stopifnot(length(raw_path) == 1L)
      stem <- tools::file_path_sans_ext(basename(raw_path))
      pred_dir <- "tabzilla/tabzilla_real_datasets_results/preds"
      target <- "target"; cutoff <- 0
    }
    dat <- readRDS(raw_path)
    dat <- dat[!is.na(dat[[target]]), , drop=FALSE]
    y <- as.numeric(as.numeric(dat[[target]]) <= cutoff)
    set.seed(sum(utf8ToInt(stem)))
    permutation <- sample(seq_len(nrow(dat)), nrow(dat), replace=FALSE)
    x <- dat[permutation, setdiff(names(dat), target), drop=FALSE]; y <- y[permutation]
    pred_path <- file.path(pred_dir, paste0(stem, "_mc_results.RDS"))
    y_path <- file.path(pred_dir, paste0(stem, "_Y.RDS"))
    if (file.exists(y_path)) stopifnot(identical(as.logical(y), as.logical(readRDS(y_path))))
    saved <- if (file.exists(pred_path)) readRDS(pred_path) else NULL
    dataset_metrics <- dataset_audit <- dataset_status <- saved_predictions <- list()
    for (fold in seq_len(n_folds)) {
      split_origin <- if (is.null(saved)) "generated_seed" else "saved_indices"
      candidates <- if (!is.null(saved)) Filter(function(z) z$fold == fold, saved) else list()
      if (length(candidates)) {
        test <- candidates[[1]]$indices
        stopifnot(all(vapply(candidates, function(z) identical(z$indices, test), logical(1))))
        stopifnot(all(vapply(candidates, function(z) identical(as.logical(y[test]), as.logical(z$truth)), logical(1))))
      } else {
        split_origin <- "generated_seed"
        set.seed(sum(utf8ToInt(stem)) + fold)
        test <- sample(seq_len(nrow(x)), floor(0.2*nrow(x)), replace=FALSE)
      }
      train <- setdiff(seq_len(nrow(x)), test)
      result <- tryCatch({
        prepared <- prepare_covariates(x[train, , drop=FALSE], x[test, , drop=FALSE])
        pooled <- fit_pooled(prepared$train, prepared$test, y[train])
        pbp <- fit_specialists(prepared$train, prepared$test, y[train])
        specialists <- list(Hybrid=pbp)
        if (with_ridge) specialists[["Ridge hybrid"]] <- fit_specialists(
          prepared$train, prepared$test, y[train], method="ridge",
          seed=sum(utf8ToInt(stem))+fold)
        base_auc <- auc_fixed(y[test], pooled$prob)
        metrics <- lapply(names(specialists), function(variant) {
          specialist <- specialists[[variant]]
          do.call(rbind, lapply(k_values, function(k) {
          hybrid <- hybrid_predictions(pooled$prob, specialist$prob, specialist$ratio, k)
          auc <- auc_fixed(y[test], hybrid$prob)
          if (is.infinite(k)) stopifnot(identical(hybrid$prob, pooled$prob), !any(hybrid$use))
          data.frame(Dataset=name, Fold=fold, Variant=variant, k=k, AUC_mean=base_auc, AUC_hybrid=auc,
                     AUC_difference=auc-base_auc, Specialisation_coverage=mean(hybrid$use),
                     Brier_mean=mean((pooled$prob-y[test])^2),
                     Brier_hybrid=mean((hybrid$prob-y[test])^2),
                     TrainN=length(train), TestN=length(test), SplitOrigin=split_origin)
          }))
        })
        audit <- do.call(rbind, lapply(names(specialists), function(variant)
          cbind(Dataset=name, Fold=fold, Variant=variant, specialists[[variant]]$audit)))
        list(metrics=do.call(rbind, metrics), audit=audit,
             predictions=list(fold=fold, test_original_rows=permutation[test],
                              train_original_rows=permutation[train], truth=y[test],
                              pooled=pooled$prob, specialist=pbp$prob, support_ratio=pbp$ratio,
                              ridge_specialist=if (with_ridge) specialists[["Ridge hybrid"]]$prob else NULL),
             pooled=pooled)
      }, error=function(e) e)
      if (inherits(result, "error")) {
        dataset_status[[length(dataset_status)+1L]] <- data.frame(
          Dataset=name, Fold=fold, Status="failed", SplitOrigin=split_origin,
          PooledConverged=NA, Warnings="", Message=conditionMessage(result))
      } else {
        dataset_metrics[[length(dataset_metrics)+1L]] <- result$metrics
        dataset_audit[[length(dataset_audit)+1L]] <- result$audit
        saved_predictions[[length(saved_predictions)+1L]] <- result$predictions
        dataset_status[[length(dataset_status)+1L]] <- data.frame(
          Dataset=name, Fold=fold, Status="ok", SplitOrigin=split_origin,
          PooledConverged=result$pooled$converged, Warnings=result$pooled$warnings,
          Message=paste("Dropped pooled columns:", result$pooled$dropped))
      }
    }
    list(metrics=do.call(rbind, dataset_metrics), audit=do.call(rbind, dataset_audit),
         status=do.call(rbind, dataset_status), predictions=saved_predictions,
         metadata=list(source=raw_path, md5=tools::md5sum(raw_path), k=k_values,
                       folds=n_folds, row_permutation=permutation,
                       character_handling="Training-level factors; unknown test levels become missing",
                       warnings_policy="GLM warnings/nonconvergence logged, fits retained like existing PbP",
                       ridge=with_ridge, ridge_cv="deviance; n>=12 and both class counts>=3; otherwise lambda=1"))
  }, error=function(e) list(metrics=NULL, audit=NULL, predictions=NULL,
       status=data.frame(Dataset=name, Fold=NA_integer_, Status="dataset_failed",
                         SplitOrigin="", PooledConverged=NA, Warnings="", Message=conditionMessage(e))))
  # Per-dataset checkpoint; existing benchmark results are never overwritten.
  saveRDS(dataset_result, file.path(out, "datasets", paste0(name, ".rds")))
  all_metrics[[length(all_metrics)+1L]] <- dataset_result$metrics
  all_audit[[length(all_audit)+1L]] <- dataset_result$audit
  all_status[[length(all_status)+1L]] <- dataset_result$status
  metrics <- do.call(rbind, all_metrics)
  if (!is.null(metrics)) write.csv(metrics, file.path(out, "metrics.csv"), row.names=FALSE)
  audit <- do.call(rbind, all_audit)
  if (!is.null(audit)) write.csv(audit, file.path(out, "pattern_audit.csv"), row.names=FALSE)
  write.csv(do.call(rbind, all_status), file.path(out, "fit_status.csv"), row.names=FALSE)
}
writeLines(c(paste("ExpectedFolds", n_folds, sep=","),
             paste("RequestedDatasets", nrow(manifest), sep=",")), file.path(out, "run_config.csv"))
cat("Finished. Outputs:", out, "\n")
