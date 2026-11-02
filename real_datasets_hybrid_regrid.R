# Re-evaluate saved predictions at new thresholds, without refitting any model.
# Rscript --vanilla real_datasets_hybrid_regrid.R --input=real_datasets_results/hybrid_ridge_threshold
source("real_datasets_hybrid_helpers.R")
args <- commandArgs(trailingOnly=TRUE)
option <- function(name, default) {
  match <- args[startsWith(args, paste0("--", name, "="))]
  if (length(match)) sub(paste0("^--", name, "="), "", tail(match, 1)) else default
}
out <- option("input", "real_datasets_results/hybrid_ridge_threshold")
k_values <- as.numeric(strsplit(option("k", "0,0.5,1,2,3,4,5,7,10,30,Inf"), ",", fixed=TRUE)[[1]])
stopifnot(!anyNA(k_values), all(k_values >= 0), !anyDuplicated(k_values),
          identical(k_values, sort(k_values)), k_values[1] == 0, is.infinite(tail(k_values,1)))
old <- read.csv(file.path(out, "metrics.csv"))
backup <- file.path(out, "metrics_before_regrid.csv")
if (!file.exists(backup)) file.copy(file.path(out, "metrics.csv"), backup)
files <- list.files(file.path(out, "datasets"), pattern="\\.rds$", full.names=TRUE)
all_metrics <- list()
for (path in files) {
  saved <- readRDS(path)
  if (!length(saved$predictions)) next
  name <- tools::file_path_sans_ext(basename(path))
  rows <- list()
  for (prediction in saved$predictions) {
    fold <- prediction$fold
    reference <- saved$metrics[saved$metrics$Fold == fold, , drop=FALSE][1,,drop=FALSE]
    specialists <- list(Hybrid=prediction$specialist)
    if (!is.null(prediction$ridge_specialist)) specialists[["Ridge hybrid"]] <- prediction$ridge_specialist
    base_auc <- auc_fixed(prediction$truth, prediction$pooled)
    for (variant in names(specialists)) for (k in k_values) {
      h <- hybrid_predictions(prediction$pooled, specialists[[variant]], prediction$support_ratio, k)
      score <- auc_fixed(prediction$truth, h$prob)
      if (is.infinite(k)) stopifnot(identical(h$prob,prediction$pooled), !any(h$use))
      rows[[length(rows)+1L]] <- data.frame(
        Dataset=name, Fold=fold, Variant=variant, k=k, AUC_mean=base_auc,
        AUC_hybrid=score, AUC_difference=score-base_auc,
        Specialisation_coverage=mean(h$use),
        Brier_mean=mean((prediction$pooled-prediction$truth)^2),
        Brier_hybrid=mean((h$prob-prediction$truth)^2),
        TrainN=reference$TrainN, TestN=reference$TestN, SplitOrigin=reference$SplitOrigin)
    }
  }
  metrics <- do.call(rbind, rows)
  saved$metrics <- metrics; saved$metadata$k <- k_values
  saveRDS(saved, path)
  all_metrics[[length(all_metrics)+1L]] <- metrics
}
metrics <- do.call(rbind, all_metrics)
if (!"Variant" %in% names(old)) old$Variant <- "Hybrid"
common <- merge(old, metrics, by=c("Dataset","Fold","Variant","k"), suffixes=c("_old","_new"))
for (col in c("AUC_mean","AUC_hybrid","AUC_difference","Specialisation_coverage","Brier_mean","Brier_hybrid")) {
  stopifnot(isTRUE(all.equal(common[[paste0(col,"_old")]], common[[paste0(col,"_new")]], tolerance=1e-12)))
}
write.csv(metrics, file.path(out, "metrics.csv"), row.names=FALSE)
cat("Re-evaluated", length(files), "dataset checkpoints at", paste(k_values, collapse=", "),
    "\nShared-threshold metrics verified unchanged. No models refitted.\n")
