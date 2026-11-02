# Export pattern support on the saved train/test splits using base R only.
# Run from the repository root: Rscript --vanilla real_datasets_pattern_support.R
# Reconstruct the training script's row permutation and validate against saved Y.
out_dir <- "real_datasets_results/pattern_support_draft"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
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
thresholds <- c(1, 5, 10, 20, 50, 100)
k_values <- c(1, 2, 3, 5, 10, 20, 50)
csv_paths <- c("real_datasets_results/real_datasets_metrics_detailed.csv",
               "tabzilla/tabzilla_real_datasets_results/real_datasets_metrics_detailed.csv")
csvs <- lapply(csv_paths, read.csv)
rows <- list()
audit <- list()
for (source_id in seq_along(csvs)) {
  scores <- csvs[[source_id]]
  for (dataset in unique(scores$Dataset)) {
    if (source_id == 2 && dataset == "colic") next
    result <- tryCatch({
      if (source_id == 1) {
        stem <- dataset
        data_file <- paste0("real_datasets/", stem, ".RDS")
        pred_dir <- "real_datasets_results/preds"
        target <- targets[[dataset]]
        cutoff <- cutoffs[[dataset]]
      } else {
        files <- list.files("tabzilla/tabzilla_real_datasets", full.names=TRUE)
        if (dataset == "Wisconsin") {
          data_file <- files[grepl("Wisconsin-breast", files, fixed=TRUE)]
        } else {
          data_file <- files[startsWith(basename(files), paste0("openml__", dataset, "__"))]
        }
        stopifnot(length(data_file) == 1)
        stem <- tools::file_path_sans_ext(basename(data_file))
        pred_dir <- "tabzilla/tabzilla_real_datasets_results/preds"
        target <- "target"
        cutoff <- 0
      }
      pred_file <- file.path(pred_dir, paste0(stem, "_mc_results.RDS"))
      y_file <- file.path(pred_dir, paste0(stem, "_Y.RDS"))
      if (!file.exists(pred_file) || !file.exists(y_file)) stop("Saved predictions/Y unavailable")
      dat <- readRDS(data_file)
      stopifnot(target %in% names(dat))
      dat <- dat[!is.na(dat[[target]]), , drop=FALSE]
      saved_y <- as.logical(readRDS(y_file))
      reconstructed_y <- as.numeric(dat[[target]]) <= cutoff
      # Seed convention is the full source file stem in the training script.
      set.seed(sum(utf8ToInt(stem)))
      permutation <- sample(seq_len(nrow(dat)), nrow(dat), replace=FALSE)
      stopifnot(identical(as.logical(reconstructed_y[permutation]), saved_y))
      x <- dat[permutation, setdiff(names(dat), target), drop=FALSE]
      saved <- readRDS(pred_file)
      folds <- sort(unique(vapply(saved, function(z) z$fold, integer(1))))
      stopifnot(identical(folds, 1:15))
      fold_rows <- list()
      brier_mismatch <- 0L
      for (fold in folds) {
        candidates <- Filter(function(z) z$fold == fold, saved)
        test <- candidates[[1]]$indices
        stopifnot(all(vapply(candidates, function(z) identical(z$indices, test), logical(1))))
        stopifnot(identical(saved_y[test], as.logical(candidates[[1]]$truth)))
        train <- setdiff(seq_len(nrow(x)), test)
        x_test <- x[test, , drop=FALSE]
        # Match training code: an unseen factor level is treated as missing.
        for (j in which(vapply(x, is.factor, logical(1)))) {
          unseen <- !is.na(x_test[[j]]) & !(x_test[[j]] %in% unique(x[[j]][train]))
          x_test[unseen, j] <- NA
        }
        train_keys <- apply(is.na(x[train, , drop=FALSE]), 1, paste, collapse="_")
        test_keys <- apply(is.na(x_test), 1, paste, collapse="_")
        counts <- table(train_keys)
        support <- as.integer(counts[test_keys])
        support[is.na(support)] <- 0L
        # Coefficient count for each pattern, including intercept and dummy
        # variables. Match PbP's removal of within-pattern constant columns.
        parameter_counts <- vapply(names(counts), function(key) {
          idx <- train[train_keys == key]
          observed <- !is.na(x[idx[1], , drop=FALSE])
          xp <- x[idx, as.logical(observed), drop=FALSE]
          varying <- vapply(xp, function(col) length(unique(col)) > 1L, logical(1))
          xp <- droplevels(xp[, varying, drop=FALSE])
          if (ncol(xp) == 0L) return(1L)
          ncol(model.matrix(~ ., data=xp))
        }, integer(1))
        test_parameters <- parameter_counts[test_keys]
        # Unseen patterns have zero training support for any positive k.
        test_parameters[is.na(test_parameters)] <- 1L
        support_ratio <- support / test_parameters
        pbp <- Filter(function(z) z$method == "PbP", candidates)
        if (length(pbp) == 1) {
          stored_brier <- scores$Value[scores$Dataset == dataset & scores$Method == "PbP" &
                                      scores$Metric == "Brier" & scores$Fold == fold]
          if (length(stored_brier) == 1 &&
              !isTRUE(all.equal(mean((pbp[[1]]$preds-as.numeric(pbp[[1]]$truth))^2),
                                stored_brier, tolerance=1e-8))) brier_mismatch <- brier_mismatch+1L
        }
        fold_rows[[length(fold_rows)+1L]] <- data.frame(
          Dataset=dataset, Fold=fold, ThresholdType="count", Threshold=thresholds,
          TrainPatterns=length(counts), TrainN=length(train), TestN=length(test),
          Coverage=vapply(thresholds, function(t) mean(support >= t), numeric(1)))
        fold_rows[[length(fold_rows)+1L]] <- data.frame(
          Dataset=dataset, Fold=fold, ThresholdType="per_coefficient", Threshold=k_values,
          TrainPatterns=length(counts), TrainN=length(train), TestN=length(test),
          Coverage=vapply(k_values, function(k) mean(support_ratio >= k), numeric(1)))
      }
      list(rows=do.call(rbind, fold_rows), brier_mismatch=brier_mismatch)
    }, error=function(e) e)
    if (inherits(result, "error")) {
      audit[[length(audit)+1L]] <- data.frame(Dataset=dataset, Status="excluded", Detail=conditionMessage(result))
    } else {
      rows[[length(rows)+1L]] <- result$rows
      audit[[length(audit)+1L]] <- data.frame(Dataset=dataset, Status="included",
                                            Detail=paste("PbP Brier mismatches vs CSV:", result$brier_mismatch))
    }
  }
}
write.csv(do.call(rbind, rows), file.path(out_dir, "split_pattern_support.csv"), row.names=FALSE)
audit <- do.call(rbind, audit)
write.csv(audit, file.path(out_dir, "support_audit.csv"), row.names=FALSE)
print(audit, row.names=FALSE)
