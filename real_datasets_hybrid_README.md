# PbP with a pooled mean-imputation fallback

Run from the repository root. R requires no additional packages. Python uses
the existing project environment.

```sh
Rscript --vanilla real_datasets_hybrid_experiment.R
.venv/bin/python real_datasets_hybrid_plot.py
```

The runner uses 38 datasets, retains benchmark colic, and ignores PbP.MinObs.
Outputs go to `real_datasets_results/hybrid_threshold/`. Existing benchmark
predictions and scores are not modified. Repeating the command recomputes the
experiment and overwrites only this experiment's outputs. Each dataset is saved
as it completes, and aggregate CSVs are updated after each dataset. The runner
does not automatically resume from checkpoints.

For a quick validation run:

```sh
Rscript --vanilla real_datasets_hybrid_experiment.R --datasets=boys,autos,colic,labor --folds=2 --out=real_datasets_results/hybrid_smoke
.venv/bin/python real_datasets_hybrid_plot.py --input real_datasets_results/hybrid_smoke
Rscript --vanilla test_real_datasets_hybrid.R
```

To add the ridge-specialist comparison (requires the installed R `glmnet` package):

```sh
Rscript --vanilla real_datasets_hybrid_experiment.R --ridge=true
.venv/bin/python real_datasets_hybrid_plot.py --input real_datasets_results/hybrid_ridge_threshold
```

This saves both variants to a separate directory, preserving the original
unregularised experiment. The pooled baseline, outer splits, pattern eligibility
rules and nominal coefficient counts are identical for both curves. Coverage can
still differ when one fitting procedure produces an unavailable prediction.

Ridge specialists use `glmnet` with alpha=0, standardised predictors and an
unpenalised intercept. Each pattern with at least 12 training observations and at
least 3 in each class uses stratified 3--5-fold inner CV, selecting the penalty
with minimum binomial deviance. The prespecified grid has 30 log-spaced values
from 1e-4 to 100. Smaller two-class patterns use fixed lambda=1 on glmnet's
average-loss scale; this choice is not selected from outer test outcomes.
For these small patterns, a base-R optimiser solves the same standardised
mean-log-loss plus lambda/2 times squared slopes objective, since glmnet refuses
patterns having just one minority-class observation. The intercept is unpenalised.
Intercept-only specialists use the pattern's empirical class frequency. No
single-class pattern is fitted in either variant. Selected penalties and the
selection rule are written to the pattern audit. Lambda is chosen once per
pattern and outer split; it is not reselected for each k.

## Prediction rule

Fit pooled mean/mode-imputation logistic regression once on the full training
split. Numeric variables use training means; categorical variables use training
modes. There is no missingness mask in this model. Entirely missing and constant
training columns are removed. Unknown test categorical levels are treated as
missing and receive training-only imputations. Safe internal feature names avoid
formula parsing errors. A pooled-model error or invalid prediction fails the
split; it is not silently replaced with a constant.

Fit one unregularised logistic regression per observed training pattern, dropping
within-pattern constant predictors and unused categorical levels. The nominal
coefficient count p_m includes the intercept and dummy-variable expansion; it is
not the number of original features or the design matrix rank. Single-class
patterns and patterns with no observed covariates are skipped, consistent with
the existing PbP implementation. An observed pattern with all constant
covariates and both classes may have an intercept-only specialist.

For k in {0, 0.5, 1, 2, 3, 4, 5, 7, 10, 30, Inf}, replace a pooled probability with
the specialist probability only if it is available and N_m / p_m >= k. No model
is refitted as k changes. Errors, non-finite specialist predictions, unseen
patterns, or pattern-specific unseen factor levels retain pooled probabilities.
Fully missing test rows receive pooled probabilities too.

GLM warnings, including separation/nonconvergence and rank deficiency, are
recorded. As in the existing PbP code, a warning alone does not disqualify finite
probabilities. Thus k isolates the support rule rather than also introducing a
new convergence-based exclusion rule.

At Inf, predictions exactly equal the pooled model. At zero, all available
specialist predictions are used. This is PbP with a pooled fallback, not the
existing PbP's constant-prevalence fallback.

## Splits and evaluation

Reuse saved indices when present and validate their labels against the
reconstructed source-row permutation. When saved indices are absent (currently
NHANES), generate 15 deterministic 80/20 splits using the training script's seed
convention. Every checkpoint saves original-row train/test indices and truth,
so later analyses do not need to reconstruct alignment. Splits generated for
NHANES are new experiment splits, not claimed to be verified against its old CSV.

Compute AUC using the Mann-Whitney formula, with fixed direction (larger means
class 1) and half credit for ties. A test split with only one class has undefined
AUC. Brier scores are also saved. No test outcomes enter prediction construction
or threshold selection.

`metrics.csv` contains one dataset/split/k row, including actual specialisation
coverage. `pattern_audit.csv` records training counts, coefficients, fit status,
warnings and available predictions per pattern. `fit_status.csv` records pooled
fit and dataset/split failures. Checkpoints under `datasets/` retain probabilities,
support ratios, split indices and source-file hashes.

The generated `.rds` checkpoints are kept locally and ignored by Git; CSV
results and summaries are retained for version control. These are new hybrid
experiment outputs, separate from the original benchmark prediction files.
Regridding requires the local checkpoints. On a fresh checkout, regenerate
them by rerunning the experiment above (with `--ridge=true` for the ridge
comparison) before running the regrid script.

## Figures

Average split-level differences within each dataset, then give every dataset
equal weight. Show the median and interquartile range across datasets. The band
describes heterogeneity, not a confidence interval. A second file shows all
individual dataset curves behind the median.

Use the same dataset cohort at every k: include only datasets with all requested
splits and finite AUC scores at every threshold. Exclusions are written to
`plot_exclusions.txt`. The threshold settings are equally spaced on the plot;
their numerical spacing is not linear. The Inf endpoint is displayed explicitly.

Generated files include `hybrid_threshold_summary.pdf/png`,
`hybrid_individual_curves.pdf/png`, `dataset_threshold_means.csv`, and
`threshold_summary.csv`.

To change the threshold grid without retraining:

```sh
Rscript --vanilla real_datasets_hybrid_regrid.R --input=real_datasets_results/hybrid_ridge_threshold --k=0,0.5,1,2,3,4,5,7,10,30,Inf
.venv/bin/python real_datasets_hybrid_plot.py --input real_datasets_results/hybrid_ridge_threshold
```

Regridding updates checkpoint metrics and metadata too, preserves the first
pre-regrid CSV, and verifies that scores at shared thresholds are unchanged.
