Exploratory pattern-support plots

Run from the repository root:
    Rscript --vanilla real_datasets_pattern_support.R
    .venv/bin/python real_datasets_support_plot.py

Each point is one dataset. Both coordinates are means over 15 saved splits.
The vertical coordinate is CSV Brier(PbP) minus CSV Brier(Mean.IMP).
No PbP models are changed or thresholded in this analysis.

For each test observation, N_m is the number of training rows with its pattern.
p_m is the number of design-matrix columns for that pattern, including the
intercept and categorical expansion, after dropping constant covariates and
unused factor levels. This is a nominal coefficient count, not matrix rank.
Coverage(k) is the fraction of test rows satisfying N_m >= k*p_m.
Unseen patterns have zero support. Coverage is a count-based diagnostic, not
verified model-fit success, class balance, or numerical stability.
Test factor levels absent from training are treated as missing, as in training.
Training masks use covariates only, after removing rows with missing targets.

37 datasets are included. NHANES has CSV scores but lacks locally saved
predictions/Y needed to reconstruct and validate its split alignment.
Benchmark colic is retained; Tabzilla colic is excluded.
Saved target permutations and test labels are checked for every included split.
Some saved PbP predictions produce different Brier scores from the CSV:
colic (2 splits), debt (5), pulplignin (2), soybean (10), Ozone (5).
The figure deliberately uses CSV performance throughout, as requested for the
draft. The support_audit.csv records these discrepancies for the results refresh.
