# phase2: survival analysis (overall survival)

**Status:** built. **Commands** (in order):
```
python run.py phase2 --step checks                                  # 2a: Kaplan-Meier, proportional hazards, collinearity
python run.py phase2 --step models                                  # 2b: members through the frozen nested CV (~10 min)
python run.py phase2 --step final --models-run M                    # 2c: final model on all training rows + patient output
python run.py phase2 --step holdout --final-run F --dry-run         # checks the code path with shuffled outcomes
python run.py phase2 --step holdout --final-run F                   # ONCE ONLY: scores the holdout; refuses a second time
```

Phase 1 predicts death as a yes/no label and ignores follow-up time. Phase 2 models the time to death, using the same data, holdout and frozen CV splits.

## Data rules
- **Event:** death, the Phase 1 target. **Time:** `overall_survival_months`.
- **Zero time:** the one patient with time 0 (censored) is set to half the smallest positive time (`zero_time_rule`).
- Everything before the holdout step uses training rows only.

## Step 2a: checks (`checks.py`, `stage.py`)
- **Kaplan-Meier curves:** overall, and by ER status, integrative cluster and standard-NPI band (good ≤ 3.4 < moderate ≤ 5.4 < poor), with log-rank tests.
- **Proportional hazards:** a Cox model on the arm-C clinical columns, then a correlation test of the scaled Schoenfeld residuals against the rank of event time, BH-adjusted across columns.
- **Collinearity** among NPI, stage, size, nodes and grade: Spearman correlations and VIFs.

## Step 2b: models (`models.py`, `cv.py`)
- **Members**, each with a small grid listed simplest first:
  - `cox`: ridge Cox with Breslow ties and optional strata (scikit-survival has no stratified Cox). The grid has unstratified and ER-stratified versions.
  - `coxnet_C` and `coxnet_C+RNA`: elastic-net Cox, with RNA penalised twice as strongly as clinical columns.
  - `rsf`: random survival forest.
- **Floors** (never eligible to win): age-only Cox, NPI-only Cox, Kaplan-Meier.
- **Selection:** inner folds choose each member's configuration by mean Harrell's C; the simplest within 1 SE wins. The `best` procedure takes, per outer split, the member with the highest inner C.
- **Metrics per outer split:**
  - Harrell's C, with risk = P(death by 120 months);
  - time-dependent AUC and Brier score at 60 and 120 months;
  - calibration-in-the-large at 60 months.
- **Reported:** corrected resampled CIs, paired comparisons, risk-group Kaplan-Meier with a log-rank test, and decile calibration per model.
- **Final member:** the highest mean outer C, then the simplest within 1 corrected SE (order cox < coxnet_C < coxnet_C+RNA < rsf).
- **Numerical failures:** a fit that fails, or gives non-finite predictions, makes that configuration ineligible in that split (logged in `inner_fit_failures`).

## Step 2c: final model and holdout (`final.py`)
- **Deployment fit:** the final member's configuration is chosen on 3 stratified folds of all training rows, then the model is refitted on all of them. 50 bootstrap refits give the patient interval.
- **Saved model:** `models/phase2_final.joblib`. `model.records(X)` returns, per patient:
  - survival at 5 and 10 years with a 95% interval;
  - the survival curve;
  - P(death by 10 years), its percentile and the risk group with that group's observed survival;
  - exact log-hazard contributions per input (Cox);
  - range flags, the model's cross-validated quality, and a disclaimer.
- **Holdout scoring:** the saved model is applied unchanged, and the floors are refitted on the training rows. Intervals come from a 2,000-resample patient bootstrap.
- **One-time guard:** the marker `outputs/frozen/phase2_holdout_scored.json` works as in Phase 1. Once complete, the command refuses to run; an interrupted scoring can be finished only with the same model file.

## Config (`phase2:`)
`time_column`, `zero_time_rule`, `horizons`, `km_groups`, `npi_bands`, `ph_alpha`, `ph_stratify_column`, `collinearity_columns`, `vif_limit`, `select_within_se`, `risk_horizon`, `members`, `floors`, `final`, `deploy`, `holdout_n_boot`, `comparisons`; plus `paths.phase2_holdout_scored`.

## Environment
`scikit-survival` 0.27.0 is installed with `pip install --no-deps -r requirements-survival.txt`. Its dependency `ecos` has no Python 3.14 wheel, and the survival-SVM models that need it are not used.

## Tests (`tests/test_phase2.py`)
- the ridge Cox matches scikit-survival (with ties), and strata get their own baselines;
- survival metrics rank a good model above a constant one;
- the log-rank, PH and VIF checks behave on synthetic data, and a hazard that reverses over time is flagged;
- grids are ordered simplest first, and the final-model rule prefers the simplest member within 1 SE;
- the zero-time rule and the NPI bands are applied as specified;
- patient records: contributions add up exactly, survival never rises, a single patient matches the batch, and out-of-range inputs are flagged;
- the Phase 2 holdout marker is separate from Phase 1's.

## Left out
Competing risks, disease-specific survival, neural and boosted survival models, Weibull models.
