# farm: the model farm

**Status:**
- **Step 3a, built:** four classical members — elastic-net logistic regression, RBF SVM, random forest, XGBoost.
- **Step 3b, planned:** the three neural members.

**Command:** `python run.py farm`, optionally with `--members logreg_en,xgboost` or `--arms C,C+RNA`.

The farm builds many candidate configurations (arm × member × RNA representation × hyperparameters), scores every one through the harness on the frozen nested CV, then evaluates **select-the-best procedures**. In each outer split a procedure picks its winner from inner-fold statistics only, and its score is that winner's outer score. A configuration's own outer score is never the headline.

## Files

| File | What it does |
|---|---|
| `preprocess.py` | In-fold preprocessing (all scikit-learn transformers, so the harness fits them on training rows only). |
| `members.py` | Builds the candidate grid: estimators, the complexity order, and the parallel worker count. |
| `stage.py` | `run_farm()`: scoring, procedures, comparisons, permutation test, figures and the generated summary. |

## Data arms (first pass)

| Arm | Columns |
|---|---|
| `C` | The 23 clinical columns, including the treatment group |
| `C+RNA` | C plus the 489 RNA genes (in one of six representations, below) |
| `strict_clinical` | C without the molecular-derived columns (PAM50, integrative cluster, 3-gene subtype, cohort) |

**One column per concept.** These are dropped from every arm:
- `oncotree_code`, which is identical to `cancer_type_detailed`;
- `cancer_type_detailed`, which is fully determined by the finer `tumor_other_histologic_subtype`;
- `cancer_type`, where one patient is a sarcoma;
- the file's NPI, a composite of size, nodes and grade that stays a floor only.

## Preprocessing (fitted inside each fold)

| | Linear members (logistic regression, SVM) | Tree members (random forest, XGBoost) |
|---|---|---|
| Numeric / ordinal | median fill + missing flag + standardise | passed through with gaps (native handling) |
| Categorical | missing as its own level, one-hot; levels under 20 patients pooled | missing as its own level, ordinal codes; levels under 20 patients pooled |
| Binary treatment flags | as is | as is |

**RNA representations (C+RNA only).** All are fitted in the fold:
- `all`: all 489 genes. For the logistic regression they are scaled by 0.5, which gives them 4× the L2 penalty of the clinical columns.
- `top15` / `top30` / `top50`: genes ranked by *selection frequency*, meaning how often an L1 logistic regression keeps them across 10 stratified half-samples of the training rows.
- `pca10` / `pca30`: principal components.

## Members and grids (small for now; in `config.yaml` → `farm.grids`)

| Member | Grid | Notes |
|---|---|---|
| `logreg_en` | C ∈ {0.01, 0.1, 1}, l1_ratio ∈ {0, 0.5, 1} | saga solver; coefficients readable |
| `svm_rbf` | C ∈ {0.1, 1, 10}, gamma = scale | RBF only; probabilities by Platt scaling inside the fold |
| `random_forest` | 300 trees, max_features ∈ {sqrt, 0.2}, min_samples_leaf ∈ {5, 20} | |
| `xgboost` | 300 rounds, learning rate 0.05, depth ∈ {2, 4}, colsample ∈ {0.5, 0.8} | hist method |

The full grid is 160 configurations: 20 for C, 120 for C+RNA (20 × 6 representations) and 20 for strict-clinical. Age enters as a plain linear term (no spline for now).

## Selection and "simpler"

A procedure picks its winner by the harness rule: best mean inner AUROC, and the simplest configuration within 1 SE wins. "Simpler" is fixed in config (`complexity_rule`):
1. member family first (logistic regression < SVM < random forest = XGBoost);
2. then fewer RNA inputs;
3. then stronger regularisation.

The procedures:
- `<arm>|<member>`: the member's own nested score;
- `<arm>|best`: the best member, chosen inside each fold, which is the arm's headline.

## Parallel execution

Configurations are scored in parallel worker processes (`farm.n_jobs`; `auto` means logical cores minus 1). Each worker runs a single thread, so XGBoost, scikit-learn and numpy don't oversubscribe the CPU. Every fit is seeded and results are assembled in a fixed order, so the output is identical for any `n_jobs`; `tests/test_farm.py` checks this.

## Outputs (`outputs/<run>/farm/`)

| Output | Content |
|---|---|
| `configuration_fold_metrics` | Every configuration on every split |
| `procedure_winners` | The winner per procedure and split |
| `procedure_summary` | Mean and corrected 95% CI per procedure |
| `winner_stability` | How often the same configuration, member and representation win |
| `overfit_gate` | The gate applied to every procedure: `overfit` (train-CV gap above 0.05 or slope below 0.8), `recalibrate` (slope above 1.2: recalibrated probabilities are primary) or `pass`. Report only; it never removes a configuration from selection. |
| `paired_comparisons` | Each procedure against the age-only floor; C+RNA against C; strict-clinical against C |
| `subgroup_metrics`, `reliability` | For the arm procedures |
| `regression_floor` | Provisional, until the final model |
| `permutation_test` | The full `C|best` procedure on shuffled labels |
| Figures | Procedure AUROC, winning members, calibration |
| `farm_summary.md` | Generated summary |

## Next (3b)

The MLP, the MLP with categorical embeddings, and the multimodal network (one encoder per block, block dropout; PyTorch, CPU). They join the same grids, procedures and comparisons.

## Tests

`tests/test_farm.py`:
- parallel output equals serial;
- the top-k selector recovers planted signal genes;
- missing categories form their own level;
- the column rules (23 clinical, strict-clinical drops, treatment kept);
- linear preprocessing is one-hot with missing flags and learns its fill values from training rows only;
- the grid counts and the complexity order;
- a real-data smoke run.
