# harness: the evaluation harness

**Status:** built (build step 2). **Command:** `python run.py harness` (runs the floors).

Every model from the farm onward is scored here, on the same frozen splits, by the same rules. On real data it has so far been run only with the floors. Its correctness was first established on synthetic data, where the right answer is known.

## Files

| File | What it does |
|---|---|
| `core.py` | `Candidate`, `Split`, `Harness`, the selection rule, split-hygiene checks and all evaluations. |
| `metrics.py` | Probability metrics: AUROC, AUPRC, log loss, Brier and Brier skill, calibration intercept and slope, reliability bins, the operating-point thresholds (fixed sensitivity; maximum F1) and their metrics (incl. F1), bootstrap AUROC intervals. |
| `floors.py` | `design_matrix()` (clinical + RNA + mutation flags + the derived `npi_standard`; never outcome columns) and the four floors. |
| `stage.py` | `run_harness()`: runs the floors on the training rows and writes every artifact plus a generated summary. |

## How one candidate is scored (`Harness.evaluate`)

A **candidate** is a name, a function that builds an unfitted estimator, the columns it uses, a complexity (lower is simpler) and whether it may win. Floors may never win.

For each of the 10 frozen outer splits:

1. **Inner layer**, inside the outer training rows:
   - for each of the 3 inner folds, fit on the other two and predict the held-out one;
   - this gives the inner AUROC (mean and SE), inner log loss and inner Brier;
   - it also gives the **inner out-of-fold predictions** for every outer training row.
2. **Fitted on the inner out-of-fold predictions only:**
   - the **operating-point threshold**, the largest threshold with at least 80% sensitivity;
   - a **Platt recaliber** (logistic regression on logit(p)).
3. **Outer layer:**
   - fit on all outer training rows and predict the outer test rows;
   - compute the metrics, using the training-fold prevalence as the Brier reference so no test label enters it;
   - apply the inner threshold to get sensitivity (recall), specificity, PPV (precision), NPV and F1;
   - a secondary operating point, the threshold maximising F1 on the inner folds (`opf1_*`; never the headline);
   - apply the recaliber to get the recalibrated metrics;
   - record the training-row AUROC (for the overfit gate) and the optimism (inner minus outer AUROC).

With `select=True`, the harness also scores the **select-the-best procedure**. In each outer split it picks a winner from the inner statistics by a fixed rule (`select_candidate`):
1. find the best mean inner AUROC;
2. among eligible candidates within 1 SE of it, the simplest wins;
3. ties go to lower inner log loss, then lower inner Brier.

It records the winner, its outer score and the optimism. The headline of later steps is this procedure's outer score, not the best member's own score.

`light=True` skips the inner layer (used by the permutation test when no selection is needed).

## Analyses

| Method | Output |
|---|---|
| `summarise` | Mean, SD and a Nadeau–Bengio corrected-t 95% CI over the outer splits for every metric. |
| `paired` | Paired differences between candidates on identical splits, corrected resampled t-test. |
| `subgroups` | AUROC (bootstrap CI), Brier, mean prediction and observed rate within treatment subgroups: chemotherapy, hormone therapy, radiotherapy, surgery type, chemo × hormone. Uses each patient's out-of-fold prediction averaged over the two repeats. |
| `overfit_gate` | Two separate questions; report only, never excludes from selection. **Overfitting** (status `overfit`): a train-versus-CV AUROC gap above 0.05, or a calibration slope below 0.8 (over-confident). **Under-confidence** (status `recalibrate`): a slope above 1.2, in which case the in-fold recalibrated probabilities become the primary ones. |
| `learning_curve` | Outer AUROC at 25/50/75/100% of the outer training rows (reported, not gating). |
| `permutation_test` | Reruns the full procedure on shuffled labels; the mean AUROC must lie within 0.5 ± 0.03. |
| `loco` | Leave-one-cohort-out on the training rows (sensitivity only). |
| `regression_floor` | Mean − 2 SD of outer-fold AUROC per model: the level a later run must not fall below. |

`check_split_hygiene` runs on construction and refuses splits where training and test overlap, inner rows fall outside the outer training rows, or a patient isn't tested exactly once per repeat.

## Floors (`floors.py`)

| Floor | Model |
|---|---|
| `majority` | Predicts the training-fold death rate for everyone (AUROC 0.5). |
| `age_only` | Logistic regression on age. |
| `npi_file` | Logistic regression on the NPI as supplied (size weight 0.02 per cm). |
| `npi_standard` | Logistic regression on the standard NPI, with an in-fold median fill and a missing flag. |

Results on the training rows: age-only 0.685 is the bar to beat.

## Settings

`config.yaml` → `harness:` holds the operating point, recalibration method, reliability bins, bootstrap count, number of permutations, chance band, 1-SE rule, learning-curve fractions, overfit gate, subgroups and the regression-floor SD multiplier. `floors:` defines the floors.

## Tests (`tests/test_harness.py`, on synthetic data)

- **Signal recovery:** a known risk ordering is recovered, with AUROC within 0.02 of the true model and a calibration slope near 1.
- **Selection rule:** the signal model beats noise, an identical but more complex copy loses, the 1-SE rule picks the simpler model, and a floor never wins.
- **Leakage detection:** shuffled labels score at chance, and a deliberately leaky procedure (gene selection on all rows) is caught.
- **Fold hygiene:** every fit sees outer-training rows only, and the threshold equals one recomputed by hand from inner predictions.
- **Statistics:** split-hygiene violations are detected, the corrected t-test returns known values, and calibration metrics are correct on calibrated synthetic probabilities.
- **Real floors:** the floors fall in their expected AUROC ranges on the frozen splits.
