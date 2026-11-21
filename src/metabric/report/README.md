# report: validation report and patient-level output

**Status:** built (build step 7). **Command:**
```
python run.py report --farm-runs A,B --ensemble-run E --agreement-run G
```

The stage turns the nested-CV results into a generated validation report, refits the final model once on all training rows for use, and produces the patient-level output. It reads only training rows: the holdout is never touched here.

## Inputs (all recorded in the manifest)

- **Ensemble run:** `final_model.csv` (which arm and option won) and the paired arm comparisons.
- **Farm runs:** the final arm's select-the-best procedure (`<arm>|best`): metrics, winners, paired comparisons with the floors, overfit gate, treatment subgroups, reliability, winner stability and the permutation test. Each member's own score comes from the first listed farm run that holds it.
- **Agreement run:** ablation, the disagreement verdict and the per-patient cause-specific risks.
- The config sections that decide how data are prepared (columns, seed, splits, holdout) must match every input run.

## Steps

1. **Validation report** (`validation_report.md`). Every sentence is computed from the inputs with conditional wording (`text.py`), and every comparison names one metric for both sides. It contains:
   - the headline: the outer-fold score of the select-and-ensemble procedure;
   - discrimination, calibration (raw and recalibrated) and error metrics next to the age-only floor;
   - threshold metrics at operating points fixed on inner folds (recall, precision, F1, specificity, NPV);
   - the overfit gate, the optimism and the permutation test;
   - paired comparisons with the floors;
   - each member's own score and winner stability;
   - treatment-subgroup metrics;
   - the omics answer (arm comparisons and ablation);
   - agreement findings;
   - the holdout status;
   - limitations.
2. **Deployment fit** (`deploy.py`). The final procedure is run once on all training rows, with `report.deploy_folds` stratified folds in the inner folds' role:
   - every member picks its configuration by the inner rule;
   - members whose predictions correlate above `ensemble.merge_spearman` are merged (logit average weighted by AUROC − 0.5);
   - the `best_single` option keeps the best merged column (simplest within 1 SE).

   **Primary probabilities** follow the overfit gate: if the out-of-fold calibration slope is above `harness.overfit_gate.recalibrate_above_slope`, a Platt map fitted on those predictions is applied.

   **Bootstrap refits** (`report.n_boot`, configuration and weights fixed) give each patient's probability interval.

   The fitted model and its explainer are saved to `models/final_model.joblib`.
3. **Patient output** (`patient.py`, `PatientExplainer.records(X)` or `.record(dict)`). Each record holds:
   - **Inputs:** echoed back, each with its status (provided, missing or not supplied) and how a gap is handled.
   - **Risk:** the primary P(death) with a bootstrap 95% interval, its percentile among the training patients' out-of-fold predictions, the risk group (quartile cutpoints of those predictions) and that group's observed death rate (Wilson interval).
   - **Cause breakdown:** P(cancer death) and P(other-cause death) from the cause-specific member (`report.cause_member`).
   - **Contributions:** interventional Shapley values of the model's log-odds over the raw input columns, from antithetic permutations against a background of training patients, grouped by clinical / RNA / DNA and worded "associated with higher/lower estimated risk". Contributions plus the base value equal the model's log-odds exactly.
   - **Cohort context:** the PAM50 subtype group (`report.context_column`) with its training count, deaths, observed death rate (Wilson interval) and a small-group flag (`report.small_group`); percentiles of the numeric inputs; prevalence of any mutations supplied.
   - **Model quality:** the nested-CV AUROC with its interval, calibration slopes and Brier skill.
   - **Flags:** numeric inputs outside the training range, and category levels never seen in training.
   - **A fixed disclaimer:** the record gives no diagnosis, no treatment advice, no causal claim, no individual certainty and no time horizon.

   There is no "models disagree" flag: the agreement analysis found that disagreement does not predict error.

## Outputs (`outputs/<run>/report/`, plus `outputs/<run>/models/`)

| Kind | Outputs |
|---|---|
| Report | `validation_report.md` |
| Deployment | `deploy_candidates`, `deploy_member_winners`, `deploy_merge`, `deploy_oof_predictions`; `models/final_model.joblib` |
| Patients | `patient_output` (every training patient: out-of-fold risk from the deployment folds, percentile, risk group, subtype, out-of-fold cause-specific risks), `risk_groups`, `cohort_context`, `example_patient_records.json` (in-sample illustration of the record format), `contribution_check` |
| Figures | reliability (final procedure vs age-only floor), treatment subgroups, observed death rate by risk group |

## Config (`report:`)

`deploy_folds`, `n_boot`, `n_background`, `n_permutations`, `top_k`, `context_column`, `small_group`, `risk_quantiles`, `risk_labels`, `cause_member`, `example_quantiles`.

## Tests (`tests/test_report.py`)

- **Wording:** flipping a difference or p-value changes the report's conclusion, the verdict depends on the metric's direction, and both sides use the same metric.
- **Shapley values:** contributions are exactly additive, and match w·(x − mean) for a linear model.
- **Deployment and patient records:**
  - probabilities lie in [0, 1] and intervals are ordered;
  - contributions plus the base value equal the model's log-odds;
  - a single-patient call matches the batch output;
  - out-of-range inputs are flagged and missing inputs are echoed with their handling.

## Holdout scoring (`holdout` stage, the last step of Phase 1)

**Command** (once only):
```
python run.py score-holdout --report-run R            # R = the report run whose models/final_model.joblib is scored
python run.py score-holdout --report-run R --dry-run  # same steps with the holdout labels shuffled; scores nothing
```

**Steps:**
1. Checks that report run R completed and used the same data, columns, splits and holdout settings as the current config.
2. Writes the one-time marker `outputs/frozen/holdout_scored.json` as "started" (with the model file's sha256), then loads the holdout.
3. Applies the saved final model unchanged: nothing is refitted, recalibrated or re-chosen on holdout labels.
4. Refits each floor (majority, age-only, the two NPI floors) once on all training rows and scores it on the holdout.
5. Applies the operating points fixed on the deployment out-of-fold predictions (training rows): the recall target and the F1-maximising threshold.
6. Computes every interval and paired difference from one patient bootstrap of the holdout (`holdout_scoring.n_boot`, seeded).
7. Writes the tables, figures, example records and the generated `holdout_report.md`, then sets the marker to "complete".

**Rules:**
- Once the marker says "complete" the command refuses to run.
- An interrupted scoring can be rerun only with the same model file.
- Agreement with cross-validation is reported, not used as a gate: whether the cross-validated estimate lies inside the holdout interval, and whether the holdout AUROC is above the regression floor.
- `--dry-run` never writes the marker; its report is labelled as meaningless.

**Outputs** (`outputs/<run>/holdout/`):
- tables: `holdout_metrics`, `holdout_floor_comparisons`, `holdout_operating_points`, `holdout_subgroups`, `holdout_risk_groups`, `holdout_predictions`;
- `holdout_example_records.json` and `holdout_report.md`;
- figures: reliability, and risk groups (holdout vs training).

**Config:** `holdout_scoring:` (`n_boot`, `alpha`) and `paths.holdout_scored`.

**Tests** (`tests/test_holdout_scoring.py`, synthetic data only):
- the scoring recovers a good model and ranks it above weak floors;
- the report text follows the results and labels a dry run;
- the marker refuses a second scoring.
