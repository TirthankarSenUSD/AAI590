# report: validation report and patient-level output

**Status:** planned (build step 7); no code yet. Design: PLAN.md sections 6 (report), 10 and 11 (A3, A4).

## Validation report

Generated from the run artifacts; nothing is typed by hand. Statements are computed with conditional wording, and every comparison uses the same metric on both sides. It contains:

- **The headline:** the outer-fold score of the select-and-ensemble procedure.
- **Each member's own score** and how often each member won (winner stability).
- **Optimism** (inner against outer).
- **Calibration:** raw and recalibrated (reliability curve, Brier skill, slope, intercept).
- **Treatment-subgroup metrics** with counts and intervals.
- **The baselines** on the same splits.
- **The holdout, scored once, last.** Nothing that depends on it is recorded before then.
- **Limitations:**
  - the any-death label;
  - a single cohort source;
  - treatment confounding;
  - weak molecular signal;
  - the file's NPI definition.

Every figure links back to its run and manifest entry.

## Patient-level output (a function, no user interface)

Returns one structured record per patient:
- **Inputs:** echoed back, with how missing values were handled.
- **Risk:** the calibrated probability of death with a bootstrap interval, its percentile among training patients, and the risk group (cutpoints from training data).
- **Contributing factors:** SHAP-based, grouped by clinical, RNA and DNA, worded "associated with".
- **Cohort context:** percentiles, mutation prevalence, and an expression-based group with its observed death rate, counts, intervals and a small-group flag.
- **Model quality:** the model's cross-validated AUROC and calibration.
- **Flags:** inputs outside the training range, and the disagreement flag if it proves useful.

It never gives a diagnosis, treatment advice, causal claims, individual certainty or a time horizon (Phase 1).

## Tests

- Probabilities lie in [0, 1].
- SHAP contributions plus the base value equal the model output.
- A single-patient call gives the same output as a batch call.
- Flipping a metric in a fake results folder changes the report's wording.
