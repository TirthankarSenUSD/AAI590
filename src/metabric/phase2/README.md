# phase2: simple survival analysis

**Status:** planned (after Phase 1 passes its success criteria); no code yet. Design: PLAN.md section 13.

Phase 1 predicts any death as a yes/no label, ignoring follow-up time. Phase 2 models time to death on the same data, holdout and frozen CV splits.

## Plan

- **Endpoint:** overall survival only (1,103 events, 801 censored). One patient has a follow-up time of 0; a rule is fixed and recorded (a small positive value, or drop the patient).
- **Descriptive:** Kaplan–Meier curves overall and by ER status, `integrative_cluster` and NPI band.
- **Models:**
  - a Cox model on clinical data;
  - a penalised Cox model on clinical data plus the omics features selected in Phase 1;
  - a random survival forest.
- **Floors:** age-only Cox, NPI-only Cox, Kaplan–Meier.
- **Validation:** the same nested CV, split IDs and holdout as Phase 1.
- **Metrics:**
  - Harrell's C-index;
  - time-dependent AUC at 5 and 10 years;
  - Kaplan–Meier by risk group with a log-rank test;
  - 5-year calibration.
- **Checks:** proportional hazards (stratify on ER status if violated); collinearity among NPI, stage, size, nodes and grade.
- **Output:** a survival curve with 5- and 10-year probabilities and a risk group.
- **Left out:** competing risks, disease-specific survival, neural and boosted survival models, Weibull models.

## Environment note

`lifelines` downgrades pandas below 3.0, so it is not installed. Survival tooling will be chosen when this phase starts and recorded in FINDINGS.
