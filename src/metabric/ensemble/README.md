# ensemble: fusion and ensembling

**Status:** planned (build step 5); no code yet. Design: PLAN.md section 7, steps 2–5.

## Steps

All four steps are rerun inside every outer training fold.

1. **Fused feature set.**
   - **What:** the best features of each data subset's winner form one set, run as an extra arm.
   - **Disagreement flag:** features where the fused model disagrees with the subset models are flagged as interaction or redundancy candidates. The comparison uses rank overlap and selection frequency, never raw SHAP magnitudes.
   - **Use:** diagnostic only.
2. **Agreement merge.** Members whose out-of-fold predictions have Spearman correlation above 0.95 (fixed in advance) are merged, weighted by (AUROC − 0.5).
3. **Stacker and averages.** These are compared against each other and against the best single member:
   - a non-negative regularised logistic regression on the members' logits (up to 7 inputs);
   - an (AUROC − 0.5)-weighted average;
   - the plain mean.
4. **Final choice** by a rule fixed beforehand: highest mean outer-fold AUROC, with the simplest option winning within 1 SE.

## Rules

- Stacker inputs are out-of-fold predictions only (A8). New patients are scored with the average of the fold models, so their predictions come from the same distribution as the out-of-fold ones (A9).
- Diversity diagnostics: prediction and error correlations, and a drop-one test (stacker AUROC without each member). Redundancy is reported as a finding; the farm stays at 7 members.
