# ensemble: fusion and ensemble

**Status:** built (build step 5). **Command:** `python run.py ensemble --farm-runs A,B,C`.

This stage combines the farm's members into ensembles, compares them with the best single member, and chooses the final model by fixed rules.

**Where the members come from:** each member's winner for each outer fold, read from one or more farm runs. For each (arm, member), the first listed run that holds it is used. Every winner is refitted and must reproduce its farm run's outer AUROC exactly (`refit_check`), so no config match on the farm grids is needed. The sections that decide how data are prepared (columns, seed, splits, holdout) must match.

**Everything happens inside each outer fold**, so the outer test rows are only ever scored.

## Steps, per arm and outer split

1. **Member predictions.**
   - Each member's winner is refitted on the 3 inner folds, giving **out-of-fold predictions on the outer training rows**.
   - It is also refitted on all outer training rows, giving predictions on the outer test rows.
   - Cause-specific members are fitted on the cause label, as in the farm.
2. **Agreement merge.** Members whose out-of-fold predictions have Spearman correlation above 0.95 (`merge_spearman`) are merged: a weighted average of their logits, with weights ∝ AUROC − 0.5.
3. **Four options**, simplest first, all fitted on the out-of-fold predictions:
   - `best_single`: the member chosen by the harness rule (best AUROC across the inner folds; simplest family within 1 SE).
   - `mean`: the plain average of the members' probabilities.
   - `weighted`: an average weighted by AUROC − 0.5.
   - `stacker`: a **non-negative** L2-penalised logistic regression on member logits (`stacker_l2`), fitted by bounded L-BFGS.
4. **Choice inside the fold.** A second-level stratified CV (`option_cv_folds` = 3) over the outer training rows fits each option on two parts and scores it on the third. The fixed rule then picks the best mean AUROC, with the simplest option within 1 SE winning. The **outer score of this chosen option is the select-and-ensemble headline**.
5. **Every option's outer score is reported too,** along with diversity measures on the outer test rows:
   - prediction Spearman and error correlation between members;
   - a **drop-one test**: the stacker's AUROC without each (merged) member.

## Final model (fixed rules)

- **Arm:** the baseline arm `C` (clinical only), unless another arm's select-and-ensemble procedure is significantly better (paired corrected t, p < 0.05 and CI above 0).
- **Model within the arm:** the option with the highest mean outer AUROC; the simplest option within 1 SE wins.
- **Regression floor:** mean − 2 SD of that option's outer-fold AUROC. Later runs of the final model must not fall below it.

**How to read the two numbers:** the nested select-and-ensemble estimate is the honest one. The final model's own mean outer AUROC is reported next to it, and it is slightly optimistic because the option was chosen on those same outer scores.

## Late fusion

`late_fusion` is a pseudo-arm whose three "members" are the best-member procedures of the clinical-only, RNA-only and DNA-only arms (`ensemble.late_fusion`), each chosen inside the fold. They are combined by the same four options and in-fold choice, and the final-model rule treats the pseudo-arm like any arm. Together with C+RNA, C+RNA+DNA and fused (early fusion) and the multimodal network (intermediate fusion), it completes the omics test across fusion styles.

## Fused arm

`fused` is a farm arm: clinical columns plus the top 15 genes (by selection frequency, inside the fold) plus the 5 known driver mutations, run with the four classical members. It enters the ensemble like any arm. The feature-disagreement diagnostic is left to step 6 (agreement analysis).

## Outputs (`outputs/<run>/ensemble/`)

| Kind | Outputs |
|---|---|
| Tables | `refit_check`, `option_fold_metrics`, `ensemble_choice`, `merged_members`, `weights`, `diversity`, `drop_one`, `summary`, `paired_comparisons` (each option vs best single; each arm vs C), `final_model`, `ensemble_summary.md` |
| Figures | options per arm, stacker weights, member prediction correlations |

## Tests

`tests/test_ensemble.py`:
- the stacker's weights are non-negative and favour the informative member;
- near-duplicates are merged and nothing else;
- every option returns valid probabilities;
- the in-fold option choice works;
- end to end on two tiny farm runs: exact refits, first listed run wins, final arm C unless significantly beaten, floor below the mean.
