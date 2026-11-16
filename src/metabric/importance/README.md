# importance: gates and feature importance

**Status:** planned (build step 4); no code yet. Design: PLAN.md section 7, step 1, and section 6 (gates).

## Gates

Every member passes the harness overfit gate before it is interpreted:
- a train-versus-CV AUROC gap of at most 0.05;
- a calibration slope within 0.8–1.2;
- learning curves are reported.

The regression floor of the final model (mean − 2 SD of its outer-fold AUROC) is derived and recorded here, before any later run is compared with it.

## Importance per member

Computed on validation rows inside the outer folds, then summarised across folds:

- **SHAP values**, or coefficients for the logistic regression, with their stability across folds.
- **Permutation importance**, plus partial dependence, ICE and interaction plots for the top 10 features (proposal; the interaction method is still open).
- **Grouping:** correlated genes are grouped into modules before ranking, so that importance shared among near-duplicates doesn't hide them.
- **Modality importance:** by grouped permutation (clinical, RNA, DNA), never by summed coefficients (A2).

## Rules

- Nothing is selected or fitted on rows a fold is scored on.
- Importance describes model behaviour, not biology, and is sensitive to correlated features. Reports say so.
- The gene crosswalk (`reference/crosswalk/`) supplies the gene-set context for interpreting genes.
