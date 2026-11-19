# importance: gates and feature importance

**Status:** built (build step 4). **Command:** `python run.py importance [--farm-run RUN_ID] [--reuse RUN_ID] [--procedures A,B]`. The default is the newest completed farm run. `--reuse` rebuilds the summaries and the detail pass from an earlier run's per-split tables, skipping the refits. `--procedures` explains only the named procedures, for example `C|logreg_cause`, with no detail pass.

This stage explains what the models chosen by the farm rely on.

**What it explains:** the **winners the farm run chose inside each outer fold**: for every member procedure (`<arm>|<member>`) and every best-member procedure (`<arm>|best`), in every one of the 10 outer splits.

**How:**
1. Each winning configuration is rebuilt from the config and **refitted on that split's outer training rows**.
2. The refit must reproduce the farm's recorded outer AUROC exactly. If it doesn't, the stage stops (`refit_check`).
3. Everything is then measured on that split's **held-out rows only**.

**Config check:** before starting, the stage confirms the farm run's config matches the current one (farm, harness, columns, seed, splits). Otherwise the winners could not be rebuilt.

## Files

| File | What it does |
|---|---|
| `explain.py` | Helpers for one fitted pipeline, listed below. |
| `stage.py` | `run_importance()`: builds the tasks, runs them in parallel (with progress lines), aggregates across splits, then writes the figures and a generated summary. |

`explain.py` contains:
- feature naming (each model input traced back to its source column);
- SHAP;
- logistic regression coefficients;
- permutation importance;
- gene grouping;
- partial dependence, ICE and Friedman's H-statistic.

## What is computed

| Measure | For | How |
|---|---|---|
| **Permutation importance** | every member, arms C, C+RNA and strict-clinical | AUROC drop when a column, or a group of correlated genes, is shuffled on the held-out rows (5 repeats). Gene groups are built on the outer training rows: average linkage on 1 − \|r\|, cut at 0.5; genes in a group share its value. |
| **Importance by data block** | every member, those arms plus C+RNA+DNA | Shuffle a whole block: all clinical columns, the treatment group, all RNA, all DNA. |
| **SHAP** | logistic regression (exact linear) and random forest / XGBoost (TreeSHAP) only | Mean \|SHAP\| per source column on the held-out rows; one-hot levels are summed back to their column. Values describe the model's own inputs (for example "RNA latent 3" when the winner uses autoencoder features). Scales differ between members, so only ranks are compared. |
| **Coefficients and odds ratios** | logistic regression (binary: death vs alive; cause-specific: cancer death vs alive and other-cause death vs alive) | Per split. Summarised as mean, SD, sign consistency across splits, and odds ratio (per SD for scaled inputs, per level for indicators). |
| **Fold stability** | all of the above | Mean rank and the share of splits in which a feature is in the top 10. |
| **Partial dependence and ICE** | the best-member procedure of each full arm, top 10 single-column features | Partial dependence: grids from the training rows (numeric: 5–95% quantiles), mean over held-out rows, summarised over splits by grid position. ICE: 50 held-out patients of split r0o0. |
| **Interactions** | same top 10 | Friedman's H² for every pair, on 60 held-out patients per split. Model-agnostic, so every member is treated alike. |
| **Learning curves** | best-member procedures | The winner refitted on 25 / 50 / 75 / 100% of the outer training rows, then scored on the held-out rows. |
| **Gates** | the explained procedures | Overfit gate from the farm run (report only), plus the provisional regression floors of the best-member procedures. The final model's floor is fixed in step 5. |

The SVM and the neural members get no SHAP. Approximate (model-agnostic) SHAP was judged not worth its cost, since they are never chosen as an arm's winner.

## Settings

All in `config.yaml` → `importance:`: the arms (`arms_full`, `arms_modality_only`), `shap_members`, `n_repeats`, `gene_groups`, `top_n` (10), `interaction_method` (`friedman_h`), `pdp_grid`, `ice_rows`, `h_rows` and `learning_curve_fractions`.

## Outputs (`outputs/<run>/importance/`)

| Kind | Outputs |
|---|---|
| Tables | `refit_check`, `permutation_importance` and `permutation_summary`, `modality_importance`, `shap_summary`, `lr_coefficients` and `lr_coefficient_summary`, `top_features`, `partial_dependence`, `ice`, `interactions`, `learning_curve`, `gate_summary`, `regression_floor_provisional`, `importance_summary.md` |
| Figures (per full arm) | permutation importance (top 15), partial dependence with ICE, interaction heatmap |
| Figures (overall) | clinical logistic regression odds ratios, reliance on each data block, learning curves |

## Caveats

- Importance describes **model behaviour, not biology**.
- **Correlated features share or split their importance.** Genes are grouped for that reason; correlated clinical columns, such as size, stage and nodes, are not grouped.

## Tests

`tests/test_importance.py`:
- permutation importance finds a planted signal and gives noise about 0;
- the H-statistic separates an interacting model from an additive one;
- correlated genes are grouped;
- feature names match every model's inputs (linear, tree, embedding; PCA, gene sets, pathways);
- exact linear SHAP adds up to the log-odds;
- end to end on a tiny farm run: every refit reproduces the farm AUROC, and all outputs are written.
