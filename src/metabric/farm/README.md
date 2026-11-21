# farm: the model farm

**Status:** built.
- **Step 3a:** four classical members — elastic-net logistic regression, RBF SVM, random forest, XGBoost.
- **Step 3b:** three neural members — MLP, MLP with categorical embeddings, multimodal network.
- **Step 3c:** all eight data arms, plus the RNA and DNA representation arms.
- **Cause-specific members:** `logreg_cause` and `xgboost_cause`, trained on alive / cancer death / other-cause death and scored as death.

**Command:** `python run.py farm`, optionally with `--members logreg_en,xgboost` or `--arms C,C+RNA`.

The farm builds many candidate configurations (arm × member × RNA/DNA representation × hyperparameters), scores every one through the harness on the frozen nested CV, then evaluates **select-the-best procedures**. In each outer split a procedure picks its winner from inner-fold statistics only, and its score is that winner's outer score. A configuration's own outer score is never the headline.

## Files

| File | What it does |
|---|---|
| `preprocess.py` | In-fold preprocessing (all scikit-learn transformers, so the harness fits them on training rows only). |
| `members.py` | Builds the candidate grid: estimators, the complexity order, and the parallel worker count. |
| `neural.py` | The three PyTorch members as scikit-learn-compatible classifiers. |
| `cause.py` | The cause-specific wrapper: a 3-class model whose P(death) = P(cancer death) + P(other-cause death). |
| `representations.py` | RNA and DNA representation transformers (gene-set scores, modules, denoising autoencoder; driver genes, count, pathway flags, bag-of-mutations embedding). |
| `stage.py` | `run_farm()`: scoring, procedures, comparisons, permutation test, figures and the generated summary. |

## Data arms

| Arm | Columns | Members |
|---|---|---|
| `C` | The 23 clinical columns, including the treatment group | all 7 |
| `C+RNA` | C plus the 489 RNA genes (in one of the RNA representations, below) | all 7 |
| `strict_clinical` | C without the molecular-derived columns (PAM50, integrative cluster, 3-gene subtype, cohort) | all 7 |
| `C+DNA` | C plus the 173 mutation flags (in one of the DNA representations) | classical 4 |
| `C+RNA+DNA` | C plus RNA plus DNA | classical 4 |
| `RNA` | RNA only | classical 4 |
| `RNA+DNA` | RNA plus DNA | classical 4 |
| `DNA` | DNA only | classical 4 |

On the five newer arms the neural members are deferred to the final high-budget run (`neural: false` in config).

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

**RNA representations.** All are fitted in the fold:
- `all`: all 489 genes. For the logistic regression they are scaled by 0.5, which gives them 4× the L2 penalty of the clinical columns.
- `top15` / `top30` / `top50`: genes ranked by *selection frequency*, meaning how often an L1 logistic regression keeps them across 10 stratified half-samples of the training rows. This is the only RNA representation that reads the outcome, and only on fitting rows.
- `pca10` / `pca30`: principal components.
- `hallmark` / `reactome`: one score per gene set (40 Hallmark sets with at least 5 matched genes; 334 Reactome sets with at least 10). Each score is the mean z-score of the set's matched genes, using the gene crosswalk.
- `modules`: co-expression modules found on the fitting rows. Average linkage on 1 − |r|, cut at 0.7, at least 5 genes. Genes are sign-aligned to the module's first principal component and the score is their mean z-score.
- `dae8` / `dae32`: denoising-autoencoder latents (489 → 128 → latent, 20% input masking, 2 seeds; the seed with the lowest held-out reconstruction loss is kept). Offered to logistic regression, the SVM and XGBoost.

**DNA representations** (from the 173 binary flags; all unsupervised):
- `flags`: all 173.
- `known5`: TP53, PIK3CA, GATA3, CDH1, MAP3K1.
- `count`: the number of mutated genes.
- `pathway`: 25 Hallmark sets with at least 3 mutation genes, each flagged if any member gene is mutated.
- `bagemb`: bag-of-mutations embedding. Each gene gets an 8-dimensional vector from an SVD of the fitting rows' patient × gene matrix, and a patient's vector is the average over their mutated genes.

**Which representations each arm tries** (fixed in config, to bound the combinations):
- C+RNA: all eleven RNA representations.
- RNA: all, top30, pca10, hallmark, modules, dae8.
- C+RNA+DNA and RNA+DNA: {pca10, hallmark} × {flags, known5, pathway}.
- C+DNA and DNA: all five DNA representations.

## Members and grids (small for now; in `config.yaml` → `farm.grids`)

| Member | Grid | Notes |
|---|---|---|
| `logreg_en` | C ∈ {0.01, 0.1, 1}, l1_ratio ∈ {0, 0.5, 1} | saga solver; coefficients readable |
| `svm_rbf` | C ∈ {0.1, 1, 10}, gamma = scale | RBF only; probabilities by Platt scaling inside the fold |
| `random_forest` | 300 trees, max_features ∈ {sqrt, 0.2}, min_samples_leaf ∈ {5, 20} | |
| `xgboost` | 300 rounds, learning rate 0.05, depth ∈ {2, 4}, colsample ∈ {0.5, 0.8} | hist method |
| `mlp` | hidden ∈ {[32], [64, 32]}, dropout 0.3, weight decay 1e-3 | linear preprocessing (one-hot, scaled) |
| `mlp_embed` | hidden ∈ {[32], [64, 32]}, dropout 0.3, weight decay 1e-3 | one learned embedding per categorical column (codes last; 0 = unseen level) |
| `multimodal` | block dropout ∈ {0.1, 0.3}, dropout 0.3, weight decay 1e-3 | one encoder per block (clinical 16, RNA 32 units); C+RNA only; all genes |
| `logreg_cause` | as `logreg_en` | multinomial; C, C+RNA (pca10, dae8) and strict-clinical |
| `xgboost_cause` | as `xgboost` | multi:softprob; same arms and representations |

**Neural guard rails** (`farm.neural`):
- AdamW with weight decay, and dropout.
- Early stopping (patience 15, at most 200 epochs) on a stratified 15% validation split carved from the *fitting* rows only.
- Predictions averaged over 2 seeds.
- One CPU thread per fit, and every seed fixed.

On C+RNA the MLPs use only the `all` and `pca10` representations (`farm.neural_rna_representations`).

The full grid is 826 configurations: C 24, C+RNA 222, strict-clinical 24, C+DNA 100, C+RNA+DNA 120, RNA 116, RNA+DNA 120, DNA 100. Age enters as a plain linear term (no spline for now).

## Cause-specific members

The label `death_from_cancer` (alive / died of disease / died of other causes) is used **only to train** these two members, inside the folds. It is never an input, and every metric is computed on binary death as for all other members.

- **Why:** age mainly predicts other-cause death and tumour burden mainly predicts cancer death. Two cause-specific scores combine as "high risk through either route", which one additive score can only approximate.
- **Fair comparison:** each cause member is compared with its binary counterpart on the same inputs (on C+RNA via `<arm>|<member>@cause_reps`).
- **Permutation test:** it shuffles the full 3-class label.
- **Result:** all six comparisons are positive, about +0.005 AUROC, two of them significant. The 1-SE rule still prefers the simpler binary logistic regression for the arm headline.

## Selection and "simpler"

A procedure picks its winner by the harness rule: best mean inner AUROC, and the simplest configuration within 1 SE wins. "Simpler" is fixed in config (`complexity_rule`):
1. member family first (logistic regression < SVM < random forest = XGBoost < MLPs < multimodal);
2. then fewer omics inputs;
3. then stronger regularisation.

The procedures:
- `<arm>|<member>`: the member's own nested score;
- `<arm>|best`: the best member, chosen inside each fold, which is the arm's headline;
- `<arm>|rep:<representation>`: the best member for one representation. A representation is **kept only if** its paired 95% CI against the arm without that block lies entirely above 0. The baselines: C for C+RNA and C+DNA; C+RNA for C+RNA+DNA; RNA for RNA+DNA; the age-only floor for RNA and DNA.

## Parallel execution

Configurations are scored in parallel worker processes (`farm.n_jobs`; `auto` means logical cores minus 1). Each worker runs a single thread, so XGBoost, scikit-learn and numpy don't oversubscribe the CPU. Every fit is seeded and results are assembled in a fixed order, so the output is identical for any `n_jobs`; `tests/test_farm.py` checks this.

The autoencoder is unsupervised and deterministic, so it is cached by its exact fitting rows and settings. Configurations sharing it run in the same worker, which means it is trained once per fold rather than once per grid setting. Tests check that cached and fresh results are identical.

## Outputs (`outputs/<run>/farm/`)

| Output | Content |
|---|---|
| `configuration_fold_metrics` | Every configuration on every split |
| `procedure_winners` | The winner per procedure and split |
| `procedure_summary` | Mean and corrected 95% CI per procedure |
| `winner_stability` | How often the same configuration, member and representation win |
| `overfit_gate` | The gate applied to every procedure: `overfit` (train-CV gap above 0.05 or slope below 0.8), `recalibrate` (slope above 1.2: recalibrated probabilities are primary) or `pass`. Report only; it never removes a configuration from selection. |
| `paired_comparisons` | Each procedure against the age-only floor; strict against C; the omics comparisons (C+RNA, C+DNA, C+RNA+DNA against C; C+RNA+DNA against C+RNA; RNA+DNA against RNA; RNA, DNA against C); every representation against its baseline |
| `subgroup_metrics`, `reliability` | For the arm procedures |
| `regression_floor` | Provisional, until the final model |
| `permutation_test` | The full best-member procedure of `farm.permutation_arm` on shuffled labels (`farm.n_permutations` shuffles) |
| Figures | Procedure AUROC, winning members, calibration |
| `farm_summary.md` | Generated summary |

## Not yet included (final high-budget run)

- The neural members on the five newer arms.
- The with/without `tumor_stage` comparison.
- Larger grids, and 5 × 5 outer / 5 inner folds.

## Tests

`tests/test_farm.py`:
- parallel output equals serial;
- the top-k selector recovers planted signal genes;
- missing categories form their own level;
- the column rules (23 clinical, strict-clinical drops, treatment kept);
- linear preprocessing is one-hot with missing flags and learns its fill values from training rows only;
- the grid counts and the complexity order;
- a real-data smoke run.

`tests/test_representations.py`:
- gene-set scores equal the mean z-scores;
- modules recover planted structure with signs aligned;
- the autoencoder is reproducible, ignores the outcome, and its cache returns identical results;
- the DNA transforms compute correctly;
- the real gene-set counts (40 / 334 / 25);
- every representation is fitted on the fitting rows only.

`tests/test_neural.py`:
- the networks learn known signal;
- the same seed gives identical predictions;
- unseen category codes are tolerated;
- multimodal runs are reproducible;
- embedding codes come last and are non-negative.
