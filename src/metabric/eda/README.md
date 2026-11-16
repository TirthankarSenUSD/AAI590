# eda: rerunnable exploratory data analysis

**Status:** built (build step 1b; item 10 completed in step 2). **Command:** `python run.py eda --mode {descriptive,modelling,both}`.

One command regenerates every EDA table and figure from the raw file. Reruns are deterministic: every table and figure CSV is byte-identical between runs.

## Two modes

| Mode | Rows | Purpose |
|---|---|---|
| `descriptive` | all 1,904 patients | Describes the cohort for the report. Labelled "full cohort, descriptive only"; **never** used for a modelling choice. Checked against 50 reference values. |
| `modelling` | the 1,523 training rows (holdout excluded) | The only EDA allowed to inform choices (merging rare levels, binning, which columns get missing flags). Also runs the split-balance checks. |

Output names carry the mode as a prefix (`descriptive__…`, `modelling__…`), and every figure is stamped with its mode label.

## What each section does (`sections.py`)

1. **`audit`:**
   - block sizes, per-column data type, missing count and distinct values;
   - the value domain of every categorical, binary and ordinal column;
   - the `'Positve'` typo count in the raw file;
   - mutation encoding (`'0'` against variant strings);
   - constant or near-constant columns (one value in at least 99% of rows);
   - the cohort coding.
2. **`outcome`:**
   - class counts and causes of death;
   - follow-up by class and cohort (box plot);
   - survivors with under 5 and under 10 years of follow-up;
   - patients with time 0;
   - death rate by cohort with Wilson intervals;
   - an overall Kaplan–Meier curve (descriptive).
3. **`missingness`:**
   - missing values per column and the most common patterns;
   - missingness by cohort (heatmap);
   - death rate with and without each missing value (Wilson intervals);
   - association of each missing indicator with cohort (Cramér's V, chi-square), age and chemotherapy.
4. **`clinical`:**
   - numeric summaries with outliers beyond 3 IQR;
   - categorical level counts with rare levels (under 20 patients);
   - histograms;
   - death rate by age band, node band, stage, grade, NPI band, PAM50 column, receptor status and cohort;
   - single-variable AUROC for each clinical column, for all deaths and with other-cause deaths set aside. Categorical columns are scored by their in-sample level death rate, which is optimistic and descriptive only.
5. **`treatment`:**
   - treatment frequency by cohort, age band, stage and PAM50 subtype;
   - crude death rate by treatment against a rate directly standardised over age band × stage;
   - stratum sizes for outcome × cohort × chemotherapy;
   - treatment combinations.
6. **`redundancy`:**
   - Spearman correlation of the numeric columns;
   - Cramér's V, plain and bias-corrected, between categorical columns;
   - identical and nested columns (the histology columns);
   - the NPI reconstruction, comparing the standard 0.2 formula with the file's 0.02 formula;
   - single-variable AUROC of the two NPI floors;
   - clinical-versus-molecular overlap: ERBB2 against HER2, GATA3 against ER, ER by IHC against ER status, PAM50 against the 3-gene subtype, mutation count against the number of mutated genes.
7. **`rna`:**
   - **Checks:** the z-score check.
   - **Per-gene association:** single-gene AUROC for death with a 1,000-permutation null and BH FDR.
   - **Correlation:** gene pairs above |r| 0.5 and 0.7; a clustered correlation heatmap; descriptive gene modules (average linkage on 1 − |r|).
   - **PCA:** the variance curve, and PC1–PC2 by outcome, by cohort and by PAM50 (small multiples).
   - **Clusters:** k-means clusters compared with `integrative_cluster` and the PAM50 column (ARI, NMI, Cramér's V).
8. **`dna`:**
   - mutation frequency, variant diversity and the top variants per gene;
   - co-occurrence and mutual exclusivity (φ);
   - single-gene association with death (Fisher exact, BH FDR, AUROC) for genes mutated in at least 1% of patients;
   - mutation burden per patient.
9. **`cohort`:**
   - clinical and outcome profile by cohort;
   - a classifier predicting cohort from RNA (L2 logistic regression, 5-fold out-of-fold AUROC), where a high value means batch structure;
   - feasibility of leave-one-cohort-out.

## Driver (`__init__.py`)

- **`run_eda(cfg, run, mode)`:** loads and validates the data. In modelling mode it also checks the frozen holdout's hash and restricts everything to the training rows. It then runs sections 1–9.
- **`split_diagnostics` (item 10, modelling mode only):**
  - balance of the holdout against the training rows;
  - `_fold_balance`, the balance of every frozen outer split (test part against training part), with an SMD heatmap;
  - stratified variables fail above a 2-point difference; others warn above |SMD| 0.10.
- **`reference_check` (descriptive only):** compares 50 computed numbers with the reference values in `config.yaml` (from PLAN.md section 17), within their tolerances. Any mismatch is logged and must be investigated.
- **`summary` (item 11):** writes `<mode>__eda_summary.md`. Every sentence is computed from the numbers and the wording is chosen by rule; nothing is typed by hand.
- **`facts` table:** every key number of the run.

`plots.py` holds the shared figure style:
- **Categorical colours:** a fixed order, never cycled.
- **Colour maps:** a one-hue sequential ramp and a blue–grey–red diverging map.
- **Marks and guides:** thin marks, a recessive grid, and a dot-plus-interval helper.

## Settings

`config.yaml` → `eda:` holds permutations, the FDR level, the rare-level minimum, thresholds, bands, the cluster k, the module cut, figure dpi and the split-balance rules; `reference_values:` holds the 50 checks.

## Tests

`tests/test_eda.py` runs both modes end to end in a throwaway project (with its own holdout and CV splits), then checks that:
- all reference values match;
- modelling mode uses only the training rows;
- the balance checks pass;
- every figure has a CSV and a caption.
