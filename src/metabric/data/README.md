# data: the data layer

**Status:** built (build step 1). Used by every stage that reads the dataset.

Turns `data/METABRIC_RNA_Mutation.csv` into a validated, cleaned `Dataset`. Nothing here is fitted to the data: anything learned from the data (imputation, scaling, encoding, selection) happens later, inside cross-validation folds.

## What `load_dataset(cfg)` does, in order

1. **Read** (`read_raw`): mutation columns are read as strings.
2. **Validate** (`validate_raw`): stops with an error listing every problem found.
   - **Shape:** 1,904 rows and 693 columns; `patient_id` unique; no duplicate rows.
   - **Columns against `config.yaml`:** the file's columns match the explicit lists exactly, by name and position. `patient_id` comes first, then the 27 clinical and 3 outcome columns, the 489 RNA columns right after `death_from_cancer`, and the 173 `_mut` columns last.
   - **RNA block:** all float, no missing values. The z-score check (|mean| ≤ 0.05, SD within 0.05 of 1) is a consistency check only; block membership comes from the explicit lists.
   - **Mutation block:** no missing values (the binarisation rule would mark them as mutated); no column that looks numeric.
   - **Target:** `overall_survival` is 0/1.
3. **Clean** (`clean`): the source typo `'Positve'` in `er_status_measured_by_ihc` becomes `'Positive'`, and `cohort` (stored as 1.0–5.0) becomes a categorical with levels 1–5. The input frame is never modified.
4. **Target** (`make_target`): `death = 1 − overall_survival`. The positive class is death, so every probability reads as risk of death.
5. **Mutation flags** (`binarise_mutations`): `'0'` → 0 and any variant string (for example `H1047R`) → 1, via `astype(str) != '0'`. The variant strings stay in the frame for later variant-level work.

The result is a `Dataset` with:
- `frame`: the cleaned file, indexed by `patient_id`;
- `y`: death 0/1;
- `mutations`: 173 binary flags;
- `blocks`: the column lists.

`Dataset.features(blocks)` returns clinical, RNA and/or DNA features and can never include the outcome columns or the ID. `Dataset.subset(ids)` restricts every part to the given patients, for example the training rows.

## Helpers

- **`missing_indicators(df, cols)`:** one `<col>__missing` 0/1 column per listed column. The list must be chosen on training rows.
- **`npi_standard(frame)`:** the standard Nottingham Prognostic Index, 0.2 × size(cm) + node stage + grade.
  - **Node stage:** 0 positive nodes → 1, 1–3 → 2, 4 or more → 3.
  - **Missing values:** the result is missing if any input is missing (89 patients).
  - **Why both exist:** the file's own `nottingham_prognostic_index` uses a 0.02 size weight instead (FINDINGS F-021). Both are kept as floors: the file's version is primary, the standard one secondary (F-028).

## Tests

`tests/test_data.py`:
- mutation binarisation;
- refusal of missing mutation values;
- the block split (no overlaps, nothing left out, no outcome or ID in the features);
- target orientation (1,103 deaths);
- the typo fix and cohort coding;
- loud validation failures;
- missing indicators;
- the NPI formula;
- a guard that the file's NPI uses the 0.02 weight.
