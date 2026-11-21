# crosswalk: dataset genes ↔ gene-set files

**Status:** built (build step 1a). **Command:** `python run.py crosswalk`.

The dataset's gene columns use lower-case symbols: RNA columns like `brca1`, mutation columns like `tp53_mut`, plus one unusual symbol, `hla-g`. The gene-set files use upper-case symbols. This stage builds the lookup between the two once, so every later stage (module and signature scores, importance interpretation, the patient output, the report) uses the same matches instead of matching on the fly.

## Inputs

- The dataset's column names (from the explicit lists in `config.yaml`).
- Exactly the two files named in `config.yaml` under `gene_sets.files`: `reference/h.all.v2026.2.Hs.symbols.gmt` (MSigDB Hallmark, 50 sets) and `reference/ReactomePathways.gmt` (Reactome, 2,874 sets). Any other file in `reference/` is ignored.
- `reference/crosswalk/symbol_aliases.csv`: a hand-maintained list of renamed symbols (currently `HIST1H2BC → H2BC4`).

## What it does

1. **Normalise:** each dataset gene column becomes a symbol by upper-casing and removing `_mut` (`tp53_mut` → `TP53`).
2. **Aliases:** applied to both sides. Every alias applied to a set member is logged, and nothing is ever guessed.
3. **Match:** exact symbol equality only, with no partial-string matching. A gene measured in both blocks (168 of the 173 mutation genes also have an RNA column) gets one row per block.
4. **Count per set:** matched RNA and DNA genes, with pass/fail on the minimum overlap. A set needs at least 5 matched RNA genes (`gene_sets.min_rna_overlap`) to be used for scores; sets below that stay in the tables for lookup.
5. **Check coverage:** compares against `gene_sets.expected_coverage` and logs a warning on any difference.

## Outputs

Written to `reference/crosswalk/` and registered in the run manifest with SHA-256:

| File | Content |
|---|---|
| `gene_crosswalk.csv` | One row per dataset gene feature: column, block (`rna`/`dna`), canonical symbol, matched, reason if not, number of sets per source, alias applied. |
| `gene_set_membership.csv` | Long form: set, source, source file, set size, matched column, block, symbol. |
| `gene_set_summary.csv` | Per set: size, matched RNA and DNA genes, pass/fail on the minimum overlap. |
| `unmatched_genes.csv` | Dataset genes found in no set (77). |

Each run also writes `crosswalk/tables/coverage.csv`, `coverage_check.csv` and `alias_log.csv`. It also saves a **snapshot** of the four tables (`snapshot_<table>.csv`) in its own run folder. `reference/crosswalk/` is rebuilt in place by later runs, so this keeps every run's provenance verifiable.

Current coverage:
- **Hallmark:** 260 of 489 RNA and 71 of 173 DNA genes; 40 of 50 sets usable for scores.
- **Reactome:** 428 RNA and 141 DNA genes; 647 of 2,874 sets usable, 334 with at least 10 genes, median overlap 1.

## Lookup helpers (`GeneSetIndex`)

| Call | Returns |
|---|---|
| `genes_of_set(name, source=None, block="rna", usable_only=False)` | The dataset columns in a set. |
| `sets_of(gene_or_column, source=None)` | The sets that contain a gene. |
| `usable_sets(source=None)` | The sets that pass the minimum overlap. |

Every answer names its source file.

## Rules

- Annotation only: this stage never reads outcome labels.
- A symbol mismatch is reported in `unmatched_genes.csv` and fixed only by an explicit alias in `symbol_aliases.csv`, never by guessing.

## Tests

`tests/test_crosswalk.py`: case-insensitive exact matching, no partial matches, both-block rows, unconfigured files ignored, alias applied and logged, lookup helpers, and a sanity check on the real tables.
