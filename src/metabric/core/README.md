# core: shared infrastructure

**Status:** built (build step 1). Used by every other stage.

This folder holds the code every stage relies on. It has no stage logic of its own and no command; it is used through the other stages.

## Files

| File | What it does |
|---|---|
| `paths.py` | `find_root()` walks up from the code to the project root (the folder holding both `config.yaml` and `run.py`, which are tracked in git); every path in the pipeline is resolved from there, so a copy of the folder placed anywhere works unchanged. `sha256_file()` hashes a file (used for frozen files, inputs and external artifacts). |
| `config.py` | `load_config()` reads `config.yaml` into a `Config` object. **Config hash:** a hash of the *parsed* YAML, so comments and whitespace do not change it; its first 8 characters go into every run_id. **Column blocks:** `Config.blocks` returns the explicit column lists. **Freezing:** `write_frozen_hash()` writes a frozen file's SHA-256 into config exactly once, as a targeted edit that keeps comments, and refuses if a hash is already there. |
| `artifacts.py` | The artifact writer and manifest (see below). |
| `stats.py` | Small statistics with no heavy dependencies: Mann–Whitney AUROC (single and vectorised), a permutation null for AUROC, permutation p-values, Benjamini–Hochberg FDR, Wilson intervals, death-rate tables, Cramér's V (plain and bias-corrected), Kaplan–Meier with Greenwood log-log intervals, the standardised mean difference, and the Nadeau–Bengio corrected resampled t-test. |

## Artifact writer (`artifacts.py`)

Every command is one **run**:

```
outputs/<run_id>/                 run_id = <local timestamp>_<first 8 chars of the config hash>
├── config.yaml                   copy of the config used
├── manifest.json                 every file the run wrote
└── <stage>/{figures,tables,logs}/
```

- **`Run(cfg, command)`** creates the run folder. It never overwrites one: a name clash gets a `-2` suffix.
- **`run.stage(name)`** returns a `Stage`. Allowed names: crosswalk, eda, splits, harness, farm, importance, ensemble, agreement, report, phase2, models.
- **Writing:**
  - `stage.table(name, df, caption)` writes a CSV.
  - `stage.figure(name, fig, data, caption)` writes a PNG at the configured dpi *and* a CSV of exactly the plotted numbers.
  - `stage.text(name, content, caption)` writes generated Markdown.
  - Every file is registered in the manifest with path, stage, run_id, config hash, seed, the creating function, the creation time and its caption.
- **Recording files outside the run folder:**
  - `run.record_input(path, role)` records an input file (size, SHA-256).
  - `run.record_external(path, role)` records a generated file that lives outside the run folder, such as `reference/crosswalk/*.csv`.
- **`run.finalize()`** validates the run: every listed file must exist, external files must still match their hash, and no file in the run folder may be unlisted. It then marks the run `complete` and points `outputs/latest.json` at it. A failed validation marks the run `invalid`; an exception marks it `failed`, and the folder is kept as a record.
- **Windows file locks:** manifest writes retry, because Windows can briefly lock the file (FINDINGS F-019).

## Tests

`tests/test_artifacts.py` (layout, manifest, unlisted-file detection, no overwrite), `tests/test_stats.py` (statistics against known values), `tests/test_portability.py` (a fresh copy elsewhere finds its own root).
