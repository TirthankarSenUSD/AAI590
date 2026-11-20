# core: shared infrastructure

**Status:** built (build step 1). Used by every other stage.

This folder holds the code every stage relies on. It has no stage logic of its own and no command; it is used through the other stages.

## Files

| File | What it does |
|---|---|
| `paths.py` | `find_root()` walks up from the code to the project root (the folder holding both `config.yaml` and `run.py`, which are tracked in git); every path in the pipeline is resolved from there, so a copy of the folder placed anywhere works unchanged. `sha256_file()` hashes a file (used for frozen files, inputs and external artifacts). |
| `config.py` | `load_config()` reads `config.yaml` into a `Config` object. **Config hash:** a hash of the *parsed* YAML, so comments and whitespace do not change it; its first 8 characters go into every run_id. **Column blocks:** `Config.blocks` returns the explicit column lists. **Freezing:** `write_frozen_hash()` writes a frozen file's SHA-256 into config exactly once, as a targeted edit that keeps comments, and refuses if a hash is already there. |
| `artifacts.py` | The artifact writer and manifest (see below). |
| `checkpoint.py` | Resumable runs: finished tasks are saved as they complete (see below). |
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

## Resumable runs (`checkpoint.py`)

Long stages can be interrupted (a crash, a reboot, a Windows update) and resumed:

- Every `Run` activates a checkpoint folder, `outputs/checkpoints/<key>/`. The key is a hash of the config hash, the package source code and the command line.
- The two parallel loops save each finished task as it completes and look it up before running it:
  - `Harness.evaluate`, used by the farm (one task per group of configurations);
  - `importance.stage._parallel`, used by importance, ensemble, agreement and report.
- **To resume, rerun the same command.** Finished tasks are loaded, the rest are computed, and the log says how many were resumed. All fits are seeded, so the results are identical to an uninterrupted run.
- A different config, code or argument list gives a new key, so stale results are never reused.
- Writes are atomic (temporary file, then rename). A result that cannot be pickled is simply not saved.
- The folder is deleted when the run completes, and kept when it fails. The manifest records how many tasks were reused and saved (`checkpoint_tasks`).

## Tests

`tests/test_artifacts.py` (layout, manifest, unlisted-file detection, no overwrite), `tests/test_stats.py` (statistics against known values), `tests/test_portability.py` (a fresh copy elsewhere finds its own root), `tests/test_checkpoint.py` (a simulated crash, then a resume that refits only the unfinished tasks and matches an uninterrupted run; checkpoints kept on failure and removed on success; a new command never reuses them).
