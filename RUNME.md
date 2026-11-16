# RUNME: reproduce the results from a fresh copy

All commands run from this folder (the project root). Paths resolve from `config.yaml`; nothing is written outside the folder.

## Environment (dedicated project env, every package from pip)
```
conda create -n metabric-v2 python=3.14.3 pip -y
conda activate metabric-v2
pip install -r requirements.txt     # exact pins; torch from the PyTorch CPU wheel index
```
Use the activated env. Do not mix in conda's numpy/MKL packages: that brings a second OpenMP runtime and torch fails to import (FINDINGS F-014, F-027).

## Build steps done so far (1, 1a, 1b, 2, 3a)
```
python run.py crosswalk             # 1a: writes reference/crosswalk/*.csv + a run with the coverage check
python run.py holdout               # ONCE ONLY: frozen 20% holdout -> outputs/frozen/holdout.csv, hash into config.yaml
python run.py splits                # ONCE ONLY: frozen nested CV splits on the training rows -> outputs/frozen/cv_splits.csv
python run.py eda --mode both       # 1b: descriptive (all patients) + modelling (training rows, holdout + fold balance) EDA
python run.py harness               # 2: floors through the nested CV (about 45 s)
python run.py farm                  # 3a: 160 configurations + permutation test (about 50 min with farm.n_jobs = auto on 8 cores)
python -m pytest -q                 # 47 tests (about 70 s; synthetic harness tests + EDA rerun)
```
- `holdout` and `splits` refuse to run if their file exists. Every later stage verifies both files against the hashes in `config.yaml`.
- Harness outputs: `outputs/<run>/harness/tables/harness_summary.md` (generated summary), `summary.csv`, `paired_comparisons.csv`,
  `subgroup_metrics.csv`, `overfit_gate.csv`, `permutation_summary.csv`, `loco.csv`, `regression_floor.csv`, plus figures.
- Each command creates `outputs/<timestamp>_<config hash>/` with `config.yaml`, `manifest.json` and `<stage>/{figures,tables,logs}`.
  `outputs/latest.json` names the newest completed run. Earlier runs are never overwritten.
- EDA outputs: `outputs/<run>/eda/tables/{descriptive,modelling}__eda_summary.md` (generated summary),
  `descriptive__reference_check.csv` (PLAN.md section 17 values), and one PNG + CSV per figure.
- pytest writes its temporary files to `.pytest_tmp/` inside this folder (safe to delete).
