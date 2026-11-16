# RUNME: reproduce the results from a fresh copy

All commands run from this folder (the project root). Paths resolve from `config.yaml`; nothing is written outside the folder.

## Environment (dedicated project env, every package from pip)
```
conda create -n metabric-v2 python=3.14.3 pip -y
conda activate metabric-v2
pip install -r requirements.txt     # exact pins; torch from the PyTorch CPU wheel index
```
Use the activated env. Do not mix in conda's numpy/MKL packages: that brings a second OpenMP runtime and torch fails to import (FINDINGS F-014, F-027).

## Build steps done so far (1, 1a, 1b)
```
python run.py crosswalk             # 1a: writes reference/crosswalk/*.csv + a run with the coverage check
python run.py holdout               # ONCE ONLY: frozen 20% holdout -> outputs/frozen/holdout.csv, hash into config.yaml
python run.py eda --mode both       # 1b: descriptive (all patients) + modelling (training rows) EDA
python -m pytest -q                 # 37 tests (about 75 s; the EDA test reruns descriptive + modelling mode)
```
- `holdout` refuses to run if the holdout exists. Every later stage verifies the file against the hash in `config.yaml`.
- Each command creates `outputs/<timestamp>_<config hash>/` with `config.yaml`, `manifest.json` and `<stage>/{figures,tables,logs}`.
  `outputs/latest.json` names the newest completed run. Earlier runs are never overwritten.
- EDA outputs: `outputs/<run>/eda/tables/{descriptive,modelling}__eda_summary.md` (generated summary),
  `descriptive__reference_check.csv` (PLAN.md section 17 values), and one PNG + CSV per figure.
- pytest writes its temporary files to `.pytest_tmp/` inside this folder (safe to delete).
