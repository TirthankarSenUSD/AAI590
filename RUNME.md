# RUNME: reproduce the results from a fresh copy

All commands run from this folder (the project root). Paths resolve from `config.yaml`; nothing is written outside the folder.

## Environment (dedicated project env, every package from pip)
```
conda create -n metabric-v2 python=3.14.3 pip -y
conda activate metabric-v2
pip install -r requirements.txt     # exact pins; torch from the PyTorch CPU wheel index
pip install --no-deps -r requirements-survival.txt   # Phase 2: scikit-survival without ecos (no Python 3.14 wheel; unused)
```
Use the activated env. Do not mix in conda's numpy/MKL packages: that brings a second OpenMP runtime and torch fails to import.

## Build steps done so far (1, 1a, 1b, 2, 3, 4)
```
python run.py crosswalk             # 1a: writes reference/crosswalk/*.csv + a run with the coverage check
python run.py holdout               # ONCE ONLY: frozen 20% holdout -> outputs/frozen/holdout.csv, hash into config.yaml
python run.py splits                # ONCE ONLY: frozen nested CV splits on the training rows -> outputs/frozen/cv_splits.csv
python run.py eda --mode both       # 1b: descriptive (all patients) + modelling (training rows, holdout + fold balance) EDA
python run.py harness               # 2: floors through the nested CV (about 45 s)
python run.py farm                  # 3: 826 configurations, 8 arms + permutation test (about 3.5 h with farm.n_jobs = auto on 8 cores)
python run.py importance            # 4: explains the newest farm run's winners (about 2.5 h; --reuse RUN_ID rebuilds summaries in 15 min)
python run.py ensemble --farm-runs FARM_3C,FARM_CAUSE                     # 5: ensembles and the final model
python run.py agreement --farm-runs FARM_3C,FARM_CAUSE --importance-run IMP # 6: model agreement and the omics ablation
python run.py report --farm-runs FARM_3C,FARM_CAUSE --ensemble-run ENS --agreement-run AGR
                                    # 7: validation report, final-model fit (models/final_model.joblib), patient output
python run.py score-holdout --report-run REP --dry-run   # checks the code path with shuffled labels (scores nothing)
python run.py score-holdout --report-run REP             # ONCE ONLY, last: scores the holdout; refuses a second time
python run.py phase2 --step checks                       # Phase 2a: Kaplan-Meier, proportional hazards, collinearity
python run.py phase2 --step models                       # Phase 2b: survival members through the nested CV (about 10 min)
python run.py phase2 --step final --models-run M         # Phase 2c: final survival model + patient survival output
python run.py phase2 --step holdout --final-run F --dry-run   # shuffled outcomes; scores nothing
python run.py phase2 --step holdout --final-run F        # ONCE ONLY: scores the holdout for Phase 2
python run.py report-pack --runs-file outputs/final_runs.yaml [--findings LOG.md]   # figures + report_material.md (seconds)
python -m pytest -q                 # 116 tests (about 25 min; includes small end-to-end farm, importance and ensemble runs)
```
## Full run (5x5 outer, 3 inner; about 14 h on 8 cores)
Run in this order. Each line is one run; if one stops, rerun the same line (it resumes from its checkpoints).
```
python run.py harness
python run.py eda --mode modelling
python run.py farm --arms C            # then the same for C+RNA, strict_clinical, C+DNA, C+RNA+DNA, RNA, RNA+DNA, DNA, C_diag
python run.py farm --members logreg_en,xgboost,logreg_cause,xgboost_cause --arms C,C+RNA,strict_clinical
python run.py farm --members logreg_en,svm_rbf,random_forest,xgboost --arms fused
python run.py ensemble  --farm-runs <the 11 farm run ids above, in that order>
python run.py agreement --farm-runs <same list> --importance-run <importance run id>
python run.py report    --farm-runs <same list> --ensemble-run <ensemble id> --agreement-run <agreement id>
```

- **Resuming:** if a run stops (crash, reboot), rerun the same command unchanged. Finished tasks are loaded from
  `outputs/checkpoints/` and only the rest is computed; the results are identical. Do not edit `config.yaml` or the code
  in between, or the run starts from scratch. The checkpoints are deleted when the run completes.
- `holdout` and `splits` refuse to run if their file exists. Every later stage verifies both files against the hashes in `config.yaml`.
- Harness outputs: `outputs/<run>/harness/tables/harness_summary.md` (generated summary), `summary.csv`, `paired_comparisons.csv`,
  `subgroup_metrics.csv`, `overfit_gate.csv`, `permutation_summary.csv`, `loco.csv`, `regression_floor.csv`, plus figures.
- Each command creates `outputs/<timestamp>_<config hash>/` with `config.yaml`, `manifest.json` and `<stage>/{figures,tables,logs}`.
  `outputs/latest.json` names the newest completed run. Earlier runs are never overwritten.
- EDA outputs: `outputs/<run>/eda/tables/{descriptive,modelling}__eda_summary.md` (generated summary),
  `descriptive__reference_check.csv` (the reference values in `config.yaml`), and one PNG + CSV per figure.
- pytest writes its temporary files to `.pytest_tmp/` inside this folder (safe to delete).
