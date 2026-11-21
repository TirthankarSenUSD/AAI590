# METABRIC multi-omics survival classification (v2)

A reproducible machine-learning pipeline that predicts overall survival in breast cancer from clinical, gene-expression (RNA) and mutation (DNA) data in the METABRIC cohort (1,904 patients).

- **Phase 1:** binary classification of death (any cause). How well can death be predicted, are the probabilities calibrated, do RNA and DNA add anything beyond clinical data, and do very different models agree on why?
- **Phase 2:** a simple time-to-event (survival) analysis on the same data and splits.

## Status

| Step | Content | State |
|---|---|---|
| 1 | Data layer, configuration, artifact writer and manifest | done |
| 1a | Gene crosswalk between dataset columns and gene-set files | done |
| 1b | Rerunnable exploratory data analysis (EDA) | done |
| 2 | Evaluation harness (nested cross-validation, baselines) | done |
| 3a | Model farm: classical members (logistic regression, SVM, random forest, XGBoost) | done |
| 3b | Model farm: neural members (MLP, MLP with embeddings, multimodal) | done |
| 3c | Model farm: all eight data arms, RNA and DNA representations | done |
| 4 | Gates and importance | done |
| 5 | Ensemble and final model | done |
| 6 | Model-agreement analysis and omics ablation | done |
| 7 | Report and patient output | done |
| – | Final run (5 × 5 outer splits) and one-time holdout scoring | done |
| P2 | Phase 2: survival analysis, patient survival output, one-time holdout scoring | done |
| – | Pre-publication cleanup | next |

## Results

All numbers come from generated reports (see `RUNME.md`). Cross-validated (CV) numbers use 25 outer splits on the 1,523 training patients. The 381 holdout patients were scored once, at the end, with the saved models.

**Phase 1: death (any cause), binary**

| | Final model (clinical data) | Age-only baseline |
|---|---|---|
| AUROC, cross-validated | 0.765 (0.736–0.794) | 0.686 |
| AUROC, holdout | 0.784 (0.737–0.829) | 0.684 |
| Brier score, holdout | 0.184 | 0.222 |
| Calibration slope, holdout (1 = ideal) | 1.12 | 0.98 |

- The final model is a merged group of four near-duplicate members (elastic-net logistic regression, cause-specific logistic regression, MLP, MLP with embeddings). It uses raw probabilities, with no recalibration.
- **Do RNA and DNA add anything?** No arm with RNA or DNA was significantly better than clinical data alone (C+RNA +0.007, p = 0.18).
- **Do the models agree on why?**
  - Their predictions agree: median correlation 0.92.
  - Their feature rankings agree only partly: half of the top-10 features are shared on clinical data, and fewer once RNA is added.
  - A clinical core is consensus: age, cohort, positive lymph nodes, tumour size and surgery type.
- Label-permutation test: at chance (mean AUROC 0.503).

**Phase 2: overall survival, time to event**

| | Final model (ridge Cox, clinical) | NPI-only Cox | Age-only Cox |
|---|---|---|---|
| Harrell's C, cross-validated | 0.675 (0.649–0.702) | 0.635 | 0.590 |
| Harrell's C, holdout | 0.688 (0.652–0.724) | 0.623 | 0.589 |
| AUC at 5 / 10 years, holdout | 0.710 / 0.742 | 0.647 / 0.659 | 0.526 / 0.631 |

- **Why Cox over the random survival forest:** the forest ranked patients slightly better (C 0.684, not significant) but its probabilities were compressed (calibration slope about 2). The pre-set rule (simplest model within one standard error) chose the Cox model, which is well calibrated.
- **RNA:** it did not improve survival prediction either.
- **Risk groups:** observed 10-year survival on the holdout is 0.85, 0.64, 0.51 and 0.28 from the low to the high group.

## Pipeline stages

Each stage lives in its own folder under `src/metabric/`, with a `README.md` that explains exactly what it does: inputs, steps, outputs, rules, settings and tests.

| # | Stage | What it does | Command | State | Details |
|---|---|---|---|---|---|
| – | Core | Paths, config and hashing, artifact writer and manifest, statistics | – | built | [core/README.md](src/metabric/core/README.md) |
| 1 | Data layer | Read, validate, clean; target (death); mutation flags; standard NPI | – | built | [data/README.md](src/metabric/data/README.md) |
| 1a | Gene crosswalk | Dataset gene columns ↔ Hallmark and Reactome gene sets | `crosswalk` | built | [crosswalk/README.md](src/metabric/crosswalk/README.md) |
| 1 / 2 | Splits | Frozen 20% holdout; frozen nested CV splits | `holdout`, `splits` | built | [splits/README.md](src/metabric/splits/README.md) |
| 1b | EDA | Descriptive and modelling-mode EDA, reference check, split balance | `eda` | built | [eda/README.md](src/metabric/eda/README.md) |
| 2 | Evaluation harness | Nested CV scoring, selection rule, floors, statistical checks | `harness` | built | [harness/README.md](src/metabric/harness/README.md) |
| 3 | Model farm | Seven models (four classical, three neural) across data arms | `farm` | built | [farm/README.md](src/metabric/farm/README.md) |
| 4 | Importance | Gates, SHAP and permutation importance, partial dependence, interactions | `importance` | built | [importance/README.md](src/metabric/importance/README.md) |
| 5 | Ensemble | Fused arm, agreement merge, stacker, final model and regression floor | `ensemble` | built | [ensemble/README.md](src/metabric/ensemble/README.md) |
| 6 | Agreement | Model-agreement analysis, disagreement flag, cause breakdown, omics ablation | `agreement` | built | [agreement/README.md](src/metabric/agreement/README.md) |
| 7 | Report | Generated validation report, final-model deployment fit, patient-level output; one-time holdout scoring | `report`, `score-holdout` | built | [report/README.md](src/metabric/report/README.md) |
| P2 | Phase 2 | Survival analysis (KM, PH checks, Cox / penalised Cox / random survival forest, patient survival output, one-time holdout) on the same splits | `phase2 --step checks|models|final|holdout` | built | [phase2/README.md](src/metabric/phase2/README.md) |

## Design principles

- **No leakage.** Every fitted step runs inside cross-validation folds. A 20% holdout (381 patients, stratified on outcome × cohort × chemotherapy) was created once, is hash-checked on every run and is scored only once, at the end.
- **Positive class is death**, so every probability reads as a risk of death.
- **Explicit column blocks:** 27 clinical, 489 RNA (z-scores) and 173 mutation columns, listed in `config.yaml` and validated against the file on every load.
- **Every run writes its artifacts to disk** under its own run folder, with a manifest that registers each file. Earlier runs are never overwritten.
- **Report text is generated from results**, never typed by hand.

## Repository layout

```
├── run.py                  command-line entry point
├── config.yaml             all settings, column lists, reference values
├── requirements.txt        exact package versions
├── requirements-survival.txt  Phase 2 survival package (installed without dependencies)
├── src/metabric/           pipeline code, one folder per stage, each with a README.md
│   ├── cli.py              commands behind run.py
│   ├── core/               paths, config, artifacts and manifest, statistics
│   ├── data/               data layer
│   ├── crosswalk/          gene crosswalk
│   ├── splits/             frozen holdout and CV splits
│   ├── eda/                exploratory data analysis
│   ├── harness/            evaluation harness and floors
│   ├── farm/              model farm (seven members)
│   ├── importance/        gates and feature importance
│   ├── ensemble/          fusion, ensemble and final model
│   ├── agreement/         model agreement and the omics ablation
│   ├── report/            validation report and patient output
│   └── phase2/            survival analysis
├── tests/                  test suite (pytest)
├── data/                   METABRIC_RNA_Mutation.csv (read-only)
├── reference/              gene-set files; crosswalk/ is built from them
├── outputs/                run artifacts (created by the pipeline)
├── PLAN.md                 design, requirements and decision log
├── FINDINGS.md             build log of findings and decisions (append-only)
└── RUNME.md                reproduction commands
```

## Setup

Requires conda (or any Python 3.14 environment). All packages come from pip; PyTorch comes from its CPU wheel index (declared in `requirements.txt`).

```bash
conda create -n metabric-v2 python=3.14.3 pip -y
conda activate metabric-v2
pip install -r requirements.txt
pip install --no-deps -r requirements-survival.txt   # scikit-survival for Phase 2 (its unused dependency ecos has no Python 3.14 wheel)
```

Do not mix conda's numpy/MKL packages into this environment: a second OpenMP runtime stops PyTorch from importing on Windows.

## Usage

Run from the repository root:

```bash
python run.py crosswalk          # build reference/crosswalk/ from the gene-set files
python run.py holdout            # create the frozen holdout (once only; refuses if it exists)
python run.py splits             # create the frozen nested CV splits (once only; refuses if they exist)
python run.py eda --mode both    # EDA: descriptive (all patients) and modelling (training rows only)
python run.py harness            # baselines ("floors") through the nested cross-validation
python run.py farm --arms C      # model farm through the same nested cross-validation (one run per arm; see RUNME.md)
python run.py importance         # gates and feature importance of the farm winners
python run.py ensemble  --farm-runs ...                       # fusion, ensembles and the final model
python run.py agreement --farm-runs ... --importance-run ...  # model agreement and the omics ablation
python run.py report    --farm-runs ... --ensemble-run ... --agreement-run ...   # validation report, final model, patient output
python run.py score-holdout --report-run REP                  # ONCE ONLY: score the holdout (--dry-run to check first)
python run.py phase2 --step checks                            # Phase 2: Kaplan-Meier, proportional hazards, collinearity
python run.py phase2 --step models                            # Phase 2: survival models through the nested CV
python run.py phase2 --step final --models-run M              # Phase 2: final survival model and patient survival output
python run.py phase2 --step holdout --final-run F             # ONCE ONLY: score the holdout for Phase 2
python -m pytest -q              # run the test suite (113 tests, about 25 min)
```

The full final run (all arms, 5 × 5 outer splits) takes about 14 hours on 8 cores. Long runs resume from checkpoints when rerun with the same command. `RUNME.md` lists the exact order.

Each command creates `outputs/<timestamp>_<config hash>/` containing a copy of the config, `manifest.json` and `<stage>/{figures,tables,logs}`. `outputs/latest.json` points at the newest completed run. Every figure is a PNG with a CSV of the plotted numbers and a caption in the manifest.

### EDA modes

- `descriptive` uses all 1,904 patients and describes the cohort for reporting. It never informs modelling choices.
- `modelling` uses the 1,523 training rows only and also checks holdout balance.

Each mode writes a generated summary (`eda/tables/<mode>__eda_summary.md`). The descriptive run is checked against 50 reference values (`descriptive__reference_check.csv`).

### Evaluation harness

All models are scored on the same frozen nested cross-validation: 5 outer folds × 5 repeats on the 1,523 training patients, with 3 inner folds inside each outer training set.
- **Inner folds** make every choice: model selection (simplest model within one standard error of the best), the operating-point threshold, and recalibration.
- **Outer folds** report the score.
- **Comparisons** between models are paired and use the corrected resampled t-test.
- **Checks:**
  - a label-permutation test, which must score at chance;
  - leave-one-cohort-out;
  - results within treatment subgroups;
  - an overfit gate (train-versus-CV gap, calibration slope).

Baselines on the training rows (mean outer-fold AUROC, 95% CI):

| Baseline | AUROC |
|---|---|
| Majority class | 0.500 |
| Age only | 0.686 (0.657–0.714) |
| NPI as supplied | 0.596 (0.566–0.625) |
| Standard NPI | 0.616 (0.586–0.645) |

## Data and gene sets

- `data/METABRIC_RNA_Mutation.csv`: 1,904 patients × 693 columns. `overall_survival` is 1 = alive and 0 = died; the pipeline models death = 1 − `overall_survival`.
- `reference/h.all.v2026.2.Hs.symbols.gmt`: MSigDB Hallmark gene sets (50 sets).
- `reference/ReactomePathways.gmt`: Reactome pathways (2,874 sets).
- `reference/crosswalk/symbol_aliases.csv`: hand-maintained renamed gene symbols.

## Selected EDA findings

The descriptive EDA reproduces all 50 reference values. Highlights, with details in `FINDINGS.md`:

- **The label mixes causes and ignores time.** 43.5% of deaths are from causes other than breast cancer. 32% of survivors have under 10 years of follow-up.
- **Clinical signal:**
  - Age alone gives AUROC 0.686 for any death, falling to 0.593 once other-cause deaths are set aside.
  - Tumour burden (NPI, nodes, size) gains when other-cause deaths are set aside.
  - In the supplied data, the Nottingham Prognostic Index (NPI) uses a 0.02 per cm tumour-size weight instead of the standard 0.2, so a standard NPI is computed as a second baseline.
- **Molecular signal is weak per gene but widespread.** Single-gene AUROCs range from 0.37 to 0.61, yet about half the genes are significant after false-discovery-rate correction. Among mutations, only GATA3 survives correction.
- **Treatment is confounded by indication.** Patients who had chemotherapy have a lower crude death rate. After adjusting for age and stage, the direction reverses. Treatment variables are read as risk given the treatment received, never as treatment effects.
- **Strong batch structure:** cohort can be predicted from RNA with AUROC 0.97.
