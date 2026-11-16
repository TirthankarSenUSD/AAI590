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
| 2 | Evaluation harness (nested cross-validation, baselines) | next |
| 3–7 | Model farm, importance, ensemble, model-agreement analysis, report | planned |

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
├── src/metabric/           pipeline code (data, crosswalk, splits, artifacts, stats, eda/)
├── tests/                  test suite (pytest)
├── data/                   METABRIC_RNA_Mutation.csv (read-only)
├── reference/              gene-set files; crosswalk/ is built from them
├── outputs/                run artifacts (created by the pipeline)
├── diagram/                pipeline diagram (HTML) and its generator
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
```

Do not mix conda's numpy/MKL packages into this environment: a second OpenMP runtime stops PyTorch from importing on Windows.

## Usage

Run from the repository root:

```bash
python run.py crosswalk          # build reference/crosswalk/ from the gene-set files
python run.py holdout            # create the frozen holdout (once only; refuses if it exists)
python run.py eda --mode both    # EDA: descriptive (all patients) and modelling (training rows only)
python -m pytest -q              # run the test suite (about 75 s)
```

Each command creates `outputs/<timestamp>_<config hash>/` containing a copy of the config, `manifest.json` and `<stage>/{figures,tables,logs}`. `outputs/latest.json` points at the newest completed run. Every figure is a PNG with a CSV of the plotted numbers and a caption in the manifest.

### EDA modes

- `descriptive` uses all 1,904 patients and describes the cohort for reporting. It never informs modelling choices.
- `modelling` uses the 1,523 training rows only and also checks holdout balance.

Each mode writes a generated summary (`eda/tables/<mode>__eda_summary.md`). The descriptive run is checked against 50 reference values (`descriptive__reference_check.csv`).

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
