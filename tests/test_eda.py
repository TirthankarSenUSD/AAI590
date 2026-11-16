"""End-to-end EDA in a throwaway project root: reference values must all match, artifacts must
validate, and modelling mode must use training rows only."""
import json

import pandas as pd
import pytest

from metabric.core.artifacts import Run
from metabric.data import load_dataset
from metabric.eda import run_eda
from metabric.splits import make_holdout


@pytest.fixture(scope="module")
def eda_run(tmp_path_factory, cfg):
    import shutil
    import re
    from pathlib import Path
    import yaml
    from metabric.core.config import Config
    root = tmp_path_factory.mktemp("edaproj")
    text = re.sub(r'(\n\s+(?:splits_)?sha256:\s*)"[0-9a-f]+"', r"\1null", (cfg.root / "config.yaml").read_text(encoding="utf-8"))
    text = text.replace("permutations: 1000", "permutations: 200")   # faster; reference values do not depend on it
    (root / "config.yaml").write_text(text, encoding="utf-8")
    (root / "run.py").write_text("# root marker", encoding="utf-8")
    (root / "data").mkdir()
    shutil.copyfile(cfg.path("data"), root / "data" / Path(cfg.raw["paths"]["data"]).name)
    c = Config(root=root, raw=yaml.safe_load(text))
    full = load_dataset(c)
    c, _, _ = make_holdout(c, full)
    from metabric.splits import make_cv_splits, training_ids
    c, _, _ = make_cv_splits(c, full.subset(training_ids(c, full)))
    run = Run(c, "test eda", run_id="eda_test")
    facts_d = run_eda(c, run, "descriptive")
    facts_m = run_eda(c, run, "modelling")
    run.finalize()
    return run, facts_d, facts_m


def test_reference_values_all_match(eda_run):
    _, fd, _ = eda_run
    assert fd["n_reference_mismatch"] == 0, fd["reference_mismatches"]


def test_modelling_mode_uses_training_rows_only(eda_run):
    _, _, fm = eda_run
    assert fm["n_patients"] == 1523 and fm["n_train"] == 1523 and fm["n_holdout"] == 381
    assert fm["n_balance_fail"] == 0
    assert fm["cv_splits_present"] and fm["n_cv_splits"] == 10 and fm["n_fold_balance_fail"] == 0


def test_every_figure_has_csv_and_caption(eda_run):
    run, _, _ = eda_run
    m = json.loads((run.dir / "manifest.json").read_text())
    assert m["status"] == "complete"
    figs = [a for a in m["artifacts"] if a["kind"] == "figure"]
    assert len(figs) > 20
    for a in figs:
        assert a["caption"] and a["data"].endswith(".csv") and (run.cfg.root / a["data"]).is_file()
    summ = (run.dir / "eda" / "tables" / "descriptive__eda_summary.md").read_text()
    assert "All 50 reference values match" in summ
    assert pd.read_csv(run.dir / "eda" / "tables" / "modelling__split_balance.csv").status.eq("ok").all()
