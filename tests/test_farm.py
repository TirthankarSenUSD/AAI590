"""Farm tests: preprocessing hygiene, top-k selection, candidate construction, and that parallel
scoring gives exactly the same result as serial scoring."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from metabric.farm import build_candidates
from metabric.farm.preprocess import CategoricalAsString, FrequencySelector, column_groups, preprocessor
from metabric.harness import Candidate, Harness

from test_harness import HCFG, make_splits, synth


def test_parallel_equals_serial():
    X, y, _ = synth(n=500)
    H = Harness(HCFG, X, y, make_splits(y))
    cands = [Candidate(f"m{c}", lambda c=c: LogisticRegression(C=c, max_iter=500), ["strong", "weak", "noise"])
             for c in (0.01, 0.1, 1.0)]
    a = H.evaluate(cands, select=True, n_jobs=1)
    b = H.evaluate(cands, select=True, n_jobs=2)
    pd.testing.assert_frame_equal(a.folds, b.folds)
    pd.testing.assert_frame_equal(a.preds, b.preds)
    pd.testing.assert_frame_equal(a.selection, b.selection)
    lines = []
    c = H.evaluate(cands, select=True, n_jobs=2, groups=[[2, 0], [1]], log=lines.append)   # grouped: same result
    assert lines and "2/2 tasks" in lines[-1]                                   # progress reported
    pd.testing.assert_frame_equal(a.folds, c.folds)
    pd.testing.assert_frame_equal(a.preds, c.preds)


def test_frequency_selector_finds_signal_genes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 200))
    y = (X[:, 3] + X[:, 7] - X[:, 11] + rng.normal(0, 1, 600) > 0).astype(int)
    sel = FrequencySelector(k=5, random_state=1).fit(X, y)
    assert {3, 7, 11} <= set(sel.selected_)
    assert sel.transform(X).shape == (600, 5)


def test_categorical_missing_is_its_own_level():
    out = CategoricalAsString().fit_transform(pd.DataFrame({"a": ["x", None, "y"], "b": pd.Categorical([1, 2, None])}))
    assert out.tolist() == [["x", "1"], ["missing", "2"], ["y", "missing"]]


def test_column_rules(cfg):
    g = column_groups(cfg, "C")
    clin = g["numeric"] + g["binary"] + g["categorical"]
    for dropped in ("oncotree_code", "cancer_type_detailed", "cancer_type", "nottingham_prognostic_index"):
        assert dropped not in clin
    assert len(clin) == 23 and not g["rna"]
    s = column_groups(cfg, "strict_clinical")
    assert not set(cfg["groups"]["molecular_derived"]) & set(s["numeric"] + s["binary"] + s["categorical"])
    assert set(cfg["groups"]["treatment"]) <= set(s["binary"] + s["categorical"])        # treatment stays in
    assert len(column_groups(cfg, "C+RNA")["rna"]) == 489


def test_linear_preprocessing_is_one_hot_and_fitted_on_training_rows_only(cfg, ds):
    g = column_groups(cfg, "C")
    cols = g["numeric"] + g["binary"] + g["categorical"]
    f = ds.frame[cols]
    pre = preprocessor(g, "linear", None, cfg, 0).fit(f.iloc[:800])
    Z = pre.transform(f.iloc[800:])
    assert not np.isnan(Z).any()
    names = list(pre.get_feature_names_out())
    assert any(n.startswith("cohort_") for n in names)                  # one-hot, not an integer code (A1)
    assert any("missingindicator" in n for n in names)                  # missing flags kept (A12)
    med = ds.frame["tumor_size"].iloc[:800].median()
    num = pre.named_transformers_["num"].named_steps["simpleimputer"]
    assert num.statistics_[g["numeric"].index("tumor_size")] == pytest.approx(med)   # learned from training rows


def test_candidate_grid_counts(cfg):
    cands, meta = build_candidates(cfg)
    by_arm = pd.Series([m.arm for m in meta.values()]).value_counts().to_dict()
    assert by_arm == {"C": 24, "C+RNA": 222, "strict_clinical": 24, "C+DNA": 100, "C+RNA+DNA": 120,
                      "RNA": 116, "RNA+DNA": 120, "DNA": 100, "fused": 20, "C_diag": 24}   # 3a-3c + fused + at diagnosis
    assert len({c.name for c in cands}) == len(cands)
    lr = [c for c in cands if meta[c.name].member == "logreg_en"]
    xgb = [c for c in cands if meta[c.name].member == "xgboost"]
    assert max(c.complexity for c in lr) < min(c.complexity for c in xgb)   # family first


def test_real_logreg_on_clinical_runs(cfg, ds):
    from metabric.harness import splits_from_table
    from metabric.harness.floors import design_matrix
    from metabric.splits import load_cv_splits, training_ids
    tr = training_ids(cfg, ds)
    H = Harness(cfg, design_matrix(ds.subset(tr)), ds.y.loc[tr], splits_from_table(load_cv_splits(cfg, tr)))
    cands, meta = build_candidates(cfg, members=["logreg_en"], arms=["C"])
    r = H.evaluate(cands[:2], select=True, n_jobs=2)
    assert r.selection.winner.isin([c.name for c in cands[:2]]).all()
    assert 0.6 < r.selection.outer_auroc.mean() < 0.85


def test_run_farm_end_to_end_small(tmp_project):
    """The whole farm stage on a tiny grid, including a member that exists on only one arm."""
    from metabric.core.artifacts import Run
    from metabric.data import load_dataset
    from metabric.farm.stage import run_farm
    from metabric.splits import make_cv_splits, make_holdout, training_ids
    ds = load_dataset(tmp_project)
    c, _, _ = make_holdout(tmp_project, ds)
    c, _, _ = make_cv_splits(c, ds.subset(training_ids(c, ds)))
    f = c.raw["farm"]
    f.update(n_jobs=2, n_permutations=2, neural_rna_representations=["pca10"], permutation_arm="C",
             grids={"logreg_en": {"C": [0.1], "l1_ratio": [0.0]}, "logreg_cause": {"C": [0.1], "l1_ratio": [0.0]},
                    "mlp": {"hidden": [[8]], "dropout": [0.3], "weight_decay": [0.001]},
                    "multimodal": {"block_dropout": [0.1], "dropout": [0.3], "weight_decay": [0.001]}},
             neural={**f["neural"], "max_epochs": 3, "n_seeds": 1}, dae={**f["dae"], "max_epochs": 3, "n_seeds": 1})
    f["arms"]["C+RNA"]["rna_reps"] = ["pca10", "hallmark", "modules", "dae8"]
    f["arms"]["C+DNA"]["dna_reps"] = ["known5", "bagemb"]
    c.raw["harness"].update(n_boot=10, chance_band=0.5)        # 2 shuffles: only the plumbing is tested here
    run = Run(c, "test farm", run_id="farm_test")
    facts = run_farm(c, run, members=["logreg_en", "logreg_cause", "mlp", "multimodal"], arms=["C", "C+RNA", "C+DNA"])
    run.finalize()
    assert "C|multimodal" not in facts and "C+RNA|multimodal" in facts and "C+RNA|best" in facts
    assert "C+DNA|mlp" not in facts and "C+DNA|best" in facts                 # neural deferred on DNA arms
    assert facts["representations_tested"] == 5 + 2      # C+RNA: 4 reps + "all" (multimodal); C+DNA: 2 reps
    assert "C|logreg_cause" in facts and "C+DNA|logreg_cause" not in facts    # cause members only on cause_arms
    pairs = pd.read_csv(run.dir / "farm" / "tables" / "paired_comparisons.csv")
    assert {"C|logreg_cause", "C+RNA|logreg_cause"} <= set(pairs[pairs.comparison.str.startswith("cause-specific")].a)
    assert facts["permutation"]["n_permutations"] == 2
    assert (run.dir / "farm" / "tables" / "farm_summary.md").is_file()


def test_farm_run_without_the_permutation_arm_skips_the_permutation_test(tmp_project):
    """Farm runs split by arm: only the run holding the permutation arm runs the test (F-072)."""
    from metabric.core.artifacts import Run
    from metabric.data import load_dataset
    from metabric.farm.stage import run_farm
    from metabric.splits import make_cv_splits, make_holdout, training_ids
    ds = load_dataset(tmp_project)
    c, _, _ = make_holdout(tmp_project, ds)
    c, _, _ = make_cv_splits(c, ds.subset(training_ids(c, ds)))
    c.raw["farm"].update(n_jobs=2, n_permutations=2, permutation_arm="C", grids={"logreg_en": {"C": [0.1], "l1_ratio": [0.0]}})
    run = Run(c, "test farm strict", run_id="farm_strict")
    facts = run_farm(c, run, members=["logreg_en"], arms=["strict_clinical"])
    run.finalize()
    assert facts["permutation"] is None and "strict_clinical|best" in facts
    assert not (run.dir / "farm" / "tables" / "permutation_test.csv").exists()
    assert "not in this run" in (run.dir / "farm" / "tables" / "farm_summary.md").read_text(encoding="utf-8")
