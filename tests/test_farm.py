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
    assert by_arm == {"C+RNA": 120, "C": 20, "strict_clinical": 20}
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
