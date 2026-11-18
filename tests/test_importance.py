"""Step 4: explanation helpers are correct on known cases, and the stage reproduces the farm's fits."""
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin

from metabric.farm import build_candidates
from metabric.importance import explain as E


class _Fixed(BaseEstimator, ClassifierMixin):
    """A known risk function of raw columns a and b (for model-agnostic checks)."""

    def __init__(self, kind="additive"):
        self.kind = kind

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        a, b = X["a"].to_numpy(float), X["b"].to_numpy(float)
        z = a * b * 2 if self.kind == "interaction" else a + b
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _xy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * X.a))).astype(int)
    return X, y


def test_permutation_importance_finds_signal_and_ignores_noise():
    from sklearn.linear_model import LogisticRegression
    X, y = _xy()
    m = LogisticRegression().fit(X, y)
    imp = E.permutation_importance(m, X, y, {"a": ["a"], "b": ["b"], "c": ["c"]}, 5, [1]).set_index("group")
    assert imp.loc["a", "importance"] > 0.15
    assert abs(imp.loc["b", "importance"]) < 0.02 and abs(imp.loc["c", "importance"]) < 0.02


def test_h_statistic_separates_interaction_from_additive():
    X, _ = _xy(n=40)
    h_int = E.h_statistic(_Fixed("interaction"), X, ["a", "b"]).h2.iloc[0]
    h_add = E.h_statistic(_Fixed("additive"), X, ["a", "b"]).h2.iloc[0]
    assert h_int > 0.5 and h_add < 1e-9                  # additive in log-odds -> exactly 0
    assert E.h_statistic(_Fixed("additive"), X, ["a", "a", "b"]).shape[0] == 1   # duplicates never paired


def test_gene_groups_join_correlated_genes_only():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(300, 1))
    X = pd.DataFrame(np.hstack([f + 0.2 * rng.normal(size=(300, 3)), rng.normal(size=(300, 4))]),
                     columns=[f"g{i}" for i in range(7)])
    groups = E.gene_groups(X, 0.5, 2)
    assert ["g0", "g1", "g2"] in groups and all([g] in groups for g in ["g3", "g4", "g5", "g6"])


@pytest.mark.parametrize("arm,member,rep", [("C", "logreg_en", "none"), ("C", "xgboost", "none"),
                                            ("C+RNA", "logreg_en", "pca10"), ("C+RNA", "random_forest", "hallmark"),
                                            ("C+DNA", "logreg_en", "pathway"), ("C", "mlp_embed", "none")])
def test_feature_map_matches_model_inputs_and_shap_adds_up(cfg, ds, arm, member, rep):
    cands, meta = build_candidates(cfg, members=[member], arms=[arm])
    c = next(c for c in cands if meta[c.name].rep == rep)
    from metabric.harness.floors import design_matrix
    X = design_matrix(ds)[c.columns]
    pipe = c.make().fit(X.iloc[:700], ds.y.iloc[:700].to_numpy())
    fm = E.feature_map(pipe)
    Z = pipe[0].transform(X.iloc[700:900])
    assert len(fm) == Z.shape[1]
    assert set(fm[fm.block == "clinical"].source) <= set(X.columns)
    if member == "logreg_en":
        phi = (Z - pipe[0].transform(X.iloc[:700]).mean(0)) * pipe[-1].coef_[0]
        p = pipe.predict_proba(X.iloc[700:900])[:, 1]
        base = pipe[-1].intercept_[0] + pipe[0].transform(X.iloc[:700]).mean(0) @ pipe[-1].coef_[0]
        np.testing.assert_allclose(phi.sum(1) + base, np.log(p / (1 - p)), atol=1e-6)    # exact linear SHAP
    if member in ("logreg_en", "xgboost", "random_forest"):
        s = E.shap_by_source(pipe, member, X.iloc[:700], X.iloc[700:900], fm)
        assert (s >= 0).all() and set(s.index) == set(fm.source)


def test_importance_stage_end_to_end_small(tmp_project):
    """Tiny farm run, then the importance stage on it: every refit must reproduce the farm's AUROC."""
    from metabric.core.artifacts import Run
    from metabric.core.config import Config
    from metabric.data import load_dataset
    from metabric.farm.stage import run_farm
    from metabric.importance.stage import run_importance
    from metabric.splits import make_cv_splits, make_holdout, training_ids
    ds = load_dataset(tmp_project)
    c, _, _ = make_holdout(tmp_project, ds)
    c, _, _ = make_cv_splits(c, ds.subset(training_ids(c, ds)))
    f = c.raw["farm"]
    f.update(n_jobs=2, n_permutations=1, permutation_arm="C", members=["logreg_en", "xgboost"],
             grids={"logreg_en": {"C": [0.1], "l1_ratio": [0.0]},
                    "xgboost": {"n_estimators": [20], "learning_rate": [0.1], "max_depth": [2], "subsample": [0.8],
                                "colsample_bytree": [0.8], "min_child_weight": [5]}})
    f["arms"] = {"C": f["arms"]["C"], "C+RNA": {**f["arms"]["C+RNA"], "rna_reps": ["pca10"]},
                 "C+RNA+DNA": {**f["arms"]["C+RNA+DNA"], "rna_reps": ["pca10"], "dna_reps": ["known5"]}}
    c.raw["harness"].update(n_boot=10, chance_band=0.5)
    c.raw["importance"].update(arms_full=["C", "C+RNA"], arms_modality_only=["C+RNA+DNA"], n_repeats=1, top_n=3,
                               pdp_grid=3, ice_rows=5, h_rows=8, learning_curve_fractions=[0.5, 1.0])
    c.file.write_text(yaml.safe_dump(c.raw), encoding="utf-8")       # the run folders copy the file
    c = Config(root=c.root, raw=yaml.safe_load(c.file.read_text(encoding="utf-8")))
    r1 = Run(c, "farm", run_id="farm_small")
    run_farm(c, r1)
    r1.finalize()
    r2 = Run(c, "importance", run_id="imp_small")
    facts = run_importance(c, r2)
    r2.finalize()
    chk = pd.read_csv(r2.dir / "importance" / "tables" / "refit_check.csv")
    assert chk.abs_diff.max() < 1e-6 and len(chk) > 0
    assert facts["farm_run"] == "farm_small" and len(facts["top"]["C"]) == 3
    assert (r2.dir / "importance" / "tables" / "importance_summary.md").is_file()
    mod = pd.read_csv(r2.dir / "importance" / "tables" / "modality_importance.csv")
    assert {"C+RNA+DNA", "C"} <= set(mod.arm) and "all DNA" in set(mod.group)
    top = pd.read_csv(r2.dir / "importance" / "tables" / "top_features.csv")
    assert not top.duplicated(["arm", "feature"]).any()                        # one entry per feature
    h = pd.read_csv(r2.dir / "importance" / "tables" / "interactions.csv")
    assert (h.feature_a != h.feature_b).all()
    r3 = Run(c, "importance --reuse", run_id="imp_reuse")                     # reuse pass 1: same summaries
    facts3 = run_importance(c, r3, reuse="imp_small")
    r3.finalize()
    a1 = pd.read_csv(r2.dir / "importance" / "tables" / "permutation_summary.csv")
    a3 = pd.read_csv(r3.dir / "importance" / "tables" / "permutation_summary.csv")
    pd.testing.assert_frame_equal(a1, a3, check_exact=False, rtol=1e-9, atol=1e-12)
    assert facts3["top"] == facts["top"]
