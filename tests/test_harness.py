"""Harness tests on synthetic data, where the right answer is known (CLAUDE.md section 9, step 2)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metabric import metrics as M
from metabric.harness import Candidate, Harness, Split, check_split_hygiene, select_candidate
from metabric.stats import corrected_resampled_ttest

HCFG = {"harness": {"operating_point_sensitivity": 0.8, "recalibration": "platt", "reliability_bins": 10, "n_boot": 50,
                    "n_permutations": 10, "chance_band": 0.03, "select_within_se": 1.0,
                    "learning_curve_fractions": [0.5, 1.0],
                    "overfit_gate": {"max_train_cv_auroc_gap": 0.05, "calibration_slope": [0.8, 1.2]},
                    "regression_floor_sd": 2}}


def make_splits(y: pd.Series, K=5, R=2, I=3, seed=0) -> list[Split]:
    out, ids = [], y.index.to_numpy()
    for r in range(R):
        for k, (tr, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=seed + r).split(ids, y)):
            tri = ids[tr]
            inner = [(pd.Index(tri[a]), pd.Index(tri[b]))
                     for a, b in StratifiedKFold(I, shuffle=True, random_state=seed + 100 + k).split(tri, y.loc[tri])]
            out.append(Split(f"r{r}o{k}", r, k, pd.Index(tri), pd.Index(ids[te]), inner))
    return out


def synth(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["strong", "weak", "noise"], index=[f"p{i}" for i in range(n)])
    lin = 1.5 * X.strong + 0.5 * X.weak - 0.2
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int), index=X.index)
    return X, y, lin


def lr(cols, name, complexity=1.0, eligible=True):
    return Candidate(name, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=1000)), cols,
                     complexity, eligible)


def test_known_risk_ordering_recovered():
    X, y, lin = synth()
    H = Harness(HCFG, X, y, make_splits(y))
    res = H.evaluate([lr(["strong", "weak", "noise"], "full", 3), lr(["strong"], "strong"),
                      lr(["weak"], "weak"), lr(["noise"], "noise")])
    a = res.folds.groupby("candidate").auroc.mean()
    assert a["full"] > a["strong"] > a["weak"] > a["noise"]
    assert abs(a["noise"] - 0.5) < 0.04
    assert abs(a["full"] - roc_auc_score(y, lin)) < 0.02          # recovers the true model's discrimination
    full = res.folds[res.folds.candidate == "full"]
    assert abs(full.cal_slope.mean() - 1) < 0.15 and abs(full.cal_intercept.mean()) < 0.15
    assert full.brier_skill.mean() > 0.1
    assert (full.op_sensitivity.mean() > 0.7)                      # operating point fixed on inner folds carries over


def test_selection_rule_signal_wins_and_simplest_within_se():
    X, y, _ = synth()
    H = Harness(HCFG, X, y, make_splits(y))
    res = H.evaluate([lr(["noise"], "noise", 1), lr(["strong", "weak"], "signal", 2),
                      lr(["strong", "weak"], "signal_complex_copy", 5),
                      lr(["strong", "weak", "noise"], "floor_like", 0, eligible=False)], select=True)
    assert set(res.selection.winner) == {"signal"}                 # beats noise; identical copy loses on complexity
    assert "optimism" in res.selection and res.selection.proc_auroc.mean() > 0.7
    st = pd.DataFrame([{"candidate": "a", "inner_auroc": 0.70, "inner_auroc_se": 0.02, "complexity": 3, "eligible": True,
                        "inner_log_loss": 0.6, "inner_brier": 0.2},
                       {"candidate": "b", "inner_auroc": 0.69, "inner_auroc_se": 0.02, "complexity": 1, "eligible": True,
                        "inner_log_loss": 0.6, "inner_brier": 0.2},
                       {"candidate": "c", "inner_auroc": 0.60, "inner_auroc_se": 0.02, "complexity": 0, "eligible": True,
                        "inner_log_loss": 0.6, "inner_brier": 0.2},
                       {"candidate": "floor", "inner_auroc": 0.99, "inner_auroc_se": 0.01, "complexity": 0, "eligible": False,
                        "inner_log_loss": 0.1, "inner_brier": 0.1}])
    assert select_candidate(st, 1.0) == "b"                        # within 1 SE of the best and simpler; floor never wins


def test_shuffled_labels_at_chance_and_leak_is_caught():
    rng = np.random.default_rng(5)
    n, p = 300, 500
    X = pd.DataFrame(rng.normal(size=(n, p)), index=[f"p{i}" for i in range(n)], columns=[f"g{j}" for j in range(p)])
    y = pd.Series(rng.integers(0, 2, n), index=X.index)
    H = Harness(HCFG, X, y, make_splits(y))
    proper = Candidate("proper", lambda: make_pipeline(SelectKBest(f_classif, k=10), LogisticRegression(max_iter=500)),
                       list(X.columns))
    proc_ok = lambda yy: H.evaluate([proper], y=yy, light=True).folds.auroc.mean()  # noqa: E731

    def proc_leaky(yy):   # deliberately leaky: selects genes using ALL rows (test folds included)
        F, _ = f_classif(X, yy)
        top = list(X.columns[np.argsort(F)[-10:]])
        c = Candidate("leaky", lambda: LogisticRegression(max_iter=500), top)
        return H.evaluate([c], y=yy, light=True).folds.auroc.mean()

    _, ok = H.permutation_test(proc_ok, 10, np.random.default_rng(0))
    _, bad = H.permutation_test(proc_leaky, 10, np.random.default_rng(0))
    assert ok["at_chance"], ok
    assert not bad["at_chance"] and bad["perm_mean"] > 0.6, bad


def test_fits_never_see_test_rows():
    X, y, _ = synth(n=400)
    splits = make_splits(y)
    seen = []

    class Spy(LogisticRegression):
        def fit(self, Xf, yf):
            seen.append(set(Xf.index))
            return super().fit(Xf, yf)

    H = Harness(HCFG, X, y, splits)
    H.evaluate([Candidate("spy", lambda: Spy(max_iter=500), ["strong", "weak"])])
    assert len(seen) == len(splits) * 4                           # 3 inner fits + 1 outer fit per split
    for rows in seen:
        assert any(rows <= set(s.train) for s in splits)           # every fit uses outer-training rows only


def test_threshold_and_recalibration_use_inner_rows_only():
    X, y, _ = synth(n=600)
    splits = make_splits(y)
    H = Harness(HCFG, X, y, splits)
    c = lr(["strong", "weak"], "m")
    res = H.evaluate([c])
    s = splits[0]
    oof = pd.Series(np.nan, index=s.train)
    for tr, va in s.inner:          # recompute the inner out-of-fold predictions by hand
        m = c.make().fit(X.loc[tr, c.columns], y.loc[tr])
        oof.loc[va] = m.predict_proba(X.loc[va, c.columns])[:, 1]
    t = M.threshold_for_sensitivity(y.loc[s.train], oof.to_numpy(), 0.8)
    assert res.folds.iloc[0].op_threshold == pytest.approx(t)


def test_split_hygiene_detects_overlap():
    X, y, _ = synth(n=200)
    splits = make_splits(y)
    bad = splits[0]
    splits[0] = Split(bad.split_id, bad.repeat, bad.outer, bad.train.append(bad.test[:1]), bad.test, bad.inner)
    with pytest.raises(AssertionError):
        check_split_hygiene(splits, X.index)


def test_corrected_ttest_known_values():
    d = np.array([0.01, 0.03, 0.02, 0.04, 0.0])
    r = corrected_resampled_ttest(d, n_train=80, n_test=20)
    se = np.sqrt((1 / 5 + 20 / 80) * d.var(ddof=1))
    assert r["se_corrected"] == pytest.approx(se)
    assert r["t"] == pytest.approx(0.02 / se)
    assert r["p"] == pytest.approx(2 * sps.t.sf(0.02 / se, 4))
    assert corrected_resampled_ttest(np.zeros(10), 80, 20)["p"] == 1.0


def test_calibration_metrics_on_calibrated_probabilities():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    m = M.binary_metrics(y, p, train_prevalence=y.mean())
    assert abs(m["cal_slope"] - 1) < 0.05 and abs(m["cal_intercept"]) < 0.05
    base = M.binary_metrics(y, np.full(len(y), y.mean()), y.mean())
    assert abs(base["brier_skill"]) < 1e-9 and base["auroc"] == 0.5
    t = M.threshold_for_sensitivity(y, p, 0.8)
    assert M.threshold_metrics(y, p, t)["sensitivity"] >= 0.8


def test_real_floors_on_frozen_splits(cfg, ds):
    from metabric.floors import design_matrix, floor_candidates
    from metabric.harness import splits_from_table
    from metabric.splits import load_cv_splits, training_ids
    tr = training_ids(cfg, ds)
    H = Harness(cfg, design_matrix(ds.subset(tr)), ds.y.loc[tr], splits_from_table(load_cv_splits(cfg, tr)))
    a = H.evaluate(floor_candidates(cfg), light=True).folds.groupby("candidate").auroc.mean()
    assert a["majority"] == 0.5
    assert 0.64 < a["age_only"] < 0.73 and 0.55 < a["npi_file"] < 0.66 and 0.55 < a["npi_standard"] < 0.67
