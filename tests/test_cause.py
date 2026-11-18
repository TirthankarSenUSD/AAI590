"""Cause-specific members: the cause of death is a training-only label; everything is scored on death."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metabric.farm.cause import CauseSpecific, cause_labels
from metabric.harness import Candidate, Harness

from test_harness import HCFG, make_splits


def _competing(n=900, seed=0):
    """Cancer death driven by x1, other-cause death by x2 (age-like); x3 is noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x1", "x2", "x3"], index=[f"p{i}" for i in range(n)])
    z1, z2 = 1.5 * X.x1 - 0.5, 1.5 * X.x2 - 0.5
    den = 1 + np.exp(z1) + np.exp(z2)
    u = rng.random(n)
    y3 = np.where(u < np.exp(z1) / den, 1, np.where(u < (np.exp(z1) + np.exp(z2)) / den, 2, 0))
    return X, pd.Series(y3, index=X.index)


class _Spy(LogisticRegression):
    seen = []

    def fit(self, X, y):
        _Spy.seen.append(int(np.max(y)))
        return super().fit(X, y)


def test_wrapper_returns_death_as_sum_of_causes():
    X, y3 = _competing()
    m = CauseSpecific(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))).fit(X, y3)
    pc, pd_ = m.predict_cause_proba(X), m.predict_proba(X)
    np.testing.assert_allclose(pd_[:, 1], pc[:, 1] + pc[:, 2])
    np.testing.assert_allclose(pd_.sum(1), 1)
    with pytest.raises(ValueError):
        CauseSpecific(make_pipeline(LogisticRegression())).fit(X, (y3 > 0).astype(int))   # needs 3 classes


def test_harness_fits_each_member_on_its_own_label_and_scores_on_death():
    X, y3 = _competing()
    y = (y3 > 0).astype(int)
    H = Harness(HCFG, X, y, make_splits(y), y_cause=y3)
    _Spy.seen = []
    binary = Candidate("binary", lambda: _Spy(max_iter=500), ["x1", "x2", "x3"])
    cause = Candidate("cause", lambda: CauseSpecific(_Spy(max_iter=500)), ["x1", "x2", "x3"], target="cause")
    r = H.evaluate([binary, cause])
    assert set(_Spy.seen[: len(_Spy.seen) // 2]) == {1} and set(_Spy.seen[len(_Spy.seen) // 2:]) == {2}
    assert set(r.preds.y.unique()) == {0, 1}                                  # scored on death
    assert r.folds.groupby("candidate").auroc.mean().min() > 0.7
    with pytest.raises(ValueError):                                            # no cause label available
        Harness(HCFG, X, y, make_splits(y)).evaluate([cause])
    bad = y3.copy()
    bad.iloc[0] = 0 if y.iloc[0] == 1 else 1
    with pytest.raises(ValueError):                                            # cause must agree with death
        Harness(HCFG, X, y, make_splits(y), y_cause=bad)


def test_permutation_test_shuffles_the_cause_label_and_is_at_chance():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"), index=[f"p{i}" for i in range(400)])
    y3 = pd.Series(rng.integers(0, 3, 400), index=X.index)
    y = (y3 > 0).astype(int)
    H = Harness(HCFG, X, y, make_splits(y), y_cause=y3)
    cause = Candidate("cause", lambda: CauseSpecific(LogisticRegression(max_iter=500)), list("abcde"), target="cause")
    seen = []

    def proc(labels):
        seen.append(int(labels.max()))
        return H.evaluate([cause], y=labels, light=True).folds.auroc.mean()

    _, s = H.permutation_test(proc, 8, np.random.default_rng(0))
    assert set(seen) == {2} and s["at_chance"]                                # the 3-class label was permuted


def test_cause_coefficients_are_two_contrasts():
    from metabric.importance.explain import lr_coefficients
    X, y3 = _competing()
    m = CauseSpecific(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))).fit(X, y3)
    fm = pd.DataFrame({"feature": list(X.columns), "source": list(X.columns), "block": "clinical"})
    c = lr_coefficients(m, fm).set_index(["contrast", "feature"]).coef
    assert c[("cancer death vs alive", "x1")] > 1 and abs(c[("cancer death vs alive", "x2")]) < 0.4
    assert c[("other-cause death vs alive", "x2")] > 1 and abs(c[("other-cause death vs alive", "x1")]) < 0.4


def test_real_cause_labels(cfg, ds):
    from metabric.splits import training_ids
    tr = training_ids(cfg, ds)
    y3 = cause_labels(ds.frame.loc[tr])                                       # training rows: no missing cause
    assert ((y3 > 0).astype(int) == ds.y.loc[tr]).all()
    assert set(y3.unique()) == {0, 1, 2}
    with pytest.raises(ValueError):
        cause_labels(ds.frame)                                                # all rows include the unknown cause
