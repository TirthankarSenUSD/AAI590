import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from metabric import stats as st


def test_auroc_matches_sklearn_and_drops_missing():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    assert np.isclose(st.auroc(s, y), roc_auc_score(y, s))
    s2 = s.copy()
    s2[:10] = np.nan
    assert np.isclose(st.auroc(s2, y), roc_auc_score(y[10:], s[10:]))


def test_auroc_columns_and_null_at_chance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 20))
    y = rng.integers(0, 2, 400)
    X[:, 0] += 2 * y                               # one informative column
    a = st.auroc_columns(X, y)
    assert np.allclose(a, [roc_auc_score(y, X[:, j]) for j in range(20)])
    null = st.auroc_permutation_null(X, y, 200, np.random.default_rng(2))
    assert abs(null.mean() - 0.5) < 0.01
    p = st.permutation_pvalues(a, null)
    assert p[0] == 1 / 201 and np.median(p[1:]) > 0.2


def test_bh_fdr_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(st.bh_fdr(p), [0.02, 0.04, 0.04, 0.02])


def test_wilson_interval():
    lo, hi = st.wilson(50, 100)
    assert np.isclose(lo, 0.4038, atol=1e-3) and np.isclose(hi, 0.5962, atol=1e-3)


def test_cramers_v_identity_and_independence():
    a = pd.Series(list("abc") * 100)
    assert np.isclose(st.cramers_v(a, a), 1.0)
    rng = np.random.default_rng(3)
    b = pd.Series(rng.choice(list("xy"), 300))
    assert st.cramers_v(a, b, bias_corrected=True) < 0.1


def test_kaplan_meier_no_censoring_is_empirical():
    t = np.array([1, 2, 3, 4])
    km = st.kaplan_meier(t, np.ones(4))
    assert np.allclose(km.survival, [1, 0.75, 0.5, 0.25, 0])


def test_rate_table_missing_level():
    g = pd.Series(["a", "a", None, "b"])
    y = pd.Series([1, 0, 1, 1])
    t = st.rate_table(g, y).set_index("level")
    assert t.loc["a", "death_rate"] == 0.5 and t.loc["missing", "n"] == 1


def test_mwu_consistency():
    rng = np.random.default_rng(4)
    x, z = rng.normal(size=50), rng.normal(0.5, size=60)
    u = mannwhitneyu(z, x).statistic
    y = np.r_[np.zeros(50), np.ones(60)]
    assert np.isclose(st.auroc(np.r_[x, z], y), u / (50 * 60))
