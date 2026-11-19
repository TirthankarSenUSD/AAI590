"""Neural farm members (step 3b): they learn known signal, are reproducible, handle unseen
categories, and plug into the harness and the farm preprocessing like any other member."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from metabric.farm import build_candidates
from metabric.farm.neural import TorchEmbedMLP, TorchMLP, TorchMultimodal
from metabric.farm.preprocess import column_groups, preprocessor

FAST = dict(max_epochs=60, patience=10, n_seeds=2)


def _data(n=800, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * X[:, 0] - 1.5 * X[:, 1])))).astype(int)
    return X, y


def test_mlp_learns_signal_and_is_reproducible():
    X, y = _data()
    a = TorchMLP(hidden=(16,), random_state=3, **FAST).fit(X[:600], y[:600]).predict_proba(X[600:])[:, 1]
    b = TorchMLP(hidden=(16,), random_state=3, **FAST).fit(X[:600], y[:600]).predict_proba(X[600:])[:, 1]
    assert roc_auc_score(y[600:], a) > 0.8
    np.testing.assert_array_equal(a, b)                        # same seed -> identical predictions
    assert ((a > 0) & (a < 1)).all()


def test_embed_mlp_uses_categories_and_tolerates_unseen_codes():
    rng = np.random.default_rng(1)
    n = 900
    cat = rng.integers(1, 5, n)                                # codes 1..4 (0 is reserved for unseen)
    y = (rng.random(n) < np.where(cat == 3, 0.85, 0.25)).astype(int)
    X = np.column_stack([rng.normal(size=(n, 3)), cat]).astype(np.float32)
    m = TorchEmbedMLP(n_cat=1, hidden=(16,), random_state=0, **FAST).fit(X[:700], y[:700])
    oracle = roc_auc_score(y[700:], (cat[700:] == 3).astype(float))      # best achievable (~0.73)
    assert roc_auc_score(y[700:], m.predict_proba(X[700:])[:, 1]) > oracle - 0.06
    X_new = X[700:].copy()
    X_new[:, -1] = 99                                          # a code never seen in fit
    assert np.isfinite(m.predict_proba(X_new)).all()


def test_multimodal_uses_both_blocks_and_block_dropout_is_seeded():
    rng = np.random.default_rng(2)
    n, n_rna = 900, 20
    clin = rng.normal(size=(n, 4))
    rna = rng.normal(size=(n, n_rna))
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * clin[:, 0] + 1.5 * rna[:, 0])))).astype(int)
    X = np.column_stack([clin, rna]).astype(np.float32)
    kw = dict(n_rna=n_rna, block_dropout=0.3, random_state=4, **FAST)
    p1 = TorchMultimodal(**kw).fit(X[:700], y[:700]).predict_proba(X[700:])[:, 1]
    p2 = TorchMultimodal(**kw).fit(X[:700], y[:700]).predict_proba(X[700:])[:, 1]
    np.testing.assert_array_equal(p1, p2)
    assert roc_auc_score(y[700:], p1) > 0.75


def test_embed_preprocessing_puts_codes_last_and_nonnegative(cfg, ds):
    g = column_groups(cfg, "C")
    cols = g["numeric"] + g["binary"] + g["categorical"]
    pre = preprocessor(g, "embed", None, cfg, 0).fit(ds.frame[cols].iloc[:800])
    Z = pre.transform(ds.frame[cols].iloc[800:])
    codes = Z[:, -len(g["categorical"]):]
    assert (codes >= 0).all() and np.allclose(codes, np.round(codes))
    assert not np.isnan(Z).any()


def test_neural_candidates_built_per_arm(cfg):
    cands, meta = build_candidates(cfg, members=["mlp", "mlp_embed", "multimodal"])
    m = pd.DataFrame([(v.arm, v.member, v.rep) for v in meta.values()], columns=["arm", "member", "rep"])
    assert set(m[m.member == "multimodal"].arm) == {"C+RNA"}               # needs two blocks
    assert set(m[(m.arm == "C+RNA") & (m.member == "mlp")].rep) == set(cfg["farm"]["neural_rna_representations"])
    assert set(m[m.arm == "C"].rep) == {"none"}


def test_fitted_networks_survive_pickling():
    """Fitted networks are saved as weights and rebuilt on load, giving identical predictions."""
    import pickle

    from metabric.farm.neural import TorchEmbedMLP, TorchMLP, TorchMultimodal
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    X[:, -1] = rng.integers(1, 4, 120)
    y = (X[:, 0] + rng.normal(size=120) > 0).astype(int)
    for m in (TorchMLP(hidden=(8,), max_epochs=5, n_seeds=1), TorchEmbedMLP(n_cat=1, hidden=(8,), max_epochs=5, n_seeds=1),
              TorchMultimodal(n_rna=3, clin_dim=4, rna_dim=4, max_epochs=5, n_seeds=1)):
        m.fit(X, y)
        m2 = pickle.loads(pickle.dumps(m))
        assert np.allclose(m.predict_proba(X), m2.predict_proba(X))
