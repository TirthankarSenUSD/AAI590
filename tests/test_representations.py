"""Step 3c representations: correct by construction, unsupervised (outcome never read), and fitted
on the fitting rows only."""
import numpy as np
import pandas as pd
import pytest

from metabric.farm.preprocess import column_groups, gene_set_positions, preprocessor
from metabric.farm.representations import (ColumnSubset, CooccurrenceEmbedding, DenoisingAE, GeneSetScorer,
                                           ModuleScorer, MutationCount, PathwayFlags)


def test_gene_set_scores_are_mean_z_of_members():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=(200, 6))
    s = GeneSetScorer(sets=[[0, 1], [2, 3, 4]]).fit(X[:150])
    Z = (X[150:] - X[:150].mean(0)) / X[:150].std(0)                 # z from the fitting rows only
    np.testing.assert_allclose(s.transform(X[150:]), np.column_stack([Z[:, [0, 1]].mean(1), Z[:, [2, 3, 4]].mean(1)]))


def test_modules_recover_planted_structure_and_align_signs():
    rng = np.random.default_rng(1)
    f1, f2 = rng.normal(size=(500, 1)), rng.normal(size=(500, 1))
    X = np.hstack([f1 + 0.3 * rng.normal(size=(500, 6)), -(f1 + 0.3 * rng.normal(size=(500, 2))),   # anti-correlated
                   f2 + 0.3 * rng.normal(size=(500, 6)), rng.normal(size=(500, 10))])
    m = ModuleScorer(distance_cut=0.5, min_size=5).fit(X)
    mods = [set(idx) for idx in m.modules_]
    assert set(range(0, 8)) in mods and set(range(8, 14)) in mods
    scores = m.transform(X)
    k = mods.index(set(range(0, 8)))
    assert abs(np.corrcoef(scores[:, k], f1[:, 0])[0, 1]) > 0.95             # signs aligned, not cancelled


def test_autoencoder_is_reproducible_and_ignores_the_outcome():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 40)) + rng.normal(size=(300, 1))
    kw = dict(latent=4, hidden=16, max_epochs=20, patience=5, n_seeds=2, random_state=3)
    a = DenoisingAE(**kw).fit(X, rng.integers(0, 2, 300)).transform(X)
    b = DenoisingAE(**kw).fit(X, rng.integers(0, 2, 300)).transform(X)      # different y: no effect
    np.testing.assert_array_equal(a, b)
    assert a.shape == (300, 4)
    from metabric.farm.representations import _AE_CACHE
    _AE_CACHE.clear()
    c = DenoisingAE(**kw).fit(X).transform(X)                                   # fresh fit
    d = DenoisingAE(**kw).fit(X).transform(X)                                   # served from the cache
    assert len(_AE_CACHE) == 1
    np.testing.assert_array_equal(c, d)
    np.testing.assert_array_equal(a, c)


def test_dna_representations():
    X = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=float)
    assert ColumnSubset([0, 3]).fit(X).transform(X).tolist() == [[1, 1], [0, 0], [1, 0]]
    assert MutationCount().fit(X).transform(X).ravel().tolist() == [2, 0, 2]
    assert PathwayFlags([[0, 1], [2]]).fit(X).transform(X).tolist() == [[1, 0], [0, 0], [1, 0]]
    e = CooccurrenceEmbedding(dim=2).fit(X)
    out = e.transform(X)
    assert out.shape == (3, 2) and np.allclose(out[1], 0)                    # no mutations -> zero vector
    np.testing.assert_allclose(out[0], e.gene_vectors_[[0, 3]].mean(0))


def test_real_gene_sets_and_arms(cfg):
    sets, names = gene_set_positions(cfg, "hallmark", "rna", 5)
    assert len(sets) == 40 and all(len(s) >= 5 for s in sets)
    assert len(gene_set_positions(cfg, "reactome", "rna", 10)[0]) == 334
    assert len(gene_set_positions(cfg, "hallmark", "dna", 3)[0]) == 25
    g = column_groups(cfg, "RNA+DNA")
    assert not (g["numeric"] or g["binary"] or g["categorical"]) and len(g["rna"]) == 489 and len(g["dna"]) == 173


@pytest.mark.parametrize("arm,rna,dna", [("C+RNA", "hallmark", None), ("RNA", "modules", None),
                                         ("DNA", None, "pathway"), ("C+DNA", None, "bagemb")])
def test_representation_preprocessing_fits_on_fitting_rows_only(cfg, ds, arm, rna, dna):
    g = column_groups(cfg, arm)
    cols = g["numeric"] + g["binary"] + g["categorical"] + g["rna"] + g["dna"]
    X = pd.concat([ds.frame[g["numeric"] + g["binary"] + g["categorical"] + g["rna"]], ds.mutations], axis=1)[cols]
    rr = cfg["farm"]["rna_representations"][rna] if rna else None
    dr = cfg["farm"]["dna_representations"][dna] if dna else None
    a = preprocessor(g, "linear", rr, cfg, 0, None, dr).fit(X.iloc[:800]).transform(X.iloc[800:])
    X2 = pd.concat([X.iloc[:800], X.iloc[800:].sample(frac=1, random_state=0).set_axis(X.index[800:])])  # scramble scored rows
    b = preprocessor(g, "linear", rr, cfg, 0, None, dr).fit(X2.iloc[:800]).transform(X.iloc[800:])
    np.testing.assert_allclose(a, b)                                         # fit never saw rows 800+
    assert np.isfinite(a).all()
