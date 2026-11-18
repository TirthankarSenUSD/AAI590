"""Preprocessing for the farm. Everything here is an sklearn transformer, so it is fitted inside
each training fold by the harness: nothing sees the rows it is scored on."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


class CategoricalAsString(BaseEstimator, TransformerMixin):
    """Stateless: categorical columns as strings, missing as its own level 'missing' (keeps the
    missingness signal for linear models, A12)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.apply(lambda s: s.astype(object).where(s.notna(), "missing").astype(str)).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


class FrequencySelector(BaseEstimator, TransformerMixin):
    """Top-k genes by selection frequency: L1 logistic regression on `n_subsamples` stratified
    half-samples of the training rows; genes ranked by how often their coefficient is non-zero
    (ties: mean |coefficient|). Fitted inside the fold only (A7)."""

    def __init__(self, k=30, n_subsamples=10, fraction=0.5, C=0.05, random_state=0):
        self.k, self.n_subsamples, self.fraction, self.C, self.random_state = k, n_subsamples, fraction, C, random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
        rng = np.random.default_rng(self.random_state)
        freq = np.zeros(X.shape[1])
        mag = np.zeros(X.shape[1])
        for _ in range(self.n_subsamples):
            idx = np.concatenate([rng.choice(np.flatnonzero(y == v), int(round(self.fraction * (y == v).sum())), replace=False)
                                  for v in (0, 1)])
            m = LogisticRegression(l1_ratio=1.0, C=self.C, solver="liblinear", max_iter=1000).fit(Xs[idx], y[idx])
            coef = np.abs(m.coef_[0])
            freq += coef > 0
            mag += coef
        order = np.lexsort((-mag, -freq))            # primary: frequency, secondary: magnitude
        self.selected_ = np.sort(order[: self.k])
        self.frequency_ = freq / self.n_subsamples
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.selected_]


def _scale(factor: float):
    return FunctionTransformer(lambda Z: Z * factor, feature_names_out="one-to-one")


def column_groups(cfg, arm: str) -> dict[str, list[str]]:
    """Clinical columns for an arm (after the one-per-concept drops) by type, plus RNA columns."""
    fc, types = cfg["farm"], cfg["clinical_types"]
    a = fc["arms"][arm]
    drop = set(fc["drop_clinical"])
    for g in a.get("drop", []):
        drop |= set(cfg["groups"][g])
    keep = [c for c in cfg.blocks["clinical"] if c not in drop] if "clinical" in a["blocks"] else []
    groups = {"numeric": [c for c in types["numeric"] + types["ordinal"] if c in keep],
              "binary": [c for c in types["binary"] if c in keep],
              "categorical": [c for c in types["categorical"] if c in keep],
              "rna": list(cfg.blocks["rna"]) if "rna" in a["blocks"] else [],
              "dna": list(cfg.blocks["dna"]) if "dna" in a["blocks"] else []}
    assert set(groups["numeric"] + groups["binary"] + groups["categorical"]) == set(keep)
    return groups


_SET_CACHE: dict = {}


def gene_set_positions(cfg, source: str, block: str, min_genes: int) -> tuple[list[list[int]], list[str]]:
    """Positions (within the RNA or DNA block) of the matched genes of every `source` gene set
    with at least `min_genes` matched genes in that block, from the frozen crosswalk tables."""
    key = (str(cfg.root), source, block, min_genes)
    if key not in _SET_CACHE:
        from ..crosswalk import GeneSetIndex
        idx = GeneSetIndex(cfg)
        pos = {c: i for i, c in enumerate(cfg.blocks[block])}
        col = "n_rna_matched" if block == "rna" else "n_dna_matched"
        s = idx.summary[(idx.summary.source == source) & (idx.summary[col] >= min_genes)]
        m = idx.membership[(idx.membership.source == source) & (idx.membership.block == block)]
        groups = m.groupby("set").column.apply(list)
        sets = [[pos[c] for c in groups[name]] for name in s.set]
        _SET_CACHE[key] = (sets, list(s.set))
    return _SET_CACHE[key]


def rna_transformer(rep: dict, kind: str, cfg, seed: int, rna_scale: float | None):
    from .representations import DenoisingAE, GeneSetScorer, ModuleScorer
    fc = cfg["farm"]
    sel = fc["topk_selection"]
    steps = []
    if "sets" in rep:
        gs = fc["gene_set_scores"]
        sets, names = gene_set_positions(cfg, rep["sets"], "rna", gs[f"{rep['sets']}_min_rna"])
        steps.append(GeneSetScorer(sets, names))
        if kind == "linear":
            steps.append(StandardScaler())
    elif "modules" in rep:
        steps.append(ModuleScorer(fc["modules"]["distance_cut"], fc["modules"]["min_size"]))
        if kind == "linear":
            steps.append(StandardScaler())
    elif "dae" in rep:
        steps.append(DenoisingAE(latent=rep["dae"], random_state=seed, **fc["dae"]))
        if kind == "linear":
            steps.append(StandardScaler())
    elif "topk" in rep:
        steps.append(FrequencySelector(rep["topk"], sel["n_subsamples"], sel["fraction"], sel["C"], seed))
        if kind == "linear":
            steps.append(StandardScaler())
    elif "pca" in rep:
        steps += [StandardScaler(), PCA(rep["pca"], random_state=seed)]
        if kind == "linear":
            steps.append(StandardScaler())
    elif kind == "linear":
        steps.append(StandardScaler())
        if rna_scale:
            steps.append(_scale(rna_scale))         # smaller scale = stronger effective penalty
    return make_pipeline(*steps) if steps else "passthrough"


def dna_transformer(rep: dict, kind: str, cfg):
    from .representations import ColumnSubset, CooccurrenceEmbedding, MutationCount, PathwayFlags
    dna = list(cfg.blocks["dna"])
    if "known" in rep:
        return ColumnSubset([dna.index(c) for c in rep["known"]])
    if "count" in rep:
        return make_pipeline(MutationCount(), StandardScaler()) if kind == "linear" else MutationCount()
    if "pathway" in rep:
        sets, names = gene_set_positions(cfg, rep["pathway"], "dna", rep["min_dna"])
        return PathwayFlags(sets, names)
    if "embed" in rep:
        e = CooccurrenceEmbedding(rep["embed"], cfg.seed)
        return make_pipeline(e, StandardScaler()) if kind == "linear" else e
    return "passthrough"                                # all 173 flags


def preprocessor(groups: dict, kind: str, rep: dict | None, cfg, seed: int, rna_scale: float | None = None,
                 dna_rep: dict | None = None):
    """kind 'linear': median fill + missing flags + scaling, one-hot categories (A1, A12).
    kind 'tree': numbers passed with gaps (native handling), categories as ordinal codes.
    kind 'embed': as 'linear' for numbers and RNA, but categories as integer codes placed last
    (for learned embeddings; 0 = a level never seen in the fitting rows)."""
    rare = cfg["farm"]["rare_level_min_count"]
    parts = []
    if kind == "linear":
        if groups["numeric"]:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                          groups["numeric"]))
        if groups["categorical"]:
            parts.append(("cat", make_pipeline(CategoricalAsString(),
                                               OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=rare,
                                                             sparse_output=False)), groups["categorical"]))
    elif kind == "embed":
        # dense part first (as for linear members), integer category codes last for the embeddings
        if groups["numeric"]:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                          groups["numeric"]))
        if groups["binary"]:
            parts.append(("bin", "passthrough", groups["binary"]))
        if groups["rna"]:
            parts.append(("rna", rna_transformer(rep or {}, "linear", cfg, seed, rna_scale), groups["rna"]))
        if groups.get("dna"):
            parts.append(("dna", dna_transformer(dna_rep or {}, "linear", cfg), groups["dna"]))
        if groups["categorical"]:
            parts.append(("cat", make_pipeline(CategoricalAsString(),
                                               OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                              min_frequency=rare),
                                               FunctionTransformer(lambda Z: Z + 1, feature_names_out="one-to-one")),
                          groups["categorical"]))              # 0 = unseen level
        return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
    else:
        if groups["numeric"]:
            parts.append(("num", "passthrough", groups["numeric"]))
        if groups["categorical"]:
            parts.append(("cat", make_pipeline(CategoricalAsString(),
                                               OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                              min_frequency=rare)), groups["categorical"]))
    if groups["binary"]:
        parts.append(("bin", "passthrough", groups["binary"]))
    if groups.get("dna"):
        parts.append(("dna", dna_transformer(dna_rep or {}, kind, cfg), groups["dna"]))
    if groups["rna"]:                                      # RNA last (the multimodal member relies on this)
        parts.append(("rna", rna_transformer(rep or {}, kind, cfg, seed, rna_scale), groups["rna"]))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
