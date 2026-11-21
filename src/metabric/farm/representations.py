"""Representation transformers for step 3c (RNA and DNA representations). All are fitted inside each
training fold by the harness and none of them reads the outcome: they are unsupervised
summaries of the RNA block or the DNA (mutation flag) block.

RNA:  GeneSetScorer (gene-set signature scores), ModuleScorer (co-expression modules found on
      the fitting rows), DenoisingAE (denoising-autoencoder latents).
DNA:  ColumnSubset (known driver genes), MutationCount, PathwayFlags (any mutation in a gene
      set), CooccurrenceEmbedding (bag-of-mutations: a vector per gene from the co-occurrence
      structure, averaged over the genes a patient carries).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


_AE_CACHE: dict = {}          # per process: (data hash, shape, settings) -> fitted autoencoder state


def _z_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0)
    return mu, np.where(sd > 0, sd, 1.0)


# ---------------------------------------------------------------------------------------- RNA
class GeneSetScorer(BaseEstimator, TransformerMixin):
    """One score per gene set: the mean of the member genes' z-scores (z from the fitting rows).
    `sets` is a list of column-position lists within the RNA block."""

    def __init__(self, sets=None, names=None):
        self.sets, self.names = sets, names

    def fit(self, X, y=None):
        self.mu_, self.sd_ = _z_stats(np.asarray(X, dtype=float))
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return np.column_stack([Z[:, idx].mean(1) for idx in self.sets])


class ModuleScorer(BaseEstimator, TransformerMixin):
    """Co-expression modules found on the fitting rows: average-linkage clustering on 1 - |r|,
    cut at `distance_cut`, modules of at least `min_size` genes. Each module's genes are sign-
    aligned to the module's first principal component, and the score is their mean z-score.
    If no module qualifies, the first principal component of all genes is used instead."""

    def __init__(self, distance_cut=0.7, min_size=5):
        self.distance_cut, self.min_size = distance_cut, min_size

    def fit(self, X, y=None):
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        X = np.asarray(X, dtype=float)
        self.mu_, self.sd_ = _z_stats(X)
        Z = (X - self.mu_) / self.sd_
        R = np.corrcoef(Z, rowvar=False)
        D = 1 - np.abs(R)
        np.fill_diagonal(D, 0)
        lab = fcluster(linkage(squareform(np.clip(D, 0, None), checks=False), method="average"),
                       t=self.distance_cut, criterion="distance")
        self.modules_, self.signs_ = [], []
        for k in sorted(set(lab)):
            idx = np.flatnonzero(lab == k)
            if len(idx) >= self.min_size:
                _, _, vt = np.linalg.svd(Z[:, idx] - Z[:, idx].mean(0), full_matrices=False)
                self.modules_.append(idx)
                self.signs_.append(np.sign(vt[0]) + (vt[0] == 0))
        if not self.modules_:
            idx = np.arange(X.shape[1])
            _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            self.modules_, self.signs_ = [idx], [np.sign(vt[0]) + (vt[0] == 0)]
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return np.column_stack([(Z[:, idx] * s).mean(1) for idx, s in zip(self.modules_, self.signs_)])


class DenoisingAE(BaseEstimator, TransformerMixin):
    """Denoising autoencoder on the RNA block (PyTorch, CPU, one thread). Inputs are z-scored on
    the fitting rows; during training a random `mask` share of inputs is zeroed and the network
    reconstructs the clean input. Several seeds are trained; the one with the lowest
    reconstruction loss on a held-out 15% of the fitting rows is kept (latent spaces of different
    seeds are not aligned, so they are not averaged). Output: the latent code."""

    def __init__(self, latent=16, hidden=128, mask=0.2, weight_decay=1e-4, lr=1e-3, max_epochs=150, patience=10,
                 batch_size=128, val_fraction=0.15, n_seeds=2, random_state=0):
        self.latent, self.hidden, self.mask, self.weight_decay, self.lr = latent, hidden, mask, weight_decay, lr
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.val_fraction, self.n_seeds, self.random_state = val_fraction, n_seeds, random_state

    def fit(self, X, y=None):
        import hashlib
        import torch
        torch.set_num_threads(1)
        X = np.asarray(X, dtype=float)
        # The autoencoder is unsupervised and deterministic, so its result depends only on the fitting
        # rows and its settings: reuse it across the grid of the downstream member (same worker).
        key = (hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest(), X.shape,
               tuple(sorted(self.get_params().items())))
        if key in _AE_CACHE:
            self.mu_, self.sd_, self.encoder_, self.val_loss_ = _AE_CACHE[key]
            return self
        self.mu_, self.sd_ = _z_stats(X)
        Z = ((X - self.mu_) / self.sd_).astype(np.float32)
        d = Z.shape[1]
        best_loss, self.encoder_ = np.inf, None
        for s in range(self.n_seeds):
            seed = int(self.random_state) * 100 + 50 + s
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            va = rng.choice(len(Z), max(1, int(round(self.val_fraction * len(Z)))), replace=False)
            tr = np.setdiff1d(np.arange(len(Z)), va)
            enc = torch.nn.Sequential(torch.nn.Linear(d, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, self.latent))
            dec = torch.nn.Sequential(torch.nn.Linear(self.latent, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, d))
            params = list(enc.parameters()) + list(dec.parameters())
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            Xt, Xv = torch.from_numpy(Z[tr]), torch.from_numpy(Z[va])
            g = torch.Generator().manual_seed(seed)
            best, best_state, wait = np.inf, None, 0
            for _ in range(self.max_epochs):
                enc.train(); dec.train()
                perm = torch.randperm(len(tr), generator=g)
                for i in range(0, len(tr), self.batch_size):
                    xb = Xt[perm[i:i + self.batch_size]]
                    noisy = xb * (torch.rand(xb.shape, generator=g) >= self.mask).float()
                    opt.zero_grad()
                    loss = torch.mean((dec(enc(noisy)) - xb) ** 2)
                    loss.backward()
                    opt.step()
                enc.eval(); dec.eval()
                with torch.no_grad():
                    vl = float(torch.mean((dec(enc(Xv)) - Xv) ** 2))
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_state = {k: v.detach().clone() for k, v in enc.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            if best < best_loss:
                enc.load_state_dict(best_state)
                best_loss, self.encoder_ = best, enc
        self.val_loss_ = best_loss
        if len(_AE_CACHE) >= 256:
            _AE_CACHE.pop(next(iter(_AE_CACHE)))
        _AE_CACHE[key] = (self.mu_, self.sd_, self.encoder_, self.val_loss_)
        return self

    def transform(self, X):
        import torch
        torch.set_num_threads(1)
        Z = ((np.asarray(X, dtype=float) - self.mu_) / self.sd_).astype(np.float32)
        self.encoder_.eval()
        with torch.no_grad():
            return self.encoder_(torch.from_numpy(Z)).numpy().astype(float)


# ---------------------------------------------------------------------------------------- DNA
class ColumnSubset(BaseEstimator, TransformerMixin):
    """Keep the listed column positions (e.g. the known driver genes)."""

    def __init__(self, idx=None):
        self.idx = idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.idx]


class MutationCount(BaseEstimator, TransformerMixin):
    """Number of mutated genes among the given flag columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float).sum(1, keepdims=True)


class PathwayFlags(BaseEstimator, TransformerMixin):
    """One 0/1 flag per gene set: any of the set's genes mutated."""

    def __init__(self, sets=None, names=None):
        self.sets, self.names = sets, names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([(X[:, idx].sum(1) > 0).astype(float) for idx in self.sets])


class CooccurrenceEmbedding(BaseEstimator, TransformerMixin):
    """Bag-of-mutations embedding. Gene vectors come from a truncated SVD of the fitting rows'
    patient x gene mutation matrix (genes that co-occur get similar vectors); a patient's vector
    is the mean of the vectors of the genes they carry (zero if none)."""

    def __init__(self, dim=8, random_state=0):
        self.dim, self.random_state = dim, random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        k = min(self.dim, len(s))
        self.gene_vectors_ = (vt[:k].T * np.sqrt(s[:k]))          # genes x k
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n = X.sum(1, keepdims=True)
        return np.where(n > 0, (X @ self.gene_vectors_) / np.where(n > 0, n, 1), 0.0)
