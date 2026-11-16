"""Neural farm members (step 3b), as scikit-learn-compatible classifiers so the harness can fit
them inside every fold like any other member.

Guard rails for ~1,200 training rows: small networks, dropout, weight decay (AdamW), early
stopping on a stratified validation split carved from the *fitting* rows only, and predictions
averaged over several seeds. One CPU thread per fit; every seed is fixed, so results are
reproducible and identical for any number of parallel workers.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def _split(y: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/validation split of the fitting rows (for early stopping only)."""
    va = np.concatenate([rng.choice(np.flatnonzero(y == v), max(1, int(round(frac * (y == v).sum()))), replace=False)
                         for v in (0, 1)])
    tr = np.setdiff1d(np.arange(len(y)), va)
    return tr, va


class _TorchBase(BaseEstimator, ClassifierMixin):
    """Shared training loop: binary cross-entropy, AdamW, early stopping, seed averaging."""

    def _build(self, torch, X: np.ndarray):          # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _forward(self, net, xb, train: bool):
        return net(xb)

    def fit(self, X, y):
        torch = _torch()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.nets_ = []
        for s in range(self.n_seeds):
            seed = int(self.random_state) * 100 + s
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            tr, va = _split(y.astype(int), self.val_fraction, rng)
            net = self._build(torch, X)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
            Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va])
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
            best, best_state, wait = np.inf, None, 0
            g = torch.Generator().manual_seed(seed)
            for _ in range(self.max_epochs):
                net.train()
                perm = torch.randperm(len(tr), generator=g)
                for i in range(0, len(tr), self.batch_size):
                    b = perm[i:i + self.batch_size]
                    opt.zero_grad()
                    loss = loss_fn(self._forward(net, Xt[b], True).squeeze(1), yt[b])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(self._forward(net, Xv, False).squeeze(1), yv))
                if vl < best - 1e-4:
                    best, wait = vl, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets_.append(net)
        return self

    def predict_proba(self, X):
        torch = _torch()
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = np.mean([torch.sigmoid(self._forward(n, X, False).squeeze(1)).numpy() for n in self.nets_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchMLP(_TorchBase):
    """Member 5: plain MLP on the preprocessed (one-hot, scaled) features."""

    def __init__(self, hidden=(64,), dropout=0.3, weight_decay=1e-3, lr=1e-3, max_epochs=200, patience=15,
                 batch_size=128, val_fraction=0.15, n_seeds=2, random_state=0):
        self.hidden, self.dropout, self.weight_decay, self.lr = hidden, dropout, weight_decay, lr
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.val_fraction, self.n_seeds, self.random_state = val_fraction, n_seeds, random_state

    def _build(self, torch, X):
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        return torch.nn.Sequential(*layers)


class _EmbedNet:
    @staticmethod
    def make(torch, n_dense, cards, emb_dims, hidden, dropout):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embs = torch.nn.ModuleList([torch.nn.Embedding(c, e) for c, e in zip(cards, emb_dims)])
                layers, d = [], n_dense + sum(emb_dims)
                for h in hidden:
                    layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
                    d = h
                layers.append(torch.nn.Linear(d, 1))
                self.mlp = torch.nn.Sequential(*layers)
                self.n_dense = n_dense

            def forward(self, x):
                dense = x[:, : self.n_dense]
                codes = x[:, self.n_dense:].long()
                parts = [dense] + [e(codes[:, i]) for i, e in enumerate(self.embs)]
                return self.mlp(torch.cat(parts, dim=1))
        return Net()


class TorchEmbedMLP(_TorchBase):
    """Member 6: MLP with a learned embedding per categorical clinical column. The last `n_cat`
    input columns are integer category codes (0 = unknown / pooled rare level)."""

    def __init__(self, n_cat=1, hidden=(64,), dropout=0.3, weight_decay=1e-3, lr=1e-3, max_epochs=200, patience=15,
                 batch_size=128, val_fraction=0.15, n_seeds=2, random_state=0):
        self.n_cat, self.hidden, self.dropout, self.weight_decay, self.lr = n_cat, hidden, dropout, weight_decay, lr
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.val_fraction, self.n_seeds, self.random_state = val_fraction, n_seeds, random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        codes = X[:, X.shape[1] - self.n_cat:]
        self.cards_ = [int(codes[:, i].max()) + 2 for i in range(self.n_cat)]   # +1 spare for unseen codes
        return super().fit(X, y)

    def _build(self, torch, X):
        emb = [min(8, (c + 1) // 2) for c in self.cards_]
        return _EmbedNet.make(torch, X.shape[1] - self.n_cat, self.cards_, emb, self.hidden, self.dropout)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32).copy()
        k = X.shape[1] - self.n_cat
        for i, c in enumerate(self.cards_):                       # codes never seen in fit -> spare index
            X[:, k + i] = np.clip(X[:, k + i], 0, c - 1)
        return super().predict_proba(X)


class TorchMultimodal(_TorchBase):
    """Member 7: one encoder per data block (clinical, RNA), concatenated into a small head. During
    training each block's encoding is dropped (zeroed) for a random subset of patients
    (`block_dropout`), so the head cannot rely on a single block. The last `n_rna` input columns
    are the RNA block."""

    def __init__(self, n_rna=489, clin_dim=16, rna_dim=32, block_dropout=0.2, dropout=0.3, weight_decay=1e-3, lr=1e-3,
                 max_epochs=200, patience=15, batch_size=128, val_fraction=0.15, n_seeds=2, random_state=0):
        self.n_rna, self.clin_dim, self.rna_dim, self.block_dropout = n_rna, clin_dim, rna_dim, block_dropout
        self.dropout, self.weight_decay, self.lr = dropout, weight_decay, lr
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.val_fraction, self.n_seeds, self.random_state = val_fraction, n_seeds, random_state

    def _build(self, torch, X):
        n_clin, n_rna, bd = X.shape[1] - self.n_rna, self.n_rna, self.block_dropout
        clin_dim, rna_dim, p = self.clin_dim, self.rna_dim, self.dropout

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clin = torch.nn.Sequential(torch.nn.Linear(n_clin, clin_dim), torch.nn.ReLU(), torch.nn.Dropout(p))
                self.rna = torch.nn.Sequential(torch.nn.Linear(n_rna, rna_dim), torch.nn.ReLU(), torch.nn.Dropout(p))
                self.head = torch.nn.Linear(clin_dim + rna_dim, 1)

            def forward(self, x, train=False):
                c, r = self.clin(x[:, :n_clin]), self.rna(x[:, n_clin:])
                if train and bd > 0:
                    keep_c = (torch.rand(len(x), 1) >= bd).float()
                    keep_r = (torch.rand(len(x), 1) >= bd).float()
                    both_off = (keep_c + keep_r) == 0                   # never drop both blocks
                    keep_c = torch.where(both_off, torch.ones_like(keep_c), keep_c)
                    c, r = c * keep_c, r * keep_r
                return self.head(torch.cat([c, r], dim=1))
        return Net()

    def _forward(self, net, xb, train):
        return net(xb, train=train)
