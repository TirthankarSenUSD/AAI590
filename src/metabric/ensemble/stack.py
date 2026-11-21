"""Ensemble building blocks (agreement merge, stacker, averages). Every function here is fitted on the
outer training rows' inner out-of-fold member predictions only, then applied to the outer test rows.

Options, simplest first: best_single < mean < weighted < stacker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from ..harness.core import select_candidate

OPTIONS = ("best_single", "mean", "weighted", "stacker")
EPS = 1e-6


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def _auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if y.min() != y.max() else 0.5


# ------------------------------------------------------------------------------------- merging
def merge_groups(P: pd.DataFrame, threshold: float) -> list[list[str]]:
    """Members whose predictions have Spearman correlation above `threshold` are joined (connected
    components). Returns groups of member names, each in input order."""
    names = list(P.columns)
    rho = P.rank().corr().to_numpy()                   # Spearman = Pearson on ranks
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if rho[i, j] > threshold:
                parent[find(j)] = find(i)
    groups: dict[int, list[str]] = {}
    for i, n in enumerate(names):
        groups.setdefault(find(i), []).append(n)
    return list(groups.values())


def merge_weights(groups: list[list[str]], aucs: dict[str, float]) -> dict[str, dict[str, float]]:
    """Within a group, weights proportional to (AUROC - 0.5), floored at a small positive value."""
    out = {}
    for g in groups:
        w = np.array([max(aucs[m] - 0.5, 1e-3) for m in g])
        out["+".join(g)] = dict(zip(g, w / w.sum()))
    return out


def apply_merge(P: pd.DataFrame, weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Merged members: weighted average of the members' logits, back to probabilities."""
    return pd.DataFrame({k: sigmoid(sum(w * logit(P[m]) for m, w in ws.items())) for k, ws in weights.items()},
                        index=P.index)


# ------------------------------------------------------------------------------------- stacker
class NonNegStacker:
    """Logistic regression on member logits with non-negative weights and an L2 penalty
    (intercept unpenalised and free). Fitted by bounded L-BFGS."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, Z: np.ndarray, y) -> "NonNegStacker":
        Z, y = np.asarray(Z, float), np.asarray(y, float)
        n, m = Z.shape

        def loss(theta):
            b, w = theta[0], theta[1:]
            s = b + Z @ w
            ll = np.sum(np.logaddexp(0, s) - y * s) / n + self.l2 * np.sum(w ** 2) / n
            r = (sigmoid(s) - y) / n
            return ll, np.r_[r.sum(), Z.T @ r + 2 * self.l2 * w / n]
        x0 = np.r_[0.0, np.full(m, 1.0 / m)]
        res = minimize(loss, x0, jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * m)
        self.intercept_, self.coef_ = float(res.x[0]), res.x[1:]
        return self

    def predict_proba(self, Z) -> np.ndarray:
        return sigmoid(self.intercept_ + np.asarray(Z, float) @ self.coef_)


# ------------------------------------------------------------------------------------- options
def fold_aucs(y: pd.Series, P: pd.DataFrame, fold_of: pd.Series) -> pd.DataFrame:
    """AUROC of each column within each fold (fold_of: fold label per row)."""
    rows = []
    for f in sorted(fold_of.unique()):
        idx = fold_of.index[fold_of == f]
        rows.append({c: _auc(y.loc[idx], P.loc[idx, c]) for c in P.columns})
    return pd.DataFrame(rows)


def choose_member(y, P, fold_of, complexity: dict[str, float], within_se: float) -> str:
    fa = fold_aucs(y, P, fold_of)
    st = pd.DataFrame({"candidate": P.columns, "inner_auroc": fa.mean().to_numpy(),
                       "inner_auroc_se": (fa.std(ddof=1) / np.sqrt(len(fa))).to_numpy(),
                       "complexity": [complexity[c] for c in P.columns], "eligible": True,
                       "inner_log_loss": 0.0, "inner_brier": 0.0})
    return select_candidate(st, within_se)


def fit_option(option: str, y, P: pd.DataFrame, fold_of, complexity, within_se, l2):
    """Fit one option on (y, P); returns a function P_new -> probabilities, plus its parameters."""
    if option == "best_single":
        m = choose_member(y, P, fold_of, complexity, within_se)
        return (lambda Q: Q[m].to_numpy()), {"member": m}
    if option == "mean":
        return (lambda Q: Q.mean(axis=1).to_numpy()), {}
    if option == "weighted":
        aucs = {c: _auc(y, P[c]) for c in P.columns}
        w = pd.Series({c: max(a - 0.5, 0.0) for c, a in aucs.items()})
        w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(w), index=w.index)
        return (lambda Q: (Q[w.index] * w.to_numpy()).sum(axis=1).to_numpy()), {"weights": w.to_dict()}
    if option == "stacker":
        s = NonNegStacker(l2).fit(logit(P.to_numpy()), y)
        cols = list(P.columns)
        return (lambda Q: s.predict_proba(logit(Q[cols].to_numpy()))), {"weights": dict(zip(cols, s.coef_)),
                                                                          "intercept": s.intercept_}
    raise ValueError(option)


def choose_option(y, P, fold_of, complexity, within_se, l2, k: int, seed: int) -> tuple[str, pd.DataFrame]:
    """Second-level CV over the outer training rows: each option is fitted on k-1 parts of the
    inner out-of-fold predictions and scored on the held part; the fixed rule (best mean AUROC,
    simplest within 1 SE, order OPTIONS) picks the option. Returns the choice and the per-part AUROCs."""
    from sklearn.model_selection import StratifiedKFold
    parts = StratifiedKFold(k, shuffle=True, random_state=seed).split(P, y)
    rows = []
    for j, (a, b) in enumerate(parts):
        ia, ib = P.index[a], P.index[b]
        for o in OPTIONS:
            f, _ = fit_option(o, y.loc[ia], P.loc[ia], fold_of.loc[ia], complexity, within_se, l2)
            rows.append({"part": j, "option": o, "auroc": _auc(y.loc[ib], f(P.loc[ib]))})
    r = pd.DataFrame(rows)
    g = r.groupby("option").auroc
    st = pd.DataFrame({"candidate": list(OPTIONS), "inner_auroc": [g.mean()[o] for o in OPTIONS],
                       "inner_auroc_se": [g.std(ddof=1)[o] / np.sqrt(k) for o in OPTIONS],
                       "complexity": range(len(OPTIONS)), "eligible": True, "inner_log_loss": 0.0, "inner_brier": 0.0})
    return select_candidate(st, within_se), r
