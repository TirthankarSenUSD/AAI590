"""Probability metrics for the harness (death = positive class)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6
BIG_C = 1e6          # effectively unpenalised (sklearn 1.8 deprecates penalty=None)


def clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p) -> np.ndarray:
    p = clip(p)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calibration_intercept(y, p) -> float:
    """Calibration-in-the-large: intercept a in logit P(y) = a + logit(p) (offset fixed). 0 = calibrated."""
    y, lp = np.asarray(y, float), logit(p)
    a = 0.0
    for _ in range(50):
        mu = _sigmoid(a + lp)
        step = (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def calibration_slope(y, p) -> float:
    """Slope b in logit P(y) = c + b logit(p). 1 = calibrated; < 1 = predictions too extreme."""
    lp = logit(p)
    if np.std(lp) < 1e-9:
        return float("nan")
    m = LogisticRegression(C=BIG_C, max_iter=1000).fit(lp.reshape(-1, 1), np.asarray(y))
    return float(m.coef_[0, 0])


def binary_metrics(y, p, train_prevalence: float) -> dict:
    """Threshold-free metrics. Brier skill uses the training-fold prevalence as the reference
    forecast, so nothing about the test labels enters the reference."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pc = clip(p)
    brier = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((train_prevalence - y) ** 2))
    one_class = y.min() == y.max()
    return {
        "n": int(len(y)), "n_deaths": int(y.sum()),
        "auroc": float("nan") if one_class else float(roc_auc_score(y, p)),
        "auprc": float("nan") if one_class else float(average_precision_score(y, p)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "brier": brier, "brier_ref": brier_ref, "brier_skill": 1 - brier / brier_ref if brier_ref > 0 else float("nan"),
        "cal_intercept": calibration_intercept(y, p), "cal_slope": calibration_slope(y, p),
        "mean_pred": float(p.mean()), "obs_rate": float(y.mean()),
    }


def reliability_table(y, p, bins: int) -> pd.DataFrame:
    """Quantile bins of predicted risk: mean prediction against observed death rate."""
    d = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), q=min(bins, len(d)), labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
    return g.reset_index()


def threshold_for_sensitivity(y, p, target: float) -> float:
    """Largest threshold t with sensitivity(p >= t) >= target. Fit on inner-fold predictions only."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return float("nan")
    need = int(np.ceil(target * len(pos)))
    return float(pos[max(need, 1) - 1])


def threshold_for_f1(y, p) -> float:
    """Threshold maximising F1 (death = positive). Fit on inner-fold predictions only; ties -> the
    higher threshold. A secondary operating point, never the headline."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)                      # predicted positives when cutting after rank k
    f1 = 2 * tp / (k + ys.sum())
    last = np.r_[ps[1:] != ps[:-1], True]              # only cut between distinct scores
    f1 = np.where(last, f1, -1)
    return float(ps[int(np.argmax(f1))])


def threshold_metrics(y, p, t: float) -> dict:
    y, pred = np.asarray(y, int), np.asarray(p, float) >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    sens, ppv = div(tp, tp + fn), div(tp, tp + fp)
    return {"threshold": t, "sensitivity": sens, "specificity": div(tn, tn + fp), "ppv": ppv, "npv": div(tn, tn + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn)}


def bootstrap_auroc_ci(y, p, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    y, p = np.asarray(y, int), np.asarray(p, float)
    if y.min() == y.max():
        return (float("nan"), float("nan"))
    vals = []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
