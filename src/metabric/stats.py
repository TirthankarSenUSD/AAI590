"""Small, dependency-light statistics used by the EDA (no lifelines / statsmodels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def auroc(score, y) -> float:
    """Mann-Whitney AUROC of `score` for y == 1. Rows with a missing score are dropped."""
    s = pd.Series(np.asarray(score, dtype=float))
    t = pd.Series(np.asarray(y)).astype(int)
    ok = s.notna().to_numpy()
    s, t = s[ok].to_numpy(), t[ok].to_numpy()
    n1, n0 = t.sum(), (1 - t).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC per column of a complete matrix (vectorised via ranks)."""
    y = np.asarray(y).astype(int)
    R = stats.rankdata(X, axis=0)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (y @ R - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_permutation_null(X: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator,
                           chunk: int = 100) -> np.ndarray:
    """(n_perm x n_cols) AUROCs under shuffled labels."""
    y = np.asarray(y).astype(int)
    R = stats.rankdata(X, axis=0)
    n1, n0 = y.sum(), len(y) - y.sum()
    out = np.empty((n_perm, X.shape[1]))
    for start in range(0, n_perm, chunk):
        b = min(chunk, n_perm - start)
        Y = np.stack([rng.permutation(y) for _ in range(b)])
        out[start:start + b] = (Y @ R - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def permutation_pvalues(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Two-sided per-column p-value on |AUROC - 0.5|."""
    exceed = (np.abs(null - 0.5) >= np.abs(obs - 0.5)[None, :]).sum(axis=0)
    return (1 + exceed) / (null.shape[0] + 1)


def bh_fdr(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def wilson(k, n, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))


def rate_table(groups: pd.Series, y: pd.Series, missing_label: str = "missing") -> pd.DataFrame:
    """Death rate per level with Wilson 95% interval. Missing values form their own level."""
    g = groups.astype(object).where(groups.notna(), missing_label)
    rows = []
    for level, idx in g.groupby(g, sort=True, observed=True).groups.items():
        k, n = int(y.loc[idx].sum()), len(idx)
        lo, hi = wilson(k, n)
        rows.append({"level": str(level), "n": n, "deaths": k, "death_rate": k / n, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def cramers_v(a: pd.Series, b: pd.Series, bias_corrected: bool = False) -> float:
    ct = pd.crosstab(a, b)  # drops missing
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    phi2 = chi2 / n
    if not bias_corrected:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else float("nan")


def kaplan_meier(time, event) -> pd.DataFrame:
    """KM estimate with Greenwood log-log 95% interval. event: 1 = death."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    times = np.unique(t[e == 1])
    s, var_sum, rows = 1.0, 0.0, [{"time": 0.0, "n_at_risk": len(t), "events": 0, "survival": 1.0,
                                     "ci_low": 1.0, "ci_high": 1.0}]
    for ti in times:
        n = int((t >= ti).sum())
        d = int(((t == ti) & (e == 1)).sum())
        s *= 1 - d / n
        if n > d:
            var_sum += d / (n * (n - d))
        if 0 < s < 1 and var_sum > 0:
            se = np.sqrt(var_sum) / abs(np.log(s))
            lo, hi = s ** np.exp(1.96 * se), s ** np.exp(-1.96 * se)
        else:
            lo = hi = s
        rows.append({"time": float(ti), "n_at_risk": n, "events": d, "survival": s, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def corrected_resampled_ttest(diffs, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test for paired per-fold differences.

    Variance is inflated by (1/J + n_test/n_train) instead of 1/J because the training sets of
    the J folds/repeats overlap. Returns mean, corrected SE, t, two-sided p and a 95% CI.
    """
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    mean = float(d.mean())
    s2 = float(d.var(ddof=1)) if J > 1 else float("nan")
    se = float(np.sqrt((1 / J + n_test / n_train) * s2)) if J > 1 else float("nan")
    if se == 0 or np.isnan(se):
        t, p = (float("nan"), 1.0) if mean == 0 or np.isnan(se) else (float("inf"), 0.0)
        half = 0.0 if se == 0 else float("nan")
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), df=J - 1))
        half = float(stats.t.ppf(0.975, df=J - 1) * se)
    return {"mean_diff": mean, "se_corrected": se, "t": t, "p": p, "ci_low": mean - half, "ci_high": mean + half, "J": J}


def smd(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0
