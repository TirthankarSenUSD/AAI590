"""Phase 2 descriptive survival and model checks (training rows only).

- Kaplan-Meier curves overall and by group, with log-rank tests.
- Proportional hazards: a Cox model on the clinical columns of the Cox members, then scaled
  Schoenfeld residuals against the rank of event time (Grambsch-Therneau style correlation test),
  BH-adjusted across columns. If the stratification column fails, the Cox members are stratified on it.
- Collinearity among NPI, stage, size, nodes and grade: Spearman correlations and VIFs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..core.stats import bh_fdr, kaplan_meier


def km_by_group(time: pd.Series, event: pd.Series, group: pd.Series | None, horizons: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """(KM curves long table, survival at the horizons per group, log-rank test or None)."""
    from sksurv.compare import compare_survival

    from .survival import structured
    g = pd.Series("all", index=time.index) if group is None else group.astype(object).where(group.notna(), "missing")
    curves, at = [], []
    for lev in sorted(g.unique(), key=str):
        m = (g == lev).to_numpy()
        km = kaplan_meier(time[m], event[m])
        km.insert(0, "group", lev)
        curves.append(km)
        for h in horizons:
            r = km[km.time <= h].iloc[-1]
            at.append({"group": lev, "n": int(m.sum()), "deaths": int(event[m].sum()), "horizon_months": h,
                       "survival": float(r.survival), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high),
                       "n_at_risk": int((time[m] >= h).sum())})
    test = None
    if group is not None:
        keep = (g != "missing").to_numpy()
        if g[keep].nunique() > 1:
            chi2, p = compare_survival(structured(time[keep], event[keep]), g[keep].astype(str).to_numpy())
            test = {"chi2": float(chi2), "df": int(g[keep].nunique() - 1), "p": float(p), "n": int(keep.sum())}
    return pd.concat(curves, ignore_index=True), pd.DataFrame(at), test


def cox_design(frame: pd.DataFrame, groups: dict, rare: int) -> pd.DataFrame:
    """Unpenalised-Cox design for the checks: numbers median-filled with a missing flag (where needed) and
    standardised; categories with rare levels pooled, missing as its own level, one level dropped."""
    parts = []
    for c in groups["numeric"]:
        v = frame[c].astype(float)
        if v.isna().any():
            parts.append(v.isna().astype(float).rename(f"{c}__missing"))
        v = v.fillna(v.median())
        parts.append(((v - v.mean()) / v.std()).rename(c))
    for c in groups["binary"]:
        parts.append(frame[c].astype(float).rename(c))
    for c in groups["categorical"]:
        v = frame[c].astype(object).where(frame[c].notna(), "missing").astype(str)
        vc = v.value_counts()
        v = v.where(v.map(vc) >= rare, "other")
        d = pd.get_dummies(v, prefix=c, prefix_sep="=", dtype=float)
        parts.append(d.drop(columns=d.sum().idxmax()))       # reference = the most frequent level
    X = pd.concat(parts, axis=1)
    return X.loc[:, X.std() > 0]


def ph_test(X: pd.DataFrame, time: pd.Series, event: pd.Series, alpha: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-column PH test: correlation of the scaled Schoenfeld residuals with the rank of event time.
    Returns (table, scaled residuals of event rows, their event times)."""
    from statsmodels.duration.hazard_regression import PHReg
    res = PHReg(time.to_numpy(), X.to_numpy(), status=event.to_numpy(), ties="breslow").fit(disp=False)
    sr = res.schoenfeld_residuals
    ev = ~np.isnan(sr).any(axis=1)
    d = int(ev.sum())
    scaled = res.params + d * sr[ev] @ res.cov_params()
    te = time.to_numpy()[ev]
    g = stats.rankdata(te)
    rows = []
    for j, c in enumerate(X.columns):
        r, p = stats.pearsonr(g, scaled[:, j])
        rows.append({"column": c, "coef": float(res.params[j]), "hazard_ratio": float(np.exp(res.params[j])),
                     "coef_p": float(res.pvalues[j]), "rho_time": float(r), "ph_p": float(p)})
    t = pd.DataFrame(rows)
    t["ph_p_bh"] = bh_fdr(t.ph_p.to_numpy())
    t["ph_violated"] = t.ph_p_bh < alpha
    return t.sort_values("ph_p").reset_index(drop=True), scaled, te


def collinearity(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Spearman correlation matrix, VIF per column) on complete rows."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    d = frame[cols].astype(float).dropna()
    corr = d.corr(method="spearman")
    Z = np.column_stack([np.ones(len(d)), ((d - d.mean()) / d.std()).to_numpy()])
    vif = pd.DataFrame({"column": cols, "vif": [float(variance_inflation_factor(Z, j + 1)) for j in range(len(cols))],
                        "n_complete": len(d)})
    return corr, vif
