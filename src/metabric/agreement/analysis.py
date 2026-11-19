"""Agreement analysis (build step 6; PLAN.md sections 8 and 9). Pure functions on tables written by
earlier stages: per-patient out-of-fold predictions (farm), permutation importance (importance).
All predictions used here are outer-fold predictions, so nothing is scored on rows a model saw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..core.stats import corrected_resampled_ttest

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ importance agreement (items 1-4)
def importance_matrix(summ: pd.DataFrame, procedures: list[str], top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model x feature matrix of mean permutation importance (union of each model's top_n), plus a
    consensus strip: mean rank across models and the share of models with the feature in their top_n."""
    s = summ[summ.procedure.isin(procedures)]
    feats = list(dict.fromkeys(s.sort_values("mean", ascending=False).groupby("procedure").head(top_n).item))
    mat = s.pivot_table(index="procedure", columns="item", values="mean").reindex(index=procedures, columns=feats)
    ranks = s.assign(r=s.groupby("procedure")["mean"].rank(ascending=False, method="min"))
    rk = ranks.pivot_table(index="procedure", columns="item", values="r").reindex(index=procedures, columns=feats)
    cons = pd.DataFrame({"mean_rank": rk.mean(), "share_in_top": (rk <= top_n).mean()})
    order = cons.sort_values(["share_in_top", "mean_rank"], ascending=[False, True]).index
    return mat[order], cons.loc[order].rename_axis("item").reset_index()


def rank_overlap(summ: pd.DataFrame, procedures: list[str], ks=(10, 20)) -> pd.DataFrame:
    """Pairwise Jaccard of top-k lists and Spearman correlation of the full importance vectors."""
    s = summ[summ.procedure.isin(procedures)]
    vec = s.pivot_table(index="item", columns="procedure", values="mean")
    rows = []
    for i, a in enumerate(procedures):
        for b in procedures[i + 1:]:
            if a not in vec or b not in vec:
                continue
            r = {"a": a, "b": b, "spearman_importance": vec[[a, b]].dropna().corr(method="spearman").iloc[0, 1]}
            for k in ks:
                ta, tb = set(vec[a].nlargest(k).index), set(vec[b].nlargest(k).index)
                r[f"jaccard_top{k}"] = len(ta & tb) / len(ta | tb)
            rows.append(r)
    return pd.DataFrame(rows)


def prediction_correlation(preds: pd.DataFrame, procedures: list[str]) -> pd.DataFrame:
    """Pairwise Spearman correlation of out-of-fold predictions (each split's test rows; mean over splits)."""
    w = preds[preds.candidate.isin(procedures)].pivot_table(index=["split_id", "patient_id"], columns="candidate", values="p")
    rows = []
    for i, a in enumerate(procedures):
        for b in procedures[i + 1:]:
            if a in w and b in w:
                rho = w[[a, b]].groupby(level=0).apply(lambda d: d.corr(method="spearman").iloc[0, 1]).mean()
                rows.append({"a": a, "b": b, "pred_spearman": float(rho)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ patients (item 6; F-058)
def disagreement(preds: pd.DataFrame, procedures: list[str], n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Per patient and split: spread of the members' logits (SD) and the error of their mean
    prediction. The question is whether disagreement predicts error *beyond* what the predicted
    risk already implies: excess error = squared error - p(1-p) (its expectation under calibration).
    The flag earns its place if Spearman(SD, excess error) is positive with a bootstrap CI above 0."""
    w = preds[preds.candidate.isin(procedures)].pivot_table(index=["split_id", "patient_id"], columns="candidate", values="p")
    y = preds.drop_duplicates(["split_id", "patient_id"]).set_index(["split_id", "patient_id"]).y.loc[w.index]
    L = _logit(w.to_numpy())
    p = 1 / (1 + np.exp(-L.mean(1)))
    d = pd.DataFrame({"y": y.to_numpy(), "p_mean": p, "sd_logit": L.std(1, ddof=1)}, index=w.index)
    d["sq_error"] = (d.y - d.p_mean) ** 2
    d["excess_error"] = d.sq_error - d.p_mean * (1 - d.p_mean)
    pt = d.groupby(level="patient_id").mean()                      # each patient once (mean over the 2 repeats)
    rho = pt[["sd_logit", "excess_error"]].corr(method="spearman").iloc[0, 1]
    boots = []
    idx = np.arange(len(pt))
    for _ in range(n_boot):
        b = pt.iloc[rng.choice(idx, len(idx))]
        boots.append(b[["sd_logit", "excess_error"]].corr(method="spearman").iloc[0, 1])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    pt["quintile"] = pd.qcut(pt.sd_logit.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    q = pt.groupby("quintile", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "sd_logit": g.sd_logit.mean(), "brier": g.sq_error.mean(), "expected_brier": (g.p_mean * (1 - g.p_mean)).mean(),
        "excess": g.excess_error.mean(), "auroc": roc_auc_score(g.y, g.p_mean) if g.y.nunique() > 1 else np.nan})).reset_index()
    verdict = {"spearman_sd_vs_excess_error": float(rho), "ci_low": float(lo), "ci_high": float(hi),
               "flag_earns_place": bool(lo > 0), "n_patients": int(len(pt)), "n_members": len(procedures)}
    return q, verdict


def robustness(preds: pd.DataFrame, procedures: list[str], combined: dict[str, list[str]]) -> pd.DataFrame:
    """Prediction stability: each patient is tested once per repeat (different training sets), so
    |p(repeat 0) - p(repeat 1)| measures how much a model's prediction for the same patient moves.
    `combined`: name -> member procedures whose mean (logit) forms a combined prediction."""
    w = preds[preds.candidate.isin(procedures)].pivot_table(index=["repeat", "patient_id"], columns="candidate", values="p")
    for name, mem in combined.items():
        w[name] = 1 / (1 + np.exp(-_logit(w[mem].to_numpy()).mean(1)))
    r0, r1 = w.xs(0, level="repeat"), w.xs(1, level="repeat")
    r1 = r1.loc[r0.index]
    diff = (r0 - r1).abs()
    return pd.DataFrame({"model": diff.columns, "mean_abs_change": diff.mean().to_numpy(),
                         "p90_abs_change": diff.quantile(0.9).to_numpy()}).sort_values("mean_abs_change")


# ------------------------------------------------------------------ the omics question (section 9)
def ablation(auc: pd.DataFrame, steps: list[tuple[str, str | None]], n_train: float, n_test: float) -> pd.DataFrame:
    """auc: rows split_id, columns procedure, values outer AUROC on identical splits. `steps`:
    (procedure, baseline or None); each procedure's AUROC with a corrected 95% CI, and its paired
    difference from the baseline (corrected resampled t)."""
    rows = []
    for p, base in steps:
        if p not in auc:
            continue
        v = auc[p].dropna()
        t = corrected_resampled_ttest(v.to_numpy(), n_train, n_test)
        r = {"procedure": p, "mean_auroc": v.mean(), "ci_low": t["ci_low"], "ci_high": t["ci_high"], "vs": base or ""}
        if base and base in auc:
            d = corrected_resampled_ttest((v - auc[base].loc[v.index]).to_numpy(), n_train, n_test)
            r.update(diff=d["mean_diff"], diff_ci_low=d["ci_low"], diff_ci_high=d["ci_high"], p=d["p"])
        rows.append(r)
    return pd.DataFrame(rows)
