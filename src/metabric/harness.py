"""Evaluation harness (build step 2; CLAUDE.md section 7, PLAN.md section 6).

Every fitted step happens inside a fold:
- outer split: fit on the outer training rows, score the outer test rows;
- inner splits (inside the outer training rows): inner AUROC (for selection and optimism),
  inner out-of-fold predictions -> the operating-point threshold and the Platt recalibrator,
  both then applied to the outer test predictions.
Identical frozen splits for every candidate, so comparisons are paired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import metrics as M
from .stats import corrected_resampled_ttest


@dataclass
class Candidate:
    name: str
    make: Callable[[], Any]           # returns an unfitted estimator with fit / predict_proba
    columns: list[str]
    complexity: float = 0.0           # lower = simpler (1-SE rule)
    eligible: bool = True             # floors are never eligible to win
    label: str = ""


@dataclass
class Split:
    split_id: str
    repeat: int
    outer: int
    train: pd.Index
    test: pd.Index
    inner: list[tuple[pd.Index, pd.Index]]


def splits_from_table(s: pd.DataFrame) -> list[Split]:
    out = []
    for sid, g in s.groupby("split_id", sort=False):
        tr = g[g.role == "train"]
        inner = [(pd.Index(tr.loc[tr.inner_fold != j, "patient_id"]), pd.Index(tr.loc[tr.inner_fold == j, "patient_id"]))
                 for j in sorted(tr.inner_fold.unique())]
        out.append(Split(sid, int(g.repeat.iloc[0]), int(g.outer_fold.iloc[0]), pd.Index(tr.patient_id),
                         pd.Index(g.loc[g.role == "test", "patient_id"]), inner))
    return out


def check_split_hygiene(splits: list[Split], all_ids: pd.Index) -> None:
    """Raise if any split leaks: train/test overlap, inner rows outside the outer train, or a
    repeat in which a patient is not tested exactly once."""
    ids = set(all_ids)
    for s in splits:
        if set(s.train) & set(s.test):
            raise AssertionError(f"{s.split_id}: train and test overlap")
        if set(s.train) | set(s.test) != ids:
            raise AssertionError(f"{s.split_id}: train + test != all training rows")
        for tr, va in s.inner:
            if set(tr) & set(va) or not (set(tr) | set(va)) <= set(s.train) or set(va) & set(s.test):
                raise AssertionError(f"{s.split_id}: inner split leaks outside the outer training rows")
    for r in {s.repeat for s in splits}:
        tested = pd.Index(np.concatenate([s.test.to_numpy() for s in splits if s.repeat == r]))
        if tested.duplicated().any() or set(tested) != ids:
            raise AssertionError(f"repeat {r}: patients not tested exactly once")


def select_candidate(stats: pd.DataFrame, within_se: float) -> str:
    """Fixed rule: best mean inner AUROC; among candidates within `within_se` SE of the best,
    the simplest wins; ties broken by lower inner log loss, then lower inner Brier."""
    s = stats[stats.eligible]
    if s.empty:
        raise ValueError("no eligible candidate")
    best = s.sort_values("inner_auroc", ascending=False).iloc[0]
    pool = s[s.inner_auroc >= best.inner_auroc - within_se * best.inner_auroc_se]
    return pool.sort_values(["complexity", "inner_log_loss", "inner_brier"]).iloc[0]["candidate"]


@dataclass
class Result:
    folds: pd.DataFrame
    preds: pd.DataFrame
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)


class Harness:
    def __init__(self, cfg, X: pd.DataFrame, y: pd.Series, splits: list[Split]):
        self.cfg, self.X, self.y, self.splits = cfg, X, y.astype(int), splits
        self.h = cfg["harness"]
        check_split_hygiene(splits, X.index)
        self.n_train = float(np.mean([len(s.train) for s in splits]))
        self.n_test = float(np.mean([len(s.test) for s in splits]))

    # -- core ------------------------------------------------------------------------------
    def _fit(self, cand: Candidate, rows: pd.Index, y: pd.Series):
        est = cand.make()
        est.fit(self.X.loc[rows, cand.columns], y.loc[rows].to_numpy())
        return est

    def _proba(self, est, cand: Candidate, rows: pd.Index) -> np.ndarray:
        return est.predict_proba(self.X.loc[rows, cand.columns])[:, 1]

    def evaluate(self, candidates: list[Candidate], y: pd.Series | None = None, select: bool = False,
                 light: bool = False) -> Result:
        """Score every candidate on every frozen split. `light` skips the inner layer (used by the
        permutation test when no selection is needed). `select` also scores the select-the-best
        procedure, choosing inside each outer training fold by the fixed rule."""
        y = self.y if y is None else y.astype(int)
        fold_rows, pred_rows, sel_rows = [], [], []
        for s in self.splits:
            ytr, yte = y.loc[s.train], y.loc[s.test]
            prev = float(ytr.mean())
            inner_stats, outer_p = [], {}
            for c in candidates:
                rec = {"candidate": c.name, "split_id": s.split_id, "repeat": s.repeat, "outer_fold": s.outer}
                if not light or select:
                    oof = pd.Series(np.nan, index=s.train)
                    aucs, lls, brs = [], [], []
                    for tr, va in s.inner:
                        p_va = self._proba(self._fit(c, tr, y), c, va)
                        oof.loc[va] = p_va
                        yv = y.loc[va].to_numpy()
                        aucs.append(roc_auc_score(yv, p_va) if yv.min() != yv.max() else np.nan)
                        pc = M.clip(p_va)
                        lls.append(-np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)))
                        brs.append(np.mean((p_va - yv) ** 2))
                    rec.update(inner_auroc=float(np.nanmean(aucs)),
                               inner_auroc_se=float(np.nanstd(aucs, ddof=1) / np.sqrt(len(aucs))),
                               inner_log_loss=float(np.mean(lls)), inner_brier=float(np.mean(brs)))
                    inner_stats.append({"candidate": c.name, "complexity": c.complexity, "eligible": c.eligible,
                                        **{k: rec[k] for k in ("inner_auroc", "inner_auroc_se", "inner_log_loss", "inner_brier")}})
                est = self._fit(c, s.train, y)
                p_te, p_tr = self._proba(est, c, s.test), self._proba(est, c, s.train)
                outer_p[c.name] = p_te
                m = M.binary_metrics(yte, p_te, prev)
                rec.update(m)
                rec["train_auroc"] = float(roc_auc_score(ytr, p_tr)) if ytr.nunique() > 1 else np.nan
                rec["optimism"] = rec.get("inner_auroc", np.nan) - m["auroc"]
                p_rc = np.full(len(p_te), np.nan)
                if not light:
                    t = M.threshold_for_sensitivity(ytr.loc[oof.index], oof.to_numpy(), self.h["operating_point_sensitivity"])
                    rec.update({f"op_{k}": v for k, v in M.threshold_metrics(yte, p_te, t).items()})
                    cal = LogisticRegression(C=M.BIG_C, max_iter=1000).fit(M.logit(oof.to_numpy()).reshape(-1, 1), ytr.loc[oof.index])
                    p_rc = cal.predict_proba(M.logit(p_te).reshape(-1, 1))[:, 1]
                    rec.update({f"recal_{k}": v for k, v in M.binary_metrics(yte, p_rc, prev).items()
                                if k in ("auroc", "log_loss", "brier", "brier_skill", "cal_intercept", "cal_slope")})
                fold_rows.append(rec)
                pred_rows.append(pd.DataFrame({"candidate": c.name, "split_id": s.split_id, "repeat": s.repeat,
                                               "outer_fold": s.outer, "patient_id": s.test, "y": yte.to_numpy(),
                                               "p": p_te, "p_recal": p_rc}))
            if select:
                st = pd.DataFrame(inner_stats)
                win = select_candidate(st, self.h["select_within_se"])
                p = outer_p[win]
                m = M.binary_metrics(yte, p, prev)
                best_inner = float(st.set_index("candidate").loc[win, "inner_auroc"])
                sel_rows.append({"split_id": s.split_id, "repeat": s.repeat, "outer_fold": s.outer, "winner": win,
                                 "winner_inner_auroc": best_inner, "outer_auroc": m["auroc"],
                                 "optimism": best_inner - m["auroc"], **{f"proc_{k}": v for k, v in m.items()}})
        return Result(pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True), pd.DataFrame(sel_rows))

    # -- summaries ---------------------------------------------------------------------------
    METRICS = ("auroc", "auprc", "log_loss", "brier", "brier_skill", "cal_intercept", "cal_slope",
               "recal_brier", "recal_cal_slope", "recal_cal_intercept", "train_auroc", "inner_auroc", "optimism",
               "op_sensitivity", "op_specificity", "op_ppv", "op_npv")

    def summarise(self, folds: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for c, g in folds.groupby("candidate", sort=False):
            for m in self.METRICS:
                if m not in g or g[m].isna().all():
                    continue
                v = g[m].dropna().to_numpy()
                t = corrected_resampled_ttest(v, self.n_train, self.n_test)
                rows.append({"candidate": c, "metric": m, "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                             "ci_low": t["ci_low"], "ci_high": t["ci_high"], "n_folds": len(v)})
        return pd.DataFrame(rows)

    def paired(self, folds: pd.DataFrame, metrics=("auroc", "brier", "log_loss")) -> pd.DataFrame:
        names = list(dict.fromkeys(folds.candidate))
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                fa = folds[folds.candidate == a].set_index("split_id")
                fb = folds[folds.candidate == b].set_index("split_id").loc[fa.index]
                for m in metrics:
                    t = corrected_resampled_ttest(fa[m] - fb[m], self.n_train, self.n_test)
                    rows.append({"a": a, "b": b, "metric": m, **t})
        return pd.DataFrame(rows)

    def subgroups(self, preds: pd.DataFrame, groups: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Metrics within treatment subgroups on each patient's out-of-fold prediction averaged over
        repeats; bootstrap 95% CI for AUROC."""
        rows = []
        for c, g in preds.groupby("candidate", sort=False):
            pp = g.groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
            gg = groups.loc[pp.index]
            for var in gg.columns:
                for lev, idx in gg.groupby(gg[var].astype(str)).groups.items():
                    sub = pp.loc[idx]
                    lo, hi = M.bootstrap_auroc_ci(sub.y, sub.p, self.h["n_boot"], rng)
                    one = sub.y.nunique() < 2
                    rows.append({"candidate": c, "subgroup": var, "level": lev, "n": len(sub), "deaths": int(sub.y.sum()),
                                 "auroc": np.nan if one else roc_auc_score(sub.y, sub.p), "auroc_ci_low": lo,
                                 "auroc_ci_high": hi, "brier": float(np.mean((sub.p - sub.y) ** 2)),
                                 "mean_pred": sub.p.mean(), "obs_rate": sub.y.mean()})
        return pd.DataFrame(rows)

    def overfit_gate(self, folds: pd.DataFrame) -> pd.DataFrame:
        gate = self.h["overfit_gate"]
        lo, hi = gate["calibration_slope"]
        rows = []
        for c, g in folds.groupby("candidate", sort=False):
            gap = float((g.train_auroc - g.auroc).mean())
            slope = float(g.cal_slope.mean()) if g.cal_slope.notna().any() else np.nan
            gap_ok = gap <= gate["max_train_cv_auroc_gap"]
            slope_ok = True if np.isnan(slope) else lo <= slope <= hi
            rows.append({"candidate": c, "train_auroc": g.train_auroc.mean(), "cv_auroc": g.auroc.mean(),
                         "train_cv_gap": gap, "gap_limit": gate["max_train_cv_auroc_gap"], "cal_slope": slope,
                         "slope_range": f"{lo}-{hi}", "gap_flag": not gap_ok,
                         "slope_flag": (not slope_ok), "status": "pass" if gap_ok and slope_ok else "flag"})
        return pd.DataFrame(rows)

    def learning_curve(self, candidates: list[Candidate], rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for s in self.splits:
            ytr, yte = self.y.loc[s.train], self.y.loc[s.test]
            for f in self.h["learning_curve_fractions"]:
                if f >= 1:
                    sub = s.train
                else:   # stratified subsample of the outer training rows
                    parts = [rng.choice(ytr.index[ytr == v], int(round(f * (ytr == v).sum())), replace=False) for v in (0, 1)]
                    sub = pd.Index(np.concatenate(parts))
                for c in candidates:
                    est = self._fit(c, sub, self.y)
                    p_te, p_tr = self._proba(est, c, s.test), self._proba(est, c, sub)
                    rows.append({"candidate": c.name, "split_id": s.split_id, "fraction": f, "n_train": len(sub),
                                 "cv_auroc": roc_auc_score(yte, p_te), "train_auroc": roc_auc_score(ytr.loc[sub], p_tr)})
        lc = pd.DataFrame(rows)
        return lc.groupby(["candidate", "fraction"], sort=False).agg(
            n_train=("n_train", "mean"), cv_auroc=("cv_auroc", "mean"), cv_auroc_sd=("cv_auroc", "std"),
            train_auroc=("train_auroc", "mean")).reset_index()

    def permutation_test(self, procedure: Callable[[pd.Series], float], n_perm: int,
                         rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
        """Run the full `procedure` on shuffled labels. It must score at chance (A6)."""
        obs = procedure(self.y)
        perm = []
        for i in range(n_perm):
            yp = pd.Series(rng.permutation(self.y.to_numpy()), index=self.y.index)
            perm.append({"permutation": i, "auroc": procedure(yp)})
        pt = pd.DataFrame(perm)
        band = self.h["chance_band"]
        summ = {"observed_auroc": obs, "perm_mean": pt.auroc.mean(), "perm_sd": pt.auroc.std(ddof=1),
                "perm_max": pt.auroc.max(), "p_value": (1 + (pt.auroc >= obs).sum()) / (n_perm + 1),
                "n_permutations": n_perm, "chance_band": band,
                "at_chance": bool(abs(pt.auroc.mean() - 0.5) <= band)}
        return pt, summ

    def loco(self, candidates: list[Candidate], cohort: pd.Series) -> pd.DataFrame:
        """Leave-one-cohort-out on the training rows (sensitivity analysis only, B8)."""
        rows = []
        coh = cohort.loc[self.X.index].astype(str)
        for k in sorted(coh.unique()):
            te, tr = coh.index[coh == k], coh.index[coh != k]
            for c in candidates:
                p = self._proba(self._fit(c, tr, self.y), c, te)
                rows.append({"candidate": c.name, "held_out_cohort": k, "n_train": len(tr),
                             **M.binary_metrics(self.y.loc[te], p, float(self.y.loc[tr].mean()))})
        return pd.DataFrame(rows)

    def regression_floor(self, folds: pd.DataFrame) -> pd.DataFrame:
        k = self.h["regression_floor_sd"]
        g = folds.groupby("candidate", sort=False).auroc
        out = pd.DataFrame({"mean_outer_auroc": g.mean(), "sd_outer_auroc": g.std(ddof=1), "n_folds": g.size()})
        out["regression_floor"] = out.mean_outer_auroc - k * out.sd_outer_auroc
        out["rule"] = f"mean - {k} x SD of outer-fold AUROC (nested CV)"
        return out.reset_index()


def reliability(preds: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows = []
    for c, g in preds.groupby("candidate", sort=False):
        pp = g.groupby("patient_id").agg(y=("y", "first"), p=("p", "mean"))
        rows.append(M.reliability_table(pp.y, pp.p, bins).assign(candidate=c))
    return pd.concat(rows, ignore_index=True)
