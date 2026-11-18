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
from ..core.stats import corrected_resampled_ttest


@dataclass
class Candidate:
    name: str
    make: Callable[[], Any]           # returns an unfitted estimator with fit / predict_proba
    columns: list[str]
    complexity: float = 0.0           # lower = simpler (1-SE rule)
    eligible: bool = True             # floors are never eligible to win
    label: str = ""
    target: str = "binary"            # "cause": trained on alive / cancer death / other-cause death (scored as death)


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


def _fit(cand: Candidate, X: pd.DataFrame, rows: pd.Index, y: pd.Series):
    est = cand.make()
    est.fit(X.loc[rows, cand.columns], y.loc[rows].to_numpy())
    return est


def _proba(est, cand: Candidate, X: pd.DataFrame, rows: pd.Index) -> np.ndarray:
    return est.predict_proba(X.loc[rows, cand.columns])[:, 1]


def _labels(c: Candidate, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(labels the candidate is fitted on, binary death labels used for every metric). `y` is either
    binary death or the 3-class cause label (0 alive, 1 cancer death, 2 other-cause death); the cause
    label is used only to fit "cause" candidates, never as an input and never for scoring."""
    y = y.astype(int)
    ybin = (y > 0).astype(int)
    if c.target == "cause":
        if y.max() < 2:
            raise ValueError(f"{c.name} needs the 3-class cause label")
        return y, ybin
    return ybin, ybin


def _score_candidate(c: Candidate, X: pd.DataFrame, y_in: pd.Series, splits: list[Split], h: dict,
                     light: bool, need_inner: bool) -> tuple[list[dict], list[pd.DataFrame]]:
    """One candidate on every split (the unit of parallel work). Everything is fitted on the
    split's training rows; the threshold and recalibrator only on inner out-of-fold predictions."""
    yfit, y = _labels(c, y_in)
    recs, preds = [], []
    for s in splits:
        ytr, yte = y.loc[s.train], y.loc[s.test]
        prev = float(ytr.mean())
        rec = {"candidate": c.name, "split_id": s.split_id, "repeat": s.repeat, "outer_fold": s.outer,
               "complexity": c.complexity, "eligible": c.eligible}
        oof = None
        if need_inner:
            oof = pd.Series(np.nan, index=s.train)
            aucs, lls, brs = [], [], []
            for tr, va in s.inner:
                p_va = _proba(_fit(c, X, tr, yfit), c, X, va)
                oof.loc[va] = p_va
                yv = y.loc[va].to_numpy()
                aucs.append(roc_auc_score(yv, p_va) if yv.min() != yv.max() else np.nan)
                pc = M.clip(p_va)
                lls.append(-np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)))
                brs.append(np.mean((p_va - yv) ** 2))
            rec.update(inner_auroc=float(np.nanmean(aucs)),
                       inner_auroc_se=float(np.nanstd(aucs, ddof=1) / np.sqrt(len(aucs))),
                       inner_log_loss=float(np.mean(lls)), inner_brier=float(np.mean(brs)))
        est = _fit(c, X, s.train, yfit)
        p_te, p_tr = _proba(est, c, X, s.test), _proba(est, c, X, s.train)
        m = M.binary_metrics(yte, p_te, prev)
        rec.update(m)
        rec["train_auroc"] = float(roc_auc_score(ytr, p_tr)) if ytr.nunique() > 1 else np.nan
        rec["optimism"] = rec.get("inner_auroc", np.nan) - m["auroc"]
        p_rc = np.full(len(p_te), np.nan)
        if not light:
            t = M.threshold_for_sensitivity(ytr.loc[oof.index], oof.to_numpy(), h["operating_point_sensitivity"])
            rec.update({f"op_{k}": v for k, v in M.threshold_metrics(yte, p_te, t).items()})
            cal = LogisticRegression(C=M.BIG_C, max_iter=1000).fit(M.logit(oof.to_numpy()).reshape(-1, 1), ytr.loc[oof.index])
            p_rc = cal.predict_proba(M.logit(p_te).reshape(-1, 1))[:, 1]
            rec.update({f"recal_{k}": v for k, v in M.binary_metrics(yte, p_rc, prev).items()
                        if k in ("auroc", "log_loss", "brier", "brier_skill", "cal_intercept", "cal_slope")})
        recs.append(rec)
        preds.append(pd.DataFrame({"candidate": c.name, "split_id": s.split_id, "repeat": s.repeat,
                                   "outer_fold": s.outer, "patient_id": s.test, "y": yte.to_numpy(),
                                   "p": p_te, "p_recal": p_rc}))
    return recs, preds


def _score_group(k: int, cands: list[Candidate], Xg: pd.DataFrame, y: pd.Series, kw: dict):
    """Several candidates in one worker, in order (the unit of parallel work). `Xg` holds only the
    columns these candidates use, so each task ships no more data than it needs."""
    return k, [_score_candidate(c, Xg[c.columns], y, **kw) for c in cands]


class _Progress:
    """Progress lines for long evaluations: every 5% of tasks or every 2 minutes, with an ETA."""

    def __init__(self, n: int, log, label: str):
        import threading
        import time
        self.n, self.log, self.label, self.done = n, log, label, 0
        self.t0 = self.last = self.last_tick = time.time()
        self.step = max(1, n // 20)
        self._stop = threading.Event()
        if log:                                   # heartbeat: shows long stretches without a finished task
            threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self):
        import time
        while not self._stop.wait(300):
            if self.done >= self.n:
                return
            if time.time() - self.last_tick >= 300:
                self.log(f"{self.label}: still running, {self.done}/{self.n} tasks done, "
                         f"elapsed {(time.time() - self.t0) / 60:.1f} min (no task finished in the last 5 min)")

    def tick(self):
        import time
        self.done += 1
        now = self.last_tick = time.time()
        if self.log and (self.done % self.step == 0 or now - self.last >= 120 or self.done == self.n):
            el = now - self.t0
            eta = el / self.done * (self.n - self.done)
            self.log(f"{self.label}: {self.done}/{self.n} tasks ({100 * self.done / self.n:.0f}%), "
                     f"elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min (rough: tasks differ in cost)")
            self.last = now
        if self.done >= self.n:
            self._stop.set()


_ID_COLS = ("candidate", "split_id", "repeat", "outer_fold", "complexity", "eligible")


def procedure(folds: pd.DataFrame, names: list[str], within_se: float) -> pd.DataFrame:
    """The select-the-best procedure over `names`: in each outer split the fixed rule picks a winner
    from the inner statistics only; the procedure's outer metrics are the winner's outer metrics."""
    f = folds[folds.candidate.isin(names)]
    rows = []
    for sid, g in f.groupby("split_id", sort=False):
        win = select_candidate(g, within_se)
        w = g[g.candidate == win].iloc[0]
        rows.append({"split_id": sid, "repeat": int(w["repeat"]), "outer_fold": int(w["outer_fold"]), "winner": win,
                     "winner_inner_auroc": w["inner_auroc"], "outer_auroc": w["auroc"],
                     "optimism": w["inner_auroc"] - w["auroc"],
                     **{f"proc_{k}": w[k] for k in g.columns if k not in _ID_COLS}})
    return pd.DataFrame(rows)


def procedure_preds(preds: pd.DataFrame, sel: pd.DataFrame, name: str) -> pd.DataFrame:
    """Out-of-fold predictions of a procedure: in each split, the winner's outer predictions."""
    win = sel.set_index("split_id").winner
    p = preds[preds.candidate == preds.split_id.map(win)].copy()
    p["candidate"] = name
    return p.reset_index(drop=True)


@dataclass
class Result:
    folds: pd.DataFrame
    preds: pd.DataFrame
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)


class Harness:
    def __init__(self, cfg, X: pd.DataFrame, y: pd.Series, splits: list[Split], y_cause: pd.Series | None = None):
        """`y`: binary death. `y_cause` (optional): 0 alive / 1 cancer death / 2 other-cause death, used
        only to fit cause-specific candidates; it must agree with `y` (death = cause > 0)."""
        self.cfg, self.X, self.y, self.splits = cfg, X, y.astype(int), splits
        self.y_cause = None
        if y_cause is not None:
            yc = y_cause.loc[y.index].astype(int)
            if not ((yc > 0).astype(int) == self.y).all():
                raise ValueError("cause label disagrees with the death label")
            self.y_cause = yc
        self.h = cfg["harness"]
        check_split_hygiene(splits, X.index)
        self.n_train = float(np.mean([len(s.train) for s in splits]))
        self.n_test = float(np.mean([len(s.test) for s in splits]))

    # -- core ------------------------------------------------------------------------------
    def _fit(self, cand: Candidate, rows: pd.Index, y: pd.Series):
        return _fit(cand, self.X, rows, y)

    def _proba(self, est, cand: Candidate, rows: pd.Index) -> np.ndarray:
        return _proba(est, cand, self.X, rows)

    def evaluate(self, candidates: list[Candidate], y: pd.Series | None = None, select: bool = False,
                 light: bool = False, n_jobs: int = 1, groups: list[list[int]] | None = None,
                 log=None, label: str = "scoring") -> Result:
        """Score every candidate on every frozen split. `light` skips the inner layer (used by the
        permutation test when no selection is needed). `select` also scores the select-the-best
        procedure, choosing inside each outer training fold by the fixed rule.

        `n_jobs > 1` scores candidates in parallel worker processes, one thread each (no
        oversubscription). Every fit is seeded and results are assembled in candidate order, so
        the output is identical for any n_jobs."""
        # labels: the cause label when available (cause candidates fit on it; everything is scored on death)
        y = (self.y_cause if self.y_cause is not None else self.y) if y is None else y.astype(int)
        kw = dict(splits=self.splits, h=self.h, light=light, need_inner=(not light) or select)
        # `groups`: candidate indices that must run in the same worker (e.g. sharing a cached,
        # unsupervised representation). Results are always assembled in candidate order.
        groups = groups or [[i] for i in range(len(candidates))]
        assert sorted(i for g in groups for i in g) == list(range(len(candidates))), "groups must partition candidates"
        def task(k, g):
            cands = [candidates[i] for i in g]
            cols = list(dict.fromkeys(c for cand in cands for c in cand.columns))
            return k, cands, self.X[cols]

        prog = _Progress(len(groups), log, label)
        gouts = {}
        if n_jobs == 1 or len(groups) == 1:
            for k, g in enumerate(groups):
                _, out = _score_group(*task(k, g), y, kw)
                gouts[k] = out
                prog.tick()
        else:
            from joblib import Parallel, delayed, parallel_config
            with parallel_config(backend="loky", inner_max_num_threads=1):
                it = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                    delayed(_score_group)(*task(k, g), y, kw) for k, g in enumerate(groups))
                for k, out in it:                       # completion order; reassembled below
                    gouts[k] = out
                    prog.tick()
        by_idx = {i: o for k, g in enumerate(groups) for i, o in zip(g, gouts[k])}
        outs = [by_idx[i] for i in range(len(candidates))]
        folds = pd.DataFrame([r for recs, _ in outs for r in recs])
        preds = pd.concat([p for _, ps in outs for p in ps], ignore_index=True)
        sel = procedure(folds, [c.name for c in candidates], self.h["select_within_se"]) if select else pd.DataFrame()
        return Result(folds, preds, sel)

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
        """Two separate questions (F-042):
        - overfitting: train-versus-CV AUROC gap above the limit, or calibration slope below the
          minimum (over-confident predictions) -> status 'overfit';
        - under-confidence: slope above `recalibrate_above_slope` -> status 'recalibrate', and the
          in-fold recalibrated probabilities become the primary ones.
        Flags are reported only; they never exclude a configuration from selection."""
        gate = self.h["overfit_gate"]
        rows = []
        for c, g in folds.groupby("candidate", sort=False):
            gap = float((g.train_auroc - g.auroc).mean())
            slope = float(g.cal_slope.mean()) if g.cal_slope.notna().any() else np.nan
            recal_slope = float(g.recal_cal_slope.mean()) if "recal_cal_slope" in g and g.recal_cal_slope.notna().any() else np.nan
            gap_flag = gap > gate["max_train_cv_auroc_gap"]
            over_conf = (not np.isnan(slope)) and slope < gate["min_calibration_slope"]
            under_conf = (not np.isnan(slope)) and slope > gate["recalibrate_above_slope"]
            status = "overfit" if (gap_flag or over_conf) else ("recalibrate" if under_conf else "pass")
            rows.append({"candidate": c, "train_auroc": g.train_auroc.mean(), "cv_auroc": g.auroc.mean(),
                         "train_cv_gap": gap, "gap_limit": gate["max_train_cv_auroc_gap"], "gap_flag": gap_flag,
                         "cal_slope": slope, "overconfident_flag": over_conf, "underconfident_flag": under_conf,
                         "recal_cal_slope": recal_slope, "status": status,
                         "primary_probabilities": "recalibrated" if under_conf else "raw",
                         "action": gate.get("action", "report")})
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
        labels = self.y_cause if self.y_cause is not None else self.y   # permute the full label (cause included)
        obs = procedure(labels)
        perm = []
        for i in range(n_perm):
            yp = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
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
