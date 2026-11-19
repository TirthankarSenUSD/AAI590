"""Deployment fit of the final model on all training rows (build step 7).

The final model is a *procedure* (ensemble stage, F-060/F-063): on one arm, every member picks its
own configuration by the inner-fold rule; members whose predictions correlate above
`ensemble.merge_spearman` are merged (logit average weighted by AUROC - 0.5); the `best_single`
option then keeps the best merged column (simplest within 1 SE). Here exactly that procedure is run
once on all training rows, with `report.deploy_folds` stratified folds in the role of the inner
folds. Nothing here reads the holdout.

Primary probabilities follow the overfit gate (F-042): if the calibration slope of the deployment
out-of-fold predictions is above `harness.overfit_gate.recalibrate_above_slope`, a Platt map fitted on
those predictions is applied; otherwise raw probabilities are primary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..ensemble import stack as S
from ..farm.members import CAUSE, FAMILY, build_candidates
from ..harness import metrics as M
from ..harness.core import select_candidate


def deployment_folds(y: pd.Series, k: int, seed: int) -> list[tuple[pd.Index, pd.Index]]:
    sk = StratifiedKFold(k, shuffle=True, random_state=seed)
    return [(y.index[a], y.index[b]) for a, b in sk.split(np.zeros(len(y)), y.to_numpy())]


def _oof_task(name, cand, X, yfit, ybin, folds):
    """Out-of-fold predictions of one candidate over the deployment folds, plus per-fold metrics."""
    oof = pd.Series(np.nan, index=X.index)
    aucs, lls, brs = [], [], []
    for tr, va in folds:
        m = cand.make().fit(X.loc[tr, cand.columns], yfit.loc[tr].to_numpy())
        p = m.predict_proba(X.loc[va, cand.columns])[:, 1]
        oof.loc[va] = p
        yv = ybin.loc[va].to_numpy()
        aucs.append(roc_auc_score(yv, p))
        lls.append(log_loss(yv, M.clip(p), labels=[0, 1]))
        brs.append(brier_score_loss(yv, p))
    return {"key": name, "oof": oof, "auroc": float(np.mean(aucs)),
            "auroc_se": float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))), "log_loss": float(np.mean(lls)),
            "brier": float(np.mean(brs))}


def _fit_task(key, cand, X, yfit):
    return {"key": key, "model": cand.make().fit(X[cand.columns], yfit.to_numpy())}


@dataclass
class FinalModel:
    """The fitted final model. `predict` gives primary P(death); `log_odds` its logit."""
    arm: str
    option: str
    group: list[str]                       # members in the chosen (merged) column
    weights: dict[str, float]              # member -> logit-average weight
    winners: dict[str, str]                # member -> chosen configuration
    models: dict[str, object]              # member -> pipeline fitted on all training rows
    columns: list[str]                     # raw input columns the model reads
    primary: str                           # "raw" or "recalibrated"
    recal: tuple[float, float] | None      # Platt (a, b): p = sigmoid(a + b * logit(p_raw))
    cause_member: str | None = None        # cause-specific member used for the cause breakdown
    cause_model: object | None = None
    boot: list[dict[str, object]] = field(default_factory=list)   # bootstrap refits of the group members
    info: dict = field(default_factory=dict)

    def _raw_logit(self, X: pd.DataFrame, models: dict | None = None) -> np.ndarray:
        models = models or self.models
        return sum(w * S.logit(models[m].predict_proba(X[self.columns])[:, 1]) for m, w in self.weights.items())

    def log_odds(self, X: pd.DataFrame, models: dict | None = None) -> np.ndarray:
        z = self._raw_logit(X, models)
        if self.recal is not None:
            z = self.recal[0] + self.recal[1] * z
        return z

    def predict(self, X: pd.DataFrame, models: dict | None = None) -> np.ndarray:
        return S.sigmoid(self.log_odds(X, models))

    def predict_interval(self, X: pd.DataFrame, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        if not self.boot:
            n = len(X)
            return np.full(n, np.nan), np.full(n, np.nan)
        B = np.vstack([self.predict(X, b) for b in self.boot])
        a = (1 - level) / 2
        return np.quantile(B, a, axis=0), np.quantile(B, 1 - a, axis=0)

    def predict_cause(self, X: pd.DataFrame) -> pd.DataFrame | None:
        if self.cause_model is None:
            return None
        p = self.cause_model.predict_cause_proba(X[self.columns])
        return pd.DataFrame(p, index=X.index, columns=["p_alive", "p_cancer_death", "p_other_death"])


def fit_final_model(cfg, X: pd.DataFrame, y: pd.Series, y_cause: pd.Series | None, arm: str, option: str,
                    parallel, log) -> tuple[FinalModel, dict[str, pd.DataFrame]]:
    """Run the final procedure once on all training rows. `parallel(fn, tasks, label)` runs tasks.
    Returns the model and the tables that document every choice."""
    if option != "best_single":
        raise NotImplementedError(f"deployment of the {option!r} option is not built yet (final option so far: best_single)")
    rc, ec, h = cfg["report"], cfg["ensemble"], cfg["harness"]
    members = [m for m in ec["members"] if m in FAMILY]
    cands, meta = build_candidates(cfg, members=members, arms=[arm])
    if not cands:
        raise ValueError(f"no candidates for arm {arm}")
    by = {c.name: c for c in cands}
    ybin = y.astype(int)
    folds = deployment_folds(ybin, rc["deploy_folds"], cfg.seed)
    fold_of = pd.Series(-1, index=y.index)
    for j, (_, va) in enumerate(folds):
        fold_of.loc[va] = j

    def yfit(c):
        if c.target == "cause":
            if y_cause is None:
                raise ValueError("cause-specific members need the cause label")
            return y_cause
        return ybin

    res = {r["key"]: r for r in parallel(_oof_task, [[c.name, c, X, yfit(c), ybin, folds] for c in cands],
                                          "deployment: candidate out-of-fold predictions")}
    stats = pd.DataFrame([{"candidate": n, "member": meta[n].member, "inner_auroc": r["auroc"], "inner_auroc_se": r["auroc_se"],
                           "inner_log_loss": r["log_loss"], "inner_brier": r["brier"], "complexity": by[n].complexity,
                           "eligible": True} for n, r in res.items()])
    # 1. each member's configuration by the inner rule
    winners = {m: select_candidate(stats[stats.member == m], h["select_within_se"])
               for m in members if (stats.member == m).any()}
    P = pd.DataFrame({m: res[w]["oof"] for m, w in winners.items()})
    # 2. agreement merge and 3. best_single among the merged columns
    aucs = {m: S._auc(ybin, P[m]) for m in P.columns}
    mw = S.merge_weights(S.merge_groups(P, ec["merge_spearman"]), aucs)
    Mx = S.apply_merge(P, mw)
    cx = {k: min(FAMILY[m] for m in k.split("+")) for k in Mx.columns}
    chosen = S.choose_member(ybin, Mx, fold_of, cx, h["select_within_se"])
    weights = mw[chosen]
    group = list(weights)
    # cause breakdown model: the configured cause-specific member (its own winner)
    cause_member = rc.get("cause_member")
    if cause_member not in winners:
        cause_member = None
    need = sorted(set(group) | ({cause_member} if cause_member else set()))
    fits = {r["key"]: r["model"] for r in parallel(_fit_task, [[m, by[winners[m]], X, yfit(by[winners[m]])] for m in need],
                                                    "deployment: final fits on all training rows")}
    # primary probabilities by the overfit gate, on the deployment out-of-fold predictions
    oof_raw = Mx[chosen].to_numpy()
    slope = M.calibration_slope(ybin.to_numpy(), oof_raw)
    recal = None
    if slope > h["overfit_gate"]["recalibrate_above_slope"]:
        lr = LogisticRegression(C=M.BIG_C, max_iter=1000).fit(S.logit(oof_raw).reshape(-1, 1), ybin.to_numpy())
        recal = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
    cols = by[winners[group[0]]].columns
    fm = FinalModel(arm=arm, option=option, group=group, weights={m: float(w) for m, w in weights.items()},
                    winners={m: winners[m] for m in group}, models={m: fits[m] for m in group}, columns=list(cols),
                    primary="recalibrated" if recal else "raw", recal=recal, cause_member=cause_member,
                    cause_model=fits.get(cause_member) if cause_member else None)
    oof_primary = S.sigmoid(S.logit(oof_raw) if recal is None else recal[0] + recal[1] * S.logit(oof_raw))
    fm.info = {"deploy_folds": rc["deploy_folds"], "oof_auroc": float(roc_auc_score(ybin, oof_raw)),
               "oof_cal_slope_raw": float(slope), "n_train": int(len(y)), "n_deaths": int(ybin.sum()),
               "n_candidates": len(cands), "merged_columns": list(Mx.columns), "chosen_column": chosen}
    # bootstrap refits of the group members (configuration, merge weights and Platt map held fixed)
    nb = int(rc["n_boot"])
    if nb:
        rng = np.random.default_rng(cfg.seed)
        tasks = []
        for b in range(nb):
            idx = y.index[rng.integers(0, len(y), len(y))]
            for m in group:
                c = by[winners[m]]
                Xb = X.loc[idx].reset_index(drop=True)
                yb = yfit(c).loc[idx].reset_index(drop=True)
                tasks.append([(b, m), c, Xb, yb])
        boot = {}
        for r in parallel(_fit_task, tasks, f"deployment: {nb} bootstrap refits"):
            boot.setdefault(r["key"][0], {})[r["key"][1]] = r["model"]
        fm.boot = [boot[b] for b in sorted(boot)]
    tables = {
        "deploy_candidates": stats.sort_values(["member", "inner_auroc"], ascending=[True, False]),
        "deploy_member_winners": pd.DataFrame([{"member": m, "winner": w, "oof_auroc": aucs[m]} for m, w in winners.items()]),
        "deploy_merge": pd.DataFrame([{"column": k, "member": m, "weight": w, "chosen": k == chosen}
                                      for k, ws in mw.items() for m, w in ws.items()]),
        "deploy_oof_predictions": pd.DataFrame({"patient_id": y.index, "y": ybin.to_numpy(), "fold": fold_of.to_numpy(),
                                                "p_raw": oof_raw, "p_primary": oof_primary}),
    }
    log(f"deployment: chosen column {chosen} (weights {fm.weights}); primary probabilities {fm.primary} "
        f"(out-of-fold slope {slope:.2f}); out-of-fold AUROC {fm.info['oof_auroc']:.3f}")
    return fm, tables
