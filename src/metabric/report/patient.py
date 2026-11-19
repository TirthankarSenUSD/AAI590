"""Patient-level output: one structured record per patient (a function, no user interface).

`PatientExplainer.records(X)` returns, for each row of raw inputs:
- the inputs echoed back, with how each missing value is handled;
- the primary probability of death with a bootstrap interval, its percentile among the training
  patients' out-of-fold predictions and the risk group (cutpoints from those predictions), with the
  group's observed death rate;
- the cause breakdown (cause-specific model): P(cancer death) and P(other-cause death);
- contributing factors: Shapley values on the model's log-odds, over the raw input columns, grouped
  by clinical / RNA / DNA and worded "associated with". Contributions plus the base value equal the
  model's log-odds exactly (each sampled permutation telescopes);
- cohort context: the PAM50 subtype group with its observed death rate (Wilson interval) and a
  small-group flag, percentiles of the numeric inputs, prevalence of any mutations supplied;
- the model's cross-validated quality (from the nested runs);
- flags: inputs outside the training range and category levels never seen in training.

It never gives a diagnosis, treatment advice, a causal claim, individual certainty or a time horizon.
Every random draw is seeded per patient, so one patient alone gives the same record as in a batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.stats import wilson
from .deploy import FinalModel

DISCLAIMER = ("This is a statistical estimate of the risk of death (any cause, no time horizon) for patients like this one in "
              "the METABRIC cohort, given the treatment they received. It is not a diagnosis, not treatment advice and "
              "not a prediction of what will happen to this person. Factors are associations in the model, not causes.")


def shapley_log_odds(f, x: pd.DataFrame, background: pd.DataFrame, cols: list[str], n_perm: int,
                     seed: int) -> tuple[pd.Series, float, float]:
    """Interventional Shapley values of f (a function of a DataFrame returning log-odds) for one row
    `x`, by antithetic permutation sampling against `background`. Returns (phi, base, value) with
    base + phi.sum() == value up to rounding."""
    rng = np.random.default_rng(seed)
    nb, m = len(background), len(cols)
    phi = pd.Series(0.0, index=cols)
    perms = []
    for _ in range(max(1, n_perm // 2)):
        p = rng.permutation(m)
        perms += [p, p[::-1]]
    xv = x.iloc[0]
    base = value = None
    for p in perms:
        states = [background.copy()]
        Z = background.copy()
        for j in p:
            Z = Z.copy()
            Z[cols[j]] = xv[cols[j]]
            states.append(Z)
        v = f(pd.concat(states, ignore_index=True)).reshape(m + 1, nb).mean(1)
        base, value = float(v[0]), float(v[-1])
        for k, j in enumerate(p):
            phi.iloc[j] += v[k + 1] - v[k]
    return phi / len(perms), base, value


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


@dataclass
class PatientExplainer:
    model: FinalModel
    blocks: dict[str, list[str]]           # clinical / rna / dna column lists
    numeric: list[str]                     # numeric clinical columns (percentiles, range flags)
    categorical: list[str]                 # categorical clinical columns (unseen-level flags)
    train_X: pd.DataFrame                  # training rows, the model's raw input columns
    train_oof: np.ndarray                  # training out-of-fold primary probabilities
    train_y: np.ndarray                    # training outcomes (death = 1)
    context_col: str                       # cohort-context group column (PAM50 subtype)
    context_values: pd.Series              # that column on the training rows
    mutation_prevalence: pd.Series         # share of training patients with each mutation
    quality: dict                          # cross-validated model quality (from the nested runs)
    risk_quantiles: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    risk_labels: list[str] = field(default_factory=lambda: ["low", "below average", "above average", "high"])
    n_background: int = 100
    n_perm: int = 20
    small_group: int = 30
    top_k: int = 5
    seed: int = 42

    def __post_init__(self):
        self.cut = np.quantile(self.train_oof, self.risk_quantiles)
        rng = np.random.default_rng(self.seed)
        n = min(self.n_background, len(self.train_X))
        self.background = self.train_X.iloc[np.sort(rng.choice(len(self.train_X), n, replace=False))].reset_index(drop=True)
        g = np.digitize(self.train_oof, self.cut)
        self.group_rates = {}
        for k, lab in enumerate(self.risk_labels):
            yy = self.train_y[g == k]
            lo, hi = wilson(int(yy.sum()), len(yy))
            self.group_rates[lab] = {"n": int(len(yy)), "deaths": int(yy.sum()), "observed_death_rate": float(yy.mean()),
                                     "ci_low": lo, "ci_high": hi}
        cv = self.context_values.astype(object).where(self.context_values.notna(), "missing")
        self.context_table = {}
        for lev in cv.unique():
            yy = self.train_y[(cv == lev).to_numpy()]
            lo, hi = wilson(int(yy.sum()), len(yy))
            self.context_table[lev] = {"n": int(len(yy)), "deaths": int(yy.sum()), "observed_death_rate": float(yy.mean()),
                                       "ci_low": lo, "ci_high": hi, "small_group": bool(len(yy) < self.small_group)}

    def _block(self, col: str) -> str:
        for b in ("rna", "dna"):
            if col in self.blocks.get(b, []):
                return b
        return "clinical"

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.model.columns:
            if c not in X.columns:
                X[c] = np.nan
        return X[self.model.columns].astype(self.train_X.dtypes.to_dict(), errors="ignore")

    def records(self, X: pd.DataFrame, ids=None) -> list[dict]:
        ids = list(X.index if ids is None else ids)
        Xp = self._prepare(X)
        p = self.model.predict(Xp)
        lo, hi = self.model.predict_interval(Xp)
        cause = self.model.predict_cause(Xp)
        return [self._one(ids[i], X.iloc[[i]], Xp.iloc[[i]], float(p[i]), float(lo[i]), float(hi[i]),
                          None if cause is None else cause.iloc[i]) for i in range(len(Xp))]

    def record(self, row: dict | pd.Series, patient_id=None) -> dict:
        X = pd.DataFrame([dict(row)])
        return self.records(X, [patient_id])[0]

    def _one(self, pid, raw: pd.DataFrame, x: pd.DataFrame, p: float, lo: float, hi: float, cause) -> dict:
        m = self.model
        # inputs, missing-value handling and range flags
        inputs, flags = {}, []
        for c in m.columns:
            v = x.iloc[0][c]
            supplied = c in raw.columns
            if pd.isna(v):
                how = ("filled with the training median, with a 'was missing' indicator" if c in self.numeric
                       else "treated as its own 'missing' category" if c in self.categorical else "handled as in training")
                inputs[c] = {"value": None, "status": "missing" if supplied else "not supplied", "handling": how}
                continue
            inputs[c] = {"value": v.item() if hasattr(v, "item") else v, "status": "provided"}
            tr = self.train_X[c]
            if c in self.numeric:
                if v < tr.min() or v > tr.max():
                    flags.append(f"{c} = {_fmt(v)} is outside the training range ({_fmt(tr.min())} to {_fmt(tr.max())})")
            elif c in self.categorical and str(v) not in set(tr.dropna().astype(str)):
                flags.append(f"{c} = {v} was never seen in training; the model treats it as an unknown level")
        # contributions on the log-odds scale
        # the same seed for every patient, so a patient's record does not depend on the batch it is in
        phi, base, value = shapley_log_odds(lambda D: m.log_odds(D), x.reset_index(drop=True), self.background,
                                            m.columns, self.n_perm, self.seed)
        by_block = {}
        for b in ("clinical", "rna", "dna"):
            cols = [c for c in m.columns if self._block(c) == b]
            if not cols:
                by_block[b] = {"used_by_model": False, "total": 0.0, "factors": []}
                continue
            s = phi[cols]
            top = s.reindex(s.abs().sort_values(ascending=False).index)[: self.top_k]
            by_block[b] = {"used_by_model": True, "total": float(s.sum()), "factors": [
                {"feature": c, "value": inputs[c].get("value"), "contribution_log_odds": float(v),
                 "wording": f"{c} = {_fmt(inputs[c].get('value')) if inputs[c].get('value') is not None else 'missing'} is "
                            f"associated with {'higher' if v > 0 else 'lower'} estimated risk in this model"}
                for c, v in top.items() if abs(v) > 1e-9]}
        # risk placement
        pct = float((self.train_oof < p).mean() * 100)
        grp = self.risk_labels[int(np.digitize(p, self.cut))]
        # cohort context
        cv = x.iloc[0].get(self.context_col) if self.context_col in x.columns else None
        lev = "missing" if cv is None or pd.isna(cv) else cv
        ctx = {"group_column": self.context_col, "group": lev,
               **self.context_table.get(lev, {"n": 0, "small_group": True, "note": "level not seen in training"})}
        perc = {c: float((self.train_X[c].dropna() < x.iloc[0][c]).mean() * 100) for c in self.numeric
                if c in m.columns and not pd.isna(x.iloc[0][c])}
        muts = [c for c in raw.columns if c.endswith("_mut") and str(raw.iloc[0][c]) not in ("0", "0.0", "nan", "False")]
        rec = {
            "patient_id": pid if pid is None or isinstance(pid, str) else (pid.item() if hasattr(pid, "item") else pid),
            "inputs": inputs,
            "risk": {"probability_of_death": p, "interval_95": [lo, hi], "interval_method":
                     f"bootstrap refits of the final model ({len(m.boot)} resamples; configuration fixed)",
                     "probabilities": m.primary, "percentile_among_training_patients": pct, "risk_group": grp,
                     "risk_group_observed": self.group_rates[grp]},
            "cause_breakdown": None if cause is None else {
                "p_cancer_death": float(cause["p_cancer_death"]), "p_other_cause_death": float(cause["p_other_death"]),
                "model": m.cause_member,
                "note": "from a separate cause-specific model; its total need not equal the probability above"},
            "contributions": {"scale": "log-odds of death", "base_value": base, "model_output": value,
                              "sum_check": base + float(phi.sum()), "by_block": by_block,
                              "method": f"interventional Shapley values, {len(phi)} inputs, {self.n_perm} permutations "
                                        f"(antithetic) against {len(self.background)} training patients"},
            "cohort_context": {"subtype": ctx, "input_percentiles": perc,
                               "mutations_supplied": {c: float(self.mutation_prevalence.get(c, np.nan)) for c in muts}},
            "model_quality": self.quality,
            "flags": flags,
            "disclaimer": DISCLAIMER,
        }
        return rec
