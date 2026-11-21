"""Phase 2 survival members. Every member has `fit(X, time, event)` and `survival(X, times)`
(the predicted probability of surviving past each time, one row per patient). All preprocessing is
fitted on the training rows passed to `fit` (A12).

- `RidgeCox`: Cox proportional hazards with a ridge penalty, Breslow ties and optional strata
  (scikit-survival has no stratified Cox). Baseline hazard per stratum (Breslow).
- members: `cox` (clinical; optionally stratified), `coxnet` (elastic-net Cox; RNA penalised more
  strongly than clinical), `rsf` (random survival forest), and the floors `cox_age`, `cox_npi`, `km`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..core.stats import kaplan_meier
from ..farm.preprocess import column_groups, preprocessor


def _step(times: np.ndarray, values: np.ndarray, at, start: float = 0.0) -> np.ndarray:
    """Right-continuous step function (value `start` before the first time) evaluated at `at`."""
    i = np.searchsorted(times, np.asarray(at, float), side="right") - 1
    return np.where(i < 0, start, values[np.clip(i, 0, None)])


class RidgeCox:
    """Cox PH, Breslow ties, penalty 0.5 * alpha * ||b||^2 on the mean negative log partial likelihood."""

    def __init__(self, alpha: float = 0.01, max_iter: int = 500):
        self.alpha, self.max_iter = alpha, max_iter

    @staticmethod
    def _groups(time, strata):
        """Per stratum: order (time descending) and, per position, the last index of its tie group."""
        out = []
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            o = idx[np.argsort(-time[idx], kind="mergesort")]
            ts = time[o]
            last = np.searchsorted(-ts, -ts, side="right") - 1
            out.append((o, last))
        return out

    def _obj(self, b, X, e, groups, n):
        eta = X @ b
        ll, grad = 0.0, np.zeros_like(b)
        for o, last in groups:
            et, Xo, eo = eta[o], X[o], e[o]
            c = et.max()
            w = np.exp(et - c)
            s0 = np.cumsum(w)[last]
            s1 = np.cumsum(w[:, None] * Xo, axis=0)[last]
            ev = eo == 1
            ll += float((et[ev] - c - np.log(s0[ev])).sum())
            grad += (Xo[ev] - s1[ev] / s0[ev, None]).sum(axis=0)
        f = -ll / n + 0.5 * self.alpha * float(b @ b)
        g = -grad / n + self.alpha * b
        return f, g

    def fit(self, X, time, event, strata=None):
        X = np.asarray(X, float)
        t, e = np.asarray(time, float), np.asarray(event, int)
        s = np.zeros(len(t), dtype=object) if strata is None else np.asarray(strata, dtype=object)
        groups = self._groups(t, s)
        r = minimize(self._obj, np.zeros(X.shape[1]), args=(X, e, groups, len(t)), jac=True, method="L-BFGS-B",
                     options={"maxiter": self.max_iter})
        self.coef_ = r.x
        self.converged_ = bool(r.success)
        eta = X @ self.coef_
        self.baseline_ = {}
        for lev in list(np.unique(s)) + ["__pooled__"]:
            m = np.ones(len(t), bool) if lev == "__pooled__" else (s == lev)
            tt, ee, rr = t[m], e[m], np.exp(eta[m])
            ut = np.unique(tt[ee == 1])
            inc = np.array([((tt == u) & (ee == 1)).sum() / rr[tt >= u].sum() for u in ut])
            self.baseline_[lev] = (ut, np.cumsum(inc))
        return self

    def survival(self, X, times, strata=None) -> np.ndarray:
        eta = np.asarray(X, float) @ self.coef_
        s = np.zeros(len(eta), dtype=object) if strata is None else np.asarray(strata, dtype=object)
        out = np.empty((len(eta), len(times)))
        for i, (lp, lev) in enumerate(zip(eta, s)):
            ut, H = self.baseline_.get(lev, self.baseline_["__pooled__"])
            out[i] = np.exp(-_step(ut, H, times) * np.exp(lp))
        return out


# ---------------------------------------------------------------- members
class _Base:
    def __init__(self, cfg, params: dict, seed: int):
        self.cfg, self.params, self.seed = cfg, dict(params), seed


class CoxMember(_Base):
    """Ridge Cox on an arm's clinical columns; `stratify` = column to stratify on (or None)."""

    def _cols(self):
        g = column_groups(self.cfg, self.params.get("arm", "C"))
        st = self.params.get("stratify")
        if st:
            g = {k: [c for c in v if c != st] for k, v in g.items()}
        return g

    def fit(self, X, time, event):
        g = self._cols()
        self.prep_ = preprocessor(g, "linear", None, self.cfg, self.seed).fit(X)
        self.m_ = RidgeCox(self.params["alpha"]).fit(self.prep_.transform(X), time, event, self._strata(X))
        return self

    def _strata(self, X):
        st = self.params.get("stratify")
        return None if not st else X[st].astype(object).where(X[st].notna(), "missing").astype(str).to_numpy()

    def survival(self, X, times):
        return self.m_.survival(self.prep_.transform(X), times, self._strata(X))


class SimpleCox(_Base):
    """Floor: unpenalised-in-effect Cox on a few numeric columns (median fill + missing flag, scaled)."""

    def fit(self, X, time, event):
        cols = self.params["columns"]
        self.prep_ = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                               StandardScaler()), cols)]).fit(X)
        self.m_ = RidgeCox(1e-6).fit(self.prep_.transform(X), time, event)
        return self

    def survival(self, X, times):
        return self.m_.survival(self.prep_.transform(X), times)


class CoxnetMember(_Base):
    """Elastic-net Cox (scikit-survival). With RNA, the RNA columns get `omics_penalty` times the clinical penalty."""

    def fit(self, X, time, event):
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        from .survival import structured
        g = column_groups(self.cfg, self.params["arm"])
        self.prep_ = preprocessor(g, "linear", {}, self.cfg, self.seed).fit(X)
        Z = self.prep_.transform(X)
        pf = np.ones(Z.shape[1])
        if g["rna"]:
            pf[-len(g["rna"]):] = self.params.get("omics_penalty", 1.0)   # RNA columns come last
        a = float(self.params["alpha"])
        y = structured(time, event)

        def fit(alphas):
            return CoxnetSurvivalAnalysis(l1_ratio=self.params["l1_ratio"], alphas=alphas, penalty_factor=pf,
                                          fit_baseline_model=True, max_iter=100000).fit(Z, y)
        try:
            self.m_ = fit([a])
        except ArithmeticError:                 # overflow at a single small alpha: warm-start a short path down to it
            self.m_ = fit(np.geomspace(20 * a, a, 10))
            if not np.isclose(self.m_.alphas_, a).any():
                raise ArithmeticError(f"coxnet path stopped before alpha {a}")
        return self

    def survival(self, X, times):
        fns = self.m_.predict_survival_function(self.prep_.transform(X), alpha=self.params["alpha"])
        return np.vstack([f(np.clip(times, f.domain[0], f.domain[1])) for f in fns])


class RSFMember(_Base):
    def fit(self, X, time, event):
        from sksurv.ensemble import RandomSurvivalForest

        from .survival import structured
        g = column_groups(self.cfg, self.params.get("arm", "C"))
        self.prep_ = preprocessor(g, "linear", None, self.cfg, self.seed).fit(X)
        mf = self.params["max_features"]
        self.m_ = RandomSurvivalForest(n_estimators=self.params["n_estimators"], min_samples_leaf=self.params["min_samples_leaf"],
                                       max_features=mf, random_state=self.seed, n_jobs=1, low_memory=False)
        self.m_.fit(self.prep_.transform(X), structured(time, event))
        return self

    def survival(self, X, times):
        S = self.m_.predict_survival_function(self.prep_.transform(X), return_array=True)
        i = np.searchsorted(self.m_.unique_times_, np.asarray(times, float), side="right") - 1
        return np.where(i[None, :] < 0, 1.0, S[:, np.clip(i, 0, None)])


class KMMember(_Base):
    """Floor: the training Kaplan-Meier curve for everyone (no ranking)."""

    def fit(self, X, time, event):
        km = kaplan_meier(time, event)
        self.t_, self.s_ = km.time.to_numpy(), km.survival.to_numpy()
        return self

    def survival(self, X, times):
        return np.tile(_step(self.t_, self.s_, times, 1.0), (len(X), 1))


KINDS = {"cox": CoxMember, "coxnet": CoxnetMember, "rsf": RSFMember, "simple_cox": SimpleCox, "km": KMMember}


def make(kind: str, cfg, params: dict, seed: int):
    return KINDS[kind](cfg, params, seed)


def columns_for(cfg, kind: str, params: dict) -> list[str]:
    """Raw input columns a member reads (so tasks ship only those)."""
    if kind == "simple_cox":
        return list(params["columns"])
    if kind == "km":
        return ["age_at_diagnosis"]
    g = column_groups(cfg, params.get("arm", "C"))
    return g["numeric"] + g["binary"] + g["categorical"] + g["rna"] + g["dna"]
