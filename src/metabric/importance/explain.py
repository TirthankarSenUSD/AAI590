"""Explanation helpers for one fitted farm pipeline (preprocessor + estimator).

Everything here is computed for a model fitted on an outer split's training rows and evaluated on
that split's test rows only.

- `feature_map`: the name, source and block of every column the estimator actually sees (one-hot
  levels map back to their clinical column; representation outputs keep their own names, e.g.
  "RNA latent 3" or a Hallmark set).
- `shap_by_source`: exact SHAP where cheap: linear (logistic regression) and TreeSHAP (random forest,
  XGBoost); summed per source column, mean |value| over the test rows.
- `lr_coefficients`: coefficients and odds ratios of the logistic regression.
- `permutation_importance`: AUROC drop when a group of raw input columns is shuffled together.
- `gene_groups`: correlated genes grouped (on training rows) so they are permuted together.
- `partial_dependence`, `ice`, `h_statistic`: model-agnostic, on the raw input columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def _steps(t):
    return t.steps if isinstance(t, Pipeline) else [("only", t)]


def _rna_names(t, cols):
    from ..farm.preprocess import FrequencySelector
    from ..farm.representations import DenoisingAE, GeneSetScorer, ModuleScorer
    if t == "passthrough":
        return list(cols)
    names = list(cols)
    for _, s in _steps(t):
        if isinstance(s, FrequencySelector):
            names = [cols[i] for i in s.selected_]
        elif s.__class__.__name__ == "PCA":
            names = [f"RNA PC{k + 1}" for k in range(s.n_components_)]
        elif isinstance(s, GeneSetScorer):
            names = [f"RNA set {n}" for n in s.names]
        elif isinstance(s, ModuleScorer):
            names = [f"RNA module {k + 1}" for k in range(len(s.modules_))]
        elif isinstance(s, DenoisingAE):
            names = [f"RNA latent {k + 1}" for k in range(s.latent)]
    return names


def _dna_names(t, cols):
    from ..farm.representations import ColumnSubset, CooccurrenceEmbedding, MutationCount, PathwayFlags
    if t == "passthrough":
        return list(cols)
    for _, s in _steps(t):
        if isinstance(s, ColumnSubset):
            return [cols[i] for i in s.idx]
        if isinstance(s, MutationCount):
            return ["DNA mutation count"]
        if isinstance(s, PathwayFlags):
            return [f"DNA pathway {n}" for n in s.names]
        if isinstance(s, CooccurrenceEmbedding):
            return [f"DNA embedding {k + 1}" for k in range(s.gene_vectors_.shape[1])]
    return list(cols)


def feature_map(pipe) -> pd.DataFrame:
    """One row per estimator input: feature (display name), source (raw column or representation
    feature it belongs to) and block (clinical / rna / dna)."""
    pre = pipe[0]
    rows = []
    for name, t, cols in pre.transformers_:
        if name == "remainder" or t == "drop":
            continue
        cols = list(cols)
        if name == "num":
            if not isinstance(t, Pipeline):               # fitted "passthrough" becomes a FunctionTransformer
                rows += [(c, c, "clinical") for c in cols]
            else:
                rows += [(c, c, "clinical") for c in cols]
                imp = t.steps[0][1]
                if getattr(imp, "indicator_", None) is not None:
                    rows += [(f"{cols[i]} missing", cols[i], "clinical") for i in imp.indicator_.features_]
        elif name == "cat":
            enc = t.steps[-1][1] if isinstance(t, Pipeline) else t
            if enc.__class__.__name__ == "OneHotEncoder":
                tokens = [f"@@{i}@@" for i in range(len(cols))]
                for out in enc.get_feature_names_out(tokens):
                    i = int(out.split("@@")[1])
                    level = out.split("@@", 2)[2].lstrip("_")
                    rows.append((f"{cols[i]}={level}", cols[i], "clinical"))
            else:
                rows += [(c, c, "clinical") for c in cols]
        elif name == "bin":
            rows += [(c, c, "clinical") for c in cols]
        elif name == "rna":
            rows += [(n, n, "rna") for n in _rna_names(t, cols)]
        elif name == "dna":
            rows += [(n, n, "dna") for n in _dna_names(t, cols)]
    return pd.DataFrame(rows, columns=["feature", "source", "block"])


def shap_by_source(pipe, member: str, X_train: pd.DataFrame, X_test: pd.DataFrame, fm: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per source over the test rows. Linear: exact interventional SHAP in log-odds,
    phi = w * (x - mean_train x). Trees: TreeSHAP (random forest: probability of death; XGBoost:
    log-odds), so magnitudes are comparable only within a member; ranks are compared across members."""
    pre, est = pipe[0], pipe[-1]
    Z = pre.transform(X_test)
    if Z.shape[1] != len(fm):
        raise ValueError("feature map does not match the transformed test rows")
    if member == "logreg_en":
        phi = (Z - pre.transform(X_train).mean(0)) * est.coef_[0]
    else:
        import shap
        sv = shap.TreeExplainer(est).shap_values(Z, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        phi = sv[:, :, 1] if sv.ndim == 3 else sv
    by_src = pd.DataFrame(phi, columns=fm.feature).T.groupby(fm.source.to_numpy()).sum().T   # sum per source, per row
    return by_src.abs().mean(axis=0)


def lr_coefficients(pipe, fm: pd.DataFrame) -> pd.DataFrame:
    """Binary model: one contrast (death vs alive). Cause-specific (multinomial) model: two contrasts,
    cancer death vs alive and other-cause death vs alive (coefficient differences against alive)."""
    est = pipe[-1]
    W = np.atleast_2d(est.coef_)
    if W.shape[0] == 1:
        parts = [("death vs alive", W[0])]
    else:
        cls = list(est.classes_)
        a = W[cls.index(0)]
        parts = [("cancer death vs alive", W[cls.index(1)] - a), ("other-cause death vs alive", W[cls.index(2)] - a)]
    return pd.concat([pd.DataFrame({"contrast": k, "feature": fm.feature, "source": fm.source, "block": fm.block,
                                    "coef": c, "odds_ratio": np.exp(c)}) for k, c in parts], ignore_index=True)


def gene_groups(X_train_rna: pd.DataFrame, distance_cut: float, min_size: int) -> list[list[str]]:
    """Correlated genes found on the training rows (average linkage on 1 - |r|, cut at distance_cut):
    groups of at least `min_size` genes; every other gene is its own group."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    genes = list(X_train_rna.columns)
    R = np.corrcoef(X_train_rna.to_numpy(dtype=float), rowvar=False)
    D = 1 - np.abs(R)
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(np.clip(D, 0, None), checks=False), method="average"), t=distance_cut,
                   criterion="distance")
    out = []
    for k in sorted(set(lab)):
        members = [genes[i] for i in np.flatnonzero(lab == k)]
        out += [members] if len(members) >= min_size else [[g] for g in members]
    return out


def permutation_importance(pipe, X_test: pd.DataFrame, y_test, groups: dict[str, list[str]], n_repeats: int,
                           seed_seq: list[int]) -> pd.DataFrame:
    """AUROC drop when each group of raw columns is shuffled jointly (rows permuted together, so the
    group's internal correlation is kept). Seeded per (task, group, repeat)."""
    y = np.asarray(y_test)
    base = roc_auc_score(y, pipe.predict_proba(X_test)[:, 1])
    rows = []
    for gi, (gname, cols) in enumerate(groups.items()):
        drops = []
        for r in range(n_repeats):
            rng = np.random.default_rng(seed_seq + [gi, r])
            Xp = X_test.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = X_test[cols].iloc[perm].to_numpy()
            drops.append(base - roc_auc_score(y, pipe.predict_proba(Xp)[:, 1]))
        rows.append({"group": gname, "n_columns": len(cols), "importance": float(np.mean(drops)),
                     "importance_sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else np.nan})
    return pd.DataFrame(rows).assign(base_auroc=base)


def _grid(x_train: pd.Series, n: int) -> tuple[list, str]:
    """Evaluation points from the training rows: quantiles for numeric columns ("numeric"), the
    observed levels (up to 12, most frequent first) for categorical or low-cardinality ones ("level")."""
    if pd.api.types.is_numeric_dtype(x_train) and not isinstance(x_train.dtype, pd.CategoricalDtype) and x_train.nunique() > 8:
        return list(np.unique(np.nanquantile(x_train.astype(float), np.linspace(0.05, 0.95, n)))), "numeric"
    vals = x_train.dropna().value_counts()
    return list(vals.index[:12]), "level"


def _set(X: pd.DataFrame, col: str, v) -> pd.DataFrame:
    """Copy of X with `col` set to v for every row (missing values allowed; dtype preserved)."""
    Xv = X.copy()
    if isinstance(Xv[col].dtype, pd.CategoricalDtype):
        Xv[col] = pd.Categorical([v] * len(Xv), categories=Xv[col].cat.categories)
    else:
        Xv[col] = pd.Series([v] * len(Xv), index=Xv.index, dtype=Xv[col].dtype)
    return Xv


def partial_dependence(pipe, X_train: pd.DataFrame, X_test: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """Mean predicted risk on the test rows with `col` set to each grid value (grid from training rows)."""
    grid, kind = _grid(X_train[col], n)
    return pd.DataFrame([{"pos": i, "value": v, "kind": kind,
                          "mean_pred": float(pipe.predict_proba(_set(X_test, col, v))[:, 1].mean())}
                         for i, v in enumerate(grid)])


def ice(pipe, X_train: pd.DataFrame, X_rows: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    rows = []
    grid, kind = _grid(X_train[col], n)
    for i, v in enumerate(grid):
        p = pipe.predict_proba(_set(X_rows, col, v))[:, 1]
        rows += [{"patient_id": pid, "pos": i, "value": v, "kind": kind, "pred": float(pi)} for pid, pi in zip(X_rows.index, p)]
    return pd.DataFrame(rows)


def h_statistic(pipe, X_rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Friedman's pairwise H^2 on the given rows, on the log-odds scale (Friedman & Popescu 2008):
    share of the joint partial dependence of (j, k) not explained by their separate partial
    dependences (0 = no interaction). Model-agnostic; a logistic regression gives ~0 by construction."""
    m = len(X_rows)
    cols = list(dict.fromkeys(cols))

    def pd_at(fix: list[str]) -> np.ndarray:
        out = np.empty(m)
        for i in range(m):
            Xi = X_rows.copy()
            for c in fix:
                Xi = _set(Xi, c, X_rows[c].iloc[i])
            p = np.clip(pipe.predict_proba(Xi)[:, 1], 1e-6, 1 - 1e-6)
            out[i] = np.log(p / (1 - p)).mean()
        return out - out.mean()

    single = {c: pd_at([c]) for c in cols}
    rows = []
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            j, k = cols[a], cols[b]
            joint = pd_at([j, k])
            den = float((joint ** 2).sum())
            h2 = float(((joint - single[j] - single[k]) ** 2).sum() / den) if den > 0 else 0.0
            rows.append({"feature_a": j, "feature_b": k, "h2": h2})
    return pd.DataFrame(rows)
