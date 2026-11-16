"""Farm members and their candidate configurations (arm x member x RNA representation x grid)."""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from ..harness import Candidate
from .preprocess import column_groups, preprocessor

FAMILY = {"logreg_en": 1, "svm_rbf": 2, "random_forest": 3, "xgboost": 3}   # simpler first
KIND = {"logreg_en": "linear", "svm_rbf": "linear", "random_forest": "tree", "xgboost": "tree"}


@dataclass(frozen=True)
class Meta:
    name: str
    arm: str
    member: str
    rep: str
    params: tuple


def resolve_n_jobs(cfg) -> int:
    n = cfg["farm"].get("n_jobs", 1)
    return max(1, (os.cpu_count() or 2) - 1) if n == "auto" else int(n)


def _estimator(member: str, p: dict, seed: int):
    if member == "logreg_en":
        return LogisticRegression(C=p["C"], l1_ratio=p["l1_ratio"], solver="saga", max_iter=3000, tol=1e-3,
                                  random_state=seed)
    if member == "svm_rbf":
        return SVC(C=p["C"], gamma=p["gamma"], probability=True, random_state=seed)   # Platt inside the fold
    if member == "random_forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_features=p["max_features"],
                                      min_samples_leaf=p["min_samples_leaf"], n_jobs=1, random_state=seed)
    if member == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                             subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                             min_child_weight=p["min_child_weight"], tree_method="hist", n_jobs=1,
                             random_state=seed, eval_metric="logloss")
    raise ValueError(member)


def _hyper_complexity(member: str, p: dict, grid: dict) -> float:
    """Within a member, stronger regularisation counts as simpler (value in [0, 1))."""
    idx = lambda k: grid[k].index(p[k]) / max(len(grid[k]), 1)  # noqa: E731
    if member == "logreg_en":
        return 0.5 * idx("C") + 0.1 * (1 - p["l1_ratio"])
    if member == "svm_rbf":
        return 0.5 * idx("C")
    if member == "random_forest":
        return 0.5 * (1 - idx("min_samples_leaf") - 1 / len(grid["min_samples_leaf"]))
    if member == "xgboost":
        return 0.5 * idx("max_depth") + 0.1 * idx("colsample_bytree")
    return 0.0


def _rna_inputs(rep: dict, n_rna: int) -> int:
    return rep.get("topk") or rep.get("pca") or (n_rna if n_rna else 0)


def build_candidates(cfg, members=None, arms=None) -> tuple[list[Candidate], dict[str, Meta]]:
    fc, seed = cfg["farm"], cfg.seed
    members = members or fc["members"]
    arms = arms or list(fc["arms"])
    cands, meta = [], {}
    for arm in arms:
        groups = column_groups(cfg, arm)
        cols = groups["numeric"] + groups["binary"] + groups["categorical"] + groups["rna"]
        reps = fc["rna_representations"] if groups["rna"] else {"none": {}}
        for member in members:
            grid = fc["grids"][member]
            keys = list(grid)
            for rep_name, rep in reps.items():
                for values in itertools.product(*(grid[k] for k in keys)):
                    p = dict(zip(keys, values))
                    scale = fc["rna_scale"] if member == "logreg_en" and rep_name == "all" else None
                    pstr = ",".join(f"{k}={v}" for k, v in p.items())
                    name = f"{arm}|{member}|{rep_name}|{pstr}"

                    def make(member=member, p=p, rep=rep, groups=groups, scale=scale):
                        return make_pipeline(preprocessor(groups, KIND[member], rep, cfg, seed, scale),
                                             _estimator(member, p, seed))

                    complexity = FAMILY[member] * 1000 + _rna_inputs(rep, len(groups["rna"])) + _hyper_complexity(member, p, grid)
                    cands.append(Candidate(name, make, cols, complexity, True, f"{member} on {arm} ({rep_name}; {pstr})"))
                    meta[name] = Meta(name, arm, member, rep_name, tuple(p.items()))
    return cands, meta
