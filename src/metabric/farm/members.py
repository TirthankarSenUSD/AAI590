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

FAMILY = {"logreg_en": 1, "logreg_cause": 1.5, "svm_rbf": 2, "random_forest": 3, "xgboost": 3,   # simpler first
          "xgboost_cause": 3.5, "mlp": 4, "mlp_embed": 4, "multimodal": 5}
KIND = {"logreg_en": "linear", "logreg_cause": "linear", "svm_rbf": "linear", "random_forest": "tree",
        "xgboost": "tree", "xgboost_cause": "tree", "mlp": "linear", "mlp_embed": "embed", "multimodal": "linear"}
CAUSE = {"logreg_cause": "logreg_en", "xgboost_cause": "xgboost"}   # cause-specific member -> its binary counterpart
NEURAL = {"mlp", "mlp_embed", "multimodal"}


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


def _estimator(member: str, p: dict, seed: int, groups: dict | None = None, nn: dict | None = None):
    if member in NEURAL:
        from .neural import TorchEmbedMLP, TorchMLP, TorchMultimodal
        common = dict(weight_decay=p["weight_decay"], dropout=p["dropout"], random_state=seed, **(nn or {}))
        if member == "mlp":
            return TorchMLP(hidden=tuple(p["hidden"]), **common)
        if member == "mlp_embed":
            return TorchEmbedMLP(n_cat=len(groups["categorical"]), hidden=tuple(p["hidden"]), **common)
        return TorchMultimodal(n_rna=len(groups["rna"]), block_dropout=p["block_dropout"], **common)
    if member in ("logreg_en", "logreg_cause"):          # 3 classes -> multinomial (cause-specific)
        return LogisticRegression(C=p["C"], l1_ratio=p["l1_ratio"], solver="saga", max_iter=3000, tol=1e-3,
                                  random_state=seed)
    if member == "svm_rbf":
        return SVC(C=p["C"], gamma=p["gamma"], probability=True, random_state=seed)   # Platt inside the fold
    if member == "random_forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_features=p["max_features"],
                                      min_samples_leaf=p["min_samples_leaf"], n_jobs=1, random_state=seed)
    if member in ("xgboost", "xgboost_cause"):            # 3 classes -> multi:softprob (cause-specific)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                             subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                             min_child_weight=p["min_child_weight"], tree_method="hist", n_jobs=1,
                             random_state=seed, eval_metric="logloss")
    raise ValueError(member)


def _hyper_complexity(member: str, p: dict, grid: dict) -> float:
    """Within a member, stronger regularisation counts as simpler (value in [0, 1))."""
    member = CAUSE.get(member, member)
    idx = lambda k: grid[k].index(p[k]) / max(len(grid[k]), 1)  # noqa: E731
    if member == "logreg_en":
        return 0.5 * idx("C") + 0.1 * (1 - p["l1_ratio"])
    if member == "svm_rbf":
        return 0.5 * idx("C")
    if member == "random_forest":
        return 0.5 * (1 - idx("min_samples_leaf") - 1 / len(grid["min_samples_leaf"]))
    if member == "xgboost":
        return 0.5 * idx("max_depth") + 0.1 * idx("colsample_bytree")
    if member in ("mlp", "mlp_embed"):
        return 0.5 * idx("hidden") + 0.1 * (1 - idx("dropout") - 1 / len(grid["dropout"]))
    if member == "multimodal":
        return 0.5 * (1 - idx("block_dropout") - 1 / len(grid["block_dropout"]))
    return 0.0


def _n_inputs(rna_rep: dict, dna_rep: dict, groups: dict, cfg) -> int:
    """Number of omics inputs a representation feeds the model (for the "simpler" order)."""
    n = 0
    if groups["rna"]:
        from .preprocess import gene_set_positions
        if "sets" in rna_rep:
            gs = cfg["farm"]["gene_set_scores"]
            n += len(gene_set_positions(cfg, rna_rep["sets"], "rna", gs[f"{rna_rep['sets']}_min_rna"])[0])
        elif "modules" in rna_rep:
            n += 20                                         # data-dependent; a fixed nominal size
        else:
            n += rna_rep.get("topk") or rna_rep.get("pca") or rna_rep.get("dae") or len(groups["rna"])
    if groups.get("dna"):
        from .preprocess import gene_set_positions
        if "known" in dna_rep:
            n += len(dna_rep["known"])
        elif "count" in dna_rep:
            n += 1
        elif "pathway" in dna_rep:
            n += len(gene_set_positions(cfg, dna_rep["pathway"], "dna", dna_rep["min_dna"])[0])
        elif "embed" in dna_rep:
            n += dna_rep["embed"]
        else:
            n += len(groups["dna"])
    return n


def _arm_reps(fc: dict, arm: str, groups: dict, member: str) -> list[tuple[str, dict, str, dict]]:
    """(rna name, rna rep, dna name, dna rep) combinations a member tries on an arm."""
    a = fc["arms"][arm]
    if groups["rna"]:
        names = a.get("rna_reps", list(fc["rna_representations"]))
        if member == "multimodal":
            names = ["all"]                                 # one encoder over the full RNA block
        elif member in NEURAL:
            names = [n for n in fc["neural_rna_representations"] if n in names]
        elif member in CAUSE:
            names = [n for n in fc["cause_rna_representations"] if n in names]
        if member not in fc["ae_members"]:
            names = [n for n in names if "dae" not in fc["rna_representations"][n]]
        rna = [(n, fc["rna_representations"][n]) for n in names]
    else:
        rna = [("", {})]
    if groups.get("dna"):
        dna = [(n, fc["dna_representations"][n]) for n in a.get("dna_reps", list(fc["dna_representations"]))]
    else:
        dna = [("", {})]
    return [(rn, rr, dn, dr) for rn, rr in rna for dn, dr in dna]


def build_candidates(cfg, members=None, arms=None) -> tuple[list[Candidate], dict[str, Meta]]:
    fc, seed = cfg["farm"], cfg.seed
    members = members or fc["members"]
    arms = arms or list(fc["arms"])
    cands, meta = [], {}
    for arm in arms:
        groups = column_groups(cfg, arm)
        cols = groups["numeric"] + groups["binary"] + groups["categorical"] + groups["rna"] + groups["dna"]
        n_blocks = sum(bool(x) for x in (groups["numeric"] + groups["categorical"], groups["rna"], groups["dna"]))
        for member in members:
            if member in NEURAL and not fc["arms"][arm].get("neural", True):
                continue                                   # neural members deferred on this arm
            if member in CAUSE and arm not in fc["cause_arms"]:
                continue                                   # cause-specific members only where configured
            if member == "multimodal" and (n_blocks < 2 or groups["dna"] or not groups["rna"]):
                continue                                   # clinical + RNA encoders only (for now)
            grid = fc["grids"][member]
            keys = list(grid)
            for rna_name, rna_rep, dna_name, dna_rep in _arm_reps(fc, arm, groups, member):
                rep_name = "+".join(x for x in (rna_name, dna_name) if x) or "none"
                for values in itertools.product(*(grid[k] for k in keys)):
                    p = dict(zip(keys, values))
                    scale = fc["rna_scale"] if member == "logreg_en" and rna_name == "all" else None
                    pstr = ",".join(f"{k}={v}" for k, v in p.items())
                    name = f"{arm}|{member}|{rep_name}|{pstr}"

                    def make(member=member, p=p, rr=rna_rep, dr=dna_rep, groups=groups, scale=scale):
                        pipe = make_pipeline(preprocessor(groups, KIND[member], rr, cfg, seed, scale, dr),
                                             _estimator(member, p, seed, groups, fc.get("neural")))
                        if member in CAUSE:
                            from .cause import CauseSpecific
                            return CauseSpecific(pipe)
                        return pipe

                    complexity = (FAMILY[member] * 1000 + _n_inputs(rna_rep, dna_rep, groups, cfg)
                                  + _hyper_complexity(member, p, grid))
                    cands.append(Candidate(name, make, cols, complexity, True, f"{member} on {arm} ({rep_name}; {pstr})",
                                           target="cause" if member in CAUSE else "binary"))
                    meta[name] = Meta(name, arm, member, rep_name, tuple(p.items()))
    return cands, meta
