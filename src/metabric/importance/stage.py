"""`python run.py importance [--farm-run RUN_ID]`: gates and feature importance (build step 4).

Explains the winners a completed farm run chose inside each outer fold. Each winning configuration
is refitted on its outer split's training rows (bit-identical to the farm fit, which is checked
against the farm's recorded outer AUROC) and explained on that split's test rows only.
"""
from __future__ import annotations

import json
import time
import zlib

import numpy as np
import pandas as pd
import yaml

from ..core import checkpoint
from ..core.artifacts import Run
from ..core.config import Config
from ..data import load_dataset
from ..eda import plots as P
from ..farm.members import build_candidates, resolve_n_jobs
from ..farm.preprocess import column_groups
from ..harness import splits_from_table
from ..harness.core import _Progress
from ..harness.floors import design_matrix
from ..splits import load_cv_splits, training_ids
from . import explain as E

# sections that decide how a winner is built and fitted (the harness section only affects scoring,
# and the winners are already chosen), so they must match the farm run for exact refits
MATCH_SECTIONS = ("farm", "columns", "seed", "clinical_types", "groups", "cv", "holdout")


def find_farm_run(cfg: Config, run_id: str | None) -> str:
    out = cfg.path("outputs")
    if run_id:
        return run_id
    cands = []
    for d in sorted(out.iterdir(), reverse=True):
        m = d / "manifest.json"
        if d.is_dir() and m.is_file():
            j = json.loads(m.read_text())
            if j["command"].startswith("farm") and j["status"] == "complete":
                cands.append(d.name)
    if not cands:
        raise FileNotFoundError("no completed farm run found; run `python run.py farm` first")
    return cands[0]


def _check_config(cfg: Config, farm_dir) -> None:
    old = yaml.safe_load((farm_dir / "config.yaml").read_text(encoding="utf-8"))
    diff = [k for k in MATCH_SECTIONS if json.dumps(old.get(k), sort_keys=True) != json.dumps(cfg.raw.get(k), sort_keys=True)]
    if diff:
        raise RuntimeError(f"config sections {diff} differ from the farm run's config: its winners cannot be rebuilt exactly")


# ---------------------------------------------------------------------------------- worker tasks
def _explain_task(key, cand, member, groups_clin, rna_cols, dna_cols, Xtr, ytr, Xte, yte, mode, s):
    """Refit one winning configuration on one outer split and explain it on the test rows. `ytr` is
    the label the candidate is fitted on (the cause label for cause-specific members); `yte` is
    binary death, used for every score."""
    pipe = cand.make().fit(Xtr[cand.columns], ytr.to_numpy())
    Xte_c = Xte[cand.columns]
    p = pipe.predict_proba(Xte_c)[:, 1]
    from sklearn.metrics import roc_auc_score
    out = {"key": key, "refit_auroc": float(roc_auc_score(yte, p))}
    seed = [s["seed"], zlib.crc32(f"{key[0]}|{key[1]}".encode())]
    modality = {}
    if groups_clin:
        modality["all clinical"] = groups_clin
        tr = [c for c in s["treatment"] if c in groups_clin]
        if tr:
            modality["treatment group"] = tr
    if rna_cols:
        modality["all RNA"] = rna_cols
    if dna_cols:
        modality["all DNA"] = dna_cols
    out["modality"] = E.permutation_importance(pipe, Xte_c, yte, modality, s["n_repeats"], seed + [0])
    if mode == "full":
        groups = {c: [c] for c in groups_clin}
        if rna_cols:
            for g in E.gene_groups(Xtr[rna_cols], s["distance_cut"], s["min_size"]):
                groups[g[0] if len(g) == 1 else "genes: " + ", ".join(g)] = g
        out["perm"] = E.permutation_importance(pipe, Xte_c, yte, groups, s["n_repeats"], seed + [1])
        fm = E.feature_map(pipe)
        if member in s["shap_members"]:
            out["shap"] = E.shap_by_source(pipe, member, Xtr[cand.columns], Xte_c, fm).rename("mean_abs_shap").reset_index() \
                .rename(columns={"index": "source"})
        if member in ("logreg_en", "logreg_cause"):
            out["coef"] = E.lr_coefficients(pipe, fm)
    return out


def _detail_task(key, cand, cols_top, Xtr, ytr, Xte, yte, split_id, s):
    """Partial dependence, ICE (one split), H-statistic and learning curve for a best-member winner."""
    from sklearn.metrics import roc_auc_score
    pipe = cand.make().fit(Xtr[cand.columns], ytr.to_numpy())
    ybin = (ytr > 0).astype(int)                          # learning-curve sampling and scores use death
    Xte_c, Xtr_c = Xte[cand.columns], Xtr[cand.columns]
    rng = np.random.default_rng([s["seed"], zlib.crc32(f"{key[0]}|{key[1]}|detail".encode())])
    feats = [c for c in cols_top if c in cand.columns]
    pdp = pd.concat([E.partial_dependence(pipe, Xtr_c, Xte_c, c, s["pdp_grid"]).assign(feature=c) for c in feats],
                    ignore_index=True) if feats else pd.DataFrame()
    ice = pd.DataFrame()
    if split_id == s["ice_split"] and feats:
        rows = Xte_c.iloc[rng.choice(len(Xte_c), min(s["ice_rows"], len(Xte_c)), replace=False)]
        ice = pd.concat([E.ice(pipe, Xtr_c, rows, c, s["pdp_grid"]).assign(feature=c) for c in feats], ignore_index=True)
    hrows = Xte_c.iloc[rng.choice(len(Xte_c), min(s["h_rows"], len(Xte_c)), replace=False)]
    h = E.h_statistic(pipe, hrows, feats) if len(feats) > 1 else pd.DataFrame()
    lc = []
    for f in s["lc_fractions"]:
        if f >= 1:
            idx = Xtr.index
        else:
            idx = pd.Index(np.concatenate([rng.choice(ybin.index[ybin == v], int(round(f * (ybin == v).sum())), replace=False)
                                           for v in (0, 1)]))
        m = cand.make().fit(Xtr.loc[idx, cand.columns], ytr.loc[idx].to_numpy())
        lc.append({"fraction": f, "n_train": len(idx), "cv_auroc": roc_auc_score(yte, m.predict_proba(Xte_c)[:, 1]),
                   "train_auroc": roc_auc_score(ybin.loc[idx], m.predict_proba(Xtr.loc[idx, cand.columns])[:, 1])})
    return {"key": key, "pdp": pdp, "ice": ice, "h": h, "lc": pd.DataFrame(lc)}


def _parallel(fn, tasks, n_jobs, log, label):
    """Run `fn(*t)` for every task (t[0] is the task's key; results carry it as r["key"]).
    Resumable: finished tasks are saved as they complete and skipped when the run is repeated.
    Results come back in task order, whatever order the workers finish in."""
    res = _parallel_any_order(fn, tasks, n_jobs, log, label)
    pos = {repr(t[0]): i for i, t in enumerate(tasks)}
    return sorted(res, key=lambda r: pos[repr(r["key"])])


def _parallel_any_order(fn, tasks, n_jobs, log, label):
    ck = checkpoint.Loop(f"parallel:{label}:{fn.__name__}")
    prog = _Progress(len(tasks), log, label)
    res, todo = [], []
    for t in tasks:
        found, v = ck.get(t[0])
        if found:
            res.append(v)
            prog.tick()
        else:
            todo.append(t)
    if res and log:
        log(f"{label}: resumed {len(res)}/{len(tasks)} tasks from checkpoints")
    if n_jobs == 1 or len(todo) <= 1:
        for t in todo:
            r = fn(*t)
            ck.put(r["key"], r)
            res.append(r)
            prog.tick()
        return res
    from joblib import Parallel, delayed, parallel_config
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for r in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(delayed(fn)(*t) for t in todo):
            ck.put(r["key"], r)
            res.append(r)
            prog.tick()
    return res


# ---------------------------------------------------------------------------------------- stage
REUSE_TABLES = ("refit_check", "modality_importance", "permutation_importance", "shap_values_by_split", "lr_coefficients")


def run_importance(cfg: Config, run: Run, farm_run: str | None = None, reuse: str | None = None,
                   procedures: list[str] | None = None) -> dict:
    """`reuse`: an earlier importance run on the same farm run whose per-split tables (pass 1: refits,
    permutation importance, SHAP, coefficients) are reused, so only the summaries and the detail pass
    are recomputed. The reused tables are recorded as inputs."""
    st = run.stage("importance")
    ic = cfg["importance"]
    farm_id = find_farm_run(cfg, farm_run)
    farm_dir = cfg.path("outputs") / farm_id
    _check_config(cfg, farm_dir)
    for f in ("procedure_winners.csv", "overfit_gate.csv", "regression_floor.csv", "procedure_summary.csv"):
        run.record_input(farm_dir / "farm" / "tables" / f, f"farm run {farm_id} (winners chosen inside the outer folds)")
    win = pd.read_csv(farm_dir / "farm" / "tables" / "procedure_winners.csv")
    full = load_dataset(cfg)
    tr_ids = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows used; holdout rows untouched)")
    run.record_input(cfg.path("cv_splits"), "frozen CV splits")
    ds = full.subset(tr_ids)
    X, y = design_matrix(ds), ds.y
    from ..farm.members import CAUSE
    members_all = list(dict.fromkeys(win.winner.str.split("|").str[1]))   # only members that appear among the winners
    y_cause = None                                     # cause label (training only) for cause-specific winners
    if win.winner.str.split("|").str[1].isin(list(CAUSE)).any():
        from ..farm.cause import cause_labels
        y_cause = cause_labels(ds.frame)
    fit_y = lambda cand, rows: (y_cause if cand.target == "cause" else y).loc[rows]  # noqa: E731
    splits = {s.split_id: s for s in splits_from_table(load_cv_splits(cfg, tr_ids))}
    arms = list(ic["arms_full"]) + list(ic["arms_modality_only"])
    if procedures:                       # explain only these procedures (full mode); no detail pass
        missing = sorted(set(procedures) - set(win.procedure))
        if missing:
            raise ValueError(f"procedures not in farm run {farm_id}: {missing}")
        win = win[win.procedure.isin(procedures)]
        arms = list(dict.fromkeys(pr.split("|")[0] for pr in procedures))
        ic = {**ic, "arms_full": arms, "arms_modality_only": []}
    cands, meta = build_candidates(cfg, members=members_all, arms=arms)
    by_name = {c.name: c for c in cands}
    n_jobs = resolve_n_jobs(cfg)
    s = {"seed": cfg.seed, "n_repeats": ic["n_repeats"], "distance_cut": ic["gene_groups"]["distance_cut"],
         "min_size": ic["gene_groups"]["min_size"], "shap_members": ic["shap_members"], "treatment": cfg["groups"]["treatment"],
         "pdp_grid": ic["pdp_grid"], "ice_rows": ic["ice_rows"], "h_rows": ic["h_rows"], "ice_split": "r0o0",
         "lc_fractions": ic["learning_curve_fractions"]}

    # tasks: one per unique (winning configuration, split); procedures share tasks
    proc_keys, tasks, seen = [], [], {}
    for arm in arms:
        mode = "full" if arm in ic["arms_full"] else "modality"
        g = column_groups(cfg, arm)
        clin = g["numeric"] + g["binary"] + g["categorical"]
        w_arm = win[win.procedure.str.startswith(f"{arm}|") & ~win.procedure.str.contains(r"\|rep:", regex=True)]
        for _, r in w_arm.iterrows():
            key = (r.winner, r.split_id)
            proc_keys.append({"arm": arm, "procedure": r.procedure, "split_id": r.split_id, "winner": r.winner,
                              "member": meta[r.winner].member, "farm_outer_auroc": r.outer_auroc, "mode": mode})
            if key in seen:
                seen[key] = "full" if "full" in (seen[key], mode) else mode
                continue
            seen[key] = mode
            sp = splits[r.split_id]
            tasks.append([key, by_name[r.winner], meta[r.winner].member, clin, g["rna"], g["dna"],
                          X.loc[sp.train], fit_y(by_name[r.winner], sp.train), X.loc[sp.test], y.loc[sp.test], mode, s])
    for t in tasks:                                       # a task shared by a full and a modality arm runs in full
        t[10] = seen[t[0]]
    pk = pd.DataFrame(proc_keys)
    t0 = time.time()
    if reuse:
        src = cfg.path("outputs") / reuse / "importance" / "tables"
        names = [n for n in REUSE_TABLES if (src / f"{n}.csv").is_file()]
        if "shap_values_by_split" not in names:          # runs before this table existed: use their SHAP summary,
            names.append("shap_summary")                 # which is exact for member procedures (one member each)
        for name in names:
            run.record_input(src / f"{name}.csv", f"pass-1 table reused from importance run {reuse}")
        old = {name: pd.read_csv(src / f"{name}.csv") for name in names}
        chk = old["refit_check"]
        cmp = chk[["procedure", "split_id", "winner"]].merge(pk[["procedure", "split_id", "winner"]], how="outer", indicator=True)
        if cmp["_merge"].ne("both").any():
            raise RuntimeError(f"importance run {reuse} does not cover the same procedures and winners")
        pk = chk
        modality, perm, coef = old["modality_importance"], old["permutation_importance"], old["lr_coefficients"]
        shap_t = old.get("shap_values_by_split")
        shap_old = None
        if shap_t is None:
            shap_old = old["shap_summary"]
            shap_old = shap_old[~shap_old.procedure.str.endswith("|best")].drop(columns=["member"], errors="ignore")
        st.log.info("importance: farm run %s; reusing pass-1 tables of importance run %s (%d refits)", farm_id, reuse, len(tasks))
    else:
        st.log.info("importance: farm run %s; %d procedure-split rows -> %d unique refits; n_jobs=%d", farm_id, len(pk), len(tasks), n_jobs)
        res = {r["key"]: r for r in _parallel(_explain_task, tasks, n_jobs, st.log.info, "explaining winners")}
        st.log.info("winners explained in %.0f s", time.time() - t0)
        pk["refit_auroc"] = [res[(w, sid)]["refit_auroc"] for w, sid in zip(pk.winner, pk.split_id)]
        pk["abs_diff"] = (pk.refit_auroc - pk.farm_outer_auroc).abs()

        def gather(kind):
            parts = []
            for r in pk.itertuples():
                d = res[(r.winner, r.split_id)].get(kind)
                if d is not None and len(d):
                    parts.append(d.assign(arm=r.arm, procedure=r.procedure, member=r.member, split_id=r.split_id))
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        modality, perm, shap_t, coef = gather("modality"), gather("perm"), gather("shap"), gather("coef")
        full_procs = pk.loc[pk["mode"] == "full", "procedure"].unique()
        perm = perm[perm.procedure.isin(full_procs)]
        # SHAP scales differ between members, so SHAP is summarised per member procedure only
        shap_t = shap_t[shap_t.procedure.isin(full_procs) & ~shap_t.procedure.str.endswith("|best")] if len(shap_t) else shap_t
        coef = coef[coef.procedure.str.endswith(("|logreg_en", "|logreg_cause"))] if len(coef) else coef
        shap_old = None
    st.table("refit_check", pk, "Refitted winners reproduce the farm's outer AUROC (abs_diff must be ~0)")
    bad = pk[pk.abs_diff > 1e-6]
    if len(bad):
        raise RuntimeError(f"{len(bad)} refits do not reproduce the farm run:\n{bad.head().to_string()}")
    full_pk = pk[pk["mode"] == "full"]
    st.table("modality_importance", modality, "Grouped permutation importance by data block (AUROC drop when the whole block is shuffled)")
    st.table("permutation_importance", perm, "Permutation importance per procedure and split (AUROC drop; correlated genes shuffled together)")
    items = _items(perm)
    # summarised per procedure: a best-member procedure's winner can be a different member in each split
    summ = _summarise(items, "importance", ic["top_n"], ["arm", "procedure"])
    st.table("permutation_summary", summ, f"Permutation importance across the 10 splits: mean, SD, mean rank, share of splits in the top {ic['top_n']}")
    if shap_t is not None:
        st.table("shap_values_by_split", shap_t, "Mean |SHAP| per source, per member procedure and split")
        shap_s = _summarise(shap_t.rename(columns={"source": "item"}), "mean_abs_shap", ic["top_n"], ["arm", "procedure"]) \
            if len(shap_t) else pd.DataFrame()
    else:
        shap_s = shap_old
    st.table("shap_summary", shap_s, "Mean |SHAP| per source across splits, member procedures only (exact: linear for logistic "
             "regression, TreeSHAP for trees; scales differ by member, compare ranks)")
    st.table("lr_coefficients", coef, "Logistic regression coefficients per split (standardised inputs; one-hot levels as indicators)")
    coef_s = _coef_summary(coef)
    st.table("lr_coefficient_summary", coef_s, "Logistic regression coefficients across splits: mean, SD, sign consistency, odds ratio per unit")

    # pass 2: details for each full arm's best-member procedure
    top = {}
    for arm in ic["arms_full"]:
        si = summ[(summ.procedure == f"{arm}|best") & (summ.n_columns == 1)].sort_values("mean", ascending=False)
        if len(si):
            top[arm] = si.item.head(ic["top_n"]).tolist()
    st.table("top_features", pd.DataFrame([{"arm": a, "rank": i + 1, "feature": f} for a, fs in top.items() for i, f in enumerate(fs)]),
             f"Top {ic['top_n']} single-column features of each best-member procedure (by mean permutation importance)")
    dtasks, dkeys = [], []
    for arm in top:
        for r in full_pk[full_pk.procedure == f"{arm}|best"].itertuples():
            sp = splits[r.split_id]
            dtasks.append([(r.winner, r.split_id, arm), by_name[r.winner], top[arm], X.loc[sp.train],
                           fit_y(by_name[r.winner], sp.train), X.loc[sp.test], y.loc[sp.test], r.split_id, s])
            dkeys.append(r)
    t1 = time.time()
    dres = {r["key"]: r for r in _parallel(_detail_task, dtasks, n_jobs, st.log.info, "partial dependence and interactions")}
    st.log.info("details in %.0f s", time.time() - t1)
    pdp, ice, hh, lc = [], [], [], []
    for r in dkeys:
        d = dres[(r.winner, r.split_id, r.arm)]
        for lst, k in ((pdp, "pdp"), (ice, "ice"), (hh, "h"), (lc, "lc")):
            if len(d[k]):
                lst.append(d[k].assign(arm=r.arm, split_id=r.split_id, winner=r.winner))
    pdp, ice, hh, lc = (pd.concat(x, ignore_index=True) if x else pd.DataFrame() for x in (pdp, ice, hh, lc))
    st.table("partial_dependence", pdp, "Partial dependence of the best-member procedure, per split (grid from training rows)")
    st.table("ice", ice, f"ICE curves for {ic['ice_rows']} held-out patients of split r0o0")
    hs = (hh.groupby(["arm", "feature_a", "feature_b"]).h2.agg(["mean", "std", "min", "max"]).reset_index()
          .sort_values(["arm", "mean"], ascending=[True, False])) if len(hh) else pd.DataFrame(columns=["arm", "feature_a", "feature_b", "mean"])
    st.table("interactions", hs, f"Friedman H^2 for pairs of the top {ic['top_n']} features, across splits ({ic['h_rows']} held-out patients per split)")
    st.table("learning_curve", lc, "Learning curves of the best-member procedures (refits on stratified fractions of the outer training rows)")

    gate = pd.read_csv(farm_dir / "farm" / "tables" / "overfit_gate.csv")
    gate = gate[gate.candidate.str.split("|").str[0].isin(ic["arms_full"]) & ~gate.candidate.str.contains("rep:")]
    st.table("gate_summary", gate, f"Overfit gate of the explained procedures (from farm run {farm_id}; report only)")
    rf = pd.read_csv(farm_dir / "farm" / "tables" / "regression_floor.csv")
    rf = rf[rf.candidate.isin([f"{a}|best" for a in ic["arms_full"]])]
    st.table("regression_floor_provisional", rf, "Provisional regression floors (mean - 2 SD outer AUROC) of the best-member procedures; "
             "the final model's floor is fixed when step 5 chooses it")

    _figures(st, summ, coef_s, pdp, ice, hs, modality, lc, top, ic)
    facts = _facts(summ, coef_s, shap_s, hs, modality, lc, top, farm_id, len(tasks), time.time() - t0, ic)
    st.text("importance_summary.md", _summary_md(run, cfg, facts, ic), "Auto-generated importance summary (computed wording)")
    return facts


def _items(perm: pd.DataFrame) -> pd.DataFrame:
    """Expand gene groups so every gene carries its group's importance (shared credit); clinical
    columns and single genes are items of their own."""
    rows = []
    perm = perm.drop(columns=[c for c in ("item", "group_size") if c in perm])
    for r in perm.itertuples():
        if r.group.startswith("genes: "):
            for g in r.group[7:].split(", "):
                rows.append({**r._asdict(), "item": g, "group_size": r.n_columns})
        else:
            rows.append({**r._asdict(), "item": r.group, "group_size": 1})
    out = pd.DataFrame(rows).drop(columns=["Index"])
    out["n_columns"] = out.group_size
    return out


def _summarise(df: pd.DataFrame, value: str, top_n: int, by: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["rank"] = d.groupby(by + ["split_id"])[value].rank(ascending=False, method="min")
    agg = d.groupby(by + ["item"]).agg(mean=(value, "mean"), sd=(value, "std"), n_splits=("split_id", "nunique"),
                                       mean_rank=("rank", "mean"), top_share=("rank", lambda r: float((r <= top_n).mean())))
    if "n_columns" in d:
        agg["n_columns"] = d.groupby(by + ["item"]).n_columns.max()
    agg = agg.reset_index()
    agg["_m"] = agg["mean"].round(12)                    # ties (e.g. genes sharing a group) broken by name
    return agg.sort_values(by + ["_m", "item"], ascending=[True] * len(by) + [False, True]).drop(columns="_m").reset_index(drop=True)


def _coef_summary(coef: pd.DataFrame) -> pd.DataFrame:
    if coef.empty:
        return coef
    if "contrast" not in coef:
        coef = coef.assign(contrast="death vs alive")
    keys = ["arm", "procedure", "contrast", "feature", "source", "block"]
    g = coef.groupby(keys)
    s = g.coef.agg(mean="mean", sd="std", n_splits="size").reset_index()
    sign = g.coef.apply(lambda c: float((np.sign(c) == np.sign(c.mean())).mean()) if c.mean() != 0 else 0.0).rename("sign_consistency")
    s = s.merge(sign.reset_index(), on=keys)
    s["odds_ratio"] = np.exp(s["mean"])
    s["abs_mean"] = s["mean"].abs()
    return s.sort_values(["arm", "procedure", "contrast", "abs_mean"], ascending=[True, True, True, False]).drop(columns="abs_mean")


def _figures(st, summ, coef_s, pdp, ice, hs, modality, lc, top, ic):
    for arm in ic["arms_full"]:
        t = summ[summ.procedure == f"{arm}|best"].head(15).iloc[::-1]
        if t.empty:
            continue
        fig, ax = P.fig(7, 0.32 * len(t) + 1.3)
        lab = [f"{i} ({int(n)} genes)" if n > 1 else i for i, n in zip(t.item, t.n_columns)]
        ax.barh(lab, t["mean"], xerr=t["sd"], color=P.CAT[0], height=0.6, error_kw={"ecolor": P.INK2, "lw": 1})
        for yv, sh in enumerate(t.top_share):
            ax.text(ax.get_xlim()[1], yv, f" top-{ic['top_n']} in {sh:.0%}", va="center", fontsize=7, color=P.INK2)
        ax.axvline(0, color=P.INK2, linewidth=1)
        ax.set_xlabel("AUROC drop when shuffled (mean ± SD over 10 splits)")
        ax.set_title(f"{arm}: what the best-member procedure relies on")
        ax.tick_params(axis="y", labelsize=7)
        P.grid(ax, "x")
        st.figure(f"permutation_importance_{arm.replace('+', '_')}", fig, t,
                  f"Permutation importance of the {arm} best-member procedure, top 15 items; correlated genes share their group's value")
    c = coef_s[(coef_s.arm == "C") & (coef_s.procedure == "C|logreg_en")].head(20).iloc[::-1] if not coef_s.empty else pd.DataFrame()
    if len(c):
        fig, ax = P.fig(6.5, 0.3 * len(c) + 1.3)
        y = np.arange(len(c))
        lo, hi = np.exp(c["mean"] - c["sd"].fillna(0)), np.exp(c["mean"] + c["sd"].fillna(0))
        ax.hlines(y, lo, hi, color=P.CAT[0], linewidth=2)
        ax.plot(c.odds_ratio, y, "o", color=P.CAT[0], markersize=6, markeredgecolor=P.SURFACE)
        ax.axvline(1, color=P.INK2, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(c.feature, fontsize=7)
        ax.set_xlabel("Odds ratio of death (per SD for numeric inputs; per level for indicators), mean ± SD over splits")
        ax.set_title("Clinical arm: logistic regression odds ratios (top 20)")
        P.grid(ax, "x")
        st.figure("lr_odds_ratios_C", fig, c, "Odds ratios of the C|logreg_en winners, mean ± SD of the coefficient across splits")
    for arm in ic["arms_full"]:
        feats = top.get(arm, [])
        if not feats:
            continue
        nc = 5
        nr = int(np.ceil(len(feats) / nc))
        fig, axes = P.fig(13, 2.6 * nr, nr, nc, squeeze=False)
        for ax, f in zip(axes.flat, feats):
            d = pdp[(pdp.arm == arm) & (pdp.feature == f)]
            if d.empty:
                ax.axis("off")
                continue
            numeric = bool((d["kind"] == "numeric").all())
            d = d.assign(value_s=d.value.astype(str))
            key = "pos" if numeric else "value_s"         # numeric grids differ by split: align by quantile position
            m = d.groupby(key, sort=True).agg(x=("value", lambda v: pd.to_numeric(v, errors="coerce").mean() if numeric else v.iloc[0]),
                                              mean=("mean_pred", "mean"), min=("mean_pred", "min"), max=("mean_pred", "max"))
            if not numeric:                                # categories: in natural order, as points with ranges
                m = m.sort_index(key=lambda s: [(0, float(v)) if str(v).replace(".", "", 1).isdigit() else (1, str(v)) for v in s])
            xs = m.x.astype(float).to_numpy() if numeric else np.arange(len(m))
            if len(ice):
                ii = ice[(ice.arm == arm) & (ice.feature == f)]
                for _, g in ii.groupby("patient_id"):
                    if numeric:
                        ax.plot(pd.to_numeric(g.value), g.pred, color=P.NEUTRAL, alpha=0.35, linewidth=0.6)
                    else:
                        gs = g.assign(value_s=g.value.astype(str))
                        gs = gs[gs.value_s.isin(m.index)]
                        ax.scatter([list(m.index).index(v) for v in gs.value_s], gs.pred, color=P.NEUTRAL, alpha=0.35, s=4)
            if numeric:
                ax.fill_between(xs, m["min"], m["max"], color=P.CAT[0], alpha=0.2, linewidth=0)
                ax.plot(xs, m["mean"], color=P.CAT[0], marker="o", markersize=3)
            else:                                          # no line between unordered categories
                ax.vlines(xs, m["min"], m["max"], color=P.CAT[0], linewidth=2, alpha=0.5)
                ax.plot(xs, m["mean"], "o", color=P.CAT[0], markersize=5)
            if not numeric:
                ax.set_xticks(list(xs))
                ax.set_xticklabels([str(v) for v in m.x], rotation=45, ha="right", fontsize=6)
            ax.set_title(f, fontsize=8)
            ax.set_ylim(0, 1)
            P.grid(ax)
        for ax in list(axes.flat)[len(feats):]:
            ax.axis("off")
        fig.suptitle(f"{arm}: partial dependence (blue: mean and range over splits; grey: ICE, split r0o0)", x=0.01, ha="left",
                     fontweight="bold", color=P.INK)
        fig.tight_layout()
        st.figure(f"partial_dependence_{arm.replace('+', '_')}", fig, pdp[pdp.arm == arm],
                  f"Partial dependence of the top {len(feats)} features ({arm} best-member procedure) with ICE curves")
        h = hs[hs.arm == arm]
        if len(h):
            mat = pd.DataFrame(0.0, index=feats, columns=feats)
            for r in h.itertuples():
                mat.loc[r.feature_a, r.feature_b] = mat.loc[r.feature_b, r.feature_a] = r[4]   # mean H^2
            fig, ax = P.fig(6.2, 5.4)
            im = ax.imshow(mat.to_numpy(), cmap=P.SEQ, vmin=0, vmax=max(0.05, float(mat.to_numpy().max())))
            ax.set_xticks(range(len(feats)))
            ax.set_xticklabels(feats, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(feats, fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.75, label="Friedman H² (mean over splits)")
            ax.spines[:].set_visible(False)
            ax.set_title(f"{arm}: pairwise interactions among the top features")
            st.figure(f"interactions_{arm.replace('+', '_')}", fig, mat.reset_index().rename(columns={"index": "feature"}),
                      f"Friedman H² between the top {len(feats)} features of the {arm} best-member procedure (0 = additive)")
    mod = modality.groupby(["arm", "procedure", "group"]).importance.mean().reset_index()
    mod = mod[~mod.procedure.str.endswith("|best")]
    arms_m = []
    if len(mod):
        arms_m = [a for a in ["C+RNA", "C+RNA+DNA"] if a in set(mod.arm)]
    if len(mod) and arms_m:
        fig, axes = P.fig(6.5 * len(arms_m), 3.6, 1, len(arms_m), squeeze=False)
        for ax, arm in zip(axes.flat, arms_m):
            m = mod[mod.arm == arm].pivot(index="procedure", columns="group", values="importance").fillna(0)
            m.index = [i.split("|")[1] for i in m.index]
            gl = [g for g in ["all clinical", "treatment group", "all RNA", "all DNA"] if g in m.columns]
            w = 0.8 / len(gl)
            for i, g in enumerate(gl):
                ax.bar(np.arange(len(m)) + i * w, m[g], width=w, color=P.CAT[i], label=g)
            ax.set_xticks(np.arange(len(m)) + 0.4 - w / 2)
            ax.set_xticklabels(m.index, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("AUROC drop when the block is shuffled")
            ax.set_title(f"{arm}: reliance on each data block")
            ax.legend(fontsize=7)
            P.grid(ax)
        fig.tight_layout()
        st.figure("modality_reliance", fig, mod, "Modality reliance per member: AUROC drop when a whole data block is shuffled (mean over splits)")
    if len(lc):
        g = lc.groupby(["arm", "fraction"]).agg(n_train=("n_train", "mean"), cv=("cv_auroc", "mean"), train=("train_auroc", "mean")).reset_index()
        fig, ax = P.fig(6.2, 3.6)
        for i, arm in enumerate(ic["arms_full"]):
            d = g[g.arm == arm]
            ax.plot(d.n_train, d.cv, marker="o", color=P.CAT[i], label=f"{arm} (held out)")
            ax.plot(d.n_train, d.train, linestyle="--", color=P.CAT[i], linewidth=1, label=f"{arm} (training rows)")
        ax.set_xlabel("Training rows used (mean over splits)")
        ax.set_ylabel("AUROC")
        ax.set_title("Learning curves of the best-member procedures")
        ax.legend(fontsize=7, ncol=2)
        P.grid(ax)
        st.figure("learning_curves", fig, g, "Learning curves: held-out AUROC (solid) and training-row AUROC (dashed) by training size")


def _facts(summ, coef_s, shap_s, hs, modality, lc, top, farm_id, n_tasks, secs, ic) -> dict:
    F = {"farm_run": farm_id, "n_refits": n_tasks, "seconds": round(secs), "top": top}
    for arm in ic["arms_full"]:
        t = summ[summ.procedure == f"{arm}|best"].head(5)
        F[f"{arm}_top5"] = [{"item": i, "mean": round(m, 4), "top_share": round(s, 2)} for i, m, s in zip(t.item, t["mean"], t.top_share)]
        h = hs[hs.arm == arm].head(3)
        F[f"{arm}_h_top"] = [{"pair": f"{a} x {b}", "h2": round(v, 3)} for a, b, v in zip(h.feature_a, h.feature_b, h["mean"])]
        l = lc[lc.arm == arm].groupby("fraction").cv_auroc.mean() if len(lc) else pd.Series(dtype=float)
        F[f"{arm}_lc"] = {float(k): round(v, 4) for k, v in l.items()}
    mod = modality.groupby(["arm", "procedure", "group"]).importance.mean()
    F["modality_best"] = {f"{a}|{g}": round(v, 4) for (a, p, g), v in mod.items() if p.endswith("|best")}
    c = coef_s[(coef_s.arm == "C") & (coef_s.procedure == "C|logreg_en")].head(5) if not coef_s.empty else pd.DataFrame()
    F["C_lr_top"] = [{"feature": f, "or": round(o, 3), "sign": round(sc, 2)} for f, o, sc in zip(c.feature, c.odds_ratio, c.sign_consistency)]
    F["cause_or"] = {}
    if not coef_s.empty:
        for (pr, ct), g in coef_s[coef_s.procedure.str.endswith("|logreg_cause")].groupby(["procedure", "contrast"]):
            g = g.assign(a=g["mean"].abs()).sort_values("a", ascending=False).head(6)
            F["cause_or"][f"{pr} | {ct}"] = [{"feature": f, "or": round(o, 3), "sign": round(sc, 2)}
                                             for f, o, sc in zip(g.feature, g.odds_ratio, g.sign_consistency)]
    if len(shap_s):
        agree = []
        for arm in ic["arms_full"]:
            for m in ("logreg_en", "random_forest", "xgboost"):
                a = shap_s[shap_s.procedure == f"{arm}|{m}"].head(ic["top_n"]).item.tolist()
                b = summ[summ.procedure == f"{arm}|{m}"].head(ic["top_n"]).item.tolist()
                if a and b:
                    agree.append({"procedure": f"{arm}|{m}", "shap_vs_permutation_top_overlap": len(set(a) & set(b))})
        F["shap_perm_overlap"] = agree
    return F


def _summary_md(run, cfg, F, ic) -> str:
    L = ["# Gates and importance (step 4) summary", "",
         f"Run `{run.run_id}`, config hash `{cfg.hash}`. Explains the winners chosen inside each outer fold by farm run "
         f"`{F['farm_run']}`: {F['n_refits']} refits, each identical to the farm fit (checked), explained on held-out rows only "
         f"({F['seconds'] // 60} min). Training rows only; the holdout is untouched.", ""]
    for arm in [a for a in ic["arms_full"] if F[f"{a}_top5"]]:
        L.append(f"## {arm} (best-member procedure)")
        L.append("- Top features by permutation importance (AUROC drop; share of splits in the top "
                 f"{ic['top_n']}): " + "; ".join(f"{t['item']} {t['mean']:.3f} ({t['top_share']:.0%})" for t in F[f"{arm}_top5"]) + ".")
        hp = F[f"{arm}_h_top"]
        if hp:
            strongest = hp[0]["h2"]
            L.append(f"- Strongest pairwise interactions (Friedman H²): " + "; ".join(f"{h['pair']} {h['h2']:.3f}" for h in hp)
                     + (" (weak: the model is close to additive)." if strongest < 0.05 else "."))
        lcv = F[f"{arm}_lc"]
        if lcv:
            ks = sorted(lcv)
            gain = lcv[ks[-1]] - lcv[ks[-2]] if len(ks) > 1 else 0
            L.append(f"- Learning curve: held-out AUROC {', '.join(f'{lcv[k]:.3f} at {k:.0%}' for k in ks)}; the last step adds "
                     f"{gain:+.3f} ({'still rising: more data may help' if gain > 0.005 else 'flat: more data of this kind is unlikely to help much'}).")
        L.append("")
    if F["C_lr_top"]:
        L += ["## Logistic regression (clinical arm)",
              "- Largest odds ratios (per SD or per level; sign consistency across splits): "
              + "; ".join(f"{c['feature']} {c['or']:.2f} ({c['sign']:.0%})" for c in F["C_lr_top"]) + ".", ""]
    if F.get("cause_or"):
        L += ["## Cause-specific logistic regression: odds ratios by cause",
              "Contrasts against alive (per SD for scaled inputs, per level for indicators; sign consistency across splits)."]
        for k, rows in F["cause_or"].items():
            L.append(f"- {k}: " + "; ".join(f"{r['feature']} {r['or']:.2f} ({r['sign']:.0%})" for r in rows) + ".")
        L.append("")
    mb = F["modality_best"]
    if mb:
        L += ["## Reliance on each data block (best-member procedures)",
              "- AUROC drop when a whole block is shuffled: " + "; ".join(f"{k} {v:.3f}" for k, v in mb.items()) + ".", ""]
    if F.get("shap_perm_overlap"):
        L += ["## SHAP and permutation importance",
              "- Overlap of the top-" + str(ic["top_n"]) + " lists (SHAP describes the model's own inputs, permutation the raw "
              "columns, so they can differ when the model uses a representation): "
              + "; ".join(f"{a['procedure']} {a['shap_vs_permutation_top_overlap']}/{ic['top_n']}" for a in F["shap_perm_overlap"]) + ".", ""]
    L += ["Importance describes model behaviour, not biology, and is sensitive to correlated features."]
    return "\n".join(L)
