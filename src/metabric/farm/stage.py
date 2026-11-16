"""`python run.py farm`: score every farm configuration on the frozen nested CV, then evaluate the
select-the-best procedures (per arm x member, and per arm across members) on the outer folds."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..core.artifacts import Run
from ..core.config import Config
from ..data import load_dataset
from ..eda import plots as P
from ..harness import Harness, procedure, procedure_preds, reliability, splits_from_table
from ..harness.floors import design_matrix, floor_candidates
from ..harness.stage import treatment_groups
from ..splits import load_cv_splits, training_ids
from .members import build_candidates, resolve_n_jobs


def _as_folds(sel: pd.DataFrame, name: str) -> pd.DataFrame:
    """Procedure rows in the same shape as candidate fold rows, so the harness summaries apply."""
    f = sel.rename(columns=lambda c: c[5:] if c.startswith("proc_") else c).copy()
    f = f.loc[:, ~f.columns.duplicated()]
    f["candidate"] = name
    return f


def run_farm(cfg: Config, run: Run, members=None, arms=None) -> dict:
    st = run.stage("farm")
    rng = np.random.default_rng(cfg.seed)
    full = load_dataset(cfg)
    run.record_input(cfg.path("data"), "dataset")
    tr = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows used; holdout rows untouched)")
    ds = full.subset(tr)
    run.record_input(cfg.path("cv_splits"), "frozen CV splits")
    H = Harness(cfg, design_matrix(ds), ds.y, splits_from_table(load_cv_splits(cfg, tr)))
    n_jobs = resolve_n_jobs(cfg)
    cands, meta = build_candidates(cfg, members, arms)
    floors = floor_candidates(cfg)
    arms = arms or list(cfg["farm"]["arms"])
    members = members or cfg["farm"]["members"]
    st.log.info("farm: %d configurations (%s) + %d floors on %d splits, n_jobs=%d", len(cands),
                {a: sum(m.arm == a for m in meta.values()) for a in arms}, len(floors), len(H.splits), n_jobs)
    t0 = time.time()
    res = H.evaluate(cands + floors, n_jobs=n_jobs)
    st.log.info("all configurations scored in %.0f s", time.time() - t0)
    folds = res.folds
    folds["arm"] = folds.candidate.map(lambda n: meta[n].arm if n in meta else "floor")
    folds["member"] = folds.candidate.map(lambda n: meta[n].member if n in meta else n)
    folds["representation"] = folds.candidate.map(lambda n: meta[n].rep if n in meta else "")
    st.table("configuration_fold_metrics", folds,
             "Every configuration and floor on every split (inner statistics drive selection; outer metrics are not a headline)")
    ws = cfg["harness"]["select_within_se"]

    # --- procedures --------------------------------------------------------------------------
    procs, sels = {}, []
    for arm in arms:
        for member in members:
            names = [n for n, m in meta.items() if m.arm == arm and m.member == member]
            procs[f"{arm}|{member}"] = names
        procs[f"{arm}|best"] = [n for n, m in meta.items() if m.arm == arm]
    pf_parts, pred_parts = [], []
    for pname, names in procs.items():
        sel = procedure(folds, names, ws)
        sel.insert(0, "procedure", pname)
        sel["winner_member"] = sel.winner.map(lambda n: meta[n].member)
        sel["winner_representation"] = sel.winner.map(lambda n: meta[n].rep)
        sels.append(sel)
        pf_parts.append(_as_folds(sel.drop(columns="procedure"), pname))
        pred_parts.append(procedure_preds(res.preds, sel, pname))
    sel_all = pd.concat(sels, ignore_index=True)
    st.table("procedure_winners", sel_all, "Select-the-best procedures: winner chosen by the inner-fold rule in every outer split, "
             "with its outer metrics (the procedure's score) and optimism")
    floor_f = folds[folds.arm == "floor"].drop(columns=["arm", "member", "representation"])
    pf = pd.concat(pf_parts + [floor_f], ignore_index=True)
    pp = pd.concat(pred_parts + [res.preds[res.preds.candidate.isin([f.name for f in floors])]], ignore_index=True)
    st.table("procedure_oof_predictions", pp, "Out-of-fold predictions of every procedure and floor")

    summ = H.summarise(pf)
    st.table("procedure_summary", summ, "Procedures and floors: mean, SD and corrected-t 95% CI over the outer splits")
    stab = (sel_all.groupby("procedure").agg(n_splits=("winner", "size"), n_distinct_winners=("winner", "nunique"),
                                             top_winner=("winner", lambda s: s.value_counts().index[0]),
                                             top_winner_share=("winner", lambda s: s.value_counts().iloc[0] / len(s)),
                                             top_member=("winner_member", lambda s: s.value_counts().index[0]),
                                             top_member_share=("winner_member", lambda s: s.value_counts().iloc[0] / len(s)),
                                             top_representation=("winner_representation", lambda s: s.value_counts().index[0]))
            .reset_index())
    st.table("winner_stability", stab, "Winner stability: how often the same configuration / member / representation wins across splits")
    gate = H.overfit_gate(pf)
    st.table("overfit_gate", gate, "Overfit gate on the procedures (winner's train AUROC minus outer AUROC; calibration slope)")

    pairs = []
    def pair(a, b, label):
        sub = pf[pf.candidate.isin([a, b])]
        p = H.paired(sub, metrics=("auroc", "brier", "log_loss"))
        if not p.empty and p.a.iloc[0] != a:          # orient as a - b
            p = p.assign(a=a, b=b, mean_diff=-p.mean_diff, t=-p.t, ci_low=-p.ci_high, ci_high=-p.ci_low)
        pairs.append(p.assign(comparison=label))
    for pname in procs:
        pair(pname, "age_only", "procedure vs age-only floor")
    for m in members + ["best"]:
        if "C+RNA" in arms and "C" in arms:
            pair(f"C+RNA|{m}", f"C|{m}", "adding RNA (C+RNA vs C)")
        if "strict_clinical" in arms and "C" in arms:
            pair(f"strict_clinical|{m}", f"C|{m}", "dropping molecular-derived clinical columns (strict vs C)")
    pair_t = pd.concat(pairs, ignore_index=True)
    st.table("paired_comparisons", pair_t, "Paired differences a - b on identical splits, corrected resampled t-test")

    best = [f"{a}|best" for a in arms]
    sub = H.subgroups(pp[pp.candidate.isin(best)], treatment_groups(ds.frame), rng)
    st.table("subgroup_metrics", sub, "Arm procedures within treatment subgroups (out-of-fold, averaged over repeats; bootstrap 95% CI)")
    rel = reliability(pp[pp.candidate.isin(best + ["age_only"])], cfg["harness"]["reliability_bins"])
    st.table("reliability", rel, "Reliability of the arm procedures and the age-only floor (quantile deciles)")
    rf = H.regression_floor(pf)
    st.table("regression_floor", rf, "Regression floor per procedure: mean - 2 SD of outer-fold AUROC (provisional until the final model)")

    perm_arm = "C" if "C" in arms else arms[0]
    perm_names = procs[f"{perm_arm}|best"]
    perm_cands = [c for c in cands if c.name in set(perm_names)]
    def proc_fn(y):
        r = H.evaluate(perm_cands, y=y, select=True, light=True, n_jobs=n_jobs)
        return float(r.selection.outer_auroc.mean())
    t1 = time.time()
    perm, perm_s = H.permutation_test(proc_fn, cfg["harness"]["n_permutations"], rng)
    st.log.info("permutation test (%s|best, %d shuffles) in %.0f s", perm_arm, len(perm), time.time() - t1)
    st.table("permutation_test", perm, f"Label-permutation test of the full {perm_arm}|best procedure (selection inside every outer fold)")
    st.table("permutation_summary", pd.DataFrame([{**perm_s, "procedure": f"{perm_arm}|best"}]), "Permutation test summary (must be at chance)")

    _figures(st, summ, sel_all, rel, arms, members)
    facts = _facts(summ, pair_t, gate, stab, perm_s, perm_arm, arms, members, len(cands), n_jobs, time.time() - t0)
    st.text("farm_summary.md", _summary_md(run, cfg, facts, arms, members), "Auto-generated farm summary (computed wording)")
    if not perm_s["at_chance"]:
        raise RuntimeError(f"permutation test not at chance: {perm_s}")
    return facts


def _figures(st, summ, sel_all, rel, arms, members):
    a = summ[summ.metric == "auroc"].set_index("candidate")
    order = [f"{arm}|{m}" for arm in arms for m in members + ["best"]]
    order = [o for o in order if o in a.index]
    fig, ax = P.fig(7, 0.33 * len(order) + 1.4)
    y = np.arange(len(order))[::-1]
    cols = [P.CAT[arms.index(o.split("|")[0])] for o in order]
    for yi, o, c in zip(y, order, cols):
        r = a.loc[o]
        ax.hlines(yi, r.ci_low, r.ci_high, color=c, linewidth=2)
        ax.plot(r["mean"], yi, "D" if o.endswith("|best") else "o", color=c, markersize=7 if o.endswith("|best") else 5,
                markeredgecolor=P.SURFACE, markeredgewidth=1.2)
    ax.axvline(a.loc["age_only", "mean"], color=P.INK2, linestyle="--", linewidth=1)
    ax.text(a.loc["age_only", "mean"], len(order) - 0.4, " age-only floor", color=P.INK2, fontsize=7, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Outer-fold AUROC of the select-the-best procedure (95% corrected CI)")
    ax.set_title("Model farm (3a): procedures by arm and member")
    P.grid(ax, "x")
    st.figure("farm_procedure_auroc", fig, a.loc[order + ["age_only"]].reset_index(),
              "Outer-fold AUROC of each select-the-best procedure (diamond = best member chosen inside the fold); dashed = age-only floor")
    wins = (sel_all[sel_all.procedure.str.endswith("|best")].groupby(["procedure", "winner_member"]).size()
            .unstack(fill_value=0).reindex(columns=members, fill_value=0))
    fig, ax = P.fig(6.4, 0.5 * len(wins) + 1.3)
    left = np.zeros(len(wins))
    for i, m in enumerate(members):
        ax.barh(wins.index, wins[m], left=left, color=P.CAT[i], label=m, height=0.6, edgecolor=P.SURFACE, linewidth=2)
        left += wins[m].to_numpy()
    ax.set_xlabel("Outer splits won (of 10)")
    ax.set_title("Which member wins inside each outer fold")
    ax.legend(loc="lower right", ncol=2)
    P.grid(ax, "x")
    st.figure("farm_winner_members", fig, wins.reset_index(), "Member chosen by the inner-fold rule in each outer split, per arm")
    fig, ax = P.fig(5, 4.6)
    ax.plot([0, 1], [0, 1], color=P.INK2, linestyle="--", linewidth=1)
    names = list(dict.fromkeys(rel.candidate))
    for i, n in enumerate(names):
        r = rel[rel.candidate == n]
        ax.plot(r.mean_pred, r.obs_rate, marker="o", markersize=4, color=P.CAT[i] if n != "age_only" else P.NEUTRAL, label=n)
    ax.set_xlabel("Mean predicted risk of death (decile)")
    ax.set_ylabel("Observed death rate")
    ax.set_title("Calibration of the arm procedures")
    ax.legend(loc="upper left", fontsize=7)
    P.grid(ax, "both")
    st.figure("farm_reliability", fig, rel, "Reliability (quantile deciles) of the arm procedures and the age-only floor; dashed = perfect")


def _facts(summ, pairs, gate, stab, perm, perm_arm, arms, members, n_cands, n_jobs, secs) -> dict:
    g = lambda c, m, k="mean": float(summ[(summ.candidate == c) & (summ.metric == m)][k].iloc[0])  # noqa: E731
    F = {"n_configurations": n_cands, "n_jobs": n_jobs, "seconds": round(secs), "floor_age": round(g("age_only", "auroc"), 4)}
    for p in [f"{a}|{m}" for a in arms for m in members + ["best"]]:
        F[p] = {"auroc": round(g(p, "auroc"), 4), "ci": [round(g(p, "auroc", "ci_low"), 4), round(g(p, "auroc", "ci_high"), 4)],
                "cal_slope": round(g(p, "cal_slope"), 3), "brier_skill": round(g(p, "brier_skill"), 4),
                "optimism": round(g(p, "optimism"), 4)}
    F["pairs"] = pairs[pairs.metric == "auroc"][["comparison", "a", "b", "mean_diff", "ci_low", "ci_high", "p"]].round(4).to_dict("records")
    F["gate_flags"] = gate[gate.status == "flag"].candidate.tolist()
    F["stability"] = stab[stab.procedure.str.endswith("|best")][["procedure", "top_member", "top_member_share", "top_representation"]].to_dict("records")
    F["permutation"] = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in perm.items()}, "procedure": f"{perm_arm}|best"}
    return F


def _summary_md(run, cfg, F, arms, members) -> str:
    L = ["# Model farm (step 3a) summary", "",
         f"Run `{run.run_id}`, config hash `{cfg.hash}`, seed {cfg.seed}. Training rows only; the holdout is untouched. "
         f"{F['n_configurations']} configurations ({', '.join(members)}) on {len(arms)} arms, scored on 10 frozen outer splits "
         f"with 3 inner folds each, n_jobs = {F['n_jobs']} ({F['seconds'] // 60} min). Each procedure picks its winner inside every "
         "outer training fold by the fixed rule (best inner AUROC; simplest within 1 SE); its score is the winner's outer score.", "",
         "| Procedure | AUROC (95% CI) | Calibration slope | Brier skill | Optimism |", "|---|---|---|---|---|"]
    for a in arms:
        for m in members + ["best"]:
            r = F[f"{a}|{m}"]
            name = f"**{a} best member**" if m == "best" else f"{a} {m}"
            L.append(f"| {name} | {r['auroc']:.3f} ({r['ci'][0]:.3f}-{r['ci'][1]:.3f}) | {r['cal_slope']:.2f} | "
                     f"{r['brier_skill']:.3f} | {r['optimism']:+.3f} |")
    L += [f"| Age-only floor | {F['floor_age']:.3f} | | | |", ""]
    sig = lambda p: "significant" if p["p"] < 0.05 else "not significant"  # noqa: E731
    for p in F["pairs"]:
        if p["a"].endswith("|best"):
            L.append(f"- {p['comparison']}: {p['a']} minus {p['b']} = {p['mean_diff']:+.3f} AUROC "
                     f"(95% CI {p['ci_low']:+.3f} to {p['ci_high']:+.3f}, p = {p['p']:.3f}; {sig(p)}).")
    for s in F["stability"]:
        L.append(f"- {s['procedure']}: `{s['top_member']}` wins {s['top_member_share']:.0%} of outer splits "
                 f"(most frequent RNA representation: {s['top_representation']}).")
    P_ = F["permutation"]
    L += [f"- Label-permutation test of the full {P_['procedure']} procedure ({P_['n_permutations']} shuffles): mean AUROC "
          f"{P_['perm_mean']:.3f} (max {P_['perm_max']:.3f}) against {P_['observed_auroc']:.3f}; "
          f"{'at chance, as required' if P_['at_chance'] else 'NOT at chance: investigate'}.",
          f"- Overfit gate: {'no procedure flagged' if not F['gate_flags'] else 'flagged: ' + ', '.join(F['gate_flags'])}.", ""]
    return "\n".join(L)
