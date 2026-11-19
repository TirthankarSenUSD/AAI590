"""`python run.py ensemble --farm-runs A,B`: fusion and ensemble (build step 5).

For every arm and outer split, each member's in-fold winner (from the farm runs) is refitted on the
inner folds (out-of-fold predictions on the outer training rows) and on the outer training rows
(predictions on the outer test rows; must reproduce the farm's AUROC exactly). On the outer
training rows only: near-duplicate members are merged, the four options are fitted, and a
second-level CV picks one option by the fixed rule. The chosen option's outer score is the
select-and-ensemble headline; every option's outer score is reported too.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from ..core.artifacts import Run
from ..core.config import Config
from ..core.stats import corrected_resampled_ttest
from ..data import load_dataset
from ..eda import plots as P
from ..farm.members import CAUSE, FAMILY, build_candidates, resolve_n_jobs
from ..harness import metrics as M
from ..harness import splits_from_table
from ..harness.floors import design_matrix
from ..importance.stage import _parallel
from ..splits import load_cv_splits, training_ids
from . import stack as S

FIT_SECTIONS = ("columns", "seed", "clinical_types", "groups", "cv", "holdout")   # farm grids checked by exact refits


def _member_task(key, cand, Xtr, yfit_tr, Xte, yte, inner):
    """Inner out-of-fold predictions on the outer training rows + outer test predictions."""
    oof = pd.Series(np.nan, index=Xtr.index)
    for tr, va in inner:
        m = cand.make().fit(Xtr.loc[tr, cand.columns], yfit_tr.loc[tr].to_numpy())
        oof.loc[va] = m.predict_proba(Xtr.loc[va, cand.columns])[:, 1]
    m = cand.make().fit(Xtr[cand.columns], yfit_tr.to_numpy())
    p = m.predict_proba(Xte[cand.columns])[:, 1]
    return {"key": key, "oof": oof, "test": pd.Series(p, index=Xte.index),
            "refit_auroc": float(roc_auc_score(yte, p))}


def run_ensemble(cfg: Config, run: Run, farm_runs: list[str]) -> dict:
    st = run.stage("ensemble")
    ec, h = cfg["ensemble"], cfg["harness"]
    # winners: each (arm, member) procedure from the first listed farm run that has it
    wins = []
    for fr in farm_runs:
        d = cfg.path("outputs") / fr
        old = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        diff = [k for k in FIT_SECTIONS if json.dumps(old.get(k), sort_keys=True) != json.dumps(cfg.raw.get(k), sort_keys=True)]
        if diff:
            raise RuntimeError(f"farm run {fr}: config sections {diff} differ; its winners cannot be rebuilt")
        run.record_input(d / "farm" / "tables" / "procedure_winners.csv", f"winners of farm run {fr}")
        w = pd.read_csv(d / "farm" / "tables" / "procedure_winners.csv")
        wins.append(w.assign(farm_run=fr))
    w = pd.concat(wins, ignore_index=True)
    w["arm"], w["member"] = w.procedure.str.split("|").str[0], w.procedure.str.split("|").str[1]
    # late fusion: each source data type's best-member procedure becomes one "member" of the pseudo-arm
    late = w[w.procedure.isin(ec.get("late_fusion", []))].drop_duplicates(["procedure", "split_id"], keep="first").copy()
    late["member"], late["arm"] = late.arm, "late_fusion"
    w = w[w.arm.isin(ec["arms"]) & w.member.isin(ec["members"])]
    w = w.drop_duplicates(["arm", "member", "split_id"], keep="first")   # first listed run wins
    if "late_fusion" in ec["arms"] and len(late):
        w = pd.concat([w, late], ignore_index=True)
    arms = [a for a in ec["arms"] if a in set(w.arm)]
    full = load_dataset(cfg)
    tr_ids = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows used; holdout rows untouched)")
    run.record_input(cfg.path("cv_splits"), "frozen CV splits")
    ds = full.subset(tr_ids)
    X, y = design_matrix(ds), ds.y
    y_cause = None
    if w.member.isin(list(CAUSE)).any():
        from ..farm.cause import cause_labels
        y_cause = cause_labels(ds.frame)
    splits = {s.split_id: s for s in splits_from_table(load_cv_splits(cfg, tr_ids))}
    src_arms = list(dict.fromkeys(w.winner.str.split("|").str[0]))      # arms the winners were built on
    src_members = list(dict.fromkeys(w.winner.str.split("|").str[1]))
    cands, meta = build_candidates(cfg, members=src_members, arms=src_arms)
    by_name = {c.name: c for c in cands}
    n_jobs = resolve_n_jobs(cfg)

    tasks, seen = [], set()
    for r in w.itertuples():
        if (r.winner, r.split_id) in seen:              # e.g. a late-fusion source equal to an arm's member winner
            continue
        seen.add((r.winner, r.split_id))
        c, sp = by_name[r.winner], splits[r.split_id]
        yfit = (y_cause if c.target == "cause" else y).loc[sp.train]
        tasks.append([(r.winner, r.split_id), c, X.loc[sp.train], yfit, X.loc[sp.test], y.loc[sp.test], sp.inner])
    st.log.info("ensemble: farm runs %s; arms %s; %d member-split refits (4 fits each); n_jobs=%d",
                farm_runs, arms, len(tasks), n_jobs)
    t0 = time.time()
    res = {r["key"]: r for r in _parallel(_member_task, tasks, n_jobs, st.log.info, "member refits")}
    st.log.info("member refits in %.0f s", time.time() - t0)
    w["refit_auroc"] = [res[(a, b)]["refit_auroc"] for a, b in zip(w.winner, w.split_id)]
    w["abs_diff"] = (w.refit_auroc - w.outer_auroc).abs()
    st.table("refit_check", w[["arm", "member", "split_id", "winner", "farm_run", "outer_auroc", "refit_auroc", "abs_diff"]],
             "Refitted member winners reproduce their farm run's outer AUROC (abs_diff must be ~0)")
    if (w.abs_diff > 1e-6).any():
        raise RuntimeError(f"refits do not reproduce the farm runs:\n{w[w.abs_diff > 1e-6].head().to_string()}")

    rows, chosen, weights, merges, div, drop = [], [], [], [], [], []
    for arm in arms:
        wa = w[w.arm == arm]
        late_order = [p.split("|")[0] for p in ec.get("late_fusion", [])]
        complexity = {m: FAMILY[m] if m in FAMILY else late_order.index(m) for m in wa.member.unique()}
        for sid, g in wa.groupby("split_id", sort=False):
            sp = splits[sid]
            members = list(g.member)
            Ptr = pd.DataFrame({m: res[(wn, sid)]["oof"] for m, wn in zip(g.member, g.winner)})
            Pte = pd.DataFrame({m: res[(wn, sid)]["test"] for m, wn in zip(g.member, g.winner)})
            ytr, yte = y.loc[sp.train], y.loc[sp.test]
            fold_of = pd.Series(-1, index=sp.train)
            for j, (_, va) in enumerate(sp.inner):
                fold_of.loc[va] = j
            # agreement merge (on the outer training rows' out-of-fold predictions)
            aucs = {m: S._auc(ytr, Ptr[m]) for m in members}
            groups = S.merge_groups(Ptr, ec["merge_spearman"])
            mw = S.merge_weights(groups, aucs)
            Mtr, Mte = S.apply_merge(Ptr, mw), S.apply_merge(Pte, mw)
            cx = {k: min(complexity[m] for m in k.split("+")) for k in Mtr.columns}
            merges += [{"arm": arm, "split_id": sid, "merged": k, "n_members": len(k.split("+"))} for k in Mtr.columns]
            # choose the option inside the fold (second-level CV), then fit every option on all training rows
            opt, part = S.choose_option(ytr, Mtr, fold_of, cx, h["select_within_se"], ec["stacker_l2"], ec["option_cv_folds"],
                                        cfg.seed)
            chosen.append({"arm": arm, "split_id": sid, "option": opt})
            prev = float(ytr.mean())
            for o in S.OPTIONS:
                f, params = S.fit_option(o, ytr, Mtr, fold_of, cx, h["select_within_se"], ec["stacker_l2"])
                p = f(Mte)
                rows.append({"arm": arm, "candidate": f"{arm}|{o}", "option": o, "split_id": sid,
                             "repeat": sp.repeat, "outer_fold": sp.outer, **M.binary_metrics(yte, p, prev),
                             "member": params.get("member", "")})
                if o == opt:
                    rows.append({"arm": arm, "candidate": f"{arm}|ensemble", "option": f"ensemble ({o})", "split_id": sid,
                                 "repeat": sp.repeat, "outer_fold": sp.outer, **M.binary_metrics(yte, p, prev),
                                 "member": params.get("member", "")})
                if "weights" in params:
                    weights += [{"arm": arm, "split_id": sid, "option": o, "member": k, "weight": v}
                                for k, v in params["weights"].items()]
            # diversity on the outer test rows: prediction and error correlations; drop-one stacker
            R = Pte.rank().corr()
            E = (Pte.sub(yte.to_numpy(), axis=0)).corr()
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    div.append({"arm": arm, "split_id": sid, "a": a, "b": b, "pred_spearman": R.loc[a, b], "error_corr": E.loc[a, b]})
            full_auc = S._auc(yte, S.fit_option("stacker", ytr, Mtr, fold_of, cx, 1, ec["stacker_l2"])[0](Mte))
            for k in Mtr.columns:
                if Mtr.shape[1] > 1:
                    keep = [c for c in Mtr.columns if c != k]
                    f, _ = S.fit_option("stacker", ytr, Mtr[keep], fold_of, cx, 1, ec["stacker_l2"])
                    drop.append({"arm": arm, "split_id": sid, "dropped": k, "auroc_full": full_auc,
                                 "auroc_without": S._auc(yte, f(Mte[keep]))})
    F = pd.DataFrame(rows)
    st.table("option_fold_metrics", F, "Every ensemble option and the in-fold-chosen ensemble, per arm and split (outer test rows)")
    st.table("ensemble_choice", pd.DataFrame(chosen), "Option chosen inside each outer fold (second-level CV, simplest within 1 SE)")
    st.table("merged_members", pd.DataFrame(merges), f"Members after merging (Spearman > {ec['merge_spearman']} on inner predictions)")
    st.table("weights", pd.DataFrame(weights), "Weights of the weighted average and of the non-negative stacker, per split")
    dv = pd.DataFrame(div)
    st.table("diversity", dv, "Prediction (Spearman) and error correlations between members on the outer test rows")
    dr = pd.DataFrame(drop)
    if len(dr):
        dr["delta"] = dr.auroc_without - dr.auroc_full
    st.table("drop_one", dr, "Drop-one test: stacker AUROC without each (merged) member, outer test rows")

    n_tr = float(np.mean([len(s.train) for s in splits.values()]))
    n_te = float(np.mean([len(s.test) for s in splits.values()]))
    summ = []
    for c, g in F.groupby("candidate", sort=False):
        for m in ("auroc", "brier_skill", "cal_slope", "log_loss"):
            v = g[m].dropna().to_numpy()
            t = corrected_resampled_ttest(v, n_tr, n_te)
            summ.append({"arm": g.arm.iloc[0], "candidate": c, "metric": m, "mean": v.mean(), "sd": v.std(ddof=1),
                         "ci_low": t["ci_low"], "ci_high": t["ci_high"]})
    summ = pd.DataFrame(summ)
    st.table("summary", summ, "Ensemble options and the select-and-ensemble procedure: mean and corrected 95% CI over 10 splits")

    def paired(a, b):
        fa, fb = F[F.candidate == a].set_index("split_id").auroc, F[F.candidate == b].set_index("split_id").auroc
        return corrected_resampled_ttest((fa - fb.loc[fa.index]).to_numpy(), n_tr, n_te)
    pairs = []
    for arm in arms:
        for o in S.OPTIONS[1:]:
            pairs.append({"a": f"{arm}|{o}", "b": f"{arm}|best_single", **paired(f"{arm}|{o}", f"{arm}|best_single")})
        if arm != ec["cross_arm_baseline"] and ec["cross_arm_baseline"] in arms:
            b = f"{ec['cross_arm_baseline']}|ensemble"
            pairs.append({"a": f"{arm}|ensemble", "b": b, **paired(f"{arm}|ensemble", b)})
    pairs = pd.DataFrame(pairs)
    st.table("paired_comparisons", pairs, "Paired AUROC differences (corrected resampled t): options vs best single member; arms vs the baseline arm")

    final = _final(F, summ, pairs, arms, ec, n_tr, n_te)
    st.table("final_model", pd.DataFrame([final]), "Final model by the fixed rules (owner, F-055; PLAN section 7 step 4) and its regression floor")
    _figures(st, summ, pd.DataFrame(weights), dv, arms)
    facts = {"final": final, "farm_runs": farm_runs, "arms": arms, "seconds": round(time.time() - t0),
             "choice": pd.DataFrame(chosen).groupby("arm").option.value_counts().to_dict()}
    st.text("ensemble_summary.md", _summary_md(run, cfg, summ, pairs, pd.DataFrame(chosen), dr, final, arms, facts),
            "Auto-generated ensemble summary (computed wording)")
    return facts


def _final(F, summ, pairs, arms, ec, n_tr, n_te) -> dict:
    base = ec["cross_arm_baseline"]
    arm = base if base in arms else arms[0]
    sig = pairs[pairs.b == f"{base}|ensemble"]
    sig = sig[(sig.p < ec["significance"]) & (sig.ci_low > 0)]
    if len(sig):                                         # an arm significantly better than the baseline arm
        a = summ[(summ.metric == "auroc") & summ.candidate.isin(sig.a)].sort_values("mean", ascending=False).iloc[0]
        arm = a.arm
    s = summ[(summ.metric == "auroc") & (summ.arm == arm) & summ.candidate.isin([f"{arm}|{o}" for o in S.OPTIONS])]
    s = s.set_index("candidate").loc[[f"{arm}|{o}" for o in S.OPTIONS]]
    best = s["mean"].max()
    se = s.loc[s["mean"].idxmax(), "sd"] / np.sqrt(10)
    pick = next(c for c in s.index if s.loc[c, "mean"] >= best - se)   # simplest within 1 SE (OPTIONS order)
    v = F[F.candidate == pick].auroc
    nested = summ[(summ.candidate == f"{arm}|ensemble") & (summ.metric == "auroc")].iloc[0]
    return {"arm": arm, "model": pick, "mean_outer_auroc": float(v.mean()), "sd_outer_auroc": float(v.std(ddof=1)),
            "regression_floor": float(v.mean() - 2 * v.std(ddof=1)),
            "select_and_ensemble_auroc": float(nested["mean"]),
            "select_and_ensemble_ci": [float(nested.ci_low), float(nested.ci_high)],
            "rule": "arm: baseline unless an arm is significantly better (p<0.05, CI>0); model: highest mean outer AUROC, "
                    "simplest within 1 SE (best_single < mean < weighted < stacker)"}


def _figures(st, summ, weights, dv, arms):
    a = summ[summ.metric == "auroc"]
    order = [f"{arm}|{o}" for arm in arms for o in list(S.OPTIONS) + ["ensemble"]]
    order = [o for o in order if o in set(a.candidate)]
    t = a.set_index("candidate").loc[order]
    fig, ax = P.fig(7, 0.33 * len(order) + 1.4)
    yv = np.arange(len(order))[::-1]
    for yi, (c, r) in zip(yv, t.iterrows()):
        col = P.CAT[arms.index(r.arm) % len(P.CAT)]
        ax.hlines(yi, r.ci_low, r.ci_high, color=col, linewidth=2)
        ax.plot(r["mean"], yi, "D" if c.endswith("|ensemble") else "o", color=col, markersize=7 if c.endswith("|ensemble") else 5,
                markeredgecolor=P.SURFACE)
    ax.set_yticks(yv)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Outer-fold AUROC (95% corrected CI)")
    ax.set_title("Ensemble options per arm (diamond = option chosen inside each fold)")
    P.grid(ax, "x")
    st.figure("ensemble_options", fig, t.reset_index(), "Outer AUROC of each ensemble option and of the select-and-ensemble procedure")
    sw = weights[weights.option == "stacker"] if len(weights) else weights
    if len(sw):
        m = sw.groupby(["arm", "member"]).weight.mean().unstack(0).fillna(0)
        fig, ax = P.fig(7, 0.32 * len(m) + 1.4)
        h = 0.8 / max(1, m.shape[1])
        for i, arm in enumerate(m.columns):
            ax.barh(np.arange(len(m)) + i * h, m[arm], height=h, color=P.CAT[i % len(P.CAT)], label=arm)
        ax.set_yticks(np.arange(len(m)) + 0.4 - h / 2)
        ax.set_yticklabels(m.index, fontsize=7)
        ax.set_xlabel("Mean non-negative stacker weight (logit scale) over splits")
        ax.set_title("What the stacker uses")
        ax.legend(fontsize=7)
        P.grid(ax, "x")
        st.figure("stacker_weights", fig, m.reset_index(), "Mean stacker weight per (merged) member and arm")
    if len(dv):
        arm = arms[0]
        d = dv[dv.arm == arm].groupby(["a", "b"]).pred_spearman.mean().reset_index()
        names = list(dict.fromkeys(list(d.a) + list(d.b)))
        mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        for r in d.itertuples():
            mat.loc[r.a, r.b] = mat.loc[r.b, r.a] = r.pred_spearman
        fig, ax = P.fig(6, 5.2)
        im = ax.imshow(mat.to_numpy(), cmap=P.SEQ, vmin=min(0.5, float(mat.to_numpy().min())), vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if mat.iat[i, j] > 0.85 else P.INK)
        fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman correlation of predictions")
        ax.spines[:].set_visible(False)
        ax.set_title(f"{arm}: how alike the members' predictions are")
        st.figure(f"member_correlation_{arm.replace('+', '_')}", fig, mat.reset_index().rename(columns={"index": "member"}),
                  f"Mean Spearman correlation of member predictions on the outer test rows ({arm})")


def _summary_md(run, cfg, summ, pairs, chosen, drop, final, arms, facts) -> str:
    a = summ[summ.metric == "auroc"].set_index("candidate")
    L = ["# Fusion and ensemble (step 5) summary", "",
         f"Run `{run.run_id}`, config hash `{cfg.hash}`. Member winners from farm runs {', '.join(f'`{r}`' for r in facts['farm_runs'])}, "
         f"each refitted exactly (checked). Every choice is made on the outer training rows; scores are outer-fold. "
         f"Training rows only; the holdout is untouched.", "",
         "| Arm | Best single | Mean | Weighted | Stacker | **Select-and-ensemble** | Option chosen in folds |", "|---|---|---|---|---|---|---|"]
    for arm in arms:
        cells = [f"{a.loc[f'{arm}|{o}', 'mean']:.3f}" for o in S.OPTIONS]
        e = a.loc[f"{arm}|ensemble"]
        ch = chosen[chosen.arm == arm].option.value_counts()
        L.append(f"| {arm} | " + " | ".join(cells) + f" | **{e['mean']:.3f}** ({e.ci_low:.3f}-{e.ci_high:.3f}) | "
                 + ", ".join(f"{k} {v}" for k, v in ch.items()) + " |")
    L.append("")
    for r in pairs.itertuples():
        L.append(f"- {r.a} minus {r.b}: {r.mean_diff:+.4f} AUROC (95% CI {r.ci_low:+.4f} to {r.ci_high:+.4f}, p = {r.p:.3f}"
                 f"{'; significant' if r.p < 0.05 else ''}).")
    if len(drop):
        d = drop.groupby(["arm", "dropped"]).delta.mean().reset_index().sort_values("delta")
        worst = d.groupby("arm").first()
        L.append("- Drop-one (stacker): largest AUROC loss when removing a member: "
                 + "; ".join(f"{arm} {r.dropped} {r.delta:+.4f}" for arm, r in worst.iterrows()) + ".")
    f = final
    L += ["", "## Final model (fixed rules)",
          f"- Arm: **{f['arm']}**; model: **{f['model'].split('|')[1]}**; mean outer AUROC {f['mean_outer_auroc']:.3f} "
          f"(SD {f['sd_outer_auroc']:.3f}); the honest nested estimate of this select-and-ensemble procedure is "
          f"{f['select_and_ensemble_auroc']:.3f} ({f['select_and_ensemble_ci'][0]:.3f}-{f['select_and_ensemble_ci'][1]:.3f}).",
          f"- Regression floor (mean - 2 SD of outer-fold AUROC): **{f['regression_floor']:.3f}**. Later runs of the final "
          "model must not fall below it.", f"- Rule: {f['rule']}.", ""]
    return "\n".join(L)
