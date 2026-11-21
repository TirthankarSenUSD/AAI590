"""`python run.py agreement --farm-runs A,B,C --importance-run R`: model-agreement analysis (step 6).

Reads per-patient out-of-fold predictions and winners from farm runs and permutation importance
from an importance run (all recorded as inputs). Adds two small refit analyses:
- leave-one-cohort-out: do the consensus features stay important in an unseen cohort?
- cause-specific risk breakdown: P(cancer death) and P(other-cause death) per patient, out of fold.
Training rows only; the holdout is untouched.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import yaml

from ..core.artifacts import Run
from ..core.config import Config
from ..data import load_dataset
from ..eda import plots as P
from ..farm.members import CAUSE, build_candidates, resolve_n_jobs
from ..farm.preprocess import column_groups
from ..harness import splits_from_table
from ..harness.floors import design_matrix
from ..importance import explain as E
from ..importance.stage import _parallel
from ..splits import load_cv_splits, training_ids
from . import analysis as A

FIT_SECTIONS = ("columns", "seed", "clinical_types", "groups", "cv", "holdout")


def _loco_task(key, cand, Xtr, ytr, Xte, yte, cols, n_repeats, seed):
    pipe = cand.make().fit(Xtr[cand.columns], ytr.to_numpy())
    imp = E.permutation_importance(pipe, Xte[cand.columns], yte, {c: [c] for c in cols}, n_repeats, [seed, len(Xte)])
    return {"key": key, "imp": imp}


def _cause_task(key, cand, Xtr, ycause_tr, Xte):
    m = cand.make().fit(Xtr[cand.columns], ycause_tr.to_numpy())
    pc = m.predict_cause_proba(Xte[cand.columns])
    return {"key": key, "p": pd.DataFrame(pc, index=Xte.index, columns=["p_alive", "p_cancer_death", "p_other_death"])}


def run_agreement(cfg: Config, run: Run, farm_runs: list[str], importance_run: str) -> dict:
    st = run.stage("agreement")
    ac = cfg["agreement"]
    out = cfg.path("outputs")
    preds, wins = [], []
    for fr in farm_runs:
        d = out / fr
        old = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        diff = [k for k in FIT_SECTIONS if json.dumps(old.get(k), sort_keys=True) != json.dumps(cfg.raw.get(k), sort_keys=True)]
        if diff:
            raise RuntimeError(f"farm run {fr}: config sections {diff} differ")
        for f in ("procedure_oof_predictions.csv", "procedure_winners.csv"):
            run.record_input(d / "farm" / "tables" / f, f"farm run {fr}")
        preds.append(pd.read_csv(d / "farm" / "tables" / "procedure_oof_predictions.csv").assign(farm_run=fr))
        wins.append(pd.read_csv(d / "farm" / "tables" / "procedure_winners.csv").assign(farm_run=fr))
    pr = pd.concat(preds, ignore_index=True)
    pr = pr[pr.candidate.str.contains("|", regex=False) & ~pr.candidate.str.contains("rep:|@", regex=True)]
    first = pr.drop_duplicates("candidate")[["candidate", "farm_run"]]          # first listed run wins per procedure
    pr = pr.merge(first, on=["candidate", "farm_run"])
    wi = pd.concat(wins, ignore_index=True)
    wi = wi.merge(wi.drop_duplicates("procedure")[["procedure", "farm_run"]], on=["procedure", "farm_run"])
    idir = out / importance_run / "importance" / "tables"
    for f in ("permutation_summary.csv", "modality_importance.csv"):
        run.record_input(idir / f, f"importance run {importance_run}")
    summ = pd.read_csv(idir / "permutation_summary.csv")
    modality = pd.read_csv(idir / "modality_importance.csv")
    full = load_dataset(cfg)
    tr_ids = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows used; holdout rows untouched)")
    run.record_input(cfg.path("cv_splits"), "frozen CV splits")
    ds = full.subset(tr_ids)
    X, y = design_matrix(ds), ds.y
    splits = {s.split_id: s for s in splits_from_table(load_cv_splits(cfg, tr_ids))}
    n_tr = float(np.mean([len(s.train) for s in splits.values()]))
    n_te = float(np.mean([len(s.test) for s in splits.values()]))
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    facts = {"farm_runs": farm_runs, "importance_run": importance_run}

    # --- items 1-3: importance agreement, overlap, prediction correlation, stability ---------------
    mats, overlaps, pcors, stab, dis_q, dis_v, rob = {}, [], [], [], [], {}, []
    for arm in ac["arms"]:
        imp_procs = [f"{arm}|{m}" for m in ac["members"] if (summ.procedure == f"{arm}|{m}").any()]
        pred_procs = [f"{arm}|{m}" for m in ac["members"] if (pr.candidate == f"{arm}|{m}").any()]
        if imp_procs:
            mat, cons = A.importance_matrix(summ, imp_procs, ac["top_n"])
            mats[arm] = (mat, cons)
            st.table(f"importance_matrix_{arm.replace('+', '_')}", mat.reset_index(), f"{arm}: mean permutation importance, model x feature")
            st.table(f"consensus_{arm.replace('+', '_')}", cons, f"{arm}: consensus strip (mean rank; share of models with the feature in their top {ac['top_n']})")
            overlaps.append(A.rank_overlap(summ, imp_procs).assign(arm=arm))
            s = summ[summ.procedure.isin(imp_procs)].sort_values("mean", ascending=False).groupby("procedure").head(ac["top_n"])
            stab.append(s.groupby("procedure").top_share.mean().rename("mean_top_share_of_top").reset_index().assign(arm=arm))
        if len(pred_procs) > 1:
            pcors.append(A.prediction_correlation(pr, pred_procs).assign(arm=arm))
            q, v = A.disagreement(pr, pred_procs, ac["n_boot"], rng)
            dis_q.append(q.assign(arm=arm))
            dis_v[arm] = v
            rob.append(A.robustness(pr, pred_procs + [f"{arm}|best"] if (pr.candidate == f"{arm}|best").any() else pred_procs,
                                    {"mean of members": pred_procs}).assign(arm=arm))
    ov = pd.concat(overlaps, ignore_index=True) if overlaps else pd.DataFrame()
    pc = pd.concat(pcors, ignore_index=True) if pcors else pd.DataFrame()
    st.table("rank_overlap", ov, "Pairwise agreement of feature importance: Jaccard of top-10 and top-20 lists, Spearman of full vectors")
    st.table("prediction_correlation", pc, "Pairwise Spearman correlation of out-of-fold predictions (mean over splits)")
    st.table("stability", pd.concat(stab, ignore_index=True) if stab else pd.DataFrame(),
             "Feature stability: mean share of the 10 splits in which each model's top features stay in its top 10")
    st.table("disagreement_quintiles", pd.concat(dis_q, ignore_index=True) if dis_q else pd.DataFrame(),
             "Patients by member disagreement (SD of logits): Brier, expected Brier p(1-p), excess error, AUROC")
    st.table("disagreement_verdict", pd.DataFrame([{**v, "arm": a} for a, v in dis_v.items()]),
             "Does disagreement predict error beyond the predicted risk? Spearman(SD, excess error) with bootstrap CI")
    rb = pd.concat(rob, ignore_index=True) if rob else pd.DataFrame()
    st.table("robustness", rb, "Prediction stability: |p(repeat 0) - p(repeat 1)| per patient (different training sets)")
    facts["disagreement"] = dis_v

    # --- modality reliance (item 7) ------------------------------------------------------------------
    mod = modality.groupby(["arm", "procedure", "group"]).importance.mean().reset_index()
    st.table("modality_reliance", mod, "AUROC drop when a whole data block is shuffled, per model (from the importance run)")

    # --- item 4: leave-one-cohort-out check of consensus features ------------------------------------
    arm = ac["loco_arm"]
    loco = pd.DataFrame()
    if arm in mats:
        cons = mats[arm][1]
        consensus = cons.item.head(ac["consensus_k"]).tolist()
        g = column_groups(cfg, arm)
        cols = g["numeric"] + g["binary"] + g["categorical"]
        cands, meta = build_candidates(cfg, members=ac["loco_members"], arms=[arm])
        by = {c.name: c for c in cands}
        coh = ds.frame["cohort"].astype(str)
        tasks = []
        for m in ac["loco_members"]:
            ww = wi[wi.procedure == f"{arm}|{m}"]
            if ww.empty:
                continue
            name = ww.winner.value_counts().index[0]                  # the member's most frequent winner
            for k in sorted(coh.unique()):
                te, trn = coh.index[coh == k], coh.index[coh != k]
                tasks.append([(m, k), by[name], X.loc[trn], y.loc[trn], X.loc[te], y.loc[te], cols, ac["n_repeats"], cfg.seed])
        res = _parallel(_loco_task, tasks, resolve_n_jobs(cfg), st.log.info, "leave-one-cohort-out importance")
        parts = []
        for r in res:
            d = r["imp"].assign(member=r["key"][0], held_out_cohort=r["key"][1])
            d["rank"] = d.importance.rank(ascending=False, method="min")
            parts.append(d)
        loco = pd.concat(parts, ignore_index=True)
        loco["consensus"] = loco.group.isin(consensus)
        st.table("loco_importance", loco, f"{arm}: permutation importance on a held-out cohort (model trained on the other cohorts)")
        ls = loco[loco.consensus].groupby(["group", "held_out_cohort"])["rank"].mean().unstack()
        st.table("loco_consensus_ranks", ls.reset_index(), f"Mean rank (across models) of the {len(consensus)} consensus features in each held-out cohort")
        facts["loco"] = {"consensus": consensus, "worst_rank": float(ls.max().max()), "mean_rank": float(ls.mean().mean()),
                         "share_in_top10": float((loco[loco.consensus]["rank"] <= 10).mean())}

    # --- ablation (omics question) ---------------------------------------------------------------------
    auc = wi.pivot_table(index="split_id", columns="procedure", values="outer_auroc")
    steps = [tuple(s) for s in ac["ablation"]]
    ab = A.ablation(auc, steps, n_tr, n_te)
    st.table("ablation", ab, "Ablation on identical splits: each arm's best-member procedure with corrected 95% CI and its paired "
             "difference from the previous step")
    facts["ablation"] = ab.round(4).to_dict("records")

    # --- cause-specific risk breakdown ----------------------------------------------------------
    cb = pd.DataFrame()
    cproc = ac["cause_procedure"]
    ww = wi[wi.procedure == cproc]
    if len(ww):
        from ..farm.cause import cause_labels
        yc = cause_labels(ds.frame)
        carm = cproc.split("|")[0]
        cands, meta = build_candidates(cfg, members=[cproc.split("|")[1]], arms=[carm])
        by = {c.name: c for c in cands}
        tasks = [[(r.winner, r.split_id), by[r.winner], X.loc[splits[r.split_id].train], yc.loc[splits[r.split_id].train],
                  X.loc[splits[r.split_id].test]] for r in ww.itertuples()]
        res = _parallel(_cause_task, tasks, resolve_n_jobs(cfg), st.log.info, "cause-specific risk")
        cb = pd.concat([r["p"].assign(split_id=r["key"][1]) for r in res]).rename_axis("patient_id").reset_index()
        cb = cb.merge(yc.rename("outcome").reset_index().rename(columns={"index": "patient_id"}), on="patient_id")
        st.table("cause_breakdown", cb, f"Out-of-fold P(cancer death) and P(other-cause death) per patient and split ({cproc})")
        pt = cb.groupby("patient_id")[["p_cancer_death", "p_other_death"]].mean().join(yc.rename("outcome"))
        pt["p_death"] = pt.p_cancer_death + pt.p_other_death
        hi = pt[pt.p_death >= pt.p_death.quantile(0.75)]
        facts["cause"] = {"n_patients": len(pt), "corr_cancer_other": float(pt[["p_cancer_death", "p_other_death"]].corr().iloc[0, 1]),
                          "high_risk_mostly_other": float((hi.p_other_death > hi.p_cancer_death).mean()),
                          "auroc_cancer_score_for_cancer_death": float(_auc((pt.outcome == 1), pt.p_cancer_death)),
                          "auroc_other_score_for_other_death": float(_auc((pt.outcome == 2), pt.p_other_death))}
        st.table("cause_patient_summary", pt.reset_index(), "Per-patient cause-specific risk (mean over the two repeats)")

    _figures(st, mats, ov, pc, pd.concat(dis_q, ignore_index=True) if dis_q else pd.DataFrame(), ab, cb, ac)
    facts["seconds"] = round(time.time() - t0)
    st.text("agreement_summary.md", _summary_md(run, cfg, facts, ov, pc, rb, mod, ac), "Auto-generated agreement summary (computed wording)")
    return facts


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def _figures(st, mats, ov, pc, dq, ab, cb, ac):
    for arm, (mat, cons) in mats.items():
        feats = list(mat.columns[:15])
        M = mat[feats]
        fig, axes = P.fig(min(14, 1.0 + 0.55 * len(feats)), 0.42 * len(M) + 2.4, 2, 1, gridspec_kw={"height_ratios": [len(M), 1.2]})
        im = axes[0].imshow(M.to_numpy(), cmap=P.SEQ, aspect="auto", vmin=0)
        axes[0].set_yticks(range(len(M)))
        axes[0].set_yticklabels([p.split("|")[1] for p in M.index], fontsize=7)
        axes[0].set_xticks([])
        axes[0].set_title(f"{arm}: which features each model relies on (permutation importance)")
        fig.colorbar(im, ax=axes[0], shrink=0.8, label="AUROC drop")
        c = cons.set_index("item").loc[feats]
        axes[1].bar(range(len(feats)), c.share_in_top, color=P.CAT[0], width=0.7)
        axes[1].set_xticks(range(len(feats)))
        axes[1].set_xticklabels(feats, rotation=60, ha="right", fontsize=7)
        axes[1].set_ylabel(f"share in top {ac['top_n']}", fontsize=7)
        axes[1].set_ylim(0, 1)
        for ax in axes:
            ax.spines[:].set_visible(False)
        fig.tight_layout()
        st.figure(f"importance_heatmap_{arm.replace('+', '_')}", fig, M.reset_index(),
                  f"{arm}: model x feature permutation importance (top 15 features) with a consensus strip")
    if len(ov) and len(pc):
        for arm in ov.arm.unique():
            o, p = ov[ov.arm == arm], pc[pc.arm == arm]
            names = list(dict.fromkeys(list(o.a) + list(o.b)))
            J = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
            for r in o.itertuples():
                J.loc[r.a, r.b] = J.loc[r.b, r.a] = r.jaccard_top10
            C = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
            for r in p.itertuples():
                if r.a in names and r.b in names:
                    C.loc[r.a, r.b] = C.loc[r.b, r.a] = r.pred_spearman
            fig, axes = P.fig(11, 4.8, 1, 2)
            lab = [n.split("|")[1] for n in names]
            for ax, Mx, t, vmin in ((axes[0], J, "Top-10 feature overlap (Jaccard)", 0), (axes[1], C, "Prediction correlation (Spearman)", 0.5)):
                im = ax.imshow(Mx.to_numpy(), cmap=P.SEQ, vmin=vmin, vmax=1)
                ax.set_xticks(range(len(lab)))
                ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=7)
                ax.set_yticks(range(len(lab)))
                ax.set_yticklabels(lab, fontsize=7)
                for i in range(len(lab)):
                    for j in range(len(lab)):
                        ax.text(j, i, f"{Mx.iat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if Mx.iat[i, j] > 0.8 else P.INK)
                ax.set_title(t, fontsize=9)
                ax.spines[:].set_visible(False)
            fig.suptitle(f"{arm}: do the models use the same features, and do they predict alike?", x=0.01, ha="left",
                         fontweight="bold", color=P.INK)
            fig.tight_layout()
            st.figure(f"agreement_matrices_{arm.replace('+', '_')}", fig, o.merge(p, on=["a", "b", "arm"], how="outer"),
                      f"{arm}: Jaccard overlap of top-10 features beside the correlation of predictions")
    if len(dq):
        fig, ax = P.fig(6.2, 3.6)
        for i, arm in enumerate(dq.arm.unique()):
            d = dq[dq.arm == arm]
            ax.plot(d.quintile.astype(int), d.excess, marker="o", color=P.CAT[i], label=arm)
        ax.axhline(0, color=P.INK2, linestyle="--", linewidth=1)
        ax.set_xlabel("Disagreement quintile (1 = models agree most)")
        ax.set_ylabel("Excess error (Brier - p(1-p))")
        ax.set_title("Is error higher where the models disagree?")
        ax.legend(fontsize=7)
        P.grid(ax)
        st.figure("disagreement_vs_error", fig, dq, "Excess error by quintile of member disagreement; above 0 = worse than the predicted risk implies")
    if len(ab):
        fig, ax = P.fig(6.6, 0.45 * len(ab) + 1.3)
        yv = np.arange(len(ab))[::-1]
        ax.hlines(yv, ab.ci_low, ab.ci_high, color=P.CAT[0], linewidth=2)
        ax.plot(ab.mean_auroc, yv, "o", color=P.CAT[0], markersize=6, markeredgecolor=P.SURFACE)
        labels = [f"{r.procedure}" + (f"  ({r['diff']:+.3f} vs {r.vs}{'*' if r.get('p', 1) < 0.05 else ''})" if r.vs else "")
                  for _, r in ab.iterrows()]
        ax.set_yticks(yv)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Outer-fold AUROC (95% corrected CI); * = paired difference p < 0.05")
        ax.set_title("The omics question: ablation on identical splits")
        P.grid(ax, "x")
        st.figure("ablation", fig, ab, "Ablation: clinical at diagnosis, + treatment, + DNA, + RNA, + both; paired differences on identical splits")
    if len(cb):
        pt = cb.groupby("patient_id").agg(pc=("p_cancer_death", "mean"), po=("p_other_death", "mean"), outcome=("outcome", "first"))
        fig, ax = P.fig(5.4, 5)
        for code, name, col in ((0, "alive", P.CAT[0]), (1, "died of cancer", P.CAT[1]), (2, "died of other causes", P.CAT[2])):
            d = pt[pt.outcome == code]
            ax.scatter(d.pc, d.po, s=7, alpha=0.55, color=col, linewidths=0, label=f"{name} (n={len(d)})")
        ax.set_xlabel("P(cancer death)")
        ax.set_ylabel("P(other-cause death)")
        ax.set_title("Each patient's risk, split by cause")
        ax.legend(fontsize=7, markerscale=2)
        P.grid(ax, "both")
        st.figure("cause_breakdown", fig, pt.reset_index(), "Out-of-fold cause-specific risks per patient (mean over repeats), coloured by actual outcome")


def _summary_md(run, cfg, F, ov, pc, rb, mod, ac) -> str:
    L = ["# Model agreement (step 6) summary", "",
         f"Run `{run.run_id}`, config hash `{cfg.hash}`. Inputs: farm runs {', '.join(f'`{r}`' for r in F['farm_runs'])}; "
         f"importance run `{F['importance_run']}`. Outer-fold predictions only; training rows only; the holdout is untouched.", ""]
    if len(ov):
        for arm in ov.arm.unique():
            o = ov[ov.arm == arm]
            L.append(f"- {arm}: top-10 feature overlap between models (Jaccard) median {o.jaccard_top10.median():.2f} "
                     f"(range {o.jaccard_top10.min():.2f}-{o.jaccard_top10.max():.2f}); importance rank correlation median "
                     f"{o.spearman_importance.median():.2f}.")
    if len(pc):
        for arm in pc.arm.unique():
            p = pc[pc.arm == arm]
            L.append(f"- {arm}: prediction correlation between models median {p.pred_spearman.median():.2f} "
                     f"(range {p.pred_spearman.min():.2f}-{p.pred_spearman.max():.2f}).")
    for arm, v in F.get("disagreement", {}).items():
        L.append(f"- {arm}: disagreement vs excess error Spearman {v['spearman_sd_vs_excess_error']:+.3f} "
                 f"(95% CI {v['ci_low']:+.3f} to {v['ci_high']:+.3f}) -> the 'models disagree' flag "
                 + ("**earns its place** in the patient output." if v["flag_earns_place"] else "**does not** earn its place (no reliable link to error)."))
    if F.get("loco"):
        lo = F["loco"]
        L.append(f"- Leave-one-cohort-out ({ac['loco_arm']}): the consensus features {', '.join(lo['consensus'])} rank on average "
                 f"{lo['mean_rank']:.1f} (worst {lo['worst_rank']:.0f}) in a held-out cohort; {lo['share_in_top10']:.0%} of "
                 "(model, cohort, feature) cases stay in the top 10.")
    if len(rb):
        for arm in rb.arm.unique():
            r = rb[rb.arm == arm].set_index("model")
            L.append(f"- {arm}: prediction stability across training sets (mean |change| per patient): "
                     + "; ".join(f"{k.split('|')[-1]} {v:.3f}" for k, v in r.mean_abs_change.items()) + ".")
    L += ["", "## The omics question (ablation, best-member procedures, identical splits)"]
    for a in F.get("ablation", []):
        line = f"- {a['procedure']}: {a['mean_auroc']:.3f} ({a['ci_low']:.3f}-{a['ci_high']:.3f})"
        if a.get("vs"):
            line += f"; vs {a['vs']}: {a['diff']:+.3f} ({a['diff_ci_low']:+.3f} to {a['diff_ci_high']:+.3f}, p = {a['p']:.3f})"
        L.append(line + ".")
    if F.get("cause"):
        c = F["cause"]
        L += ["", "## Cause-specific risk breakdown",
              f"- {c['n_patients']} patients. P(cancer death) and P(other-cause death) correlate {c['corr_cancer_other']:+.2f}; among the "
              f"top quarter by total risk, {c['high_risk_mostly_other']:.0%} are high-risk mainly through other-cause death.",
              f"- The cancer score separates cancer deaths with AUROC {c['auroc_cancer_score_for_cancer_death']:.3f}; the other-cause "
              f"score separates other-cause deaths with AUROC {c['auroc_other_score_for_other_death']:.3f}."]
    L += ["", "Caveats: permutation importance is sensitive to correlated features; agreement among models can reflect shared data "
          "limits rather than truth; this describes model behaviour, not biology."]
    return "\n".join(L)
