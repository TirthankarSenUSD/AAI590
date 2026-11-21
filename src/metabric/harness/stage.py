"""`python run.py harness`: run the floors through the frozen nested CV and write the artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.artifacts import Run
from ..core.config import Config
from ..data import load_dataset
from ..eda import plots as P
from .floors import design_matrix, floor_candidates
from .core import Harness, reliability, splits_from_table
from ..splits import load_cv_splits, training_ids


def treatment_groups(f: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame(index=f.index)
    g["chemotherapy"] = f["chemotherapy"].map({0: "no", 1: "yes"})
    g["hormone_therapy"] = f["hormone_therapy"].map({0: "no", 1: "yes"})
    g["radio_therapy"] = f["radio_therapy"].map({0: "no", 1: "yes"})
    g["type_of_breast_surgery"] = f["type_of_breast_surgery"].astype(object).fillna("missing")
    g["chemo_x_hormone"] = "chemo " + g["chemotherapy"] + " / hormone " + g["hormone_therapy"]
    return g


def run_harness(cfg: Config, run: Run) -> dict:
    st = run.stage("harness")
    rng = np.random.default_rng(cfg.seed)
    full = load_dataset(cfg)
    run.record_input(cfg.path("data"), "dataset")
    tr_ids = training_ids(cfg, full)                       # verifies the holdout hash
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows used; holdout rows untouched)")
    ds = full.subset(tr_ids)
    sp = load_cv_splits(cfg, tr_ids)                       # verifies the CV split hash
    run.record_input(cfg.path("cv_splits"), "frozen CV splits")
    splits = splits_from_table(sp)
    H = Harness(cfg, design_matrix(ds), ds.y, splits)
    cands = floor_candidates(cfg)
    st.log.info("harness: %d training rows, %d splits (%s outer x %s repeats, %s inner), %d floors",
                len(ds.y), len(splits), cfg["cv"]["outer_folds"], cfg["cv"]["outer_repeats"], cfg["cv"]["inner_folds"], len(cands))

    res = H.evaluate(cands)
    st.table("fold_metrics", res.folds, "Per-split metrics of every floor (outer test rows; thresholds and recalibration fitted on inner folds)")
    st.table("oof_predictions", res.preds, "Out-of-fold predicted probability of death per patient, split and floor")
    summ = H.summarise(res.folds)
    st.table("summary", summ, "Mean, SD and corrected-t 95% CI over the outer folds (Nadeau-Bengio variance)")
    pair = H.paired(res.folds)
    st.table("paired_comparisons", pair, "Paired differences a - b on identical splits, corrected resampled t-test")
    sub = H.subgroups(res.preds, treatment_groups(ds.frame), rng)
    st.table("subgroup_metrics", sub, "Metrics within treatment subgroups (out-of-fold predictions averaged over repeats; bootstrap 95% CI)")
    gate = H.overfit_gate(res.folds)
    st.table("overfit_gate", gate, "Overfit gate: overfitting (train-CV gap, slope < 0.8) and under-confidence (slope > 1.2: recalibrated probabilities primary); report only")
    lc = H.learning_curve(cands, rng)
    st.table("learning_curve", lc, "Learning curves: outer-fold AUROC by fraction of the outer training rows (reported, not gating)")
    rel = reliability(res.preds, cfg["harness"]["reliability_bins"])
    st.table("reliability", rel, "Reliability (quantile bins) of out-of-fold predictions averaged over repeats")
    loco = H.loco(cands, ds.frame["cohort"])
    st.table("loco", loco, "Leave-one-cohort-out on the training rows (sensitivity only)")
    rf = H.regression_floor(res.folds)
    st.table("regression_floor", rf, "Regression floor per floor model: mean - 2 SD of outer-fold AUROC")
    age = next(c for c in cands if c.name == "age_only")
    proc = lambda y: H.evaluate([age], y=y, light=True).folds.auroc.mean()  # noqa: E731
    perm, perm_s = H.permutation_test(proc, cfg["harness"]["n_permutations"], rng)
    st.table("permutation_test", perm, "Label-permutation test of the age-only procedure: AUROC per shuffled-label run")
    st.table("permutation_summary", pd.DataFrame([perm_s]), "Permutation test summary (must be at chance)")
    if not perm_s["at_chance"]:
        raise RuntimeError(f"permutation test not at chance: {perm_s}")

    _figures(st, res, summ, rel, lc, sub, cands)
    facts = _facts(summ, pair, gate, rf, perm_s, loco, sub, len(ds.y), len(splits))
    st.text("harness_summary.md", _summary_md(run, cfg, facts, cands), "Auto-generated harness summary (computed wording)")
    return facts


def _figures(st, res, summ, rel, lc, sub, cands):
    names = [c.name for c in cands]
    col = {n: P.CAT[i] for i, n in enumerate(names)}
    f = res.folds
    fig, ax = P.fig(6.4, 3.4)
    for i, n in enumerate(names):
        v = f[f.candidate == n].auroc.to_numpy()
        jitter = np.linspace(-0.15, 0.15, len(v))
        ax.plot(v, i + jitter, "o", color=col[n], markersize=5, alpha=0.8)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=P.INK, linewidth=2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0.5, color=P.INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("Outer-fold AUROC (dots = splits, bar = mean)")
    ax.set_title("Floors on the frozen nested CV")
    P.grid(ax, "x")
    st.figure("floors_fold_auroc", fig, f[["candidate", "split_id", "auroc"]], "Outer-fold AUROC of each floor on the 10 frozen splits; bar = mean")
    fig, ax = P.fig(5, 4.6)
    ax.plot([0, 1], [0, 1], color=P.INK2, linestyle="--", linewidth=1)
    for n in names:
        if n == "majority":
            continue
        r = rel[rel.candidate == n]
        ax.plot(r.mean_pred, r.obs_rate, marker="o", color=col[n], label=n, markersize=5)
    ax.set_xlabel("Mean predicted risk of death (decile)")
    ax.set_ylabel("Observed death rate")
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0.2, 1)
    ax.set_title("Calibration of the floors (out-of-fold)")
    ax.legend(loc="upper left")
    P.grid(ax, "both")
    st.figure("floors_reliability", fig, rel, "Reliability curves (quantile deciles) of out-of-fold predictions; dashed = perfect calibration")
    fig, ax = P.fig(6, 3.6)
    for n in names:
        r = lc[lc.candidate == n]
        ax.plot(r.n_train, r.cv_auroc, marker="o", color=col[n], label=n, markersize=5)
    ax.set_xlabel("Training rows used (mean over splits)")
    ax.set_ylabel("Outer-fold AUROC")
    ax.set_title("Learning curves of the floors")
    ax.legend(loc="lower right", ncol=2)
    P.grid(ax)
    st.figure("floors_learning_curve", fig, lc, "Outer-fold AUROC against training size (stratified subsamples of the outer training rows)")
    s = sub[sub.candidate != "majority"].copy()
    s["row"] = s.subgroup + ": " + s.level
    rows = list(dict.fromkeys(s.row))
    fig, axes = P.fig(11, 0.32 * len(rows) + 1.6, 1, 3, sharey=True)
    for ax, n in zip(axes, [x for x in names if x != "majority"]):
        t = s[s.candidate == n].set_index("row").loc[rows]
        P.dot_ci(ax, [f"{r} (n={k})" for r, k in zip(rows, t.n)], t.auroc, t.auroc_ci_low, t.auroc_ci_high, color=col[n], ref=0.5)
        ax.set_title(n, fontsize=9)
        ax.set_xlim(0.3, 0.9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Floor AUROC within treatment subgroups (bootstrap 95% CI)", x=0.01, ha="left", fontweight="bold", color=P.INK)
    fig.tight_layout()
    st.figure("floors_subgroup_auroc", fig, s, "AUROC of each floor within treatment subgroups, bootstrap 95% CI; dashed = chance")


def _facts(summ, pair, gate, rf, perm, loco, sub, n, n_splits) -> dict:
    g = lambda c, m, k="mean": float(summ[(summ.candidate == c) & (summ.metric == m)][k].iloc[0])  # noqa: E731
    F = {"n_train_rows": n, "n_splits": n_splits}
    for c in summ.candidate.unique():
        for m in ("auroc", "brier_skill", "cal_slope", "cal_intercept", "log_loss", "op_sensitivity", "op_specificity"):
            if ((summ.candidate == c) & (summ.metric == m)).any():
                F[f"{c}_{m}"] = round(g(c, m), 4)
        F[f"{c}_auroc_ci"] = [round(g(c, "auroc", "ci_low"), 4), round(g(c, "auroc", "ci_high"), 4)]
    for a, b in [("age_only", "npi_file"), ("npi_standard", "npi_file")]:
        r = pair[(pair.a == a) & (pair.b == b) & (pair.metric == "auroc")]
        if r.empty:
            r = pair[(pair.a == b) & (pair.b == a) & (pair.metric == "auroc")]
            sign = -1
        else:
            sign = 1
        r = r.iloc[0]
        F[f"diff_{a}_vs_{b}"] = {"mean": round(sign * r.mean_diff, 4), "p": round(r.p, 4),
                                 "ci": sorted([round(sign * r.ci_low, 4), round(sign * r.ci_high, 4)])}
    F["gate_overfit"] = gate[gate.status == "overfit"].candidate.tolist()
    F["gate_recalibrate"] = gate[gate.status == "recalibrate"].candidate.tolist()
    F["regression_floor"] = {r.candidate: round(r.regression_floor, 4) for r in rf.itertuples()}
    F["permutation"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in perm.items()}
    lo = loco[loco.candidate == "age_only"]
    F["loco_age_auroc_range"] = [round(lo.auroc.min(), 3), round(lo.auroc.max(), 3)]
    s = sub[(sub.candidate == "age_only") & (sub.subgroup == "chemotherapy")].set_index("level")
    F["age_auroc_by_chemo"] = {k: round(v, 3) for k, v in s.auroc.items()}
    return F


def _summary_md(run, cfg, F, cands) -> str:
    best = max(["age_only", "npi_file", "npi_standard"], key=lambda c: F[f"{c}_auroc"])
    d = F["diff_age_only_vs_npi_file"]
    ds = F["diff_npi_standard_vs_npi_file"]
    sig = lambda x: "a significant" if x["p"] < 0.05 else "no significant"  # noqa: E731
    P_ = F["permutation"]
    L = [f"# Harness summary: floors on the frozen nested CV", "",
         f"Run `{run.run_id}`, config hash `{cfg.hash}`, seed {cfg.seed}. Training rows only ({F['n_train_rows']:,}); the "
         f"holdout is untouched. {F['n_splits']} outer splits ({cfg['cv']['outer_folds']} folds x {cfg['cv']['outer_repeats']} repeats), "
         f"{cfg['cv']['inner_folds']} inner folds each. Every number below is computed in this run.", "",
         "## Floors (never eligible to win)", "",
         "| Floor | AUROC (95% CI) | Brier skill | Calibration slope | Sensitivity / specificity at the inner-fold 80% operating point |",
         "|---|---|---|---|---|"]
    for c in cands:
        n = c.name
        ci = F[f"{n}_auroc_ci"]
        slope = F.get(f"{n}_cal_slope")
        L.append(f"| {c.label} | {F[f'{n}_auroc']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {F[f'{n}_brier_skill']:.3f} | "
                 f"{'n/a' if slope is None or np.isnan(slope) else f'{slope:.2f}'} | "
                 f"{F[f'{n}_op_sensitivity']:.2f} / {F[f'{n}_op_specificity']:.2f} |")
    L += ["", f"- The strongest floor is `{best}` (AUROC {F[f'{best}_auroc']:.3f}). Any model must beat this on the same splits.",
          f"- Age-only minus NPI as supplied: {d['mean']:+.3f} AUROC (95% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}, p = {d['p']:.3f}): "
          f"{sig(d)} difference.",
          f"- Standard NPI minus NPI as supplied: {ds['mean']:+.3f} AUROC (95% CI {ds['ci'][0]:+.3f} to {ds['ci'][1]:+.3f}, p = {ds['p']:.3f}): "
          f"{sig(ds)} difference.",
          f"- Label-permutation test ({P_['n_permutations']} shuffles of the age-only procedure): mean AUROC {P_['perm_mean']:.3f} "
          f"(max {P_['perm_max']:.3f}) against {P_['observed_auroc']:.3f} observed; "
          f"{'at chance, as required' if P_['at_chance'] else 'NOT at chance: investigate leakage'}.",
          f"- Overfit gate (report only; never excludes from selection): overfitting (train-CV gap > 0.05 or slope < 0.8) in "
          f"{'no floor' if not F['gate_overfit'] else ', '.join(F['gate_overfit'])}; under-confident (slope > 1.2, recalibrated "
          f"probabilities are primary) in {'no floor' if not F['gate_recalibrate'] else ', '.join(F['gate_recalibrate'])}.",
          f"- Leave-one-cohort-out, age-only AUROC ranges {F['loco_age_auroc_range'][0]:.3f}-{F['loco_age_auroc_range'][1]:.3f} across held-out cohorts.",
          f"- Age-only AUROC within chemotherapy groups: " + ", ".join(f"{k} {v:.3f}" for k, v in F["age_auroc_by_chemo"].items()) + ".",
          "- Regression floors (mean - 2 SD of outer-fold AUROC): " + ", ".join(f"{k} {v:.3f}" for k, v in F["regression_floor"].items()) + ".",
          ""]
    return "\n".join(L)
