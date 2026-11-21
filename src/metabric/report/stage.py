"""`python run.py report --farm-runs A,B --ensemble-run E --agreement-run G`: validation report and
patient-level output (build step 7).

1. Reads the results of the nested runs (recorded as inputs) and writes `validation_report.md`,
   whose every sentence is computed from them.
2. Fits the final model once on all training rows (`deploy.fit_final_model`) and saves it, with
   the patient explainer, to `models/final_model.joblib`.
3. Writes the patient output: a table for every training patient (out-of-fold risk, risk group,
   cause breakdown, subtype) and full structured records for a few example patients.
Training rows only; the holdout is untouched.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import yaml

from ..core.artifacts import Run
from ..core.config import Config
from ..core.stats import corrected_resampled_ttest
from ..data import load_dataset
from ..eda import plots as P
from ..farm.cause import cause_labels
from ..farm.members import resolve_n_jobs
from ..harness import splits_from_table
from ..harness.floors import design_matrix
from ..importance.stage import _parallel
from ..splits import load_cv_splits, training_ids
from . import text as T
from .deploy import fit_final_model
from .patient import PatientExplainer

FIT_SECTIONS = ("columns", "seed", "clinical_types", "groups", "cv", "holdout")
FLOORS = ("age_only", "npi_file", "npi_standard", "majority")


def _read(run: Run, d, rel: str, role: str) -> pd.DataFrame:
    p = d / rel
    run.record_input(p, role)
    return pd.read_csv(p)


def _check(cfg, d, name):
    old = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    diff = [k for k in FIT_SECTIONS if json.dumps(old.get(k), sort_keys=True) != json.dumps(cfg.raw.get(k), sort_keys=True)]
    if diff:
        raise RuntimeError(f"{name}: config sections {diff} differ from the current config")


def _metrics(summary: pd.DataFrame, cand: str) -> dict:
    s = summary[summary.candidate == cand]
    return {r.metric: {"mean": float(r.mean), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high)} for r in s.itertuples()}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def run_report(cfg: Config, run: Run, farm_runs: list[str], ensemble_run: str, agreement_run: str) -> dict:
    st = run.stage("report")
    rc, h = cfg["report"], cfg["harness"]
    out = cfg.path("outputs")
    t0 = time.time()
    for r in farm_runs + [ensemble_run, agreement_run]:
        _check(cfg, out / r, r)
    # ---------------------------------------------------------------- results of the nested runs
    ed, ad = out / ensemble_run / "ensemble" / "tables", out / agreement_run / "agreement" / "tables"
    final = _read(run, ed, "final_model.csv", f"ensemble run {ensemble_run}").iloc[0]
    arm, option = final["arm"], final["model"].split("|")[1]
    proc = f"{arm}|best"
    ens_pairs = _read(run, ed, "paired_comparisons.csv", f"ensemble run {ensemble_run}")
    fd = None
    for fr in farm_runs:                                   # the first farm run holding the arm's procedure
        if (out / fr / "farm" / "tables" / "procedure_winners.csv").is_file():
            w = pd.read_csv(out / fr / "farm" / "tables" / "procedure_winners.csv")
            if (w.procedure == proc).any():
                fd = out / fr / "farm" / "tables"
                farm_main = fr
                break
    if fd is None:
        raise RuntimeError(f"no farm run in {farm_runs} holds {proc}")
    role = f"farm run {farm_main}"
    summ = _read(run, fd, "procedure_summary.csv", role)
    wins = _read(run, fd, "procedure_winners.csv", role)
    pairs = _read(run, fd, "paired_comparisons.csv", role)
    gate = _read(run, fd, "overfit_gate.csv", role)
    sub = _read(run, fd, "subgroup_metrics.csv", role)
    stab = _read(run, fd, "winner_stability.csv", role)
    rel = _read(run, fd, "reliability.csv", role)
    perm = _read(run, fd, "permutation_test.csv", role)
    members = []
    seen = set()
    for fr in farm_runs:                                   # each member's own procedure; first listed run wins
        s = pd.read_csv(out / fr / "farm" / "tables" / "procedure_summary.csv")
        if fr != farm_main:
            run.record_input(out / fr / "farm" / "tables" / "procedure_summary.csv", f"farm run {fr}")
        s = s[(s.metric == "auroc") & s.candidate.str.startswith(f"{arm}|") & ~s.candidate.str.contains(r"\|best|@|rep:", regex=True)]
        for r in s.itertuples():
            if r.candidate not in seen:
                seen.add(r.candidate)
                members.append({"procedure": r.candidate, "mean": r.mean, "ci_low": r.ci_low, "ci_high": r.ci_high})
    abl = _read(run, ad, "ablation.csv", f"agreement run {agreement_run}")
    dis = _read(run, ad, "disagreement_verdict.csv", f"agreement run {agreement_run}")
    cps = _read(run, ad, "cause_patient_summary.csv", f"agreement run {agreement_run}")

    # ---------------------------------------------------------------- data (training rows only)
    full = load_dataset(cfg)
    tr_ids = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows used; holdout rows untouched)")
    run.record_input(cfg.path("cv_splits"), "frozen CV splits (sizes for the corrected intervals)")
    ds = full.subset(tr_ids)
    X, y = design_matrix(ds), ds.y.astype(int)
    splits = splits_from_table(load_cv_splits(cfg, tr_ids))
    n_tr = float(np.mean([len(s.train) for s in splits]))
    n_te = float(np.mean([len(s.test) for s in splits]))

    met = _metrics(summ, proc)
    wp = wins[wins.procedure == proc]
    for pre in ("op", "opf1"):                             # F1 per outer fold at each operating point
        if f"proc_{pre}_ppv" in wp and f"{pre}_f1" not in met:
            pp, rr = wp[f"proc_{pre}_ppv"], wp[f"proc_{pre}_sensitivity"]
            f1 = (2 * pp * rr / (pp + rr)).to_numpy()
            ci = corrected_resampled_ttest(f1, n_tr, n_te)
            met[f"{pre}_f1"] = {"mean": float(f1.mean()), "ci_low": float(ci["ci_low"]), "ci_high": float(ci["ci_high"])}
    ag = dis[dis.arm == arm]
    pt = cps.set_index("patient_id")
    from sklearn.metrics import roc_auc_score
    hi = pt[pt.p_death >= pt.p_death.quantile(0.75)]
    F = {
        "arm": arm, "procedure": proc, "baseline": cfg["ensemble"]["cross_arm_baseline"], "config_hash": cfg.hash,
        "runs": {"farm": ", ".join(farm_runs), "ensemble": ensemble_run, "agreement": agreement_run, "report": run.run_id},
        "n_train": int(len(y)), "n_deaths": int(y.sum()), "base_rate": float(y.mean()), "n_splits": len(splits),
        "outer_folds": cfg["cv"]["outer_folds"], "outer_repeats": cfg["cv"]["outer_repeats"], "inner_folds": cfg["cv"]["inner_folds"],
        "n_holdout": int(len(full.ids) - len(tr_ids)), "holdout_scored": False, "floor_sd": h["regression_floor_sd"],
        "op_target": h["operating_point_sensitivity"], "context_col": rc["context_column"],
        "headline": {"option": option, "nested_auroc": float(final["select_and_ensemble_auroc"]),
                     "nested_ci": [float(v) for v in json.loads(final["select_and_ensemble_ci"])],
                     "mean_outer_auroc": float(final["mean_outer_auroc"]), "sd_outer_auroc": float(final["sd_outer_auroc"]),
                     "regression_floor": float(final["regression_floor"])},
        "metrics": met, "floor_metrics": _metrics(summ, "age_only"),
        "floor_compare": [{"b": r.b, "metric": r.metric, "diff": r.mean_diff, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p}
                          for r in pairs[(pairs.a == proc) & pairs.b.isin(FLOORS) & pairs.metric.isin(["auroc", "brier"])].itertuples()],
        "members": members,
        "winner_stability": stab[stab.procedure == proc].iloc[0].to_dict() if (stab.procedure == proc).any() else None,
        "optimism": float(wp.optimism.mean()),
        "gate": gate[gate.candidate == proc].iloc[0].to_dict(),
        "subgroups": sub[sub.candidate == proc].to_dict("records"),
        "perm": {"n": len(perm), "mean": float(perm.auroc.mean()), "max": float(perm.auroc.max()), "band": h["chance_band"]},
        "arm_compare": [{"arm": r.a.split("|")[0], "diff": r.mean_diff, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p}
                        for r in ens_pairs[ens_pairs.b == f"{cfg['ensemble']['cross_arm_baseline']}|ensemble"].itertuples()],
        "ablation": abl.replace({np.nan: None}).to_dict("records"),
        "disagreement": None if ag.empty else {"rho": float(ag.spearman_sd_vs_excess_error.iloc[0]), "ci_low": float(ag.ci_low.iloc[0]),
                                                "ci_high": float(ag.ci_high.iloc[0]), "earns": bool(ag.flag_earns_place.iloc[0])},
        "cause": {"auroc_cancer": float(roc_auc_score(pt.outcome == 1, pt.p_cancer_death)),
                  "auroc_other": float(roc_auc_score(pt.outcome == 2, pt.p_other_death)),
                  "high_mostly_other": float((hi.p_other_death > hi.p_cancer_death).mean())},
    }

    # ---------------------------------------------------------------- deployment fit
    n_jobs = resolve_n_jobs(cfg)

    def par(fn, tasks, label):
        return _parallel(fn, tasks, n_jobs, st.log.info, label)
    fm, tabs = fit_final_model(cfg, X, y, cause_labels(ds.frame), arm, option, par, st.log.info)
    for k, df in tabs.items():
        st.table(k, df, {"deploy_candidates": "Deployment fit: every candidate's out-of-fold metrics on all training rows",
                         "deploy_member_winners": "Deployment fit: each member's configuration chosen by the inner rule",
                         "deploy_merge": "Deployment fit: merged members (Spearman above the merge threshold) and their weights",
                         "deploy_oof_predictions": "Deployment fit: out-of-fold predictions of the chosen column (raw and primary)"}[k])
    F["deploy"] = {"chosen_column": fm.info["chosen_column"], "weights": fm.weights, "primary": fm.primary,
                   "oof_cal_slope_raw": fm.info["oof_cal_slope_raw"], "oof_auroc": fm.info["oof_auroc"]}
    oof = tabs["deploy_oof_predictions"].set_index("patient_id")

    # ---------------------------------------------------------------- patient explainer
    types = cfg["clinical_types"]
    quality = {"nested_cv_auroc": F["headline"]["nested_auroc"], "nested_cv_auroc_ci": F["headline"]["nested_ci"],
               "calibration_slope_raw": met.get("cal_slope", {}).get("mean"),
               "calibration_slope_recalibrated": met.get("recal_cal_slope", {}).get("mean"),
               "brier_skill": met.get("brier_skill", {}).get("mean"),
               "evaluated_on": "nested cross-validation on the training rows; holdout not yet scored"}
    ex = PatientExplainer(model=fm, blocks=cfg.blocks, numeric=[c for c in types["numeric"] + types["ordinal"] if c in fm.columns],
                          categorical=[c for c in types["categorical"] if c in fm.columns], train_X=X[fm.columns],
                          train_oof=oof.p_primary.to_numpy(), train_y=y.to_numpy(), context_col=rc["context_column"],
                          context_values=ds.frame[rc["context_column"]], mutation_prevalence=ds.mutations.mean(),
                          quality=quality, risk_quantiles=rc["risk_quantiles"], risk_labels=rc["risk_labels"],
                          n_background=rc["n_background"], n_perm=rc["n_permutations"], small_group=rc["small_group"],
                          top_k=rc["top_k"], seed=cfg.seed)
    mdir = run.dir / "models"
    mdir.mkdir(exist_ok=True)
    bundle = mdir / "final_model.joblib"
    joblib.dump({"model": fm, "explainer": ex, "config_hash": cfg.hash, "run_id": run.run_id}, bundle)
    Xc = X[fm.columns].iloc[:50]
    if not np.allclose(joblib.load(bundle)["model"].predict(Xc), fm.predict(Xc)):
        raise RuntimeError("the saved model does not reproduce the fitted model's predictions")
    run._register(bundle, "models", "model", "Final model fitted on all training rows, with its patient explainer "
                  "(load with joblib; explainer.records(X) gives patient records)", "metabric.report.stage")

    # ---------------------------------------------------------------- patient output
    cut = ex.cut
    po = pd.DataFrame({"patient_id": oof.index, "died": oof.y.to_numpy(), "p_death_oof": oof.p_primary.to_numpy()})
    po["percentile"] = [float((oof.p_primary.to_numpy() < p).mean() * 100) for p in po.p_death_oof]
    po["risk_group"] = [rc["risk_labels"][int(np.digitize(p, cut))] for p in po.p_death_oof]
    po[rc["context_column"]] = ds.frame.loc[po.patient_id, rc["context_column"]].to_numpy()
    po = po.merge(cps[["patient_id", "p_cancer_death", "p_other_death"]], on="patient_id", how="left")
    st.table("patient_output", po, "Every training patient: out-of-fold P(death) from the deployment folds, percentile, risk group, "
             "subtype, and out-of-fold cause-specific risks (agreement run); no patient is scored by a model trained on them")
    gr = pd.DataFrame([{"risk_group": k, **v} for k, v in ex.group_rates.items()])
    st.table("risk_groups", gr, "Risk groups (cutpoints = quartiles of training out-of-fold risk): observed death rate, Wilson 95% CI")
    ctx = pd.DataFrame([{"group": k, **v} for k, v in ex.context_table.items()]).sort_values("n", ascending=False)
    st.table("cohort_context", ctx, f"Cohort context by {rc['context_column']}: training patients, deaths, observed rate, Wilson CI")
    # example records: patients at fixed quantiles of out-of-fold risk (in-sample illustration of the function)
    order = oof.p_primary.sort_values()
    ids = list(dict.fromkeys(order.index[min(int(q * len(order)), len(order) - 1)] for q in rc["example_quantiles"]))
    raw = pd.concat([ds.frame[cfg.blocks["clinical"]], ds.mutations], axis=1).loc[ids]
    recs = ex.records(raw.reset_index(drop=True), ids)
    st.text("example_patient_records.json", json.dumps(_jsonable(recs), indent=2, default=str),
            "Structured patient records for example training patients (in-sample: the model was fitted on them; illustration of "
            "the output format only)")
    chk = pd.DataFrame([{"patient_id": r["patient_id"], "base": r["contributions"]["base_value"],
                         "sum_contributions": r["contributions"]["sum_check"] - r["contributions"]["base_value"],
                         "model_log_odds": r["contributions"]["model_output"],
                         "abs_error": abs(r["contributions"]["sum_check"] - r["contributions"]["model_output"])} for r in recs])
    st.table("contribution_check", chk, "Shapley contributions plus base value against the model's log-odds (must match)")
    if chk.abs_error.max() > 1e-6:
        raise RuntimeError("Shapley contributions do not add up to the model output")

    # ---------------------------------------------------------------- figures and report
    _figures(st, rel, proc, sub[sub.candidate == proc], gr)
    st.text("validation_report.md", T.validation_report(F), "Validation report (generated from the run results)")
    F["seconds"] = round(time.time() - t0)
    st.log.info("report done in %d s", F["seconds"])
    return F


def _figures(st, rel, proc, sub, gr):
    d = rel[rel.candidate.isin([proc, "age_only"])]
    fig, ax = P.fig(5.2, 5)
    ax.plot([0, 1], [0, 1], color=P.NEUTRAL, linestyle="--", linewidth=1)
    for i, (c, g) in enumerate(d.groupby("candidate", sort=False)):
        ax.plot(g.mean_pred, g.obs_rate, marker="o", color=P.CAT[i], label=c)
    ax.set_xlabel("Mean predicted P(death), raw (outer folds, quantile bins)")
    ax.set_ylabel("Observed death rate")
    ax.set_title("Reliability: final procedure vs age-only floor")
    ax.legend()
    P.grid(ax, "both")
    st.figure("reliability", fig, d, "Reliability curve of the outer-fold predictions (raw probabilities) with the age-only floor")
    if len(sub):
        s = sub.reset_index(drop=True)
        fig, ax = P.fig(6.4, 0.38 * len(s) + 1.3)
        yv = np.arange(len(s))[::-1]
        ax.hlines(yv, s.auroc_ci_low, s.auroc_ci_high, color=P.CAT[0], linewidth=2)
        ax.plot(s.auroc, yv, "o", color=P.CAT[0])
        ax.set_yticks(yv)
        ax.set_yticklabels([f"{r.subgroup}: {r.level} (n={r.n})" for r in s.itertuples()], fontsize=7)
        ax.set_xlabel("AUROC (95% bootstrap CI)")
        ax.set_title("Discrimination within treatment subgroups")
        P.grid(ax, "x")
        st.figure("subgroups", fig, s, "AUROC of the final procedure within each treatment subgroup (risk given treatment received)")
    fig, ax = P.fig(5.6, 3.4)
    x = np.arange(len(gr))
    ax.bar(x, gr.observed_death_rate, color=P.CAT[1], width=0.6)
    ax.errorbar(x, gr.observed_death_rate, yerr=[gr.observed_death_rate - gr.ci_low, gr.ci_high - gr.observed_death_rate],
                fmt="none", ecolor=P.INK2, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.risk_group}\n(n={r.n})" for r in gr.itertuples()], fontsize=8)
    ax.set_ylabel("Observed death rate")
    ax.set_title("Observed death rate by risk group")
    P.grid(ax, "y")
    st.figure("risk_groups", fig, gr, "Observed death rate (Wilson 95% CI) in the risk groups of the patient output")
