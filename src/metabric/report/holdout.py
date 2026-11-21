"""`python run.py score-holdout --report-run R [--dry-run]`: score the frozen holdout once, last.

1. Loads the final model saved by report run R (`models/final_model.joblib`) and checks that the
   report run used the same data, columns, splits and holdout settings as the current config.
2. Applies the model unchanged to the holdout patients. Nothing is refitted, recalibrated or
   re-chosen on holdout labels. The floors are refitted once on all training rows and scored on
   the holdout; the operating points are the thresholds fixed on the deployment out-of-fold
   predictions (training rows only).
3. Every interval and paired difference comes from one patient bootstrap of the holdout.
4. A one-time marker (`paths.holdout_scored`) records the scoring. Once it says "complete" the
   command refuses to run again. A scoring interrupted before completion may be rerun with the
   same model only.

`--dry-run` runs every step with the holdout labels shuffled (the numbers are meaningless and are
labelled as such). It never writes the marker; it checks the code path and shows the report format.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..core.artifacts import Run
from ..core.config import Config
from ..core.paths import sha256_file
from ..core.stats import wilson
from ..data import load_dataset
from ..eda import plots as P
from ..farm.cause import CAUSE_CODES
from ..harness.core import _fit, _proba
from ..harness.floors import design_matrix, floor_candidates
from ..harness.metrics import binary_metrics, threshold_for_f1, threshold_for_sensitivity, threshold_metrics
from ..harness.stage import treatment_groups
from ..splits import load_holdout
from ..splits.frozen import HoldoutError
from . import text as T
from .stage import _check, _jsonable

METRICS = ("auroc", "auprc", "log_loss", "brier", "brier_skill", "cal_slope", "cal_intercept")
PAIRED = ("auroc", "brier")
OP_KEYS = ("sensitivity", "specificity", "ppv", "npv", "f1")


# ---------------------------------------------------------------- one-time marker
def read_marker(cfg: Config, key: str = "holdout_scored") -> dict | None:
    p = cfg.path(key)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def check_marker(cfg: Config, bundle_sha: str, key: str = "holdout_scored") -> dict | None:
    """Refuse a second scoring. An interrupted scoring ("started") may be rerun with the same model."""
    m = read_marker(cfg, key)
    if m is None:
        return None
    if m.get("status") == "complete":
        raise HoldoutError(f"the holdout was already scored (run {m.get('run_id')}, {m.get('completed')}); "
                           "it is scored once only")
    if m.get("bundle_sha256") != bundle_sha:
        raise HoldoutError("an interrupted holdout scoring used a different final model; the holdout cannot be "
                           "rescored with another model")
    return m


def write_marker(cfg: Config, key: str = "holdout_scored", **fields) -> None:
    p = cfg.path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    m = (read_marker(cfg, key) or {}) | fields
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------- scoring (pure: arrays in, tables out)
def _pct(v: np.ndarray, a: float) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    return (float(np.quantile(v, a / 2)), float(np.quantile(v, 1 - a / 2))) if len(v) else (float("nan"), float("nan"))


def _boot_p(d: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a paired difference (share of resamples on the other side of 0)."""
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else float("nan")


def evaluate_holdout(y: np.ndarray, p: np.ndarray, floors: dict[str, np.ndarray], train_prev: float,
                     thresholds: dict[str, float], groups: pd.DataFrame, cut: np.ndarray, labels: list[str],
                     train_rates: dict, n_boot: int, alpha: float, seed: int,
                     cause: pd.DataFrame | None = None, y_cause: np.ndarray | None = None) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    ok = np.array([y[i].min() != y[i].max() for i in idx])
    idx = idx[ok]

    def boot(fn):
        return np.array([fn(i) for i in idx])

    # model and floors: threshold-free metrics
    rows, bm = [], {}
    for name, q in {"final_model": p, **floors}.items():
        pt = binary_metrics(y, q, train_prev)
        B = {k: np.empty(len(idx)) for k in METRICS}
        for j, i in enumerate(idx):
            m = binary_metrics(y[i], q[i], train_prev)
            for k in METRICS:
                B[k][j] = m[k]
        bm[name] = B
        for k in METRICS:
            lo, hi = _pct(B[k], alpha)
            rows.append({"model": name, "metric": k, "value": pt[k], "ci_low": lo, "ci_high": hi})
    metrics = pd.DataFrame(rows)

    # paired differences, final model minus each floor, on the same resamples
    comp = []
    for name in floors:
        pt_m, pt_f = binary_metrics(y, p, train_prev), binary_metrics(y, floors[name], train_prev)
        for k in PAIRED:
            d = bm["final_model"][k] - bm[name][k]
            lo, hi = _pct(d, alpha)
            comp.append({"a": "final_model", "b": name, "metric": k, "diff": pt_m[k] - pt_f[k], "ci_low": lo, "ci_high": hi,
                         "p": _boot_p(d)})
    comp = pd.DataFrame(comp)

    # operating points fixed on training rows, applied unchanged
    ops = []
    for name, t in thresholds.items():
        pt = threshold_metrics(y, p, t)
        B = {k: np.empty(len(idx)) for k in OP_KEYS}
        for j, i in enumerate(idx):
            m = threshold_metrics(y[i], p[i], t)
            for k in OP_KEYS:
                B[k][j] = m[k]
        for k in OP_KEYS:
            lo, hi = _pct(B[k], alpha)
            ops.append({"operating_point": name, "threshold": t, "metric": k, "value": pt[k], "ci_low": lo, "ci_high": hi})
    ops = pd.DataFrame(ops)

    # treatment subgroups (risk given the treatment received)
    sub = []
    for col in groups.columns:
        for lev, gi in groups.groupby(col).groups.items():
            m = groups.index.get_indexer(gi)
            yy, pp = y[m], p[m]
            one = yy.min() == yy.max()
            if one:
                a, lo, hi = float("nan"), float("nan"), float("nan")
            else:
                a = float(roc_auc_score(yy, pp))
                r2 = np.random.default_rng(seed + 1)
                bs = []
                for _ in range(n_boot):
                    i = r2.integers(0, len(yy), len(yy))
                    if yy[i].min() != yy[i].max():
                        bs.append(roc_auc_score(yy[i], pp[i]))
                lo, hi = _pct(np.array(bs), alpha)
            sub.append({"subgroup": col, "level": lev, "n": int(len(yy)), "deaths": int(yy.sum()), "auroc": a,
                        "auroc_ci_low": lo, "auroc_ci_high": hi, "mean_pred": float(pp.mean()), "obs_rate": float(yy.mean())})
    sub = pd.DataFrame(sub)

    # risk groups with the training cutpoints
    g = np.digitize(p, cut)
    rg = []
    for k, lab in enumerate(labels):
        yy = y[g == k]
        lo, hi = wilson(int(yy.sum()), len(yy)) if len(yy) else (float("nan"), float("nan"))
        tr = train_rates.get(lab, {})
        rg.append({"risk_group": lab, "n": int(len(yy)), "deaths": int(yy.sum()),
                   "observed_death_rate": float(yy.mean()) if len(yy) else float("nan"), "ci_low": lo, "ci_high": hi,
                   "mean_pred": float(p[g == k].mean()) if len(yy) else float("nan"),
                   "training_observed_death_rate": tr.get("observed_death_rate")})
    rg = pd.DataFrame(rg)

    # reliability (quantile bins of predicted risk)
    from ..harness.metrics import reliability_table
    rel = reliability_table(y, p, 10)

    cz = None
    if cause is not None and y_cause is not None:
        yc = np.asarray(y_cause)
        ok_c = yc >= 0
        cz = {"auroc_cancer": float(roc_auc_score(yc[ok_c] == 1, cause.p_cancer_death.to_numpy()[ok_c])),
              "auroc_other": float(roc_auc_score(yc[ok_c] == 2, cause.p_other_death.to_numpy()[ok_c])),
              "n": int(ok_c.sum())}
    return {"metrics": metrics, "floor_comparisons": comp, "operating_points": ops, "subgroups": sub, "risk_groups": rg,
            "reliability": rel, "cause": cz, "n_boot_used": int(len(idx)), "groups_index": g}


# ---------------------------------------------------------------- stage
def _report_inputs(cfg: Config, report_run: str) -> dict:
    out = cfg.path("outputs")
    rd = out / report_run
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    if man.get("status") != "complete" or not man["command"].startswith("report "):
        raise RuntimeError(f"{report_run} is not a completed report run")
    tok = man["command"].split()
    ens = tok[tok.index("--ensemble-run") + 1]
    return {"dir": rd, "ensemble_run": ens, "bundle": rd / "models" / "final_model.joblib",
            "final": pd.read_csv(out / ens / "ensemble" / "tables" / "final_model.csv").iloc[0],
            "oof": pd.read_csv(rd / "report" / "tables" / "deploy_oof_predictions.csv")}


def run_holdout(cfg: Config, run: Run, report_run: str, dry_run: bool = False) -> dict:
    st = run.stage("holdout")
    hc, h, rc = cfg["holdout_scoring"], cfg["harness"], cfg["report"]
    t0 = time.time()
    R = _report_inputs(cfg, report_run)
    _check(cfg, R["dir"], report_run)
    bundle_sha = sha256_file(R["bundle"])
    run.record_input(R["bundle"], f"final model saved by report run {report_run}")
    run.record_input(R["dir"] / "report" / "tables" / "deploy_oof_predictions.csv", "deployment out-of-fold predictions (thresholds)")
    run.record_input(cfg.path("outputs") / R["ensemble_run"] / "ensemble" / "tables" / "final_model.csv",
                     "cross-validated estimate and regression floor")
    if not dry_run:
        prev = check_marker(cfg, bundle_sha)
        if prev:
            st.log.info("resuming an interrupted scoring started %s (same model)", prev.get("started"))
        write_marker(cfg, status="started", started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     run_id=run.run_id, report_run=report_run, bundle_sha256=bundle_sha, config_hash=cfg.hash)

    full = load_dataset(cfg)
    hold = load_holdout(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (scored here, once)")
    ho = pd.Index(hold.loc[hold["set"] == "holdout", "patient_id"])
    tr = pd.Index(hold.loc[hold["set"] == "train", "patient_id"])
    X = design_matrix(full)
    y_all = full.y.astype(int)
    yc_all = full.frame["death_from_cancer"].map(CAUSE_CODES)       # one patient has no recorded cause: -1, left out of the cause AUROCs
    y_ho, yc_ho = y_all.loc[ho].to_numpy(), yc_all.loc[ho].fillna(-1).astype(int).to_numpy()
    if dry_run:                                             # shuffled labels: the numbers carry no information
        perm = np.random.default_rng(cfg.seed + 7).permutation(len(ho))
        y_ho, yc_ho = y_ho[perm], yc_ho[perm]

    b = joblib.load(R["bundle"])
    fm, ex = b["model"], b["explainer"]
    Xho = X.loc[ho, fm.columns]
    p = fm.predict(Xho)
    lo, hi = fm.predict_interval(Xho)
    cause = fm.predict_cause(Xho)

    floors = {}
    for c in floor_candidates(cfg):
        est = _fit(c, X, tr, y_all)
        floors[c.name] = _proba(est, c, X, ho)

    oof = R["oof"]
    thr = {f"{int(round(h['operating_point_sensitivity'] * 100))}% recall target":
           threshold_for_sensitivity(oof.y, oof.p_primary, h["operating_point_sensitivity"]),
           "F1-maximising": threshold_for_f1(oof.y, oof.p_primary)}
    grp = treatment_groups(full.frame.loc[ho]).reset_index(drop=True)
    E = evaluate_holdout(y_ho, p, floors, float(y_all.loc[tr].mean()), thr, grp, ex.cut, rc["risk_labels"], ex.group_rates,
                         int(hc["n_boot"]), float(hc["alpha"]), cfg.seed, cause, yc_ho)

    tag = " (DRY RUN: holdout labels shuffled; numbers meaningless)" if dry_run else ""
    st.table("holdout_metrics", E["metrics"], "Holdout: final model and floors, threshold-free metrics, 95% bootstrap CI" + tag)
    st.table("holdout_floor_comparisons", E["floor_comparisons"], "Holdout: final model minus each floor, paired bootstrap" + tag)
    st.table("holdout_operating_points", E["operating_points"],
             "Holdout: operating points with thresholds fixed on the deployment out-of-fold predictions" + tag)
    st.table("holdout_subgroups", E["subgroups"], "Holdout: AUROC within treatment subgroups (risk given treatment received)" + tag)
    st.table("holdout_risk_groups", E["risk_groups"], "Holdout: observed death rate in the training-defined risk groups" + tag)
    pred = pd.DataFrame({"patient_id": ho, "died": y_ho, "p_death": p, "p_low": lo, "p_high": hi,
                         "risk_group": [rc["risk_labels"][k] for k in E["groups_index"]]})
    if cause is not None:
        pred["p_cancer_death"], pred["p_other_death"] = cause.p_cancer_death.to_numpy(), cause.p_other_death.to_numpy()
    st.table("holdout_predictions", pred, "Holdout: every patient's P(death) from the final model, interval and risk group" + tag)

    m = E["metrics"].set_index(["model", "metric"])
    fin = E["metrics"][E["metrics"].model == "final_model"].set_index("metric")
    age = E["metrics"][E["metrics"].model == "age_only"].set_index("metric")
    cv_auc = float(R["final"]["select_and_ensemble_auroc"])
    cv_ci = [float(v) for v in json.loads(R["final"]["select_and_ensemble_ci"])]
    F = {"dry_run": dry_run, "runs": {"report": report_run, "ensemble": R["ensemble_run"], "holdout": run.run_id},
         "bundle_sha256": bundle_sha, "config_hash": cfg.hash, "n": int(len(ho)), "deaths": int(y_ho.sum()),
         "base_rate": float(y_ho.mean()), "train_base_rate": float(y_all.loc[tr].mean()), "n_boot": E["n_boot_used"],
         "final": {k: {"mean": float(r.value), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high)} for k, r in fin.iterrows()},
         "floor_metrics_full": {k: {"mean": float(r.value), "ci_low": float(r.ci_low), "ci_high": float(r.ci_high)}
                                for k, r in age.iterrows()},
         "floors": {f: {"auroc": float(m.loc[(f, "auroc"), "value"]), "brier": float(m.loc[(f, "brier"), "value"])} for f in floors},
         "floor_comparisons": E["floor_comparisons"].to_dict("records"),
         "operating_points": E["operating_points"].to_dict("records"),
         "subgroups": E["subgroups"].to_dict("records"), "risk_groups": E["risk_groups"].to_dict("records"),
         "cause": E["cause"],
         "cv": {"auroc": cv_auc, "ci": cv_ci, "regression_floor": float(R["final"]["regression_floor"])},
         "model": {"chosen_column": fm.info.get("chosen_column"), "primary": fm.primary, "arm": fm.arm}}

    _figures(st, E, tag)
    order = np.argsort(p)
    ids = list(dict.fromkeys(int(order[min(int(q * len(order)), len(order) - 1)]) for q in rc["example_quantiles"]))
    raw = pd.concat([full.frame[cfg.blocks["clinical"]], full.mutations], axis=1).loc[ho[ids]]
    recs = ex.records(raw.reset_index(drop=True), list(ho[ids]))
    st.text("holdout_example_records.json", json.dumps(_jsonable(recs), indent=2, default=str),
            "Structured patient records for example holdout patients (out of sample: the model never saw them)" + tag)
    st.text("holdout_report.md", T.holdout_report(F), "Holdout report (generated from the results)" + tag)
    F["seconds"] = round(time.time() - t0)
    if not dry_run:
        write_marker(cfg, status="complete", completed=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     holdout_auroc=F["final"]["auroc"]["mean"])
        run.record_external(cfg.path("holdout_scored"), "one-time holdout scoring marker", "metabric.report.holdout")
    st.log.info("holdout %sscored in %d s: AUROC %.3f", "dry-run " if dry_run else "", F["seconds"], F["final"]["auroc"]["mean"])
    return F


def _figures(st, E, tag):
    rel = E["reliability"]
    fig, ax = P.fig(5.2, 5)
    ax.plot([0, 1], [0, 1], color=P.NEUTRAL, linestyle="--", linewidth=1)
    ax.plot(rel.mean_pred, rel.obs_rate, marker="o", color=P.CAT[0], label="final model")
    ax.set_xlabel("Mean predicted P(death) (holdout, quantile bins)")
    ax.set_ylabel("Observed death rate")
    ax.set_title("Holdout reliability")
    ax.legend()
    P.grid(ax, "both")
    st.figure("holdout_reliability", fig, rel, "Reliability of the final model on the holdout (10 quantile bins)" + tag)
    gr = E["risk_groups"]
    fig, ax = P.fig(5.6, 3.4)
    x = np.arange(len(gr))
    ax.bar(x - 0.18, gr.observed_death_rate, color=P.CAT[1], width=0.36, label="holdout")
    ax.errorbar(x - 0.18, gr.observed_death_rate, yerr=[gr.observed_death_rate - gr.ci_low, gr.ci_high - gr.observed_death_rate],
                fmt="none", ecolor=P.INK2, capsize=3)
    ax.bar(x + 0.18, gr.training_observed_death_rate.astype(float), color=P.CAT[0], width=0.36, label="training (out-of-fold)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.risk_group}\n(n={r.n})" for r in gr.itertuples()], fontsize=8)
    ax.set_ylabel("Observed death rate")
    ax.set_title("Risk groups: holdout vs training")
    ax.legend(fontsize=8)
    P.grid(ax, "y")
    st.figure("holdout_risk_groups", fig, gr, "Observed death rate (Wilson 95% CI) in the training-defined risk groups" + tag)
