"""Phase 2 step 2c: the final survival model, the patient survival output, and the one-time holdout scoring.

`python run.py phase2 --step final --models-run M`
  1. Reads the final member chosen in models run M (`final_choice`) and its cross-validated metrics.
  2. Chooses that member's configuration on all training rows with `deploy_folds` stratified folds (best mean
     C, simplest within 1 SE, as in the nested CV), refits it on all training rows, and keeps the out-of-fold
     risks for the risk-group cutpoints and percentiles. 50 bootstrap refits give the patient interval.
  3. Saves `models/phase2_final.joblib` and writes the final model's cross-validated risk groups and
     calibration (from M's outer predictions), example patient records and a generated report.

`python run.py phase2 --step holdout --final-run F [--dry-run]`: scores the frozen holdout once with the
saved model (marker `paths.phase2_holdout_scored`; same guard as Phase 1). `--dry-run` shuffles the
holdout outcomes (time and event together) and writes no marker.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from ..core.artifacts import Run
from ..core.config import Config
from ..core.paths import sha256_file
from ..core.stats import kaplan_meier, wilson
from ..data import load_dataset
from ..eda import plots as P
from ..harness.floors import design_matrix
from ..report.deploy import deployment_folds
from ..report.holdout import check_marker, write_marker
from ..report.stage import _jsonable
from ..splits import load_holdout, training_ids
from .cv import calibration_by_model, configurations, survival_metrics
from .models import columns_for, make
from .survival import structured, survival_data

DISCLAIMER = ("This is a statistical estimate of overall survival (death from any cause) for patients like this one in the "
              "METABRIC cohort, given the treatment they received. It is not a diagnosis, not treatment advice and not a "
              "prediction of what will happen to this person. Factors are associations in the model, not causes.")
LABELS = ["low", "below average", "above average", "high"]


# ---------------------------------------------------------------- the saved model
@dataclass
class SurvivalModel:
    member: str
    config: str
    params: dict
    model: object                       # fitted member (CoxMember)
    columns: list[str]                  # raw input columns
    horizons: list[float]
    risk_horizon: float
    train_oof_risk: np.ndarray          # out-of-fold P(death by risk_horizon), training rows
    cut: np.ndarray                     # risk-group cutpoints (quartiles of train_oof_risk)
    group_km: dict                      # risk group -> KM survival at the horizons (training, out-of-fold)
    boot: list = field(default_factory=list)
    zbar: np.ndarray | None = None      # mean transformed design row (training): reference for contributions
    numeric_range: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    curve_months: list = field(default_factory=list)

    def survival(self, X: pd.DataFrame, times) -> np.ndarray:
        return self.model.survival(X[self.columns], times)

    def interval(self, X: pd.DataFrame, times, level: float = 0.95):
        if not self.boot:
            return None, None
        B = np.stack([b.survival(X[self.columns], times) for b in self.boot])
        a = (1 - level) / 2
        return np.quantile(B, a, axis=0), np.quantile(B, 1 - a, axis=0)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Log-hazard contribution of each raw column relative to the average training patient (Cox: exact)."""
        m = self.model
        Z = m.prep_.transform(X[self.columns])
        names = list(m.prep_.get_feature_names_out())
        raw = sorted(self.columns, key=len, reverse=True)

        def owner(n):
            for c in raw:
                if n == c or n.startswith(c + "_") or n == f"missingindicator_{c}":
                    return c
            return n
        own = [owner(n) for n in names]
        C = (Z - self.zbar) * m.m_.coef_
        return pd.DataFrame(C, columns=own, index=X.index).T.groupby(level=0).sum().T

    def records(self, X: pd.DataFrame, ids=None) -> list[dict]:
        ids = list(X.index if ids is None else ids)
        H = self.horizons
        S = self.survival(X, H + [self.risk_horizon])
        lo, hi = self.interval(X, H)
        curve = self.survival(X, self.curve_months)
        C = self.contributions(X)
        out = []
        for i, pid in enumerate(ids):
            risk = 1 - S[i, -1]
            g = int(np.digitize(risk, self.cut))
            c = C.iloc[i].sort_values(key=np.abs, ascending=False)
            flags = [f"{k} = {X.iloc[i][k]} is outside the training range {v}" for k, v in self.numeric_range.items()
                     if pd.notna(X.iloc[i][k]) and not (v[0] <= X.iloc[i][k] <= v[1])]
            out.append({
                "patient_id": pid,
                "inputs": {k: (None if pd.isna(v) else v) for k, v in X.iloc[i][self.columns].items()},
                "missing_inputs": [k for k in self.columns if pd.isna(X.iloc[i][k])],
                "survival": {f"{int(h)}_months": {"probability": float(S[i, j]),
                                                   "interval_95": None if lo is None else [float(lo[i, j]), float(hi[i, j])]}
                             for j, h in enumerate(H)},
                "survival_curve": [{"month": int(t), "survival": float(s)} for t, s in zip(self.curve_months, curve[i])],
                "risk": {"p_death_by_months": int(self.risk_horizon), "value": float(risk),
                         "percentile_among_training": float((self.train_oof_risk < risk).mean() * 100),
                         "risk_group": LABELS[g], "group_observed": self.group_km[LABELS[g]]},
                "contributions": {"reference": "average training patient", "unit": "log hazard",
                                  "sum": float(c.sum()),
                                  "higher_hazard": [{"input": k, "log_hazard": float(v)} for k, v in c.items() if v > 0][:5],
                                  "lower_hazard": [{"input": k, "log_hazard": float(v)} for k, v in c.items() if v < 0][:5],
                                  "wording": "associated with higher / lower hazard of death in the model"},
                "flags": flags,
                "model_quality": self.quality,
                "disclaimer": DISCLAIMER,
            })
        return out


# ---------------------------------------------------------------- deployment fit
def _fit_eval(name, kind, params, cfg, seed, X, sv, tr, te, risk_h):
    m = make(kind, cfg, params, seed).fit(X.loc[tr], sv.time.loc[tr].to_numpy(), sv.event.loc[tr].to_numpy())
    S = m.survival(X.loc[te], [risk_h])
    return name, te, 1 - S[:, 0]


def run_final(cfg: Config, run: Run, models_run: str) -> dict:
    from sksurv.metrics import concordance_index_censored
    st = run.stage("phase2")
    pc, dc = cfg["phase2"], cfg["phase2"]["deploy"]
    t0 = time.time()
    md = cfg.path("outputs") / models_run / "phase2" / "tables"
    for f in ("final_choice.csv", "summary.csv", "outer_predictions.csv", "calibration_by_model.csv"):
        run.record_input(md / f, f"phase2 models run {models_run}")
    fc = pd.read_csv(md / "final_choice.csv")
    member = str(fc.loc[fc.chosen, "model"].iloc[0])
    summ = pd.read_csv(md / "summary.csv")
    full = load_dataset(cfg)
    tr = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows only are used)")
    ds = full.subset(tr)
    sv, _ = survival_data(cfg, ds)
    X = design_matrix(ds)
    H, risk_h = [float(h) for h in pc["horizons"]], float(pc["risk_horizon"])
    confs = [c for c in configurations(pc["members"]) if c["member"] == member]
    if confs[0]["kind"] != "cox":               # the patient contributions below are exact for the ridge Cox only
        raise NotImplementedError(f"the final-model step supports the cox member; {member} is {confs[0]['kind']}")
    cols = columns_for(cfg, confs[0]["kind"], confs[0]["params"])
    folds = deployment_folds(sv.event, dc["deploy_folds"], cfg.seed)

    # configuration: best mean C over the deployment folds, simplest within 1 SE
    rows, oof = [], {}
    for c in confs:
        cs = []
        o = pd.Series(np.nan, index=sv.index)
        for a, b in folds:
            _, te, r = _fit_eval(c["name"], c["kind"], c["params"], cfg, cfg.seed, X[cols], sv, a, b, risk_h)
            o.loc[te] = r
            cs.append(concordance_index_censored(sv.event.loc[te].to_numpy().astype(bool), sv.time.loc[te].to_numpy(), r)[0])
        oof[c["name"]] = o
        rows.append({"config": c["name"], "complexity": c["complexity"], "mean_c": float(np.mean(cs)),
                     "se": float(np.std(cs, ddof=1) / np.sqrt(len(cs)))})
    dt = pd.DataFrame(rows)
    top = dt.loc[dt.mean_c.idxmax()]
    dt["within_se"] = dt.mean_c >= top.mean_c - pc["select_within_se"] * top.se
    chosen = dt[dt.within_se].sort_values("complexity").iloc[0].config
    dt["chosen"] = dt.config == chosen
    st.table("deploy_configurations", dt, f"Deployment: {member} configurations on {dc['deploy_folds']} folds of all training rows")
    cconf = next(c for c in confs if c["name"] == chosen)
    model = make(cconf["kind"], cfg, cconf["params"], cfg.seed).fit(X[cols], sv.time.to_numpy(), sv.event.to_numpy())
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(dc["n_boot"]):
        i = rng.integers(0, len(sv), len(sv))
        boot.append(make(cconf["kind"], cfg, cconf["params"], cfg.seed).fit(X[cols].iloc[i], sv.time.to_numpy()[i],
                                                                            sv.event.to_numpy()[i]))
    o = oof[chosen].to_numpy()
    cut = np.quantile(o, [0.25, 0.5, 0.75])
    g = np.digitize(o, cut)
    gkm = {}
    for k, lab in enumerate(LABELS):
        km = kaplan_meier(sv.time.to_numpy()[g == k], sv.event.to_numpy()[g == k])
        gkm[lab] = {"n": int((g == k).sum()), **{f"km_survival_{int(h)}": float(km[km.time <= h].survival.iloc[-1]) for h in H}}
    num = [c for c in cfg["clinical_types"]["numeric"] + cfg["clinical_types"]["ordinal"] if c in cols]
    met = lambda k: summ[(summ.model == member) & (summ.metric == k)].iloc[0]  # noqa: E731
    cal = pd.read_csv(md / "calibration_by_model.csv")
    quality = {"cv_harrell_c": [float(met("c_index")["mean"]), float(met("c_index")["ci_low"]), float(met("c_index")["ci_high"])],
               "cv_auc_5y": float(met("auc_60")["mean"]), "cv_auc_10y": float(met("auc_120")["mean"]),
               "cv_calibration_slope": {int(r.horizon_months): float(r.logit_slope_deciles) for r in cal[cal.model == member].itertuples()},
               "evaluated_on": "nested cross-validation on the training rows"}
    sm = SurvivalModel(member=member, config=chosen, params=cconf["params"], model=model, columns=cols, horizons=H, risk_horizon=risk_h,
                       train_oof_risk=o, cut=cut, group_km=gkm, boot=boot,
                       zbar=model.prep_.transform(X[cols]).mean(axis=0),
                       numeric_range={c: [float(X[c].min()), float(X[c].max())] for c in num},
                       quality=quality, curve_months=list(range(0, int(dc["curve_max_months"]) + 1, int(dc["curve_step_months"]))))
    mdir = run.dir / "models"
    mdir.mkdir(exist_ok=True)
    bundle = mdir / "phase2_final.joblib"
    joblib.dump({"model": sm, "config_hash": cfg.hash, "run_id": run.run_id, "models_run": models_run}, bundle)
    if not np.allclose(joblib.load(bundle)["model"].survival(X.iloc[:50], H), sm.survival(X.iloc[:50], H)):
        raise RuntimeError("the saved Phase 2 model does not reproduce the fitted model")
    run._register(bundle, "models", "model", "Final Phase 2 survival model fitted on all training rows (load with joblib; "
                  ".records(X) gives patient survival records)", "metabric.phase2.final")

    # cross-validated risk groups and calibration of the final member (models run, first repeat)
    pr = pd.read_csv(md / "outer_predictions.csv")
    pr = pr[(pr.model == member) & (pr.repeat == 0)]
    cvrg = _cv_risk_groups(st, pr, H)
    # contributions check and example records (in-sample illustration)
    C = sm.contributions(X.iloc[:200])
    lp = np.asarray(model.prep_.transform(X[cols].iloc[:200]), float) @ model.m_.coef_ - float(sm.zbar @ model.m_.coef_)
    err = float(np.abs(C.sum(axis=1).to_numpy() - lp).max())
    if err > 1e-8:
        raise RuntimeError(f"contributions do not add up to the log hazard (error {err})")
    order = np.argsort(o)
    ids = list(dict.fromkeys(int(order[min(int(q * len(o)), len(o) - 1)]) for q in cfg["report"]["example_quantiles"]))
    recs = sm.records(X.iloc[ids], list(X.index[ids]))
    st.text("phase2_example_records.json", json.dumps(_jsonable(recs), indent=2, default=str),
            "Patient survival records for example training patients (in-sample; illustration of the output format)")
    F = {"run": run.run_id, "models_run": models_run, "member": member, "config": chosen, "deploy": dt.to_dict("records"),
         "quality": quality, "risk_groups_train_oof": gkm, "cv_risk_groups": cvrg["groups"], "cv_logrank": cvrg["logrank"],
         "contribution_error": err, "n_boot": dc["n_boot"], "seconds": round(time.time() - t0),
         "coefficients": dict(zip(model.prep_.get_feature_names_out(), map(float, model.m_.coef_)))}
    st.table("final_coefficients", pd.DataFrame({"feature": list(F["coefficients"]), "coef": list(F["coefficients"].values()),
                                                  "hazard_ratio_per_unit": np.exp(list(F["coefficients"].values()))}).sort_values(
             "coef", key=np.abs, ascending=False), "Final Cox model: coefficients (standardised numeric inputs; one-hot categories)")
    from .text import final_summary
    st.text("phase2_final_summary.md", final_summary(F), "Phase 2 final model summary (generated)")
    st.log.info("phase2 final model %s (%s) fitted in %d s", member, chosen, F["seconds"])
    return F


def _cv_risk_groups(st, p: pd.DataFrame, H) -> dict:
    from sksurv.compare import compare_survival
    p = p.assign(group=[LABELS[k] for k in p.risk_group])
    rows, curves = [], []
    for lab in LABELS:
        g = p[p.group == lab]
        km = kaplan_meier(g.time, g.event)
        km.insert(0, "group", lab)
        curves.append(km)
        r = {"risk_group": lab, "n": int(len(g)), "deaths": int(g.event.sum())}
        for h in H:
            k = km[km.time <= h].iloc[-1]
            r |= {f"km_survival_{int(h)}": float(k.survival), f"ci_low_{int(h)}": float(k.ci_low),
                  f"ci_high_{int(h)}": float(k.ci_high), f"mean_predicted_{int(h)}": float(g[f"S_{int(h)}"].mean())}
        rows.append(r)
    chi2, pv = compare_survival(structured(p.time, p.event), p.group.to_numpy())
    st.table("final_cv_risk_groups", pd.DataFrame(rows), "Final model, cross-validated: KM survival by risk group vs mean predicted")
    curves = pd.concat(curves, ignore_index=True)
    fig, ax = P.fig(6.0, 4.2)
    for i, (g, d) in enumerate(curves.groupby("group", sort=False)):
        ax.step(d.time, d.survival, where="post", color=P.CAT[i], label=f"{g} (n={int(d.n_at_risk.iloc[0])})")
        ax.fill_between(d.time, d.ci_low, d.ci_high, step="post", color=P.CAT[i], alpha=0.12, linewidth=0)
    ax.set_xlabel("Months since diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Final model: survival by risk group (log-rank p = {pv:.2g})")
    ax.legend(fontsize=7)
    P.grid(ax, "both")
    st.figure("final_cv_risk_group_km", fig, curves, "Final model, outer-fold predictions (first repeat): KM by risk group")
    cal = calibration_by_model(p.assign(model="final"), H)
    st.table("final_cv_calibration", cal, "Final model, cross-validated calibration by decile")
    return {"groups": rows, "logrank": {"chi2": float(chi2), "df": 3, "p": float(pv)}}


# ---------------------------------------------------------------- one-time holdout scoring
def run_phase2_holdout(cfg: Config, run: Run, final_run: str, dry_run: bool = False) -> dict:
    from .models import make as mk
    st = run.stage("phase2")
    pc = cfg["phase2"]
    t0 = time.time()
    bundle = cfg.path("outputs") / final_run / "models" / "phase2_final.joblib"
    bsha = sha256_file(bundle)
    run.record_input(bundle, f"final Phase 2 model from run {final_run}")
    key = "phase2_holdout_scored"
    if not dry_run:
        prev = check_marker(cfg, bsha, key)
        if prev:
            st.log.info("resuming an interrupted Phase 2 holdout scoring started %s", prev.get("started"))
        write_marker(cfg, key, status="started", started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     run_id=run.run_id, final_run=final_run, bundle_sha256=bsha, config_hash=cfg.hash)
    sm: SurvivalModel = joblib.load(bundle)["model"]
    full = load_dataset(cfg)
    h = load_holdout(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (scored here, once, for Phase 2)")
    ho = pd.Index(h.loc[h["set"] == "holdout", "patient_id"])
    tr = pd.Index(h.loc[h["set"] == "train", "patient_id"])
    sv_all, _ = survival_data(cfg, full)
    X = design_matrix(full)
    svh = sv_all.loc[ho].copy()
    if dry_run:                                             # shuffle outcomes (time and event together)
        perm = np.random.default_rng(cfg.seed + 7).permutation(len(ho))
        svh[["time", "event"]] = svh[["time", "event"]].to_numpy()[perm]
    svt = sv_all.loc[tr]
    H, risk_h = sm.horizons, sm.risk_horizon
    times = H + ([risk_h] if risk_h not in H else [])
    ri = times.index(risk_h)
    preds = {"final": sm.survival(X.loc[ho], times)}
    for f in configurations(pc["floors"]):
        m = mk(f["kind"], cfg, f["params"], cfg.seed).fit(X.loc[tr, columns_for(cfg, f["kind"], f["params"])], svt.time.to_numpy(),
                                                           svt.event.to_numpy())
        preds[f["name"]] = m.survival(X.loc[ho, columns_for(cfg, f["kind"], f["params"])], times)
    trt, tre = svt.time.to_numpy(), svt.event.to_numpy()
    te_t, te_e = svh.time.to_numpy(), svh.event.to_numpy()
    point = {k: survival_metrics(trt, tre, te_t, te_e, S, H, ri) for k, S in preds.items()}
    n, B = len(ho), int(pc["holdout_n_boot"])
    rng = np.random.default_rng(cfg.seed)
    boot = {k: [] for k in preds}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if te_e[i].sum() == 0:
            continue
        for k, S in preds.items():
            boot[k].append(survival_metrics(trt, tre, te_t[i], te_e[i], S[i], H, ri))
    keys = ["c_index", "auc_60", "auc_120", "brier_60", "brier_120", "cal_large_60"]
    rows = []
    for k in preds:
        bd = pd.DataFrame(boot[k])
        for m in keys:
            rows.append({"model": k, "metric": m, "value": point[k][m], "ci_low": float(bd[m].quantile(0.025)),
                         "ci_high": float(bd[m].quantile(0.975))})
    met = pd.DataFrame(rows)
    comp = []
    for f in [x for x in preds if x != "final"]:
        a, b = pd.DataFrame(boot["final"]), pd.DataFrame(boot[f])
        for m in ("c_index", "auc_60", "auc_120", "brier_60"):
            d = (a[m] - b[m]).dropna().to_numpy()
            comp.append({"a": "final", "b": f, "metric": m, "diff": point["final"][m] - point[f][m],
                         "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
                         "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    comp = pd.DataFrame(comp)
    tag = " (DRY RUN: holdout outcomes shuffled; numbers meaningless)" if dry_run else ""
    st.table("phase2_holdout_metrics", met, "Phase 2 holdout: metrics with 95% patient-bootstrap CI" + tag)
    st.table("phase2_holdout_comparisons", comp, "Phase 2 holdout: final model minus each floor, paired bootstrap" + tag)
    S = preds["final"]
    g = np.digitize(1 - S[:, ri], sm.cut)
    rg = []
    for k, lab in enumerate(LABELS):
        m = g == k
        km = kaplan_meier(te_t[m], te_e[m]) if m.any() else None
        r = {"risk_group": lab, "n": int(m.sum()), "deaths": int(te_e[m].sum())}
        for j, hh in enumerate(H):
            r[f"km_survival_{int(hh)}"] = float(km[km.time <= hh].survival.iloc[-1]) if km is not None else float("nan")
            r[f"mean_predicted_{int(hh)}"] = float(S[m, j].mean()) if m.any() else float("nan")
            r[f"training_km_survival_{int(hh)}"] = sm.group_km[lab][f"km_survival_{int(hh)}"]
        rg.append(r)
    rg = pd.DataFrame(rg)
    st.table("phase2_holdout_risk_groups", rg, "Phase 2 holdout: KM survival in the training-defined risk groups" + tag)
    p = pd.DataFrame({"model": "final", "time": te_t, "event": te_e, **{f"S_{int(t)}": S[:, j] for j, t in enumerate(times)}})
    cal = calibration_by_model(p, H)
    st.table("phase2_holdout_calibration", cal, "Phase 2 holdout: calibration by decile of predicted death" + tag)
    pred = pd.DataFrame({"patient_id": ho, "time": te_t, "event": te_e, **{f"S_{int(t)}": S[:, j] for j, t in enumerate(times)},
                         "risk_group": [LABELS[k] for k in g]})
    st.table("phase2_holdout_predictions", pred, "Phase 2 holdout: every patient's predicted survival and risk group" + tag)
    from sksurv.compare import compare_survival
    chi2, pv = compare_survival(structured(te_t, te_e), np.array([LABELS[k] for k in g]))
    F = {"dry_run": dry_run, "run": run.run_id, "final_run": final_run, "bundle_sha256": bsha, "member": sm.member, "config": sm.config,
         "n": n, "events": int(te_e.sum()), "n_boot": len(boot["final"]), "metrics": met.to_dict("records"),
         "comparisons": comp.to_dict("records"), "risk_groups": rg.to_dict("records"), "logrank": {"chi2": float(chi2), "p": float(pv)},
         "calibration": cal.to_dict("records"), "cv": sm.quality}
    from .text import holdout_summary
    st.text("phase2_holdout_report.md", holdout_summary(F), "Phase 2 holdout report (generated)" + tag)
    if not dry_run:
        write_marker(cfg, key, status="complete", completed=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     holdout_c_index=point["final"]["c_index"])
        run.record_external(cfg.path(key), "one-time Phase 2 holdout scoring marker", "metabric.phase2.final")
    st.log.info("phase2 holdout %sscored in %d s: C %.3f", "dry-run " if dry_run else "", time.time() - t0, point["final"]["c_index"])
    return F
