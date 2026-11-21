"""`python run.py phase2 --step models`: the survival members through the frozen nested CV (step 2b).

1. Inner folds choose each member's configuration: best mean inner Harrell's C, the simplest within
   `select_within_se` SE of it (grids are listed simplest first).
2. Each member is refitted with its chosen configuration on the outer training rows and scored on the
   outer test rows; the floors likewise. The `best` procedure takes, per outer split, the member whose
   chosen configuration has the highest mean inner C.
3. Metrics per outer split: Harrell's C (risk = predicted P(death by `risk_horizon`)), time-dependent AUC and
   Brier score at the horizons, and calibration-in-the-large at the first horizon (mean predicted death
   probability minus the Kaplan-Meier death probability of the test rows).
4. Means with corrected resampled 95% CIs; paired comparisons on identical splits; risk groups
   (training-fold quartiles) with Kaplan-Meier curves and a log-rank test, and a 5-year calibration plot,
   from the first repeat (each patient once).
Training rows only; the holdout is untouched. Resumable (checkpoints).
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from ..core.artifacts import Run
from ..core.config import Config
from ..core.stats import corrected_resampled_ttest, kaplan_meier
from ..data import load_dataset
from ..eda import plots as P
from ..farm.members import resolve_n_jobs
from ..harness import splits_from_table
from ..harness.floors import design_matrix
from ..importance.stage import _parallel
from ..splits import load_cv_splits, training_ids
from .models import columns_for, make
from .survival import structured, survival_data

METRICS = ("c_index", "auc_60", "auc_120", "brier_60", "brier_120", "cal_large_60")
LOWER_BETTER = {"brier_60", "brier_120"}


# ---------------------------------------------------------------- configurations
def configurations(spec: dict) -> list[dict]:
    """Every member configuration: name, member, kind, params, complexity (grid position, simplest first)."""
    out = []
    for member, m in spec.items():
        grid = m.get("grid", {})
        keys = list(grid)
        for k, vals in enumerate(itertools.product(*[grid[x] for x in keys]) if keys else [()]):
            p = dict(m.get("fixed", {})) | dict(zip(keys, vals))
            name = member + ("" if not keys else "[" + ",".join(f"{a}={b}" for a, b in zip(keys, vals)) + "]")
            out.append({"name": name, "member": member, "kind": m["kind"], "params": p, "complexity": k})
    return out


# ---------------------------------------------------------------- metrics
def survival_metrics(tr_time, tr_event, te_time, te_event, S: np.ndarray, horizons: list[float], risk_col: int) -> dict:
    """S: test survival at `horizons` (n x len(horizons)). Risk for the C-index = 1 - S[:, risk_col]."""
    from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc
    tt = np.minimum(np.asarray(te_time, float), float(np.max(tr_time)) - 1e-6)   # IPCW needs test times within training follow-up
    ytr, yte = structured(tr_time, tr_event), structured(tt, te_event)
    risk = 1 - S[:, risk_col]
    out = {"c_index": float(concordance_index_censored(np.asarray(te_event, bool), np.asarray(te_time, float), risk)[0])}
    for j, h in enumerate(horizons):
        h = float(h)
        est = 1 - S[:, j]
        try:
            out[f"auc_{int(h)}"] = float(cumulative_dynamic_auc(ytr, yte, est, [h])[0][0]) if np.ptp(est) > 0 else 0.5
        except ValueError:
            out[f"auc_{int(h)}"] = float("nan")
        try:
            out[f"brier_{int(h)}"] = float(brier_score(ytr, yte, S[:, [j]], [h])[1][0])
        except ValueError:
            out[f"brier_{int(h)}"] = float("nan")
    km = kaplan_meier(te_time, te_event)
    s_obs = float(km[km.time <= horizons[0]].survival.iloc[-1])
    out[f"cal_large_{int(horizons[0])}"] = float((1 - S[:, 0]).mean() - (1 - s_obs))
    return out


# ---------------------------------------------------------------- tasks (module level: they run in worker processes)
def _inner_task(key, kind, params, cfg, seed, X, sv, tr, te, horizons, risk_h):
    try:
        m = make(kind, cfg, params, seed).fit(X.loc[tr], sv.time.loc[tr].to_numpy(), sv.event.loc[tr].to_numpy())
    except ArithmeticError:                     # numerical failure: this configuration is not eligible in this split
        return {"key": key, "c": float("nan")}
    S = m.survival(X.loc[te], [risk_h])
    if not np.isfinite(S).all():                # diverged fit: same handling as a numerical failure
        return {"key": key, "c": float("nan")}
    from sksurv.metrics import concordance_index_censored
    c = concordance_index_censored(sv.event.loc[te].to_numpy().astype(bool), sv.time.loc[te].to_numpy(), 1 - S[:, 0])[0]
    return {"key": key, "c": float(c)}


def _outer_task(key, kind, params, cfg, seed, X, sv, tr, te, horizons, risk_h):
    t0 = time.time()
    try:
        m = make(kind, cfg, params, seed).fit(X.loc[tr], sv.time.loc[tr].to_numpy(), sv.event.loc[tr].to_numpy())
    except ArithmeticError:                     # numerical failure: recorded as missing metrics for this split
        return {"key": key, "failed": True}
    times = list(horizons) + ([risk_h] if risk_h not in horizons else [])
    S = m.survival(X.loc[te], times)
    Str = m.survival(X.loc[tr], [risk_h])
    if not (np.isfinite(S).all() and np.isfinite(Str).all()):
        return {"key": key, "failed": True}
    return {"key": key, "test_ids": np.asarray(te), "S": S, "times": times, "train_risk": 1 - Str[:, 0],
            "seconds": time.time() - t0}


# ---------------------------------------------------------------- stage
def run_models(cfg: Config, run: Run) -> dict:
    st = run.stage("phase2")
    pc = cfg["phase2"]
    t0 = time.time()
    full = load_dataset(cfg)
    tr_ids = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows only are used)")
    run.record_input(cfg.path("cv_splits"), "frozen nested CV splits (same as Phase 1)")
    ds = full.subset(tr_ids)
    sv, rule = survival_data(cfg, ds)
    X = design_matrix(ds)
    splits = splits_from_table(load_cv_splits(cfg, tr_ids))
    H, risk_h = [float(h) for h in pc["horizons"]], float(pc["risk_horizon"])
    ri = H.index(risk_h) if risk_h in H else len(H)
    seed, n_jobs = cfg.seed, resolve_n_jobs(cfg)
    members, floors = configurations(pc["members"]), configurations(pc["floors"])
    cols = {c["name"]: columns_for(cfg, c["kind"], c["params"]) for c in members + floors}

    def par(fn, tasks, label):
        return _parallel(fn, tasks, n_jobs, st.log.info, label)

    # 1. inner folds: every member configuration
    tasks = [((s.split_id, k, c["name"]), c["kind"], c["params"], cfg, seed, X[cols[c["name"]]], sv, a, b, H, risk_h)
             for s in splits for k, (a, b) in enumerate(s.inner) for c in members]
    st.log.info("inner: %d fits (%d configurations x %d splits x %d inner folds)", len(tasks), len(members), len(splits),
                len(splits[0].inner))
    inner = pd.DataFrame([{"split_id": r["key"][0], "inner_fold": r["key"][1], "config": r["key"][2], "c": r["c"]}
                          for r in par(_inner_task, tasks, "phase2 inner")])
    meta = pd.DataFrame(members).set_index("name")
    failed = inner[inner.c.isna()]
    if len(failed):
        st.log.info("inner: %d fits failed numerically (%s); not eligible in their splits", len(failed), ", ".join(sorted(set(failed.config))))
    st.table("inner_fit_failures", failed, "Inner fits that failed numerically (configuration not eligible in that split)")
    inner = inner[~inner.set_index(["split_id", "config"]).index.isin(failed.set_index(["split_id", "config"]).index)]
    agg = inner.groupby(["split_id", "config"]).c.agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg["member"] = agg.config.map(meta.member)
    agg["complexity"] = agg.config.map(meta.complexity)
    chosen = []
    for (sid, mem), g in agg.groupby(["split_id", "member"]):
        b = g.sort_values("mean", ascending=False).iloc[0]
        pool = g[g["mean"] >= b["mean"] - pc["select_within_se"] * b["se"]]
        w = pool.sort_values(["complexity", "mean"], ascending=[True, False]).iloc[0]
        chosen.append({"split_id": sid, "member": mem, "config": w.config, "inner_c": float(w["mean"])})
    chosen = pd.DataFrame(chosen)
    best = chosen.loc[chosen.groupby("split_id").inner_c.idxmax(), ["split_id", "member"]].rename(columns={"member": "best"})
    chosen = chosen.merge(best, on="split_id")
    st.table("inner_scores", agg, "Inner folds: mean Harrell's C per configuration and outer split")
    st.table("chosen_configurations", chosen, "Configuration chosen per member and outer split (best inner C; simplest within 1 SE); "
             "`best` = member with the highest inner C")

    # 2. outer folds: chosen configurations + floors
    byname = {c["name"]: c for c in members + floors}
    otasks = []
    for s in splits:
        ch = chosen[chosen.split_id == s.split_id]
        for name in list(ch.config) + [f["name"] for f in floors]:
            c = byname[name]
            otasks.append(((s.split_id, name), c["kind"], c["params"], cfg, seed, X[cols[name]], sv, s.train, s.test, H, risk_h))
    outer = {r["key"]: r for r in par(_outer_task, otasks, "phase2 outer")}

    # 3. metrics per outer split
    rows, preds = [], []
    for s in splits:
        ch = chosen[chosen.split_id == s.split_id].set_index("member")
        models = {m: ch.loc[m, "config"] for m in ch.index} | {f["name"]: f["name"] for f in floors}
        models["best"] = ch.loc[ch.best.iloc[0], "config"]
        trt, tre = sv.time.loc[s.train].to_numpy(), sv.event.loc[s.train].to_numpy()
        tet, tee = sv.time.loc[s.test].to_numpy(), sv.event.loc[s.test].to_numpy()
        for model, name in models.items():
            r = outer[(s.split_id, name)]
            if r.get("failed"):
                st.log.info("outer: %s failed numerically in %s; its metrics are missing for that split", name, s.split_id)
                rows.append({"model": model, "config": name, "split_id": s.split_id, "repeat": s.repeat, "outer_fold": s.outer,
                             **{k: float("nan") for k in METRICS}})
                continue
            m = survival_metrics(trt, tre, tet, tee, r["S"], H, ri)
            rows.append({"model": model, "config": name, "split_id": s.split_id, "repeat": s.repeat, "outer_fold": s.outer, **m})
            if model == "best" or s.repeat == 0:
                cut = np.quantile(r["train_risk"], [0.25, 0.5, 0.75])
                preds.append(pd.DataFrame({"model": model, "split_id": s.split_id, "repeat": s.repeat, "patient_id": r["test_ids"],
                                           "time": tet, "event": tee, **{f"S_{int(h)}": r["S"][:, j] for j, h in enumerate(r["times"])},
                                           "risk_group": np.digitize(1 - r["S"][:, ri], cut)}))
    folds = pd.DataFrame(rows)
    preds = pd.concat(preds, ignore_index=True)
    st.table("outer_metrics", folds, "Outer folds: metrics per model and split (best = per-split member chosen on inner C)")
    st.table("outer_predictions", preds[preds.repeat == 0], "Outer-fold survival predictions, first repeat (each training patient once)")

    n_tr = float(np.mean([len(s.train) for s in splits]))
    n_te = float(np.mean([len(s.test) for s in splits]))
    summ = []
    for model, g in folds.groupby("model", sort=False):
        for k in METRICS:
            v = g[k].dropna().to_numpy()
            ci = corrected_resampled_ttest(v, n_tr, n_te)
            summ.append({"model": model, "metric": k, "mean": float(v.mean()), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                         "n_splits": int(len(v))})
    summ = pd.DataFrame(summ)
    st.table("summary", summ, "Mean over outer splits with corrected resampled 95% CI")
    comp = []
    for a, b in pc["comparisons"]:
        A = folds[folds.model == a].set_index("split_id")
        B = folds[folds.model == b].set_index("split_id")
        for k in ("c_index", "auc_60", "auc_120", "brier_60", "brier_120"):
            d = (A[k] - B.loc[A.index, k]).dropna().to_numpy()
            t = corrected_resampled_ttest(d, n_tr, n_te)
            comp.append({"a": a, "b": b, "metric": k, "diff": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"], "p": t["p"]})
    comp = pd.DataFrame(comp)
    st.table("paired_comparisons", comp, "Paired differences a - b on identical splits (corrected resampled t-test)")
    wins = chosen.drop_duplicates("split_id").best.value_counts().rename_axis("member").reset_index(name="splits_won")
    st.table("best_member_wins", wins, "How often each member was the best on inner C")
    final, fin_tab = choose_final(folds, pc["final"], n_tr, n_te)
    st.table("final_choice", fin_tab, "Final Phase 2 model: highest mean outer C, simplest within 1 corrected SE "
             "(complexity order from config)")
    cal_m = calibration_by_model(preds[preds.repeat == 0], H)
    st.table("calibration_by_model", cal_m, "Calibration per model (first repeat): deciles of predicted death probability vs "
             "Kaplan-Meier; logit slope across deciles (1 = calibrated, > 1 = predictions too compressed)")

    # 4. risk groups and calibration (first repeat, best procedure: each training patient once)
    rg, lr, cal = _risk_groups(st, preds[(preds.model == "best") & (preds.repeat == 0)], H, ri)
    F = {"run": run.run_id, "config_hash": cfg.hash, "n": int(len(sv)), "events": int(sv.event.sum()), "n_splits": len(splits),
         "horizons": H, "risk_horizon": risk_h, "summary": summ.to_dict("records"), "comparisons": comp.to_dict("records"),
         "wins": wins.to_dict("records"), "risk_groups": rg.to_dict("records"), "logrank": lr, "calibration": cal.to_dict("records"),
         "final": final, "final_table": fin_tab.to_dict("records"), "calibration_by_model": cal_m.to_dict("records"),
         "configs_chosen": chosen.groupby(["member", "config"]).size().rename("n").reset_index().to_dict("records"),
         "seconds": round(time.time() - t0)}
    _figures(st, summ)
    from .text import models_summary
    st.text("phase2_models_summary.md", models_summary(F), "Phase 2 models summary (generated from the results)")
    st.text("phase2_models_facts.json", json.dumps(F, indent=2, default=float), "Phase 2 models: facts behind the summary")
    st.log.info("phase2 models done in %d s", F["seconds"])
    return F


def choose_final(folds: pd.DataFrame, rule: dict, n_tr: float, n_te: float) -> tuple[str, pd.DataFrame]:
    """Highest mean outer C among the members; the simplest (config order) within `within_se` corrected SE wins."""
    order = rule["complexity_order"]
    rows = []
    for k, m in enumerate(order):
        v = folds.loc[folds.model == m, "c_index"].dropna().to_numpy()
        t = corrected_resampled_ttest(v - v.mean(), n_tr, n_te)          # SE of the mean, corrected for overlap
        rows.append({"model": m, "complexity": k, "mean_c": float(v.mean()), "se_corrected": float(t["se_corrected"])})
    t = pd.DataFrame(rows)
    top = t.loc[t.mean_c.idxmax()]
    t["within_se_of_best"] = t.mean_c >= top.mean_c - rule["within_se"] * top.se_corrected
    final = str(t[t.within_se_of_best].sort_values("complexity").iloc[0].model)
    t["chosen"] = t.model == final
    return final, t


def calibration_by_model(p: pd.DataFrame, H) -> pd.DataFrame:
    """Per model and horizon: mean predicted vs KM-observed death probability in deciles, and the logit slope across deciles."""
    rows = []
    lg = lambda v: np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))  # noqa: E731
    for m, g in p.groupby("model", sort=False):
        for h in H:
            h = int(h)
            pred = 1 - g[f"S_{h}"]
            if pred.nunique() < 10:
                continue
            d = pd.qcut(pred.rank(method="first"), 10, labels=False)
            pr, ob = [], []
            for k in range(10):
                x = g[d == k]
                km = kaplan_meier(x.time, x.event)
                pr.append(float(pred[d == k].mean()))
                ob.append(float(1 - km[km.time <= h].survival.iloc[-1]))
            pr, ob = np.array(pr), np.array(ob)
            rows.append({"model": m, "horizon_months": h, "logit_slope_deciles": float(np.polyfit(lg(pr), lg(ob), 1)[0]),
                         "pred_lowest_decile": pr[0], "obs_lowest_decile": ob[0], "pred_highest_decile": pr[-1],
                         "obs_highest_decile": ob[-1]})
    return pd.DataFrame(rows)


def _risk_groups(st, p: pd.DataFrame, H, ri):
    from sksurv.compare import compare_survival
    labels = ["low", "below average", "above average", "high"]
    p = p.assign(group=[labels[k] for k in p.risk_group])
    curves, rows = [], []
    for lab in labels:
        g = p[p.group == lab]
        km = kaplan_meier(g.time, g.event)
        km.insert(0, "group", lab)
        curves.append(km)
        r = {"risk_group": lab, "n": int(len(g)), "deaths": int(g.event.sum())}
        for j, h in enumerate(H):
            k = km[km.time <= h].iloc[-1]
            r |= {f"km_survival_{int(h)}": float(k.survival), f"ci_low_{int(h)}": float(k.ci_low), f"ci_high_{int(h)}": float(k.ci_high),
                  f"mean_predicted_{int(h)}": float(g[f"S_{int(h)}"].mean())}
        rows.append(r)
    rg = pd.DataFrame(rows)
    chi2, pv = compare_survival(structured(p.time, p.event), p.group.to_numpy())
    lr = {"chi2": float(chi2), "df": 3, "p": float(pv)}
    st.table("risk_groups", rg, "Risk groups (quartiles of training-fold predicted risk): KM survival vs mean predicted, first repeat")
    curves = pd.concat(curves, ignore_index=True)
    fig, ax = P.fig(6.0, 4.2)
    for i, (g, d) in enumerate(curves.groupby("group", sort=False)):
        ax.step(d.time, d.survival, where="post", color=P.CAT[i], label=f"{g} (n={int(d.n_at_risk.iloc[0])})")
        ax.fill_between(d.time, d.ci_low, d.ci_high, step="post", color=P.CAT[i], alpha=0.12, linewidth=0)
    for h in H:
        ax.axvline(h, color=P.NEUTRAL, linestyle=":", linewidth=1)
    ax.set_xlabel("Months since diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Survival by predicted risk group (log-rank p = {pv:.2g})")
    ax.legend(fontsize=7)
    P.grid(ax, "both")
    st.figure("risk_group_km", fig, curves, "Kaplan-Meier by risk group (outer-fold predictions, first repeat; cutpoints from each "
              "training fold)")
    # 5-year calibration: deciles of predicted death by the first horizon vs KM death probability
    h = int(H[0])
    p = p.assign(pred=1 - p[f"S_{h}"])
    p["decile"] = pd.qcut(p.pred.rank(method="first"), 10, labels=False)
    cal = []
    for d, g in p.groupby("decile"):
        km = kaplan_meier(g.time, g.event)
        k = km[km.time <= h].iloc[-1]
        cal.append({"decile": int(d), "n": int(len(g)), "mean_predicted": float(g.pred.mean()), "observed": float(1 - k.survival),
                    "obs_ci_low": float(1 - k.ci_high), "obs_ci_high": float(1 - k.ci_low)})
    cal = pd.DataFrame(cal)
    fig, ax = P.fig(5.2, 5)
    ax.plot([0, 1], [0, 1], color=P.NEUTRAL, linestyle="--", linewidth=1)
    ax.errorbar(cal.mean_predicted, cal.observed, yerr=[cal.observed - cal.obs_ci_low, cal.obs_ci_high - cal.observed],
                fmt="o", color=P.CAT[0], ecolor=P.INK2, capsize=2)
    ax.set_xlabel(f"Mean predicted P(death by {h} months)")
    ax.set_ylabel(f"Observed (Kaplan-Meier) P(death by {h} months)")
    ax.set_title(f"{h // 12}-year calibration (outer folds, first repeat)")
    P.grid(ax, "both")
    st.figure("calibration_5y", fig, cal, f"{h // 12}-year calibration by decile of predicted risk; observed = 1 - KM with 95% CI")
    return rg, lr, cal


def _figures(st, summ):
    d = summ[summ.metric == "c_index"].sort_values("mean")
    fig, ax = P.fig(6.0, 0.4 * len(d) + 1.4)
    y = np.arange(len(d))
    ax.hlines(y, d.ci_low, d.ci_high, color=P.CAT[0], linewidth=2)
    ax.plot(d["mean"], y, "o", color=P.CAT[0])
    ax.set_yticks(y)
    ax.set_yticklabels(d.model, fontsize=8)
    ax.axvline(0.5, color=P.NEUTRAL, linestyle=":", linewidth=1)
    ax.set_xlabel("Harrell's C (mean over outer splits, corrected 95% CI)")
    ax.set_title("Discrimination: members, best procedure and floors")
    P.grid(ax, "x")
    st.figure("c_index", fig, d, "Harrell's C per model over the outer splits with corrected resampled 95% CIs")
