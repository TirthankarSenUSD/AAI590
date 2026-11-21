"""`python run.py phase2 --step checks`: descriptive survival and the model checks (Phase 2, step 2a).

Training rows only; the holdout is untouched. Writes the Kaplan-Meier tables and figures, the
proportional-hazards test and the collinearity check, and a generated summary with the decision the
checks imply (stratify the Cox members on the configured column or not).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..core.artifacts import Run
from ..core.config import Config
from ..data import load_dataset, npi_standard
from ..eda import plots as P
from ..farm.preprocess import column_groups
from ..splits import training_ids
from .checks import collinearity, cox_design, km_by_group, ph_test
from .survival import npi_band, survival_data


def run_checks(cfg: Config, run: Run) -> dict:
    st = run.stage("phase2")
    pc = cfg["phase2"]
    t0 = time.time()
    full = load_dataset(cfg)
    tr = training_ids(cfg, full)
    run.record_input(cfg.path("holdout"), "frozen holdout (training rows only are used)")
    ds = full.subset(tr)
    sv, rule = survival_data(cfg, ds)
    f = ds.frame.copy()
    f["npi_band"] = npi_band(f, pc["npi_bands"])
    f["npi_standard"] = npi_standard(f)
    H = pc["horizons"]
    F = {"n": int(len(sv)), "events": int(sv.event.sum()), "zero_rule": rule, "horizons": H, "config_hash": cfg.hash,
         "run": run.run_id, "median_followup_censored": float(sv.time[sv.event == 0].median())}

    # ---------------------------------------------------------------- Kaplan-Meier
    curves, at, _ = km_by_group(sv.time, sv.event, None, H)
    st.table("km_overall", curves, "Kaplan-Meier overall survival, training rows (Greenwood log-log 95% CI)")
    F["km_overall"] = at.to_dict("records")
    F["km_groups"] = {}
    _km_figure(st, curves, "km_overall", "Overall survival (training rows)")
    for col in pc["km_groups"]:
        c, a, test = km_by_group(sv.time, sv.event, f[col], H)
        st.table(f"km_{col}", a, f"Survival at {H} months by {col} (KM, 95% CI), with counts")
        _km_figure(st, c, f"km_{col}", f"Overall survival by {col} (log-rank p = {test['p']:.2g})" if test else col)
        F["km_groups"][col] = {"at": a.to_dict("records"), "logrank": test}
    st.table("km_logrank", pd.DataFrame([{"grouping": k, **v["logrank"]} for k, v in F["km_groups"].items() if v["logrank"]]),
             "Log-rank tests (missing group left out)")

    # ---------------------------------------------------------------- proportional hazards
    groups = column_groups(cfg, "C")
    X = cox_design(f, groups, cfg["farm"]["rare_level_min_count"])
    ph, scaled, te = ph_test(X, sv.time, sv.event, pc["ph_alpha"])
    st.table("ph_test", ph, "Proportional hazards: Cox on the arm-C clinical columns (training rows); scaled Schoenfeld "
             "residuals vs rank of event time; BH across columns")
    strat = pc["ph_stratify_column"]
    s_rows = ph[ph.column.str.startswith(strat + "=") | (ph.column == strat)]
    F["ph"] = {"n_columns": int(len(ph)), "violated": ph[ph.ph_violated].column.tolist(),
               "strat_column": strat, "strat_rows": s_rows.to_dict("records"),
               "stratify": bool(s_rows.ph_violated.any())}
    _ph_figure(st, ph, X, scaled, te)

    # ---------------------------------------------------------------- collinearity
    corr, vif = collinearity(f, pc["collinearity_columns"])
    st.table("collinearity_spearman", corr.reset_index(names="column"), "Spearman correlations among NPI, stage, size, nodes, grade")
    vif["over_limit"] = vif.vif > pc["vif_limit"]
    st.table("collinearity_vif", vif, f"Variance inflation factors (complete rows; limit {pc['vif_limit']})")
    F["vif"] = vif.to_dict("records")
    F["cox_uses_npi"] = "nottingham_prognostic_index" in groups["numeric"] or "npi_standard" in groups["numeric"]

    st.text("phase2_checks_summary.md", _summary(F), "Phase 2 checks summary (generated from the results)")
    st.text("phase2_checks_facts.json", json.dumps(F, indent=2, default=float), "Phase 2 checks: facts behind the summary")
    st.log.info("phase2 checks done in %d s; stratify on %s: %s", time.time() - t0, strat, F["ph"]["stratify"])
    return F


def _km_figure(st, curves, name, title):
    fig, ax = P.fig(6.0, 4.2)
    for i, (g, d) in enumerate(curves.groupby("group", sort=False)):
        col = P.CAT[i % len(P.CAT)]
        ax.step(d.time, d.survival, where="post", color=col, label=f"{g} (n={int(d.n_at_risk.iloc[0])})")
        ax.fill_between(d.time, d.ci_low, d.ci_high, step="post", color=col, alpha=0.12, linewidth=0)
    for h in (60, 120):
        ax.axvline(h, color=P.NEUTRAL, linestyle=":", linewidth=1)
    ax.set_xlabel("Months since diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    if curves.group.nunique() > 1:
        ax.legend(fontsize=7)
    P.grid(ax, "both")
    st.figure(name, fig, curves, f"{title}: Kaplan-Meier with 95% CI; dotted lines at 5 and 10 years")


def _ph_figure(st, ph, X, scaled, te):
    top = ph.head(4).column.tolist()
    fig, axes = P.fig(8.0, 5.6, nrows=2, ncols=2)
    rows = []
    order = np.argsort(te)
    for ax, c in zip(np.ravel(axes), top):
        j = list(X.columns).index(c)
        r = scaled[order, j]
        tt = te[order]
        k = max(len(r) // 15, 5)
        sm = pd.Series(r).rolling(k, center=True, min_periods=k // 2).mean().to_numpy()
        ax.scatter(tt, r, s=4, color=P.NEUTRAL)
        ax.plot(tt, sm, color=P.CAT[0])
        ax.axhline(float(ph.loc[ph.column == c, "coef"].iloc[0]), color=P.INK2, linestyle="--", linewidth=1)
        p = float(ph.loc[ph.column == c, "ph_p_bh"].iloc[0])
        ax.set_title(f"{c} (BH p = {p:.2g})", fontsize=8)
        ax.set_xlabel("Event time (months)", fontsize=7)
        ax.set_ylabel("Scaled Schoenfeld residual (log HR)", fontsize=7)
        rows += [{"column": c, "time": float(a), "scaled_residual": float(b), "smoothed": float(s)} for a, b, s in zip(tt, r, sm)]
    st.figure("ph_residuals", fig, pd.DataFrame(rows), "Scaled Schoenfeld residuals vs event time for the four columns with the "
              "smallest PH p-values; a flat smooth at the Cox coefficient (dashed) means proportional hazards")


def _summary(F) -> str:
    H = F["horizons"]
    o = {r["horizon_months"]: r for r in F["km_overall"]}
    z = F["zero_rule"]
    L = ["# Phase 2 checks: descriptive survival, proportional hazards, collinearity", "",
         f"Run `{F['run']}` (config hash `{F['config_hash']}`). Training rows only: {F['n']} patients, {F['events']} deaths; "
         f"median follow-up of censored patients {F['median_followup_censored']:.0f} months.", "",
         f"- Zero-time rule (`{z['rule']}`): patients {z['patients']} ({z['events_at_zero']} deaths among them)"
         + (f" set to {z['new_time']:.3f} months." if "new_time" in z else "; none in these rows."), "",
         "## Kaplan-Meier", ""]
    L += [f"- Overall survival at {h} months: {o[h]['survival']:.3f} ({o[h]['ci_low']:.3f}-{o[h]['ci_high']:.3f}), "
          f"{o[h]['n_at_risk']} at risk." for h in H]
    for col, v in F["km_groups"].items():
        lr = v["logrank"]
        parts = [f"{r['group']} {r['survival']:.2f}" for r in v["at"] if r["horizon_months"] == H[0]]
        L.append(f"- By `{col}`: {H[0]}-month survival " + ", ".join(parts)
                 + (f"; log-rank chi2 {lr['chi2']:.1f} on {lr['df']} df, p = {lr['p']:.2g}." if lr else "."))
    ph = F["ph"]
    L += ["", "## Proportional hazards", "",
          f"- {len(ph['violated'])} of {ph['n_columns']} design columns fail the PH test after BH adjustment"
          + (f": {', '.join(ph['violated'])}." if ph["violated"] else ".")]
    for r in ph["strat_rows"]:
        L.append(f"- `{r['column']}`: time correlation of the scaled residuals {r['rho_time']:+.3f}, BH p = {r['ph_p_bh']:.2g}.")
    L.append(f"- **Decision (rule fixed in config):** the Cox members are " + ("" if ph["stratify"] else "**not** ")
             + f"stratified on `{ph['strat_column']}`" + (" (its PH test fails)." if ph["stratify"] else " (its PH test passes)."))
    others = [c for c in ph["violated"] if not c.startswith(ph["strat_column"])]
    if others:
        L.append(f"- Other columns that fail ({', '.join(others)}) are reported; their hazard ratios are averages over follow-up, "
                 "and the random survival forest does not assume proportional hazards.")
    L += ["", "## Collinearity (complete rows)", ""]
    for r in F["vif"]:
        L.append(f"- `{r['column']}`: VIF {r['vif']:.1f}" + (" (over the limit)" if r["over_limit"] else "") + ".")
    L.append("- The Cox members " + ("use" if F["cox_uses_npi"] else "do not use") + " an NPI column next to its components "
             "(NPI is a composite of size, nodes and grade and is a floor only).")
    return "\n".join(L)
