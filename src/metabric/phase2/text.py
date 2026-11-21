"""Phase 2 summary text, generated from the results (no sentence typed by hand)."""
from __future__ import annotations

from ..report.text import compare_line, verdict

NAMES = {"c_index": "Harrell's C", "auc_60": "AUC at 5 years", "auc_120": "AUC at 10 years", "brier_60": "Brier at 5 years",
         "brier_120": "Brier at 10 years", "cal_large_60": "Calibration-in-the-large at 5 years (predicted - observed)"}


def _m(F, model, metric):
    for r in F["summary"]:
        if r["model"] == model and r["metric"] == metric:
            return r
    return None


def _fmt(r, nd=3):
    return "n/a" if r is None else f"{r['mean']:.{nd}f} ({r['ci_low']:.{nd}f} to {r['ci_high']:.{nd}f})"


def models_summary(F: dict) -> str:
    models = list(dict.fromkeys(r["model"] for r in F["summary"]))
    order = ["best"] + [m for m in models if m not in ("best", "cox_age", "cox_npi", "km")] + [m for m in ("cox_age", "cox_npi", "km") if m in models]
    L = ["# Phase 2: survival models through the nested CV (overall survival)", "",
         f"Run `{F['run']}` (config hash `{F['config_hash']}`). Training rows only: {F['n']} patients, {F['events']} deaths; "
         f"{F['n_splits']} outer splits (the Phase 1 frozen splits). Risk score for the C-index and risk groups: predicted "
         f"P(death by {int(F['risk_horizon'])} months). Means with corrected resampled 95% CIs.", "",
         "## 1. Discrimination, error and calibration", "",
         "| Model | " + " | ".join(NAMES[k] for k in ("c_index", "auc_60", "auc_120", "brier_60", "brier_120", "cal_large_60")) + " |",
         "|---|" + "---|" * 6]
    for m in order:
        L.append(f"| {m} | " + " | ".join(_fmt(_m(F, m, k)) for k in ("c_index", "auc_60", "auc_120", "brier_60", "brier_120",
                                                                         "cal_large_60")) + " |")
    L += ["", "`best` is the procedure: in each outer split, the member with the highest inner C. Floors (`cox_age`, `cox_npi`, "
          "`km`) are never eligible. Calibration-in-the-large near 0 means the average predicted 5-year death probability "
          "matches the Kaplan-Meier estimate.", "",
          "- Best member per outer split: " + ", ".join(f"{w['member']} {w['splits_won']}" for w in F["wins"]) + "."]
    fin = F["final"]
    ft = {r["model"]: r for r in F["final_table"]}
    top = max(F["final_table"], key=lambda r: r["mean_c"])
    L += ["", "## Final Phase 2 model", "",
          f"- Rule (fixed beforehand, as in Phase 1): highest mean outer C, the simplest within 1 corrected SE "
          f"(order: {', '.join(r['model'] for r in sorted(F['final_table'], key=lambda r: r['complexity']))}).",
          f"- Highest mean C: `{top['model']}` {top['mean_c']:.3f} (corrected SE {top['se_corrected']:.3f}); within 1 SE: "
          + ", ".join(f"`{r['model']}` {r['mean_c']:.3f}" for r in F["final_table"] if r["within_se_of_best"]) + ".",
          f"- **Final model: `{fin}`** (mean outer C {ft[fin]['mean_c']:.3f})."]
    L += ["", "## Calibration by model (first repeat, deciles of predicted death probability)", "",
          "| Model | Horizon (months) | Logit slope across deciles | Lowest decile: predicted / observed | Highest decile: predicted / observed |",
          "|---|---|---|---|---|"]
    for r in F["calibration_by_model"]:
        L.append(f"| {r['model']} | {r['horizon_months']} | {r['logit_slope_deciles']:.2f} | {r['pred_lowest_decile']:.2f} / "
                 f"{r['obs_lowest_decile']:.2f} | {r['pred_highest_decile']:.2f} / {r['obs_highest_decile']:.2f} |")
    comp = sorted({r["model"] for r in F["calibration_by_model"] if r["logit_slope_deciles"] > 1.5 and r["model"] not in ("km",)})
    L += ["", "A slope of 1 means calibrated; above 1 means the predictions are too compressed toward the average. "
          + (f"Too compressed (slope above 1.5 at some horizon): {', '.join(comp)}." if comp else "No model has a slope above 1.5.")]
    L += ["", "## 2. Paired comparisons (identical splits)", ""]
    for r in F["comparisons"]:
        if r["metric"] in ("c_index", "auc_60", "auc_120", "brier_60"):
            L.append("- " + compare_line(r["a"], r["b"], r["metric"], r["diff"], r["ci_low"], r["ci_high"], r["p"]))
    om = [r for r in F["comparisons"] if r["a"] == "coxnet_C+RNA" and r["b"] == "coxnet_C" and r["metric"] == "c_index"]
    if om:
        v = verdict("c_index", om[0]["diff"], om[0]["ci_low"], om[0]["ci_high"], om[0]["p"])
        L += ["", "**Omics for survival:** adding RNA to the penalised Cox model "
              + ("significantly improved" if v == "significantly better" else "significantly worsened" if v == "significantly worse"
                 else "did not measurably change") + f" Harrell's C ({om[0]['diff']:+.3f})."]
    L += ["", "## 3. Risk groups (best procedure, first repeat, each patient once)", "",
          "| Risk group | n | Deaths | KM 5-year survival (95% CI) | Mean predicted | KM 10-year survival (95% CI) | Mean predicted |",
          "|---|---|---|---|---|---|---|"]
    for r in F["risk_groups"]:
        L.append(f"| {r['risk_group']} | {r['n']} | {r['deaths']} | {r['km_survival_60']:.3f} ({r['ci_low_60']:.3f}-{r['ci_high_60']:.3f}) "
                 f"| {r['mean_predicted_60']:.3f} | {r['km_survival_120']:.3f} ({r['ci_low_120']:.3f}-{r['ci_high_120']:.3f}) | "
                 f"{r['mean_predicted_120']:.3f} |")
    lr = F["logrank"]
    L += ["", f"Log-rank test across the four groups: chi2 {lr['chi2']:.1f} on {lr['df']} df, p = {lr['p']:.2g}.", "",
          "## 4. Configurations chosen on the inner folds", ""]
    for r in F["configs_chosen"]:
        L.append(f"- {r['config']}: {r['n']} of {F['n_splits']} splits.")
    L += ["", "## 5. Limitations", "",
          "- Overall survival mixes cancer and other-cause deaths; competing risks are out of scope.",
          "- Age does not meet proportional hazards (its effect grows with follow-up); Cox hazard ratios are averages over follow-up.",
          "- Single cohort source; the holdout has not been used here."]
    return "\n".join(L)


def final_summary(F: dict) -> str:
    q = F["quality"]
    L = ["# Phase 2: final survival model", "",
         f"Run `{F['run']}`, from models run `{F['models_run']}`. Final member `{F['member']}` (chosen by the rule in the models run); "
         f"configuration on all training rows: `{F['config']}`.", "",
         "## Cross-validated performance (nested CV, training rows)", "",
         f"- Harrell's C {q['cv_harrell_c'][0]:.3f} ({q['cv_harrell_c'][1]:.3f} to {q['cv_harrell_c'][2]:.3f}); AUC at 5 years "
         f"{q['cv_auc_5y']:.3f}, at 10 years {q['cv_auc_10y']:.3f}.",
         "- Calibration slope across deciles: " + ", ".join(f"{k} months {v:.2f}" for k, v in q["cv_calibration_slope"].items()) + ".",
         "", "| Risk group (cross-validated) | n | Deaths | KM 5-year survival | Mean predicted | KM 10-year survival | Mean predicted |",
         "|---|---|---|---|---|---|---|"]
    for r in F["cv_risk_groups"]:
        L.append(f"| {r['risk_group']} | {r['n']} | {r['deaths']} | {r['km_survival_60']:.3f} ({r['ci_low_60']:.3f}-{r['ci_high_60']:.3f}) | "
                 f"{r['mean_predicted_60']:.3f} | {r['km_survival_120']:.3f} ({r['ci_low_120']:.3f}-{r['ci_high_120']:.3f}) | "
                 f"{r['mean_predicted_120']:.3f} |")
    lr = F["cv_logrank"]
    L += ["", f"Log-rank across the risk groups: chi2 {lr['chi2']:.1f} on {lr['df']} df, p = {lr['p']:.2g}.", "",
          "## Deployment fit (all training rows)", ""]
    for r in F["deploy"]:
        L.append(f"- `{r['config']}`: mean C {r['mean_c']:.3f} (SE {r['se']:.3f})" + (" **chosen**" if r["chosen"] else "") + ".")
    L += [f"- Patient interval: {F['n_boot']} bootstrap refits. Contributions add up to the log hazard (max error "
          f"{F['contribution_error']:.1e}).", "", "## Patient output", "",
          "`model.records(X)` returns, per patient: the inputs and which are missing; survival at 5 and 10 years with a 95% bootstrap "
          "interval; the survival curve every 12 months; P(death by 10 years), its percentile among training patients and the risk "
          "group with that group's observed (Kaplan-Meier) survival; the inputs most associated with higher and lower hazard "
          "(log-hazard contributions against the average training patient); out-of-range flags; the model's cross-validated "
          "quality; and a disclaimer. No diagnosis, treatment advice or causal claim."]
    return "\n".join(L)


def holdout_summary(F: dict) -> str:
    m = {(r["model"], r["metric"]): r for r in F["metrics"]}

    def f(k, mod="final"):
        r = m[(mod, k)]
        return f"{r['value']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})"
    c, cv = m[("final", "c_index")], F["cv"]["cv_harrell_c"]
    L = ["# Phase 2 holdout report: overall survival", ""]
    if F["dry_run"]:
        L += ["> **DRY RUN.** The holdout outcomes (time and event together) were shuffled, so every number below is meaningless. "
              "This run checks the code path and the format only; the Phase 2 holdout has not been scored.", ""]
    L += [f"Model: `{F['member']}` (`{F['config']}`) from run `{F['final_run']}` (file sha256 `{F['bundle_sha256'][:12]}`), applied "
          f"unchanged. Holdout: {F['n']} patients, {F['events']} deaths. Intervals: 95% patient bootstrap ({F['n_boot']} resamples). "
          f"Run `{F['run']}`.", "", "## 1. Headline", "",
          f"- **Holdout Harrell's C:** {f('c_index')}. Cross-validated: {cv[0]:.3f} ({cv[1]:.3f}-{cv[2]:.3f}); the cross-validated "
          "estimate " + ("lies inside" if c["ci_low"] <= cv[0] <= c["ci_high"] else "lies outside") + " the holdout interval.",
          "- Nothing was refitted or re-chosen after scoring.", "",
          "| Metric | Final model | Age-only Cox | NPI-only Cox | Kaplan-Meier |", "|---|---|---|---|---|"]
    for k, lab in NAMES.items():
        L.append(f"| {lab} | {f(k)} | {f(k, 'cox_age')} | {f(k, 'cox_npi')} | {f(k, 'km')} |")
    L += ["", "## 2. Against the floors (paired bootstrap, final minus floor)", ""]
    for r in F["comparisons"]:
        L.append("- " + compare_line("final model", r["b"], r["metric"], r["diff"], r["ci_low"], r["ci_high"], r["p"]))
    L += ["", "## 3. Risk groups (training cutpoints)", "",
          "| Risk group | n | Deaths | KM 5-year | Predicted 5-year | Training KM 5-year | KM 10-year | Predicted 10-year | Training KM 10-year |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in F["risk_groups"]:
        L.append(f"| {r['risk_group']} | {r['n']} | {r['deaths']} | {r['km_survival_60']:.3f} | {r['mean_predicted_60']:.3f} | "
                 f"{r['training_km_survival_60']:.3f} | {r['km_survival_120']:.3f} | {r['mean_predicted_120']:.3f} | "
                 f"{r['training_km_survival_120']:.3f} |")
    L += ["", f"Log-rank across the groups: chi2 {F['logrank']['chi2']:.1f}, p = {F['logrank']['p']:.2g}.", "",
          "## 4. Calibration (deciles of predicted death probability)", ""]
    for r in F["calibration"]:
        L.append(f"- {r['horizon_months']} months: logit slope across deciles {r['logit_slope_deciles']:.2f}; lowest decile predicted "
                 f"{r['pred_lowest_decile']:.2f} vs observed {r['obs_lowest_decile']:.2f}; highest {r['pred_highest_decile']:.2f} vs "
                 f"{r['obs_highest_decile']:.2f}.")
    L += ["", "## 5. Limitations", "",
          f"- One holdout of {F['n']} patients; decile calibration rests on about {F['n'] // 10} patients per decile.",
          "- Overall survival mixes causes of death; age does not meet proportional hazards.",
          "- Same cohort source as the training rows (internal validation)."]
    return "\n".join(L)
