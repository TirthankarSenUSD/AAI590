"""Validation report text, generated from a dictionary of results (no sentence is typed by hand).

Every comparison names one metric for both sides, and every verdict is computed from the paired
difference, its interval and p-value, so changing a number changes the wording.
"""
from __future__ import annotations

LOWER_IS_BETTER = {"brier", "log_loss", "recal_brier", "brier_60", "brier_120"}


def verdict(metric: str, diff: float, lo: float, hi: float, p: float, alpha: float = 0.05) -> str:
    """'better', 'worse' or 'no significant difference' for A minus B on `metric`."""
    if not (p < alpha) or (lo <= 0 <= hi):
        return "no significant difference"
    up = diff > 0
    good = (not up) if metric in LOWER_IS_BETTER else up
    return "significantly better" if good else "significantly worse"


def compare_line(a: str, b: str, metric: str, diff: float, lo: float, hi: float, p: float, alpha: float = 0.05) -> str:
    v = verdict(metric, diff, lo, hi, p, alpha)
    return (f"{a} vs {b} ({metric} on both sides): difference {diff:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}, p = {p:.3f}); "
            + (f"{a} is {v}." if v != "no significant difference" else "no significant difference."))


def _ci(d: dict, k: str, nd: int = 3) -> str:
    m = d.get(k)
    if not m:
        return "n/a"
    return f"{m['mean']:.{nd}f} ({m['ci_low']:.{nd}f}-{m['ci_high']:.{nd}f})"


def omics_conclusion(arm_rows: list[dict], baseline: str, alpha: float = 0.05) -> str:
    better = [r for r in arm_rows if verdict("auroc", r["diff"], r["ci_low"], r["ci_high"], r["p"], alpha) == "significantly better"]
    worse = [r for r in arm_rows if verdict("auroc", r["diff"], r["ci_low"], r["ci_high"], r["p"], alpha) == "significantly worse"]
    if better:
        s = (f"{', '.join(r['arm'] for r in better)} {'was' if len(better) == 1 else 'were'} significantly better than "
             f"{baseline} on AUROC, so the molecular data add measurable signal in this cohort.")
    else:
        s = (f"No arm was significantly better than {baseline} on AUROC; adding RNA or DNA did not measurably improve "
             "prediction of death in this cohort.")
    if worse:
        s += f" Significantly worse than {baseline}: {', '.join(r['arm'] for r in worse)}."
    return s


def validation_report(F: dict) -> str:
    a = F["arm"]
    hd = F["headline"]
    met, flo = F["metrics"], F["floor_metrics"]
    L = [f"# Validation report: METABRIC Phase 1 (death, any cause)", "",
         f"Generated from runs: farm `{F['runs']['farm']}`, ensemble `{F['runs']['ensemble']}`, agreement "
         f"`{F['runs']['agreement']}`; this report run `{F['runs']['report']}` (config hash `{F['config_hash']}`). "
         f"Training rows only ({F['n_train']} patients, {F['n_deaths']} deaths, base rate {F['base_rate']:.3f}); "
         f"nested cross-validation with {F['n_splits']} outer splits.", "",
         "## 1. Headline", "",
         f"- **Final model:** arm `{a}`, option `{hd['option']}`, chosen by rules fixed before the results were seen "
         f"(arm: `{F['baseline']}` unless another arm is significantly better; option: highest mean outer AUROC, simplest within 1 SE).",
         f"- **Honest estimate (select-and-ensemble procedure, outer folds):** AUROC {hd['nested_auroc']:.3f} "
         f"(95% CI {hd['nested_ci'][0]:.3f}-{hd['nested_ci'][1]:.3f}).",
         f"- The chosen option's own mean outer AUROC is {hd['mean_outer_auroc']:.3f} (SD {hd['sd_outer_auroc']:.3f}); it is "
         "slightly optimistic because the option was chosen on those scores.",
         f"- **Regression floor** for later runs: {hd['regression_floor']:.3f} (mean - {F['floor_sd']} SD of outer-fold AUROC).",
         f"- **Deployment fit:** on all training rows the same procedure kept `{F['deploy']['chosen_column']}` "
         f"(weights {', '.join(f'{k} {v:.2f}' for k, v in F['deploy']['weights'].items())}); "
         f"{F['deploy']['primary']} probabilities are primary (out-of-fold calibration slope {F['deploy']['oof_cal_slope_raw']:.2f}).", ""]
    L += ["## 2. Discrimination, calibration and error (procedure `" + F["procedure"] + "`, mean over outer folds, corrected 95% CI)", "",
          "| Metric | Model | Age-only floor |", "|---|---|---|"]
    for k, lab in [("auroc", "AUROC"), ("auprc", "AUPRC"), ("log_loss", "Log loss"), ("brier", "Brier score"),
                   ("brier_skill", "Brier skill (vs base rate)"), ("cal_slope", "Calibration slope, raw"),
                   ("cal_intercept", "Calibration intercept, raw"), ("recal_cal_slope", "Calibration slope, recalibrated"),
                   ("recal_brier", "Brier score, recalibrated")]:
        if k in met:
            L.append(f"| {lab} | {_ci(met, k)} | {_ci(flo, k)} |")
    L += ["", "Thresholds are fixed on the inner folds and applied unchanged to each outer fold.", "",
          "| Operating point | Recall (sensitivity) | Precision (PPV) | F1 | Specificity | NPV |", "|---|---|---|---|---|---|"]
    for pre, lab in [("op", f"{F['op_target']:.0%} recall target"), ("opf1", "F1-maximising")]:
        if f"{pre}_sensitivity" in met:
            L.append(f"| {lab} | {_ci(met, pre + '_sensitivity')} | {_ci(met, pre + '_ppv')} | {_ci(met, pre + '_f1')} | "
                     f"{_ci(met, pre + '_specificity')} | {_ci(met, pre + '_npv')} |")
    if "opf1_sensitivity" not in met:
        L.append("| F1-maximising | not in this farm run (added later; appears in the next farm run) | | | | |")
    g = F["gate"]
    L += ["", f"- **Overfit gate:** train-vs-CV AUROC gap {g['train_cv_gap']:.3f} (limit {g['gap_limit']:.2f}); raw calibration "
          f"slope {g['cal_slope']:.2f}; status **{g['status']}**; primary probabilities in the farm: {g['primary_probabilities']}.",
          f"- **Optimism** (winner's inner AUROC minus its outer AUROC): mean {F['optimism']:+.3f}.",
          f"- **Permutation test** of the full procedure ({F['perm']['n']} label shuffles): mean AUROC {F['perm']['mean']:.3f}, "
          f"max {F['perm']['max']:.3f}; " + ("at chance, as required." if abs(F['perm']['mean'] - 0.5) <= F['perm']['band']
                                              else "**not at chance: investigate before trusting any result.**"), ""]
    L += ["## 3. Against the floors (paired, identical splits)", ""]
    for r in F["floor_compare"]:
        L.append("- " + compare_line(F["procedure"], r["b"], r["metric"], r["diff"], r["ci_low"], r["ci_high"], r["p"]))
    L += ["", "## 4. Each member's own score and winner stability", "", "| Member procedure | AUROC (95% CI) |", "|---|---|"]
    for r in F["members"]:
        L.append(f"| {r['procedure']} | {r['mean']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f}) |")
    ws = F["winner_stability"]
    if ws:
        L += ["", f"- In the {ws['n_splits']} outer splits the `{F['procedure']}` procedure picked {ws['n_distinct_winners']} distinct "
              f"configurations; the most frequent member was `{ws['top_member']}` ({ws['top_member_share']:.0%} of splits)."]
    L += ["", "## 5. Treatment subgroups (risk given the treatment received, not treatment effect)", "",
          "| Subgroup | Level | n | Deaths | AUROC (95% bootstrap CI) | Mean predicted | Observed |", "|---|---|---|---|---|---|---|"]
    for r in F["subgroups"]:
        L.append(f"| {r['subgroup']} | {r['level']} | {r['n']} | {r['deaths']} | {r['auroc']:.3f} ({r['auroc_ci_low']:.3f}-"
                 f"{r['auroc_ci_high']:.3f}) | {r['mean_pred']:.3f} | {r['obs_rate']:.3f} |")
    L += ["", "## 6. Do RNA and DNA add anything?", "", omics_conclusion(F["arm_compare"], F["baseline"]), ""]
    for r in F["arm_compare"]:
        L.append("- " + compare_line(r["arm"], F["baseline"], "auroc", r["diff"], r["ci_low"], r["ci_high"], r["p"]))
    if F.get("ablation"):
        L += ["", "Ablation on identical splits (best-member procedures):", ""]
        for r in F["ablation"]:
            line = f"- {r['procedure']}: AUROC {r['mean_auroc']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})"
            if r.get("vs"):
                line += f"; vs {r['vs']}: {r['diff']:+.3f}, {verdict('auroc', r['diff'], r['diff_ci_low'], r['diff_ci_high'], r['p'])}"
            L.append(line + ".")
    L += ["", "## 7. Model agreement and the patient output", ""]
    d = F.get("disagreement")
    if d:
        L.append(f"- Member disagreement vs excess error on `{a}`: Spearman {d['rho']:+.3f} (95% CI {d['ci_low']:+.3f} to "
                 f"{d['ci_high']:+.3f}); the 'models disagree' flag " + ("**is** included in the patient output." if d["earns"]
                                                                        else "**is not** included (it does not predict error)."))
    c = F.get("cause")
    if c:
        L.append(f"- Cause breakdown: the cancer-death score separates cancer deaths with AUROC {c['auroc_cancer']:.3f}; the "
                 f"other-cause score separates other-cause deaths with AUROC {c['auroc_other']:.3f}. Among the top quarter by total "
                 f"risk, {c['high_mostly_other']:.0%} are high-risk mainly through other-cause death.")
    L += ["- Patient output: probability with a bootstrap interval, percentile and risk group, cause breakdown, Shapley "
          f"contributions grouped by data type, cohort context by `{F['context_col']}`, input-range flags.", "",
          "## 8. Holdout", "",
          ("The holdout has been scored: see the holdout section." if F.get("holdout_scored") else
           f"Not scored. The {F['n_holdout']} holdout patients have not been used for any number in this report; they are "
           "scored once, last."), "",
          "## 9. Limitations", "",
          "- The label is death from any cause with no time horizon: it mixes cancer and other-cause deaths and ignores "
          "follow-up time (handled in Phase 2).",
          "- Single cohort source (METABRIC); no external validation.",
          "- Treatment follows age, stage and subtype, so treatment columns describe risk given the treatment received, "
          "not treatment effects.",
          "- The molecular signal is weak; " + ("no RNA or DNA arm was significantly better than clinical data alone."
                                                 if not any(verdict("auroc", r["diff"], r["ci_low"], r["ci_high"], r["p"]) ==
                                                            "significantly better" for r in F["arm_compare"])
                                                 else "the gain from RNA or DNA is small."),
          "- The file's NPI uses a 0.02 size weight (the standard NPI uses 0.2).",
          "- Importance and contributions describe the model's behaviour, not biology; they are sensitive to correlated features.",
          f"- Compute budget: {F['n_splits']} outer splits ({F['outer_folds']} folds x {F['outer_repeats']} repeats), "
          f"{F['inner_folds']} inner folds, small hyperparameter grids."]
    return "\n".join(L)


FLOOR_LABELS = {"age_only": "age-only floor", "npi_file": "NPI floor (as supplied)", "npi_standard": "standard-NPI floor",
                "majority": "majority-class floor"}


def holdout_report(F: dict) -> str:
    """Holdout report: the final model applied once, unchanged, to the frozen holdout."""
    fin, cv = F["final"], F["cv"]
    a = fin["auroc"]
    L = ["# Holdout report: METABRIC Phase 1 (death, any cause)", ""]
    if F["dry_run"]:
        L += ["> **DRY RUN.** The holdout labels were shuffled before scoring, so every number below is meaningless "
              "(expected near chance). This run checks the code path and the report format only; the holdout has not been scored.", ""]
    L += [f"Model: the final model saved by report run `{F['runs']['report']}` (arm `{F['model']['arm']}`, column "
          f"`{F['model']['chosen_column']}`, {F['model']['primary']} probabilities; file sha256 `{F['bundle_sha256'][:12]}`), "
          f"applied unchanged. Holdout: {F['n']} patients, {F['deaths']} deaths (rate {F['base_rate']:.3f}; training "
          f"{F['train_base_rate']:.3f}). Intervals: 95% patient bootstrap of the holdout ({F['n_boot']} resamples). "
          f"This run `{F['runs']['holdout']}` (config hash `{F['config_hash']}`).", "",
          "## 1. Headline", "",
          f"- **Holdout AUROC:** {a['mean']:.3f} (95% CI {a['ci_low']:.3f}-{a['ci_high']:.3f}).",
          f"- **Cross-validated estimate** (select-and-ensemble procedure, outer folds): {cv['auroc']:.3f} "
          f"(95% CI {cv['ci'][0]:.3f}-{cv['ci'][1]:.3f}). The cross-validated estimate "
          + ("lies inside" if a["ci_low"] <= cv["auroc"] <= a["ci_high"] else "lies outside")
          + f" the holdout interval; holdout minus cross-validated: {a['mean'] - cv['auroc']:+.3f}.",
          f"- **Regression floor** ({cv['regression_floor']:.3f}): the holdout AUROC is "
          + ("above" if a["mean"] >= cv["regression_floor"] else "below") + " it.",
          "- Nothing was refitted, recalibrated or re-chosen after scoring.", "",
          "## 2. Discrimination, calibration and error", "",
          "| Metric | Final model | Age-only floor |", "|---|---|---|"]
    names = {"auroc": "AUROC", "auprc": "AUPRC", "log_loss": "Log loss", "brier": "Brier score",
             "brier_skill": "Brier skill (vs training base rate)", "cal_slope": "Calibration slope",
             "cal_intercept": "Calibration intercept"}
    fl = F.get("floor_metrics_full", {})
    for k, lab in names.items():
        L.append(f"| {lab} | {_ci(fin, k)} | {_ci(fl, k) if fl else 'see table'} |")
    L += ["", "Calibration slope 1 and intercept 0 mean calibrated; a slope below 1 means the predictions are too extreme.", "",
          "## 3. Against the floors (paired bootstrap, final model minus floor)", ""]
    for r in F["floor_comparisons"]:
        L.append("- " + compare_line("final model", FLOOR_LABELS.get(r["b"], r["b"]), r["metric"], r["diff"], r["ci_low"],
                                     r["ci_high"], r["p"]))
    L += ["", "## 4. Operating points (thresholds fixed on training rows, applied unchanged)", "",
          "| Operating point | Threshold | Recall (sensitivity) | Precision (PPV) | F1 | Specificity | NPV |", "|---|---|---|---|---|---|---|"]
    ops = {}
    for r in F["operating_points"]:
        ops.setdefault((r["operating_point"], r["threshold"]), {})[r["metric"]] = {"mean": r["value"], "ci_low": r["ci_low"],
                                                                                  "ci_high": r["ci_high"]}
    for (name, t), d in ops.items():
        L.append(f"| {name} | {t:.3f} | {_ci(d, 'sensitivity')} | {_ci(d, 'ppv')} | {_ci(d, 'f1')} | {_ci(d, 'specificity')} | "
                 f"{_ci(d, 'npv')} |")
    L += ["", "## 5. Treatment subgroups (risk given the treatment received, not treatment effect)", "",
          "| Subgroup | Level | n | Deaths | AUROC (95% bootstrap CI) | Mean predicted | Observed |", "|---|---|---|---|---|---|---|"]
    for r in F["subgroups"]:
        auc = "n/a" if r["auroc"] != r["auroc"] else f"{r['auroc']:.3f} ({r['auroc_ci_low']:.3f}-{r['auroc_ci_high']:.3f})"
        L.append(f"| {r['subgroup']} | {r['level']} | {r['n']} | {r['deaths']} | {auc} | {r['mean_pred']:.3f} | {r['obs_rate']:.3f} |")
    small = [r for r in F["subgroups"] if r["n"] < 30]
    if small:
        L += ["", f"Subgroups with fewer than 30 holdout patients ({', '.join(f'{r['subgroup']}: {r['level']}' for r in small)}) "
              "have wide intervals; read them as descriptive."]
    L += ["", "## 6. Risk groups (cutpoints from the training out-of-fold risk)", "",
          "| Risk group | n | Deaths | Observed death rate (Wilson 95% CI) | Mean predicted | Training observed rate |", "|---|---|---|---|---|---|"]
    for r in F["risk_groups"]:
        L.append(f"| {r['risk_group']} | {r['n']} | {r['deaths']} | {r['observed_death_rate']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f}) "
                 f"| {r['mean_pred']:.3f} | {r['training_observed_death_rate']:.3f} |")
    rates = [r["observed_death_rate"] for r in F["risk_groups"]]
    L += ["", "Observed death rates " + ("rise" if all(x < y for x, y in zip(rates, rates[1:])) else "do not rise")
          + " monotonically across the risk groups."]
    if F.get("cause"):
        c = F["cause"]
        L += ["", "## 7. Cause breakdown", "",
              f"- The cancer-death score separates cancer deaths from everyone else with AUROC {c['auroc_cancer']:.3f}; the other-cause "
              f"score separates other-cause deaths with AUROC {c['auroc_other']:.3f} ({c['n']} patients with a recorded cause)."]
    L += ["", "## 8. Limitations", "",
          f"- One holdout of {F['n']} patients: its interval is wider than the cross-validated one, and a single split can "
          "differ from the average by chance.",
          "- Same cohort source (METABRIC) as the training rows; this is internal, not external, validation.",
          "- The label is death from any cause with no time horizon (handled in Phase 2).",
          "- Treatment columns describe risk given the treatment received, not treatment effects."]
    return "\n".join(L)
