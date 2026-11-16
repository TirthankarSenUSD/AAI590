"""Rerunnable EDA stage (PLAN.md section 17).

Modes:
  descriptive - all 1,904 patients; for the report only, never for modelling choices.
  modelling   - training rows only (frozen holdout excluded). The only EDA allowed to inform choices.
The holdout rows are read in modelling mode for one purpose only: the split-balance check (item 10).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..artifacts import Run, Stage, _caller
from ..config import Config
from ..data import Dataset, load_dataset, read_raw
from ..splits import load_holdout
from . import plots as P
from . import sections as S

LABELS = {"descriptive": "full cohort, descriptive only",
          "modelling": "training rows only, holdout excluded"}


@dataclass
class Ctx:
    cfg: Config
    stage: Stage
    mode: str
    ds: Dataset
    rng: np.random.Generator
    facts: dict = field(default_factory=dict)

    frame = property(lambda s: s.ds.frame)
    y = property(lambda s: s.ds.y)
    muts = property(lambda s: s.ds.mutations)
    log = property(lambda s: s.stage.log)
    label = property(lambda s: LABELS[s.mode])

    def table(self, name, df, caption, index=False):
        return self.stage.table(f"{self.mode}__{name}", df, f"[{self.label}] {caption}", index=index, creator=_caller(2))

    def figure(self, name, fig, data, caption):
        fig.text(0.995, 0.005, self.label, ha="right", va="bottom", fontsize=6, color=P.INK2)
        return self.stage.figure(f"{self.mode}__{name}", fig, data, f"[{self.label}] {caption}", creator=_caller(2))

    def text(self, name, content, caption):
        return self.stage.text(f"{self.mode}__{name}", content, f"[{self.label}] {caption}", creator=_caller(2))


def run_eda(cfg: Config, run: Run, mode: str) -> dict:
    if mode not in LABELS:
        raise ValueError(mode)
    stage = run.stage("eda")
    full = load_dataset(cfg)
    recorded = {e["path"] for e in run.manifest["inputs"]}
    if cfg.raw["paths"]["data"] not in recorded:
        run.record_input(cfg.path("data"), "dataset")
    raw = read_raw(cfg).set_index(cfg.blocks["id"][0])
    hold = None
    if mode == "modelling":
        hold = load_holdout(cfg, full)                    # verifies the frozen hash
        if cfg.raw["paths"]["holdout"] not in recorded:
            run.record_input(cfg.path("holdout"), "frozen holdout (read-only)")
        train = pd.Index(hold.loc[hold["set"] == "train", "patient_id"])
        ds, raw = full.subset(train), raw.loc[train]
    else:
        ds = full
    ctx = Ctx(cfg, stage, mode, ds, np.random.default_rng(cfg.seed))
    stage.log.info("EDA mode=%s (%s): %d patients", mode, ctx.label, len(ds.y))
    S.audit(ctx, raw)
    S.outcome(ctx)
    S.missingness(ctx)
    S.clinical(ctx)
    S.treatment(ctx)
    S.redundancy(ctx)
    S.rna(ctx)
    S.dna(ctx)
    S.cohort(ctx)
    if mode == "modelling":
        split_diagnostics(ctx, full, hold, run)
    else:
        reference_check(ctx)
    ctx.table("facts", pd.DataFrame([{"key": k, "value": json.dumps(v, default=_np)} for k, v in ctx.facts.items()]),
              "Key numbers computed in this run (feed the reference check and the summary)")
    ctx.text("eda_summary.md", summary(ctx, run), "Auto-generated EDA summary (computed wording)")
    return ctx.facts


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


# 10 --------------------------------------------------------------------------------------------
def split_diagnostics(ctx: Ctx, full: Dataset, hold: pd.DataFrame, run: Run) -> None:
    """Balance of the frozen holdout against the training rows. Fails if a stratified variable is off."""
    sp = run.stage("splits")
    sp.table("holdout_assignment", hold, "Copy of the frozen holdout assignment (patient_id, set, stratum)", creator=_caller(1))
    sp.table("holdout_strata", hold.groupby(["stratum", "set"]).size().unstack(fill_value=0).reset_index(),
             "Patients per stratum (outcome x cohort x chemotherapy) in train and holdout", creator=_caller(1))
    rule = ctx.cfg["eda"]["split_balance"]
    s = hold.set_index("patient_id").loc[full.ids, "set"]
    f = full.frame
    tr, ho = s == "train", s == "holdout"
    rows = []

    def add(var, level, x, stratified):
        p1, p2 = float(x[tr].mean()), float(x[ho].mean())
        sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        diff = p2 - p1
        smd = diff / sd if sd > 0 else 0.0
        status = ("fail" if abs(diff) > rule["max_abs_diff_stratified"] else "ok") if stratified else \
                 ("warn" if abs(smd) > rule["warn_smd"] else "ok")
        rows.append({"variable": var, "level": level, "train": p1, "holdout": p2, "diff": diff, "smd": smd,
                     "stratified": stratified, "status": status})

    add("death", "1", full.y, True)
    for c in sorted(f["cohort"].astype(str).unique()):
        add("cohort", c, f["cohort"].astype(str) == c, True)
    add("chemotherapy", "1", f["chemotherapy"], True)
    add("hormone_therapy", "1", f["hormone_therapy"], False)
    add("radio_therapy", "1", f["radio_therapy"], False)
    add("type_of_breast_surgery", "MASTECTOMY", f["type_of_breast_surgery"] == "MASTECTOMY", False)
    add("er_status", "Positive", f["er_status"] == "Positive", False)
    add("tumor_stage", "missing", f["tumor_stage"].isna(), False)
    a_tr, a_ho = f.loc[tr, "age_at_diagnosis"], f.loc[ho, "age_at_diagnosis"]
    from ..stats import smd as _smd
    sm = _smd(a_ho, a_tr)
    rows.append({"variable": "age_at_diagnosis", "level": "mean", "train": a_tr.mean(), "holdout": a_ho.mean(),
                 "diff": a_ho.mean() - a_tr.mean(), "smd": sm, "stratified": False,
                 "status": "warn" if abs(sm) > rule["warn_smd"] else "ok"})
    bal = pd.DataFrame(rows)
    ctx.table("split_balance", bal, f"Holdout versus training balance; stratified variables fail above "
              f"{rule['max_abs_diff_stratified']:.0%} absolute difference, others warn above |SMD| {rule['warn_smd']}")
    ctx.facts.update(n_train=int(tr.sum()), n_holdout=int(ho.sum()), n_balance_fail=int((bal.status == "fail").sum()),
                     n_balance_warn=int((bal.status == "warn").sum()),
                     max_abs_diff_stratified=round(float(bal[bal.stratified]["diff"].abs().max()), 4),
                     max_abs_smd_other=round(float(bal[~bal.stratified]["smd"].abs().max()), 4))
    b = bal[bal.level != "mean"].reset_index(drop=True)
    fig, ax = P.fig(6.2, 0.3 * len(b) + 1.2)
    yy = np.arange(len(b))[::-1]
    lab = [f"{v} = {lv}" for v, lv in zip(b.variable, b.level)]
    ax.hlines(yy, b.train, b.holdout, color=P.NEUTRAL, linewidth=2)
    ax.plot(b.train, yy, "o", color=P.CAT[0], label=f"train (n={int(tr.sum())})", markersize=6)
    ax.plot(b.holdout, yy, "o", color=P.CAT[1], label=f"holdout (n={int(ho.sum())})", markersize=6)
    ax.set_yticks(yy)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("Share of patients")
    ax.set_title("Holdout balance against training rows")
    ax.legend(loc="lower right")
    P.grid(ax, "x")
    ctx.figure("split_balance", fig, b, "Share of patients per variable level in training rows and the frozen holdout")
    if ctx.facts["n_balance_fail"]:
        raise RuntimeError(f"split balance rule broken:\n{bal[bal.status == 'fail'].to_string(index=False)}")


def reference_check(ctx: Ctx) -> pd.DataFrame:
    rows = []
    for r in ctx.cfg["reference_values"]:
        obs = ctx.facts.get(r["key"])
        ok = obs is not None and abs(float(obs) - float(r["expected"])) <= float(r["tol"]) + 1e-9
        rows.append({"key": r["key"], "expected": r["expected"], "observed": obs, "tolerance": r["tol"], "match": bool(ok)})
    rc = pd.DataFrame(rows)
    ctx.table("reference_check", rc, "Reference values (PLAN.md section 17) against this run")
    bad = rc[~rc.match]
    ctx.facts["n_reference_values"] = len(rc)
    ctx.facts["n_reference_mismatch"] = len(bad)
    ctx.facts["reference_mismatches"] = bad[["key", "expected", "observed"]].to_dict("records")
    if len(bad):
        ctx.log.warning("reference mismatches:\n%s", bad.to_string(index=False))
    else:
        ctx.log.info("all %d reference values match", len(rc))
    return rc


# 11 --------------------------------------------------------------------------------------------
def _pct(x, d=1):
    return f"{x:.{d}f}%"


def summary(ctx: Ctx, run: Run) -> str:
    F = ctx.facts
    n = F["n_patients"]
    L = [f"# EDA summary: {ctx.label}", "",
         f"Run `{run.run_id}`, config hash `{ctx.cfg.hash}`, seed {ctx.cfg.seed}. {n:,} patients. "
         "Every number below is computed in this run; wording is chosen by rule from the numbers.", ""]
    if ctx.mode == "descriptive":
        L += ["> Full cohort, descriptive only. Nothing here may inform a modelling choice (A13).", ""]
    else:
        L += ["> Training rows only (frozen holdout excluded). This is the only EDA allowed to inform modelling choices.", ""]
    L += ["## Outcome",
          f"- {F['n_died']:,} of {n:,} patients died ({_pct(F['pct_died'])}); the positive class is death.",
          f"- Causes: {F['cod_disease']} died of disease, {F['cod_other']} of other causes, {F['cod_unknown']} unknown. "
          f"Other-cause deaths are {_pct(100 * F['cod_other'] / F['n_died'])} of all deaths, so the label mixes causes.",
          f"- {F['survivors_fu_lt10y']} of {F['n_alive']} survivors ({_pct(F['pct_survivors_fu_lt10y'])}) have under 10 years of "
          f"follow-up and {F['survivors_fu_lt5y']} under 5 years: the label ignores time (Phase 2).",
          f"- {F['n_time_zero']} patient(s) with follow-up time 0 (Phase 2 rule needed).", ""]
    L += ["## Missingness",
          f"- {F['any_clinical_missing']} patients ({_pct(F['pct_any_clinical_missing'])}) lack at least one clinical value; "
          f"the RNA and DNA blocks have {F['rna_dna_missing_cells']} missing cells.",
          f"- `tumor_stage` is missing for {F['stage_missing']} patients ({_pct(F['pct_stage_missing'])}): "
          f"{_pct(F['pct_stage_missing_cohort4'])} of cohort 4 and {_pct(F['pct_stage_missing_cohort5'])} of cohort 5, "
          "so its missingness marks cohort.",
          f"- Death rate is {_pct(F['death_rate_missing_grade'])} with grade missing (n = {F['n_missing_grade']}) and "
          f"{_pct(F['death_rate_missing_mutation_count'])} with mutation count missing (n = {F['n_missing_mutation_count']}), "
          f"against {_pct(F['pct_died'])} overall: missingness is informative.", ""]
    aa, ae = F["auroc_age_all"], F["auroc_age_excl_other"]
    L += ["## Clinical signal",
          f"- Age alone gives AUROC {aa:.3f} for any death and {ae:.3f} with other-cause deaths set aside "
          f"({'it falls' if ae < aa - 0.01 else 'about the same'}): age {'dominates the any-death label' if aa > max(F['auroc_npi_all'], F['auroc_size_all']) else 'is not the strongest single variable'}.",
          f"- With other-cause deaths set aside, tumour burden {'gains' if F['auroc_npi_excl_other'] > F['auroc_npi_all'] else 'does not gain'}: "
          f"NPI {F['auroc_npi_all']:.3f} -> {F['auroc_npi_excl_other']:.3f}, nodes {F['auroc_nodes_all']:.3f} -> "
          f"{F['auroc_nodes_excl_other']:.3f}, size {F['auroc_size_all']:.3f} -> {F['auroc_size_excl_other']:.3f}.",
          f"- Treatment follows the patient: median age {F['median_age_chemo']:.1f} with chemotherapy versus "
          f"{F['median_age_no_chemo']:.1f} without. Crude treatment-outcome associations are not treatment effects.", ""]
    nf, nmf = F["n_genes_fdr"], F["n_mut_fdr"]
    L += ["## Molecular signal",
          f"- Single-gene AUROCs range {F['gene_auroc_min']:.3f}-{F['gene_auroc_max']:.3f} (median distance from 0.5: "
          f"{F['gene_auroc_median_dist']:.3f}); {F['n_genes_outside_40_60']} genes fall outside 0.40-0.60. "
          f"{'No gene' if nf == 0 else f'{nf} of 489 genes'} pass{'es' if nf == 1 else ''} FDR q < {ctx.cfg['eda']['fdr_alpha']} "
          f"against the permutation null (null 95% range {F['null_auroc_95'][0]:.3f}-{F['null_auroc_95'][1]:.3f}).",
          f"- RNA structure: {F['n_pairs_gt_05']} of {F['n_pairs_total']:,} gene pairs have |r| > 0.5 and {F['n_pairs_gt_07']} > 0.7; "
          f"{F['n_pcs_50']} / {F['n_pcs_80']} / {F['n_pcs_90']} principal components reach 50 / 80 / 90% of variance, "
          f"so the expression signal is {'diffuse' if F['n_pcs_80'] > 100 else 'concentrated'}.",
          f"- {F['n_mut_ge1pct']} mutation columns are mutated in at least 1% of patients ({F['n_mut_lt1pct']} below). "
          f"Their AUROCs range {F['mut_auroc_min']:.3f}-{F['mut_auroc_max']:.3f}; "
          f"{'none survives' if nmf == 0 else ('1 survives' if nmf == 1 else f'{nmf} survive')} FDR correction"
          f" (q < {ctx.cfg['eda']['fdr_alpha']}: {', '.join(r['gene'] for r in F['mut_nominal'] if r['bh_q'] < ctx.cfg['eda']['fdr_alpha']) or 'none'}; "
          f"{len(F['mut_nominal'])} at nominal p < 0.05 before correction).",
          f"- Strongest co-occurrence: {F['max_cooccurrence']['pair']} (phi {F['max_cooccurrence']['phi']}, "
          f"{F['max_cooccurrence']['n_both']} patients with both); strongest mutual exclusivity: "
          f"{F['max_exclusivity']['pair']} (phi {F['max_exclusivity']['phi']}).", ""]
    ca = F["cohort_rna_auroc_macro"]
    word = "strong" if ca >= 0.8 else "moderate" if ca >= 0.65 else "little"
    L += ["## Redundancy and batch",
          f"- ERBB2 expression separates `her2_status` with AUROC {F['auroc_erbb2_her2']:.4f} and GATA3 separates `er_status` "
          f"with {F['auroc_gata3_er']:.3f}; ER IHC agrees with `er_status` for {_pct(F['er_ihc_agreement_pct'])} after the typo fix. "
          "ESR1 is not in the expression panel.",
          f"- The file's NPI matches the standard 0.2 x size(cm) + node stage + grade within 0.1 for "
          f"{_pct(F['npi_standard_pct_within_0.1'])} of complete rows, but a 0.02 size weight within 0.01 for "
          f"{_pct(F['npi_file_fit_pct_within_0.01'])}"
          + (": the file's NPI under-weights tumour size tenfold relative to the standard formula. "
             if F['npi_file_fit_pct_within_0.01'] > 90 and F['npi_standard_pct_within_0.1'] < 50 else ". ")
          + f"As single variables, the file's NPI gives AUROC {F['auroc_npi_file_same_rows']:.3f} and the standard NPI "
          f"{F['auroc_npi_standard_all']:.3f} on the {F['n_npi_standard']:,} rows where the standard NPI can be computed "
          f"({'the standard NPI is stronger' if F['auroc_npi_standard_all'] > F['auroc_npi_file_same_rows'] + 0.005 else 'the standard NPI is weaker' if F['auroc_npi_standard_all'] < F['auroc_npi_file_same_rows'] - 0.005 else 'about the same'}). "
          + "`mutation_count` tracks the number of mutated genes "
          f"(Spearman {F['spearman_mutcount_nmut']:.3f}).",
          ""]
    L += [f"- Cohort can be predicted from RNA with macro AUROC {ca:.3f} (out-of-fold): {word} batch structure "
          f"{'that models could exploit; leave-one-cohort-out matters' if word != 'little' else ''}.", ""]
    L += [f"- Stratification: {F['n_strata']} outcome x cohort x chemotherapy strata, smallest {F['min_stratum']}; "
          f"{F['n_strata_below_folds']} below the {ctx.cfg['cv']['outer_folds']}-fold count.", ""]
    if ctx.mode == "descriptive":
        k, tot = F["n_reference_mismatch"], F["n_reference_values"]
        L += ["## Reference check", (f"- All {tot} reference values match." if k == 0 else
              f"- {k} of {tot} reference values differ: " + "; ".join(
                  f"`{m['key']}` expected {m['expected']}, observed {m['observed']}" for m in F["reference_mismatches"])
              + ". Each is investigated in FINDINGS.md."), ""]
    else:
        L += ["## Split balance",
              f"- {F['n_train']} training and {F['n_holdout']} holdout patients. Largest difference on a stratified variable: "
              f"{F['max_abs_diff_stratified']:.3f}; {F['n_balance_fail']} failures, {F['n_balance_warn']} warnings "
              f"(largest |SMD| on other variables {F['max_abs_smd_other']:.3f}).", ""]
    return "\n".join(L)
