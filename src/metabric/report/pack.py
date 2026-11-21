"""`python run.py report-pack --runs-file FILE`: collect the material for the written report in one place.

Reads only saved results (nothing is refitted or rescored):
1. For every section, the generated summary of the final run of that stage, and every figure the run
   registered (PNG + the CSV of the plotted numbers + the caption from its manifest). Figures are copied into
   this run's `report/figures/` with a global number, so the report can cite "Figure k".
2. Two Phase 2 holdout figures (risk-group Kaplan-Meier, calibration by decile) drawn from the saved holdout
   predictions; the holdout scoring run itself wrote tables only.
3. `report_material.md`: provenance, study design (from the config), key results (from the result tables),
   every section's generated summary with its figures and tables, consolidated limitations, and figure and
   table indexes. Every number in it comes from a result file.

FILE (YAML) names the final run of each stage; see `RUNME.md`.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..core.artifacts import Run
from ..core.config import Config
from ..core.stats import kaplan_meier
from ..eda import plots as P

# (key, title, runs-file key, file filters: summary names, figure-name prefix or None, what the section is for)
SECTIONS = [
    ("data", "Data and cohort (descriptive EDA, all patients)", "eda_descriptive", ["descriptive__eda_summary.md"], "descriptive__",
     "Describes the cohort for the report. Never used for modelling choices."),
    ("splits", "Training rows, holdout and split balance (modelling EDA)", "eda_modelling", ["modelling__eda_summary.md"], "modelling__",
     "Training rows only: balance of the frozen holdout and the nested CV folds."),
    ("harness", "Evaluation design and baselines (harness)", "harness", ["harness_summary.md"], None,
     "Nested CV, selection rule, baselines (floors), permutation test, leave-one-cohort-out, overfit gate."),
    ("farm", "Model farm (clinical arm)", "farm_main", ["farm_summary.md"], None,
     "The seven members and the cause-specific members on the clinical arm: per-member scores, winners, checks."),
    ("importance", "Feature importance and gates", "importance", ["importance_summary.md"], None,
     "Coefficients, SHAP, permutation importance, partial dependence, interactions (run on the 10-split farm; not rerun "
     "for the final 25-split run)."),
    ("ensemble", "Ensembles, arm comparison and the final model", "ensemble", ["ensemble_summary.md"], None,
     "Agreement merge, stacker, averages, arm comparison (the omics question) and the final-model rule."),
    ("agreement", "Model agreement, disagreement flag, cause breakdown, omics ablation", "agreement", ["agreement_summary.md"], None,
     "Do very different models agree on why? Importance heatmaps, rank overlap, prediction correlation, stability."),
    ("validation", "Phase 1 validation report and patient output", "report", ["validation_report.md"], None,
     "Headline Phase 1 results with calibration, operating points, subgroups, risk groups and the deployment fit."),
    ("holdout", "Phase 1 holdout (scored once)", "holdout", ["holdout_report.md"], None,
     "The final Phase 1 model applied once, unchanged, to the 381 holdout patients."),
    ("p2_checks", "Phase 2: descriptive survival and model checks", "phase2_checks", ["phase2_checks_summary.md"], None,
     "Kaplan-Meier curves, proportional hazards, collinearity."),
    ("p2_models", "Phase 2: survival models through the nested CV", "phase2_models", ["phase2_models_summary.md"], None,
     "Cox, penalised Cox (with and without RNA), random survival forest and floors; calibration; the final-model rule."),
    ("p2_final", "Phase 2: final survival model and patient survival output", "phase2_final", ["phase2_final_summary.md"], None,
     "The ridge Cox refitted on all training rows; cross-validated risk groups; the patient survival record."),
    ("p2_holdout", "Phase 2 holdout (scored once)", "phase2_holdout", ["phase2_holdout_report.md"], None,
     "The final survival model applied once, unchanged, to the 381 holdout patients."),
]


def _manifest(out: Path, run_id: str) -> dict:
    m = json.loads((out / run_id / "manifest.json").read_text(encoding="utf-8"))
    if m.get("status") != "complete":
        raise RuntimeError(f"run {run_id} is not complete")
    return m


def _demote(md: str, by: int = 2) -> str:
    """Push a summary's headings down so it nests under its section."""
    return "\n".join(("#" * by + ln) if re.match(r"^#{1,4} ", ln) else ln for ln in md.splitlines())


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def findings_appendix(text: str) -> list[str]:
    """An appendix from a findings log whose entries start with `### <id> · <date> · <topic> · <title>`:
    an index table, entries per topic, then every entry in full (headings pushed down one level)."""
    head = re.compile(r"^### (\S+) · (\S+) · ([^·]+?) · (.+)$", re.M)
    hits = list(head.finditer(text))
    if not hits:
        return []
    entries = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        entries.append({"id": m.group(1), "date": m.group(2), "topic": m.group(3).strip(), "title": m.group(4).strip(),
                        "body": text[m.end():end].strip()})
    topics: dict[str, list[str]] = {}
    for e in entries:
        topics.setdefault(e["topic"], []).append(e["id"])
    L = ["## Appendix A. Findings and decisions log", "",
         f"Every entry of the build log ({len(entries)} entries), in the order they were written. Later entries correct or "
         "supersede earlier ones; an entry is never rewritten. Run ids mentioned in early entries may refer to development "
         "runs that are no longer kept.", "",
         "### A.1 Entries by topic", ""]
    L += [f"- **{t}** ({len(ids)}): {', '.join(ids)}" for t, ids in sorted(topics.items(), key=lambda kv: kv[0].lower())]
    L += ["", "### A.2 Index", "", "| Entry | Date | Topic | Title |", "|---|---|---|---|"]
    L += [f"| {e['id']} | {e['date']} | {e['topic']} | {e['title']} |" for e in entries]
    L += ["", "### A.3 Entries in full", ""]
    for e in entries:
        L += [f"#### {e['id']} · {e['date']} · {e['topic']} · {e['title']}", "", _demote(e["body"], 1), ""]
    return L


def run_pack(cfg: Config, run: Run, runs_file: str, findings: str | None = None) -> dict:
    st = run.stage("report")
    t0 = time.time()
    root = cfg.root
    rf = Path(runs_file) if Path(runs_file).is_absolute() else root / runs_file
    runs = yaml.safe_load(rf.read_text(encoding="utf-8"))
    run.record_input(rf, "final runs per stage")
    out = cfg.path("outputs")
    man = {k: _manifest(out, v) for k, v in runs.items() if isinstance(v, str)}
    figdir = st.dir / "figures"
    fig_rows, tab_rows, sections = [], [], []
    n = 0
    for key, title, rk, summaries, prefix, purpose in SECTIONS:
        if rk not in man:
            continue
        m, rid = man[rk], runs[rk]
        arts = m["artifacts"]
        text = []
        for s in summaries:
            hit = [a for a in arts if a["path"].endswith("/" + s)]
            if hit:
                p = root / hit[0]["path"]
                run.record_input(p, f"generated summary ({rid})")
                text.append(p.read_text(encoding="utf-8"))
        figs = [a for a in arts if a.get("kind") == "figure" and (prefix is None or Path(a["path"]).name.startswith(prefix))]
        refs = []
        for a in figs:
            n += 1
            src = root / a["path"]
            name = f"F{n:02d}_{key}__{src.stem}"
            shutil.copyfile(src, figdir / f"{name}.png")
            if a.get("data"):
                shutil.copyfile(root / a["data"], figdir / f"{name}.csv")
                run._register(figdir / f"{name}.csv", "report", "figure_data", f"Data for {name}.png", "metabric.report.pack")
            run._register(figdir / f"{name}.png", "report", "figure", f"Figure {n}: {a.get('caption', '')}", "metabric.report.pack",
                          data_path=(figdir / f"{name}.csv") if a.get("data") else None)
            row = {"figure": n, "section": key, "file": f"figures/{name}.png", "data": f"figures/{name}.csv" if a.get("data") else "",
                   "caption": a.get("caption", ""), "source": a["path"]}
            fig_rows.append(row)
            refs.append(row)
        if key == "p2_holdout":
            for nm, cap, fn in _p2_holdout_figures(root, arts):
                n += 1
                name = f"F{n:02d}_{key}__{nm}"
                fig, data = fn()
                st.figure(name, fig, data, f"Figure {n}: {cap}")
                row = {"figure": n, "section": key, "file": f"figures/{name}.png", "data": f"figures/{name}.csv", "caption": cap,
                       "source": f"drawn from the saved holdout predictions of {rid}"}
                fig_rows.append(row)
                refs.append(row)
        tabs = [a for a in arts if a.get("kind") == "table" and (prefix is None or Path(a["path"]).name.startswith(prefix))]
        for a in tabs:
            tab_rows.append({"section": key, "table": Path(a["path"]).name, "path": a["path"], "caption": a.get("caption", "")})
        sections.append({"key": key, "title": title, "run": rid, "command": m["command"], "config_hash": m["config_hash"],
                         "completed": m.get("completed"), "purpose": purpose, "text": text, "figures": refs,
                         "tables": [t for t in tab_rows if t["section"] == key]})
    fi, ti = pd.DataFrame(fig_rows), pd.DataFrame(tab_rows)
    st.table("figure_index", fi, "Every figure in the report pack: number, section, file, caption, source run")
    st.table("table_index", ti, "Every result table of the final runs: section, file, caption")
    md = _material(cfg, runs, sections, _key_results(root, man, runs), fi, ti)
    n_find = 0
    if findings:
        fp = Path(findings) if Path(findings).is_absolute() else root / findings
        run.record_input(fp, "findings and decisions log (appendix)")
        app = findings_appendix(fp.read_text(encoding="utf-8"))
        n_find = sum(1 for ln in app if ln.startswith("#### "))
        md += "\n\n" + "\n".join(app)
    st.text("report_material.md", md, "Material for the written report: design, key results, every section's generated summary "
            "with its figures and tables, limitations, indexes (generated from saved results)")
    F = {"figures": int(len(fi)), "tables": int(len(ti)), "sections": len(sections), "findings": n_find,
         "seconds": round(time.time() - t0)}
    st.log.info("report pack: %d figures, %d tables, %d sections, %d findings entries in %d s", F["figures"], F["tables"],
                F["sections"], F["findings"], F["seconds"])
    return F


# ---------------------------------------------------------------- Phase 2 holdout figures from the saved predictions
def _p2_holdout_figures(root: Path, arts: list[dict]):
    hit = [a for a in arts if a["path"].endswith("/phase2_holdout_predictions.csv")]
    if not hit:
        return []
    p = pd.read_csv(root / hit[0]["path"])
    labels = ["low", "below average", "above average", "high"]

    def km_fig():
        curves = []
        fig, ax = P.fig(6.0, 4.2)
        for i, lab in enumerate(labels):
            g = p[p.risk_group == lab]
            km = kaplan_meier(g.time, g.event)
            km.insert(0, "group", lab)
            curves.append(km)
            ax.step(km.time, km.survival, where="post", color=P.CAT[i], label=f"{lab} (n={len(g)})")
            ax.fill_between(km.time, km.ci_low, km.ci_high, step="post", color=P.CAT[i], alpha=0.12, linewidth=0)
        for h in (60, 120):
            ax.axvline(h, color=P.NEUTRAL, linestyle=":", linewidth=1)
        ax.set_xlabel("Months since diagnosis")
        ax.set_ylabel("Survival probability")
        ax.set_title("Phase 2 holdout: survival by risk group (training cutpoints)")
        ax.legend(fontsize=7)
        P.grid(ax, "both")
        return fig, pd.concat(curves, ignore_index=True)

    def cal_fig():
        rows = []
        fig, ax = P.fig(5.2, 5)
        ax.plot([0, 1], [0, 1], color=P.NEUTRAL, linestyle="--", linewidth=1)
        for i, h in enumerate((60, 120)):
            pred = 1 - p[f"S_{h}"]
            d = pd.qcut(pred.rank(method="first"), 10, labels=False)
            for k in range(10):
                g = p[d == k]
                km = kaplan_meier(g.time, g.event)
                s = km[km.time <= h].iloc[-1]
                rows.append({"horizon_months": h, "decile": k, "n": len(g), "mean_predicted": float(pred[d == k].mean()),
                             "observed": float(1 - s.survival), "obs_ci_low": float(1 - s.ci_high), "obs_ci_high": float(1 - s.ci_low)})
            r = pd.DataFrame([x for x in rows if x["horizon_months"] == h])
            ax.errorbar(r.mean_predicted, r.observed, yerr=[r.observed - r.obs_ci_low, r.obs_ci_high - r.observed], fmt="o",
                        color=P.CAT[i], ecolor=P.CAT[i], capsize=2, label=f"{h // 12} years")
        ax.set_xlabel("Mean predicted P(death by horizon)")
        ax.set_ylabel("Observed (Kaplan-Meier) P(death by horizon)")
        ax.set_title("Phase 2 holdout: calibration by decile")
        ax.legend()
        P.grid(ax, "both")
        return fig, pd.DataFrame(rows)

    return [("risk_group_km", "Phase 2 holdout: Kaplan-Meier survival by risk group (cutpoints from the training rows), 95% CI",
             km_fig),
            ("calibration", "Phase 2 holdout: predicted vs observed (Kaplan-Meier) death probability by decile at 5 and 10 years, "
             "95% CI", cal_fig)]


# ---------------------------------------------------------------- key results from the result tables
def _table(root: Path, man: dict, key: str, name: str) -> pd.DataFrame | None:
    if key not in man:
        return None
    hit = [a for a in man[key]["artifacts"] if a["path"].endswith("/" + name)]
    return pd.read_csv(root / hit[0]["path"]) if hit else None


def _key_results(root: Path, man: dict, runs: dict) -> list[str]:
    L = []
    fm = _table(root, man, "ensemble", "final_model.csv")
    hm = _table(root, man, "holdout", "holdout_metrics.csv")
    if fm is not None:
        r = fm.iloc[0]
        ci = json.loads(r["select_and_ensemble_ci"])
        L.append(f"- **Phase 1, cross-validated AUROC** (select-and-ensemble procedure, 25 outer splits): "
                 f"{r['select_and_ensemble_auroc']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}); final model `{r['model']}`; regression floor "
                 f"{r['regression_floor']:.3f}.")
    if hm is not None:
        g = hm.set_index(["model", "metric"])

        def v(mod, k):
            x = g.loc[(mod, k)]
            return f"{x.value:.3f} ({x.ci_low:.3f}-{x.ci_high:.3f})"
        L.append(f"- **Phase 1, holdout** (381 patients, once): AUROC {v('final_model', 'auroc')}; Brier {v('final_model', 'brier')}; "
                 f"calibration slope {v('final_model', 'cal_slope')}; age-only AUROC {v('age_only', 'auroc')}.")
    ps = _table(root, man, "phase2_models", "summary.csv")
    fc = _table(root, man, "phase2_models", "final_choice.csv")
    if ps is not None and fc is not None:
        mem = fc.loc[fc.chosen, "model"].iloc[0]
        s = ps[(ps.model == mem) & (ps.metric == "c_index")].iloc[0]
        L.append(f"- **Phase 2, cross-validated Harrell's C** (final member `{mem}`): {s['mean']:.3f} ({s.ci_low:.3f}-{s.ci_high:.3f}).")
    p2 = _table(root, man, "phase2_holdout", "phase2_holdout_metrics.csv")
    if p2 is not None:
        g = p2.set_index(["model", "metric"])
        c, a5, a10 = g.loc[("final", "c_index")], g.loc[("final", "auc_60")], g.loc[("final", "auc_120")]
        L.append(f"- **Phase 2, holdout** (once): Harrell's C {c.value:.3f} ({c.ci_low:.3f}-{c.ci_high:.3f}); AUC at 5 years "
                 f"{a5.value:.3f}, at 10 years {a10.value:.3f}.")
    return L


# ---------------------------------------------------------------- the document
def _material(cfg: Config, runs: dict, sections: list[dict], key: list[str], fi: pd.DataFrame, ti: pd.DataFrame) -> str:
    b = cfg.blocks
    cv, hc = cfg["cv"], cfg["holdout"]
    L = ["# Report material: METABRIC Phase 1 (death) and Phase 2 (overall survival)", "",
         "Everything needed to write the final report, generated from the saved results of the final runs. Figures are "
         "numbered globally (Figure k) and copied to `../figures/` next to this file, each with the CSV of the plotted "
         "numbers. Section text below each heading is the generated summary of that stage, reproduced unchanged.", "",
         "## 1. Provenance", "",
         "| Section | Run | Command | Config hash | Completed |", "|---|---|---|---|---|"]
    for s in sections:
        cmd = s["command"] if len(s["command"]) < 90 else s["command"][:87] + "..."
        L.append(f"| {s['title']} | `{s['run']}` | `{cmd}` | `{s['config_hash']}` | {s['completed'] or ''} |")
    extra = {k: v for k, v in runs.items() if k not in {s2[2] for s2 in SECTIONS}}
    if extra:
        L += ["", "Other runs used by the final runs: " + ", ".join(f"{k} `{v}`" if isinstance(v, str) else f"{k} " +
                                                                    ", ".join(f"`{x}`" for x in v) for k, v in extra.items()) + "."]
    L += ["", "## 2. Study design (from the configuration)", "",
          f"- **Data:** METABRIC, `{Path(cfg.raw['paths']['data']).name}`: {len(b['clinical'])} clinical inputs, {len(b['rna'])} mRNA "
          f"z-score columns, {len(b['dna'])} mutation columns. Outcome columns are never features.",
          f"- **Target (Phase 1):** death from any cause = 1 - `{cfg['target']['source_column']}` (positive class = death).",
          f"- **Target (Phase 2):** time to death, `{cfg['phase2']['time_column']}`; horizons {cfg['phase2']['horizons']} months.",
          f"- **Holdout:** {hc['fraction']:.0%} of patients, frozen once, stratified on outcome, cohort and chemotherapy; scored once "
          "per phase, at the end.",
          f"- **Nested cross-validation:** {cv['outer_folds']} outer folds x {cv['outer_repeats']} repeats on the training rows, "
          f"{cv['inner_folds']} inner folds; identical frozen splits for every model; seed {cfg.seed}.",
          f"- **Selection rule:** best mean inner score; the simplest candidate within {cfg['harness']['select_within_se']} SE wins. "
          "Paired comparisons use the corrected resampled t-test.",
          f"- **Phase 1 members:** {', '.join(cfg['farm']['members'])}, plus cause-specific members trained on the cause of death "
          "(training label only).",
          f"- **Data arms:** {', '.join(cfg['farm']['arms'])}.",
          f"- **Phase 2 members:** {', '.join(cfg['phase2']['members'])}; floors {', '.join(cfg['phase2']['floors'])}.",
          f"- **Baselines (Phase 1 floors):** {', '.join(cfg['floors'])}.", "",
          "## 3. Key results", ""] + key
    L += ["", "## 4. Results by section", ""]
    for i, s in enumerate(sections, 1):
        L += [f"### 4.{i} {s['title']}", "", f"*Purpose:* {s['purpose']}  ", f"*Run:* `{s['run']}`", ""]
        if s["figures"]:
            L += ["**Figures**", ""]
            for f in s["figures"]:
                L.append(f"- **Figure {f['figure']}.** {f['caption']}  \n  `{f['file']}`" + (f" (data `{f['data']}`)" if f["data"] else ""))
            L.append("")
        for t in s["text"]:
            L += ["**Generated summary**", "", _demote(t, 3), ""]
        if s["tables"]:
            L += ["<details><summary>Result tables of this section</summary>", ""]
            L += [f"- `{t['path']}`: {t['caption']}" for t in s["tables"]]
            L += ["", "</details>", ""]
    L += ["## 5. Limitations (as stated in each generated report)", ""]
    for s in sections:
        for t in s["text"]:
            m = re.search(r"#+ \d*\.? ?Limitations\s*\n(.*?)(\n#|\Z)", t, flags=re.S)
            if m:
                L += [f"**{s['title']}**", ""] + [ln.strip() for ln in m.group(1).splitlines() if ln.strip().startswith("- ")] + [""]
    L += ["", "## 6. Figure index", "", "| Figure | Section | File | Caption |", "|---|---|---|---|"]
    L += [f"| {r.figure} | {r.section} | `{r.file}` | {r.caption} |" for r in fi.itertuples()]
    L += ["", f"## 7. Table index", "", f"{len(ti)} result tables; the full list with captions is in `table_index.csv` next to this "
          "file, and each section above lists its own."]
    L += ["", "## 8. Reproduction", "", "Commands, environment and run order: `RUNME.md`. Frozen files: holdout "
          f"`{Path(cfg.raw['paths']['holdout']).name}` (sha256 `{str(hc.get('sha256'))[:12]}`), CV splits "
          f"`{Path(cfg.raw['paths']['cv_splits']).name}` (sha256 `{str(cv.get('splits_sha256'))[:12]}`)."]
    return "\n".join(L)
