"""EDA sections 1-9 (PLAN.md section 17). Each writes tables/figures through ctx and records
key numbers in ctx.facts (used by the reference check and the generated summary)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from ..core import stats as st
from . import plots as P

OTHER = "Died of Other Causes"


def _band(x: pd.Series, edges, labels=None) -> pd.Series:
    return pd.cut(x, bins=edges, right=False, labels=labels)


def _age_labels(edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(f"<{b}" if a == 0 else (f"{a}+" if b >= 150 else f"{a}-{b - 1}"))
    return out


def bands(ctx) -> dict[str, pd.Series]:
    f, e = ctx.frame, ctx.cfg["eda"]
    return {
        "age_band": _band(f["age_at_diagnosis"], e["age_bands"], _age_labels(e["age_bands"])),
        "node_band": _band(f["lymph_nodes_examined_positive"], e["node_bands"], ["0", "1-3", "4-9", "10+"]),
        "npi_band": _band(f["nottingham_prognostic_index"], e["npi_bands"], ["good (<3.4)", "moderate (3.4-5.4)", "poor (>=5.4)"]),
    }


# 1 ---------------------------------------------------------------------------------------------
def audit(ctx, raw: pd.DataFrame) -> None:
    f, b = ctx.frame, ctx.cfg.blocks
    ctx.facts.update(n_patients=len(f), n_columns=f.shape[1] + 1)
    ctx.table("audit_blocks", pd.DataFrame(
        [{"block": k, "n_columns": len(v)} for k, v in b.items()]), "Column blocks from the explicit config lists")
    rows = []
    for blk in ("clinical", "outcome", "rna"):
        for c in b[blk]:
            s = f[c]
            rows.append({"column": c, "block": blk, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
                         "n_unique": int(s.nunique())})
    for c in b["dna"]:
        rows.append({"column": c, "block": "dna", "dtype": "str (binarised)", "n_missing": int(f[c].isna().sum()),
                     "n_unique": int(f[c].nunique())})
    ctx.table("audit_columns", pd.DataFrame(rows), "Per-column dtype, missing count and distinct values")
    types = ctx.cfg["clinical_types"]
    dom = []
    for c in types["categorical"] + types["binary"] + types["ordinal"]:
        vc = f[c].astype(object).where(f[c].notna(), "<missing>").value_counts()
        dom += [{"column": c, "value": str(k), "n": int(v)} for k, v in vc.items()]
    ctx.table("audit_value_domains", pd.DataFrame(dom), "Value domain of every categorical, binary and ordinal clinical column")
    typo = int((raw["er_status_measured_by_ihc"] == "Positve").sum())
    ctx.facts["er_ihc_typo_rows"] = typo
    enc = []
    for c in b["dna"]:
        s = f[c].astype(str)
        enc.append({"column": c, "n_zero": int((s == "0").sum()), "n_variant": int((s != "0").sum()),
                    "n_distinct_strings": int(s.nunique()), "n_missing": int(f[c].isna().sum())})
    ctx.table("audit_mutation_encoding", pd.DataFrame(enc), "Mutation columns: '0' versus variant strings")
    feats = pd.concat([f[b["clinical"]].astype(object), ctx.muts], axis=1)
    near = []
    for c in feats.columns:
        top = feats[c].value_counts(normalize=True, dropna=False)
        if top.iloc[0] >= 0.99:
            near.append({"column": c, "top_value": str(top.index[0]), "top_fraction": float(top.iloc[0])})
    rna_sd = f[b["rna"]].std()
    near += [{"column": c, "top_value": "sd", "top_fraction": float(v)} for c, v in rna_sd[rna_sd < 1e-8].items()]
    ctx.table("audit_near_constant", pd.DataFrame(near, columns=["column", "top_value", "top_fraction"]),
              "Constant or near-constant columns (one value in at least 99% of rows)")
    ctx.facts["n_near_constant"] = len(near)
    ctx.facts["n_near_constant_mut"] = sum(r["column"].endswith("_mut") for r in near)
    ctx.table("audit_cohort_coding", f["cohort"].value_counts().sort_index().rename_axis("cohort").reset_index(name="n"),
              "Cohort coding (stored as float 1.0-5.0 in the file, treated as categorical)")
    ctx.log.info("audit: %d patients, typo rows %d, near-constant columns %d", len(f), typo, len(near))


# 2 ---------------------------------------------------------------------------------------------
def outcome(ctx) -> None:
    f, y = ctx.frame, ctx.y
    n = len(y)
    ctx.facts.update(n_died=int(y.sum()), n_alive=int((y == 0).sum()), pct_died=round(100 * y.mean(), 2))
    ctx.table("outcome_classes", pd.DataFrame({"class": ["died (positive)", "alive"], "n": [int(y.sum()), int((y == 0).sum())],
                                               "pct": [100 * y.mean(), 100 * (1 - y.mean())]}), "Outcome classes")
    cod = f["death_from_cancer"].astype(object).where(f["death_from_cancer"].notna(), "<missing>").value_counts()
    ctx.table("outcome_cause_of_death", cod.rename_axis("death_from_cancer").reset_index(name="n"), "Cause of death")
    ctx.facts.update(cod_disease=int(cod.get("Died of Disease", 0)), cod_other=int(cod.get(OTHER, 0)),
                     cod_unknown=int(cod.get("<missing>", 0)))
    unk = f[f["death_from_cancer"].isna()]
    ctx.facts["cod_unknown_overall_survival"] = unk["overall_survival"].tolist()
    t = f["overall_survival_months"]
    alive = y == 0
    ctx.facts.update(survivors_fu_lt10y=int((alive & (t < 120)).sum()), survivors_fu_lt5y=int((alive & (t < 60)).sum()),
                     pct_survivors_fu_lt10y=round(100 * (alive & (t < 120)).sum() / alive.sum(), 2),
                     n_time_zero=int((t == 0).sum()))
    g = pd.DataFrame({"cohort": f["cohort"].astype(str), "class": np.where(y == 1, "died", "alive"), "months": t})
    summ = (g.groupby(["cohort", "class"])["months"].describe()[["count", "min", "25%", "50%", "75%", "max"]].reset_index())
    summ = pd.concat([g.groupby("class")["months"].describe()[["count", "min", "25%", "50%", "75%", "max"]]
                      .reset_index().assign(cohort="all"), summ])
    ctx.table("outcome_followup", summ, "Follow-up (months) by class and cohort")
    fig, ax = P.fig(7, 3.8)
    cohorts = sorted(g.cohort.unique())
    pos, data, cols, plot_rows = [], [], [], []
    for i, c in enumerate(cohorts):
        for j, (cls, col) in enumerate([("alive", P.ALIVE), ("died", P.DIED)]):
            v = g[(g.cohort == c) & (g["class"] == cls)].months
            pos.append(i * 3 + j)
            data.append(v.to_numpy())
            cols.append(col)
            q = v.quantile([0, .25, .5, .75, 1]).to_list()
            plot_rows.append({"cohort": c, "class": cls, "n": len(v), "min": q[0], "q1": q[1], "median": q[2], "q3": q[3], "max": q[4]})
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                    medianprops={"color": P.INK, "linewidth": 1.5}, whiskerprops={"color": P.INK2},
                    capprops={"color": P.INK2})
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(cohorts))])
    ax.set_xticklabels([f"cohort {c}" for c in cohorts])
    ax.axhline(120, color=P.INK2, linestyle="--", linewidth=1)
    ax.text(len(cohorts) * 3 - 1.2, 123, "10 years", color=P.INK2, fontsize=7)
    ax.set_ylabel("Follow-up (months)")
    ax.set_title("Follow-up by cohort: alive (blue) and died (orange)")
    P.grid(ax)
    ctx.figure("outcome_followup_by_cohort", fig, pd.DataFrame(plot_rows),
               "Follow-up time by cohort and class (box = quartiles, whiskers 1.5 IQR, outliers hidden)")
    rt = st.rate_table(f["cohort"].astype(str), y)
    ctx.table("outcome_rate_by_cohort", rt, "Death rate by cohort with Wilson 95% intervals")
    fig, ax = P.fig(5.5, 3)
    P.dot_ci(ax, [f"cohort {c}" for c in rt.level], rt.death_rate, rt.ci_low, rt.ci_high, ref=y.mean(), ref_label="overall")
    ax.set_xlabel("Death rate")
    ax.set_title("Death rate by cohort")
    ctx.figure("outcome_rate_by_cohort", fig, rt, "Death rate by cohort, Wilson 95% intervals; dashed line = overall rate")
    km = st.kaplan_meier(t, y)
    ctx.table("outcome_km_overall", km, "Kaplan-Meier overall survival (descriptive), Greenwood log-log 95% CI")
    fig, ax = P.fig(6, 3.6)
    ax.step(km.time, km.survival, where="post", color=P.CAT[0])
    ax.fill_between(km.time, km.ci_low, km.ci_high, step="post", color=P.CAT[0], alpha=0.18, linewidth=0)
    ax.set_xlabel("Months since diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Overall survival (Kaplan-Meier)")
    P.grid(ax)
    ctx.figure("outcome_km_overall", fig, km[["time", "survival", "ci_low", "ci_high", "n_at_risk", "events"]],
               "Kaplan-Meier overall survival with 95% band (descriptive only; time-to-event is Phase 2)")


# 3 ---------------------------------------------------------------------------------------------
def missingness(ctx) -> None:
    f, y, clin = ctx.frame, ctx.y, ctx.cfg.blocks["clinical"]
    miss = f[clin].isna()
    per = pd.DataFrame({"column": clin, "n_missing": miss.sum().to_numpy(), "pct_missing": 100 * miss.mean().to_numpy()})
    per = per.sort_values("n_missing", ascending=False)
    ctx.table("missing_per_column", per, "Missing values per clinical column (RNA and DNA blocks have none)")
    ctx.facts["rna_dna_missing_cells"] = int(f[ctx.cfg.blocks["rna"] + ctx.cfg.blocks["dna"]].isna().sum().sum())
    anym = miss.any(axis=1)
    ctx.facts.update(any_clinical_missing=int(anym.sum()), pct_any_clinical_missing=round(100 * anym.mean(), 2))
    cols = per.loc[per.n_missing > 0, "column"].tolist()
    pat = miss[cols].apply(lambda r: " + ".join(c for c in cols if r[c]) or "(complete)", axis=1)
    pt = pat.value_counts().head(12).rename_axis("pattern").reset_index(name="n")
    pt["pct"] = 100 * pt.n / len(f)
    ctx.table("missing_patterns", pt, "Most common missingness patterns")
    byc = miss[cols].groupby(f["cohort"].astype(str)).mean().T * 100
    ctx.table("missing_by_cohort", byc.rename_axis("column").reset_index(), "Percent missing by cohort")
    fig, ax = P.fig(6.5, 0.35 * len(cols) + 1.2)
    im = ax.imshow(byc.to_numpy(), cmap=P.SEQ, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(byc.shape[1]))
    ax.set_xticklabels([f"cohort {c}" for c in byc.columns])
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(byc.shape[0]):
        for j in range(byc.shape[1]):
            v = byc.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else P.INK)
    fig.colorbar(im, ax=ax, label="% missing", shrink=0.8)
    ax.set_title("Missingness by cohort (%)")
    ax.spines[:].set_visible(False)
    ctx.figure("missing_by_cohort", fig, byc.rename_axis("column").reset_index(), "Percent missing per clinical column and cohort")
    st_ = f["tumor_stage"].isna()
    bycoh = st_.groupby(f["cohort"].astype(str)).mean() * 100
    ctx.facts.update(stage_missing=int(st_.sum()), pct_stage_missing=round(100 * st_.mean(), 2),
                     pct_stage_missing_cohort4=round(float(bycoh.get("4", np.nan)), 2),
                     pct_stage_missing_cohort5=round(float(bycoh.get("5", np.nan)), 2))
    rows = []
    for c in cols:
        for state, m in (("missing", miss[c]), ("present", ~miss[c])):
            k, n = int(y[m].sum()), int(m.sum())
            lo, hi = st.wilson(k, n)
            rows.append({"column": c, "state": state, "n": n, "deaths": k, "death_rate": k / n if n else np.nan,
                         "ci_low": lo, "ci_high": hi})
    dr = pd.DataFrame(rows)
    ctx.table("missing_death_rate", dr, "Death rate with and without each missing value (Wilson 95% CI)")
    g = dr[(dr.column == "neoplasm_histologic_grade") & (dr.state == "missing")].iloc[0]
    mc = dr[(dr.column == "mutation_count") & (dr.state == "missing")].iloc[0]
    ctx.facts.update(death_rate_missing_grade=round(100 * g.death_rate, 2), n_missing_grade=int(g.n),
                     death_rate_missing_mutation_count=round(100 * mc.death_rate, 2), n_missing_mutation_count=int(mc.n))
    m = dr[dr.state == "missing"].reset_index(drop=True)
    fig, ax = P.fig(6, 0.32 * len(m) + 1.2)
    P.dot_ci(ax, [f"{c} (n={n})" for c, n in zip(m.column, m.n)], m.death_rate, m.ci_low, m.ci_high,
             color=P.CAT[1], ref=y.mean(), ref_label="overall")
    ax.set_xlabel("Death rate among patients with the value missing")
    ax.set_title("Death rate when a value is missing")
    ctx.figure("missing_death_rate", fig, m, "Death rate among patients missing each column, Wilson 95% CI; dashed = overall rate")
    assoc = []
    for c in cols:
        ind = miss[c].astype(int)
        tab = pd.crosstab(ind, f["cohort"].astype(str))
        p = stats.chi2_contingency(tab)[1] if tab.shape[0] > 1 else np.nan
        assoc.append({"column": c, "cramers_v_with_cohort": st.cramers_v(ind, f["cohort"].astype(str)),
                      "chi2_p_cohort": p, "mean_age_missing": f.loc[miss[c], "age_at_diagnosis"].mean(),
                      "mean_age_present": f.loc[~miss[c], "age_at_diagnosis"].mean(),
                      "chemo_rate_missing": f.loc[miss[c], "chemotherapy"].mean(),
                      "chemo_rate_present": f.loc[~miss[c], "chemotherapy"].mean()})
    ctx.table("missing_associations", pd.DataFrame(assoc),
              "Association of each missing indicator with cohort (Cramer's V, chi-square), age and chemotherapy")


# 4 ---------------------------------------------------------------------------------------------
def clinical(ctx) -> None:
    f, y, types = ctx.frame, ctx.y, ctx.cfg["clinical_types"]
    num = types["numeric"] + types["ordinal"]
    rows = []
    for c in num:
        s = f[c].dropna()
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        out = s[(s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)]
        rows.append({"column": c, "n": len(s), "n_missing": int(f[c].isna().sum()), "mean": s.mean(), "sd": s.std(),
                     "min": s.min(), "q1": q1, "median": s.median(), "q3": q3, "max": s.max(),
                     "n_outliers_3iqr": len(out), "outlier_values": " ".join(f"{v:g}" for v in sorted(out.unique())[-10:])})
    ctx.table("clinical_numeric_summary", pd.DataFrame(rows), "Numeric and ordinal clinical columns; outliers beyond 3 IQR")
    rare_min = ctx.cfg["eda"]["rare_level_min_count"]
    lv = []
    for c in types["categorical"] + types["binary"]:
        vc = f[c].astype(object).where(f[c].notna(), "<missing>").value_counts()
        lv += [{"column": c, "level": str(k), "n": int(v), "pct": 100 * v / len(f), "rare": bool(v < rare_min)} for k, v in vc.items()]
    lv = pd.DataFrame(lv)
    ctx.table("clinical_level_counts", lv, f"Level counts of categorical columns; rare = fewer than {rare_min} patients")
    ctx.facts["n_rare_levels"] = int(lv.rare.sum())
    hist = []
    fig, axes = P.fig(10, 5.2, 2, 4)
    for ax, c in zip(axes.flat, num):
        s = f[c].dropna()
        bins = np.arange(s.min() - 0.5, s.max() + 1.5) if c in types["ordinal"] else 20
        cnt, edges = np.histogram(s, bins=bins)
        ax.bar(edges[:-1], cnt, width=np.diff(edges) * 0.9, align="edge", color=P.CAT[0])
        ax.set_title(c, fontsize=8)
        P.grid(ax)
        hist += [{"variable": c, "bin_left": a, "bin_right": b, "count": int(k)} for a, b, k in zip(edges[:-1], edges[1:], cnt)]
    for ax in list(axes.flat)[len(num):]:
        ax.axis("off")
    fig.suptitle("Distributions of numeric and ordinal clinical columns", x=0.01, ha="left", fontweight="bold", color=P.INK)
    fig.tight_layout()
    ctx.figure("clinical_distributions", fig, pd.DataFrame(hist), "Histograms of numeric and ordinal clinical columns (missing excluded)")
    bd = bands(ctx)
    groups = {"age_band": bd["age_band"], "node_band": bd["node_band"], "tumor_stage": f["tumor_stage"],
              "neoplasm_histologic_grade": f["neoplasm_histologic_grade"], "npi_band": bd["npi_band"],
              "pam50_+_claudin-low_subtype": f["pam50_+_claudin-low_subtype"], "er_status": f["er_status"],
              "pr_status": f["pr_status"], "her2_status": f["her2_status"], "cohort": f["cohort"].astype(str)}
    rr = []
    for name, s in groups.items():
        s2 = s.astype(object)
        t = st.rate_table(s2.map(lambda v: f"{v:g}" if isinstance(v, float) and not pd.isna(v) else v), y)
        if isinstance(s.dtype, pd.CategoricalDtype):
            order = [str(c) for c in s.cat.categories] + ["missing"]
            t["__o"] = t.level.map({k: i for i, k in enumerate(order)})
            t = t.sort_values("__o").drop(columns="__o")
        rr.append(t.assign(variable=name))
    rr = pd.concat(rr)[["variable", "level", "n", "deaths", "death_rate", "ci_low", "ci_high"]]
    ctx.table("clinical_death_rate_by_group", rr, "Death rate by clinical group with Wilson 95% CI")
    fig, axes = P.fig(11, 9, 4, 3)
    for ax, (name, t) in zip(axes.flat, rr.groupby("variable", sort=False)):
        P.dot_ci(ax, [f"{lv} ({n})" for lv, n in zip(t.level, t.n)], t.death_rate, t.ci_low, t.ci_high, ref=y.mean())
        ax.set_title(name, fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    fig.suptitle("Death rate by group (n in brackets; dashed = overall)", x=0.01, ha="left", fontweight="bold", color=P.INK)
    fig.tight_layout()
    ctx.figure("clinical_death_rate_by_group", fig, rr, "Death rate by clinical group, Wilson 95% CI; dashed = overall rate")
    excl = ctx.frame["death_from_cancer"].notna() & (ctx.frame["death_from_cancer"] != OTHER)
    au = []
    for c in ctx.cfg.blocks["clinical"]:
        kind = "numeric" if c in num else "categorical (level death rate)"
        def score(mask):
            if c in num:
                return f.loc[mask, c].astype(float)
            s = f.loc[mask, c].astype(object).where(f.loc[mask, c].notna(), "<missing>")
            return s.map(y[mask].groupby(s).mean())
        all_mask = pd.Series(True, index=f.index)
        a_all, a_ex = st.auroc(score(all_mask), y), st.auroc(score(excl), y[excl])
        au.append({"variable": c, "kind": kind, "auroc_all_deaths": a_all, "auroc_other_cause_set_aside": a_ex,
                   "n_all": int(f[c].notna().sum()) if c in num else len(f),
                   "n_other_set_aside": int(f.loc[excl, c].notna().sum()) if c in num else int(excl.sum())})
    au = pd.DataFrame(au)
    au["oriented_all"] = np.maximum(au.auroc_all_deaths, 1 - au.auroc_all_deaths)
    au = au.sort_values("oriented_all", ascending=False)
    ctx.table("clinical_univariate_auroc", au, "Single-variable AUROC for death; categorical columns scored by their "
              "in-sample level death rate (optimistic, descriptive); other-cause deaths set aside in the second column")
    a = au.set_index("variable")
    ctx.facts.update(auroc_age_all=round(a.loc["age_at_diagnosis", "auroc_all_deaths"], 4),
                     auroc_age_excl_other=round(a.loc["age_at_diagnosis", "auroc_other_cause_set_aside"], 4),
                     n_excl_other=int(excl.sum()))
    for c, k in [("tumor_size", "size"), ("tumor_stage", "stage"), ("nottingham_prognostic_index", "npi"),
                 ("lymph_nodes_examined_positive", "nodes")]:
        ctx.facts[f"auroc_{k}_all"] = round(a.loc[c, "auroc_all_deaths"], 4)
        ctx.facts[f"auroc_{k}_excl_other"] = round(a.loc[c, "auroc_other_cause_set_aside"], 4)
    top = au.head(15).iloc[::-1]
    fig, ax = P.fig(6.5, 4.6)
    ax.barh(top.variable, top.oriented_all, color=P.CAT[0], height=0.6, label="all deaths")
    ax.plot(np.maximum(top.auroc_other_cause_set_aside, 1 - top.auroc_other_cause_set_aside), top.variable, "o",
            color=P.CAT[1], markersize=6, label="other-cause deaths set aside")
    ax.axvline(0.5, color=P.INK2, linewidth=1)
    ax.set_xlim(0.45, 0.8)
    ax.set_xlabel("AUROC (oriented, max(a, 1-a))")
    ax.set_title("Single-variable AUROC for death (top 15)")
    ax.legend(loc="lower right")
    P.grid(ax, "x")
    ctx.figure("clinical_univariate_auroc", fig, top, "Single-variable AUROC for death, top 15 clinical columns")


# 5 ---------------------------------------------------------------------------------------------
def treatment(ctx) -> None:
    f, y = ctx.frame, ctx.y
    bd = bands(ctx)
    tr = {"chemotherapy": f["chemotherapy"], "hormone_therapy": f["hormone_therapy"], "radio_therapy": f["radio_therapy"],
          "mastectomy": (f["type_of_breast_surgery"] == "MASTECTOMY").astype(float).where(f["type_of_breast_surgery"].notna())}
    by = {"cohort": f["cohort"].astype(str), "age_band": bd["age_band"].astype(str),
          "tumor_stage": f["tumor_stage"].map(lambda v: "missing" if pd.isna(v) else f"{v:g}"),
          "pam50": f["pam50_+_claudin-low_subtype"]}
    rows = []
    for tn, tv in tr.items():
        for bn, bv in by.items():
            for lev, idx in tv.groupby(bv, observed=True).groups.items():
                v = tv.loc[idx].dropna()
                rows.append({"treatment": tn, "by": bn, "level": lev, "n": len(v), "n_treated": int(v.sum()), "rate": v.mean()})
    fq = pd.DataFrame(rows)
    ctx.table("treatment_frequency", fq, "Treatment frequency by cohort, age band, stage and PAM50 subtype (confounding picture)")
    age = fq[fq.by == "age_band"]
    order = _age_labels(ctx.cfg["eda"]["age_bands"])
    fig, ax = P.fig(6.5, 3.8)
    for i, (tn, t) in enumerate(age.groupby("treatment", sort=False)):
        t = t.set_index("level").reindex(order)
        ax.plot(order, t.rate, marker="o", color=P.CAT[i], label=tn, markersize=5)
    ax.set_ylabel("Share treated")
    ax.set_xlabel("Age at diagnosis")
    ax.set_ylim(0, 1)
    ax.set_title("Treatment follows age")
    ax.legend(loc="upper left", ncol=2)
    P.grid(ax)
    ctx.figure("treatment_by_age", fig, age, "Share of patients receiving each treatment by age band")
    med = f.groupby("chemotherapy")["age_at_diagnosis"].median()
    ctx.facts.update(median_age_chemo=round(float(med.get(1)), 2), median_age_no_chemo=round(float(med.get(0)), 2))
    strata = bd["age_band"].astype(str) + "|stage " + by["tumor_stage"]
    cmp_rows = []
    for tn, tv in tr.items():
        ok = tv.notna()
        d = pd.DataFrame({"t": tv[ok].astype(int), "y": y[ok], "s": strata[ok]})
        crude = d.groupby("t")["y"].mean()
        means = d.pivot_table(index="s", columns="t", values="y", aggfunc="mean")
        sizes = d.pivot_table(index="s", columns="t", values="y", aggfunc="size")
        both = means.dropna().index                      # strata containing treated and untreated patients
        w = sizes.loc[both].sum(axis=1)
        std_t = float((means.loc[both, 1] * w).sum() / w.sum())
        std_u = float((means.loc[both, 0] * w).sum() / w.sum())
        cmp_rows.append({"treatment": tn, "crude_rate_treated": crude[1], "crude_rate_untreated": crude[0],
                         "std_rate_treated": std_t, "std_rate_untreated": std_u, "n_strata_used": len(both),
                         "n_patients_used": int(w.sum()), "n_patients": int(ok.sum())})
    ctx.table("treatment_crude_vs_standardised", pd.DataFrame(cmp_rows),
              "Crude death rate by treatment versus rate directly standardised over age band x stage strata "
              "(strata with both groups only). Association, not treatment effect.")
    s = pd.DataFrame({"outcome": np.where(y == 1, "died", "alive"), "cohort": f["cohort"].astype(str), "chemotherapy": f["chemotherapy"]})
    sz = s.value_counts().rename("n").reset_index().sort_values("n")
    ctx.table("treatment_strata_sizes", sz, "Stratum sizes outcome x cohort x chemotherapy (the splitting rule)")
    folds = ctx.cfg["cv"]["outer_folds"]
    ctx.facts.update(n_strata=len(sz), min_stratum=int(sz.n.min()), n_strata_below_folds=int((sz.n < folds).sum()))
    combo = pd.DataFrame({"hormone": f["hormone_therapy"], "radio": f["radio_therapy"], "chemo": f["chemotherapy"],
                          "surgery": f["type_of_breast_surgery"].astype(object).fillna("<missing>"), "y": y})
    cb = combo.groupby(["chemo", "hormone", "radio", "surgery"])["y"].agg(n="size", death_rate="mean").reset_index()
    ctx.table("treatment_combinations", cb.sort_values("n", ascending=False), "Treatment combinations with counts and crude death rate")


# 6 ---------------------------------------------------------------------------------------------
def redundancy(ctx) -> None:
    f, y, types = ctx.frame, ctx.y, ctx.cfg["clinical_types"]
    num = types["numeric"] + types["ordinal"]
    sp = f[num].corr(method="spearman")
    ctx.table("redundancy_spearman", sp.rename_axis("variable").reset_index(), "Spearman correlation of numeric and ordinal clinical columns")
    fig, ax = P.fig(6, 5)
    im = ax.imshow(sp.to_numpy(), cmap=P.DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)))
    ax.set_xticklabels(num, rotation=45, ha="right")
    ax.set_yticks(range(len(num)))
    ax.set_yticklabels(num)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{sp.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(sp.iat[i, j]) > 0.6 else P.INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.spines[:].set_visible(False)
    ax.set_title("Spearman correlation, numeric clinical")
    ctx.figure("redundancy_spearman", fig, sp.rename_axis("variable").reset_index(), "Spearman correlation matrix of numeric and ordinal clinical columns")
    npi = "nottingham_prognostic_index"
    ctx.facts.update(spearman_npi_nodes=round(sp.loc[npi, "lymph_nodes_examined_positive"], 4),
                     spearman_npi_grade=round(sp.loc[npi, "neoplasm_histologic_grade"], 4),
                     spearman_npi_stage=round(sp.loc[npi, "tumor_stage"], 4),
                     spearman_npi_size=round(sp.loc[npi, "tumor_size"], 4))
    cat = types["categorical"] + types["binary"]
    cf = f[cat].astype(object)
    plain = pd.DataFrame(np.nan, index=cat, columns=cat)
    corr = plain.copy()
    for i, a in enumerate(cat):
        for b in cat[i:]:
            v1 = 1.0 if a == b else st.cramers_v(cf[a], cf[b])
            v2 = 1.0 if a == b else st.cramers_v(cf[a], cf[b], bias_corrected=True)
            plain.loc[a, b] = plain.loc[b, a] = v1
            corr.loc[a, b] = corr.loc[b, a] = v2
    ctx.table("redundancy_cramers_v_plain", plain.rename_axis("variable").reset_index(), "Cramer's V (plain) between categorical clinical columns")
    ctx.table("redundancy_cramers_v_corrected", corr.rename_axis("variable").reset_index(), "Cramer's V (bias-corrected, Bergsma) between categorical clinical columns")
    fig, ax = P.fig(8.5, 7.5)
    im = ax.imshow(corr.to_numpy(dtype=float), cmap=P.SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(cat)))
    ax.set_xticklabels(cat, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cat)))
    ax.set_yticklabels(cat, fontsize=7)
    for i in range(len(cat)):
        for j in range(len(cat)):
            v = corr.iat[i, j]
            if i != j and v >= 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else P.INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Cramer's V (bias-corrected)")
    ax.spines[:].set_visible(False)
    ax.set_title("Association between categorical clinical columns")
    ctx.figure("redundancy_cramers_v", fig, corr.rename_axis("variable").reset_index(),
               "Bias-corrected Cramer's V between categorical clinical columns; values >= 0.30 printed")
    for key, a, b in [("cramers_er_3gene", "er_status", "3-gene_classifier_subtype"),
                      ("cramers_er_pam50", "er_status", "pam50_+_claudin-low_subtype"),
                      ("cramers_her2_3gene", "her2_status", "3-gene_classifier_subtype")]:
        ctx.facts[key] = round(plain.loc[a, b], 4)
        ctx.facts[key + "_corrected"] = round(corr.loc[a, b], 4)
    ctx.facts["cramers_pam50_3gene"] = round(plain.loc["pam50_+_claudin-low_subtype", "3-gene_classifier_subtype"], 4)
    nest = []
    for a, b in [("cancer_type_detailed", "oncotree_code"), ("tumor_other_histologic_subtype", "cancer_type_detailed"),
                 ("cancer_type_detailed", "cancer_type"), ("tumor_other_histologic_subtype", "oncotree_code")]:
        d = f[[a, b]].dropna()
        a_to_b = d.groupby(a)[b].nunique().max() == 1
        b_to_a = d.groupby(b)[a].nunique().max() == 1
        nest.append({"a": a, "b": b, "levels_a": d[a].nunique(), "levels_b": d[b].nunique(),
                     "a_determines_b": bool(a_to_b), "b_determines_a": bool(b_to_a),
                     "identical_up_to_labels": bool(a_to_b and b_to_a),
                     "missing_same_rows": bool((f[a].isna() == f[b].isna()).all()),
                     "cramers_v": st.cramers_v(f[a], f[b])})
    ctx.table("redundancy_nesting", pd.DataFrame(nest), "Identical or nested categorical columns")
    d = f[["tumor_size", "lymph_nodes_examined_positive", "neoplasm_histologic_grade", npi]].dropna()
    node_stage = np.select([d.lymph_nodes_examined_positive == 0, d.lymph_nodes_examined_positive <= 3], [1, 2], 3)
    base = node_stage + d.neoplasm_histologic_grade
    formulas = {"standard": (0.2, "0.2 x size(cm) + node stage (0:1, 1-3:2, 4+:3) + grade  [CLAUDE.md 3]"),
                "file_fit": (0.02, "0.02 x size(cm) + node stage + grade  [what the file matches]")}
    recs, npi_rows = {}, []
    for key, (coef, text) in formulas.items():
        rec = coef * d.tumor_size / 10 + base
        diff = d[npi] - rec
        recs[key] = rec
        npi_rows.append({"formula_id": key, "formula": text, "n_complete": len(d),
                         "pct_abs_diff_lt_0.01": 100 * (diff.abs() < 0.01).mean(),
                         "pct_abs_diff_lt_0.1": 100 * (diff.abs() < 0.1).mean(), "median_abs_diff": diff.abs().median(),
                         "max_abs_diff": diff.abs().max(), "spearman": stats.spearmanr(d[npi], rec)[0]})
        ctx.facts[f"npi_{key}_pct_within_0.01"] = round(100 * (diff.abs() < 0.01).mean(), 2)
        ctx.facts[f"npi_{key}_pct_within_0.1"] = round(100 * (diff.abs() < 0.1).mean(), 2)
    ctx.table("redundancy_npi_reconstruction", pd.DataFrame(npi_rows),
              "NPI reconstruction: the file's NPI against the standard formula and against a 0.02 size coefficient")
    fig, ax = P.fig(5, 4.5)
    for i, (key, rec) in enumerate(recs.items()):
        ax.scatter(rec, d[npi], s=8, color=P.CAT[i], alpha=0.5, linewidths=0, label=formulas[key][1].split("  [")[0])
    allv = pd.concat([*recs.values(), d[npi]])
    lim = [allv.min() - 0.2, allv.max() + 0.2]
    ax.plot(lim, lim, color=P.INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("Reconstructed NPI")
    ax.set_ylabel("NPI in file")
    ax.set_title("NPI in the file uses a 0.02 size weight")
    ax.legend(loc="upper left", markerscale=2)
    P.grid(ax, "both")
    ctx.figure("redundancy_npi_reconstruction", fig,
               pd.DataFrame({"standard_0.2": recs["standard"], "file_fit_0.02": recs["file_fit"], "file": d[npi]}),
               "NPI in the file against the standard formula (0.2 x size cm) and a 0.02 size weight; dashed = identity")
    # the two NPI floors (F-028): file NPI (primary) and standard NPI (secondary), descriptive AUROCs
    from ..data import npi_standard
    ns = npi_standard(f)
    assert np.allclose(ns.loc[d.index], recs["standard"]), "npi_standard disagrees with the EDA reconstruction"
    excl = f["death_from_cancer"].notna() & (f["death_from_cancer"] != OTHER)
    comp = ns.notna()
    fl = []
    for name, score, rows in [("npi_file (as supplied, 0.02 size weight)", f[npi], pd.Series(True, index=f.index)),
                              ("npi_standard (0.2 size weight)", ns, comp),
                              ("npi_file on the npi_standard rows", f[npi], comp)]:
        fl.append({"index": name, "n": int((rows & score.notna()).sum()),
                   "auroc_all_deaths": st.auroc(score[rows], y[rows]),
                   "auroc_other_cause_set_aside": st.auroc(score[rows & excl], y[rows & excl])})
    fl = pd.DataFrame(fl)
    ctx.table("redundancy_npi_floors", fl, "Single-variable AUROC of the file's NPI and the standard NPI (the two NPI floors); "
              "descriptive, not a model result")
    ctx.facts.update(n_npi_standard=int(comp.sum()), auroc_npi_standard_all=round(fl.auroc_all_deaths[1], 4),
                     auroc_npi_standard_excl_other=round(fl.auroc_other_cause_set_aside[1], 4),
                     auroc_npi_file_same_rows=round(fl.auroc_all_deaths[2], 4))
    rna = ctx.frame
    er_both = f[["er_status_measured_by_ihc", "er_status"]].dropna()
    agree = (er_both.iloc[:, 0] == er_both.iloc[:, 1])
    nmut = ctx.muts.sum(axis=1)
    rho_mc = stats.spearmanr(f["mutation_count"], nmut, nan_policy="omit")[0]
    mol = [
        {"check": "ERBB2 expression vs her2_status (AUROC, Positive)", "value": st.auroc(rna["erbb2"], f["her2_status"] == "Positive"), "n": len(f)},
        {"check": "GATA3 expression vs er_status (AUROC, Positive)", "value": st.auroc(rna["gata3"], f["er_status"] == "Positive"), "n": len(f)},
        {"check": "ESR1 expression vs er_status", "value": np.nan, "n": 0, "note": "ESR1 is not among the 489 expression columns"},
        {"check": "ER IHC vs er_status agreement (%)", "value": 100 * agree.mean(), "n": len(er_both)},
        {"check": "PAM50 subtype column vs 3-gene subtype (Cramer's V)", "value": ctx.facts["cramers_pam50_3gene"], "n": int(f[["pam50_+_claudin-low_subtype", "3-gene_classifier_subtype"]].dropna().shape[0])},
        {"check": "mutation_count vs number of mutated genes among the 173 (Spearman)", "value": rho_mc, "n": int(f["mutation_count"].notna().sum())},
    ]
    ctx.table("redundancy_clinical_vs_molecular", pd.DataFrame(mol), "Clinical-versus-molecular overlap checks")
    ctx.facts.update(auroc_erbb2_her2=round(mol[0]["value"], 4), auroc_gata3_er=round(mol[1]["value"], 4),
                     er_ihc_agreement_pct=round(100 * agree.mean(), 2), er_ihc_n_both=len(er_both),
                     spearman_mutcount_nmut=round(rho_mc, 4))


# 7 ---------------------------------------------------------------------------------------------
def rna(ctx) -> dict:
    f, y, genes = ctx.frame, ctx.y, ctx.cfg.blocks["rna"]
    X = f[genes].to_numpy(dtype=float)
    mu, sd = X.mean(0), X.std(0, ddof=1)
    ctx.table("rna_zscore_check", pd.DataFrame([{"n_genes": len(genes), "max_abs_mean": np.abs(mu).max(), "min_sd": sd.min(),
                                                 "max_sd": sd.max(), "n_missing": int(np.isnan(X).sum())}]),
              "z-score check of the RNA block (rows in this mode)")
    yy = y.to_numpy()
    obs = st.auroc_columns(X, yy)
    null = st.auroc_permutation_null(X, yy, ctx.cfg["eda"]["permutations"], ctx.rng)
    p = st.permutation_pvalues(obs, null)
    q = st.bh_fdr(p)
    tab = pd.DataFrame({"gene": genes, "auroc": obs, "dist_from_0.5": np.abs(obs - 0.5), "perm_p": p, "bh_q": q})
    tab = tab.sort_values("dist_from_0.5", ascending=False)
    ctx.table("rna_gene_auroc", tab, f"Single-gene AUROC for death with a {null.shape[0]}-permutation null and BH FDR")
    alpha = ctx.cfg["eda"]["fdr_alpha"]
    lo95, hi95 = np.quantile(null, [0.025, 0.975])
    ctx.facts.update(gene_auroc_min=round(obs.min(), 4), gene_auroc_max=round(obs.max(), 4),
                     gene_auroc_median_dist=round(float(np.median(np.abs(obs - 0.5))), 4),
                     n_genes_outside_40_60=int(((obs < 0.4) | (obs > 0.6)).sum()),
                     n_genes_fdr=int((q < alpha).sum()), null_auroc_95=[round(lo95, 4), round(hi95, 4)],
                     top_genes=tab.gene.head(5).tolist())
    edges = np.linspace(0.35, 0.65, 31)
    c_obs, _ = np.histogram(obs, edges)
    fig, ax = P.fig(6.4, 3.6)
    ax.axvspan(lo95, hi95, color=P.NEUTRAL, alpha=0.35, linewidth=0, label="95% of permutation null")
    ax.bar(edges[:-1], c_obs, width=np.diff(edges) * 0.9, align="edge", color=P.CAT[0], label="observed genes")
    ax.set_xlabel("Single-gene AUROC for death")
    ax.set_ylabel("Genes")
    ax.set_title("Single-gene AUROCs against the permutation null")
    ax.legend(loc="upper right")
    P.grid(ax)
    ctx.figure("rna_gene_auroc_hist", fig, pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "n_genes": c_obs,
                                                         "null_q025": lo95, "null_q975": hi95}),
               "Histogram of 489 single-gene AUROCs; grey band = central 95% of the pooled permutation null")
    R = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(len(genes), 1)
    r = R[iu]
    t1, t2 = ctx.cfg["eda"]["corr_thresholds"]
    ctx.facts.update(n_pairs_total=len(r), n_pairs_gt_05=int((np.abs(r) > t1).sum()), n_pairs_gt_07=int((np.abs(r) > t2).sum()))
    pairs = pd.DataFrame({"gene_a": np.array(genes)[iu[0]], "gene_b": np.array(genes)[iu[1]], "r": r})
    pairs = pairs[np.abs(pairs.r) > t1].sort_values("r", key=np.abs, ascending=False)
    ctx.table("rna_correlated_pairs", pairs, f"Gene pairs with |r| above {t1} (Pearson)")
    D = 1 - np.abs(R)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    order = leaves_list(Z)
    Ro = R[np.ix_(order, order)]
    fig, ax = P.fig(7.5, 7)
    im = ax.imshow(Ro, cmap=P.DIV, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title("Gene-gene correlation, clustered (average linkage on 1-|r|)")
    ax.spines[:].set_visible(False)
    og = [genes[i] for i in order]
    ctx.figure("rna_correlation_clustered", fig, pd.DataFrame(np.round(Ro, 3), columns=og).assign(gene=og)[["gene", *og]],
               "Clustered 489 x 489 gene correlation matrix (rows/columns in plotted order)")
    cl = fcluster(Z, t=ctx.cfg["eda"]["module_distance_cut"], criterion="distance")
    sizes = pd.Series(cl).value_counts()
    keep = sizes[sizes >= ctx.cfg["eda"]["module_min_size"]].index
    mod = pd.DataFrame({"gene": genes, "cluster": cl})
    mod["module"] = mod.cluster.map({c: f"M{i + 1}" for i, c in enumerate(sizes.loc[keep].index)}).fillna("")
    ctx.table("rna_modules", mod[mod.module != ""].sort_values(["module", "gene"])[["module", "gene"]],
              "Descriptive gene modules (average linkage on 1-|r|, cut at the configured distance, min size); "
              "modelling modules are refitted inside folds")
    ctx.facts["n_modules"] = len(keep)
    ctx.facts["n_genes_in_modules"] = int((mod.module != "").sum())
    from sklearn.decomposition import PCA
    pca = PCA(random_state=ctx.cfg.seed).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    targets = ctx.cfg["eda"]["pca_variance_targets"]
    npcs = {t: int(np.searchsorted(cum, t) + 1) for t in targets}
    ctx.facts.update(n_pcs_50=npcs[0.5], n_pcs_80=npcs[0.8], n_pcs_90=npcs[0.9],
                     pc1_var=round(float(pca.explained_variance_ratio_[0]), 4))
    vc = pd.DataFrame({"pc": np.arange(1, len(cum) + 1), "var_ratio": pca.explained_variance_ratio_, "cumulative": cum})
    ctx.table("rna_pca_variance", vc, "PCA of the RNA block: variance per component")
    fig, ax = P.fig(6, 3.6)
    ax.plot(vc.pc, vc.cumulative, color=P.CAT[0])
    for t, k in npcs.items():
        ax.plot([k], [t], "o", color=P.CAT[0], markersize=6)
        ax.annotate(f"{int(t * 100)}%: {k} PCs", (k, t), xytext=(6, -12), textcoords="offset points", fontsize=7, color=P.INK2)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_ylim(0, 1.02)
    ax.set_title("RNA variance is spread over many components")
    P.grid(ax)
    ctx.figure("rna_pca_variance", fig, vc, "Cumulative RNA variance explained by principal components")
    S = pca.transform(X)[:, :max(2, ctx.cfg["eda"]["cluster_n_pcs"])]
    sc = pd.DataFrame({"pc1": S[:, 0], "pc2": S[:, 1], "death": yy, "cohort": f["cohort"].astype(str).to_numpy(),
                       "pam50": f["pam50_+_claudin-low_subtype"].to_numpy()}, index=f.index)
    fig, ax = P.fig(5.4, 4.6)
    for v, col, lab in [(0, P.ALIVE, "alive"), (1, P.DIED, "died")]:
        m = sc.death == v
        ax.scatter(sc.pc1[m], sc.pc2[m], s=7, color=col, alpha=0.55, linewidths=0, label=lab)
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("RNA PC1-PC2 by outcome")
    ax.legend(markerscale=2)
    ctx.figure("rna_pca_by_outcome", fig, sc.reset_index()[["patient_id", "pc1", "pc2", "death"]], "RNA PC1 against PC2 coloured by outcome")
    for key, title in [("cohort", "cohort"), ("pam50", "PAM50 subtype column")]:
        levels = sorted(sc[key].unique())
        nc = 4 if len(levels) > 5 else len(levels)
        nr = int(np.ceil(len(levels) / nc))
        fig, axes = P.fig(2.6 * nc, 2.5 * nr, nr, nc, sharex=True, sharey=True, squeeze=False)
        for ax, lv in zip(axes.flat, levels):
            m = sc[key] == lv
            ax.scatter(sc.pc1[~m], sc.pc2[~m], s=3, color=P.NEUTRAL, alpha=0.4, linewidths=0)
            ax.scatter(sc.pc1[m], sc.pc2[m], s=4, color=P.CAT[0], alpha=0.7, linewidths=0)
            ax.set_title(f"{lv} (n={int(m.sum())})", fontsize=8)
        for ax in list(axes.flat)[len(levels):]:
            ax.axis("off")
        fig.suptitle(f"RNA PC1-PC2, one panel per {title} (others in grey)", x=0.01, ha="left", fontweight="bold", color=P.INK)
        fig.tight_layout()
        ctx.figure(f"rna_pca_by_{key}", fig, sc.reset_index()[["patient_id", "pc1", "pc2", key]],
                   f"RNA PC1 against PC2, small multiples by {title}")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    k_rows = []
    Sk = S[:, :ctx.cfg["eda"]["cluster_n_pcs"]]
    for k in ctx.cfg["eda"]["cluster_k"]:
        lab = KMeans(n_clusters=k, n_init=10, random_state=ctx.cfg.seed).fit_predict(Sk)
        for ref in ["integrative_cluster", "pam50_+_claudin-low_subtype"]:
            k_rows.append({"k": k, "reference": ref, "ari": adjusted_rand_score(f[ref], lab),
                           "nmi": normalized_mutual_info_score(f[ref], lab),
                           "cramers_v": st.cramers_v(pd.Series(lab, index=f.index), f[ref])})
        ct = pd.crosstab(pd.Series(lab, index=f.index, name=f"kmeans_k{k}"), f["integrative_cluster"])
        ctx.table(f"rna_kmeans_k{k}_vs_intclust", ct.reset_index(), f"k-means (k={k}) clusters on {Sk.shape[1]} RNA PCs against integrative_cluster")
    kr = pd.DataFrame(k_rows)
    ctx.table("rna_clusters_vs_labels", kr, "Unsupervised RNA clusters (k-means on PCs) against integrative_cluster and the PAM50 subtype column")
    ctx.facts["kmeans_best"] = kr.sort_values("ari", ascending=False).iloc[0][["k", "reference", "ari"]].to_dict()
    return {"pca_scores": S}


# 8 ---------------------------------------------------------------------------------------------
def dna(ctx) -> None:
    f, y, M = ctx.frame, ctx.y, ctx.muts
    n = len(M)
    thr = ctx.cfg["eda"]["mutation_freq_threshold"]
    freq = M.mean()
    rows = []
    for c in M.columns:
        s = f[c].astype(str)
        toks = s[s != "0"].str.split().explode()
        top = toks.value_counts()
        rows.append({"column": c, "gene": c[:-4].upper(), "n_mutated": int(M[c].sum()), "pct_mutated": 100 * freq[c],
                     "n_distinct_variants": int(top.size), "top_variant": top.index[0] if top.size else "",
                     "top_variant_n": int(top.iloc[0]) if top.size else 0})
    fr = pd.DataFrame(rows).sort_values("n_mutated", ascending=False)
    ctx.table("dna_mutation_frequency", fr, "Mutation frequency per gene with variant diversity")
    common = freq[freq >= thr].index.tolist()
    ctx.facts.update(n_mut_lt1pct=int((freq < thr).sum()), n_mut_ge1pct=len(common),
                     top_mutated=fr.gene.head(5).tolist(), pct_tp53=round(100 * freq.get("tp53_mut", np.nan), 2),
                     pct_pik3ca=round(100 * freq.get("pik3ca_mut", np.nan), 2))
    var = []
    for c in fr.column.head(30):
        s = f[c].astype(str)
        vc = s[s != "0"].str.split().explode().value_counts().head(10)
        var += [{"gene": c[:-4].upper(), "variant": k, "n": int(v)} for k, v in vc.items()]
    ctx.table("dna_top_variants", pd.DataFrame(var), "Most frequent variants per gene (30 most mutated genes, top 10 each)")
    top30 = fr.head(30).iloc[::-1]
    fig, ax = P.fig(6, 6.5)
    ax.barh(top30.gene, top30.pct_mutated, color=P.CAT[0], height=0.65)
    ax.axvline(100 * thr, color=P.INK2, linestyle="--", linewidth=1)
    ax.set_xlabel("% of patients mutated")
    ax.set_title("30 most frequently mutated genes")
    ax.tick_params(axis="y", labelsize=7)
    P.grid(ax, "x")
    ctx.figure("dna_mutation_frequency", fig, top30[["gene", "n_mutated", "pct_mutated"]], "Thirty most frequently mutated genes")
    Mc = M[common].to_numpy(dtype=float)
    C = np.corrcoef(Mc, rowvar=False)
    iu = np.triu_indices(len(common), 1)
    co = pd.DataFrame({"gene_a": np.array(common)[iu[0]], "gene_b": np.array(common)[iu[1]], "phi": C[iu],
                       "n_both": (Mc.T @ Mc)[iu].astype(int)})
    # deterministic order: |phi| rounded (ties differ only at ~1e-15 across BLAS builds), then gene names
    co = (co.assign(_k=co.phi.abs().round(10)).sort_values(["_k", "gene_a", "gene_b"], ascending=[False, True, True])
          .drop(columns="_k"))
    ctx.table("dna_cooccurrence_pairs", co.head(50), "Strongest pairwise mutation associations by |phi| "
              "(positive = co-occurrence, negative = mutual exclusivity), genes mutated in at least 1%")
    pos, neg = co.sort_values("phi").iloc[-1], co.sort_values("phi").iloc[0]
    name = lambda r: f"{r.gene_a[:-4].upper()}-{r.gene_b[:-4].upper()}"  # noqa: E731
    ctx.facts["max_cooccurrence"] = {"pair": name(pos), "phi": round(float(pos.phi), 3), "n_both": int(pos.n_both)}
    ctx.facts["max_exclusivity"] = {"pair": name(neg), "phi": round(float(neg.phi), 3), "n_both": int(neg.n_both)}
    t20 = fr.column.head(20).tolist()
    C20 = np.corrcoef(M[t20].to_numpy(float), rowvar=False)
    np.fill_diagonal(C20, np.nan)
    fig, ax = P.fig(6.5, 5.6)
    vmax = float(np.nanmax(np.abs(C20)))
    im = ax.imshow(C20, cmap=P.DIV, vmin=-vmax, vmax=vmax)
    lab = [c[:-4].upper() for c in t20]
    ax.set_xticks(range(20))
    ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(20))
    ax.set_yticklabels(lab, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75, label="phi")
    ax.spines[:].set_visible(False)
    ax.set_title("Co-occurrence among the 20 most mutated genes")
    ctx.figure("dna_cooccurrence", fig, pd.DataFrame(C20, columns=lab).assign(gene=lab)[["gene", *lab]],
               "Phi correlation of mutation flags, 20 most frequently mutated genes (diagonal blank)")
    assoc = []
    for c in common:
        m = M[c].astype(bool)
        a, b = int(y[m].sum()), int(m.sum() - y[m].sum())
        c_, d = int(y[~m].sum()), int((~m).sum() - y[~m].sum())
        orr, p = stats.fisher_exact([[a, b], [c_, d]])
        assoc.append({"gene": c[:-4].upper(), "n_mutated": int(m.sum()), "death_rate_mutated": a / (a + b),
                      "death_rate_wildtype": c_ / (c_ + d), "odds_ratio": orr, "fisher_p": p, "auroc": st.auroc(M[c], y)})
    asc = pd.DataFrame(assoc)
    asc["bh_q"] = st.bh_fdr(asc.fisher_p)
    asc = asc.sort_values("fisher_p")
    ctx.table("dna_association", asc, "Single-gene mutation association with death (Fisher exact, BH FDR), genes mutated in at least 1%")
    alpha = ctx.cfg["eda"]["fdr_alpha"]
    ctx.facts.update(mut_auroc_min=round(asc.auroc.min(), 4), mut_auroc_max=round(asc.auroc.max(), 4),
                     n_mut_fdr=int((asc.bh_q < alpha).sum()),
                     mut_nominal=asc[asc.fisher_p < 0.05][["gene", "death_rate_mutated", "death_rate_wildtype", "fisher_p", "bh_q"]]
                     .round(4).to_dict("records"))
    nmut = M.sum(axis=1)
    fig, ax = P.fig(6, 3.4)
    vals = np.arange(0, int(nmut.max()) + 1)
    cnt = nmut.value_counts().reindex(vals, fill_value=0)
    ax.bar(vals, cnt, color=P.CAT[0], width=0.8)
    ax.set_xlabel("Mutated genes per patient (of 173)")
    ax.set_ylabel("Patients")
    ax.set_title("Mutation burden per patient")
    P.grid(ax)
    ctx.figure("dna_mutation_burden", fig, pd.DataFrame({"n_mutated_genes": vals, "patients": cnt.to_numpy()}),
               "Number of mutated genes among the 173 per patient")
    ctx.facts["median_mutated_genes"] = float(nmut.median())


# 9 ---------------------------------------------------------------------------------------------
def cohort(ctx) -> None:
    f, y = ctx.frame, ctx.y
    coh = f["cohort"].astype(str)
    rows = []
    for c in ["age_at_diagnosis", "tumor_size", "lymph_nodes_examined_positive", "nottingham_prognostic_index", "mutation_count"]:
        rows += [{"variable": c, "statistic": "median", "cohort": k, "value": v} for k, v in f[c].groupby(coh).median().items()]
    for c, val in [("er_status", "Positive"), ("her2_status", "Positive"), ("pr_status", "Positive"),
                   ("type_of_breast_surgery", "MASTECTOMY"), ("inferred_menopausal_state", "Post")]:
        rows += [{"variable": c, "statistic": f"share {val}", "cohort": k, "value": v} for k, v in (f[c] == val).groupby(coh).mean().items()]
    for c in ["chemotherapy", "hormone_therapy", "radio_therapy"]:
        rows += [{"variable": c, "statistic": "share treated", "cohort": k, "value": v} for k, v in f[c].groupby(coh).mean().items()]
    rows += [{"variable": "death", "statistic": "rate", "cohort": k, "value": v} for k, v in y.groupby(coh).mean().items()]
    ctx.table("cohort_clinical_profile", pd.DataFrame(rows), "Clinical and outcome profile by cohort")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    X = f[ctx.cfg.blocks["rna"]].to_numpy(float)
    cv = StratifiedKFold(ctx.cfg["eda"]["cohort_classifier_folds"], shuffle=True, random_state=ctx.cfg.seed)
    clf = LogisticRegression(C=0.05, max_iter=3000)
    proba = cross_val_predict(clf, X, coh, cv=cv, method="predict_proba")
    classes = sorted(coh.unique())
    res = [{"cohort": c, "n": int((coh == c).sum()), "auroc_one_vs_rest": roc_auc_score(coh == c, proba[:, i])} for i, c in enumerate(classes)]
    res.append({"cohort": "macro", "n": len(coh), "auroc_one_vs_rest": float(np.mean([r["auroc_one_vs_rest"] for r in res]))})
    ctx.table("cohort_from_rna_classifier", pd.DataFrame(res),
              "Predicting cohort from RNA (L2 logistic regression, C=0.05, 5-fold CV out-of-fold); high AUROC = batch structure")
    ctx.facts["cohort_rna_auroc_macro"] = round(res[-1]["auroc_one_vs_rest"], 4)
    ctx.facts["cohort_rna_auroc_by_cohort"] = {r["cohort"]: round(r["auroc_one_vs_rest"], 3) for r in res[:-1]}
    lo = []
    for c in classes:
        m = coh == c
        lo.append({"held_out_cohort": c, "n_held_out": int(m.sum()), "deaths_held_out": int(y[m].sum()),
                   "alive_held_out": int((y[m] == 0).sum()), "chemo_held_out": int(f.loc[m, "chemotherapy"].sum()),
                   "n_training": int((~m).sum()), "stage_missing_pct_held_out": 100 * f.loc[m, "tumor_stage"].isna().mean()})
    ctx.table("cohort_loco_feasibility", pd.DataFrame(lo), "Leave-one-cohort-out feasibility (stratum sizes when each cohort is held out)")
