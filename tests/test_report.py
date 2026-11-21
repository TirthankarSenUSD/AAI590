"""Step 7: report wording, Shapley additivity, deployment fit and patient records."""
import copy

import numpy as np
import pandas as pd

from metabric.report import text as T
from metabric.report.patient import shapley_log_odds


def _close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, z, tol) for x, z in zip(a, b))
    if isinstance(a, float):
        return abs(a - b) <= tol * max(1.0, abs(a))
    return a == b


def test_wording_follows_the_numbers():
    rows = [{"arm": "C+RNA", "diff": 0.008, "ci_low": -0.007, "ci_high": 0.023, "p": 0.25}]
    assert T.omics_conclusion(rows, "C").startswith("No arm was significantly better")
    flipped = [{"arm": "C+RNA", "diff": 0.030, "ci_low": 0.010, "ci_high": 0.050, "p": 0.01}]
    assert "C+RNA was significantly better" in T.omics_conclusion(flipped, "C")
    # same metric on both sides, and direction depends on the metric
    line = T.compare_line("C|best", "age_only", "brier", -0.026, -0.035, -0.016, 0.001)
    assert "brier on both sides" in line and "significantly better" in line
    assert T.verdict("auroc", -0.026, -0.035, -0.016, 0.001) == "significantly worse"
    assert T.verdict("auroc", 0.02, -0.01, 0.05, 0.2) == "no significant difference"


def test_shapley_is_additive_and_exact_for_an_additive_model():
    rng = np.random.default_rng(0)
    bg = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
    w = np.array([1.0, -2.0, 0.5, 0.0])
    f = lambda D: D[list("abcd")].to_numpy() @ w + 0.3
    x = pd.DataFrame([[1.0, 2.0, -1.0, 5.0]], columns=list("abcd"))
    phi, base, value = shapley_log_odds(f, x, bg, list("abcd"), 6, 1)
    assert abs(base + phi.sum() - value) < 1e-9 and abs(value - f(x)[0]) < 1e-9
    assert np.allclose(phi.to_numpy(), w * (x.iloc[0].to_numpy() - bg.mean().to_numpy()))
    # an interaction: still exactly additive (efficiency holds for every sampled permutation)
    g = lambda D: (D.a * D.b + D.c).to_numpy()
    phi, base, value = shapley_log_odds(g, x, bg, list("abcd"), 4, 2)
    assert abs(base + phi.sum() - value) < 1e-9


def test_deployment_and_patient_records(cfg, ds):
    from metabric.core.config import Config
    from metabric.farm.cause import cause_labels
    from metabric.harness.floors import design_matrix
    from metabric.report.deploy import fit_final_model
    from metabric.report.patient import PatientExplainer
    raw = copy.deepcopy(cfg.raw)
    raw["farm"]["grids"]["logreg_en"] = {"C": [0.1], "l1_ratio": [0.0]}
    raw["farm"]["grids"]["logreg_cause"] = {"C": [0.1], "l1_ratio": [0.0]}
    raw["ensemble"]["members"] = ["logreg_en", "logreg_cause"]
    raw["report"].update(n_boot=3)
    c = Config(root=cfg.root, raw=raw)
    ids = ds.ids[:500]
    sub = ds.subset(ids)
    X, y = design_matrix(sub), sub.y.astype(int)
    par = lambda fn, tasks, label: [fn(*t) for t in tasks]
    fm, tabs = fit_final_model(c, X, y, cause_labels(sub.frame), "C", "best_single", par, lambda *a: None)
    assert set(fm.group) <= {"logreg_en", "logreg_cause"} and abs(sum(fm.weights.values()) - 1) < 1e-9
    assert len(fm.boot) == 3 and fm.cause_model is not None
    p = fm.predict(X)
    assert ((p >= 0) & (p <= 1)).all()
    oof = tabs["deploy_oof_predictions"]
    assert oof.p_raw.notna().all() and len(oof) == len(y)           # every patient scored out of fold
    types = c["clinical_types"]
    ex = PatientExplainer(model=fm, blocks=c.blocks, numeric=[k for k in types["numeric"] + types["ordinal"] if k in fm.columns],
                          categorical=[k for k in types["categorical"] if k in fm.columns], train_X=X[fm.columns],
                          train_oof=oof.p_primary.to_numpy(), train_y=y.to_numpy(), context_col="pam50_+_claudin-low_subtype",
                          context_values=sub.frame["pam50_+_claudin-low_subtype"], mutation_prevalence=sub.mutations.mean(),
                          quality={}, n_background=30, n_perm=4, seed=1)
    rows = pd.concat([sub.frame[c.blocks["clinical"]], sub.mutations], axis=1).iloc[:3]
    batch = ex.records(rows.reset_index(drop=True), list(rows.index))
    for r in batch:
        assert 0 <= r["risk"]["probability_of_death"] <= 1
        lo, hi = r["risk"]["interval_95"]
        assert 0 <= lo <= hi <= 1
        cb = r["contributions"]
        assert abs(cb["sum_check"] - cb["model_output"]) < 1e-8          # contributions + base = model output
        assert abs(cb["model_output"] - np.log(r["risk"]["probability_of_death"] / (1 - r["risk"]["probability_of_death"]))) < 1e-6
        assert not cb["by_block"]["rna"]["used_by_model"]
        assert r["cohort_context"]["subtype"]["n"] > 0
    single = ex.record(rows.iloc[1], rows.index[1])
    assert _close(single, batch[1])                                    # single == batch (up to float rounding)
    # out-of-range input is flagged, missing input is echoed with its handling
    odd = rows.iloc[0].copy()
    odd["age_at_diagnosis"] = 150.0
    odd["tumor_size"] = np.nan
    r = ex.record(odd, "odd")
    assert any("age_at_diagnosis" in f for f in r["flags"])
    assert r["inputs"]["tumor_size"]["status"] == "missing" and "median" in r["inputs"]["tumor_size"]["handling"]


def test_every_member_pipeline_can_be_saved(cfg):
    """The deployment bundle is saved with joblib (plain pickle): no lambdas inside fitted objects."""
    import pickle

    from metabric.farm.members import build_candidates
    cands, meta = build_candidates(cfg, arms=["C", "C+RNA"])
    seen = set()
    for c in cands:
        if meta[c.name].member not in seen:
            seen.add(meta[c.name].member)
            pickle.dumps(c.make())
    assert {"logreg_en", "mlp_embed", "multimodal"} <= seen


def test_report_pack_helpers():
    from metabric.report.pack import SECTIONS, _demote
    assert _demote("# A\ntext\n## B", 2) == "### A\ntext\n#### B"
    keys = [s[0] for s in SECTIONS]
    assert len(keys) == len(set(keys)) and len({s[2] for s in SECTIONS}) == len(SECTIONS)


def test_findings_appendix_indexes_and_keeps_every_entry():
    from metabric.report.pack import findings_appendix
    log = "# Log\n\nintro\n\n### X-001 · 2026-01-01 · data · First\nBody one.\n**Status:** done.\n\n### X-002 · 2026-01-02 · design · Second\n#### sub\nBody two.\n"
    L = findings_appendix(log)
    s = "\n".join(L)
    assert "| X-001 | 2026-01-01 | data | First |" in s and "| X-002 | 2026-01-02 | design | Second |" in s
    assert "Body one." in s and "Body two." in s and "##### sub" in s           # nested headings pushed down
    assert "**data** (1): X-001" in s and findings_appendix("no entries") == []
