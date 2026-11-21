"""Phase 2: survival data rules, the ridge Cox (against scikit-survival), strata, members, metrics,
the checks, and the configuration grid. Synthetic data, or training rows only."""
import numpy as np
import pandas as pd
import pytest

from metabric.phase2.checks import collinearity, km_by_group, ph_test
from metabric.phase2.cv import configurations, survival_metrics
from metabric.phase2.models import KMMember, RidgeCox
from metabric.phase2.survival import npi_band, structured, survival_data


def _sim(n=600, seed=0, strata_effect=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    s = rng.integers(0, 2, n)
    base = np.where(s == 1, 3.0, 1.0) if strata_effect else 1.0
    T = rng.exponential(1 / (base * np.exp(X @ np.array([0.9, -0.5, 0.0])))) * 40
    C = rng.exponential(60, n)
    return X, np.round(np.minimum(T, C), 1), (T <= C).astype(int), s


def test_ridge_cox_matches_scikit_survival_with_ties():
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    X, t, e, _ = _sim()
    m = RidgeCox(0.01).fit(X, t, e)
    ref = CoxPHSurvivalAnalysis(alpha=0.01 * len(t), ties="breslow").fit(X, structured(t, e))
    assert m.converged_ and np.abs(m.coef_ - ref.coef_).max() < 1e-4
    S = m.survival(X[:20], [10, 30])
    R = np.array([f([10, 30]) for f in ref.predict_survival_function(X[:20])])
    assert np.abs(S - R).max() < 1e-4
    assert np.all(np.diff(m.survival(X[:5], [1, 10, 30, 60]), axis=1) <= 1e-12)    # survival never rises


def test_stratified_cox_uses_a_baseline_per_stratum():
    X, t, e, s = _sim(strata_effect=True)
    m = RidgeCox(0.001).fit(X, t, e, strata=s)
    assert set(m.baseline_) == {0, 1, "__pooled__"}
    x0 = np.zeros((1, 3))
    s0, s1 = m.survival(x0, [20], [0])[0, 0], m.survival(x0, [20], [1])[0, 0]
    assert s1 < s0                                                      # stratum 1 has the higher baseline hazard
    assert abs(m.coef_[0] - 0.9) < 0.2 and abs(m.coef_[1] + 0.5) < 0.2
    from metabric.phase2.models import _step
    ut, H = m.baseline_["__pooled__"]
    assert m.survival(x0, [20], ["unseen"])[0, 0] == pytest.approx(np.exp(-_step(ut, H, [20])[0]))   # unseen -> pooled baseline


def test_survival_metrics_rank_a_good_model_above_a_constant():
    X, t, e, _ = _sim(n=900)
    tr, te = np.arange(600), np.arange(600, 900)
    m = RidgeCox(0.01).fit(X[tr], t[tr], e[tr])
    good = survival_metrics(t[tr], e[tr], t[te], e[te], m.survival(X[te], [30, 60]), [30, 60], 1)
    km = KMMember(None, {}, 0).fit(None, t[tr], e[tr])
    const = survival_metrics(t[tr], e[tr], t[te], e[te], km.survival(np.zeros((300, 1)), [30, 60]), [30, 60], 1)
    assert good["c_index"] > 0.65 and const["c_index"] == pytest.approx(0.5)
    assert good["auc_30"] > 0.65 and const["auc_30"] == pytest.approx(0.5)
    assert good["brier_30"] < const["brier_30"]
    assert abs(good["cal_large_30"]) < 0.1


def test_km_logrank_ph_and_vif_behave_on_synthetic_data():
    X, t, e, s = _sim(n=800, strata_effect=True)
    idx = pd.RangeIndex(800)
    curves, at, test = km_by_group(pd.Series(t, idx), pd.Series(e, idx), pd.Series(np.where(s == 1, "b", "a"), idx), [20, 40])
    assert test["p"] < 0.001 and set(at.group) == {"a", "b"}
    Xp, tp, ep, _ = _sim(n=800, seed=3)                                 # proportional hazards, nothing left out
    ph, _, _ = ph_test(pd.DataFrame(Xp, columns=["x0", "x1", "x2"]), pd.Series(tp), pd.Series(ep), 0.05)
    assert not ph.ph_violated.any()
    rng = np.random.default_rng(4)                                      # x's effect reverses over time: must be flagged
    x = rng.integers(0, 2, 1500).astype(float)
    T = np.where(x == 1, rng.uniform(0, 30, 1500), rng.uniform(15, 45, 1500))
    ph2, _, _ = ph_test(pd.DataFrame({"x": x}), pd.Series(T), pd.Series(np.ones(1500, int)), 0.05)
    assert ph2.ph_violated.all()
    f = pd.DataFrame({"a": X[:, 0], "b": X[:, 1]})
    f["c"] = f.a + f.b + np.random.default_rng(1).normal(scale=0.1, size=800)
    _, vif = collinearity(f, ["a", "b", "c"])
    assert vif.set_index("column").loc["c", "vif"] > 10


def test_configurations_are_listed_simplest_first(cfg):
    cs = configurations(cfg["phase2"]["members"])
    cox = [c for c in cs if c["member"] == "cox"]
    assert cox[0]["params"] == {"stratify": None, "alpha": 0.1} and cox[0]["complexity"] == 0
    assert len({c["name"] for c in cs}) == len(cs)


def test_survival_data_zero_time_rule_and_npi_bands(cfg, ds):
    sv, rule = survival_data(cfg, ds)
    assert (sv.time > 0).all() and rule["patients"] == [284] and rule["events_at_zero"] == 0
    assert rule["new_time"] == pytest.approx(0.05)
    assert sv.event.sum() == 1103
    b = npi_band(ds.frame, cfg["phase2"]["npi_bands"])
    assert set(b.dropna()) == {"good", "moderate", "poor"} and b.isna().sum() == 89


def test_final_rule_prefers_the_simplest_member_within_one_se():
    from metabric.phase2.cv import choose_final
    rng = np.random.default_rng(0)
    rows = []
    for k in range(25):
        base = rng.normal(0, 0.02)
        rows += [{"model": "cox", "split_id": k, "c_index": 0.675 + base + rng.normal(0, 0.005)},
                 {"model": "rsf", "split_id": k, "c_index": 0.683 + base + rng.normal(0, 0.005)},
                 {"model": "weak", "split_id": k, "c_index": 0.60 + base + rng.normal(0, 0.005)}]
    folds = pd.DataFrame(rows)
    f, t = choose_final(folds, {"complexity_order": ["weak", "cox", "rsf"], "within_se": 1.0}, 1218, 305)
    assert f == "cox" and not t.set_index("model").loc["weak", "within_se_of_best"]
    f2, _ = choose_final(folds, {"complexity_order": ["weak", "cox", "rsf"], "within_se": 0.0}, 1218, 305)
    assert f2 == "rsf"


def test_patient_survival_records_add_up_and_match_batch(cfg, ds):
    from metabric.harness.floors import design_matrix
    from metabric.phase2.final import SurvivalModel
    from metabric.phase2.models import columns_for, make
    from metabric.splits import training_ids
    tr = training_ids(cfg, ds)[:600]                                     # training rows only
    sub = ds.subset(tr)
    sv, _ = survival_data(cfg, sub)
    X = design_matrix(sub)
    p = {"stratify": None, "alpha": 0.1}
    cols = columns_for(cfg, "cox", p)
    m = make("cox", cfg, p, 0).fit(X[cols], sv.time.to_numpy(), sv.event.to_numpy())
    boot = [make("cox", cfg, p, 0).fit(X[cols].iloc[i], sv.time.to_numpy()[i], sv.event.to_numpy()[i])
            for i in [np.random.default_rng(k).integers(0, 600, 600) for k in range(5)]]
    risk = 1 - m.survival(X[cols], [120])[:, 0]
    sm = SurvivalModel("cox", "cox[x]", p, m, cols, [60.0, 120.0], 120.0, risk, np.quantile(risk, [0.25, 0.5, 0.75]),
                       {g: {"n": 1} for g in ["low", "below average", "above average", "high"]}, boot,
                       zbar=m.prep_.transform(X[cols]).mean(axis=0), numeric_range={"age_at_diagnosis": [30.0, 80.0]},
                       curve_months=list(range(0, 241, 12)))
    C = sm.contributions(X.iloc[:50])
    lp = m.prep_.transform(X[cols].iloc[:50]) @ m.m_.coef_ - sm.zbar @ m.m_.coef_
    assert np.abs(C.sum(axis=1).to_numpy() - lp).max() < 1e-10                       # exact for the Cox model
    recs = sm.records(X.iloc[:5], list(X.index[:5]))
    one = sm.records(X.iloc[[3]], [X.index[3]])[0]
    for h in ("60_months", "120_months"):                                            # single patient == batch
        assert one["survival"][h]["probability"] == pytest.approx(recs[3]["survival"][h]["probability"], abs=1e-12)
        assert one["survival"][h]["interval_95"] == pytest.approx(recs[3]["survival"][h]["interval_95"], abs=1e-12)
    assert one["risk"] == recs[3]["risk"]
    for r in recs:
        s = [c["survival"] for c in r["survival_curve"]]
        assert s[0] == pytest.approx(1.0) and all(a >= b - 1e-12 for a, b in zip(s, s[1:]))
        five = r["survival"]["60_months"]
        assert 0 <= five["probability"] <= 1 and five["interval_95"][0] <= five["interval_95"][1]
        assert r["survival"]["120_months"]["probability"] <= five["probability"]
        assert r["risk"]["risk_group"] in ("low", "below average", "above average", "high") and "disclaimer" in r
    old = X.iloc[[0]].copy()
    old["age_at_diagnosis"] = 95.0
    assert sm.records(old)[0]["flags"]                                                # outside the training range


def test_phase2_holdout_marker_is_separate_from_phase1(tmp_project):
    from metabric.report.holdout import check_marker, read_marker, write_marker
    from metabric.splits.frozen import HoldoutError
    write_marker(tmp_project, "phase2_holdout_scored", status="complete", bundle_sha256="a")
    assert read_marker(tmp_project) is None                                           # Phase 1 marker untouched
    with pytest.raises(HoldoutError, match="already scored"):
        check_marker(tmp_project, "a", "phase2_holdout_scored")
