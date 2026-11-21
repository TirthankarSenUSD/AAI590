"""One-time holdout scoring: the scoring function on synthetic data, the report text, and the guard
that refuses a second scoring. No test touches the real holdout labels."""
import numpy as np
import pandas as pd
import pytest

from metabric.report import text as T
from metabric.report.holdout import check_marker, evaluate_holdout, read_marker, write_marker
from metabric.splits.frozen import HoldoutError


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * z))).astype(int)
    p = 1 / (1 + np.exp(-1.5 * z))                               # the true risk: a good, calibrated model
    floors = {"weak": 1 / (1 + np.exp(-0.3 * (z + rng.normal(scale=2, size=n)))), "majority": np.full(n, y.mean())}
    groups = pd.DataFrame({"chemotherapy": np.where(rng.random(n) < 0.3, "yes", "no")})
    cut = np.quantile(p, [0.25, 0.5, 0.75])
    labels = ["low", "below average", "above average", "high"]
    rates = {lab: {"observed_death_rate": 0.1 + 0.25 * k} for k, lab in enumerate(labels)}
    yc = np.where(y == 1, rng.integers(1, 3, n), 0)
    cause = pd.DataFrame({"p_cancer_death": p * 0.5, "p_other_death": p * 0.5})
    return y, p, floors, groups, cut, labels, rates, cause, yc


def test_evaluate_holdout_recovers_a_good_model_and_beats_weak_floors():
    y, p, floors, groups, cut, labels, rates, cause, yc = _synthetic()
    E = evaluate_holdout(y, p, floors, float(y.mean()), {"F1-maximising": 0.5}, groups, cut, labels, rates,
                         n_boot=200, alpha=0.05, seed=1, cause=cause, y_cause=yc)
    m = E["metrics"].set_index(["model", "metric"])
    a = m.loc[("final_model", "auroc")]
    assert 0.75 < a.value < 0.9 and a.ci_low < a.value < a.ci_high
    assert abs(m.loc[("final_model", "cal_slope"), "value"] - 1) < 0.4
    c = E["floor_comparisons"].set_index(["b", "metric"])
    assert c.loc[("weak", "auroc"), "diff"] > 0 and c.loc[("weak", "auroc"), "ci_low"] > 0 and c.loc[("weak", "auroc"), "p"] < 0.05
    assert c.loc[("majority", "brier"), "diff"] < 0                # lower Brier is better
    assert set(E["operating_points"].metric) == {"sensitivity", "specificity", "ppv", "npv", "f1"}
    rg = E["risk_groups"]
    assert rg.n.sum() == len(y) and rg.observed_death_rate.is_monotonic_increasing
    assert set(E["subgroups"].level) == {"yes", "no"}
    assert 0 < E["cause"]["auroc_cancer"] < 1
    # same seed -> identical results
    E2 = evaluate_holdout(y, p, floors, float(y.mean()), {"F1-maximising": 0.5}, groups, cut, labels, rates,
                          n_boot=200, alpha=0.05, seed=1, cause=cause, y_cause=yc)
    pd.testing.assert_frame_equal(E["metrics"], E2["metrics"])


def test_holdout_report_text_is_generated_and_marks_a_dry_run():
    y, p, floors, groups, cut, labels, rates, cause, yc = _synthetic()
    E = evaluate_holdout(y, p, floors, float(y.mean()), {"F1-maximising": 0.5}, groups, cut, labels, rates,
                         n_boot=100, alpha=0.05, seed=1, cause=cause, y_cause=yc)
    met = lambda name: {r.metric: {"mean": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high}  # noqa: E731
                        for r in E["metrics"][E["metrics"].model == name].itertuples()}
    F = {"dry_run": True, "runs": {"report": "R", "ensemble": "E", "holdout": "H"}, "bundle_sha256": "ab" * 32,
         "config_hash": "x", "n": len(y), "deaths": int(y.sum()), "base_rate": float(y.mean()), "train_base_rate": 0.58,
         "n_boot": E["n_boot_used"], "final": met("final_model"), "floor_metrics_full": met("weak"),
         "floor_comparisons": E["floor_comparisons"].to_dict("records"), "operating_points": E["operating_points"].to_dict("records"),
         "subgroups": E["subgroups"].to_dict("records"), "risk_groups": E["risk_groups"].to_dict("records"), "cause": E["cause"],
         "cv": {"auroc": 0.0, "ci": [0.0, 0.01], "regression_floor": 0.99}, "model": {"chosen_column": "c", "primary": "raw", "arm": "C"}}
    s = T.holdout_report(F)
    assert "DRY RUN" in s and "lies outside" in s and "is below it" in s
    assert "rise monotonically" in s
    F["dry_run"] = False
    F["cv"] = {"auroc": F["final"]["auroc"]["mean"], "ci": [0.7, 0.9], "regression_floor": 0.5}
    s = T.holdout_report(F)
    assert "DRY RUN" not in s and "lies inside" in s and "is above it" in s


def test_marker_refuses_a_second_scoring(tmp_project):
    cfg = tmp_project
    assert read_marker(cfg) is None and check_marker(cfg, "sha1") is None
    write_marker(cfg, status="started", bundle_sha256="sha1", run_id="r1")
    assert check_marker(cfg, "sha1")["status"] == "started"        # interrupted: same model may finish it
    with pytest.raises(HoldoutError, match="different final model"):
        check_marker(cfg, "sha2")
    write_marker(cfg, status="complete", completed="t")
    with pytest.raises(HoldoutError, match="already scored"):
        check_marker(cfg, "sha1")
    assert read_marker(cfg)["run_id"] == "r1"                        # fields are merged, not replaced
