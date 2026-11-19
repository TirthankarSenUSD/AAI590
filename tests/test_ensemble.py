"""Step 5: ensemble building blocks behave as specified, and the stage runs end to end."""
import numpy as np
import pandas as pd
import pytest
import yaml

from metabric.ensemble import stack as S


def _members(n=600, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-1.5 * z))).astype(int))
    good = 1.5 * z + 0.3 * rng.normal(size=n)
    P = pd.DataFrame({"good": S.sigmoid(good),
                      "copy": S.sigmoid(good + 0.1 * rng.normal(size=n)),     # near-duplicate, slightly noisier
                      "noise": S.sigmoid(rng.normal(size=n)),
                      "anti": S.sigmoid(-1.5 * z)})
    fold = pd.Series(np.arange(n) % 3)
    return y, P, fold


def test_stacker_weights_are_non_negative_and_favour_signal():
    y, P, _ = _members()
    s = S.NonNegStacker(1.0).fit(S.logit(P.to_numpy()), y)
    w = dict(zip(P.columns, s.coef_))
    assert all(v >= 0 for v in w.values())
    assert w["anti"] < 1e-6 and w["noise"] < 0.2 and w["good"] + w["copy"] > 0.5


def test_merge_joins_near_duplicates_only():
    y, P, _ = _members()
    groups = S.merge_groups(P, 0.95)
    assert ["good", "copy"] in groups and ["noise"] in groups and ["anti"] in groups
    mw = S.merge_weights(groups, {c: S._auc(y, P[c]) for c in P.columns})
    M = S.apply_merge(P, mw)
    assert list(M.columns) == ["good+copy", "noise", "anti"] and ((M > 0) & (M < 1)).all().all()


def test_options_and_choice():
    y, P, fold = _members()
    cx = {"good": 1, "copy": 2, "noise": 3, "anti": 4}
    for o in S.OPTIONS:
        f, params = S.fit_option(o, y, P, fold, cx, 1.0, 1.0)
        p = f(P)
        assert p.shape == (len(P),) and ((p >= 0) & (p <= 1)).all()
    f, params = S.fit_option("best_single", y, P, fold, cx, 1.0, 1.0)
    assert params["member"] == "good"                                  # simplest within 1 SE of the best
    opt, parts = S.choose_option(y, P[["good", "noise"]], fold, cx, 1.0, 1.0, 3, 0)
    assert opt in S.OPTIONS and set(parts.option) == set(S.OPTIONS)


def test_ensemble_stage_end_to_end_small(tmp_project):
    from metabric.core.artifacts import Run
    from metabric.core.config import Config
    from metabric.data import load_dataset
    from metabric.ensemble.stage import run_ensemble
    from metabric.farm.stage import run_farm
    from metabric.splits import make_cv_splits, make_holdout, training_ids
    ds = load_dataset(tmp_project)
    c, _, _ = make_holdout(tmp_project, ds)
    c, _, _ = make_cv_splits(c, ds.subset(training_ids(c, ds)))
    f = c.raw["farm"]
    f.update(n_jobs=2, n_permutations=1, permutation_arm="C",
             grids={"logreg_en": {"C": [0.1], "l1_ratio": [0.0]}, "logreg_cause": {"C": [0.1], "l1_ratio": [0.0]},
                    "xgboost": {"n_estimators": [20], "learning_rate": [0.1], "max_depth": [2], "subsample": [0.8],
                                "colsample_bytree": [0.8], "min_child_weight": [5]}})
    c.raw["harness"].update(n_boot=10, chance_band=0.5)
    c.raw["ensemble"].update(arms=["C", "strict_clinical", "late_fusion"], members=["logreg_en", "logreg_cause", "xgboost"],
                             late_fusion=["C|best", "strict_clinical|best"])
    c.file.write_text(yaml.safe_dump(c.raw), encoding="utf-8")
    c = Config(root=c.root, raw=yaml.safe_load(c.file.read_text(encoding="utf-8")))
    r1 = Run(c, "farm", run_id="farm_a")
    run_farm(c, r1, members=["logreg_en", "xgboost"], arms=["C", "strict_clinical"])
    r1.finalize()
    r2 = Run(c, "farm", run_id="farm_b")
    run_farm(c, r2, members=["logreg_en", "logreg_cause"], arms=["C", "strict_clinical"])
    r2.finalize()
    r3 = Run(c, "ensemble", run_id="ens")
    facts = run_ensemble(c, r3, ["farm_a", "farm_b"])
    r3.finalize()
    chk = pd.read_csv(r3.dir / "ensemble" / "tables" / "refit_check.csv")
    assert chk.abs_diff.max() < 1e-6 and set(chk[chk.arm != "late_fusion"].member) == {"logreg_en", "xgboost", "logreg_cause"}
    assert set(chk[chk.member == "logreg_en"].farm_run) == {"farm_a"}           # first listed run wins
    assert facts["final"]["arm"] == "C"                                         # baseline unless significantly beaten
    assert facts["final"]["regression_floor"] < facts["final"]["mean_outer_auroc"]
    F = pd.read_csv(r3.dir / "ensemble" / "tables" / "option_fold_metrics.csv")
    assert set(F.candidate) >= {"C|best_single", "C|mean", "C|weighted", "C|stacker", "C|ensemble", "late_fusion|ensemble"}
    lf = chk[chk.arm == "late_fusion"]
    assert set(lf.member) == {"C", "strict_clinical"} and lf.abs_diff.max() < 1e-6       # one model per source arm
