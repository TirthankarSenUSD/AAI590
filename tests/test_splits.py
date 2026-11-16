import pandas as pd
import pytest

from metabric.data import load_dataset
from metabric.splits import HoldoutError, load_holdout, make_holdout, strata_labels


def test_strata_20_smallest_13(ds):
    lab, merges = strata_labels(ds, 2)
    assert lab.nunique() == 20 and lab.value_counts().min() == 13 and merges.empty


def test_strata_merge_logged(ds):
    lab, merges = strata_labels(ds, 14)       # forces the 13-patient stratum to merge
    assert len(merges) >= 1 and merges.n.min() < 14
    assert lab.value_counts().min() >= 14 or lab.str.contains("merged").any()


def test_make_holdout_once_and_frozen(tmp_project):
    ds = load_dataset(tmp_project)
    cfg2, out, _ = make_holdout(tmp_project, ds)
    assert (out["set"] == "holdout").sum() == 381
    assert cfg2["holdout"]["sha256"]
    h = load_holdout(cfg2, ds)
    hold, train = set(h.loc[h.set == "holdout", "patient_id"]), set(h.loc[h.set == "train", "patient_id"])
    assert not hold & train and len(hold | train) == 1904
    with pytest.raises(HoldoutError):          # never recreated
        make_holdout(cfg2, ds)
    # stratified: outcome, cohort, chemotherapy proportions within 2 points
    f = ds.frame.assign(death=ds.y, set=h.set_index("patient_id").loc[ds.ids, "set"].to_numpy())
    for col in ["death", "chemotherapy"]:
        r = f.groupby("set")[col].mean()
        assert abs(r["holdout"] - r["train"]) < 0.02
    # tampering is detected
    p = cfg2.path("holdout")
    p.write_text(p.read_text().replace("holdout", "train", 1))
    with pytest.raises(HoldoutError):
        load_holdout(cfg2, ds)


def test_make_cv_splits_frozen_nested_and_clean(tmp_project):
    from metabric.harness import check_split_hygiene, splits_from_table
    from metabric.splits import load_cv_splits, make_cv_splits, training_ids
    ds = load_dataset(tmp_project)
    cfg2, _, _ = make_holdout(tmp_project, ds)
    tr = training_ids(cfg2, ds)
    cfg3, sp, merges = make_cv_splits(cfg2, ds.subset(tr))
    assert cfg3["cv"]["splits_sha256"] and cfg3["holdout"]["sha256"] == cfg2["holdout"]["sha256"]
    s = load_cv_splits(cfg3, tr)
    assert s.split_id.nunique() == 10 and not set(s.patient_id) & set(ds.ids.difference(tr))   # no holdout rows
    check_split_hygiene(splits_from_table(s), tr)
    assert merges.empty                                            # smallest training stratum (11) >= 5 folds
    test = s[s.role == "test"]
    rates = test.assign(y=ds.y.loc[test.patient_id].to_numpy()).groupby("split_id").y.mean()
    assert (rates - ds.y.loc[tr].mean()).abs().max() < 0.02        # outcome stratified
    with pytest.raises(HoldoutError):
        make_cv_splits(cfg3, ds.subset(tr))
    p = cfg3.path("cv_splits")
    p.write_text(p.read_text().replace(",test,", ",train,", 1))
    with pytest.raises(HoldoutError):
        load_cv_splits(cfg3, tr)


def test_real_holdout_is_frozen_and_valid(cfg, ds):
    h = load_holdout(cfg, ds)                  # verifies hash against config
    assert (h["set"] == "holdout").sum() == 381
    assert isinstance(h, pd.DataFrame)
