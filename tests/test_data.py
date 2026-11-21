import pandas as pd
import pytest

from metabric.data import (DataValidationError, binarise_mutations, clean, make_target, missing_indicators,
                           read_raw, validate_raw)


def test_binarise_mutations_strings():
    df = pd.DataFrame({"a_mut": ["0", "H1047R", "Q546H G1049R", "0"], "b_mut": ["0", "0", "0", "X1_splice"]})
    out = binarise_mutations(df, ["a_mut", "b_mut"])
    assert out["a_mut"].tolist() == [0, 1, 1, 0]
    assert out["b_mut"].tolist() == [0, 0, 0, 1]
    assert str(out.dtypes.iloc[0]) == "int8"


def test_binarise_rejects_missing():
    df = pd.DataFrame({"a_mut": ["0", None]})
    with pytest.raises(DataValidationError):
        binarise_mutations(df, ["a_mut"])


def test_block_partition_disjoint_exhaustive(cfg, ds):
    b = cfg.blocks
    allcols = b["id"] + b["clinical"] + b["outcome"] + b["rna"] + b["dna"]
    assert len(allcols) == len(set(allcols)) == 693
    assert (len(b["clinical"]), len(b["outcome"]), len(b["rna"]), len(b["dna"])) == (27, 3, 489, 173)
    X = ds.features()
    assert X.shape == (1904, 27 + 489 + 173)
    for leak in b["outcome"] + b["id"]:
        assert leak not in X.columns
    assert ds.frame.index.name == "patient_id"


def test_target_orientation(ds):
    # file: overall_survival 1 = alive (801), 0 = died (1103); model positive class = death
    assert ds.y.name == "death"
    assert int(ds.y.sum()) == 1103 and int((ds.y == 0).sum()) == 801
    assert (ds.y == 1 - ds.frame["overall_survival"]).all()


def test_typo_normalised_and_cohort_categorical(ds):
    assert "Positve" not in set(ds.frame["er_status_measured_by_ihc"].dropna())
    assert set(ds.frame["er_status_measured_by_ihc"].dropna()) == {"Positive", "Negative"}
    assert isinstance(ds.frame["cohort"].dtype, pd.CategoricalDtype)
    assert ds.frame["cohort"].value_counts().sort_index().tolist() == [481, 286, 734, 236, 167]


def test_mutation_flags_match_rule(cfg, ds):
    raw = read_raw(cfg).set_index("patient_id")
    expected = (raw[cfg.blocks["dna"]].astype(str) != "0").astype("int8")
    pd.testing.assert_frame_equal(ds.mutations, expected)
    assert int((ds.mutations.mean() >= 0.01).sum()) == 115


def test_validation_fails_loudly(cfg):
    raw = read_raw(cfg)
    with pytest.raises(DataValidationError):
        validate_raw(raw.drop(columns=["brca1"]), cfg)
    bad = raw.copy()
    bad.loc[0, "tp53_mut"] = None
    with pytest.raises(DataValidationError):
        validate_raw(bad, cfg)
    with pytest.raises(DataValidationError):
        validate_raw(pd.concat([raw, raw.iloc[:1]]), cfg)


def test_missing_indicators_explicit_columns(ds):
    mi = missing_indicators(ds.frame, ["tumor_stage", "tumor_size"])
    assert list(mi.columns) == ["tumor_stage__missing", "tumor_size__missing"]
    assert int(mi["tumor_stage__missing"].sum()) == 501


def test_npi_standard_formula_and_missing():
    from metabric.data import npi_standard
    f = pd.DataFrame({"tumor_size": [22.0, 10.0, None, 30.0], "lymph_nodes_examined_positive": [10.0, 0.0, 2.0, 2.0],
                      "neoplasm_histologic_grade": [3.0, 3.0, 2.0, None]})
    out = npi_standard(f)
    assert out.name == "npi_standard"
    assert out.iloc[0] == pytest.approx(0.2 * 2.2 + 3 + 3) and out.iloc[1] == pytest.approx(0.2 * 1.0 + 1 + 3)
    assert out.iloc[2:].isna().all()


def test_file_npi_uses_0_02_size_weight(ds):
    """Guards the data fact: the supplied NPI is 0.02 x size(cm) + node stage + grade."""
    from metabric.data import npi_standard
    f = ds.frame
    ns = npi_standard(f)
    file_fit = ns - 0.2 * f["tumor_size"] / 10 + 0.02 * f["tumor_size"] / 10
    ok = (f["nottingham_prognostic_index"] - file_fit).abs() < 0.01
    assert ns.notna().sum() == 1815 and ok[ns.notna()].mean() > 0.99


def test_clean_does_not_mutate_input(cfg):
    raw = read_raw(cfg)
    before = raw["er_status_measured_by_ihc"].copy()
    clean(raw, cfg)
    pd.testing.assert_series_equal(raw["er_status_measured_by_ihc"], before)
    assert make_target(raw, cfg).sum() == 1103
