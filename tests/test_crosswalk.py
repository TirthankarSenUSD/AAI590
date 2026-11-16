import pandas as pd
import pytest
import yaml

from metabric.core.config import Config
from metabric.crosswalk import GeneSetIndex, build_crosswalk, canonical_symbol, read_gmt, write_crosswalk


def _mini(tmp_path, aliases="old_symbol,current_symbol,note\n"):
    root = tmp_path / "mini"
    (root / "reference" / "crosswalk").mkdir(parents=True)
    (root / "reference" / "a.gmt").write_text(
        "SET_A\turl\tBRCA1\tTP53\tBRCA\tORC6L\tX1\tX2\tX3\n"
        "SET_B\turl\tbrca2\tTP53\n", encoding="utf-8")
    (root / "reference" / "b.gmt").write_text("SET_C\turl\tHLA-G\n", encoding="utf-8")
    (root / "reference" / "ignored.gmt").write_text("SET_Z\turl\tBRCA1\n", encoding="utf-8")
    (root / "reference" / "crosswalk" / "symbol_aliases.csv").write_text(aliases, encoding="utf-8")
    raw = {"seed": 42, "paths": {"reference": "reference", "crosswalk": "reference/crosswalk"},
           "gene_sets": {"files": {"src_a": "a.gmt", "src_b": "b.gmt"},
                         "aliases": "reference/crosswalk/symbol_aliases.csv", "min_rna_overlap": 2},
           "columns": {"id": "patient_id", "clinical": [], "outcome": [],
                       "rna": ["brca1", "brca2", "hla-g", "brca1x", "orc6"], "dna": ["tp53_mut", "brca1_mut"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return Config(root=root, raw=raw)


def test_canonical_symbol():
    assert canonical_symbol("brca1") == "BRCA1"
    assert canonical_symbol("tp53_mut") == "TP53"
    assert canonical_symbol("hla-g") == "HLA-G"


def test_rules_exact_match_both_blocks_only_configured_files(tmp_path):
    cfg = _mini(tmp_path)
    res = build_crosswalk(cfg)
    xw = res.crosswalk.set_index(["column", "block"])
    assert xw.loc[("brca1", "rna"), "n_sets_src_a"] == 1
    assert xw.loc[("brca1_mut", "dna"), "n_sets_src_a"] == 1        # gene in both blocks -> both rows
    assert xw.loc[("brca2", "rna"), "matched"]                       # case-insensitive
    assert not xw.loc[("brca1x", "rna"), "matched"]                  # no partial matching (BRCA != BRCA1X)
    assert xw.loc[("hla-g", "rna"), "n_sets_src_b"] == 1
    assert not xw.loc[("orc6", "rna"), "matched"]                    # no alias listed -> not guessed
    assert "SET_Z" not in set(res.summary.set)                       # unconfigured file ignored
    s = res.summary.set_index("set")
    assert s.loc["SET_A", "n_rna_matched"] == 1 and s.loc["SET_A", "n_dna_matched"] == 2
    assert not s.loc["SET_A", "passes_min_overlap"]
    assert set(res.unmatched.column) == {"brca1x", "orc6"}


def test_alias_applied_and_reported(tmp_path):
    cfg = _mini(tmp_path, "old_symbol,current_symbol,note\nORC6L,ORC6,renamed\n")
    res = build_crosswalk(cfg)
    assert res.crosswalk.set_index("column").loc["orc6", "matched"]
    assert res.alias_log.set_symbol.tolist() == ["ORC6L"]


def test_lookup_helpers(tmp_path):
    cfg = _mini(tmp_path)
    write_crosswalk(build_crosswalk(cfg), cfg)
    idx = GeneSetIndex(cfg)
    g = idx.genes_of_set("SET_A", block=None)
    assert set(g.column) == {"brca1", "tp53_mut", "brca1_mut"} and set(g.source) == {"src_a"}
    sets = idx.sets_of("tp53_mut")
    assert set(sets.set) == {"SET_A", "SET_B"} and "source" in sets
    assert set(idx.sets_of("TP53").set) == {"SET_A", "SET_B"}
    with pytest.raises(ValueError):
        idx.genes_of_set("SET_A", usable_only=True)


def test_read_gmt_dedup_and_case(tmp_path):
    p = tmp_path / "x.gmt"
    p.write_text("S\td\tabc\tABC\tdef\n", encoding="utf-8")
    assert read_gmt(p) == {"S": ["ABC", "DEF"]}


def test_real_crosswalk_tables_match_expected_coverage(cfg):
    """Sanity check on the written tables (run `python run.py crosswalk` first)."""
    idx = GeneSetIndex(cfg)
    xw = idx.crosswalk
    assert len(xw) == 489 + 173
    assert set(xw.block) == {"rna", "dna"}
    s = idx.summary
    assert int(s[s.source == "hallmark"].passes_min_overlap.sum()) == 40
    assert set(s.source) == {"hallmark", "reactome"}
    assert isinstance(xw, pd.DataFrame)
