"""Gene crosswalk (build step 1a): dataset gene columns <-> gene-set files in reference/.

Rules (CLAUDE.md section 3b): only the gene-set files named in config; upper-case and strip
`_mut`; exact symbol matching only (no partial strings); a gene measured in both blocks gets
one row per block; aliases come only from the hand-maintained symbol_aliases.csv and every
applied alias is reported. Annotation only: outcome labels are never read here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import Config

OUTPUT_FILES = ("gene_crosswalk.csv", "gene_set_membership.csv", "gene_set_summary.csv", "unmatched_genes.csv")


def canonical_symbol(column: str) -> str:
    """`brca1` -> BRCA1, `tp53_mut` -> TP53, `hla-g` -> HLA-G."""
    sym = column[:-4] if column.endswith("_mut") else column
    return sym.strip().upper()


def read_gmt(path: Path) -> dict[str, list[str]]:
    """GMT: name <tab> description/url <tab> gene ... Genes upper-cased, de-duplicated in order."""
    sets: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                if line.strip():
                    raise ValueError(f"{path.name}:{line_no}: malformed GMT line")
                continue
            if parts[0] in sets:
                raise ValueError(f"{path.name}: duplicate set name {parts[0]!r}")
            genes = list(dict.fromkeys(g.strip().upper() for g in parts[2:] if g.strip()))
            sets[parts[0]] = genes
    return sets


def read_aliases(path: Path) -> dict[str, str]:
    if not Path(path).is_file():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for _, r in df.iterrows():
        old, new = r["old_symbol"].strip().upper(), r["current_symbol"].strip().upper()
        if old and new:
            out[old] = new
    return out


@dataclass
class CrosswalkResult:
    crosswalk: pd.DataFrame
    membership: pd.DataFrame
    summary: pd.DataFrame
    unmatched: pd.DataFrame
    coverage: pd.DataFrame
    alias_log: pd.DataFrame
    source_files: list[Path]


def build_crosswalk(cfg: Config) -> CrosswalkResult:
    ref = cfg.path("reference")
    files = {src: ref / name for src, name in cfg["gene_sets"]["files"].items()}
    aliases = read_aliases(cfg.root / cfg["gene_sets"]["aliases"])
    min_rna = int(cfg["gene_sets"]["min_rna_overlap"])

    # dataset features: one row per (column, block)
    feats = [(c, "rna") for c in cfg.blocks["rna"]] + [(c, "dna") for c in cfg.blocks["dna"]]
    feat = pd.DataFrame(feats, columns=["column", "block"])
    feat["symbol_raw"] = feat["column"].map(canonical_symbol)
    feat["symbol"] = feat["symbol_raw"].map(lambda s: aliases.get(s, s))
    feat["alias_applied"] = feat["symbol"] != feat["symbol_raw"]
    if feat.duplicated(["symbol", "block"]).any():
        dups = feat[feat.duplicated(["symbol", "block"], keep=False)]
        raise ValueError(f"two columns map to the same symbol within a block: {dups.to_dict('records')}")
    by_symbol: dict[str, list[tuple[str, str]]] = {}
    for c, b, s in feat[["column", "block", "symbol"]].itertuples(index=False):
        by_symbol.setdefault(s, []).append((c, b))

    member_rows, summary_rows, alias_rows, gmts = [], [], [], {}
    for src, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"gene-set file named in config is missing: {path}")
        gmts[src] = read_gmt(path)
        for set_name, genes in gmts[src].items():
            n_rna = n_dna = 0
            for g in genes:
                g2 = aliases.get(g, g)
                if g2 != g:
                    alias_rows.append({"source": src, "set": set_name, "set_symbol": g, "mapped_to": g2})
                for col, block in by_symbol.get(g2, []):
                    member_rows.append({"set": set_name, "source": src, "source_file": path.name,
                                        "set_size": len(genes), "column": col, "block": block, "symbol": g2})
                    n_rna += block == "rna"
                    n_dna += block == "dna"
            summary_rows.append({"set": set_name, "source": src, "source_file": path.name, "set_size": len(genes),
                                 "n_rna_matched": n_rna, "n_dna_matched": n_dna, "min_rna_overlap": min_rna,
                                 "passes_min_overlap": n_rna >= min_rna})
    membership = pd.DataFrame(member_rows, columns=["set", "source", "source_file", "set_size", "column", "block", "symbol"])
    summary = pd.DataFrame(summary_rows)

    xw = feat[["column", "block", "symbol", "alias_applied"]].copy()
    for src in files:
        counts = membership[membership.source == src].groupby(["column", "block"]).size()
        xw[f"n_sets_{src}"] = [int(counts.get((c, b), 0)) for c, b in zip(xw.column, xw.block)]
    n_cols = [f"n_sets_{s}" for s in files]
    xw["matched"] = xw[n_cols].sum(axis=1) > 0
    xw["reason"] = np.where(xw["matched"], "", "symbol not found in any configured gene-set file")
    xw = xw[["column", "block", "symbol", "matched", "reason", *n_cols, "alias_applied"]]
    unmatched = xw.loc[~xw.matched, ["column", "block", "symbol", "reason"]].reset_index(drop=True)

    cov = []
    for src in files:
        s = summary[summary.source == src]
        cov.append({"source": src, "file": files[src].name, "n_sets": len(s),
                    "n_unique_genes_in_file": len({aliases.get(g, g) for gs in gmts[src].values() for g in gs}),
                    "rna_covered": int((xw.block.eq("rna") & xw[f"n_sets_{src}"].gt(0)).sum()),
                    "dna_covered": int((xw.block.eq("dna") & xw[f"n_sets_{src}"].gt(0)).sum()),
                    "sets_pass": int(s.passes_min_overlap.sum()),
                    "sets_ge10": int((s.n_rna_matched >= 10).sum()),
                    "median_rna_overlap": float(s.n_rna_matched.median()),
                    "min_rna_overlap": min_rna})
    cov.append({"source": "any", "file": "", "n_sets": len(summary), "n_unique_genes_in_file": np.nan,
                "rna_covered": int((xw.block.eq("rna") & xw.matched).sum()),
                "dna_covered": int((xw.block.eq("dna") & xw.matched).sum()),
                "sets_pass": int(summary.passes_min_overlap.sum()),
                "sets_ge10": int((summary.n_rna_matched >= 10).sum()),
                "median_rna_overlap": float(summary.n_rna_matched.median()), "min_rna_overlap": min_rna})
    alias_log = pd.DataFrame(alias_rows, columns=["source", "set", "set_symbol", "mapped_to"])
    return CrosswalkResult(xw, membership, summary, unmatched, pd.DataFrame(cov), alias_log, list(files.values()))


def check_coverage(cov: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for src, exp in cfg["gene_sets"]["expected_coverage"].items():
        got = cov.set_index("source").loc[src]
        for k, v in exp.items():
            rows.append({"source": src, "quantity": k, "expected": v, "observed": got[k],
                         "match": bool(np.isclose(float(got[k]), float(v)))})
    return pd.DataFrame(rows)


def write_crosswalk(res: CrosswalkResult, cfg: Config) -> list[Path]:
    out = cfg.path("crosswalk")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in zip(OUTPUT_FILES, (res.crosswalk, res.membership, res.summary, res.unmatched)):
        p = out / name
        df.to_csv(p, index=False, lineterminator="\n")
        paths.append(p)
    return paths


class GeneSetIndex:
    """Lookup helpers over the written crosswalk tables. Every answer names its source."""

    def __init__(self, cfg: Config):
        d = cfg.path("crosswalk")
        missing = [f for f in OUTPUT_FILES if not (d / f).is_file()]
        if missing:
            raise FileNotFoundError(f"crosswalk not built (missing {missing}); run `python run.py crosswalk`")
        self.crosswalk = pd.read_csv(d / "gene_crosswalk.csv")
        self.membership = pd.read_csv(d / "gene_set_membership.csv")
        self.summary = pd.read_csv(d / "gene_set_summary.csv")

    def genes_of_set(self, set_name: str, source: str | None = None, block: str | None = "rna",
                     usable_only: bool = False) -> pd.DataFrame:
        """Dataset columns in a set -> DataFrame(set, source, column, block, symbol)."""
        s = self.summary[self.summary.set == set_name]
        if source:
            s = s[s.source == source]
        if s.empty:
            raise KeyError(f"unknown set {set_name!r} (source={source})")
        if len(s) > 1:
            raise KeyError(f"set {set_name!r} exists in several sources {list(s.source)}; pass source=")
        if usable_only and not bool(s.passes_min_overlap.iloc[0]):
            raise ValueError(f"{set_name} has {int(s.n_rna_matched.iloc[0])} matched RNA genes, "
                             f"below the minimum of {int(s.min_rna_overlap.iloc[0])}; lookup only")
        m = self.membership[(self.membership.set == set_name) & (self.membership.source == s.source.iloc[0])]
        if block:
            m = m[m.block == block]
        return m[["set", "source", "column", "block", "symbol"]].reset_index(drop=True)

    def sets_of(self, gene_or_column: str, source: str | None = None) -> pd.DataFrame:
        """Sets containing a gene symbol (any case) or dataset column -> DataFrame(set, source, column, block)."""
        key = gene_or_column
        m = self.membership
        hit = m[(m.column == key) | (m.symbol == canonical_symbol(key))]
        if source:
            hit = hit[hit.source == source]
        return hit[["set", "source", "column", "block", "set_size"]].drop_duplicates().reset_index(drop=True)

    def usable_sets(self, source: str | None = None) -> pd.DataFrame:
        s = self.summary[self.summary.passes_min_overlap]
        return s[s.source == source] if source else s
