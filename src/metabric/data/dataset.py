"""Data layer: read, validate (fail loudly), clean, build the target and the feature blocks.

Nothing here is fitted to the data. Missing indicators take an explicit column list; which
columns get indicators must be decided on training rows only (A12, A13).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.config import Config

ER_IHC = "er_status_measured_by_ihc"
TYPO_FIXES = {ER_IHC: {"Positve": "Positive"}}


class DataValidationError(ValueError):
    pass


def read_raw(cfg: Config) -> pd.DataFrame:
    dna = cfg.blocks["dna"]
    # mutation columns are strings by definition; read them as such
    return pd.read_csv(cfg.path("data"), low_memory=False, dtype={c: str for c in dna})


def validate_raw(df: pd.DataFrame, cfg: Config) -> dict:
    """Check the file against the explicit block lists. Raise on any difference."""
    b = cfg.blocks
    problems: list[str] = []
    expected = b["id"] + b["clinical"] + b["outcome"] + b["rna"] + b["dna"]
    cols = list(df.columns)
    if len(cols) != len(expected) or set(cols) != set(expected):
        problems.append(f"columns: file has {len(cols)}, config lists {len(expected)}; "
                        f"missing={sorted(set(expected) - set(cols))[:10]} extra={sorted(set(cols) - set(expected))[:10]}")
    else:
        n_head = 1 + len(b["clinical"]) + len(b["outcome"])
        if cols[0] != b["id"][0]:
            problems.append("patient_id is not the first column")
        if set(cols[1:n_head]) != set(b["clinical"] + b["outcome"]):
            problems.append("clinical/outcome columns are not in positions 1..30")
        if cols[n_head - 1] != "death_from_cancer":
            problems.append("death_from_cancer is not the last column before the RNA block")
        if cols[n_head:n_head + len(b["rna"])] != b["rna"]:
            problems.append("RNA block does not start right after death_from_cancer in config order")
        if cols[-len(b["dna"]):] != b["dna"]:
            problems.append("mutation block is not the final columns in config order")
    for name, size in [("clinical", 27), ("outcome", 3), ("rna", 489), ("dna", 173)]:
        if len(b[name]) != size:
            problems.append(f"block {name}: {len(b[name])} columns, expected {size}")
    if not all(c.endswith("_mut") for c in b["dna"]):
        problems.append("a DNA column does not end in _mut")
    if df.shape[0] != 1904:
        problems.append(f"rows: {df.shape[0]}, expected 1904")
    if not df[b["id"][0]].is_unique:
        problems.append("patient_id not unique")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate rows")
    if not problems:
        rna = df[b["rna"]]
        if not all(pd.api.types.is_float_dtype(t) for t in rna.dtypes):
            problems.append("RNA block contains non-float columns")
        if rna.isna().any().any():
            problems.append("RNA block has missing values")
        # z-score heuristic: consistency check only (A10)
        m, s = rna.mean().abs().max(), rna.std()
        if m > 0.05 or (s - 1).abs().max() > 0.05:
            problems.append(f"RNA block fails the z-score check (max |mean| {m:.3f}, SD range {s.min():.3f}-{s.max():.3f})")
        dna = df[b["dna"]]
        if dna.isna().any().any():
            problems.append("mutation block has missing values (binarisation rule would mark them mutated)")
        numeric_like = [c for c in b["dna"] if pd.to_numeric(dna[c], errors="coerce").notna().all()
                        and (dna[c].astype(str) != "0").any()]
        if numeric_like:
            problems.append(f"mutation columns that look numeric: {numeric_like[:5]}")
        y = df[cfg["target"]["source_column"]]
        if set(y.unique()) != {0, 1}:
            problems.append("overall_survival is not 0/1")
    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))
    return {"rows": df.shape[0], "columns": df.shape[1], **{k: len(v) for k, v in b.items()}}


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in TYPO_FIXES.items():
        out[col] = out[col].replace(mapping)
    # cohort is stored as float 1.0-5.0 but is categorical
    out["cohort"] = pd.Categorical(out["cohort"].astype("Int64"), categories=[1, 2, 3, 4, 5])
    return out


def make_target(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Positive class is death: death = 1 - overall_survival."""
    src = df[cfg["target"]["source_column"]]
    y = (1 - src).astype("int8")
    y.name = cfg["target"]["name"]
    return y


def binarise_mutations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """'0' means not mutated; any other string is a variant."""
    if df[cols].isna().any().any():
        raise DataValidationError("missing values in mutation columns")
    return (df[cols].astype(str) != "0").astype("int8")


def missing_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One `<col>__missing` 0/1 column per listed column. Choose `cols` on training rows."""
    return pd.DataFrame({f"{c}__missing": df[c].isna().astype("int8") for c in cols}, index=df.index)


def npi_standard(frame: pd.DataFrame) -> pd.Series:
    """Standard Nottingham Prognostic Index: 0.2 x size(cm) + node stage + grade.

    Node stage: 0 positive nodes -> 1, 1-3 -> 2, 4 or more -> 3. `tumor_size` is in mm.
    Missing if any input is missing. Deterministic (nothing fitted). The file's own
    `nottingham_prognostic_index` uses a 0.02 size weight instead (FINDINGS F-021); both are
    kept: the file's NPI is the primary floor, this one the secondary floor (F-028).
    """
    nodes = frame["lymph_nodes_examined_positive"]
    node_stage = pd.Series(np.select([nodes == 0, nodes <= 3], [1.0, 2.0], 3.0), index=frame.index)
    node_stage[nodes.isna()] = np.nan
    out = 0.2 * frame["tumor_size"] / 10 + node_stage + frame["neoplasm_histologic_grade"]
    return out.rename("npi_standard")


@dataclass
class Dataset:
    frame: pd.DataFrame          # cleaned file, index = patient_id (ID dropped from columns)
    y: pd.Series                 # death (1) / alive (0)
    mutations: pd.DataFrame      # binary flags for the 173 mutation columns
    blocks: dict[str, list[str]]

    @property
    def ids(self) -> pd.Index:
        return self.frame.index

    def features(self, blocks=("clinical", "rna", "dna")) -> pd.DataFrame:
        """Feature matrix. Never contains outcome columns or the ID; DNA comes binarised."""
        parts = []
        for b in blocks:
            if b == "dna":
                parts.append(self.mutations)
            elif b in ("clinical", "rna"):
                parts.append(self.frame[self.blocks[b]])
            else:
                raise ValueError(f"{b!r} is not a feature block")
        return pd.concat(parts, axis=1)

    def subset(self, ids) -> "Dataset":
        ids = pd.Index(ids)
        return Dataset(self.frame.loc[ids], self.y.loc[ids], self.mutations.loc[ids], self.blocks)


def load_dataset(cfg: Config) -> Dataset:
    raw = read_raw(cfg)
    validate_raw(raw, cfg)
    df = clean(raw, cfg)
    id_col = cfg.blocks["id"][0]
    df = df.set_index(id_col, drop=True)
    y = make_target(df, cfg)
    muts = binarise_mutations(df, cfg.blocks["dna"])
    return Dataset(frame=df, y=y, mutations=muts, blocks=cfg.blocks)




__all__ = ["DataValidationError", "Dataset", "load_dataset", "read_raw", "validate_raw", "clean",
           "make_target", "binarise_mutations", "missing_indicators", "npi_standard"]
