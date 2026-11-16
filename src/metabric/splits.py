"""Frozen holdout. Created once, never regenerated, verified by hash on every use.

Only the holdout is made here (build step 1). Outer and inner folds belong to the harness
(build step 2) and will use the same `strata_labels` rule.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config, write_holdout_hash
from .data import Dataset, sha256_file


class HoldoutError(RuntimeError):
    pass


def strata_labels(ds: Dataset, min_count: int) -> tuple[pd.Series, pd.DataFrame]:
    """outcome x cohort x chemotherapy. Strata below `min_count` are merged across cohorts within
    the same outcome x chemotherapy cell (and the merge is returned for logging)."""
    f = ds.frame
    lab = (ds.y.astype(str) + "|c" + f["cohort"].astype(str) + "|chemo" + f["chemotherapy"].astype(str))
    lab.name = "stratum"
    counts = lab.value_counts()
    small = counts[counts < min_count]
    merges = []
    if len(small):
        for s in small.index:
            y, _, chemo = s.split("|")
            merges.append({"stratum": s, "n": int(counts[s]), "merged_into": f"{y}|merged|{chemo}"})
            lab = lab.where(lab != s, f"{y}|merged|{chemo}")
    return lab, pd.DataFrame(merges, columns=["stratum", "n", "merged_into"])


def make_holdout(cfg: Config, ds: Dataset) -> tuple[Config, pd.DataFrame, pd.DataFrame]:
    path = cfg.path("holdout")
    if path.exists() or cfg["holdout"].get("sha256"):
        raise HoldoutError(f"holdout already exists ({path.name} / hash in config); it is frozen and is never recreated")
    labels, merges = strata_labels(ds, int(cfg["holdout"]["min_stratum_count"]))
    train_ids, hold_ids = train_test_split(ds.ids.to_numpy(), test_size=float(cfg["holdout"]["fraction"]),
                                           stratify=labels.to_numpy(), random_state=cfg.seed, shuffle=True)
    hold = set(hold_ids)
    out = pd.DataFrame({"patient_id": ds.ids, "set": ["holdout" if i in hold else "train" for i in ds.ids],
                        "stratum": labels.to_numpy()})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")
    cfg = write_holdout_hash(cfg, sha256_file(path))
    return cfg, out, merges


def load_holdout(cfg: Config, ds: Dataset | None = None) -> pd.DataFrame:
    path: Path = cfg.path("holdout")
    expected = cfg["holdout"].get("sha256")
    if not path.is_file() or not expected:
        raise HoldoutError("no frozen holdout: run `python run.py holdout` once")
    got = sha256_file(path)
    if got != expected:
        raise HoldoutError(f"holdout file hash {got[:12]} differs from config {expected[:12]}: file was changed")
    h = pd.read_csv(path)
    if ds is not None and set(h.patient_id) != set(ds.ids):
        raise HoldoutError("holdout patient_ids do not match the dataset")
    if set(h["set"]) != {"train", "holdout"}:
        raise HoldoutError("holdout file has unexpected set labels")
    return h


def training_ids(cfg: Config, ds: Dataset) -> pd.Index:
    h = load_holdout(cfg, ds)
    return pd.Index(h.loc[h["set"] == "train", "patient_id"])
