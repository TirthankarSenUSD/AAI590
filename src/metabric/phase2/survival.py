"""Survival data for Phase 2: overall survival time and event, the zero-time rule, NPI bands.

Death = event (1 - overall_survival, the Phase 1 target); time = `overall_survival_months`.
Nothing here is fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import Config
from ..data import Dataset, npi_standard


def survival_data(cfg: Config, ds: Dataset) -> tuple[pd.DataFrame, dict]:
    """(frame with `time`, `event` indexed by patient_id, record of the zero-time rule)."""
    pc = cfg["phase2"]
    t = ds.frame[pc["time_column"]].astype(float).copy()
    e = ds.y.astype(int)
    if t.isna().any() or (t < 0).any():
        raise ValueError("survival times must be present and non-negative")
    zero = t.index[t <= 0]
    rule = {"rule": pc["zero_time_rule"], "patients": [int(i) for i in zero], "events_at_zero": int(e.loc[zero].sum())}
    if len(zero):
        if pc["zero_time_rule"] != "half_min_positive":
            raise ValueError(f"unknown zero_time_rule {pc['zero_time_rule']!r}")
        new = float(t[t > 0].min()) / 2
        t.loc[zero] = new
        rule["new_time"] = new
    return pd.DataFrame({"time": t, "event": e}), rule


def structured(time, event) -> np.ndarray:
    """scikit-survival target: a record array (event: bool, time: float)."""
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, dtype=float))


def npi_band(frame: pd.DataFrame, cuts: list[float]) -> pd.Series:
    """Standard NPI bands (good / moderate / poor); missing when the standard NPI is missing."""
    v = npi_standard(frame)
    lab = pd.Series(np.select([v <= cuts[0], v <= cuts[1]], ["good", "moderate"], "poor"), index=frame.index, dtype=object)
    lab[v.isna()] = np.nan
    return lab.rename("npi_band")
