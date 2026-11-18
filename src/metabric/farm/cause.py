"""Cause-specific members: trained on alive / cancer death / other-cause death, scored as death.

The 3-class label comes from `death_from_cancer` and is used **only as a training label inside the
folds**: the wrapped pipeline never sees it as an input, and every metric is computed on binary
death exactly as for the other members. P(death) = P(cancer death) + P(other-cause death).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

CAUSE_CODES = {"Living": 0, "Died of Disease": 1, "Died of Other Causes": 2}
CAUSE_NAMES = {0: "alive", 1: "cancer death", 2: "other-cause death"}


def cause_labels(frame: pd.DataFrame) -> pd.Series:
    """0 alive / 1 cancer death / 2 other-cause death. Rows without a recorded cause are rejected."""
    y = frame["death_from_cancer"].map(CAUSE_CODES)
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} rows have no recorded cause of death")
    return y.astype(int).rename("cause")


class CauseSpecific(BaseEstimator, ClassifierMixin):
    """Wraps a multi-class pipeline; `predict_proba` returns [P(alive), P(death)]."""

    def __init__(self, pipe=None):
        self.pipe = pipe

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        if set(np.unique(y)) - {0, 1, 2} or y.max() < 2:
            raise ValueError("CauseSpecific needs labels 0 (alive), 1 (cancer death), 2 (other-cause death)")
        self.pipe.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_cause_proba(self, X) -> np.ndarray:
        """Columns: alive, cancer death, other-cause death."""
        p = self.pipe.predict_proba(X)
        cls = list(self.pipe.classes_)
        return p[:, [cls.index(0), cls.index(1), cls.index(2)]]

    def predict_proba(self, X) -> np.ndarray:
        p = self.predict_cause_proba(X)
        death = p[:, 1] + p[:, 2]
        return np.column_stack([1 - death, death])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getitem__(self, i):                 # pipe-like access for the explanation helpers
        return self.pipe[i]
