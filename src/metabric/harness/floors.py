"""Design matrix and the floor models (always reported, never eligible to win)."""
from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data import Dataset, npi_standard
from .core import Candidate
from .metrics import BIG_C


def design_matrix(ds: Dataset) -> pd.DataFrame:
    """Raw inputs a candidate may draw columns from: clinical (as in the file, cleaned), RNA,
    binary mutation flags, plus deterministic derived columns. Nothing here is fitted."""
    f = ds.frame
    X = pd.concat([f[ds.blocks["clinical"]], f[ds.blocks["rna"]], ds.mutations, npi_standard(f)], axis=1)
    assert not set(ds.blocks["outcome"]) & set(X.columns)
    return X


def _lr():
    return LogisticRegression(C=BIG_C, max_iter=1000)


def floor_candidates(cfg) -> list[Candidate]:
    fl = cfg["floors"]
    return [
        Candidate("majority", lambda: DummyClassifier(strategy="prior"), ["age_at_diagnosis"], 0, False,
                  "Majority class (training-fold death rate for everyone)"),
        Candidate("age_only", lambda: make_pipeline(StandardScaler(), _lr()), fl["age_only"]["features"], 1, False,
                  "Age-only logistic regression"),
        Candidate("npi_file", lambda: make_pipeline(StandardScaler(), _lr()), fl["npi_file"]["features"], 1, False,
                  fl["npi_file"]["label"] + " logistic regression"),
        Candidate("npi_standard", lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                        StandardScaler(), _lr()),
                  fl["npi_standard"]["features"], 2, False,
                  fl["npi_standard"]["label"] + " logistic regression (in-fold median fill + missing flag)"),
    ]
