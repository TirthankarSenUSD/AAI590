"""Evaluation harness (build step 2): nested CV scoring, selection rule, floors, checks."""
from .core import (Candidate, Harness, Result, Split, check_split_hygiene, procedure, procedure_preds, reliability,  # noqa: F401
                   select_candidate, splits_from_table)
