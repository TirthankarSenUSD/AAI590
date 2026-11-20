"""Resumable runs: a crashed run, rerun with the same command, skips finished tasks and gives the
same results as an uninterrupted run."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from metabric.core import checkpoint
from metabric.harness.core import Candidate, Harness, Split

FITS = {"n": 0}
CRASH = {"on": None}


class Counting(LogisticRegression):
    def fit(self, X, y, *a, **k):
        if CRASH["on"] is not None and self.C == CRASH["on"]:
            raise RuntimeError("simulated crash")
        FITS["n"] += 1
        return super().fit(X, y, *a, **k)


def _setup(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=pd.RangeIndex(1000, 1000 + n))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-X.a.to_numpy()))).astype(int), index=X.index)
    splits = []
    for k, (tr, te) in enumerate(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y)):
        tr_i, te_i = X.index[tr], X.index[te]
        inner = [(tr_i[a], tr_i[b]) for a, b in StratifiedKFold(2, shuffle=True, random_state=1).split(tr_i, y.loc[tr_i])]
        splits.append(Split(f"r0o{k}", 0, k, tr_i, te_i, inner))
    cands = [Candidate(f"lr{c}", (lambda c=c: Counting(C=c)), list("abc"), complexity=i) for i, c in enumerate([0.01, 0.1, 1.0, 10.0])]
    return X, y, splits, cands


def test_loop_roundtrip_and_namespaces(tmp_path):
    checkpoint.activate(tmp_path, "cfg", "farm --arms C")
    a, b = checkpoint.Loop("x"), checkpoint.Loop("x")              # second loop with the same label: own namespace
    a.put(("k", 1), {"v": 1})
    assert a.get(("k", 1)) == (True, {"v": 1}) and b.get(("k", 1)) == (False, None)
    a.put("bad", lambda z: z)                                         # unpicklable: skipped, no crash
    assert a.get("bad") == (False, None)
    checkpoint.deactivate(remove=True)
    assert not any((tmp_path / "checkpoints").iterdir())
    assert checkpoint.Loop("x").get(("k", 1)) == (False, None)        # inactive: no-op


def test_crash_then_resume_reuses_finished_tasks_and_matches(tmp_path, cfg):
    X, y, splits, cands = _setup()
    ref = Harness(cfg, X, y, splits).evaluate(cands, select=True)    # uninterrupted, no checkpoints
    # run 1 crashes on the third candidate (candidates are scored in order with n_jobs=1)
    checkpoint.activate(tmp_path, "cfg", "farm --arms C")
    CRASH["on"] = 1.0
    with pytest.raises(RuntimeError):
        Harness(cfg, X, y, splits).evaluate(cands, select=True)
    checkpoint.deactivate(remove=False)
    # run 2: same command, same config and code -> the first two candidates come from checkpoints
    CRASH["on"] = None
    checkpoint.activate(tmp_path, "cfg", "farm --arms C")
    FITS["n"] = 0
    res = Harness(cfg, X, y, splits).evaluate(cands, select=True)
    per_cand = len(splits) * (len(splits[0].inner) + 1)               # inner fits + outer fit, per candidate
    assert FITS["n"] == 2 * per_cand                                   # only the two unfinished candidates refitted
    assert checkpoint.stats()["reused"] == 2
    pd.testing.assert_frame_equal(res.folds, ref.folds)
    pd.testing.assert_frame_equal(res.preds, ref.preds)
    pd.testing.assert_frame_equal(res.selection, ref.selection)
    checkpoint.deactivate(remove=True)
    # a different command (or config / code) never reuses them
    checkpoint.activate(tmp_path, "cfg", "farm --arms C")
    checkpoint.deactivate(remove=True)
    checkpoint.activate(tmp_path, "cfg", "farm --arms C+RNA")
    FITS["n"] = 0
    Harness(cfg, X, y, splits).evaluate(cands, select=True)
    assert FITS["n"] == 4 * per_cand
    checkpoint.deactivate(remove=True)


def test_run_keeps_checkpoints_on_failure_and_removes_them_on_success(tmp_project):
    from metabric.core.artifacts import Run
    r = Run(tmp_project, "farm --arms C")
    d = r.checkpoint_dir
    checkpoint.Loop("x").put("k", 1)
    r.fail(RuntimeError("boom"))
    assert d.is_dir() and any(d.rglob("*.pkl"))                       # kept for the resume
    r2 = Run(tmp_project, "farm --arms C")
    assert r2.checkpoint_dir == d and checkpoint.Loop("x").get("k") == (True, 1)
    r2.finalize()
    assert not d.exists()
