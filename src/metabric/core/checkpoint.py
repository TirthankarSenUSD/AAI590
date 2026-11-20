"""Resumable runs: every finished unit of parallel work is saved as it completes.

A `Run` activates a checkpoint folder, `outputs/checkpoints/<key>/`, where the key is a hash of
- the config hash,
- the source code of the package,
- the run's command line (stage and arguments, e.g. `farm --arms C`).

The parallel loops (`Harness.evaluate`, `importance.stage._parallel`) look up each task there
before running it and save its result when it finishes. If a run crashes or the machine restarts,
rerunning the **same command with the same config and code** skips every finished task. All fits
are seeded, so the results are identical to an uninterrupted run. Any change to the config, the
code or the arguments gives a new key, so stale results are never reused.

The folder is deleted when the run completes. Writes are atomic (temporary file, then rename), so
a crash during a write leaves no half-written checkpoint.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import shutil
from pathlib import Path

_ACTIVE: dict = {"dir": None, "calls": {}, "reused": 0, "saved": 0}


def code_hash(pkg_dir: Path | None = None) -> str:
    pkg_dir = pkg_dir or Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(pkg_dir.rglob("*.py")):
        h.update(p.relative_to(pkg_dir).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_key(config_hash: str, command: str) -> str:
    return hashlib.sha256(f"{config_hash}|{code_hash()}|{command}".encode()).hexdigest()[:20]


def activate(outputs: Path, config_hash: str, command: str) -> Path:
    d = Path(outputs) / "checkpoints" / run_key(config_hash, command)
    d.mkdir(parents=True, exist_ok=True)
    _ACTIVE.update(dir=d, calls={}, reused=0, saved=0)
    return d


def deactivate(remove: bool) -> None:
    d = _ACTIVE["dir"]
    _ACTIVE.update(dir=None, calls={})
    if remove and d is not None and d.exists():
        shutil.rmtree(d, ignore_errors=True)


def active() -> bool:
    return _ACTIVE["dir"] is not None


def stats() -> dict:
    return {"dir": str(_ACTIVE["dir"]) if _ACTIVE["dir"] else None, "reused": _ACTIVE["reused"], "saved": _ACTIVE["saved"]}


class Loop:
    """Checkpoints of one parallel loop. `label` names the loop; the n-th loop with the same label
    in a run gets its own namespace, so repeated loops never collide."""

    def __init__(self, label: str):
        self.dir = None
        if _ACTIVE["dir"] is None:
            return
        n = _ACTIVE["calls"].get(label, 0)
        _ACTIVE["calls"][label] = n + 1
        ns = hashlib.sha256(f"{label}|{n}".encode()).hexdigest()[:12]
        self.dir = _ACTIVE["dir"] / ns
        self.dir.mkdir(exist_ok=True)

    def _path(self, key) -> Path:
        return self.dir / (hashlib.sha256(repr(key).encode()).hexdigest()[:24] + ".pkl")

    def get(self, key):
        """(found, value)."""
        if self.dir is None:
            return False, None
        p = self._path(key)
        if not p.is_file():
            return False, None
        try:
            with open(p, "rb") as f:
                v = pickle.load(f)
        except Exception:                       # unreadable: recompute
            return False, None
        _ACTIVE["reused"] += 1
        return True, v

    def put(self, key, value) -> None:
        if self.dir is None:
            return
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:                       # not picklable: this task is simply not checkpointed
            tmp.unlink(missing_ok=True)
            return
        os.replace(tmp, p)
        _ACTIVE["saved"] += 1
