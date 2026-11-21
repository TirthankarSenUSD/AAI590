"""Artifact writer and manifest (PLAN.md section 18).

Layout: outputs/<run_id>/<stage>/{figures,tables,logs}/ plus config.yaml and manifest.json
in the run folder. A run is complete only when its manifest validates: every listed file
exists and no file in the run folder is unlisted. outputs/latest.json points at the newest
completed run (a pointer file, not a symlink: see FINDINGS).
"""
from __future__ import annotations

import inspect
import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from .config import Config
from .paths import sha256_file

STAGES = ("crosswalk", "eda", "splits", "harness", "farm", "importance", "ensemble",
          "agreement", "report", "holdout", "phase2", "models")
SUBDIRS = ("figures", "tables", "logs")
RUN_FILES = ("manifest.json", "config.yaml")
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "PyYAML")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _caller(depth: int = 2) -> str:
    frame = inspect.stack()[depth]
    mod = inspect.getmodule(frame.frame)
    return f"{mod.__name__ if mod else '?'}.{frame.function}"


def _pkg_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


class Run:
    def __init__(self, cfg: Config, command: str, run_id: str | None = None):
        self.cfg = cfg
        self.outputs = cfg.path("outputs")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = run_id or f"{stamp}_{cfg.hash[:8]}"
        rid, k = base, 1
        while (self.outputs / rid).exists():          # never overwrite an earlier run
            k += 1
            rid = f"{base}-{k}"
        self.run_id = rid
        self.dir = self.outputs / rid
        self.dir.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(cfg.file, self.dir / "config.yaml")
        self.manifest = {
            "run_id": rid, "command": command, "config_hash": cfg.hash, "seed": cfg.seed,
            "created": _now(), "completed": None, "status": "running", "platform": sys.platform,
            "packages": _pkg_versions(), "inputs": [], "external_artifacts": [], "artifacts": [],
        }
        self._stages: dict[str, Stage] = {}
        # resumable: rerunning the same command with the same config and code reuses finished tasks
        from . import checkpoint
        self.checkpoint_dir = checkpoint.activate(self.outputs, cfg.hash, command)
        self.manifest["checkpoints"] = self._rel(self.checkpoint_dir)
        self._write()

    # ------------------------------------------------------------------
    def stage(self, name: str) -> "Stage":
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}; allowed: {STAGES}")
        if name not in self._stages:
            self._stages[name] = Stage(self, name)
        return self._stages[name]

    def _rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.cfg.root).as_posix()

    def record_input(self, path: Path, role: str) -> dict:
        path = Path(path)
        entry = {"path": self._rel(path), "role": role, "bytes": path.stat().st_size,
                 "sha256": sha256_file(path)}
        self.manifest["inputs"].append(entry)
        self._write()
        return entry

    def record_external(self, path: Path, role: str, creator: str | None = None) -> dict:
        """A generated file that lives outside the run folder (e.g. reference/crosswalk/*)."""
        path = Path(path)
        entry = {"path": self._rel(path), "role": role, "bytes": path.stat().st_size,
                 "sha256": sha256_file(path), "creator": creator or _caller(), "created": _now()}
        self.manifest["external_artifacts"].append(entry)
        self._write()
        return entry

    def _register(self, path: Path, stage: str, kind: str, caption: str | None, creator: str,
                  data_path: Path | None = None) -> dict:
        entry = {"path": self._rel(path), "stage": stage, "kind": kind, "run_id": self.run_id,
                 "config_hash": self.cfg.hash, "seed": self.cfg.seed, "creator": creator,
                 "created": _now(), "caption": caption}
        if data_path is not None:
            entry["data"] = self._rel(data_path)
        self.manifest["artifacts"].append(entry)
        self._write()
        return entry

    def _write(self) -> None:
        """Atomic replace; on Windows the target can be briefly locked (indexer, editor), so retry."""
        import time
        text = json.dumps(self.manifest, indent=2)
        tmp, target = self.dir / "manifest.json.tmp", self.dir / "manifest.json"
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(20):
            try:
                tmp.replace(target)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        target.write_text(text, encoding="utf-8")   # last resort: direct write
        tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        problems = []
        listed = {a["path"] for a in self.manifest["artifacts"]}
        for p in listed:
            if not (self.cfg.root / p).is_file():
                problems.append(f"listed but missing: {p}")
        for e in self.manifest["external_artifacts"]:
            p = self.cfg.root / e["path"]
            if not p.is_file() or sha256_file(p) != e["sha256"]:
                problems.append(f"external artifact missing or changed: {e['path']}")
        allowed = listed | {self._rel(self.dir / f) for f in RUN_FILES}
        for f in self.dir.rglob("*"):
            if f.is_file() and self._rel(f) not in allowed:
                problems.append(f"unlisted file: {self._rel(f)}")
        return problems

    def finalize(self) -> None:
        for st in self._stages.values():
            st._close_log()
        problems = self.validate()
        if problems:
            self.manifest["status"] = "invalid"
            self.manifest["problems"] = problems
            self._write()
            raise RuntimeError("manifest validation failed:\n" + "\n".join(problems))
        from . import checkpoint
        self.manifest["checkpoint_tasks"] = {k: v for k, v in checkpoint.stats().items() if k != "dir"}
        checkpoint.deactivate(remove=True)                # finished: checkpoints are no longer needed
        self.manifest.pop("checkpoints", None)
        self.manifest["status"] = "complete"
        self.manifest["completed"] = _now()
        self._write()
        pointer = {"run_id": self.run_id, "path": self._rel(self.dir), "command": self.manifest["command"],
                   "completed": self.manifest["completed"]}
        (self.outputs / "latest.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    def fail(self, exc: BaseException) -> None:
        for st in self._stages.values():
            st._close_log()
        from . import checkpoint
        self.manifest["checkpoint_tasks"] = {k: v for k, v in checkpoint.stats().items() if k != "dir"}
        checkpoint.deactivate(remove=False)               # kept: rerun the same command to resume
        self.manifest["status"] = "failed"
        self.manifest["error"] = f"{type(exc).__name__}: {exc}"
        self._write()


class Stage:
    def __init__(self, run: Run, name: str):
        self.run, self.name = run, name
        self.dir = run.dir / name
        for s in SUBDIRS:
            (self.dir / s).mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "logs" / f"{name}.log"
        self.log = logging.getLogger(f"metabric.{run.run_id}.{name}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        self._handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.log.addHandler(self._handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(f"[{name}] %(message)s"))
        self.log.addHandler(console)
        self._log_registered = False

    def table(self, name: str, df: pd.DataFrame, caption: str | None = None, index: bool = False,
              creator: str | None = None) -> Path:
        path = self.dir / "tables" / f"{name}.csv"
        df.to_csv(path, index=index, lineterminator="\n")
        self.run._register(path, self.name, "table", caption, creator or _caller())
        return path

    def figure(self, name: str, fig, data: pd.DataFrame, caption: str, creator: str | None = None) -> Path:
        """Save a PNG plus the CSV of exactly the plotted numbers; caption goes in the manifest."""
        import matplotlib.pyplot as plt
        png = self.dir / "figures" / f"{name}.png"
        csv = self.dir / "figures" / f"{name}.csv"
        fig.savefig(png, dpi=self.run.cfg["eda"]["figure_dpi"], bbox_inches="tight",
                    facecolor=fig.get_facecolor(), metadata={"Software": None})
        plt.close(fig)
        data.to_csv(csv, index=False, lineterminator="\n")
        who = creator or _caller()
        self.run._register(png, self.name, "figure", caption, who, data_path=csv)
        self.run._register(csv, self.name, "figure_data", f"Data for {name}.png", who)
        return png

    def text(self, name: str, content: str, caption: str | None = None, creator: str | None = None) -> Path:
        path = self.dir / "tables" / name
        path.write_text(content, encoding="utf-8", newline="\n")
        self.run._register(path, self.name, "text", caption, creator or _caller())
        return path

    def _close_log(self) -> None:
        for h in list(self.log.handlers):
            h.close()
            self.log.removeHandler(h)
        if not self._log_registered and self.log_path.exists():
            self.run._register(self.log_path, self.name, "log", f"{self.name} stage log", "metabric.core.artifacts")
            self._log_registered = True
