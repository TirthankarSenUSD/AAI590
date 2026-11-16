"""Load config.yaml, hash it, resolve paths from the project root."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .paths import find_root


@dataclass(frozen=True)
class Config:
    root: Path
    raw: dict = field(repr=False)

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def hash(self) -> str:
        """Hash of the parsed config (insensitive to comments and whitespace)."""
        canon = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(canon).hexdigest()[:10]

    def path(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]

    @property
    def file(self) -> Path:
        return self.root / "config.yaml"

    def __getitem__(self, key):
        return self.raw[key]

    # column blocks -------------------------------------------------------
    @property
    def blocks(self) -> dict[str, list[str]]:
        c = self.raw["columns"]
        return {"id": [c["id"]], "clinical": list(c["clinical"]), "outcome": list(c["outcome"]),
                "rna": list(c["rna"]), "dna": list(c["dna"])}


def load_config(root: str | Path | None = None) -> Config:
    root = Path(root).resolve() if root else find_root()
    with open(root / "config.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(root=root, raw=raw)


def write_frozen_hash(cfg: Config, section: str, field_name: str, sha: str) -> Config:
    """Record a frozen file's hash in config.yaml once (a targeted edit that keeps comments)."""
    if cfg.raw[section].get(field_name):
        raise RuntimeError(f"config.yaml already holds {section}.{field_name}; the file is frozen")
    text = cfg.file.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(\s+{re.escape(field_name)}:\s*)null(.*)$", re.M)
    new, n = pattern.subn(lambda m: f'{m.group(1)}"{sha}"{m.group(2)}', text, count=1)
    if n != 1:
        raise RuntimeError(f"could not find '{field_name}: null' in config.yaml")
    cfg.file.write_text(new, encoding="utf-8", newline="\n")
    out = load_config(cfg.root)
    if out.raw[section].get(field_name) != sha:
        raise RuntimeError(f"hash written to the wrong place for {section}.{field_name}")
    return out


def write_holdout_hash(cfg: Config, sha: str) -> Config:
    return write_frozen_hash(cfg, "holdout", "sha256", sha)
