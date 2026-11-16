import shutil
from pathlib import Path

import pytest
import yaml

from metabric.config import Config, load_config
from metabric.data import load_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(ROOT)


@pytest.fixture(scope="session")
def ds(cfg):
    return load_dataset(cfg)


@pytest.fixture
def tmp_project(tmp_path, cfg) -> Config:
    """A throwaway project root (config copy + markers) so tests never write into real outputs/."""
    root = tmp_path / "proj"
    root.mkdir()
    raw = dict(cfg.raw)
    raw["holdout"] = dict(raw["holdout"], sha256=None)
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    import re
    text = re.sub(r'(\n\s+sha256:\s*)"[0-9a-f]+"', r"\1null", text)
    (root / "config.yaml").write_text(text, encoding="utf-8")
    (root / "CLAUDE.md").write_text("marker", encoding="utf-8")
    (root / "data").mkdir()
    shutil.copyfile(cfg.path("data"), root / "data" / Path(cfg.raw["paths"]["data"]).name)
    return Config(root=root, raw=yaml.safe_load(text))
