"""The code must work from a fresh copy of the folder placed elsewhere (B6)."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_copy_resolves_its_own_root(tmp_path):
    dst = tmp_path / "elsewhere" / "pipeline_copy"
    dst.mkdir(parents=True)
    for f in ("config.yaml", "CLAUDE.md", "run.py"):
        shutil.copyfile(ROOT / f, dst / f)
    for d in ("src", "data"):
        shutil.copytree(ROOT / d, dst / d, ignore=shutil.ignore_patterns("__pycache__"))
    code = ("import sys; sys.path.insert(0, 'src'); sys.dont_write_bytecode = True\n"
            "from metabric.config import load_config; from metabric.data import load_dataset\n"
            "c = load_config(); d = load_dataset(c)\n"
            "print(c.root); print(d.frame.shape[0])")
    out = subprocess.run([sys.executable, "-c", code], cwd=dst, capture_output=True, text=True, check=True)
    root_line, n = out.stdout.strip().splitlines()
    assert Path(root_line).resolve() == dst.resolve()
    assert n == "1904"
