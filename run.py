"""Entry point: python run.py <crosswalk|holdout|eda> [...]. Puts src/ on the path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from metabric.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
