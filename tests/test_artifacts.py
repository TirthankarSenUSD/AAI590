import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from metabric.artifacts import Run  # noqa: E402


def test_run_layout_and_manifest(tmp_project):
    run = Run(tmp_project, "test", run_id="r1")
    st = run.stage("eda")
    for sub in ("figures", "tables", "logs"):
        assert (run.dir / "eda" / sub).is_dir()
    st.table("t", pd.DataFrame({"a": [1, 2]}), "a table")
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    st.figure("f", fig, pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "a figure")
    st.log.info("hello")
    run.finalize()
    m = json.loads((run.dir / "manifest.json").read_text())
    assert m["status"] == "complete" and m["seed"] == 42 and m["config_hash"] == tmp_project.hash
    kinds = sorted(a["kind"] for a in m["artifacts"])
    assert kinds == ["figure", "figure_data", "log", "table"]
    fig_entry = next(a for a in m["artifacts"] if a["kind"] == "figure")
    assert fig_entry["caption"] == "a figure" and fig_entry["data"].endswith("f.csv")
    assert (run.dir / "config.yaml").is_file()
    latest = json.loads((tmp_project.path("outputs") / "latest.json").read_text())
    assert latest["run_id"] == "r1"


def test_unlisted_file_fails_validation(tmp_project):
    run = Run(tmp_project, "test", run_id="r2")
    run.stage("eda")
    (run.dir / "eda" / "tables" / "sneaky.csv").write_text("x")
    with pytest.raises(RuntimeError, match="unlisted"):
        run.finalize()
    assert json.loads((run.dir / "manifest.json").read_text())["status"] == "invalid"


def test_reruns_never_overwrite(tmp_project):
    a = Run(tmp_project, "test", run_id="same")
    b = Run(tmp_project, "test", run_id="same")
    assert a.dir != b.dir and b.run_id == "same-2"


def test_unknown_stage_rejected(tmp_project):
    with pytest.raises(ValueError):
        Run(tmp_project, "test", run_id="r3").stage("nonsense")
