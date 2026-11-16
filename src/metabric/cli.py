"""Command line: python run.py <command> [options]. Each invocation is one run."""
from __future__ import annotations

import argparse
import sys

from .artifacts import Run
from .config import load_config
from .data import load_dataset


def cmd_crosswalk(args) -> str:
    from .crosswalk import build_crosswalk, check_coverage, write_crosswalk
    cfg = load_config()
    run = Run(cfg, "crosswalk")
    st = run.stage("crosswalk")
    try:
        run.record_input(cfg.path("data"), "dataset (column names only)")
        for name in cfg["gene_sets"]["files"].values():
            run.record_input(cfg.path("reference") / name, "gene-set file")
        alias = cfg.root / cfg["gene_sets"]["aliases"]
        if alias.is_file():
            run.record_input(alias, "symbol aliases (hand-maintained)")
        res = build_crosswalk(cfg)
        for p in write_crosswalk(res, cfg):
            run.record_external(p, "gene crosswalk table", "metabric.crosswalk.write_crosswalk")
        st.table("coverage", res.coverage, "Gene-set coverage of the 489 RNA and 173 DNA columns per source")
        chk = check_coverage(res.coverage, cfg)
        st.table("coverage_check", chk, "Observed coverage against the expected values in CLAUDE.md 3b")
        st.table("alias_log", res.alias_log, "Aliases applied to gene-set symbols (from symbol_aliases.csv)")
        st.log.info("coverage:\n%s", res.coverage.to_string(index=False))
        bad = chk[~chk.match]
        if len(bad):
            st.log.warning("coverage differs from expected:\n%s", bad.to_string(index=False))
        else:
            st.log.info("coverage matches all expected values")
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_holdout(args) -> str:
    from .splits import load_holdout, make_holdout
    cfg = load_config()
    ds = load_dataset(cfg)
    cfg, hold, merges = make_holdout(cfg, ds)           # refuses if it already exists
    run = Run(cfg, "holdout")                           # run starts after the hash is in config
    st = run.stage("splits")
    try:
        run.record_input(cfg.path("data"), "dataset")
        h = load_holdout(cfg, ds)
        run.record_external(cfg.path("holdout"), "frozen holdout (created by this run)", "metabric.splits.make_holdout")
        _register_holdout_copy(st, h, merges)
        st.log.info("holdout created: %d holdout / %d train, sha256 %s", (h["set"] == "holdout").sum(),
                    (h["set"] == "train").sum(), cfg["holdout"]["sha256"])
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def _register_holdout_copy(st, h, merges=None):
    import pandas as pd
    st.table("holdout_assignment", h, "Copy of the frozen holdout assignment (patient_id, set, stratum)")
    counts = (h.groupby(["stratum", "set"]).size().unstack(fill_value=0).reset_index())
    st.table("holdout_strata", counts, "Patients per stratum (outcome x cohort x chemotherapy) in train and holdout")
    st.table("holdout_merges", merges if merges is not None else pd.DataFrame(columns=["stratum", "n", "merged_into"]),
             "Strata merged before the holdout split (empty = none)")


def cmd_eda(args) -> str:
    from .eda import run_eda
    cfg = load_config()
    modes = ["descriptive", "modelling"] if args.mode == "both" else [args.mode]
    run = Run(cfg, f"eda --mode {args.mode}")
    try:
        for m in modes:
            run_eda(cfg, run, m)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="run.py", description="METABRIC v2 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("crosswalk", help="build reference/crosswalk/ from the configured gene-set files")
    sub.add_parser("holdout", help="create the frozen 20%% holdout (once only)")
    e = sub.add_parser("eda", help="rerunnable EDA stage")
    e.add_argument("--mode", choices=["descriptive", "modelling", "both"], default="both")
    args = p.parse_args(argv)
    rid = {"crosswalk": cmd_crosswalk, "holdout": cmd_holdout, "eda": cmd_eda}[args.cmd](args)
    print(f"run complete: outputs/{rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
