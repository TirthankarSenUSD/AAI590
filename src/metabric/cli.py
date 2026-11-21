"""Command line: python run.py <command> [options]. Each invocation is one run."""
from __future__ import annotations

import argparse
import sys

from .core.artifacts import Run
from .core.config import load_config
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
        # Snapshot of the four tables inside the run: reference/crosswalk/ is rebuilt in place by later
        # runs, so the run folder keeps its own copy and its provenance stays verifiable.
        for name, df in zip(("gene_crosswalk", "gene_set_membership", "gene_set_summary", "unmatched_genes"),
                            (res.crosswalk, res.membership, res.summary, res.unmatched)):
            st.table(f"snapshot_{name}", df, f"Snapshot of reference/crosswalk/{name}.csv as built by this run")
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


def cmd_splits(args) -> str:
    from .splits import load_cv_splits, make_cv_splits, training_ids
    cfg = load_config()
    full = load_dataset(cfg)
    tr = training_ids(cfg, full)                        # verifies the holdout hash
    cfg, sp, merges = make_cv_splits(cfg, full.subset(tr))   # refuses if they already exist
    run = Run(cfg, "splits")
    st = run.stage("splits")
    try:
        run.record_input(cfg.path("data"), "dataset")
        run.record_input(cfg.path("holdout"), "frozen holdout (training rows only are split)")
        load_cv_splits(cfg, tr)
        run.record_external(cfg.path("cv_splits"), "frozen CV splits (created by this run)", "metabric.splits.make_cv_splits")
        sizes = (sp.groupby(["split_id", "role"]).size().unstack().reset_index())
        st.table("cv_split_sizes", sizes, "Rows per split: outer train and test sizes")
        st.table("cv_strata_merges", merges, "Strata merged before outer/inner splitting (empty = none)")
        st.log.info("CV splits created: %d splits, sha256 %s; merges: %d", sp.split_id.nunique(),
                    cfg["cv"]["splits_sha256"], len(merges))
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_harness(args) -> str:
    from .harness.stage import run_harness
    cfg = load_config()
    run = Run(cfg, "harness")
    try:
        run_harness(cfg, run)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_farm(args) -> str:
    from .farm.stage import run_farm
    cfg = load_config()
    run = Run(cfg, "farm" + (f" --members {','.join(args.members)}" if args.members else "")
              + (f" --arms {','.join(args.arms)}" if args.arms else ""))
    try:
        run_farm(cfg, run, args.members, args.arms)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_importance(args) -> str:
    from .importance.stage import run_importance
    cfg = load_config()
    run = Run(cfg, "importance" + (f" --farm-run {args.farm_run}" if args.farm_run else "")
              + (f" --reuse {args.reuse}" if args.reuse else "")
              + (f" --procedures {','.join(args.procedures)}" if args.procedures else ""))
    try:
        run_importance(cfg, run, args.farm_run, args.reuse, args.procedures)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_ensemble(args) -> str:
    from .ensemble.stage import run_ensemble
    cfg = load_config()
    run = Run(cfg, f"ensemble --farm-runs {','.join(args.farm_runs)}")
    try:
        run_ensemble(cfg, run, args.farm_runs)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_agreement(args) -> str:
    from .agreement.stage import run_agreement
    cfg = load_config()
    run = Run(cfg, f"agreement --farm-runs {','.join(args.farm_runs)} --importance-run {args.importance_run}")
    try:
        run_agreement(cfg, run, args.farm_runs, args.importance_run)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_report(args) -> str:
    from .report.stage import run_report
    cfg = load_config()
    run = Run(cfg, f"report --farm-runs {','.join(args.farm_runs)} --ensemble-run {args.ensemble_run} "
                   f"--agreement-run {args.agreement_run}")
    try:
        run_report(cfg, run, args.farm_runs, args.ensemble_run, args.agreement_run)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


def cmd_score_holdout(args) -> str:
    from .report.holdout import run_holdout
    cfg = load_config()
    run = Run(cfg, f"score-holdout --report-run {args.report_run}" + (" --dry-run" if args.dry_run else ""))
    try:
        run_holdout(cfg, run, args.report_run, dry_run=args.dry_run)
        run.finalize()
    except BaseException as exc:
        run.fail(exc)
        raise
    return run.run_id


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
    sub.add_parser("splits", help="create the frozen nested CV splits on the training rows (once only)")
    sub.add_parser("harness", help="run the floors through the evaluation harness")
    f = sub.add_parser("farm", help="run the model farm through the harness (members and arms from config)")
    f.add_argument("--members", type=lambda s: s.split(","), default=None, help="comma-separated subset of farm.members")
    f.add_argument("--arms", type=lambda s: s.split(","), default=None, help="comma-separated subset of farm.arms")
    imp = sub.add_parser("importance", help="gates and feature importance of a completed farm run's winners")
    imp.add_argument("--farm-run", default=None, help="farm run id (default: the newest completed farm run)")
    imp.add_argument("--reuse", default=None, help="importance run id whose per-split tables are reused (skips the refits)")
    imp.add_argument("--procedures", type=lambda s: s.split(","), default=None,
                     help="explain only these procedures, e.g. C|logreg_cause (no partial-dependence pass)")
    en = sub.add_parser("ensemble", help="fusion and ensemble of the farm winners (step 5)")
    en.add_argument("--farm-runs", type=lambda s: s.split(","), required=True,
                    help="farm run ids holding the member winners; for each (arm, member) the first run listed wins")
    ag = sub.add_parser("agreement", help="model-agreement analysis and the omics ablation (step 6)")
    ag.add_argument("--farm-runs", type=lambda s: s.split(","), required=True,
                    help="farm runs with per-patient predictions; for each procedure the first run listed wins")
    ag.add_argument("--importance-run", required=True, help="importance run with permutation importance")
    rp = sub.add_parser("report", help="validation report, final-model deployment fit and patient output (step 7)")
    rp.add_argument("--farm-runs", type=lambda s: s.split(","), required=True,
                    help="farm runs; the first holding the final arm's procedure supplies its metrics")
    rp.add_argument("--ensemble-run", required=True, help="ensemble run that chose the final model")
    rp.add_argument("--agreement-run", required=True, help="agreement run (ablation, disagreement, cause breakdown)")
    sh = sub.add_parser("score-holdout", help="score the frozen holdout ONCE with the saved final model (last step of Phase 1)")
    sh.add_argument("--report-run", required=True, help="report run whose models/final_model.joblib is scored")
    sh.add_argument("--dry-run", action="store_true", help="shuffle the holdout labels: check the code path, score nothing")
    e = sub.add_parser("eda", help="rerunnable EDA stage")
    e.add_argument("--mode", choices=["descriptive", "modelling", "both"], default="both")
    args = p.parse_args(argv)
    rid = {"crosswalk": cmd_crosswalk, "holdout": cmd_holdout, "splits": cmd_splits, "harness": cmd_harness,
           "farm": cmd_farm, "importance": cmd_importance, "ensemble": cmd_ensemble, "agreement": cmd_agreement, "report": cmd_report,
           "score-holdout": cmd_score_holdout, "eda": cmd_eda}[args.cmd](args)
    print(f"run complete: outputs/{rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
