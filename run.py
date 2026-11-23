"""Command-line entry point. Run ``python run.py --help`` for the list of commands.

Checks that use the saved outputs of the published run (no API key, a few minutes):
    verify    recompute every code-derived number in the manuscript from the saved outputs and the
              ground truth, and report PASSED or FAILED for each one
    figures   redraw every code-generated figure and table from the saved outputs
    demo      classify three example terms and compare the result with the saved output
              (the classification needs an API key; without one the saved output is shown)

New runs (need OPENROUTER_API_KEY and NCBI_EMAIL in the environment; they call a paid model API):
    classify          run the agent on a list of HPO terms
    baseline          run the no-retrieval baseline on the ground-truth terms and score it
    model-selection   rerun the model selection (Supplementary Figure S1)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEMO_TERMS = ["HP:0001249", "HP:0002414", "HP:0000574"]   # Intellectual disability, Spina bifida, Thick eyebrow


def fresh(path: str) -> Path:
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise SystemExit(f"{path} is not empty. Choose a new output folder or delete it.")
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dotenv() -> None:
    """Reads KEY=VALUE lines from .env in the repository root, if it exists. Existing variables win."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_verify(args) -> None:
    from hpo_severity import verify
    failed = verify.run(str(fresh(args.output)))
    raise SystemExit(1 if failed else 0)


def cmd_figures(args) -> None:
    from hpo_severity import baseline, figures, gene_level, gene_table, paths
    out = fresh(args.output)
    paths.prepare()
    work = out / "_work"
    print("Scoring the agent and the baseline against the ground truth ...")
    baseline.score_published(work / "baseline")
    table = paths.GENE_TABLE
    if args.rebuild_gene_table:
        print("Rebuilding the gene-disease table from the inputs ...")
        table = gene_table.build(work / "gene_table" / "gene_disease_table.csv")
    print("Figure 7 ...")
    gene_level.run(work / "gene_level", table)
    print("Figures 2 to 6 and S1 to S3 ...")
    figures.run(out, work / "baseline" / "per_term_results.csv", work / "gene_level")
    print(f"\nFigures in {out / 'main'} and {out / 'supplementary'}; reports in {out / 'data_and_reports'}")


def _set_agent_environment(input_csv: Path, release: str, output: Path, workers: int | None, index: str | None) -> None:
    from hpo_severity import paths
    os.environ["INPUT_CSV_PATH"] = str(input_csv)
    os.environ["HPO_ONTOLOGY_PATH"] = str(paths.HPO_RELEASES[release])
    os.environ["OUTPUT_DIR"] = str(output)
    if index:
        os.environ["HPO_INDEX_PATH"] = str(Path(index).resolve())
    os.environ.setdefault("RUN_VERSION", "run1")
    if workers:
        os.environ["MAX_WORKERS"] = str(workers)


def cmd_classify(args) -> None:
    from hpo_severity import paths
    load_dotenv()
    paths.prepare()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _set_agent_environment(Path(args.input).resolve(), args.hpo_release, out.resolve(), args.workers, args.index)
    from hpo_severity import agent       # reads the settings above when imported
    agent.main(["--limit", str(args.limit)] if args.limit else [])


def cmd_baseline(args) -> None:
    from hpo_severity import baseline, paths
    load_dotenv()
    paths.prepare()
    argv = ["--output", args.output, "--workers", str(args.workers)]
    argv += ["--limit", str(args.limit)] if args.limit else []
    argv += ["--dry-run"] if args.dry_run else []
    baseline.main(argv)


def cmd_model_selection(args) -> None:
    from hpo_severity import paths
    load_dotenv()
    paths.prepare()
    os.environ["OUTPUT_DIR"] = str(Path(args.output).resolve())
    os.environ["HPO_ONTOLOGY_PATH"] = str(paths.HPO_RELEASES[args.hpo_release])
    from hpo_severity import model_selection
    model_selection.run_hyperparameter_tuning()


def cmd_demo(args) -> None:
    import pandas as pd
    from hpo_severity import paths
    load_dotenv()
    paths.prepare()
    out = fresh(args.output)
    saved = pd.read_csv(paths.CLASSIFICATIONS, low_memory=False).set_index("hpo_id")
    expert = pd.read_excel(paths.GROUND_TRUTH).set_index("hpo_id")
    terms = pd.DataFrame({"hpo_id": DEMO_TERMS})
    terms.to_csv(out / "demo_terms.csv", index=False)
    new = None
    if os.environ.get("OPENROUTER_API_KEY"):
        import time
        _set_agent_environment(out / "demo_terms.csv", args.hpo_release, out.resolve(), 3, args.index)
        from hpo_severity import agent
        start = time.time()
        agent.main([])
        print(f"\nDemo classification took {time.time() - start:.0f} s.")
        files = sorted(out.glob("hpo_classifications_*.csv"))
        new = pd.read_csv(files[0]).set_index("hpo_id") if files else None
    else:
        print("OPENROUTER_API_KEY is not set, so no new classification was run. The saved output of the published run is shown.")
        print("To classify the terms: export OPENROUTER_API_KEY=... and NCBI_EMAIL=..., then run python run.py demo again.\n")
    rows = []
    for h in DEMO_TERMS:
        s = saved.loc[h]
        row = {"hpo_id": h, "name": s.hpo_name,
               "expert": f"{expert.loc[h, 'Final_classification_category']} / tier {expert.loc[h, 'Final classification_tier']}",
               "saved run": f"{s.classification_category} / tier {s.classification_tier}"}
        if new is not None and h in new.index:
            row["this run"] = f"{new.loc[h, 'classification_category']} / tier {new.loc[h, 'classification_tier']}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "demo_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nWritten to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify", help="recompute every manuscript number and report PASSED or FAILED")
    p.add_argument("--output", default="outputs/verify")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("figures", help="redraw every figure from the saved outputs")
    p.add_argument("--output", default="outputs/figures")
    p.add_argument("--rebuild-gene-table", action="store_true",
                   help="rebuild the gene-disease table from the inputs instead of using the published one")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("demo", help="classify three example terms and compare with the saved output")
    p.add_argument("--output", default="outputs/demo")
    p.add_argument("--hpo-release", choices=["2025-05-06", "2026-09-01"], default="2025-05-06")
    p.add_argument("--index", help="folder of the HPO record index (default data/inputs/hpo_record_index)")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("classify", help="run the agent on a list of HPO terms (needs an API key)")
    p.add_argument("--input", default=str(ROOT / "data" / "hpo_terms.csv"), help="CSV with an hpo_id column")
    p.add_argument("--hpo-release", choices=["2025-05-06", "2026-09-01"], default="2025-05-06",
                   help="HPO release the agent reads (default 2025-05-06, as in the manuscript)")
    p.add_argument("--output", default="outputs/classify")
    p.add_argument("--limit", type=int, help="classify only the first N terms not done yet")
    p.add_argument("--workers", type=int, help="terms processed at the same time (default 9)")
    p.add_argument("--index", help="folder of the HPO record index (default data/inputs/hpo_record_index)")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("baseline", help="run the no-retrieval baseline and score it (needs an API key)")
    p.add_argument("--output", default="outputs/baseline")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="write one prompt and a token estimate, no API calls")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("model-selection", help="rerun the model selection (needs an API key)")
    p.add_argument("--output", default="outputs/model_selection")
    p.add_argument("--hpo-release", choices=["2025-05-06", "2026-09-01"], default="2025-05-06")
    p.set_defaults(func=cmd_model_selection)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
