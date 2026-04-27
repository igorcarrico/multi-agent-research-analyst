"""CLI entrypoint.

Examples:
    python main.py "Impact of AI in banking fraud detection"
    python main.py "Carbon capture economics" --no-compare --max-iter 1
    python main.py "..." --output report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from agents import BaselineAgent
from evaluation import compare_outputs
from graph import run_workflow
from utils import StepLogger, load_config, summarize_run


def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Research Analyst")
    parser.add_argument("query", help="Research topic")
    parser.add_argument("--context", default="", help="Optional extra context")
    parser.add_argument("--language", default="Portuguese", help="Output language (e.g. Portuguese, English)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-compare", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--max-iter", type=int, help="Override max review loops")
    parser.add_argument("--threshold", type=float, help="Override quality threshold")
    parser.add_argument("--output", help="Write final report to this file (.md or .pdf)")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    if args.max_iter is not None:
        cfg.workflow.max_iterations = args.max_iter
    if args.threshold is not None:
        cfg.workflow.quality_threshold = args.threshold

    multi_logger = StepLogger(log_file="logs/multiagent.jsonl")

    if args.no_compare:
        final = run_workflow(
            cfg, multi_logger, query=args.query, context=args.context, language=args.language,
        )
        baseline_report = None
        comparison = None
    else:
        baseline_logger = StepLogger(log_file="logs/baseline.jsonl")

        def _baseline():
            return BaselineAgent(cfg, baseline_logger).run(
                {"query": args.query, "language": args.language}
            )["report"]

        def _multi():
            return run_workflow(
                cfg, multi_logger, query=args.query, context=args.context, language=args.language,
            )

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_b = ex.submit(_baseline)
            f_m = ex.submit(_multi)
            baseline_report = f_b.result()
            final = f_m.result()

        comp_logger = StepLogger(log_file="logs/comparison.jsonl")
        comparison = compare_outputs(
            cfg,
            comp_logger,
            query=args.query,
            analysis=final.get("analysis", ""),
            baseline_report=baseline_report,
            multiagent_report=final.get("final_report", ""),
            language=args.language,
        )

    _print_section("FINAL REPORT (multi-agent)")
    print(final["final_report"])

    _print_section("EVALUATION")
    print(json.dumps(
        {
            "scores": final.get("scores", {}),
            "weighted_score": final.get("weighted_score"),
            "iterations": final.get("iteration", 0),
        },
        indent=2,
    ))

    if comparison:
        _print_section("BASELINE vs MULTI-AGENT")
        print(json.dumps(
            {k: v for k, v in comparison.items() if k != "verdict"}, indent=2,
        ))
        print("\nJudge verdict:")
        print(comparison["verdict"])

    _print_section("COST & TOKENS")
    all_events = [e.to_dict() for e in multi_logger.events]
    if not args.no_compare:
        all_events += [e.to_dict() for e in baseline_logger.events]
        all_events += [e.to_dict() for e in comp_logger.events]
    print(json.dumps(summarize_run(all_events), indent=2))

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".pdf":
            from utils.pdf import markdown_to_pdf
            out_path.write_bytes(markdown_to_pdf(final["final_report"], title=args.query))
        else:
            out_path.write_text(final["final_report"], encoding="utf-8")
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
