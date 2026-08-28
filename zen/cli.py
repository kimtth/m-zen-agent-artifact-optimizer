"""Command-line interface for the rebuilt optimizer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .domain.core import (
    DEFAULT_CATEGORIES,
    QUICK_CATEGORIES,
    ArtifactError,
    OptimizeConfig,
    load_artifact,
    parse_aggressive_limit,
)
from .optimization.service import optimize, write_outputs
from .runtime.progress import ProgressBar
from .selfcheck import run as run_selfcheck


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="zen")
    result.add_argument("--version", action="version", version=f"zen {__version__}")
    result.add_argument(
        "--target-model",
        default=os.getenv("ZEN_TARGET_MODEL", "gpt-5-mini"),
        help="model that will consume the optimized artifact",
    )
    result.add_argument(
        "--strong-model",
        default=os.getenv("ZEN_STRONG_MODEL", "gpt-5"),
        help="contract, judge, and reflection model",
    )
    result.add_argument(
        "--generator-model",
        default=os.getenv("ZEN_GENERATOR_MODEL", "gpt-5-mini"),
        help="lower-cost synthetic-case model",
    )
    result.add_argument(
        "--budget",
        type=int,
        default=int(os.getenv("ZEN_BUDGET", "600")),
        help="total application model-call budget",
    )
    result.add_argument(
        "--max-metric-calls",
        type=int,
        default=int(os.getenv("ZEN_MAX_METRIC_CALLS", "120")),
        help="GEPA metric-call budget",
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--cache-dir", type=Path, default=Path(".zen-cache"))

    commands = result.add_subparsers(dest="command", required=True)
    optimize_parser = commands.add_parser("optimize", help="optimize one artifact")
    optimize_parser.add_argument("path", type=Path)
    optimize_parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the candidate and report (default: next to PATH)",
    )
    optimize_parser.add_argument(
        "--quick",
        action="store_true",
        help="use a 10-case, single-holdout profile for a fast illustrative result",
    )
    optimize_parser.add_argument(
        "--aggressive",
        nargs="?",
        const="100",
        metavar="LINES|PERCENT",
        type=_aggressive_limit,
        help="produce a minimal effective skill body; cap it at LINES or PERCENT of source (default: 100)",
    )
    detect_parser = commands.add_parser("detect", help="validate and inspect one artifact")
    detect_parser.add_argument("path", type=Path)
    commands.add_parser("selfcheck", help="run offline invariant checks")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "selfcheck":
        return _selfcheck()
    if args.command == "detect":
        try:
            artifact = load_artifact(args.path)
        except ArtifactError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        metadata = "frozen metadata" if artifact.immutable_prefix else "no metadata"
        print(f"{artifact.path}: supported ({artifact.body_tokens} mutable tokens, {metadata})")
        return 0
    if args.budget < 1 or args.max_metric_calls < 1:
        print("error: budgets must be at least 1", file=sys.stderr)
        return 1

    config = OptimizeConfig(
        target_model=args.target_model,
        strong_model=args.strong_model,
        generator_model=args.generator_model,
        seed=args.seed,
        max_metric_calls=args.max_metric_calls,
        total_call_budget=args.budget,
        holdout_repetitions=1 if args.quick else 3,
        categories=QUICK_CATEGORIES if args.quick else DEFAULT_CATEGORIES,
        split=(6, 2, 2) if args.quick else None,
        aggressive_limit=args.aggressive,
    )
    progress = ProgressBar()
    try:
        result = optimize(args.path, config, args.cache_dir, progress.update)
        candidate, report = write_outputs(result, args.output_dir)
    except Exception as exc:  # noqa: BLE001 - CLI converts domain/provider failures to exit status.
        progress.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        progress.close()
        print("cancelled", file=sys.stderr)
        return 130
    progress.close()
    print(f"Decision: {result.decision}")
    print(f"Why: {result.message or 'quality passed and communication cost fell'}")
    print(f"Source: {args.path} (unchanged)")
    if candidate is not None:
        print(f"Candidate: {candidate}")
    print(f"Report: {report}")
    print(f"Run data: {result.run_directory}")
    return 0


def _selfcheck() -> int:
    failed = False
    for name, passed, detail in run_selfcheck():
        mark = "PASS" if passed else "FAIL"
        print(f"{mark} {name}{': ' + detail if detail else ''}")
        failed = failed or not passed
    return int(failed)


def _aggressive_limit(value: str):
    try:
        return parse_aggressive_limit(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
