"""Command-line interface for the rebuilt optimizer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .domain.core import (
    AGGRESSIVE_DEFAULT_PERCENT,
    DEFAULT_CATEGORIES,
    QUICK_CATEGORIES,
    AggressiveLimit,
    ArtifactError,
    OptimizeConfig,
    load_artifact,
    parse_aggressive_limit,
)
from .optimization.service import optimize, write_outputs
from .runtime.progress import ProgressBar
from .selfcheck import run as run_selfcheck


class _Parser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.command == "optimize":
            defaults = OptimizeConfig(engine=parsed.engine)
            if parsed.budget is None:
                parsed.budget = defaults.total_call_budget
            if not hasattr(parsed, "aggressive"):
                parsed.aggressive = defaults.aggressive_limit
        return parsed


def parser() -> argparse.ArgumentParser:
    result = _Parser(prog="zen")
    result.add_argument("--version", action="version", version=f"zen {__version__}")
    result.add_argument(
        "--target-model",
        default=os.getenv("ZEN_TARGET_MODEL", OptimizeConfig.target_model),
        help="model that will consume the optimized artifact",
    )
    result.add_argument(
        "--strong-model",
        default=os.getenv("ZEN_STRONG_MODEL", OptimizeConfig.strong_model),
        help="contract, judge, and reflection model",
    )
    result.add_argument(
        "--generator-model",
        default=os.getenv("ZEN_GENERATOR_MODEL", OptimizeConfig.generator_model),
        help="weak model for synthetic-case generation",
    )
    result.add_argument(
        "--engine", choices=("semantic", "gepa"), default="semantic",
        help="optimizer engine (default: semantic; gepa retains the legacy search)",
    )
    result.add_argument(
        "--budget",
        type=int,
        default=os.getenv("ZEN_BUDGET"),
        help="total application model-call budget (default: semantic 32, GEPA 600)",
    )
    result.add_argument(
        "--max-metric-calls",
        type=int,
        default=os.getenv("ZEN_MAX_METRIC_CALLS"),
        help="GEPA-only metric-call budget (default: 120)",
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--cache-dir", type=Path, default=Path(".zen-cache"))

    commands = result.add_subparsers(dest="command", required=True)
    optimize_parser = commands.add_parser("optimize", help="optimize one artifact")
    optimize_parser.add_argument("path", type=Path)
    optimize_parser.add_argument(
        "--engine", choices=("semantic", "gepa"), default=argparse.SUPPRESS,
        help="optimizer engine (default: semantic)",
    )
    optimize_parser.add_argument(
        "--budget", type=int, default=argparse.SUPPRESS,
        help="total model-call budget (default: semantic 32, GEPA 600)",
    )
    optimize_parser.add_argument(
        "--max-metric-calls", type=int, default=argparse.SUPPRESS,
        help="GEPA-only metric-call budget (default: 120)",
    )
    optimize_parser.add_argument(
        "--compare-concision", action="store_true",
        help="GEPA-only diagnostic concision arm (uses additional calls; never selected)",
    )
    optimize_parser.add_argument(
        "--focus", choices=("all", "task", "communication"), default="all",
        help="mutations within an explicit zen comment-delimited section",
    )
    optimize_parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the candidate and report (default: next to PATH)",
    )
    optimize_parser.add_argument(
        "--quick",
        action="store_true",
        help="semantic: 4 cases instead of 6; GEPA: 10 cases and one holdout repetition",
    )
    optimize_parser.add_argument(
        "--accept-draft", action="store_true",
        help="explicitly accept a semantic REVIEW_REQUIRED draft as user-confirmed; preserve model warnings",
    )
    aggressive_options = optimize_parser.add_mutually_exclusive_group()
    aggressive_options.add_argument(
        "--aggressive",
        nargs="?",
        const=AggressiveLimit(percent=AGGRESSIVE_DEFAULT_PERCENT),
        default=argparse.SUPPRESS,
        metavar="LINES|PERCENT",
        type=_aggressive_limit,
        help="optional body line cap (flag alone: 50%%; semantic default: none; GEPA default: 50%%)",
    )
    aggressive_options.add_argument(
        "--no-aggressive", dest="aggressive", action="store_const", const=None,
        default=argparse.SUPPRESS,
        help="disable the body line cap",
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
    if args.engine == "semantic":
        incompatible = [
            name for enabled, name in (
                (args.max_metric_calls is not None, "--max-metric-calls"),
                (args.compare_concision, "--compare-concision"),
            ) if enabled
        ]
        if incompatible:
            print(f"error: {', '.join(incompatible)} require --engine gepa", file=sys.stderr)
            return 1
    elif args.accept_draft:
        print("error: --accept-draft requires --engine semantic", file=sys.stderr)
        return 1
    if args.budget < 1 or (args.max_metric_calls is not None and args.max_metric_calls < 1):
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
        compare_concision=args.compare_concision,
        focus=args.focus,
        engine=args.engine,
        quick=args.quick,
        output_dir=args.output_dir,
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
        label = ("Draft (review required)" if result.decision == "REVIEW_REQUIRED"
             else "Candidate (confirmed)" if result.decision == "CONFIRMED"
             else "Candidate")
        print(f"{label}: {candidate}")
    print(f"Report: {report}")
    print(f"Run data: {result.run_directory}")
    if args.engine == "semantic" and result.decision == "REVIEW_REQUIRED" and candidate is not None:
        print("Review the draft and report. Acceptance keeps all model warnings and unmet gates.")
        try:
            accepted = True if args.accept_draft else _ask_acceptance()
            if accepted is not None:
                from .optimization.semantic_outputs import write_semantic_outputs

                candidate, report = write_semantic_outputs(
                    result, args.output_dir, user_acceptance=accepted,
                )
                choice = result.details["user_acceptance"]["status"]
                print(f"User acceptance: {choice}")
                if choice == "ACCEPTED":
                    print(f"Decision: CONFIRMED (by user)\nCandidate: {candidate}")
                elif choice == "BLOCKED":
                    print(f"Why: {result.details['user_acceptance']['reason']}")
                print(f"Report: {report}")
        except KeyboardInterrupt:
            print("cancelled; draft and report retained", file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001 - preserve draft on consent/publication failure.
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 1 if result.decision in ("INCONCLUSIVE", "REVIEW_REQUIRED", "ERROR") else 0


def _ask_acceptance() -> bool | None:
    """Never infer approval from EOF, redirected input, or an empty answer."""
    if not sys.stdin.isatty():
        print("No interactive input; draft retained without confirmation. Explicit consent: --accept-draft.")
        return None
    while True:
        try:
            answer = input("Accept this draft and mark it CONFIRMED by user? [y/N]: ").strip().lower()
        except EOFError:
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please answer yes or no.")


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
