"""Fast offline checks for invariants that do not need a model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .domain.core import (
    Aggregate,
    BehaviorContract,
    CaseOutcome,
    Constraint,
    EvaluationCase,
    OptimizeConfig,
    ReaderQuestion,
    load_artifact,
)
from .pipeline.gate import decide
from .pipeline.synthesis import parse_contract, split_cases


def run() -> list[tuple[str, bool, str]]:
    checks = [_config_check(), _artifact_check(), _contract_check(), _split_check(), _gate_check()]
    return checks


def _config_check() -> tuple[str, bool, str]:
    semantic = OptimizeConfig()
    gepa = OptimizeConfig(engine="gepa")
    ok = (
        semantic.engine == "semantic"
        and semantic.total_call_budget == 32
        and semantic.max_metric_calls is None
        and semantic.max_body_lines("Keep all rules.\n") is None
        and not semantic.quick
        and semantic.output_dir is None
        and OptimizeConfig(quick=True).quick
        and gepa.total_call_budget == 600
        and gepa.max_metric_calls == 120
        and gepa.max_body_lines("First.\nSecond.\n") == 1
        and not OptimizeConfig(engine="gepa", aggressive_limit=None).aggressive
    )
    return "semantic defaults and explicit GEPA legacy profile", ok, ""


def _artifact_check() -> tuple[str, bool, str]:
    with TemporaryDirectory(prefix=".zen-selfcheck-", dir=Path.cwd()) as directory:
        path = Path(directory) / "english.instructions.md"
        source = "---\r\napplyTo: '**/*.py'\r\n---\r\nExplain the result first.\r\n"
        path.write_bytes(source.encode("utf-8"))
        artifact = load_artifact(path)
        candidate = artifact.render("Explain the result.\n")
        ok = candidate.startswith(artifact.immutable_prefix) and "\r\n" in candidate
        return "artifact metadata and line endings stay frozen", ok, ""


def _contract_check() -> tuple[str, bool, str]:
    source = "Explain the result first."
    value = {
        "language": "English",
        "purpose": "Clear results",
        "obligations": [
            {
                "id": "O1",
                "statement": "Explain the result first",
                "severity": "critical",
                "source_evidence": source,
            }
        ],
        "prohibitions": [],
    }
    contract = parse_contract(value, source)
    ok = isinstance(contract, BehaviorContract) and contract.language == "English"
    return "contracts preserve language metadata", ok, ""


def _split_check() -> tuple[str, bool, str]:
    cases = [
        EvaluationCase(
            id=f"case-{index}",
            category="normal",
            family=f"family-{index}",
            inquiry="Question",
            context={"value": index},
            obligations=("O1",),
            must_include=("result",),
            must_not=(),
            reader_questions=(ReaderQuestion("what", "What changed?", "result"),),
            constraints=(Constraint("limit", "max_output_tokens", 20),),
        )
        for index in range(50)
    ]
    dataset = split_cases(cases, (30, 10, 10), 7)
    families = [
        {case.family for case in dataset.train},
        {case.family for case in dataset.validation},
        {case.family for case in dataset.holdout},
    ]
    ok = [len(dataset.train), len(dataset.validation), len(dataset.holdout)] == [30, 10, 10]
    ok = ok and not (families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
    return "dataset split is deterministic and family-safe", ok, ""


def _gate_check() -> tuple[str, bool, str]:
    cases = (
        CaseOutcome("A", 10, 10, 10, {"quality": 10}, {"what": 10}),
        CaseOutcome("B", 10, 10, 10, {"quality": 10}, {"what": 10}),
    )
    baseline = Aggregate(100, 2, 2, 0, 100, 80, cases)
    boundary = replace(
        cases[0], behavior_passes=9, understanding_passes=9,
        rule_passes={"quality": 9}, reader_passes={"what": 9},
    )
    candidate = Aggregate(70, 2, 2, 0, 70, 50, (boundary, cases[1]))
    ok = decide(baseline, candidate).accepted
    for field in ("behavior_passes", "understanding_passes"):
        regressed = replace(candidate, cases=(replace(boundary, **{field: 8}), cases[1]))
        decision = decide(baseline, regressed)
        ok = ok and not decision.accepted and not decision.inconclusive
    critical = decide(baseline, replace(candidate, critical_failures=1))
    incomplete = decide(baseline, replace(candidate, evidence_errors=("missing trial",)))
    ok = ok and not critical.accepted and incomplete.inconclusive
    return "bounded trial-rate quality gates outrank token reduction", ok, ""
