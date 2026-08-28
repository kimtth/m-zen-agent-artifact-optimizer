"""Fast offline checks for invariants that do not need a model."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from .domain.core import (
    Aggregate,
    BehaviorContract,
    Constraint,
    EvaluationCase,
    ReaderQuestion,
    load_artifact,
)
from .pipeline.gate import decide
from .pipeline.synthesis import parse_contract, split_cases


def run() -> list[tuple[str, bool, str]]:
    checks = [_artifact_check(), _contract_check(), _split_check(), _gate_check()]
    return checks


def _artifact_check() -> tuple[str, bool, str]:
    with TemporaryDirectory() as directory:
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
    baseline = Aggregate(100, 10, 10, 0, 100, 80)
    candidate = Aggregate(70, 10, 10, 0, 70, 50)
    accepted = decide(baseline, candidate).accepted
    regressed = Aggregate(60, 9, 10, 0, 50, 40)
    rejected = not decide(baseline, regressed).accepted
    return "quality outranks token reduction", accepted and rejected, ""
