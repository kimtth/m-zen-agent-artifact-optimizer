from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

from zen.domain.core import (
    AggressiveLimit,
    BehaviorContract,
    BehaviorResult,
    CaseEvaluation,
    Check,
    Dataset,
    EvaluationCase,
    OptimizeConfig,
    ReaderQuestion,
    Rule,
    UnderstandingAnswer,
    UnderstandingResult,
    load_artifact,
)
from zen.optimization.metric import Baseline
from zen.optimization.proposer import CompressionProposer
from zen.optimization.report import render_report
from zen.optimization.service import Optimizer, _within_limit, write_outputs
from zen.runtime.harness import CandidatePolicy
from zen.runtime.lm import CallBudget, FunctionModel


def _generator(_system: str, user: str) -> str:
    data = json.loads(user)
    offset = data["case_offset"]
    cases = [
        {
            "family": f"{data['category']}-{offset + index}",
            "inquiry": f"{data['category']} 질문 {offset + index}",
            "context": {"변경": "캐시"},
            "obligations": ["O1"],
            "must_include": ["결과를 설명한다"],
            "must_not": [],
            "reader_questions": [
                {
                    "id": "what",
                    "question": "무엇이 바뀌었나요?",
                    "criterion": "결과를 찾는다",
                    "applicable": True,
                }
            ],
            "constraints": [],
        }
        for index in range(data["count"])
    ]
    return json.dumps({"cases": cases}, ensure_ascii=False)


def _strong(system: str, user: str) -> str:
    data = json.loads(user)
    if "mutable_body" in data:
        return json.dumps(
            {
                "language": "한국어",
                "purpose": "결과 설명",
                "obligations": [
                    {
                        "id": "O1",
                        "statement": "결과를 설명한다",
                        "severity": "critical",
                        "source_evidence": "결과를 먼저 설명하세요.",
                    }
                ],
                "prohibitions": [],
            },
            ensure_ascii=False,
        )
    if "cases" in data:
        return json.dumps({"accepted": [case["id"] for case in data["cases"]]})
    if "semantic criteria" in system:
        return json.dumps(
            {
                "checks": [
                    {
                        "rule": "O1",
                        "passed": True,
                        "evidence": "결과",
                        "feedback": "통과",
                    }
                ],
                "criteria_passed": True,
                "criteria_feedback": "",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {"answers": [{"id": "what", "correct": True, "evidence": "결과"}]},
        ensure_ascii=False,
    )


def test_end_to_end_accepts_candidate_without_modifying_source(tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    source = "결과를 먼저 설명하세요. 같은 결론을 반복하지 말고 이유와 영향을 간결하게 설명하세요.\n"
    path.write_text(source, encoding="utf-8")
    task = FunctionModel(
        lambda _system, _user: "[[ ## answer ## ]]\n결과\n[[ ## completed ## ]]",
        "target",
    )
    strong = FunctionModel(_strong, "strong")
    generator = FunctionModel(_generator, "generator")
    budget = CallBudget(1000)
    progress: list[tuple[int, str]] = []
    optimizer = Optimizer(
        OptimizeConfig(),
        task,
        strong,
        generator,
        budget,
        tmp_path / "cache",
        lambda percent, message: progress.append((percent, message)),
    )
    optimizer._compile = MethodType(
        lambda self, artifact, contract, dataset, baselines, run_directory: "결과를 설명하세요.",
        optimizer,
    )

    result = optimizer.run(path)
    output_directory = tmp_path / "out"
    candidate, report = write_outputs(result, output_directory)

    assert result.decision == "ACCEPT", result.message
    assert path.read_text(encoding="utf-8") == source
    assert candidate is not None and candidate.parent == output_directory
    assert candidate is not None and candidate.name == "AGENTS.optimized.md"
    assert candidate is not None and candidate.read_text(encoding="utf-8") == "결과를 설명하세요."
    assert report.parent == output_directory
    assert report.name == "AGENTS.optimize.report.md"
    assert "Decision: ACCEPT" in report.read_text(encoding="utf-8")
    assert progress[0] == (0, "Preparing artifact")
    assert progress[-1] == (100, "Finished: ACCEPT")


def test_rejected_run_removes_stale_candidate(tmp_path: Path) -> None:
    source = tmp_path / "debug.agent.md"
    source.write_text("Keep the source.", encoding="utf-8")
    output_directory = tmp_path / "out"
    stale = output_directory / "debug.agent.optimized.md"
    output_directory.mkdir()
    stale.write_text("stale", encoding="utf-8")
    result = SimpleNamespace(
        artifact_path=str(source),
        decision="REJECT",
        candidate_body=None,
        gate=None,
        message="quality regressed",
    )

    candidate, report = write_outputs(result, output_directory)

    assert candidate is None
    assert not stale.exists()
    assert report.name == "debug.agent.optimize.report.md"


def test_report_uses_only_the_artifact_filename() -> None:
    result = SimpleNamespace(
        artifact_path=r"D:\\Code\\zen-less-in-out\\examples\\in\\AGENTS.md",
        decision="REJECT",
        gate=None,
        message="quality regressed",
    )

    report = render_report(result)

    assert "# Zen optimization: AGENTS.md" in report
    assert "D:\\Code" not in report


def test_aggressive_mode_rejects_a_candidate_over_100_lines(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(
        "\n".join("Keep this skill focused." for _ in range(102)),
        encoding="utf-8",
    )
    artifact = load_artifact(path)
    policy = CandidatePolicy(
        artifact.body,
        OptimizeConfig(aggressive_limit=AggressiveLimit(lines=100)).max_body_lines(artifact.body),
    )

    try:
        policy.validate("\n".join("Keep this skill focused." for _ in range(101)))
    except ValueError as exc:
        assert "100" in str(exc)
    else:
        raise AssertionError("aggressive mode accepted a body over 100 lines")


def test_aggressive_percentage_resolves_from_original_body(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text("\n".join("Keep this skill focused." for _ in range(101)), encoding="utf-8")
    artifact = load_artifact(path)
    limit = OptimizeConfig(aggressive_limit=AggressiveLimit(percent=50)).max_body_lines(
        artifact.body
    )

    assert limit == 51
    try:
        CandidatePolicy(artifact.body, limit).validate(
            "\n".join("Keep this skill focused." for _ in range(52))
        )
    except ValueError as exc:
        assert "51" in str(exc)
    else:
        raise AssertionError("aggressive percentage accepted a body over its limit")


def test_aggressive_mode_allows_gepa_to_evaluate_oversized_source(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text("\n".join("Keep this skill focused." for _ in range(201)), encoding="utf-8")
    artifact = load_artifact(path)
    source_policy = CandidatePolicy(artifact.body)
    candidate_limit = OptimizeConfig(aggressive_limit=AggressiveLimit(lines=200)).max_body_lines(
        artifact.body
    )

    source_policy.validate(artifact.body)
    try:
        CandidatePolicy(artifact.body, candidate_limit).validate(artifact.body)
    except ValueError as exc:
        assert "200" in str(exc)
    else:
        raise AssertionError("aggressive mode accepted an oversized source as a candidate")


def test_aggressive_proposer_retries_an_oversized_draft() -> None:
    responses = iter(
        (
            "\n".join("Keep this skill focused." for _ in range(201)),
            "Keep this skill focused.",
        )
    )
    proposer = CompressionProposer(
        FunctionModel(lambda _system, _user: next(responses), "strong"),
        BehaviorContract("English", "Focus", (Rule("O1", "Focus", "Focus"),)),
        "\n".join("Keep this skill focused." for _ in range(201)),
        200,
    )

    proposal = proposer({"answer": "Keep this skill focused."}, {}, ["answer"])

    assert proposal["answer"] == "Keep this skill focused."


def test_aggressive_selection_prefers_the_best_candidate_within_the_limit() -> None:
    oversized = "\n".join("Keep this skill focused." for _ in range(201))
    optimized = SimpleNamespace(
        detailed_results=SimpleNamespace(
            candidates=[
                {"answer": oversized},
                {"answer": "Answer briefly."},
                {"answer": "Answer briefly and explain why it matters."},
            ],
            val_aggregate_scores=[0.9, 0.4, 0.7],
        )
    )

    assert _within_limit(optimized, oversized, 200) == (
        "Answer briefly and explain why it matters."
    )
    assert _within_limit(optimized, oversized, None) == oversized




def test_evaluation_skips_failed_case_and_continues(tmp_path: Path) -> None:
    calls = 0

    def task_answer(_system: str, _user: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary target failure")
        return "[[ ## answer ## ]]\n결과\n[[ ## completed ## ]]"

    task = FunctionModel(task_answer, "target")
    strong = FunctionModel(_strong, "strong")
    progress: list[tuple[int, str]] = []
    optimizer = Optimizer(
        OptimizeConfig(),
        task,
        strong,
        strong,
        CallBudget(100),
        tmp_path / "cache",
        lambda percent, message: progress.append((percent, message)),
    )
    optimizer.contract = BehaviorContract(
        "한국어",
        "결과 설명",
        (Rule("O1", "결과를 설명한다", "결과"),),
    )
    first = EvaluationCase(
        "first",
        "normal",
        "first-family",
        "첫 결과는?",
        {},
        ("O1",),
        ("결과",),
        (),
        (ReaderQuestion("what", "무엇?", "결과"),),
    )
    second = EvaluationCase(
        "second",
        "normal",
        "second-family",
        "둘째 결과는?",
        {},
        ("O1",),
        ("결과",),
        (),
        (ReaderQuestion("what", "무엇?", "결과"),),
    )

    results = optimizer._evaluate(
        "결과를 설명하세요.",
        (first, second),
        1,
        40,
        50,
        "Evaluating",
    )

    assert calls == 2
    assert results[0].behavior.critical_failure
    assert "temporary target failure" in results[0].feedback
    assert results[1].behavior.passed
    assert any("skipped first" in message for _, message in progress)
    assert progress[-1] == (50, "Evaluating")


def test_real_gepa_compile_uses_feedback_and_returns_valid_body(tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("결과를 먼저 명확하고 간결하게 설명하세요.\n", encoding="utf-8")
    artifact = load_artifact(path)
    contract = BehaviorContract(
        "한국어",
        "결과 설명",
        (Rule("O1", "결과를 설명한다", "결과를 먼저 명확하고 간결하게 설명하세요."),),
    )

    def judge(system: str, user: str) -> str:
        if "Rewrite one instruction body" in system:
            return "결과를 설명하세요."
        if "semantic criteria" in system:
            return json.dumps(
                {
                    "checks": [
                        {
                            "rule": "O1",
                            "passed": True,
                            "evidence": "결과",
                            "feedback": "통과",
                        }
                    ],
                    "criteria_passed": True,
                    "criteria_feedback": "",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"answers": [{"id": "what", "correct": True, "evidence": "결과"}]},
            ensure_ascii=False,
        )

    case = EvaluationCase(
        "train",
        "normal",
        "train-family",
        "무엇인가요?",
        {},
        ("O1",),
        ("결과",),
        (),
        (ReaderQuestion("what", "무엇?", "결과"),),
    )
    validation = EvaluationCase(
        "validation",
        "normal",
        "validation-family",
        "결과를 말하세요.",
        {},
        ("O1",),
        ("결과",),
        (),
        (ReaderQuestion("what", "무엇?", "결과"),),
    )
    passed = CaseEvaluation(
        "train",
        BehaviorResult(True, False, (Check("O1", True, "critical", "결과", "통과"),)),
        UnderstandingResult(True, 1.0, 2, (UnderstandingAnswer("what", True, "결과"),)),
        2,
        "통과",
    )
    baselines = {
        "train": Baseline(passed, artifact.body_tokens),
        "validation": Baseline(
            CaseEvaluation(
                "validation",
                passed.behavior,
                passed.understanding,
                passed.output_tokens,
                passed.feedback,
            ),
            artifact.body_tokens,
        ),
    }
    task = FunctionModel(
        lambda _system, _user: "[[ ## answer ## ]]\n결과\n[[ ## completed ## ]]",
        "target",
    )
    strong = FunctionModel(judge, "strong")
    budget = CallBudget(100)
    optimizer = Optimizer(
        OptimizeConfig(max_metric_calls=8),
        task,
        strong,
        strong,
        budget,
        tmp_path / "cache",
    )
    optimizer.contract = contract
    run_directory = tmp_path / "gepa-run"
    run_directory.mkdir()

    candidate = optimizer._compile(
        artifact,
        contract,
        Dataset((case,), (validation,), (), {}),
        baselines,
        run_directory,
    )

    assert candidate == "결과를 설명하세요."
