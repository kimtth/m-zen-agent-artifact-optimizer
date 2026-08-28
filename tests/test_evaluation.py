from __future__ import annotations

import json
from pathlib import Path

from zen.domain.core import (
    BehaviorContract,
    Constraint,
    EvaluationCase,
    ReaderQuestion,
    Rule,
    RunRecord,
)
from zen.pipeline.evaluation import EvaluationCache, Evaluator
from zen.runtime.lm import FunctionModel


def _model(system: str, user: str) -> str:
    data = json.loads(user)
    if "semantic criteria" in system:
        return json.dumps(
            {
                "checks": [
                    {
                        "rule": rule["id"],
                        "passed": True,
                        "evidence": "캐시",
                        "feedback": "통과",
                    }
                    for rule in data["obligations"]
                ],
                "criteria_passed": True,
                "criteria_feedback": "",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "answers": [
                {"id": question["id"], "correct": True, "evidence": "캐시"}
                for question in data["questions"]
            ]
        },
        ensure_ascii=False,
    )


def test_semantic_behavior_and_reader_evidence_pass(tmp_path: Path) -> None:
    contract = BehaviorContract(
        "한국어",
        "변경 설명",
        (Rule("O1", "변경을 설명한다", "변경을 설명하세요."),),
    )
    case = EvaluationCase(
        "case",
        "normal",
        "family",
        "무엇이 바뀌었나요?",
        {"변경": "캐시"},
        ("O1",),
        ("캐시 변경",),
        (),
        (ReaderQuestion("what", "무엇?", "변경"),),
        (Constraint("length", "max_output_tokens", 20),),
    )
    run = RunRecord("case", "캐시 적용", 10, 4, 1)

    result = Evaluator(FunctionModel(_model), EvaluationCache(tmp_path)).evaluate(
        contract, case, run
    )

    assert result.behavior.passed
    assert result.understanding.passed
    assert 0 < result.understanding.tokens <= result.output_tokens


def test_unknown_constraint_fails_closed(tmp_path: Path) -> None:
    contract = BehaviorContract(
        "한국어", "목적", (Rule("O1", "규칙", "규칙"),)
    )
    case = EvaluationCase(
        "case",
        "normal",
        "family",
        "질문",
        {},
        ("O1",),
        (),
        (),
        (ReaderQuestion("what", "무엇?", "결과"),),
        (Constraint("mystery", "unknown", 1),),
    )
    run = RunRecord("case", "캐시", 5, 2, 1)

    result = Evaluator(FunctionModel(_model), EvaluationCache(tmp_path)).evaluate(
        contract, case, run
    )

    assert not result.behavior.passed
    assert result.behavior.critical_failure


def test_non_activation_case_without_obligations_can_pass(tmp_path: Path) -> None:
    contract = BehaviorContract(
        "한국어", "목적", (Rule("O1", "규칙", "규칙"),)
    )
    case = EvaluationCase(
        "case",
        "irrelevant",
        "family",
        "관련 없는 질문",
        {},
        (),
        (),
        ("불필요한 활성화",),
        (ReaderQuestion("what", "무엇?", "답"),),
    )
    run = RunRecord("case", "캐시", 5, 2, 1)

    result = Evaluator(FunctionModel(_model), EvaluationCache(tmp_path)).evaluate(
        contract, case, run
    )

    assert result.behavior.passed
    assert not result.behavior.critical_failure
