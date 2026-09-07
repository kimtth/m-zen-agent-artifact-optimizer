from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from zen.domain.core import (
    BehaviorContract,
    Constraint,
    EvaluationCase,
    ReaderQuestion,
    Rule,
    RunRecord,
    count_tokens,
)
from zen.pipeline import evaluation
from zen.pipeline.evaluation import (
    EvaluationCache,
    Evaluator,
    _evaluation_from_dict,
    _evaluation_to_dict,
)
from zen.runtime.lm import BudgetExceeded, FunctionModel


def _model(system: str, user: str) -> str:
    data = json.loads(user)
    if "semantic criteria" in system:
        return json.dumps(
            {
                "checks": [
                    {
                        "rule": rule["id"],
                        "passed": True,
                        "evidence": "캐시" if rule["kind"] in {"obligation", "must_include"} else None,
                        "rationale": "Required meaning is present; forbidden behavior is absent.",
                        "feedback": "통과",
                    }
                    for rule in data["checks"]
                ],
            },
            ensure_ascii=False,
        )
    if "reader_answers" in data:
        return json.dumps(
            {
                "answers": [
                    {"id": question["id"], "correct": True, "evidence": "캐시", "feedback": "Supported by the answer and facts."}
                    for question in data["questions"]
                ]
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "answers": [
                {"id": question["id"], "response": "캐시", "evidence": "캐시"}
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


def _inputs() -> tuple[BehaviorContract, EvaluationCase, RunRecord]:
    contract = BehaviorContract(
        "language-secret", "purpose-secret",
        (Rule("O1", "Describe the change", "source-secret"), Rule("O2", "Explain why", "why")),
        (Rule("P1", "Do not invent facts", "no invention"), Rule("P2", "No unrelated advice", "no advice", "minor")),
    )
    case = EvaluationCase(
        "case", "normal", "family", "inquiry-secret",
        {"change": "캐시", "private_fact": "context-secret"},
        ("O1", "O2"), ("Describe the cache", "Explain its benefit"),
        ("Do not activate unrelated skills", "Do not claim execution"),
        (
            ReaderQuestion("what", "What changed?", "criterion-secret"),
            ReaderQuestion("why", "Why?", "benefit-secret"),
            ReaderQuestion("skipped", "not-applicable-secret", "ignored-secret", False),
        ),
    )
    return contract, case, RunRecord("case", "캐시 적용", 10, 4, 1, trial_id="trial-1")


def _stage(system: str, user: str) -> str:
    if "semantic criteria" in system:
        return "behavior"
    return "grader" if "reader_answers" in json.loads(user) else "reader"


@pytest.mark.parametrize("explicit_unknown", [True, False])
def test_reader_protocol_distinguishes_stated_unknown_from_missing_information(
    tmp_path: Path, explicit_unknown: bool,
) -> None:
    contract, case, run = _inputs()
    case = replace(case, reader_questions=(ReaderQuestion("why", "Why?", "Reason is unknown."),))
    answer = "The reason is unknown." if explicit_unknown else "The account system will change."
    run = replace(run, answer=answer, output_tokens=count_tokens(answer))
    calls = []

    def complete(system: str, user: str) -> str:
        data = json.loads(user)
        calls.append(data)
        if "reader_answers" not in data:
            assert set(data) == {"answer", "questions"}
            assert "explicitly states that a fact is unknown" in system
            assert "cite that statement" in system
            return json.dumps({"answers": [{
                "id": "why", "response": "The reason is unknown.",
                "evidence": answer if explicit_unknown else None,
            }]})
        assert "If the reader citation is null" in system
        return json.dumps({"answers": [{
            "id": "why", "correct": explicit_unknown,
            "evidence": answer if explicit_unknown else None,
            "feedback": "Explicit uncertainty is cited." if explicit_unknown else "No supporting citation.",
        }]})

    result = Evaluator(FunctionModel(complete), EvaluationCache(tmp_path))._understanding(
        contract, case, run,
    )
    assert result.passed is explicit_unknown
    assert not result.error
    assert len(calls) == (2 if explicit_unknown else 1)


def _mutating_model(stage: str, mutate: Callable[[dict[str, Any]], None]) -> FunctionModel:
    def complete(system: str, user: str) -> str:
        result = json.loads(_model(system, user))
        if _stage(system, user) == stage:
            mutate(result)
        return json.dumps(result, ensure_ascii=False)

    return FunctionModel(complete)


CHECK_IDS = ("O1", "O2", "P1", "P2", "must_include:1", "must_include:2", "must_not:1", "must_not:2")


def test_individual_checks_have_stable_ids_and_no_aggregate_criteria(tmp_path: Path) -> None:
    contract, case, run = _inputs()
    result = Evaluator(FunctionModel(_model), EvaluationCache(tmp_path)).evaluate(contract, case, run)

    assert tuple(check.rule for check in result.behavior.checks) == CHECK_IDS
    assert result.behavior.passed
    assert not result.error
    assert next(check for check in result.behavior.checks if check.rule == "P2").severity == "minor"
    assert all(answer.response == "캐시" and answer.feedback for answer in result.understanding.answers)


@pytest.mark.parametrize("rule_id", CHECK_IDS)
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_each_behavior_check_is_required_exactly_once(
    tmp_path: Path, rule_id: str, mutation: str
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        item = next(check for check in value["checks"] if check["rule"] == rule_id)
        if mutation == "missing":
            value["checks"].remove(item)
        else:
            value["checks"].append(item.copy())

    result = Evaluator(_mutating_model("behavior", mutate), EvaluationCache(tmp_path)).evaluate(*_inputs())

    assert not result.behavior.passed
    assert mutation in result.behavior.error
    assert rule_id in result.behavior.error
    assert result.behavior.error in result.error


@pytest.mark.parametrize("stage", ["behavior", "reader", "grader"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected", "nonstring", "not-object"])
def test_judge_response_ids_are_exact(
    tmp_path: Path, stage: str, mutation: str
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        items = value["checks" if stage == "behavior" else "answers"]
        field = "rule" if stage == "behavior" else "id"
        if mutation == "missing":
            items.pop()
        elif mutation == "duplicate":
            items.append(items[0].copy())
        elif mutation == "unexpected":
            items[0][field] += " "
        elif mutation == "nonstring":
            items[0][field] = 1
        else:
            items[0] = "not an object"

    result = Evaluator(_mutating_model(stage, mutate), EvaluationCache(tmp_path)).evaluate(*_inputs())

    measured = result.behavior if stage == "behavior" else result.understanding
    assert not measured.passed
    assert measured.error
    assert measured.error in result.error


@pytest.mark.parametrize("stage", ["behavior", "grader"])
@pytest.mark.parametrize("invalid_bool", ["false", "true", 0, 1, None, [], {}])
def test_judge_booleans_are_not_coerced(tmp_path: Path, stage: str, invalid_bool: Any) -> None:
    def mutate(value: dict[str, Any]) -> None:
        if stage == "behavior":
            value["checks"][0]["passed"] = invalid_bool
        else:
            value["answers"][0]["correct"] = invalid_bool

    result = Evaluator(_mutating_model(stage, mutate), EvaluationCache(tmp_path)).evaluate(*_inputs())

    measured = result.behavior if stage == "behavior" else result.understanding
    assert not measured.passed
    assert "JSON boolean" in measured.error


@pytest.mark.parametrize("rule_id,passed", [("O1", False), ("must_include:1", False), ("P1", True), ("must_not:1", True)])
@pytest.mark.parametrize("rationale", [None, "", "   ", "The specified meaning is absent from the entire answer."])
def test_absence_checks_require_explicit_rationale(
    tmp_path: Path, rule_id: str, passed: bool, rationale: str | None
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        item = next(check for check in value["checks"] if check["rule"] == rule_id)
        item.update(passed=passed, evidence=None, rationale=rationale)

    result = Evaluator(_mutating_model("behavior", mutate), EvaluationCache(tmp_path)).evaluate(*_inputs())

    if rationale and rationale.strip():
        assert not result.behavior.error
        check = next(check for check in result.behavior.checks if check.rule == rule_id)
        assert check.passed is passed
        assert check.evidence is None
        assert rationale in check.feedback
    else:
        assert "rationale" in result.behavior.error


@pytest.mark.parametrize("rule_id,passed", [("O1", True), ("must_include:1", True), ("P1", False), ("must_not:1", False)])
@pytest.mark.parametrize("evidence", [None, "", " ", "fabricated evidence", "캐시"])
def test_positive_claims_and_prohibition_violations_require_exact_evidence(
    tmp_path: Path, rule_id: str, passed: bool, evidence: str | None
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        item = next(check for check in value["checks"] if check["rule"] == rule_id)
        item.update(passed=passed, evidence=evidence)

    result = Evaluator(_mutating_model("behavior", mutate), EvaluationCache(tmp_path)).evaluate(*_inputs())

    if evidence == "캐시":
        assert not result.error
        check = next(check for check in result.behavior.checks if check.rule == rule_id)
        assert check.passed is passed
        assert result.behavior.critical_failure is (not passed)
    else:
        assert "exact substring" in result.behavior.error


def test_reader_receives_only_output_and_questions_and_grader_gets_actual_response(tmp_path: Path) -> None:
    contract, case, run = _inputs()
    stages = []

    def complete(system: str, user: str) -> str:
        stage = _stage(system, user)
        stages.append(stage)
        data = json.loads(user)
        if stage == "reader":
            assert data == {
                "answer": run.answer,
                "questions": [{"id": question.id, "question": question.question} for question in case.reader_questions if question.applicable],
            }
            assert "-secret" not in system + user
        elif stage == "grader":
            assert data["answer"] == run.answer
            assert data["context"] == case.context
            assert data["inquiry"] == case.inquiry
            assert data["questions"][0]["criterion"] == "criterion-secret"
            assert data["reader_answers"] == [
                {"id": question.id, "response": "캐시", "evidence": "캐시"}
                for question in case.reader_questions if question.applicable
            ]
        return _model(system, user)

    result = Evaluator(FunctionModel(complete), EvaluationCache(tmp_path)).evaluate(contract, case, run)

    assert stages == ["behavior", "reader", "grader"]
    assert result.understanding.passed
    assert not result.error


@pytest.mark.parametrize("stage", ["reader", "grader"])
@pytest.mark.parametrize("evidence", ["fabricated evidence", "", " ", 123])
def test_reader_and_grader_reject_fabricated_or_invalid_evidence(
    tmp_path: Path, stage: str, evidence: Any
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        value["answers"][0]["evidence"] = evidence

    result = Evaluator(_mutating_model(stage, mutate), EvaluationCache(tmp_path)).evaluate(*_inputs())

    assert not result.understanding.passed
    assert result.understanding.accuracy == 0
    assert result.understanding.tokens == result.output_tokens
    assert "exact substring" in result.understanding.error
    assert result.understanding.error in result.error


@pytest.mark.parametrize("stage", ["reader", "grader"])
def test_correct_grade_requires_both_reader_and_grader_citations(tmp_path: Path, stage: str) -> None:
    def mutate(value: dict[str, Any]) -> None:
        value["answers"][0]["evidence"] = None

    result = Evaluator(_mutating_model(stage, mutate), EvaluationCache(tmp_path)).evaluate(*_inputs())

    assert not result.understanding.passed
    if stage == "reader":
        assert not result.error
        assert result.understanding.accuracy == 0.5
        assert result.understanding.answers[0].feedback == "Reader supplied no supporting citation."
    else:
        assert "exact substring" in result.understanding.error


def test_incorrect_reader_response_is_a_quality_failure_not_a_grader_error(tmp_path: Path) -> None:
    def complete(system: str, user: str) -> str:
        value = json.loads(_model(system, user))
        if _stage(system, user) == "reader":
            value["answers"][0].update(response="Cannot determine.", evidence=None)
        elif _stage(system, user) == "grader":
            assert [q["id"] for q in json.loads(user)["questions"]] == ["why"]
            assert all(q["evidence"] is not None for q in json.loads(user)["reader_answers"])
        return json.dumps(value)

    result = Evaluator(FunctionModel(complete), EvaluationCache(tmp_path)).evaluate(*_inputs())

    assert not result.understanding.passed
    assert result.understanding.accuracy == 0.5
    assert not result.error
    assert result.understanding.answers[0].response == "Cannot determine."
    assert result.understanding.answers[0].feedback == "Reader supplied no supporting citation."


@pytest.mark.parametrize("stage", ["behavior", "reader", "grader"])
@pytest.mark.parametrize("failure", ["exception", "json", "schema"])
def test_model_failures_propagate_distinct_errors(tmp_path: Path, stage: str, failure: str) -> None:
    def complete(system: str, user: str) -> str:
        if _stage(system, user) == stage:
            if failure == "exception":
                raise RuntimeError("scripted failure")
            return "not JSON" if failure == "json" else "{}"
        return _model(system, user)

    result = Evaluator(FunctionModel(complete), EvaluationCache(tmp_path)).evaluate(*_inputs())

    measured = result.behavior if stage == "behavior" else result.understanding
    assert measured.error
    assert measured.error in result.error
    assert measured.error in result.feedback
    assert not measured.passed
    if stage == "grader":
        assert all(answer.response == "캐시" for answer in result.understanding.answers)
        assert all(answer.feedback == measured.error for answer in result.understanding.answers)


@pytest.mark.parametrize("stage", ["behavior", "reader", "grader"])
def test_budget_exhaustion_is_not_swallowed(tmp_path: Path, stage: str) -> None:
    def complete(system: str, user: str) -> str:
        if _stage(system, user) == stage:
            raise BudgetExceeded("offline budget")
        return _model(system, user)

    with pytest.raises(BudgetExceeded, match="offline budget"):
        Evaluator(FunctionModel(complete), EvaluationCache(tmp_path)).evaluate(*_inputs())


def test_execution_error_skips_judges_and_propagates(tmp_path: Path) -> None:
    contract, case, run = _inputs()

    def complete(system: str, user: str) -> str:
        pytest.fail("No judge calls are needed for execution errors")

    result = Evaluator(FunctionModel(complete), EvaluationCache(tmp_path)).evaluate(
        contract, case, replace(run, error="target failed")
    )

    assert result.error == "target failed"
    assert not result.behavior.passed
    assert not result.behavior.error  # Execution errors are not judge errors.
    assert result.understanding.error == "target failed"


def test_cache_is_trial_specific_and_replays_only_the_same_trial(tmp_path: Path) -> None:
    calls = []

    def complete(system: str, user: str) -> str:
        calls.append(_stage(system, user))
        return _model(system, user)

    contract, case, run = _inputs()
    evaluator = Evaluator(FunctionModel(complete), EvaluationCache(tmp_path))
    first = evaluator.evaluate(contract, case, run)
    assert evaluator.evaluate(contract, case, run) == first
    assert calls == ["behavior", "reader", "grader"]
    evaluator.evaluate(contract, case, replace(run, trial_id="trial-2"))
    assert len(calls) == 6
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize("version", ["_EVALUATION_CACHE_VERSION", "_BEHAVIOR_PROMPT_VERSION", "_READER_PROMPT_VERSION", "_READER_GRADER_PROMPT_VERSION"])
def test_cache_key_versions_every_evaluation_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    evaluator = Evaluator(FunctionModel(_model), EvaluationCache(tmp_path))
    key = evaluator._key(*_inputs())
    monkeypatch.setattr(evaluation, version, "test-next-version")
    assert evaluator._key(*_inputs()) != key


def test_serialization_roundtrips_new_fields_and_defaults(tmp_path: Path) -> None:
    result = Evaluator(FunctionModel(_model), EvaluationCache(tmp_path)).evaluate(*_inputs())
    changed = replace(
        result,
        behavior=replace(result.behavior, error="behavior error"),
        understanding=replace(result.understanding, error="reader error"),
        trial=3, error="combined error",
    )
    assert _evaluation_from_dict(json.loads(json.dumps(_evaluation_to_dict(changed)))) == changed

    legacy = _evaluation_to_dict(result)
    for field in ("cache_version", "trial", "error"):
        legacy.pop(field)
    legacy["behavior"].pop("error")
    legacy["understanding"].pop("error")
    for answer in legacy["understanding"]["answers"]:
        answer.pop("response")
        answer.pop("feedback")
    restored = _evaluation_from_dict(legacy)
    assert restored.trial == 0
    assert restored.error == restored.behavior.error == restored.understanding.error == ""
    assert all(answer.response == answer.feedback == "" for answer in restored.understanding.answers)
