from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from zen.domain.core import (
    QUICK_CATEGORIES,
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
from zen.optimization.proposer import CompressionProposer, validate_focus
from zen.optimization.report import render_report
from zen.optimization.service import Optimizer, _within_limit, write_outputs
from zen.runtime.harness import CandidatePolicy
from zen.runtime.lm import BudgetExceeded, CallBudget, FunctionModel


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
    return _judge_response(system, data)


def _judge_response(system: str, data: dict) -> str:
    if "semantic criteria" in system:
        return json.dumps({"checks": [
            {"rule": rule["id"], "passed": True, "evidence": "결과",
             "rationale": "통과", "feedback": "통과"}
            for rule in data["checks"]
        ]}, ensure_ascii=False)
    if "reader_answers" in data:
        return json.dumps({"answers": [
            {"id": q["id"], "correct": True, "evidence": "결과", "feedback": "통과"}
            for q in data["questions"]
        ]}, ensure_ascii=False)
    return json.dumps({"answers": [
        {"id": q["id"], "response": "결과", "evidence": "결과"}
        for q in data["questions"]
    ]}, ensure_ascii=False)


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
        OptimizeConfig(engine="gepa"),
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
            "[]",
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

    proposer({"answer": "Keep this skill focused."}, {}, ["answer"])
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
        OptimizeConfig(engine="gepa"),
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


@pytest.mark.parametrize("focus", ["all", "task", "communication"])
def test_real_gepa_compile_uses_feedback_and_returns_valid_body(tmp_path: Path, focus: str) -> None:
    path = tmp_path / "AGENTS.md"
    prefix = "" if focus == "all" else f"\t동결\n<!-- zen:{focus} -->"
    suffix = "" if focus == "all" else f"<!-- /zen:{focus} -->\n\t동결\n"
    path.write_text(prefix + "결과를 먼저 명확하고 간결하게 설명하세요.\n" + suffix, encoding="utf-8", newline="")
    artifact = load_artifact(path)
    contract = BehaviorContract(
        "한국어",
        "결과 설명",
        (Rule("O1", "결과를 설명한다", "결과를 먼저 명확하고 간결하게 설명하세요."),),
    )

    def judge(system: str, user: str) -> str:
        if system.startswith("Propose a removal-only"):
            return "[]"
        if "Rewrite one instruction body" in system:
            return "결과를 설명하세요."
        return _judge_response(system, json.loads(user))

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
        OptimizeConfig(engine="gepa", max_metric_calls=16, focus=focus, aggressive_limit=None),
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

    assert candidate == prefix + "결과를 설명하세요." + suffix


def _small_optimizer(tmp_path: Path, *, control: bool = False, fail: bool = False) -> Optimizer:
    def answer(_system: str, _user: str) -> str:
        if fail:
            raise RuntimeError("provider unavailable")
        return "[[ ## answer ## ]]\n결과\n[[ ## completed ## ]]"

    optimizer = Optimizer(
        OptimizeConfig(
            engine="gepa",
            categories=QUICK_CATEGORIES, split=(6, 2, 2), holdout_repetitions=1,
            compare_concision=control,
        ),
        FunctionModel(answer, "target"), FunctionModel(_strong, "strong"),
        FunctionModel(_generator, "generator"), CallBudget(1000), tmp_path / "cache",
    )
    optimizer._compile = MethodType(
        lambda self, artifact, contract, dataset, baselines, run_directory: "결과를 설명하세요.",
        optimizer,
    )
    return optimizer


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "AGENTS.md"
    path.write_text("결과를 먼저 설명하세요. 같은 결론을 반복하지 말고 이유와 영향을 간결하게 설명하세요.\n", encoding="utf-8")
    return path


def test_control_is_measured_but_never_selected(tmp_path: Path) -> None:
    optimizer = _small_optimizer(tmp_path, control=True)
    result = optimizer.run(_source(tmp_path))
    assert result.decision == "ACCEPT", result.message
    assert result.candidate_body == "결과를 설명하세요."
    control = result.details["concision_control"]
    assert control["status"] == "MEASURED"
    assert control["selected"] is False
    assert control["aggregate"]["artifact_tokens"] > result.gate.baseline.artifact_tokens
    folder = Path(result.run_directory)
    assert (folder / "control-holdout.json").is_file()
    values = json.loads((folder / "candidate-holdout.json").read_text(encoding="utf-8"))
    assert values[0]["behavior"]["checks"]
    assert values[0]["understanding"]["answers"][0]["response"] == "결과"
    assert "diagnostic only" in render_report(result)
    trial_ids = []
    for phase in ("baseline-holdout", "baseline-search", "candidate-validation",
                  "candidate-holdout", "control-holdout"):
        evidence = json.loads((folder / f"{phase}.json").read_text(encoding="utf-8"))
        trial_ids.extend(value["details"]["trial_id"] for value in evidence)
    assert len(trial_ids) == len(set(trial_ids))
    baseline = json.loads((folder / "baseline-holdout.json").read_text(encoding="utf-8"))
    seal = json.loads((folder / "holdout-seal.json").read_text(encoding="utf-8"))
    encoded = json.dumps(baseline, ensure_ascii=False, sort_keys=True).encode("utf-8")
    assert seal["sha256"] == hashlib.sha256(encoded).hexdigest()


def test_source_change_blocks_candidate_publication(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = _small_optimizer(tmp_path).run(source)
    assert result.decision == "ACCEPT"
    source.write_text("new user content", encoding="utf-8")
    candidate, report = write_outputs(result)
    assert candidate is None
    assert result.decision == "INCONCLUSIVE"
    assert "INCONCLUSIVE" in report.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == "new user content"
    summary = json.loads((Path(result.run_directory) / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "INCONCLUSIVE"


def test_failed_baseline_is_inconclusive_not_a_savings_measurement(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = _small_optimizer(tmp_path, fail=True).run(source)
    assert result.decision == "INCONCLUSIVE"
    candidate, report = write_outputs(result)
    assert candidate is None
    assert "INCONCLUSIVE" in report.read_text(encoding="utf-8")
    assert result.gate is None


def test_budget_failure_retains_inconclusive_summary(tmp_path: Path) -> None:
    from zen.runtime.lm import BudgetExceeded

    optimizer = _small_optimizer(tmp_path)

    def exhausted(*_args):
        raise BudgetExceeded("test budget")

    optimizer._compile = exhausted
    result = optimizer.run(_source(tmp_path))
    assert result.decision == "INCONCLUSIVE"
    summary = json.loads((Path(result.run_directory) / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "INCONCLUSIVE"
    assert "test budget" in summary["message"]


def test_control_budget_failure_does_not_replace_valid_candidate(tmp_path: Path) -> None:
    from zen.runtime.lm import BudgetExceeded

    optimizer = _small_optimizer(tmp_path, control=True)
    evaluate = optimizer._evaluate

    def with_control_failure(instructions, *args):
        if "Be concise where compatible" in instructions:
            raise BudgetExceeded("control budget")
        return evaluate(instructions, *args)

    optimizer._evaluate = with_control_failure
    result = optimizer.run(_source(tmp_path))
    assert result.decision == "ACCEPT"
    assert result.details["concision_control"]["status"] == "INCONCLUSIVE"


def test_each_evaluation_arm_gets_new_target_and_judge_trials(tmp_path: Path) -> None:
    optimizer = _small_optimizer(tmp_path)
    optimizer.contract = BehaviorContract("한국어", "결과", (Rule("O1", "결과", "결과"),))
    case = EvaluationCase("case", "normal", "family", "결과?", {}, ("O1",), (), (),
                          (ReaderQuestion("what", "무엇?", "결과"),))
    target_calls, judge_calls = [], []
    target = optimizer.task_model
    judge = optimizer.strong_model
    target_complete, judge_complete = target.complete, judge.complete

    def track_target(system, user):
        target_calls.append(user)
        return target_complete(system, user)

    def track_judge(system, user):
        judge_calls.append(user)
        return judge_complete(system, user)

    target.complete, judge.complete = track_target, track_judge
    for _ in range(2):
        values = optimizer._evaluate("결과를 설명하세요.", (case,), 3, 0, 100, "test")
        assert [v.trial for v in values] == [0, 1, 2]
    assert len(target_calls) == 6
    assert len(judge_calls) == 18


def test_focus_requires_explicit_boundary_before_any_model_call(tmp_path: Path) -> None:
    from dataclasses import replace

    optimizer = _small_optimizer(tmp_path)
    optimizer.config = replace(optimizer.config, focus="communication")
    with pytest.raises(ValueError, match="absent"):
        optimizer.run(_source(tmp_path))


@pytest.mark.parametrize("focus", ["task", "communication"])
def test_aggressive_fallback_preserves_frozen_whitespace(focus: str) -> None:
    prefix = f"\tFrozen prefix\n<!-- zen:{focus} -->\n"
    suffix = f"\n<!-- /zen:{focus} -->\n\tFrozen suffix\n\n"
    original = prefix + "Explain the result.\n" * 10 + suffix
    candidate = prefix + "Explain the result." + suffix
    limit = len(candidate.splitlines())
    optimized = SimpleNamespace(detailed_results=SimpleNamespace(
        candidates=[{"answer": original}, {"answer": candidate}],
        val_aggregate_scores=[0.9, 0.8],
    ))

    selected = _within_limit(optimized, original, limit)

    assert selected == candidate
    validate_focus(original, focus).validate(selected)
    CandidatePolicy(original, limit).validate(selected)


@pytest.mark.parametrize("failure", ["rule", "reader", "error"])
def test_validation_rejection_reports_actual_reasons(tmp_path: Path, failure: str) -> None:
    optimizer = _small_optimizer(tmp_path, control=True)
    progress = []
    optimizer.progress = lambda percent, message: progress.append((percent, message))
    evaluate = optimizer.evaluator.evaluate
    failed_cases = []

    def fail_validation(contract, case, run):
        value = evaluate(contract, case, run)
        if case in optimizer.dataset.validation and progress[-1][0] >= 78:
            failed_cases.append(case.id)
            if failure == "rule":
                checks = tuple(replace(c, passed=False) if c.rule == "O1" else c
                               for c in value.behavior.checks)
                return replace(value, behavior=replace(
                    value.behavior, passed=False, critical_failure=True, checks=checks,
                ))
            if failure == "reader":
                return replace(value, understanding=replace(
                    value.understanding, passed=False, accuracy=0.0,
                    answers=tuple(replace(a, correct=False) for a in value.understanding.answers),
                ))
            return replace(value, error="validation judge unavailable")
        return value

    optimizer.evaluator.evaluate = fail_validation
    result = optimizer.run(_source(tmp_path))

    decision = "INCONCLUSIVE" if failure == "error" else "REJECT"
    assert result.decision == decision
    reason = {
        "rule": "no successful behavior/reader trial",
        "reader": "no successful behavior/reader trial",
        "error": "evaluation error",
    }[failure]
    expected = f"{failed_cases[0]}: {reason}"
    assert expected in result.message
    assert expected in result.details["validation_reasons"]
    if failure != "error":
        dimension = "behavior" if failure == "rule" else "reader"
        assert (
            f"overall {dimension} pass rate dropped by more than 5 percentage points"
            in result.details["validation_reasons"]
        )
    if failure == "rule":
        assert f"observed {len(failed_cases)} critical failure(s)" in result.message
    report = render_report(result)
    assert expected in report
    assert "Validation reasons:" in report
    assert f"Decision: {decision}" in report
    assert all(message == f"Finished: {decision}"
               for _, message in progress if message.startswith("Finished:"))
    folder = Path(result.run_directory)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert expected in summary["details"]["validation_reasons"]
    if failure == "error":
        values = json.loads((folder / "candidate-validation.json").read_text(encoding="utf-8"))
        assert all(value["error"] == "validation judge unavailable" for value in values)
        assert "Validation evidence is incomplete or contains errors." in report
    assert not (folder / "candidate-holdout.json").exists()
    assert "concision_control" not in result.details


def _inject_phase_trials(
    optimizer: Optimizer,
    transform: Callable[[str, EvaluationCase, CaseEvaluation], CaseEvaluation],
) -> list[str]:
    evaluate_phase = optimizer._evaluate
    evaluate = optimizer.evaluator.evaluate
    phases = []
    trials = {}

    def track_phase(instructions, cases, repetitions, start, end, message, journal_path=None):
        assert journal_path is not None
        phases.append(journal_path.stem)
        trials.clear()
        return evaluate_phase(instructions, cases, repetitions, start, end, message, journal_path)

    def inject(contract, case, run):
        trial = trials.get(case.id, 0)
        trials[case.id] = trial + 1
        value = replace(evaluate(contract, case, run), trial=trial)
        return transform(phases[-1], case, value)

    optimizer._evaluate = track_phase
    optimizer.evaluator.evaluate = inject
    return phases


def _quality_trial(
    value: CaseEvaluation, *, behavior: bool = True, reader: bool = True,
) -> CaseEvaluation:
    # A noncritical check permits behavior loss without introducing a critical failure.
    check = Check("quality", behavior, "noncritical", "결과", "scripted quality outcome")
    return replace(
        value,
        behavior=replace(
            value.behavior, passed=value.behavior.passed and behavior,
            checks=(*value.behavior.checks, check),
        ),
        understanding=replace(
            value.understanding, passed=value.understanding.passed and reader,
            accuracy=value.understanding.accuracy if reader else 0.0,
            answers=tuple(replace(a, correct=a.correct and reader)
                          for a in value.understanding.answers),
        ),
    )


@pytest.mark.parametrize(("repetitions", "behavior_loss", "reader_loss"), [
    (10, 1, 0), (10, 0, 1), (10, 1, 1),
    (10, 2, 0), (10, 0, 2),
    (9, 1, 0), (9, 0, 1),
])
def test_holdout_trial_rate_allowance_is_independent_and_not_pooled_with_validation(
    tmp_path: Path, repetitions: int, behavior_loss: int, reader_loss: int,
) -> None:
    optimizer = _small_optimizer(tmp_path)
    optimizer.config = replace(optimizer.config, holdout_repetitions=repetitions)

    def inject(phase, case, value):
        selected = phase == "candidate-holdout" and case == optimizer.dataset.holdout[0]
        return _quality_trial(
            value, behavior=not (selected and value.trial < behavior_loss),
            reader=not (selected and value.trial < reader_loss),
        )

    phases = _inject_phase_trials(optimizer, inject)
    source = _source(tmp_path)
    original = source.read_bytes()
    result = optimizer.run(source)

    total = 2 * repetitions
    reasons = tuple(
        f"overall {dimension} pass rate dropped by more than 5 percentage points"
        for dimension, loss in (("behavior", behavior_loss), ("reader", reader_loss))
        if 20 * loss > total
    )
    decision = "REJECT" if reasons else "ACCEPT"
    assert result.decision == decision, result.message
    assert result.gate is not None
    assert result.gate.reasons == reasons
    assert not result.gate.inconclusive
    assert phases == [
        "baseline-holdout", "baseline-search", "candidate-validation", "candidate-holdout",
    ]
    baseline, candidate = result.gate.baseline, result.gate.candidate
    assert baseline.total_trials == candidate.total_trials == total
    assert baseline.behavior_trial_passes == baseline.understanding_trial_passes == total
    assert candidate.behavior_trial_passes == total - behavior_loss
    assert candidate.understanding_trial_passes == total - reader_loss
    assert baseline.behavior_pass_rate == baseline.understanding_pass_rate == 1.0
    assert candidate.behavior_pass_rate == pytest.approx((total - behavior_loss) / total)
    assert candidate.understanding_pass_rate == pytest.approx((total - reader_loss) / total)
    assert baseline.behavior_passes == candidate.behavior_passes == 2
    assert baseline.understanding_passes == candidate.understanding_passes == 2
    assert candidate.critical_failures == 0
    folder = Path(result.run_directory)
    validation = json.loads((folder / "candidate-validation.json").read_text(encoding="utf-8"))
    assert len(validation) == 2
    assert all(v["behavior"]["passed"] and v["understanding"]["passed"] for v in validation)
    if repetitions == 9:
        # Pooling the two successful validation trials would wrongly allow this loss.
        assert 20 * max(behavior_loss, reader_loss) == total + len(validation)
        assert result.decision == "REJECT"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == decision
    published, report = write_outputs(result, tmp_path / "out")
    assert (published is not None) == (decision == "ACCEPT")
    assert f"Decision: {decision}" in report.read_text(encoding="utf-8")
    assert source.read_bytes() == original


@pytest.mark.parametrize("error_scope", ["evaluation", "behavior", "reader"])
def test_allowed_holdout_drop_cannot_hide_evaluation_errors(
    tmp_path: Path, error_scope: str,
) -> None:
    optimizer = _small_optimizer(tmp_path)
    optimizer.config = replace(optimizer.config, holdout_repetitions=10)
    error = "holdout judge unavailable"

    def inject(phase, case, value):
        selected = (
            phase == "candidate-holdout"
            and case == optimizer.dataset.holdout[0] and value.trial == 0
        )
        value = _quality_trial(value, behavior=not selected, reader=not selected)
        if not selected:
            return value
        if error_scope == "behavior":
            return replace(value, behavior=replace(value.behavior, error=error))
        if error_scope == "reader":
            return replace(value, understanding=replace(value.understanding, error=error))
        return replace(value, error=error)

    phases = _inject_phase_trials(optimizer, inject)
    result = optimizer.run(_source(tmp_path))

    assert result.decision == "INCONCLUSIVE", result.message
    assert phases[-1] == "candidate-holdout"
    assert result.gate is not None and result.gate.inconclusive
    assert result.gate.candidate.total_trials == 20
    assert result.gate.candidate.behavior_trial_passes == 19
    assert result.gate.candidate.understanding_trial_passes == 19
    assert result.gate.reasons == (f"{optimizer.dataset.holdout[0].id}: evaluation error",)
    assert result.gate.communication_reduction == 0.0
    assert result.candidate_body is None
    published, report = write_outputs(result, tmp_path / "out")
    assert published is None
    assert "Decision: INCONCLUSIVE" in report.read_text(encoding="utf-8")
    folder = Path(result.run_directory)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "INCONCLUSIVE"
    values = json.loads((folder / "candidate-holdout.json").read_text(encoding="utf-8"))
    assert len(values) == 20
    assert sum(bool(v["error"] or v["behavior"]["error"] or v["understanding"]["error"])
               for v in values) == 1


@pytest.mark.parametrize("dimension", ["behavior", "reader"])
def test_validation_per_case_floor_rejects_even_an_exact_five_point_drop(
    tmp_path: Path, dimension: str,
) -> None:
    optimizer = _small_optimizer(tmp_path, control=True)
    optimizer.config = replace(
        optimizer.config, split=(6, 20, 2),
        categories=(replace(QUICK_CATEGORIES[0], raw=28, retained=28),),
    )

    def inject(phase, case, value):
        selected = (
            phase == "candidate-validation" and case == optimizer.dataset.validation[0]
        )
        return _quality_trial(
            value, behavior=not (selected and dimension == "behavior"),
            reader=not (selected and dimension == "reader"),
        )

    phases = _inject_phase_trials(optimizer, inject)
    result = optimizer.run(_source(tmp_path))

    assert result.decision == "REJECT", result.message
    assert result.gate is None
    expected = f"{optimizer.dataset.validation[0].id}: no successful behavior/reader trial"
    assert result.details["validation_reasons"] == [expected]
    baseline = result.details["validation_baseline"]["cases"]
    candidate = result.details["validation_candidate"]["cases"]
    pass_key = "behavior_passes" if dimension == "behavior" else "understanding_passes"
    assert sum(c["trials"] for c in baseline) == sum(c["trials"] for c in candidate) == 20
    assert sum(c[pass_key] for c in baseline) == 20
    assert sum(c[pass_key] for c in candidate) == 19
    assert all(c["trials"] == 1 for c in candidate)
    assert phases == ["baseline-holdout", "baseline-search", "candidate-validation"]
    assert "concision_control" not in result.details
    folder = Path(result.run_directory)
    assert not (folder / "candidate-holdout.json").exists()
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["details"]["validation_reasons"] == [expected]
    published, report = write_outputs(result, tmp_path / "out")
    assert published is None
    assert expected in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("interrupt_at", ["target", "judge"])
def test_evaluation_journal_survives_budget_exhaustion(tmp_path: Path, interrupt_at: str) -> None:
    optimizer = _small_optimizer(tmp_path)
    optimizer.contract = BehaviorContract("한국어", "결과", (Rule("O1", "결과", "결과"),))
    case = EvaluationCase("case", "normal", "family", "결과?", {}, ("O1",), (), (),
                          (ReaderQuestion("what", "무엇?", "결과"),))
    journal = tmp_path / "trials.json"
    model = optimizer.task_model if interrupt_at == "target" else optimizer.strong_model
    complete = model.complete
    budget = CallBudget(1 if interrupt_at == "target" else 3)
    attempts = 0

    def limited(system, user):
        nonlocal attempts
        attempts += 1
        if budget.calls == budget.limit:
            # The first judged trial must be on disk before the next call is attempted.
            saved = json.loads(journal.read_text(encoding="utf-8"))
            assert len(saved) == 1
            assert saved[0]["trial"] == 0
        budget.claim()
        return complete(system, user)

    model.complete = limited
    with pytest.raises(BudgetExceeded):
        optimizer._evaluate("결과를 설명하세요.", (case,), 3, 0, 100, "test", journal)

    values = json.loads(journal.read_text(encoding="utf-8"))
    assert len(values) == 1
    assert values[0]["error"] == ""
    assert values[0]["behavior"]["checks"][0]["rule"] == "O1"
    assert values[0]["understanding"]["answers"][0]["response"] == "결과"
    trial_id = values[0]["details"]["trial_id"]
    run = json.loads((tmp_path / "cache" / "model-runs" / f"{trial_id}.json").read_text(encoding="utf-8"))
    assert run["trial_id"] == trial_id
    assert attempts == budget.limit + 1  # No retry or budget bypass.


@pytest.mark.parametrize("phase", [
    "baseline-holdout", "baseline-search", "candidate-validation", "candidate-holdout", "control-holdout",
])
def test_phase_budget_failure_preserves_evidence_and_control_isolation(tmp_path: Path, phase: str) -> None:
    optimizer = _small_optimizer(tmp_path, control=True)
    evaluate_phase = optimizer._evaluate
    judge = optimizer.evaluator.evaluate
    active_phase = None
    attempts = 0
    seen_phases = []
    compile_calls = []
    compile_candidate = optimizer._compile

    def track_compile(*args):
        dataset, baselines = args[2:4]
        assert set(baselines) == {case.id for case in (*dataset.train, *dataset.validation)}
        assert set(baselines).isdisjoint(case.id for case in dataset.holdout)
        compile_calls.append(tuple(seen_phases))
        return compile_candidate(*args)

    def track_phase(instructions, cases, repetitions, start, end, message, journal_path=None):
        nonlocal active_phase
        assert journal_path is not None
        active_phase = journal_path.stem
        seen_phases.append(active_phase)
        return evaluate_phase(instructions, cases, repetitions, start, end, message, journal_path)

    def limited_judge(contract, case, run):
        nonlocal attempts
        if active_phase == phase:
            attempts += 1
            if attempts == 2:
                raise BudgetExceeded("phase budget")
        return judge(contract, case, run)

    optimizer._compile = track_compile
    optimizer._evaluate = track_phase
    optimizer.evaluator.evaluate = limited_judge
    result = optimizer.run(_source(tmp_path))

    assert attempts == 2
    folder = Path(result.run_directory)
    values = json.loads((folder / f"{phase}.json").read_text(encoding="utf-8"))
    assert len(values) == 1
    assert values[0]["trial"] == 0
    assert values[0]["details"]["trial_id"]
    assert seen_phases[-1] == phase
    assert len(seen_phases) == len(set(seen_phases))
    assert all(phases == ("baseline-holdout", "baseline-search") for phases in compile_calls)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if phase == "control-holdout":
        assert len(compile_calls) == 1
        assert result.decision == summary["decision"] == "ACCEPT"
        assert result.candidate_body == "결과를 설명하세요."
        assert result.details["concision_control"] == {
            "status": "INCONCLUSIVE", "reason": "phase budget", "selected": False,
        }
    else:
        assert result.decision == summary["decision"] == "INCONCLUSIVE"
        assert result.candidate_body is None
        assert "phase budget" in result.message
        assert "concision_control" not in result.details
