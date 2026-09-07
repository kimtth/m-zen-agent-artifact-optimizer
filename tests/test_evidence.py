from dataclasses import replace
from pathlib import Path

import pytest

from zen.domain.core import (
    BehaviorResult,
    CaseEvaluation,
    Check,
    EvaluationCase,
    UnderstandingAnswer,
    UnderstandingResult,
)
from zen.pipeline.gate import aggregate, decide, quality_reasons
from zen.runtime.harness import RunCache, Runner
from zen.runtime.lm import FunctionModel


def evaluation(case: str, passed: bool = True, critical: bool = False) -> CaseEvaluation:
    return CaseEvaluation(
        case,
        BehaviorResult(passed, critical, (Check("O1", passed, "preference", "fact", "check"),)),
        UnderstandingResult(True, 1.0, 10, (UnderstandingAnswer("what", True, "fact"),)),
        20,
        "feedback",
    )


def test_swapped_case_failures_cannot_cancel_each_other() -> None:
    baseline = aggregate([evaluation("A", False), evaluation("B")], 100, 1)
    candidate = aggregate([evaluation("A"), evaluation("B", False)], 50, 1)
    decision = decide(baseline, candidate)
    assert not decision.accepted
    assert any("B" in reason for reason in decision.reasons)


def test_incomplete_repetitions_are_not_acceptance_evidence() -> None:
    baseline = aggregate([evaluation("A")], 100, 3)
    candidate = aggregate([evaluation("A")], 50, 3)
    decision = decide(baseline, candidate)
    assert not decision.accepted
    assert decision.inconclusive


def test_any_critical_failure_blocks_acceptance_even_if_baseline_failed() -> None:
    baseline = aggregate([evaluation("A", False, True)], 100, 1)
    candidate = aggregate([evaluation("A", False, True)], 50, 1)
    assert not decide(baseline, candidate).accepted


def test_different_case_sets_are_inconclusive() -> None:
    baseline = aggregate([evaluation("A")], 100, 1)
    candidate = aggregate([evaluation("B")], 50, 1)
    assert decide(baseline, candidate).inconclusive


def test_run_cache_uses_complete_case_content(tmp_path: Path) -> None:
    calls = []

    def answer(system: str, user: str) -> str:
        calls.append(user)
        return "[[ ## answer ## ]]\nfact\n[[ ## completed ## ]]"

    case = EvaluationCase("same", "normal", "family", "first", {}, (), (), (), ())
    runner = Runner(FunctionModel(answer), RunCache(tmp_path))
    runner.run("Explain.", case)
    runner.run("Explain.", replace(case, inquiry="second"))
    assert len(calls) == 2


def test_rule_regression_is_diagnostic_with_overall_quality_and_case_floor_preserved() -> None:
    before = evaluation("A", False)
    after = replace(
        before,
        behavior=BehaviorResult(False, False, (
            Check("O1", True, "preference", "fact", "fixed"),
            Check("O2", False, "preference", None, "lost"),
        )),
    )
    before = replace(before, behavior=BehaviorResult(False, False, (
        Check("O1", False, "preference", None, "missing"),
        Check("O2", True, "preference", "fact", "present"),
    )))
    successful = replace(
        before, trial=1,
        behavior=replace(before.behavior, passed=True, checks=tuple(
            replace(check, passed=True) for check in before.behavior.checks
        )),
    )
    baseline = aggregate([before, successful], 100, 2)
    candidate = aggregate([after, successful], 50, 2)
    assert baseline.cases[0].rule_passes["O2"] == 2
    assert candidate.cases[0].rule_passes["O2"] == 1
    assert baseline.behavior_pass_rate == candidate.behavior_pass_rate == 0.5
    assert decide(baseline, candidate).accepted


def test_minority_critical_failure_is_not_erased() -> None:
    baseline = [replace(evaluation("A"), trial=i) for i in range(3)]
    candidate = [replace(evaluation("A", i != 1, i == 1), trial=i) for i in range(3)]
    summary = aggregate(candidate, 50, 3)
    assert summary.critical_failures == 1
    assert not decide(aggregate(baseline, 100, 3), summary).accepted


def test_duplicate_trial_indices_are_inconclusive() -> None:
    values = [evaluation("A")] * 3
    assert decide(aggregate(values, 100, 3), aggregate(values, 50, 3)).inconclusive


def test_errors_are_excluded_from_token_medians_and_invalidate_gate() -> None:
    valid = replace(evaluation("A"), trial=0)
    failed = replace(valid, trial=1, error="network", output_tokens=0)
    summary = aggregate([valid, failed], 50, 2)
    assert summary.median_output_tokens == 20
    assert decide(summary, summary).inconclusive


def test_missing_frozen_case_is_inconclusive() -> None:
    summary = aggregate([evaluation("A")], 100, 1, ("A", "B"))
    assert decide(summary, summary).inconclusive


def test_case_majority_rule_and_question_counts_may_decrease_at_exact_boundary() -> None:
    def trials(first_case_passes: int) -> list[CaseEvaluation]:
        values = []
        for case, passes in (("A", first_case_passes), ("B", 10)):
            for trial in range(10):
                passed = trial < passes
                value = evaluation(case, passed)
                values.append(replace(
                    value, trial=trial,
                    understanding=replace(
                        value.understanding, passed=passed, accuracy=float(passed),
                        answers=(UnderstandingAnswer("what", passed, "fact"),),
                    ),
                ))
        return values

    baseline = aggregate(trials(6), 100, 10)
    candidate = aggregate(trials(5), 50, 10)
    assert baseline.behavior_passes == baseline.understanding_passes == 2
    assert candidate.behavior_passes == candidate.understanding_passes == 1
    assert baseline.total_trials == candidate.total_trials == 20
    assert baseline.behavior_trial_passes == baseline.understanding_trial_passes == 16
    assert candidate.behavior_trial_passes == candidate.understanding_trial_passes == 15
    assert baseline.behavior_pass_rate == baseline.understanding_pass_rate == 0.8
    assert candidate.behavior_pass_rate == candidate.understanding_pass_rate == 0.75
    assert baseline.cases[0].rule_passes == {"O1": 6}
    assert candidate.cases[0].rule_passes == {"O1": 5}
    assert baseline.cases[0].reader_passes == {"what": 6}
    assert candidate.cases[0].reader_passes == {"what": 5}
    assert not quality_reasons(baseline, candidate)
    assert decide(baseline, candidate).accepted


def test_per_case_behavior_and_reader_success_can_be_different_trials() -> None:
    behavior_only = evaluation("A")
    behavior_only = replace(
        behavior_only,
        understanding=replace(
            behavior_only.understanding, passed=False, accuracy=0.0,
            answers=(UnderstandingAnswer("what", False, None),),
        ),
    )
    reader_only = replace(evaluation("A", False), trial=1)
    baseline = aggregate([behavior_only, reader_only], 100, 2)
    candidate = aggregate([behavior_only, reader_only], 50, 2)
    assert candidate.behavior_passes == candidate.understanding_passes == 0
    assert candidate.behavior_pass_rate == candidate.understanding_pass_rate == 0.5
    assert decide(baseline, candidate).accepted


def test_case_losses_can_be_offset_within_a_dimension_when_each_case_has_success() -> None:
    baseline = [
        replace(evaluation(case, case != "A" or trial != 0), trial=trial)
        for case in ("A", "B") for trial in range(3)
    ]
    candidate = [
        replace(evaluation(case, case != "B" or trial != 0), trial=trial)
        for case in ("A", "B") for trial in range(3)
    ]
    assert decide(aggregate(baseline, 100, 3), aggregate(candidate, 50, 3)).accepted


def test_one_critical_failure_blocks_an_otherwise_allowed_five_point_drop() -> None:
    baseline = [replace(evaluation("A"), trial=i) for i in range(20)]
    candidate = [replace(evaluation("A", i != 0, i == 0), trial=i) for i in range(20)]
    before, after = aggregate(baseline, 100, 20), aggregate(candidate, 50, 20)
    assert quality_reasons(before, after) == ["observed 1 critical failure(s)"]
    assert not decide(before, after).accepted


@pytest.mark.parametrize("stage", ["evaluation", "behavior", "understanding"])
@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_any_stage_error_on_either_side_invalidates_rates_and_gate(stage: str, side: str) -> None:
    value = evaluation("A")
    failed = (
        replace(value, error="network") if stage == "evaluation"
        else replace(value, **{stage: replace(getattr(value, stage), error="network")})
    )
    complete = aggregate([value], 100, 1)
    incomplete = aggregate([failed], 50, 1)
    assert incomplete.behavior_pass_rate is incomplete.understanding_pass_rate is None
    before, after = (incomplete, complete) if side == "baseline" else (complete, incomplete)
    decision = decide(before, after)
    assert not decision.accepted
    assert decision.inconclusive
    assert decision.communication_reduction == 0
    assert quality_reasons(before, after)


def test_empty_evaluations_never_have_pass_rates_or_savings() -> None:
    before, after = aggregate([], 100, 1), aggregate([], 50, 1)
    assert before.total_trials == after.total_trials == 0
    assert before.behavior_pass_rate is before.understanding_pass_rate is None
    decision = decide(before, after)
    assert not decision.accepted
    assert decision.inconclusive
    assert decision.communication_reduction == 0


@pytest.mark.parametrize("coverage", ["trials", "rules", "questions"])
def test_unequal_coverage_remains_inconclusive(coverage: str) -> None:
    value = evaluation("A")
    candidate_values = [value]
    repetitions = 1
    if coverage == "trials":
        candidate_values.append(replace(value, trial=1))
        repetitions = 2
    elif coverage == "rules":
        candidate_values = [replace(value, behavior=replace(
            value.behavior, checks=(replace(value.behavior.checks[0], rule="O2"),),
        ))]
    else:
        candidate_values = [replace(value, understanding=replace(
            value.understanding, answers=(UnderstandingAnswer("different", True, "fact"),),
        ))]
    before = aggregate([value], 100, 1)
    after = aggregate(candidate_values, 50, repetitions)
    assert not before.evidence_errors
    assert not after.evidence_errors
    decision = decide(before, after)
    assert not decision.accepted
    assert decision.inconclusive
    assert decision.communication_reduction == 0


@pytest.mark.parametrize("stage", ["behavior", "understanding"])
@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_inconsistent_requirements_within_trials_are_inconclusive(stage: str, change: str) -> None:
    first = evaluation("A")
    field = "checks" if stage == "behavior" else "answers"
    result = getattr(first, stage)
    values = getattr(result, field) * 2 if change == "duplicate" else ()
    second = replace(first, trial=1, **{stage: replace(result, **{field: values})})
    summary = aggregate([first, second], 50, 2)
    assert summary.behavior_pass_rate is summary.understanding_pass_rate is None
    assert decide(summary, summary).inconclusive