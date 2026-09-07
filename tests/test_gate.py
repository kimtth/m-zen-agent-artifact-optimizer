from dataclasses import replace

import pytest

from zen.domain.core import Aggregate, CaseOutcome
from zen.pipeline.gate import decide, quality_reasons, quality_regressed


def summary(
    behavior: int = 20, reader: int = 20, trials: int = 20,
    artifact_tokens: int = 900,
) -> Aggregate:
    return Aggregate(
        artifact_tokens, int(behavior > trials // 2), int(reader > trials // 2),
        0, 100, 100,
        (CaseOutcome("A", trials, behavior, reader, {"O1": behavior}, {"what": reader}),),
    )


def test_shorter_quality_preserving_candidate_is_accepted() -> None:
    baseline = summary()
    candidate = summary(artifact_tokens=650)
    assert decide(baseline, candidate).accepted


def test_shorter_candidate_with_behavior_regression_is_rejected() -> None:
    baseline = summary()
    candidate = summary(behavior=18, artifact_tokens=500)
    decision = decide(baseline, candidate)
    assert not decision.accepted
    assert "overall behavior pass rate dropped by more than 5 percentage points" in decision.reasons


def test_default_minimum_communication_reduction_is_three_percent() -> None:
    baseline = summary()

    assert decide(baseline, summary(artifact_tokens=870)).accepted

    decision = decide(baseline, summary(artifact_tokens=875))
    assert not decision.accepted
    assert any("below 3.0%" in r for r in decision.reasons)


@pytest.mark.parametrize("dimension", ["behavior", "reader"])
@pytest.mark.parametrize(("trials", "before", "after", "accepted"), [
    (20, 20, 19, True),
    (20, 19, 18, True),
    (20, 10, 9, True),
    (20, 20, 18, False),
    (100, 100, 94, False),
    (100, 50, 44, False),
    (21, 21, 20, True),
    (19, 19, 18, False),
    (3, 3, 2, False),
    (2, 2, 1, False),
    (1, 1, 0, False),
])
def test_independent_absolute_rate_boundary(
    dimension: str, trials: int, before: int, after: int, accepted: bool,
) -> None:
    baseline_counts = {"behavior": trials, "reader": trials, dimension: before}
    candidate_counts = {**baseline_counts, dimension: after}
    baseline = summary(**baseline_counts, trials=trials)
    candidate = summary(**candidate_counts, trials=trials, artifact_tokens=650)
    decision = decide(baseline, candidate)
    assert decision.accepted is accepted
    assert not decision.inconclusive
    assert bool(quality_reasons(baseline, candidate)) is not accepted
    assert quality_regressed(baseline, candidate) is not accepted
    if not accepted:
        assert (
            f"overall {dimension} pass rate dropped by more than 5 percentage points"
            in decision.reasons
        )


@pytest.mark.parametrize(("behavior", "reader", "rejected"), [
    (19, 19, ()),
    (18, 19, ("behavior",)),
    (19, 18, ("reader",)),
    (18, 18, ("behavior", "reader")),
])
def test_each_dimension_has_its_own_allowance(
    behavior: int, reader: int, rejected: tuple[str, ...],
) -> None:
    baseline = summary()
    candidate = summary(behavior, reader, artifact_tokens=650)
    assert quality_reasons(baseline, candidate) == [
        f"overall {dimension} pass rate dropped by more than 5 percentage points"
        for dimension in rejected
    ]
    assert decide(baseline, candidate).accepted is (not rejected)


@pytest.mark.parametrize(("behavior", "reader", "dimension"), [
    (12, 8, "reader"), (8, 12, "behavior"),
])
def test_improvement_in_one_dimension_does_not_offset_the_other(
    behavior: int, reader: int, dimension: str,
) -> None:
    baseline = summary(10, 10)
    candidate = summary(behavior, reader, artifact_tokens=650)
    assert quality_reasons(baseline, candidate) == [
        f"overall {dimension} pass rate dropped by more than 5 percentage points"
    ]


@pytest.mark.parametrize("dimension", ["behavior", "reader"])
def test_per_case_success_floor_still_blocks_an_allowed_overall_drop(dimension: str) -> None:
    cases = tuple(CaseOutcome(str(i), 1, 1, 1, {}, {}) for i in range(20))
    baseline = replace(summary(), cases=cases)
    changed = replace(cases[0], **{
        "behavior_passes" if dimension == "behavior" else "understanding_passes": 0,
    })
    candidate = replace(baseline, artifact_tokens=650, cases=(changed, *cases[1:]))
    assert quality_reasons(baseline, candidate) == ["0: no successful behavior/reader trial"]
    assert not decide(baseline, candidate).accepted


@pytest.mark.parametrize(("field", "reason"), [
    ("artifact_tokens", "artifact token count increased"),
    ("median_output_tokens", "median output token count increased"),
    ("median_understanding_tokens", "median understanding-token count increased"),
])
def test_token_nonincrease_guards_remain(field: str, reason: str) -> None:
    baseline = summary()
    candidate = replace(summary(19, 19, artifact_tokens=650), **{
        field: getattr(baseline, field) + 1,
    })
    decision = decide(baseline, candidate)
    assert not decision.accepted
    assert not decision.inconclusive
    assert reason in decision.reasons


def test_legacy_majority_counts_never_supply_trial_evidence() -> None:
    baseline = Aggregate(900, 10, 10, 0, 100, 100)
    candidate = replace(baseline, artifact_tokens=650)
    assert baseline.total_trials == 0
    assert baseline.behavior_trial_passes == baseline.understanding_trial_passes == 0
    assert baseline.behavior_pass_rate is baseline.understanding_pass_rate is None
    decision = decide(baseline, candidate)
    assert not decision.accepted
    assert decision.inconclusive
    assert decision.communication_reduction == 0
    assert quality_reasons(baseline, candidate) == ["no evaluation trials"]


def test_trial_rates_are_weighted_by_trials_not_majority_counts_or_case_rates() -> None:
    value = replace(summary(), behavior_passes=999, understanding_passes=888, cases=(
        CaseOutcome("A", 2, 1, 2, {}, {}),
        CaseOutcome("B", 18, 18, 9, {}, {}),
    ))
    assert value.total_trials == 20
    assert value.behavior_trial_passes == 19
    assert value.understanding_trial_passes == 11
    assert value.behavior_pass_rate == 0.95
    assert value.understanding_pass_rate == 0.55
    assert value.behavior_passes == 999
    assert value.understanding_passes == 888


@pytest.mark.parametrize("invalid_cases", [
    (),
    (CaseOutcome("A", 0, 0, 0, {}, {}),),
    (CaseOutcome("A", 1, 2, 1, {}, {}),),
    (CaseOutcome("A", 1, 1, -1, {}, {}),),
    (CaseOutcome("A", 1, 1, 1, {}, {}),) * 2,
])
def test_empty_or_invalid_case_outcomes_are_inconclusive(
    invalid_cases: tuple[CaseOutcome, ...],
) -> None:
    baseline = replace(summary(), cases=invalid_cases)
    candidate = replace(baseline, artifact_tokens=650)
    decision = decide(baseline, candidate)
    assert not decision.accepted
    assert decision.inconclusive
    assert decision.communication_reduction == 0
