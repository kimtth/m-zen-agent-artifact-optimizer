from zen.domain.core import Aggregate
from zen.pipeline.gate import decide


def test_shorter_quality_preserving_candidate_is_accepted() -> None:
    baseline = Aggregate(900, 10, 10, 0, 200, 150)
    candidate = Aggregate(650, 10, 10, 0, 150, 100)
    assert decide(baseline, candidate).accepted


def test_shorter_candidate_with_behavior_regression_is_rejected() -> None:
    baseline = Aggregate(900, 10, 10, 0, 200, 150)
    candidate = Aggregate(500, 9, 10, 0, 100, 80)
    decision = decide(baseline, candidate)
    assert not decision.accepted
    assert "behavior pass count regressed" in decision.reasons


def test_default_minimum_communication_reduction_is_three_percent() -> None:
    baseline = Aggregate(900, 10, 10, 0, 100, 100)

    assert decide(baseline, Aggregate(870, 10, 10, 0, 100, 100)).accepted

    decision = decide(baseline, Aggregate(875, 10, 10, 0, 100, 100))
    assert not decision.accepted
    assert any("below 3.0%" in r for r in decision.reasons)
