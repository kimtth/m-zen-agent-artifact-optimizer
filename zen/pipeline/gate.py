"""Step 6: aggregate repeated runs and apply the acceptance contract."""

from __future__ import annotations

from collections import defaultdict
from statistics import median

from ..domain.core import Aggregate, CaseEvaluation, GateDecision


def aggregate(
    evaluations: list[CaseEvaluation],
    artifact_tokens: int,
    repetitions: int,
) -> Aggregate:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    grouped: dict[str, list[CaseEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.case_id].append(evaluation)
    behavior_passes = 0
    understanding_passes = 0
    critical_failures = 0
    for case_runs in grouped.values():
        required = len(case_runs) // 2 + 1
        behavior_passes += sum(item.behavior.passed for item in case_runs) >= required
        understanding_passes += sum(item.understanding.passed for item in case_runs) >= required
        critical_failures += sum(item.behavior.critical_failure for item in case_runs) >= required
    return Aggregate(
        artifact_tokens=artifact_tokens,
        behavior_passes=behavior_passes,
        understanding_passes=understanding_passes,
        critical_failures=critical_failures,
        median_output_tokens=median(item.output_tokens for item in evaluations),
        median_understanding_tokens=median(
            item.understanding.tokens for item in evaluations
        ),
    )


def decide(
    baseline: Aggregate,
    candidate: Aggregate,
    minimum_reduction: float = 0.03,
) -> GateDecision:
    reasons = []
    new_critical = max(0, candidate.critical_failures - baseline.critical_failures)
    if new_critical:
        reasons.append(f"introduced {new_critical} critical failure(s)")
    if candidate.behavior_passes < baseline.behavior_passes:
        reasons.append("behavior pass count regressed")
    if candidate.understanding_passes < baseline.understanding_passes:
        reasons.append("reader understanding regressed")
    if candidate.artifact_tokens > baseline.artifact_tokens:
        reasons.append("artifact token count increased")
    if candidate.median_output_tokens > baseline.median_output_tokens:
        reasons.append("median output token count increased")
    if candidate.median_understanding_tokens > baseline.median_understanding_tokens:
        reasons.append("median understanding-token count increased")
    reduction = 0.0
    if baseline.communication_tokens:
        reduction = (
            baseline.communication_tokens - candidate.communication_tokens
        ) / baseline.communication_tokens
    if reduction < minimum_reduction:
        reasons.append(
            f"communication-token reduction {reduction:.1%} is below {minimum_reduction:.1%}"
        )
    return GateDecision(not reasons, tuple(reasons), baseline, candidate, reduction)


def quality_regressed(baseline: Aggregate, candidate: Aggregate) -> bool:
    return (
        candidate.critical_failures > baseline.critical_failures
        or candidate.behavior_passes < baseline.behavior_passes
        or candidate.understanding_passes < baseline.understanding_passes
    )
