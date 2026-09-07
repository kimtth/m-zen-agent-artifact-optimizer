"""Aggregate aligned trials and gate independent behavior and reader pass rates."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from ..domain.core import Aggregate, CaseEvaluation, CaseOutcome, GateDecision


def aggregate(
    evaluations: list[CaseEvaluation],
    artifact_tokens: int,
    repetitions: int,
    expected_case_ids: tuple[str, ...] | None = None,
) -> Aggregate:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    grouped: dict[str, list[CaseEvaluation]] = defaultdict(list)
    errors = []
    for item in evaluations:
        grouped[item.case_id].append(item)
        if item.error or item.behavior.error or item.understanding.error:
            errors.append(f"{item.case_id}: evaluation error")
    if not grouped:
        errors.append("no evaluation trials")
    if expected_case_ids is not None and (
        len(set(expected_case_ids)) != len(expected_case_ids)
        or set(grouped) != set(expected_case_ids)
    ):
        errors.append("case coverage does not match the frozen dataset")
    cases = []
    for case_id, runs in sorted(grouped.items()):
        if len(runs) != repetitions or {r.trial for r in runs} != set(range(repetitions)):
            errors.append(f"{case_id}: missing or duplicate trials")
        rule_sets = [tuple(c.rule for c in r.behavior.checks) for r in runs]
        reader_sets = [tuple(a.question for a in r.understanding.answers) for r in runs]
        for label, sets in (("rules", rule_sets), ("reader questions", reader_sets)):
            if any(len(s) != len(set(s)) or set(s) != set(sets[0]) for s in sets):
                errors.append(f"{case_id}: inconsistent {label}")
        rules = Counter({key: 0 for key in rule_sets[0]})
        readers = Counter({key: 0 for key in reader_sets[0]})
        for run in runs:
            rules.update(c.rule for c in run.behavior.checks if c.passed)
            readers.update(a.question for a in run.understanding.answers if a.correct)
        cases.append(CaseOutcome(
            case_id, len(runs), sum(r.behavior.passed for r in runs),
            sum(r.understanding.passed for r in runs), dict(rules), dict(readers),
        ))
    valid = [r for r in evaluations if not (r.error or r.behavior.error or r.understanding.error)]
    return Aggregate(
        artifact_tokens,
        sum(c.behavior_passes > c.trials // 2 for c in cases),
        sum(c.understanding_passes > c.trials // 2 for c in cases),
        sum(r.behavior.critical_failure for r in valid),
        median(r.output_tokens for r in valid) if valid else 0,
        median(r.understanding.tokens for r in valid) if valid else 0,
        tuple(cases), tuple(dict.fromkeys(errors)),
    )


def evidence_problems(baseline: Aggregate, candidate: Aggregate) -> list[str]:
    problems = [*baseline.evidence_errors, *candidate.evidence_errors]
    for summary in (baseline, candidate):
        if not summary.cases:
            problems.append("no evaluation trials")
        if len({c.case_id for c in summary.cases}) != len(summary.cases):
            problems.append("duplicate case outcomes")
        for case in summary.cases:
            if (case.trials < 1 or not 0 <= case.behavior_passes <= case.trials
                    or not 0 <= case.understanding_passes <= case.trials):
                problems.append(f"{case.case_id}: invalid trial counts")
    before = {c.case_id: c for c in baseline.cases}
    after = {c.case_id: c for c in candidate.cases}
    if before.keys() != after.keys():
        problems.append("baseline and candidate case sets differ")
    for key in before.keys() & after.keys():
        left, right = before[key], after[key]
        if (left.trials != right.trials or left.rule_passes.keys() != right.rule_passes.keys()
                or left.reader_passes.keys() != right.reader_passes.keys()):
            problems.append(f"{key}: trial or requirement coverage differs")
    return sorted(set(problems))


def quality_reasons(baseline: Aggregate, candidate: Aggregate) -> list[str]:
    problems = evidence_problems(baseline, candidate)
    if problems:
        return problems
    reasons = []
    if candidate.critical_failures:
        reasons.append(f"observed {candidate.critical_failures} critical failure(s)")
    for label, before, after in (
        ("behavior", baseline.behavior_trial_passes, candidate.behavior_trial_passes),
        ("reader", baseline.understanding_trial_passes, candidate.understanding_trial_passes),
    ):
        # Compare the absolute drop with 1/20 exactly, without float subtraction.
        if 20 * (before * candidate.total_trials - after * baseline.total_trials) > (
            baseline.total_trials * candidate.total_trials
        ):
            reasons.append(f"overall {label} pass rate dropped by more than 5 percentage points")
    for right in candidate.cases:
        if not right.behavior_passes or not right.understanding_passes:
            reasons.append(f"{right.case_id}: no successful behavior/reader trial")
    return reasons


def decide(baseline: Aggregate, candidate: Aggregate, minimum_reduction: float = 0.03) -> GateDecision:
    problems = evidence_problems(baseline, candidate)
    if problems:
        return GateDecision(False, tuple(problems), baseline, candidate, 0.0, True)
    reasons = quality_reasons(baseline, candidate)
    if candidate.artifact_tokens > baseline.artifact_tokens:
        reasons.append("artifact token count increased")
    if candidate.median_output_tokens > baseline.median_output_tokens:
        reasons.append("median output token count increased")
    if candidate.median_understanding_tokens > baseline.median_understanding_tokens:
        reasons.append("median understanding-token count increased")
    reduction = (
        (baseline.communication_tokens - candidate.communication_tokens) / baseline.communication_tokens
        if baseline.communication_tokens else 0.0
    )
    if reduction < minimum_reduction:
        reasons.append(f"communication-token reduction {reduction:.1%} is below {minimum_reduction:.1%}")
    return GateDecision(not reasons, tuple(reasons), baseline, candidate, reduction)


def quality_regressed(baseline: Aggregate, candidate: Aggregate) -> bool:
    return bool(quality_reasons(baseline, candidate))