from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_optimize import (
    _inject_phase_trials,
    _quality_trial,
    _small_optimizer,
    _source,
)

from zen.domain.core import Aggregate, CaseOutcome, OptimizationResult
from zen.optimization.report import render_report
from zen.pipeline.gate import decide


def _stats(*, candidate: bool = False) -> Aggregate:
    cases = tuple(
        CaseOutcome(
            str(index), 3,
            passes := 1 if candidate and index == 0 else 2 if candidate and index == 1 else 3,
            passes, {"quality": passes}, {"reader": passes},
        )
        for index in range(20)
    )
    majority = sum(case.behavior_passes > 1 for case in cases)
    return Aggregate(80 if candidate else 100, majority, majority, 0, 10, 5, cases)


def _result(baseline: Aggregate, candidate: Aggregate) -> OptimizationResult:
    gate = decide(baseline, candidate)
    return OptimizationResult(
        "INCONCLUSIVE" if gate.inconclusive else "ACCEPT" if gate.accepted else "REJECT",
        "AGENTS.md", "Be brief." if gate.accepted else None, None, None, gate, 10, "",
    )


def test_report_accepts_exact_five_points_for_both_rates_and_labels_majorities() -> None:
    result = _result(_stats(), _stats(candidate=True))

    report = render_report(result)

    assert "Decision: ACCEPT" in report
    assert "bounded observed quality-loss allowance" in report
    assert "Holdout (baseline → candidate):" in report
    for dimension in ("Behavior", "Reader"):
        assert (
            f"{dimension} trial pass rate: 60/60 (100.0%) → 57/60 (95.0%); "
            "change -5.0 percentage points."
        ) in report
        assert f"{dimension} majority-passing cases (diagnostic only): 20 → 19" in report
    assert "Behavior passes:" not in report
    assert "5 absolute percentage points" in report
    assert "independently, separately in validation and holdout; never pooled" in report
    assert "floor(N/20) net lost passes" in report
    assert "per-case and critical constraints can be stricter" in report
    assert "not human comprehension or a statistical guarantee" in report
    assert "no per-case/rule pass loss" not in report


@pytest.mark.parametrize("path", [
    r"C:\Users\Example User\private\result.md",
    "/home/example/private/result.md",
    r"\\server\share\result.md",
    "./out/result.md", "out/result.md",
])
def test_gepa_reports_omit_paths_from_diagnostics(path: str) -> None:
    result = OptimizationResult(
        "INCONCLUSIVE", path, None, None, None, None, 0, "/private/run",
        f"Unable to read '{path}': permission denied.",
    )
    report = render_report(result)
    assert path not in report
    assert "permission denied" in report
    assert result.message == f"Unable to read '{path}': permission denied."


def test_rejection_does_not_imply_all_quality_loss_is_forbidden() -> None:
    candidate = _stats(candidate=True)
    cases = (*candidate.cases[:-1], replace(candidate.cases[-1], behavior_passes=2))
    result = _result(_stats(), replace(candidate, cases=cases))

    report = render_report(result)

    assert "Decision: REJECT" in report
    assert "overall behavior pass rate dropped by more than 5 percentage points" in report
    assert "quality-loss allowance" in report
    assert "when quality or the required reduction regresses" not in report


@pytest.mark.parametrize("failure", ["missing", "unaligned", "coverage", "error"])
def test_invalid_holdout_rates_and_savings_are_unavailable(
    tmp_path: Path, failure: str,
) -> None:
    baseline, candidate = _stats(), _stats(candidate=True)
    if failure == "missing":
        candidate = replace(candidate, cases=())
    elif failure == "unaligned":
        candidate = replace(candidate, cases=candidate.cases[:-1])
    elif failure == "coverage":
        candidate = replace(candidate, cases=(
            replace(candidate.cases[0], reader_passes={"different": 1}), *candidate.cases[1:],
        ))
    else:
        candidate = replace(candidate, evidence_errors=("reader failed",))
    result = _result(baseline, candidate)

    report = render_report(result)
    optimizer = _small_optimizer(tmp_path)
    optimizer._save_summary(tmp_path, result)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert "Decision: INCONCLUSIVE" in report
    assert "trial pass rates and token comparison: unavailable" in report
    assert "Token savings: unavailable" in report
    assert "100.0%" not in report
    assert "95.0%" not in report
    assert "Artifact tokens:" not in report
    assert "tokens fell by" not in report
    assert saved["gate"]["communication_reduction"] is None
    for side in ("baseline", "candidate"):
        stats = saved["gate"][side]
        assert stats["behavior_pass_rate"] is None
        assert stats["understanding_pass_rate"] is None
        assert "behavior_trial_passes" in stats
        assert "total_trials" in stats


def _assert_serialized_rates(stats: dict, trials: int, behavior: int, reader: int) -> None:
    assert stats["total_trials"] == trials
    assert stats["behavior_trial_passes"] == behavior
    assert stats["understanding_trial_passes"] == reader
    assert stats["behavior_pass_rate"] == behavior / trials
    assert stats["understanding_pass_rate"] == reader / trials
    assert "behavior_passes" in stats
    assert "understanding_passes" in stats


def test_success_serializes_validation_holdout_control_and_policy(tmp_path: Path) -> None:
    optimizer = _small_optimizer(tmp_path, control=True)
    optimizer.config = replace(optimizer.config, holdout_repetitions=10)

    def inject(phase, case, value):
        loss = phase == "candidate-holdout" and case == optimizer.dataset.holdout[0]
        return _quality_trial(
            value, behavior=not (loss and value.trial == 0),
            reader=not (loss and value.trial == 0),
        )

    _inject_phase_trials(optimizer, inject)
    result = optimizer.run(_source(tmp_path))
    folder = Path(result.run_directory)
    saved = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    experiment = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))

    assert result.decision == saved["decision"] == "ACCEPT", result.message
    _assert_serialized_rates(saved["gate"]["baseline"], 20, 20, 20)
    _assert_serialized_rates(saved["gate"]["candidate"], 20, 19, 19)
    for side in ("baseline", "candidate"):
        _assert_serialized_rates(saved["details"][f"validation_{side}"], 2, 2, 2)
    assert saved["details"]["validation_reasons"] == []
    control = saved["details"]["concision_control"]
    assert control["selected"] is False
    _assert_serialized_rates(control["aggregate"], 20, 20, 20)
    assert saved["details"] == json.loads(json.dumps(result.details))
    assert "5 absolute percentage points" in experiment["gate"]
    assert "validation and holdout (not pooled)" in experiment["gate"]
    assert "no observed candidate critical failures" in experiment["gate"]
    assert "per candidate case" in experiment["gate"]
    assert "tokens must not increase" in experiment["gate"]
    assert "reduction >= 3.0%" in experiment["gate"]
    assert "no per-case/rule pass loss" not in experiment["gate"]
    assert "floor(N/20)" in experiment["quality_rate_semantics"]
    report = render_report(result)
    assert "Validation (baseline → candidate):" in report
    assert "Holdout (baseline → candidate):" in report
    assert "Control majority-passing cases (diagnostic only), behavior / reader: 2 / 2" in report
    assert "never selected" in report


@pytest.mark.parametrize("failure", ["quality", "error", "unaligned"])
def test_validation_failure_preserves_reasons_and_serialized_evidence(
    tmp_path: Path, failure: str,
) -> None:
    optimizer = _small_optimizer(tmp_path, control=True)
    evaluate = optimizer._evaluate

    def fail_validation(instructions, cases, repetitions, start, end, message, journal_path=None):
        values = evaluate(instructions, cases, repetitions, start, end, message, journal_path)
        if failure == "quality":
            values = [_quality_trial(value) for value in values]
        if journal_path.stem == "candidate-validation":
            if failure == "unaligned":
                return values[:-1]
            if failure == "error":
                return [replace(value, error="judge unavailable") for value in values]
            return [
                replace(value, understanding=replace(value.understanding, passed=False))
                for value in values
            ]
        return values

    optimizer._evaluate = fail_validation
    result = optimizer.run(_source(tmp_path))
    saved = json.loads((Path(result.run_directory) / "summary.json").read_text(encoding="utf-8"))
    report = render_report(result)

    assert result.gate is None
    assert "Validation reasons:" in report
    assert "Validation (baseline → candidate):" in report
    for reason in result.details["validation_reasons"]:
        assert reason in report
    assert saved["details"]["validation_reasons"] == result.details["validation_reasons"]
    assert "concision_control" not in saved["details"]
    if failure == "quality":
        assert "Decision: REJECT" in report
        assert "overall reader pass rate dropped by more than 5 percentage points" in report
        _assert_serialized_rates(saved["details"]["validation_candidate"], 2, 2, 0)
    else:
        assert "Decision: INCONCLUSIVE" in report
        assert "Token savings: unavailable" in report
        assert "100.0%" not in report
        for side in ("baseline", "candidate"):
            stats = saved["details"][f"validation_{side}"]
            assert stats["behavior_pass_rate"] is None
            assert stats["understanding_pass_rate"] is None


def test_summary_includes_details_before_outer_run_merge(tmp_path: Path) -> None:
    optimizer = _small_optimizer(tmp_path)
    optimizer.details = {"validation_reasons": [], "concision_control": {"selected": False}}
    result = _result(_stats(), _stats(candidate=True))
    result.details = {"result_detail": "retained"}

    optimizer._save_summary(tmp_path, result)

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["details"] == {**optimizer.details, "result_detail": "retained"}
