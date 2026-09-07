"""Final review resolves model flags, not missing evidence or hard publication gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_semantic import (
    CANDIDATE,
    SOURCE,
    _answer,
    _final,
    _json,
    _make,
    _pair,
    _review,
)

from zen.domain.core import AggressiveLimit
from zen.optimization.semantic import write_semantic_outputs
from zen.runtime.lm import ModelError


def _shared_pair():
    value = _pair()
    value["important_constraint_violations"] = [{
        "description": "Answer lacks an explanation of why it matters.",
        "rationale": "The judge expected a significance paragraph.",
    }]
    value["baseline_defects"] = ["Baseline also lacks a significance paragraph."]
    return value


def _confirm(system, user):
    value = _final(system, user)
    value["decision"] = "CONFIRMED"
    value["rationale"] = "The source requires a result, not a significance paragraph; both answers give it."
    for finding in value["findings"]:
        finding["classification"] = ("evaluator_error" if finding["id"].startswith("instruction/")
                                     else "shared_baseline_defect")
        finding["rationale"] = "Source and draft retain the same rule; the compared answers have no new loss."
    return value


def _harness(tmp_path, **kwargs):
    responses = {"PAIR": [_shared_pair()], "FINAL": [_confirm]}
    responses.update(kwargs.pop("responses", {}))
    return _make(tmp_path, responses=responses, **kwargs)


def test_shared_baseline_defect_can_be_confirmed_without_rewriting(tmp_path):
    harness = _harness(tmp_path)
    result = harness.run()
    assert result.decision == "CONFIRMED", result.message
    assert result.details["confirmation_origin"] == "model"
    assert result.details["preliminary"]["decision"] == "REVIEW_REQUIRED"
    assert result.details["phases"]["validation"]["pairs"][0]["status"] == "MATERIAL_DIFFERENCE"
    assert result.details["reasons"] == []
    assert result.details["errors"] == []
    assert harness.script.counts["FINAL"] == 1
    assert result.calls == 16
    assert result.details["budget"]["used"] == result.calls
    assert [stage for stage, _, _ in harness.script.calls][-1] == "FINAL"
    assert harness.script.counts["REWRITE"] == 1
    assert harness.script.counts["REPAIR"] == 0
    assert Path(result.details["artifacts"]["optimized"]).read_text(encoding="utf-8") == CANDIDATE
    manifest = _json(harness.output / "AGENTS.optimize.manifest.json")
    assert manifest["decision"] == "CONFIRMED"
    report = (Path(result.run_directory) / "report.md").read_text(encoding="utf-8")
    assert "Initial decision: REVIEW_REQUIRED" in report
    assert "shared_baseline_defect" in report
    assert "not independent or human verification" in report
    assert _json(Path(result.run_directory) / "final-review.json")["assessment"]["decision"] == "CONFIRMED"


def test_final_review_has_source_draft_all_cases_and_raw_answers(tmp_path):
    harness = _harness(tmp_path)
    result = harness.run()
    bundle = _json(Path(result.run_directory) / "final-review-input.json")
    assert bundle["source"] == SOURCE
    assert bundle["candidate"] == CANDIDATE
    assert len(bundle["findings"]) == 1
    assert "tokens" not in bundle  # No reward signal for overriding quality judgments.
    for phase in ("validation", "holdout"):
        assert len(bundle["phases"][phase]) == 2
        for record in bundle["phases"][phase]:
            assert record["case"]["required_meaning"]
            assert record["baseline"]["answer"]
            assert record["candidate"]["answer"]
            assert record["review"]


@pytest.mark.parametrize("phase", ["validation", "holdout"])
def test_one_material_regression_in_either_phase_blocks_confirmation(tmp_path, phase):
    def reject(system, user):
        value = _confirm(system, user)
        value["decision"] = "REVIEW_REQUIRED"
        value["phases"][phase][0]["outcome"] = "candidate_regression"
        value["rationale"] = "A new candidate defect remains."
        return value

    result = _harness(tmp_path, responses={"FINAL": [reject]}).run()
    assert result.decision == "REVIEW_REQUIRED"
    assert result.details["final_review"]["status"] == "COMPLETE"
    assert result.details["artifacts"]["optimized"] is None


@pytest.mark.parametrize("defect", [
    "missing_finding", "duplicate_finding", "unknown_finding", "no_rationale", "unknown_ref",
    "missing_arm_ref", "missing_holdout", "wrong_case", "duplicate_case", "regression",
    "uncertain", "instruction_loss", "phase_regression",
])
def test_incomplete_or_contradictory_final_confirmation_is_rejected(tmp_path, defect):
    def malformed(system, user):
        value = _confirm(system, user)
        if defect == "missing_finding":
            value["findings"] = []
        elif defect == "duplicate_finding":
            value["findings"].append(value["findings"][0])
        elif defect == "unknown_finding":
            value["findings"][0]["id"] = "unknown"
        elif defect == "no_rationale":
            value["findings"][0]["rationale"] = ""
        elif defect == "unknown_ref":
            value["findings"][0]["evidence_refs"].append("imaginary")
        elif defect == "missing_arm_ref":
            value["findings"][0]["evidence_refs"] = ["source", "candidate"]
        elif defect == "missing_holdout":
            del value["phases"]["holdout"]
        elif defect == "wrong_case":
            value["phases"]["holdout"][0]["case_id"] = "nonexistent"
        elif defect == "duplicate_case":
            value["phases"]["validation"][1] = value["phases"]["validation"][0]
        elif defect in ("regression", "uncertain"):
            value["findings"][0]["classification"] = "candidate_regression" if defect == "regression" else "uncertain"
        elif defect == "instruction_loss":
            value["instruction_assessment"]["outcome"] = "material_loss"
        else:
            value["phases"]["holdout"][0]["outcome"] = "candidate_regression"
        return value

    harness = _harness(tmp_path, responses={"FINAL": [malformed, malformed]})
    result = harness.run()
    assert result.decision == "REVIEW_REQUIRED"
    assert harness.script.counts["FINAL"] == 2
    assert result.details["errors"]
    assert result.details["artifacts"]["optimized"] is None


def test_direct_omission_cannot_be_excused_as_shared_answer_failure(tmp_path):
    def wrong(system, user):
        value = _confirm(system, user)
        for finding in value["findings"]:
            finding["classification"] = "shared_baseline_defect"
        return value

    harness = _harness(tmp_path, responses={
        "REVIEW": [_review(severity="material")] * 2,
        "FINAL": [wrong, wrong],
    })
    assert harness.run().decision == "REVIEW_REQUIRED"
    assert harness.script.counts["FINAL"] == 2


def test_final_can_resolve_erroneous_direct_flag(tmp_path):
    harness = _harness(tmp_path, responses={"REVIEW": [_review(severity="material")] * 2})
    result = harness.run()
    assert result.decision == "CONFIRMED"
    assert harness.script.counts["REPAIR"] == 1
    findings = result.details["final_review"]["assessment"]["findings"]
    assert any(item["id"].startswith("instruction/") for item in findings)


@pytest.mark.parametrize("responses,config", [
    ({"TASK": [ModelError("target unavailable")]}, {}),
    ({"REVIEW": ["bad", "bad"]}, {}),
    ({"PAIR": ["bad", "bad"]}, {}),
    ({"REWRITE": [{"body": SOURCE}]}, {}),
    ({"REWRITE": [{"body": "Report.\nKeep secrets.\n"}]}, {"aggressive_limit": AggressiveLimit(lines=1)}),
    ({"TASK": [_answer("Long output. " * 3000)] * 8}, {}),
])
def test_final_cannot_override_hard_blockers(tmp_path, responses, config):
    harness = _harness(tmp_path, responses=responses, **config)
    result = harness.run()
    assert result.decision == "REVIEW_REQUIRED"
    assert harness.script.counts["FINAL"] == 0
    assert result.details["final_review"]["status"] == "SKIPPED"


def test_no_final_review_when_initial_checks_pass(tmp_path):
    harness = _make(tmp_path)
    result = harness.run()
    assert result.decision == "VERIFIED"
    assert result.details["final_review"]["status"] == "SKIPPED"
    assert harness.script.counts["FINAL"] == 0


@pytest.mark.parametrize("budget,expected_calls", [(15, 0), (16, 1)])
def test_final_review_respects_remaining_shared_budget(tmp_path, budget, expected_calls):
    harness = _harness(tmp_path, budget=budget, responses={"FINAL": ["malformed", _confirm]})
    result = harness.run()
    assert result.decision == "REVIEW_REQUIRED"
    assert result.calls <= budget
    assert harness.script.counts["FINAL"] == expected_calls
    assert result.details["errors"]
    assert Path(result.details["artifacts"]["draft"]).is_file()


def test_final_schema_retry_can_succeed(tmp_path):
    harness = _harness(tmp_path, responses={"FINAL": ["invalid JSON", _confirm]})
    result = harness.run()
    assert result.decision == "CONFIRMED"
    assert harness.script.counts["FINAL"] == 2
    assert result.calls == 17


def test_final_provider_failure_is_not_retried(tmp_path):
    harness = _harness(tmp_path, responses={"FINAL": [ModelError("FINAL_PROVIDER_FAILURE")]})
    result = harness.run()
    assert result.decision == "REVIEW_REQUIRED"
    assert harness.script.counts["FINAL"] == 1
    assert "FINAL_PROVIDER_FAILURE" in json.dumps(_json(Path(result.run_directory) / "model-calls.json"))


def test_source_change_during_final_review_prevents_publication(tmp_path):
    harness = _harness(tmp_path)

    def before(stage, system, user):
        if stage == "FINAL":
            harness.source.write_bytes(b"User edit wins.\n")

    harness.script.before = before
    result = harness.run()
    assert result.decision == "REVIEW_REQUIRED"
    assert result.details["artifacts"]["optimized"] is None
    assert result.details["artifacts"]["draft"] is None
    assert harness.source.read_bytes() == b"User edit wins.\n"


def test_confirmed_output_is_revoked_on_late_source_change(tmp_path):
    harness = _harness(tmp_path)
    result = harness.run()
    assert result.decision == "CONFIRMED"
    accepted = Path(result.details["artifacts"]["optimized"])
    harness.source.write_bytes(b"Later edit wins.\n")
    write_semantic_outputs(result, harness.output)
    assert result.decision == "REVIEW_REQUIRED"
    assert not accepted.exists()
    assert _json(harness.output / "AGENTS.optimize.manifest.json")["decision"] == "REVIEW_REQUIRED"


def test_candidate_edit_during_final_review_never_publishes_confirmation(tmp_path):
    harness = _harness(tmp_path)

    def before(stage, system, user):
        if stage == "FINAL":
            artifacts = harness.optimizer.result.details["artifacts"]
            Path(artifacts["run_draft"]).write_bytes(b"Unreviewed content")

    harness.script.before = before
    result = harness.run()
    assert result.decision == "REVIEW_REQUIRED"
    assert result.details["errors"]
    assert result.details["artifacts"]["optimized"] is None
    assert _json(harness.output / "AGENTS.optimize.manifest.json")["decision"] == "REVIEW_REQUIRED"


@pytest.mark.parametrize("key", ["run_draft", "run_optimized"])
def test_late_candidate_evidence_edit_downgrades_publication(tmp_path, key):
    harness = _harness(tmp_path)
    result = harness.run()
    assert result.decision == "CONFIRMED"
    Path(result.details["artifacts"][key]).write_bytes(b"Changed evidence")
    write_semantic_outputs(result, harness.output)
    assert result.decision == "REVIEW_REQUIRED"
    assert result.details["artifacts"]["optimized"] is None
    assert _json(harness.output / "AGENTS.optimize.manifest.json")["decision"] == "REVIEW_REQUIRED"