"""Deterministic semantic-engine contracts; no provider calls or benefit claims."""

from __future__ import annotations

import codecs
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from zen.domain.core import AggressiveLimit, ModelResponse, OptimizeConfig, count_tokens
from zen.optimization.semantic import SemanticOptimizer, write_semantic_outputs
from zen.runtime.lm import CallBudget, ModelError

SOURCE = (
    "Report the result clearly. Never disclose secrets. "
    "Describe the result without repeating the result unnecessarily. "
    "Be clear and concise when explaining the result to the user. "
    "Keep all secrets private and do not reveal any secret information.\n"
)
CANDIDATE = "Report the result; never disclose secrets."
REPAIRED = "State the result concisely; keep secrets private."
STAGES = ("REWRITE", "REPAIR", "REVIEW", "CASES", "PAIR", "FINAL")


def _review(
    *, kind: str = "omission", severity: str | None = None,
) -> dict[str, Any]:
    return {
        "differences": [] if severity is None else [{
            "kind": kind,
            "severity": severity,
            "description": "The candidate no longer prohibits disclosure of secrets.",
            "rationale": "A request for credentials could now receive secret information.",
        }],
        "rationale": "Equivalent meaning" if severity is None else "Meaning changed",
    }


def _pair(**kwargs: Any) -> dict[str, Any]:
    return {
        **_review(**kwargs),
        "important_constraint_violations": [],
        "baseline_defects": [],
    }


def _cases(count: int = 2) -> dict[str, Any]:
    return {
        phase: [{
            "id": f"{phase}-{index}",
            "kind": "representative" if index == 0 else "boundary",
            "inquiry": f"{phase} inquiry {index}: explain the result without secrets",
            "context": {"fact": f"{phase} result {index} is complete"},
            "required_meaning": ["Explain the supplied result"],
            "important_constraints": ["Do not disclose secrets"],
        } for index in range(count)]
        for phase in ("validation", "holdout")
    }


def _answer(text: str = "The result is complete.") -> str:
    return f"[[ ## answer ## ]]\n{text}\n[[ ## completed ## ]]"


def _final(_system: str, user: str) -> dict:
    """Conservative scripted default; tests explicitly opt into justified confirmation."""
    payload = json.loads(user)
    payload = payload.get("request", payload)
    return {
        "decision": "REVIEW_REQUIRED", "rationale": "The disputed evidence remains uncertain.",
        "instruction_assessment": {"outcome": "preserved", "rationale": "Source prohibitions remain."},
        "findings": [{
            "id": finding["id"], "classification": "uncertain",
            "rationale": "The available evidence does not resolve this flag.",
            "evidence_refs": finding["required_evidence_refs"],
        } for finding in payload["findings"]],
        "phases": {phase: [{
            "case_id": record["case_id"], "outcome": "no_material_regression",
            "rationale": "Both answers report the same supplied result.",
        } for record in records] for phase, records in payload["phases"].items()},
    }


@dataclass
class Script:
    budget: CallBudget
    case_count: int = 2
    responses: dict[str, list[Any]] = field(default_factory=dict)
    before: Callable[[str, str, str], None] | None = None
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    def respond(self, stage: str, system: str, user: str) -> ModelResponse:
        # Like CopilotModel, the model owns the shared budget claim, even on errors.
        self.budget.claim()
        self.calls.append((stage, system, user))
        self.counts[stage] += 1
        if self.before is not None:
            self.before(stage, system, user)
        default = {
            "REWRITE": {"body": CANDIDATE},
            "REPAIR": {"body": REPAIRED},
            "REVIEW": _review(),
            "CASES": _cases(self.case_count),
            "PAIR": _pair(),
            "TASK": _answer(),
            "FINAL": _final,
        }
        replies = self.responses.get(stage, [])
        value = replies.pop(0) if replies else default[stage]
        if callable(value):
            value = value(system, user)
        if isinstance(value, Exception):
            raise value
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return ModelResponse(text, count_tokens(system + user), count_tokens(text))


class ScriptedModel:
    def __init__(self, script: Script, name: str, *, task: bool = False):
        self.script = script
        self.name = name
        self.task = task

    def complete(self, system: str, user: str) -> ModelResponse:
        if self.task:
            stage = "TASK"
        else:
            matches = [stage for stage in STAGES if f"SEMANTIC_{stage}" in system]
            if not matches:
                matches = [stage for stage in STAGES if f"SEMANTIC_{stage}" in user]
            assert len(matches) == 1, f"Unrecognized/ambiguous semantic stage: {system}"
            stage = matches[0]
        return self.script.respond(stage, system, user)


@dataclass
class Harness:
    optimizer: SemanticOptimizer
    script: Script
    source: Path
    output: Path

    def run(self):
        return self.optimizer.run(self.source)


def _make(
    root: Path,
    *,
    source: str | bytes = SOURCE,
    quick: bool = True,
    budget: int = 32,
    responses: dict[str, list[Any]] | None = None,
    before: Callable[[str, str, str], None] | None = None,
    **config: Any,
) -> Harness:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "AGENTS.md"
    path.write_bytes(source.encode("utf-8") if isinstance(source, str) else source)
    output = root / "out"
    script = Script(CallBudget(budget), 2 if quick else 3, responses or {}, before)
    optimizer = SemanticOptimizer(
        OptimizeConfig(
            engine="semantic", quick=quick, total_call_budget=budget,
            output_dir=output, **config,
        ),
        ScriptedModel(script, "offline-task", task=True),
        ScriptedModel(script, "offline-strong"),
        ScriptedModel(script, "offline-generator"),
        script.budget,
        root / "cache",
    )
    return Harness(optimizer, script, path, output)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(harness: Harness) -> dict[str, Any]:
    data = _json(harness.output / f"{harness.source.stem}.optimize.manifest.json")
    return data.get("artifacts", data)


def _evidence(result) -> dict[str, Any]:
    run = Path(result.run_directory)
    summary = _json(run / "summary.json")
    assert (run / "report.md").read_text(encoding="utf-8").strip()
    assert _json(run / "model-calls.json")
    assert summary["decision"] == result.decision
    assert result.details["engine"] == "semantic"
    assert isinstance(result.details["errors"], list)
    assert isinstance(result.details["reviews"], list)
    assert isinstance(result.details["phase"], str)
    return result.details["artifacts"]


def _has_draft(result) -> None:
    artifacts = _evidence(result)
    assert artifacts["drafts"]
    for key in ("draft", "run_draft"):
        path = Path(artifacts[key])
        assert path.is_file()
        assert path.name.endswith(".draft.md")
        assert path.read_bytes()


def _unverified(harness: Harness, result) -> None:
    assert result.decision == "REVIEW_REQUIRED", result.message
    _has_draft(result)
    assert result.details["artifacts"]["optimized"] is None
    assert result.details["artifacts"]["run_optimized"] is None
    write_semantic_outputs(result, harness.output)
    assert _manifest(harness)["optimized"] is None


def test_semantic_defaults_are_small_budget_without_an_implicit_cap() -> None:
    config = OptimizeConfig()
    assert config.engine == "semantic"
    assert config.total_call_budget == 32
    assert config.aggressive_limit is None
    assert config.max_body_lines(SOURCE) is None


def test_all_semantic_reports_omit_locations_but_manifest_retains_them(tmp_path: Path) -> None:
    harness = _make(tmp_path, budget=1)
    result = harness.run()
    draft = result.details["artifacts"]["draft"]
    error = f"Could not read '{draft}': permission denied."
    result.details["errors"].append(error)
    _, report = write_semantic_outputs(result, harness.output, user_acceptance=True)
    paths = [str(tmp_path), str(harness.source), draft, result.run_directory,
             result.details["artifacts"]["optimized"]]
    for path in (report, harness.output / "AGENTS.optimize.report.md",
                 Path(result.run_directory) / "report.md"):
        text = path.read_text(encoding="utf-8")
        assert all(location not in text for location in paths)
        assert "permission denied" in text
        assert "Confirmed by user" in text
        assert "Draft: available" in text
        assert "Accepted candidate (VERIFIED or CONFIRMED): available" in text
    manifest = _json(harness.output / "AGENTS.optimize.manifest.json")
    assert manifest["artifacts"]["draft"] == draft
    assert manifest["run_directory"] == result.run_directory
    assert error in result.details["errors"]


@pytest.mark.parametrize("accept", [True, False])
def test_user_choice_is_separate_from_model_verification(tmp_path: Path, accept: bool) -> None:
    harness = _make(tmp_path, budget=1)
    original = harness.source.read_bytes()
    result = harness.run()
    reasons = list(result.details["reasons"])
    errors = list(result.details["errors"])
    calls = result.calls
    candidate, report = write_semantic_outputs(result, harness.output, user_acceptance=accept)
    assert result.decision == ("CONFIRMED" if accept else "REVIEW_REQUIRED")
    assert result.details["model_assessment"]["decision"] == "REVIEW_REQUIRED"
    assert result.details["reasons"] == reasons
    assert result.details["errors"] == errors
    assert result.calls == calls
    assert harness.source.read_bytes() == original
    assert result.details["user_acceptance"]["status"] == ("ACCEPTED" if accept else "DECLINED")
    manifest = _json(harness.output / "AGENTS.optimize.manifest.json")
    assert manifest["decision"] == ("CONFIRMED" if accept else "REVIEW_REQUIRED")
    assert manifest["user_acceptance"] == result.details["user_acceptance"]
    assert bool(manifest["artifacts"]["optimized"]) is accept
    assert candidate is not None
    assert candidate.name.endswith(".optimized.md" if accept else ".draft.md")
    assert candidate.read_bytes() == Path(result.details["artifacts"]["run_draft"]).read_bytes()
    assert "User acceptance: " + ("ACCEPTED" if accept else "DECLINED") in report.read_text()
    if accept:
        assert result.details["confirmation_origin"] == "user"
        assert "Confirmed by user" in report.read_text()
        assert "Model decision before user choice: REVIEW_REQUIRED" in report.read_text()
    assert write_semantic_outputs(result, harness.output)[0] == candidate


@pytest.mark.parametrize("changed", ["source", "run_draft", "draft"])
def test_user_acceptance_cannot_bypass_changed_files(tmp_path: Path, changed: str) -> None:
    harness = _make(tmp_path, budget=1)
    result = harness.run()
    path = harness.source if changed == "source" else Path(result.details["artifacts"][changed])
    path.write_bytes(b"User edit must not be accepted as the reviewed text.")
    write_semantic_outputs(result, harness.output, user_acceptance=True)
    assert result.details["user_acceptance"]["status"] == "BLOCKED"
    assert result.decision != "CONFIRMED"
    assert result.details["artifacts"]["optimized"] is None
    assert not list(harness.output.glob("*.optimized.md"))
    assert path.read_bytes() == b"User edit must not be accepted as the reviewed text."


def test_user_confirmation_retains_material_findings_and_unmet_line_gate(tmp_path: Path) -> None:
    harness = _make(
        tmp_path, aggressive_limit=AggressiveLimit(lines=1),
        responses={
            "REWRITE": [{"body": "Report results.\nNever disclose secrets.\n"}],
            "PAIR": [_pair(severity="material")],
        },
    )
    result = harness.run()
    assert result.decision == "REVIEW_REQUIRED"
    phase = json.dumps(result.details["phases"])
    final = json.dumps(result.details["final_review"])
    reasons = list(result.details["reasons"])
    _, report = write_semantic_outputs(result, harness.output, user_acceptance=True)
    assert result.decision == "CONFIRMED"
    assert result.details["reasons"] == reasons
    assert json.dumps(result.details["phases"]) == phase
    assert json.dumps(result.details["final_review"]) == final
    text = report.read_text()
    assert "material omission" in text
    assert all(reason in text for reason in reasons)


def test_no_draft_cannot_be_user_confirmed(tmp_path: Path) -> None:
    harness = _make(tmp_path, responses={"REWRITE": [ModelError("offline failure")]})
    result = harness.run()
    candidate, _ = write_semantic_outputs(result, harness.output, user_acceptance=True)
    assert result.decision == "ERROR"
    assert candidate is None
    assert result.details["user_acceptance"]["status"] == "BLOCKED"


def test_user_accepted_output_is_revoked_after_source_change(tmp_path: Path) -> None:
    harness = _make(tmp_path, budget=1)
    result = harness.run()
    accepted, _ = write_semantic_outputs(result, harness.output, user_acceptance=True)
    harness.source.write_bytes(b"New source.")
    write_semantic_outputs(result, harness.output)
    assert not accepted.exists()
    assert _manifest(harness)["optimized"] is None
    assert result.decision != "CONFIRMED"
    assert result.details["user_acceptance"]["status"] == "BLOCKED"


@pytest.mark.parametrize("quick,count", [(True, 2), (False, 3)])
def test_clean_end_to_end_verification_and_persistent_evidence(
    tmp_path: Path, quick: bool, count: int,
) -> None:
    harness = _make(tmp_path, quick=quick)
    original = harness.source.read_bytes()
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    assert harness.source.read_bytes() == original
    artifacts = _evidence(result)
    assert artifacts["drafts"]
    for key in ("draft", "optimized", "run_draft", "run_optimized"):
        assert Path(artifacts[key]).is_file()
        assert Path(artifacts[key]).read_text(encoding="utf-8") == CANDIDATE
    assert Path(artifacts["optimized"]).name.endswith(".optimized.md")
    assert Path(artifacts["draft"]).parent == harness.output
    assert Path(artifacts["run_draft"]).parent == Path(result.run_directory)
    for phase in ("validation", "holdout"):
        assert len(result.details["phases"][phase]["pairs"]) == count
    assert harness.script.counts["TASK"] == 4 * count
    assert harness.script.counts["PAIR"] == 2 * count
    assert harness.script.counts["REWRITE"] == 1
    assert harness.script.counts["REPAIR"] == 0
    assert result.calls == harness.script.budget.calls
    assert result.details["budget"] == {"limit": 32, "used": result.calls}
    assert result.calls <= 32
    write_semantic_outputs(result, harness.output)
    manifest = _manifest(harness)
    assert Path(manifest["optimized"]) == Path(artifacts["optimized"])
    assert Path(manifest["draft"]) == Path(artifacts["draft"])


def test_draft_exists_in_both_locations_before_first_review(tmp_path: Path) -> None:
    harness = _make(tmp_path)
    observed = []

    def before(stage: str, _system: str, _user: str) -> None:
        if stage == "REVIEW":
            external = list(harness.output.glob("*.draft.md"))
            internal = list((tmp_path / "cache").rglob("*.draft.md"))
            assert external and internal
            assert external[-1].read_text(encoding="utf-8") == CANDIDATE
            assert internal[-1].read_text(encoding="utf-8") == CANDIDATE
            observed.append(stage)

    harness.script.before = before
    assert harness.run().decision == "VERIFIED"
    assert observed == ["REVIEW"]


def test_one_repair_is_persisted_before_rereview_and_frozen_before_cases(
    tmp_path: Path,
) -> None:
    harness = _make(tmp_path, responses={"REVIEW": [_review(severity="material"), _review()]})
    observations = []

    def before(stage: str, _system: str, _user: str) -> None:
        if stage == "REVIEW":
            bodies = {p.read_text(encoding="utf-8") for p in harness.output.glob("*.draft.md")}
            observations.append(bodies)
            if harness.script.counts["REVIEW"] == 2:
                assert {CANDIDATE, REPAIRED} <= bodies
                internal = {
                    p.read_text(encoding="utf-8")
                    for p in (tmp_path / "cache").rglob("*.draft.md")
                }
                assert {CANDIDATE, REPAIRED} <= internal

    harness.script.before = before
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    assert observations[0] == {CANDIDATE}
    assert harness.script.counts["REPAIR"] == 1
    assert len(result.details["artifacts"]["drafts"]) == 2
    assert Path(result.details["artifacts"]["optimized"]).read_text(encoding="utf-8") == REPAIRED
    stages = [stage for stage, _, _ in harness.script.calls]
    assert stages[:5] == ["REWRITE", "REVIEW", "REPAIR", "REVIEW", "CASES"]
    assert all(stage not in ("REWRITE", "REPAIR") for stage in stages[5:])


@pytest.mark.parametrize("kind", ["omission", "addition", "conflict"])
def test_material_direct_difference_after_one_repair_prevents_verification(
    tmp_path: Path, kind: str,
) -> None:
    harness = _make(tmp_path, responses={
        "REVIEW": [_review(kind=kind, severity="material")] * 2,
    })
    result = harness.run()
    _unverified(harness, result)
    assert harness.script.counts["REPAIR"] == 1
    assert harness.script.counts["REVIEW"] == 2
    assert harness.script.counts["CASES"] == 1
    assert harness.script.counts["TASK"] > 0


def test_minor_difference_is_not_a_material_failure(tmp_path: Path) -> None:
    minor = _review(severity="minor")
    harness = _make(tmp_path, responses={"REVIEW": [minor, minor], "PAIR": [_pair(severity="minor")]})
    result = harness.run()
    assert result.decision == "VERIFIED", result.message


@pytest.mark.parametrize("phase", ["validation", "holdout"])
@pytest.mark.parametrize("failure", ["material", "important_constraint"])
def test_paired_failures_in_either_phase_block_verification(
    tmp_path: Path, phase: str, failure: str,
) -> None:
    bad = _pair(severity="material") if failure == "material" else _pair()
    if failure == "important_constraint":
        bad["important_constraint_violations"] = [{
            "description": "A secret is disclosed in the candidate output.",
            "rationale": "The case explicitly prohibits secret disclosure.",
        }]
    prefix = [_pair()] * (2 if phase == "holdout" else 0)
    harness = _make(tmp_path, responses={"PAIR": [*prefix, bad]})
    result = harness.run()
    _unverified(harness, result)
    assert result.details["phases"][phase]["pairs"]


def test_baseline_defects_are_recorded_but_are_not_candidate_failures(tmp_path: Path) -> None:
    pair = _pair()
    pair["baseline_defects"] = ["The baseline repeats the result unnecessarily."]
    harness = _make(tmp_path, responses={"PAIR": [pair]})
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    assert pair["baseline_defects"][0] in json.dumps(result.details)


def test_cases_use_source_only_and_target_never_sees_case_grading_criteria(tmp_path: Path) -> None:
    cases = _cases()
    required = "PRIVATE_REQUIRED_MEANING_SENTINEL"
    constraint = "PRIVATE_IMPORTANT_CONSTRAINT_SENTINEL"
    for phase_cases in cases.values():
        for case in phase_cases:
            case["required_meaning"] = [required]
            case["important_constraints"] = [constraint]
    harness = _make(tmp_path, responses={"CASES": [cases]})
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    stage, system, user = next(call for call in harness.script.calls if call[0] == "CASES")
    assert stage == "CASES"
    assert SOURCE.strip() in system + user or SOURCE.strip() in json.dumps(json.loads(user))
    assert CANDIDATE not in system + user
    tasks = [system + user for stage, system, user in harness.script.calls if stage == "TASK"]
    assert tasks
    assert all(required not in prompt and constraint not in prompt for prompt in tasks)
    pair_prompts = [system + user for stage, system, user in harness.script.calls if stage == "PAIR"]
    assert any(required in prompt and constraint in prompt for prompt in pair_prompts)
    assert any("holdout inquiry" in prompt for prompt in tasks)
    assert not any("holdout inquiry" in system + user
                   for stage, system, user in harness.script.calls if stage in ("REWRITE", "REPAIR"))


@pytest.mark.parametrize("stage", ["REWRITE", "REVIEW", "CASES", "PAIR"])
def test_structured_stages_allow_one_schema_repair_retry(tmp_path: Path, stage: str) -> None:
    harness = _make(tmp_path, responses={stage: ["INVALID_SCHEMA_SENTINEL"]})
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    expected = 5 if stage == "PAIR" else 2
    assert harness.script.counts[stage] == expected
    journal = json.dumps(_json(Path(result.run_directory) / "model-calls.json"))
    assert "INVALID_SCHEMA_SENTINEL" in journal


def test_repair_stage_also_allows_only_one_schema_retry(tmp_path: Path) -> None:
    harness = _make(tmp_path, responses={
        "REVIEW": [_review(severity="material"), _review()],
        "REPAIR": ["not JSON", {"body": REPAIRED}],
    })
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    assert harness.script.counts["REPAIR"] == 2
    assert len(result.details["artifacts"]["drafts"]) == 2


@pytest.mark.parametrize("invalid", [
    "not JSON", {}, {"body": ""}, {"body": " \t\r\n"}, {"body": 42},
    {"body": None}, {"body": ["Report results."]},
])
def test_initial_generation_schema_failure_is_error_with_report_and_no_draft(
    tmp_path: Path, invalid: Any,
) -> None:
    harness = _make(tmp_path, responses={"REWRITE": [invalid, invalid]})
    result = harness.run()
    assert result.decision == "ERROR", result.message
    artifacts = _evidence(result)
    assert result.details["errors"]
    assert artifacts["draft"] is None
    assert artifacts["run_draft"] is None
    assert artifacts["optimized"] is None
    assert artifacts["drafts"] == []
    assert harness.script.counts == {"REWRITE": 2}
    assert not list(harness.output.glob("*.draft.md"))
    write_semantic_outputs(result, harness.output)
    assert _manifest(harness)["optimized"] is None


@pytest.mark.parametrize("stage", ["REWRITE", "REVIEW", "REPAIR", "CASES", "PAIR"])
def test_provider_errors_are_not_retried_and_raw_error_is_journaled(
    tmp_path: Path, stage: str,
) -> None:
    responses = {stage: [ModelError("OFFLINE_PROVIDER_FAILURE_SENTINEL")]}
    if stage == "REPAIR":
        responses["REVIEW"] = [_review(severity="material")]
    harness = _make(tmp_path, responses=responses)
    result = harness.run()
    if stage == "REWRITE":
        assert result.decision == "ERROR"
        assert result.details["artifacts"]["draft"] is None
    else:
        _unverified(harness, result)
    assert result.details["errors"]
    if stage != "PAIR":
        assert harness.script.counts[stage] == 1
    else:
        # Later cases may still be evaluated; the failing case itself must not retry.
        prompts = [(system, user) for name, system, user in harness.script.calls if name == stage]
        assert len(prompts) == len(set(prompts))
    journal = json.dumps(_json(Path(result.run_directory) / "model-calls.json"))
    assert "OFFLINE_PROVIDER_FAILURE_SENTINEL" in journal


def test_direct_review_schema_failure_does_not_discard_successful_pair_evidence(
    tmp_path: Path,
) -> None:
    harness = _make(tmp_path, responses={"REVIEW": ["bad review", "still bad review"]})
    result = harness.run()
    _unverified(harness, result)
    assert harness.script.counts["REVIEW"] == 2
    assert harness.script.counts["CASES"] == 1
    assert harness.script.counts["TASK"] == 8
    assert harness.script.counts["PAIR"] == 4
    for phase in ("validation", "holdout"):
        assert len(result.details["phases"][phase]["pairs"]) == 2
    assert result.details["errors"]


@pytest.mark.parametrize("stage", ["REPAIR", "CASES", "PAIR"])
def test_exhausted_schema_retries_retain_prior_drafts(tmp_path: Path, stage: str) -> None:
    responses = {stage: ["bad first response", "bad second response"]}
    if stage == "REPAIR":
        responses["REVIEW"] = [_review(severity="material")]
    harness = _make(tmp_path, responses=responses)
    result = harness.run()
    _unverified(harness, result)
    assert result.details["errors"]
    if stage != "PAIR":
        assert harness.script.counts[stage] == 2
    assert Path(result.details["artifacts"]["draft"]).read_text(encoding="utf-8") == CANDIDATE


@pytest.mark.parametrize("defect", [
    "missing_holdout", "too_few", "duplicate_id", "no_boundary",
    "no_representative", "empty_inquiry", "non_object_context",
])
def test_invalid_case_sets_cannot_produce_verified_evidence(tmp_path: Path, defect: str) -> None:
    cases = _cases()
    if defect == "missing_holdout":
        del cases["holdout"]
    elif defect == "too_few":
        cases["holdout"].pop()
    elif defect == "duplicate_id":
        cases["holdout"][0]["id"] = cases["validation"][0]["id"]
    elif defect == "no_boundary":
        for case in cases["validation"]:
            case["kind"] = "representative"
    elif defect == "no_representative":
        for case in cases["holdout"]:
            case["kind"] = "boundary"
    elif defect == "empty_inquiry":
        cases["validation"][0]["inquiry"] = ""
    elif defect == "non_object_context":
        cases["validation"][0]["context"] = []
    harness = _make(tmp_path, responses={"CASES": [cases, cases]})
    result = harness.run()
    _unverified(harness, result)
    assert harness.script.counts["CASES"] == 2
    assert harness.script.counts["TASK"] == 0


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 10, 14])
def test_budget_exhaustion_retains_draft_and_never_reports_verification(
    tmp_path: Path, limit: int,
) -> None:
    harness = _make(tmp_path, budget=limit)
    result = harness.run()
    _unverified(harness, result)
    assert harness.script.budget.calls <= limit
    assert result.calls == harness.script.budget.calls
    assert result.details["budget"] == {"limit": limit, "used": result.calls}
    assert result.details["errors"]


def test_target_failure_cannot_be_overridden_by_a_clean_pair_judge(tmp_path: Path) -> None:
    harness = _make(tmp_path, responses={"TASK": [ModelError("target failed")]})
    result = harness.run()
    _unverified(harness, result)
    assert result.details["errors"]
    assert "target failed" in json.dumps(_json(Path(result.run_directory) / "model-calls.json"))


def test_empty_target_answer_is_incomplete_not_verified(tmp_path: Path) -> None:
    harness = _make(tmp_path, responses={"TASK": [_answer("")]})
    _unverified(harness, harness.run())


@pytest.mark.parametrize("candidate", [SOURCE, SOURCE + "More redundant instructions.\n"])
def test_non_smaller_candidate_is_still_persisted_but_not_verified(
    tmp_path: Path, candidate: str,
) -> None:
    harness = _make(tmp_path, responses={"REWRITE": [{"body": candidate}]})
    result = harness.run()
    _unverified(harness, result)
    assert Path(result.details["artifacts"]["draft"]).read_text(encoding="utf-8") == candidate


def test_explicit_line_cap_affects_status_not_first_draft_persistence(tmp_path: Path) -> None:
    candidate = "Report results.\nNever disclose secrets.\n"
    harness = _make(tmp_path, aggressive_limit=AggressiveLimit(lines=1),
                    responses={"REWRITE": [{"body": candidate}]})
    result = harness.run()
    _unverified(harness, result)
    assert Path(result.details["artifacts"]["draft"]).read_text(encoding="utf-8") == candidate


def test_larger_output_is_allowed_when_combined_holdout_tokens_decrease(tmp_path: Path) -> None:
    def target(system: str, user: str) -> str:
        return _answer("Complete. Additional useful detail." if CANDIDATE in system + user else "Done.")

    baseline_tokens = count_tokens(SOURCE) + count_tokens("Done.")
    candidate_tokens = count_tokens(CANDIDATE) + count_tokens("Complete. Additional useful detail.")
    assert candidate_tokens <= baseline_tokens * 0.97
    harness = _make(tmp_path, responses={"TASK": [target] * 8})
    result = harness.run()
    assert result.decision == "VERIFIED", result.message


def test_less_than_three_percent_combined_reduction_is_not_verified(tmp_path: Path) -> None:
    long_output = "The result remains complete. " * 1500
    assert (
        (count_tokens(SOURCE) - count_tokens(CANDIDATE))
        / (count_tokens(SOURCE) + count_tokens(long_output))
    ) < 0.03
    harness = _make(tmp_path, responses={"TASK": [_answer(long_output)] * 8})
    _unverified(harness, harness.run())


def test_acceptance_uses_holdout_metrics_not_better_validation_outputs(tmp_path: Path) -> None:
    def target(system: str, user: str) -> str:
        candidate = CANDIDATE in system + user
        holdout = "holdout inquiry" in user
        return _answer("Expanded output. " * 500 if candidate and holdout else "Done.")

    harness = _make(tmp_path, responses={"TASK": [target] * 8})
    result = harness.run()
    _unverified(harness, result)
    assert len(result.details["phases"]["holdout"]["pairs"]) == 2


def test_bom_frontmatter_and_source_bytes_survive_full_body_rewrite(tmp_path: Path) -> None:
    prefix = codecs.BOM_UTF8 + b"---\r\nname: semantic-test\nversion: 1\r\n---\r\n"
    source = prefix + SOURCE.replace("\n", "\r\n").encode("utf-8")
    replacement = "Report results.\nNever disclose secrets.\n"
    harness = _make(tmp_path, source=source, responses={"REWRITE": [{"body": replacement}]})
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    expected = prefix + replacement.replace("\n", "\r\n").encode("utf-8")
    assert harness.source.read_bytes() == source
    for key in ("draft", "optimized", "run_draft", "run_optimized"):
        assert Path(result.details["artifacts"][key]).read_bytes() == expected


@pytest.mark.parametrize("focus", ["task", "communication"])
def test_focus_preserves_frozen_mixed_newlines_tabs_and_trailing_spaces(
    tmp_path: Path, focus: str,
) -> None:
    prefix = (
        codecs.BOM_UTF8 + b"---\r\nname: focused\n---\r\n"
        + b"\tFrozen header  \n"
        + f"<!-- zen:{focus} -->".encode()
    )
    suffix = f"<!-- /zen:{focus} -->\n\tFrozen footer  \r\nNo final newline  ".encode()
    source = prefix + ("\r\n" + SOURCE * 3).encode() + suffix
    replacement = "\nReport results.\nNever disclose secrets.\n"
    harness = _make(tmp_path, source=source, focus=focus,
                    responses={"REWRITE": [{"body": replacement}]})
    result = harness.run()
    assert result.decision == "VERIFIED", result.message
    expected = prefix + replacement.replace("\n", "\r\n").encode() + suffix
    assert harness.source.read_bytes() == source
    for key in ("draft", "optimized", "run_draft", "run_optimized"):
        assert Path(result.details["artifacts"][key]).read_bytes() == expected


def test_source_change_during_rewrite_blocks_external_draft_publication(tmp_path: Path) -> None:
    harness = _make(tmp_path)
    changed = b"User edited the source while generation was in progress.\n"

    def mutate(stage: str, _system: str, _user: str) -> None:
        if stage == "REWRITE":
            harness.source.write_bytes(changed)

    harness.script.before = mutate
    result = harness.run()
    assert result.decision != "VERIFIED"
    assert harness.source.read_bytes() == changed
    assert not list(harness.output.glob("*.draft.md"))
    assert not list(harness.output.glob("*.optimized.md"))
    assert result.details["errors"]


def test_source_change_after_draft_revokes_only_this_runs_external_artifacts(tmp_path: Path) -> None:
    harness = _make(tmp_path)
    harness.output.mkdir()
    unrelated = harness.output / "another-run.optimized.md"
    unrelated.write_bytes(b"Previous accepted evidence, do not delete.")
    changed = b"A concurrent user edit must win.\n"
    observed = []

    def mutate(stage: str, _system: str, _user: str) -> None:
        if stage == "REVIEW":
            observed.extend(harness.output.glob("*.draft.md"))
            assert observed
            harness.source.write_bytes(changed)

    harness.script.before = mutate
    result = harness.run()
    assert result.decision != "VERIFIED"
    assert harness.source.read_bytes() == changed
    assert observed and all(not path.exists() for path in observed)
    assert unrelated.read_bytes() == b"Previous accepted evidence, do not delete."
    assert result.details["artifacts"]["draft"] is None
    assert result.details["artifacts"]["optimized"] is None
    write_semantic_outputs(result, harness.output)
    assert not any(path.exists() for path in observed)
    assert _manifest(harness)["optimized"] is None


def test_late_source_change_is_checked_again_by_output_writer(tmp_path: Path) -> None:
    harness = _make(tmp_path)
    result = harness.run()
    assert result.decision == "VERIFIED"
    externals = [
        Path(result.details["artifacts"][key]) for key in ("draft", "optimized")
    ]
    harness.source.write_bytes(b"Edited after optimization, before publication.\n")
    write_semantic_outputs(result, harness.output)
    assert result.decision != "VERIFIED"
    assert all(not path.exists() for path in externals)
    assert _manifest(harness)["optimized"] is None
    assert harness.source.read_bytes() == b"Edited after optimization, before publication.\n"


def test_new_unverified_manifest_does_not_point_to_previous_accepted_candidate(
    tmp_path: Path,
) -> None:
    first = _make(tmp_path)
    accepted = first.run()
    assert accepted.decision == "VERIFIED"
    write_semantic_outputs(accepted, first.output)
    old_candidate = Path(accepted.details["artifacts"]["optimized"])
    old_bytes = old_candidate.read_bytes()
    old_reports = set(first.output.glob("*.report.md"))
    assert old_reports

    second = _make(tmp_path, responses={"REVIEW": [_review(severity="material")] * 2})
    rejected = second.run()
    _unverified(second, rejected)
    assert old_candidate.read_bytes() == old_bytes
    assert accepted.run_directory != rejected.run_directory
    assert accepted.details["artifacts"]["draft"] != rejected.details["artifacts"]["draft"]
    assert old_reports < set(second.output.glob("*.report.md"))
    assert _manifest(second)["optimized"] is None
    assert Path(_manifest(second)["draft"]) == Path(rejected.details["artifacts"]["draft"])


def test_repeated_successful_runs_have_unique_evidence_and_make_fresh_task_calls(
    tmp_path: Path,
) -> None:
    harness = _make(tmp_path)
    first = harness.run()
    first_calls = harness.script.counts["TASK"]
    second = harness.run()
    assert first.decision == second.decision == "VERIFIED"
    assert harness.script.counts["TASK"] == 2 * first_calls == 16
    assert first.run_directory != second.run_directory
    for key in ("draft", "optimized", "run_draft", "run_optimized"):
        assert first.details["artifacts"][key] != second.details["artifacts"][key]
        assert Path(first.details["artifacts"][key]).exists()
        assert Path(second.details["artifacts"][key]).exists()


def test_output_writer_honors_an_explicit_destination(tmp_path: Path) -> None:
    harness = _make(tmp_path)
    result = harness.run()
    destination = tmp_path / "published"
    write_semantic_outputs(result, destination)
    manifest_path = destination / "AGENTS.optimize.manifest.json"
    assert manifest_path.is_file()
    manifest = _json(manifest_path)
    artifacts = manifest.get("artifacts", manifest)
    for key in ("draft", "optimized"):
        assert Path(artifacts[key]).parent == destination
        assert Path(artifacts[key]).is_file()
    assert list(destination.glob("*.report.md"))


def test_current_manifest_and_report_exist_before_first_review(tmp_path: Path) -> None:
    harness = _make(tmp_path)
    observations = []

    def before(stage: str, _system: str, _user: str) -> None:
        if stage == "REVIEW":
            manifest = _json(harness.output / "AGENTS.optimize.manifest.json")
            assert manifest["decision"] == "REVIEW_REQUIRED"
            assert manifest["artifacts"]["optimized"] is None
            assert Path(manifest["artifacts"]["draft"]).read_text(encoding="utf-8") == CANDIDATE
            assert (harness.output / "AGENTS.optimize.report.md").is_file()
            observations.append(stage)

    harness.script.before = before
    assert harness.run().decision == "VERIFIED"
    assert observations == ["REVIEW"]


def test_pair_judge_receives_source_for_checking_generated_criteria(tmp_path: Path) -> None:
    harness = _make(tmp_path)
    assert harness.run().decision == "VERIFIED"
    prompts = [json.loads(user) for stage, _, user in harness.script.calls if stage == "PAIR"]
    assert len(prompts) == 4
    assert all(prompt["source"] == SOURCE for prompt in prompts)


def test_complete_material_differences_are_not_reported_as_incomplete(tmp_path: Path) -> None:
    harness = _make(tmp_path, responses={"PAIR": [_pair(severity="material")]})
    result = harness.run()
    _unverified(harness, result)
    assert not result.details["errors"]
    assert "validation: material candidate differences reported." in result.details["reasons"]
    assert not any("incomplete" in reason for reason in result.details["reasons"])


def test_atomic_publication_retries_only_local_permission_errors(tmp_path: Path, monkeypatch) -> None:
    from zen.optimization.semantic_outputs import atomic_bytes

    original_replace = Path.replace
    attempts = []

    def replace(path, destination):
        attempts.append(destination)
        if len(attempts) < 3:
            raise PermissionError("temporary Windows destination lock")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    path = tmp_path / "manifest.json"
    atomic_bytes(path, b"current")
    assert path.read_bytes() == b"current"
    assert len(attempts) == 3
    assert not list(tmp_path.glob("*.tmp"))
