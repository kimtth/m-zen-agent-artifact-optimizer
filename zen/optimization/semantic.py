"""Draft-first semantic rewriting with bounded, separate verification evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from uuid import uuid4

from ..domain.core import (
    EvaluationCase,
    ModelResponse,
    OptimizationResult,
    OptimizeConfig,
    count_tokens,
    load_artifact,
    parse_json,
)
from ..runtime.harness import RunCache, Runner
from ..runtime.lm import BudgetExceeded, CallBudget, TextModel, fresh_calls
from ..runtime.progress import ProgressCallback
from . import semantic_checks as checks
from .proposer import validate_focus
from .semantic_adjudication import FINAL, evidence_bundle, validate_final
from .semantic_outputs import (
    atomic_bytes,
    atomic_json,
    digest,
    publish_file,
    revoke_publication,
    save_summary,
    source_matches,
    write_semantic_outputs,
)

__all__ = ["SemanticOptimizer", "write_semantic_outputs"]


class _JournalModel:
    def __init__(self, model: TextModel, root: Path, journal: list):
        self.model, self.root, self.journal = model, root, journal
        self.name = model.name

    def complete(self, system: str, user: str) -> ModelResponse:
        entry = {"model": self.name, "system": system, "user": user}
        self.journal.append(entry)
        try:
            with fresh_calls():
                response = self.model.complete(system, user)
            entry.update(asdict(response))
            return response
        except Exception as exc:
            entry["error"] = str(exc)
            raise
        finally:
            atomic_json(self.root / "model-calls.json", self.journal)


class SemanticOptimizer:
    def __init__(
        self, config: OptimizeConfig, task_model: TextModel, strong_model: TextModel,
        generator_model: TextModel, budget: CallBudget, cache_root: Path,
        progress: ProgressCallback | None = None,
    ):
        self.config = config
        self.models = task_model, strong_model, generator_model
        self.budget, self.cache_root = budget, cache_root.resolve()
        self.progress = progress or (lambda _percent, _message: None)

    def run(self, path: Path) -> OptimizationResult:
        self.artifact = load_artifact(path.resolve())
        self.scope = validate_focus(self.artifact.body, self.config.focus)
        run_id = uuid4().hex
        self.root = self.cache_root / "runs" / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        source_bytes = (b"\xef\xbb\xbf" if self.artifact.bom else b"") + self.artifact.source.encode("utf-8")
        count = 2 if self.config.quick else 3
        self.result = OptimizationResult(
            "ERROR", str(self.artifact.path), None, None, None, None,
            self.budget.calls, str(self.root), "Rewrite has not completed.", {
                "engine": "semantic", "run_id": run_id, "source_sha256": digest(source_bytes),
                "output_directory": str((self.config.output_dir or self.artifact.path.parent).resolve()),
                "phase": "rewrite", "errors": [], "reviews": [], "reasons": [],
                "final_review": {"status": "NOT_RUN", "reason": "Evaluation is pending."},
                "publications": [], "budget": {"limit": self.budget.limit, "used": self.budget.calls},
                "artifacts": {"draft": None, "optimized": None, "run_draft": None,
                              "run_optimized": None, "drafts": []},
                "phases": {phase: {"expected": count, "pairs": []}
                           for phase in ("validation", "holdout")},
            },
        )
        self.details = self.result.details
        self.journal: list[dict] = []
        self.task, self.strong, self.generator = (
            _JournalModel(model, self.root, self.journal) for model in self.models
        )
        self.runner = Runner(self.task, RunCache(self.root / "task-runs"), run_id)
        atomic_bytes(self.root / "source.md", source_bytes)
        atomic_json(self.root / "experiment.json", {
            "source_sha256": self.details["source_sha256"],
            "config": {**asdict(self.config), "output_dir": self.details["output_directory"]},
            "contract": "draft first; one repair; one schema retry; no provider retries; "
                        "frozen candidate; separate complete validation and holdout; "
                        "no material losses; smaller artifact and configured combined reduction; "
                        "one final adjudication of eligible disputed evidence may yield CONFIRMED",
        })
        atomic_json(self.root / "model-calls.json", [])
        save_summary(self.result)
        try:
            self._run(count)
        except Exception as exc:  # noqa: BLE001 - persist initialized runs even on failures.
            self._error(str(exc))
        self._decide()
        try:
            self._adjudicate()
        except Exception as exc:  # noqa: BLE001 - retain draft if final evidence cannot be processed.
            self._error(str(exc))
            self.details["final_review"] = {"status": "ERROR", "reason": str(exc)}
        self.result.calls = self.budget.calls
        self.details["budget"]["used"] = self.budget.calls
        if not source_matches(self.result):
            revoke_publication(self.result)
        self._save_accepted()
        write_semantic_outputs(self.result)
        self.progress(100, f"Finished: {self.result.decision} ({self.budget.calls} calls)")
        return self.result

    def _error(self, message: str) -> None:
        self.details["errors"].append(f"{self.details['phase']}: {message}")

    def _structured(self, model, system, payload, validator) -> dict | None:
        """Retry malformed data once; never retry provider failures or exhausted budgets."""
        request = json.dumps(payload, ensure_ascii=False)
        for attempt in range(2):
            self.progress(self.percent, f"{self.details['phase']} — call {self.budget.calls + 1}/{self.budget.limit}")
            try:
                response = model.complete(system, request)
            except Exception as exc:  # noqa: BLE001 - provider errors are evidence, not retries.
                self._error(str(exc))
                return None
            try:
                return validator(parse_json(response.text))
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 1:
                    self._error(f"invalid structured response after two attempts: {exc}")
                    return None
                request = json.dumps({
                    "request": payload, "schema_error": str(exc),
                    "instruction": "Return the required valid JSON shape. Do not add commentary.",
                }, ensure_ascii=False)
        return None

    def _stage(self, phase: str, percent: int) -> None:
        self.details["phase"], self.percent = phase, percent
        self.progress(percent, phase)

    def _save_draft(self, replacement: str) -> None:
        # Normalize only the mutable region; frozen mixed-newline text remains byte-exact.
        replacement = replacement.replace("\r\n", "\n").replace("\r", "\n")
        replacement = replacement.replace("\n", self.artifact.newline)
        body = self.scope.replace(self.artifact.body, replacement)
        data = self.artifact.immutable_prefix + body
        raw = (b"\xef\xbb\xbf" if self.artifact.bom else b"") + data.encode("utf-8")
        self.result.candidate_body = body
        self.details["candidate_bytes_sha256"] = digest(raw)
        artifacts = self.details["artifacts"]
        name = f"{self.artifact.path.stem}.{self.details['run_id']}.r{len(artifacts['drafts']) + 1}.draft.md"
        internal = self.root / name
        atomic_bytes(internal, raw)
        entry = {"run_draft": str(internal), "draft": None, "sha256": digest(raw)}
        artifacts["drafts"].append(entry)
        artifacts["run_draft"] = str(internal)
        external = publish_file(self.result, internal, Path(self.details["output_directory"]))
        entry["draft"] = artifacts["draft"] = str(external) if external else None
        self.result.decision = "REVIEW_REQUIRED"
        self.result.message = "Draft saved; verification is pending."
        self.result.calls = self.budget.calls
        self.details["budget"]["used"] = self.budget.calls
        save_summary(self.result)
        write_semantic_outputs(self.result)
        self.progress(self.percent, f"Draft saved: {external or internal}")

    def _run(self, count: int) -> None:
        self._stage("rewrite", 3)
        payload = {
            "source": self.artifact.body, "focus": self.config.focus,
            "mutable_region": self.scope.extract(self.artifact.body),
            "max_body_lines": self.config.max_body_lines(self.artifact.body),
        }
        rewrite = self._structured(self.strong, checks.REWRITE, payload, checks.validate_body)
        if rewrite is None:
            return
        self._save_draft(rewrite["body"])
        if not source_matches(self.result):
            return
        self._stage("instruction review", 15)
        review = self._review()
        if review and checks.material(review):
            self._stage("repair", 20)
            repair = self._structured(self.strong, checks.REPAIR, {
                **payload, "draft": self.result.candidate_body, "findings": review,
            }, checks.validate_body)
            if repair:
                self._save_draft(repair["body"])
                self._stage("instruction re-review", 25)
                self._review()
        if not source_matches(self.result):
            return
        # All mutation is now over. Neither validation nor holdout feeds a repair.
        self.details["candidate_sha256"] = digest(self.result.candidate_body.encode("utf-8"))
        self._stage("source-based cases", 30)
        cases = self._structured(self.generator, checks.CASES, {
            "source": self.artifact.body, "cases_per_phase": count,
        }, lambda value: checks.validate_cases(value, count))
        if cases is None:
            return
        self.cases = cases
        atomic_json(self.root / "cases.json", cases)
        for phase, start in (("validation", 35), ("holdout", 65)):
            for index, case in enumerate(cases[phase]):
                self._stage(f"{phase} {index + 1}/{count}", start + round(25 * index / count))
                if self.budget.calls >= self.budget.limit:
                    self._error("application model-call budget exhausted; remaining cases not evaluated")
                    return
                pair = self._pair(case, phase)
                self.details["phases"][phase]["pairs"].append(pair)
                atomic_json(self.root / f"{phase}.json", self.details["phases"][phase])
                save_summary(self.result)

    def _review(self) -> dict | None:
        review = self._structured(self.strong, checks.REVIEW, {
            "source": self.artifact.body, "candidate": self.result.candidate_body,
        }, checks.validate_review)
        if review:
            self.details["reviews"].append(review)
            atomic_json(self.root / "instruction-reviews.json", self.details["reviews"])
        return review

    def _pair(self, case: dict, phase: str) -> dict:
        evidence = {"case_id": case["id"], "status": "INCOMPLETE", "review": None}
        task = EvaluationCase(case["id"], case["kind"], case["id"], case["inquiry"],
                              case["context"], (), (), (), ())
        for arm, body in (("baseline", self.artifact.body), ("candidate", self.result.candidate_body)):
            self.progress(self.percent, f"{self.details['phase']}: {arm} answer — call {self.budget.calls + 1}/{self.budget.limit}")
            try:
                with fresh_calls():
                    run = self.runner.run(body, task, scope=f"{phase}:{arm}")
            except BudgetExceeded as exc:
                self._error(str(exc))
                return evidence
            evidence[arm] = run.to_dict()
            if run.error:
                self._error(f"{case['id']} {arm}: {run.error}")
        if any(evidence.get(arm, {}).get("error", "missing") for arm in ("baseline", "candidate")):
            return evidence
        review = self._structured(self.strong, checks.PAIR, {
            "source": self.artifact.body,
            "case": case, "baseline": evidence["baseline"]["answer"],
            "candidate": evidence["candidate"]["answer"],
        }, lambda value: checks.validate_review(value, paired=True))
        if review:
            evidence["review"] = review
            evidence["status"] = "MATERIAL_DIFFERENCE" if (
                checks.material(review) or review["important_constraint_violations"]
            ) else "PASS"
        return evidence

    def _decide(self) -> None:
        result, details = self.result, self.details
        if result.candidate_body is None:
            result.decision = "ERROR"
            result.message = "No usable draft was generated; see the recorded errors."
            return
        reasons = details["reasons"] = []
        hard = details["hard_blockers"] = []
        if not details["reviews"]:
            hard.append("Instruction review is incomplete.")
        elif checks.material(details["reviews"][-1]):
            reasons.append("Instruction review reports material differences.")
        for phase, evidence in details["phases"].items():
            if len(evidence["pairs"]) != evidence["expected"] or any(
                pair["status"] == "INCOMPLETE" for pair in evidence["pairs"]
            ):
                hard.append(f"{phase}: incomplete evidence.")
            if any(pair["status"] == "MATERIAL_DIFFERENCE" for pair in evidence["pairs"]):
                reasons.append(f"{phase}: material candidate differences reported.")
        original_tokens = self.artifact.tokens
        candidate_tokens = count_tokens(self.artifact.immutable_prefix + result.candidate_body)
        metrics = details["tokens"] = {"baseline_artifact": original_tokens,
                                      "candidate_artifact": candidate_tokens}
        if candidate_tokens >= original_tokens:
            hard.append("The draft artifact is not smaller than the source.")
        cap = self.config.max_body_lines(self.artifact.body)
        if cap is not None and len(result.candidate_body.splitlines()) > cap:
            hard.append(f"Draft exceeds the requested {cap}-line body cap.")
        holdout = details["phases"]["holdout"]
        pairs = holdout["pairs"]
        if len(pairs) == holdout["expected"] and all(pair["review"] is not None for pair in pairs):
            before = median(pair["baseline"]["output_tokens"] for pair in pairs)
            after = median(pair["candidate"]["output_tokens"] for pair in pairs)
            reduction = 1 - (candidate_tokens + after) / (original_tokens + before)
            metrics.update(baseline_median_output=before, candidate_median_output=after,
                           combined_reduction=reduction)
            if reduction < self.config.communication_reduction:
                hard.append(f"Combined holdout token reduction is below {self.config.communication_reduction:.1%}.")
        else:
            hard.append("Combined holdout token reduction is unavailable.")
        if details["errors"]:
            hard.append("Errors remain; incomplete evidence cannot verify a draft.")
        reasons.extend(hard)
        result.decision = "REVIEW_REQUIRED" if reasons else "VERIFIED"
        result.message = ("Draft retained for review. " + " ".join(reasons)) if reasons else (
            "Draft passed direct and paired semantic checks and reduced combined holdout tokens."
        )
    def _adjudicate(self) -> None:
        """Adjudicate only quality disputes; never fill holes in execution evidence."""
        result, details = self.result, self.details
        details["preliminary"] = {"decision": result.decision, "reasons": list(details["reasons"])}
        if result.decision != "REVIEW_REQUIRED" or details.get("hard_blockers"):
            details["final_review"] = {"status": "SKIPPED", "reason": (
                "No disputed findings." if result.decision == "VERIFIED"
                else "Missing evidence or deterministic requirements cannot be overridden."
            )}
            return
        self._stage("final adjudication", 95)
        if not source_matches(result):
            details["final_review"] = {"status": "SKIPPED", "reason": "Source changed."}
            return
        candidate = result.candidate_body
        raw = Path(details["artifacts"]["run_draft"]).read_bytes()
        if (digest(candidate.encode("utf-8")) != details["candidate_sha256"]
                or digest(raw) != details["candidate_bytes_sha256"]):
            raise ValueError("frozen candidate changed before final adjudication")
        bundle = evidence_bundle(self.artifact.body, candidate, details, self.cases)
        atomic_json(self.root / "final-review-input.json", bundle)
        evidence_hash = digest((self.root / "final-review-input.json").read_bytes())
        if self.budget.calls >= self.budget.limit:
            self._error("no remaining call budget for final adjudication")
            details["final_review"] = {"status": "SKIPPED", "reason": "Call budget exhausted."}
            return
        review = self._structured(self.strong, FINAL, bundle, lambda value: validate_final(value, bundle))
        record = details["final_review"] = {
            "status": "COMPLETE" if review else "ERROR", "model": self.strong.name,
            "input_sha256": evidence_hash, "assessment": review,
        }
        atomic_json(self.root / "final-review.json", record)
        if not review:
            result.message = "Draft retained: final adjudication was incomplete. See errors and initial findings."
            return
        if review["decision"] == "CONFIRMED":
            result.decision = "CONFIRMED"
            details["confirmation_origin"] = "model"
            details["reasons"] = []
            result.message = (
                "Confirmed by final model adjudication; initial flags were resolved without "
                "material candidate regressions. Not independent or human verification. " + review["rationale"]
            )
        else:
            result.message = "Draft retained after final adjudication: " + review["rationale"]

    def _save_accepted(self) -> None:
        result, details = self.result, self.details
        if result.decision in ("VERIFIED", "CONFIRMED") and source_matches(result):
            artifacts = details["artifacts"]
            draft = Path(artifacts["run_draft"])
            if digest(draft.read_bytes()) != details["candidate_bytes_sha256"]:
                result.decision = "REVIEW_REQUIRED"
                result.message = "Frozen draft changed; accepted output was not published."
                self._error(result.message)
                return
            optimized = self.root / f"{self.artifact.path.stem}.{details['run_id']}.optimized.md"
            atomic_bytes(optimized, draft.read_bytes())
            artifacts["run_optimized"] = str(optimized)