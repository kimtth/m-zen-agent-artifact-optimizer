"""End-to-end orchestration for one dataset-free optimization run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import dspy

from ..domain.core import (
    Aggregate,
    Artifact,
    BehaviorResult,
    CaseEvaluation,
    Check,
    Dataset,
    EvaluationCase,
    OptimizationResult,
    OptimizeConfig,
    UnderstandingAnswer,
    UnderstandingResult,
    count_tokens,
    load_artifact,
    optimized_path,
    report_path,
    write_json,
)
from ..pipeline.evaluation import EvaluationCache, Evaluator
from ..pipeline.gate import aggregate, decide, evidence_problems, quality_reasons
from ..pipeline.synthesis import generate_cases, generate_contract, split_cases
from ..runtime.harness import ArtifactProgram, CandidatePolicy, RunCache, Runner
from ..runtime.lm import (
    BudgetExceeded,
    CallBudget,
    CopilotModel,
    DSPyModel,
    ResponseCache,
    TextModel,
    fresh_calls,
)
from ..runtime.progress import ProgressCallback
from .metric import Baseline, ZenMetric
from .proposer import CompressionProposer, validate_focus
from .report import render_report


class Optimizer:
    def __init__(
        self,
        config: OptimizeConfig,
        task_model: TextModel,
        strong_model: TextModel,
        generator_model: TextModel,
        budget: CallBudget,
        cache_root: Path,
        progress: ProgressCallback | None = None,
    ):
        self.config = config
        self.task_model = task_model
        self.strong_model = strong_model
        self.generator_model = generator_model
        self.budget = budget
        self.cache_root = cache_root
        self.runner = Runner(task_model, RunCache(cache_root / "model-runs"))
        self.evaluator = Evaluator(strong_model, EvaluationCache(cache_root / "judgments"))
        self.contract = None
        self.progress = progress or (lambda _percent, _message: None)

    def run(self, path: Path) -> OptimizationResult:
        # Run-local evidence never certifies a new run through historical replay.
        path = path.resolve()
        artifact = load_artifact(path)
        validate_focus(artifact.body, self.config.focus)
        self.runner.namespace = str(uuid4())
        run_directory = self._run_directory(artifact)
        run_directory.mkdir(parents=True, exist_ok=False)
        self.dataset = None
        self.contract = None
        self.details = {
            "source_sha256": hashlib.sha256(
                (b"\xef\xbb\xbf" if artifact.bom else b"") + artifact.source.encode("utf-8")
            ).hexdigest(),
            "optimizer": "GEPA",
            "focus": self.config.focus,
            "sampling": "fresh application calls; provider independence is not established",
            "token_provenance": "local o200k_base estimates, not provider usage or billing",
        }
        write_json(run_directory / "experiment.json", {
            **self.details, "config": asdict(self.config),
            "gate": (
                "aligned complete error-free trials; behavior and reader trial pass rates may each "
                "drop by at most 5 absolute percentage points, independently in validation and "
                "holdout (not pooled); no observed candidate critical failures; at least one "
                "successful behavior trial and one successful reader trial per candidate case; "
                "holdout artifact, median output, and median reader tokens must not increase; "
                f"combined artifact + median output reduction >= {self.config.communication_reduction:.1%}"
            ),
            "quality_rate_semantics": (
                "trial pass counts / total trials; majority-passing case counts are diagnostic only; "
                "for N aligned trials per phase, the rate gate alone permits floor(N/20) net lost "
                "passes per metric, with per-case and critical constraints potentially stricter; "
                "bounded observed quality loss, not a human or statistical guarantee"
            ),
        })
        try:
            result = self._run(artifact, run_directory)
        except Exception as exc:  # noqa: BLE001 - retain evidence and a decision on interrupted experiments.
            result = OptimizationResult(
                "INCONCLUSIVE", str(path), None, self.contract, self.dataset, None,
                self.budget.calls, str(run_directory), f"evaluation did not complete: {exc}",
            )
        result.details = {**self.details, **result.details}
        result.calls = self.budget.calls
        self._save_summary(run_directory, result)
        self.progress(100, f"Finished: {result.decision}")
        return result

    def _run(self, artifact: Artifact, run_directory: Path) -> OptimizationResult:
        self.progress(0, "Preparing artifact")
        path = artifact.path
        max_body_lines = self.config.max_body_lines(artifact.body)

        self.progress(3, "Deriving behavior contract")
        contract = generate_contract(artifact, self.strong_model)
        self.contract = contract
        self.progress(10, "Generating evaluation cases")
        cases = generate_cases(
            contract,
            artifact.body,
            self.generator_model,
            self.strong_model,
            self.config.categories,
            self.config.seed,
            lambda current, total, name: self.progress(
                10 + round(20 * current / total),
                f"Generating cases ({name}, {current}/{total})",
            ),
        )
        self.progress(31, "Splitting sealed dataset")
        dataset = split_cases(cases, self.config.sizes_for(len(cases)), self.config.seed)
        dataset = Dataset(
            dataset.train,
            dataset.validation,
            dataset.holdout,
            {
                **dataset.metadata,
                "models": {
                    "target": self.task_model.name,
                    "generator": self.generator_model.name,
                    "judge": self.strong_model.name,
                    "reflection": self.strong_model.name,
                },
                "mode": "aggressive" if self.config.aggressive else "standard",
                "max_body_lines": max_body_lines,
            },
        )
        write_json(run_directory / "contract.json", contract.to_dict())
        write_json(run_directory / "dataset.json", dataset.to_dict())
        self.dataset = dataset

        baseline_holdout = self._evaluate(
            artifact.body,
            dataset.holdout,
            self.config.holdout_repetitions,
            32,
            45,
            "Measuring baseline holdout",
            run_directory / "baseline-holdout.json",
        )
        self._seal_holdout(run_directory, baseline_holdout)
        baseline_aggregate = aggregate(
            baseline_holdout, artifact.tokens, self.config.holdout_repetitions,
            tuple(case.id for case in dataset.holdout),
        )
        if baseline_aggregate.evidence_errors:
            raise RuntimeError("; ".join(baseline_aggregate.evidence_errors))

        baseline_search = self._evaluate(
            artifact.body,
            (*dataset.train, *dataset.validation),
            1,
            45,
            60,
            "Measuring search baseline",
            run_directory / "baseline-search.json",
        )
        search_summary = aggregate(
            baseline_search, artifact.tokens, 1,
            tuple(case.id for case in (*dataset.train, *dataset.validation)),
        )
        if search_summary.evidence_errors:
            raise RuntimeError("; ".join(search_summary.evidence_errors))
        baselines = {
            evaluation.case_id: Baseline(evaluation, artifact.body_tokens)
            for evaluation in baseline_search
        }
        self.progress(60, "Optimizing instructions with GEPA")
        candidate_body = self._compile(artifact, contract, dataset, baselines, run_directory)
        self.progress(78, "Validating optimized candidate")
        try:
            CandidatePolicy(artifact.body, max_body_lines).validate(candidate_body)
            validate_focus(artifact.body, self.config.focus).validate(candidate_body)
        except ValueError as exc:
            result = OptimizationResult(
                decision="REJECT",
                artifact_path=str(path),
                candidate_body=None,
                contract=contract,
                dataset=dataset,
                gate=None,
                calls=self.budget.calls,
                run_directory=str(run_directory),
                message=f"the final GEPA candidate did not meet the requested limit: {exc}",
            )
            self._save_summary(run_directory, result)
            self.progress(100, "Finished: REJECT")
            return result

        validation_ids = {case.id for case in dataset.validation}
        baseline_validation = [
            evaluation
            for evaluation in baseline_search
            if evaluation.case_id in validation_ids
        ]
        candidate_validation = self._evaluate(
            candidate_body,
            dataset.validation,
            1,
            78,
            86,
            "Validating optimized candidate",
            run_directory / "candidate-validation.json",
        )
        validation_baseline = aggregate(
            baseline_validation, artifact.tokens, 1, tuple(case.id for case in dataset.validation)
        )
        validation_candidate = aggregate(
            candidate_validation,
            count_tokens(artifact.render(candidate_body)),
            1,
            tuple(case.id for case in dataset.validation),
        )
        problems = evidence_problems(validation_baseline, validation_candidate)
        reasons = problems or quality_reasons(validation_baseline, validation_candidate)
        before, after = _comparison_dicts(validation_baseline, validation_candidate)
        self.details.update({
            "validation_reasons": reasons,
            "validation_evidence_problems": problems,
            "validation_baseline": before,
            "validation_candidate": after,
        })
        if reasons:
            result = OptimizationResult(
                decision="INCONCLUSIVE" if problems else "REJECT",
                artifact_path=str(path),
                candidate_body=None,
                contract=contract,
                dataset=dataset,
                gate=None,
                calls=self.budget.calls,
                run_directory=str(run_directory),
                message="; ".join(reasons),
            )
            self._save_summary(run_directory, result)
            self.progress(100, f"Finished: {result.decision}")
            return result

        candidate_holdout = self._evaluate(
            candidate_body,
            dataset.holdout,
            self.config.holdout_repetitions,
            86,
            98,
            "Measuring candidate holdout",
            run_directory / "candidate-holdout.json",
        )
        candidate_aggregate = aggregate(
            candidate_holdout,
            count_tokens(artifact.render(candidate_body)),
            self.config.holdout_repetitions,
            tuple(case.id for case in dataset.holdout),
        )
        gate = decide(
            baseline_aggregate,
            candidate_aggregate,
            self.config.communication_reduction,
        )
        result = OptimizationResult(
            decision="INCONCLUSIVE" if gate.inconclusive else "ACCEPT" if gate.accepted else "REJECT",
            artifact_path=str(path),
            candidate_body=candidate_body if gate.accepted else None,
            contract=contract,
            dataset=dataset,
            gate=gate,
            calls=self.budget.calls,
            run_directory=str(run_directory),
            message="" if gate.accepted else "; ".join(gate.reasons),
        )
        if self.config.compare_concision:
            self.details["concision_control"] = self._control(artifact, dataset, run_directory)
        self._save_summary(run_directory, result)
        self.progress(100, f"Finished: {result.decision}")
        return result

    def _control(self, artifact: Artifact, dataset: Dataset, run_directory: Path) -> dict:
        """Measure a fixed control after search; never select it or expose it to GEPA."""
        suffix = (
            "Be concise where compatible with all preceding instructions. Preserve the required "
            "language, facts, warnings, uncertainty, and completeness; omit only unnecessary repetition."
        )
        body = artifact.body + "\n\n" + suffix
        write_json(run_directory / "concision-control.json", {
            "instruction": body, "purpose": "diagnostic only; not eligible for selection",
            "suffix_language": "English; requests preservation of the original output language",
        })
        try:
            values = self._evaluate(
                body, dataset.holdout, self.config.holdout_repetitions, 98, 99,
                "Measuring diagnostic concision control",
                run_directory / "control-holdout.json",
            )
        except BudgetExceeded as exc:
            return {"status": "INCONCLUSIVE", "reason": str(exc), "selected": False}
        summary = aggregate(
            values, count_tokens(artifact.render(body)), self.config.holdout_repetitions,
            tuple(c.id for c in dataset.holdout),
        )
        return {
            "status": "INCONCLUSIVE" if summary.evidence_errors else "MEASURED",
            "selected": False, "aggregate": summary.to_dict(),
        }

    def _compile(
        self,
        artifact: Artifact,
        contract,
        dataset: Dataset,
        baselines: dict[str, Baseline],
        run_directory: Path,
    ) -> str:
        # GEPA evaluates the source program before proposing a replacement. The source
        # may exceed an aggressive cap, so only proposals and the final candidate use it.
        policy = CandidatePolicy(artifact.body)
        max_body_lines = self.config.max_body_lines(artifact.body)
        program = ArtifactProgram(artifact.body, policy)
        task_lm = DSPyModel(self.task_model)
        reflection_lm = DSPyModel(self.strong_model)
        metric = ZenMetric(contract, self.evaluator, baselines, artifact.body)
        proposer = CompressionProposer(
            self.strong_model,
            contract,
            artifact.body,
            max_body_lines,
            focus=self.config.focus,
        )
        optimizer = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            instruction_proposer=proposer,
            reflection_minibatch_size=3,
            max_metric_calls=self.config.max_metric_calls,
            candidate_selection_strategy="pareto",
            use_merge=True,
            track_stats=True,
            seed=self.config.seed,
            num_threads=1,
            log_dir=str(run_directory / "gepa"),
            warn_on_score_mismatch=False,
        )
        train = [_example(case) for case in dataset.train]
        validation = [_example(case) for case in dataset.validation]
        adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
        try:
            with fresh_calls(), dspy.context(lm=task_lm, adapter=adapter):
                optimized = optimizer.compile(program, trainset=train, valset=validation)
        finally:
            write_json(run_directory / "proposal-audit.json", proposer.history)
        selected = str(optimized.answer.signature.instructions)
        if self.config.focus == "all":
            selected = selected.strip()
        return _within_limit(optimized, selected, max_body_lines)

    def _evaluate(
        self,
        instructions: str,
        cases: tuple[EvaluationCase, ...],
        repetitions: int,
        start_percent: int,
        end_percent: int,
        message: str,
        journal_path: Path | None = None,
    ) -> list[CaseEvaluation]:
        """Persist each completed trial when requested; exhaustion propagates without retry."""
        results = []
        journal = []
        if journal_path is not None:
            write_json(journal_path, journal)
        total = len(cases) * repetitions
        completed = 0
        scope = str(uuid4())
        self.progress(start_percent, message)
        for repetition in range(repetitions):
            for case in cases:
                status_message = message
                trial_id = ""
                try:
                    with fresh_calls():
                        run = self.runner.run(instructions, case, repetition, scope=scope)
                    trial_id = run.trial_id
                    if run.error:
                        evaluation = _failed_evaluation(case, run.error)
                        status_message = f"{message} (skipped {case.id})"
                    else:
                        if self.contract is None:
                            raise RuntimeError("behavior contract is not initialized")
                        with fresh_calls():
                            evaluation = self.evaluator.evaluate(self.contract, case, run)
                except BudgetExceeded:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad case must not restart a run.
                    evaluation = _failed_evaluation(case, str(exc))
                    status_message = f"{message} (skipped {case.id})"
                results.append(replace(evaluation, trial=repetition))
                if journal_path is not None:
                    journal.append(_evaluation_dict(results[-1], trial_id=trial_id))
                    write_json(journal_path, journal)
                completed += 1
                self.progress(
                    start_percent
                    + round((end_percent - start_percent) * completed / max(total, 1)),
                    status_message,
                )
        return results

    def _run_directory(self, artifact: Artifact) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        digest = hashlib.sha256(str(artifact.path.resolve()).encode("utf-8")).hexdigest()[:10]
        return self.cache_root / "runs" / f"{stamp}-{digest}"

    def _seal_holdout(
        self, run_directory: Path, evaluations: list[CaseEvaluation]
    ) -> None:
        path = run_directory / "baseline-holdout.json"
        # Seal the completed journal, including its run-trial provenance, without rewriting it.
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = [_evaluation_dict(item) for item in evaluations]
            write_json(path, value)
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        write_json(
            run_directory / "holdout-seal.json",
            {"sha256": hashlib.sha256(encoded).hexdigest()},
        )

    def _save_summary(self, run_directory: Path, result: OptimizationResult) -> None:
        result.details = {**getattr(self, "details", {}), **result.details}
        gate = None
        if result.gate:
            before, after = _comparison_dicts(result.gate.baseline, result.gate.candidate)
            unavailable = result.gate.inconclusive or bool(
                evidence_problems(result.gate.baseline, result.gate.candidate)
            )
            if unavailable:
                for stats in (before, after):
                    stats["behavior_pass_rate"] = stats["understanding_pass_rate"] = None
            gate = {
                **asdict(result.gate),
                "baseline": before,
                "candidate": after,
                "communication_reduction": (
                    None if unavailable else result.gate.communication_reduction
                ),
            }
        write_json(
            run_directory / "summary.json",
            {
                "decision": result.decision,
                "artifact": result.artifact_path,
                "calls": result.calls,
                "message": result.message,
                "gate": gate,
                "details": result.details,
            },
        )


def _comparison_dicts(baseline: Aggregate, candidate: Aggregate) -> tuple[dict, dict]:
    before, after = baseline.to_dict(), candidate.to_dict()
    if evidence_problems(baseline, candidate):
        for stats in (before, after):
            stats["behavior_pass_rate"] = stats["understanding_pass_rate"] = None
    return before, after


def optimize(
    path: Path,
    config: OptimizeConfig,
    cache_root: Path | None = None,
    progress: ProgressCallback | None = None,
) -> OptimizationResult:
    if config.engine == "semantic":
        if config.max_metric_calls is not None or config.compare_concision:
            raise ValueError("max_metric_calls and compare_concision require engine='gepa'")
        from .semantic import SemanticOptimizer

        optimizer_type = SemanticOptimizer
    else:
        optimizer_type = Optimizer
        # Legacy GEPA serializes its config as JSON; publication remains a separate step.
        config = replace(config, output_dir=None)
    artifact_path = path.resolve()
    root = cache_root or Path.cwd() / ".zen-cache"
    budget = CallBudget(config.total_call_budget)
    responses = ResponseCache(root / "model-calls")
    task = CopilotModel(
        config.target_model,
        budget,
        working_directory=artifact_path.parent,
        cache=responses,
    )
    strong = CopilotModel(
        config.strong_model,
        budget,
        working_directory=artifact_path.parent,
        cache=responses,
    )
    generator = CopilotModel(
        config.generator_model,
        budget,
        working_directory=artifact_path.parent,
        cache=responses,
    )
    optimizer = optimizer_type(config, task, strong, generator, budget, root, progress)
    return optimizer.run(artifact_path)


def write_outputs(
    result: OptimizationResult, output_directory: Path | None = None
) -> tuple[Path | None, Path]:
    if getattr(result, "details", {}).get("engine") == "semantic":
        from .semantic import write_semantic_outputs

        return write_semantic_outputs(result, output_directory)
    source = Path(result.artifact_path)
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
    target = source if output_directory is None else output_directory / source.name
    report = report_path(target)
    candidate_path = optimized_path(target)
    artifact = None
    if result.decision == "ACCEPT":
        expected = getattr(result, "details", {}).get("source_sha256")
        try:
            artifact = load_artifact(source)
        except (OSError, ValueError):
            artifact = None
        actual = hashlib.sha256(
            (b"\xef\xbb\xbf" if artifact.bom else b"") + artifact.source.encode("utf-8")
        ).hexdigest() if artifact else None
        if not expected or actual != expected:
            result.decision = "INCONCLUSIVE"
            result.candidate_body = None
            result.gate = None
            result.message = "source changed or its evaluated hash is unavailable; candidate was not written"
            run_directory = getattr(result, "run_directory", "")
            if run_directory and Path(run_directory).is_dir():
                write_json(Path(run_directory) / "summary.json", {
                    "decision": result.decision, "artifact": result.artifact_path,
                    "calls": result.calls, "message": result.message, "gate": None,
                    "details": result.details,
                })
    report.write_text(render_report(result), encoding="utf-8", newline="\n")
    if result.decision == "ACCEPT" and result.candidate_body is not None:
        assert artifact is not None  # Render only the snapshot whose hash was verified.
        candidate_path.write_bytes(artifact.candidate_bytes(result.candidate_body))
        return candidate_path, report
    candidate_path.unlink(missing_ok=True)
    return None, report


def _within_limit(optimized: object, selected: str, max_lines: int | None) -> str:
    # GEPA returns its highest-scoring program, which can be the unmodified source.
    # An aggressive run still needs the best proposal that also meets the line limit.
    if max_lines is None or len(selected.splitlines()) <= max_lines:
        return selected
    results = getattr(optimized, "detailed_results", None)
    if results is None:
        return selected
    ranked = sorted(
        zip(results.candidates, results.val_aggregate_scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for candidate, _score in ranked:
        body = (
            str(candidate.get("answer", ""))
            if isinstance(candidate, dict)
            else str(candidate.answer.signature.instructions)
        )
        if body.strip() and len(body.splitlines()) <= max_lines:
            return body
    return selected


def _example(case: EvaluationCase) -> dspy.Example:
    context = json.dumps(case.context, ensure_ascii=False, sort_keys=True)
    return dspy.Example(inquiry=case.inquiry, context=context, case=case).with_inputs(
        "inquiry", "context"
    )


def _evaluation_dict(value: CaseEvaluation, *, trial_id: str | None = None) -> dict[str, object]:
    return {
    **asdict(value),
    **({"details": {"trial_id": trial_id}} if trial_id is not None else {}),
        "case_id": value.case_id,
        "behavior_passed": value.behavior.passed,
        "critical_failure": value.behavior.critical_failure,
        "understanding_passed": value.understanding.passed,
        "understanding_accuracy": value.understanding.accuracy,
        "understanding_tokens": value.understanding.tokens,
        "output_tokens": value.output_tokens,
        "feedback": value.feedback,
    }


def _failed_evaluation(case: EvaluationCase, error: str) -> CaseEvaluation:
    message = error.strip() or "unknown model error"
    behavior = BehaviorResult(
        False,
        True,
        (Check("execution", False, "critical", None, f"Skipped case: {message}"),),
    )
    questions = tuple(
        UnderstandingAnswer(question.id, False, None)
        for question in case.reader_questions
        if question.applicable
    )
    understanding = UnderstandingResult(False, 0.0, 0, questions)
    return CaseEvaluation(
        case.id,
        behavior,
        understanding,
        0,
        f"Case skipped after model error: {message}",
        error=message,
    )
