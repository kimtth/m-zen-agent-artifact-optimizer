"""Step 5: GEPA feedback metric with quality gates before token reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dspy

from ..domain.core import (
    BehaviorContract,
    BehaviorResult,
    CaseEvaluation,
    EvaluationCase,
    RunRecord,
    count_tokens,
)
from ..pipeline.evaluation import Evaluator

_UNUSABLE_RULES = frozenset({"execution", "judge"})


@dataclass(frozen=True)
class Baseline:
    evaluation: CaseEvaluation
    instruction_tokens: int


class ZenMetric:
    def __init__(
        self,
        contract: BehaviorContract,
        evaluator: Evaluator,
        baselines: dict[str, Baseline],
        original_instructions: str,
    ):
        self.contract = contract
        self.evaluator = evaluator
        self.baselines = baselines
        self.original_instructions = original_instructions

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
        program_trace: Any = None,
    ) -> dspy.Prediction:
        del pred_name, program_trace
        case: EvaluationCase = gold.case
        answer = str(getattr(pred, "answer", ""))
        instructions = (
            getattr(pred, "zen_instructions", None)
            or _instructions_from_trace(pred_trace or trace)
            or self.original_instructions
        )
        run = RunRecord(
            case_id=case.id,
            answer=answer,
            input_tokens=count_tokens(instructions + case.inquiry),
            output_tokens=count_tokens(answer),
            latency_ms=0,
            error="" if answer else "empty answer",
        )
        evaluation = self.evaluator.evaluate(self.contract, case, run)
        baseline = self.baselines[case.id]
        input_reduction = _reduction(
            baseline.instruction_tokens, count_tokens(instructions)
        )
        output_reduction = _reduction(
            baseline.evaluation.output_tokens, evaluation.output_tokens
        )

        # A real artifact rarely satisfies every derived rule, so a hard zero on any
        # critical miss flattens the search space and leaves GEPA no gradient. Only an
        # unusable answer scores zero; every other result is graded.
        if _unusable(evaluation.behavior, answer):
            score = 0.0
        elif not evaluation.behavior.passed or not evaluation.understanding.passed:
            score = 0.79 * _quality(evaluation)
        else:
            score = 0.90 + 0.05 * input_reduction + 0.05 * output_reduction
        score = max(0.0, min(1.0, score))
        feedback = (
            f"{evaluation.feedback}\n"
            f"- Instruction reduction versus baseline: {input_reduction:.1%}.\n"
            f"- Output reduction versus baseline: {output_reduction:.1%}."
        )
        return dspy.Prediction(score=score, feedback=feedback)


def _unusable(behavior: BehaviorResult, answer: str) -> bool:
    """True when no answer exists or the judge could not evaluate it."""
    if not answer.strip():
        return True
    return any(
        not check.passed and check.rule in _UNUSABLE_RULES for check in behavior.checks
    )


def _quality(evaluation: CaseEvaluation) -> float:
    """Grade critical rules, all rules, and reader understanding together."""
    checks = evaluation.behavior.checks
    critical = [check for check in checks if check.severity == "critical"]
    critical_rate = (
        sum(check.passed for check in critical) / len(critical) if critical else 1.0
    )
    behavior_rate = sum(check.passed for check in checks) / max(len(checks), 1)
    return (critical_rate + behavior_rate + evaluation.understanding.accuracy) / 3


def _instructions_from_trace(trace: Any) -> str | None:
    if not trace:
        return None
    for item in trace:
        try:
            module = item[0]
            instructions = module.signature.instructions
        except (AttributeError, IndexError, TypeError):
            continue
        if isinstance(instructions, str):
            return instructions
    return None


def _reduction(baseline: int, candidate: int) -> float:
    if baseline <= 0:
        return 0.0
    value = (baseline - candidate) / baseline
    return max(-1.0, min(1.0, value))
