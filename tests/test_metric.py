from __future__ import annotations

import dspy

from zen.domain.core import (
    BehaviorContract,
    BehaviorResult,
    CaseEvaluation,
    Check,
    EvaluationCase,
    ReaderQuestion,
    Rule,
    UnderstandingAnswer,
    UnderstandingResult,
)
from zen.optimization.metric import Baseline, ZenMetric

_CONTRACT = BehaviorContract(
    "English",
    "Explain the result",
    (Rule("O1", "Explain the result", "Explain the result."),),
)
_CASE = EvaluationCase(
    "case",
    "normal",
    "family",
    "What changed?",
    {},
    ("O1", "O2"),
    (),
    (),
    (ReaderQuestion("what", "What changed?", "The answer states the change."),),
)


class _StubEvaluator:
    def __init__(self, evaluation: CaseEvaluation):
        self.evaluation = evaluation

    def evaluate(self, _contract, _case, _run) -> CaseEvaluation:
        return self.evaluation


def _evaluation(passed_rules: int, total_rules: int, accuracy: float) -> CaseEvaluation:
    checks = tuple(
        Check(f"O{index + 1}", index < passed_rules, "critical", "quote", "feedback")
        for index in range(total_rules)
    )
    behavior = BehaviorResult(passed_rules == total_rules, passed_rules < total_rules, checks)
    understanding = UnderstandingResult(
        accuracy == 1.0,
        accuracy,
        4,
        (UnderstandingAnswer("what", accuracy == 1.0, "quote"),),
    )
    return CaseEvaluation("case", behavior, understanding, 20, "feedback")


def _score(evaluation: CaseEvaluation, answer: str = "The cache changed.") -> float:
    metric = ZenMetric(
        _CONTRACT,
        _StubEvaluator(evaluation),
        {"case": Baseline(_evaluation(2, 2, 1.0), 40)},
        "Explain the result.",
    )
    prediction = dspy.Prediction(answer=answer, zen_instructions="Explain the result.")
    return float(metric(dspy.Example(case=_CASE), prediction).score)


def test_partial_rule_failures_are_graded_not_zeroed() -> None:
    worse = _score(_evaluation(1, 3, 0.0))
    better = _score(_evaluation(2, 3, 1.0))

    assert 0 < worse < better < 0.79


def test_unusable_answer_scores_zero() -> None:
    unusable = _evaluation(2, 2, 1.0)
    judge_failed = CaseEvaluation(
        unusable.case_id,
        BehaviorResult(
            False,
            True,
            (Check("judge", False, "critical", None, "judge failure"),),
        ),
        unusable.understanding,
        unusable.output_tokens,
        unusable.feedback,
    )

    assert _score(judge_failed) == 0.0
    assert _score(_evaluation(2, 2, 1.0), answer="  ") == 0.0


def test_full_quality_outranks_every_partial_result() -> None:
    assert _score(_evaluation(2, 2, 1.0)) >= 0.90
