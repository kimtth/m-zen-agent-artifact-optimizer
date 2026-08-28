"""Step 4: rule-driven behavior and reader-understanding evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..domain.core import (
    BehaviorContract,
    BehaviorResult,
    CaseEvaluation,
    Check,
    EvaluationCase,
    RunRecord,
    UnderstandingAnswer,
    UnderstandingResult,
    count_tokens,
    parse_json,
    write_json,
)
from ..runtime.lm import BudgetExceeded, TextModel

_BEHAVIOR_PROMPT_VERSION = "behavior-judge-v1"
_READER_PROMPT_VERSION = "reader-judge-v1"


class EvaluationCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"


class Evaluator:
    def __init__(self, model: TextModel, cache: EvaluationCache):
        self.model = model
        self.cache = cache

    def evaluate(
        self,
        contract: BehaviorContract,
        case: EvaluationCase,
        run: RunRecord,
    ) -> CaseEvaluation:
        key = self._key(contract, case, run)
        path = self.cache.path_for(key)
        if path.is_file():
            return _evaluation_from_dict(json.loads(path.read_text(encoding="utf-8")))

        behavior = self._behavior(contract, case, run)
        understanding = self._understanding(contract, case, run)
        feedback = _feedback(behavior, understanding, run)
        result = CaseEvaluation(case.id, behavior, understanding, run.output_tokens, feedback)
        write_json(path, _evaluation_to_dict(result))
        return result

    def _behavior(
        self, contract: BehaviorContract, case: EvaluationCase, run: RunRecord
    ) -> BehaviorResult:
        deterministic = _check_constraints(case, run)
        if run.error:
            deterministic.append(Check("execution", False, "critical", run.error, "Produce a valid answer."))
        relevant = {rule.id: rule for rule in contract.obligations if rule.id in case.obligations}
        system = """Judge an answer against supplied rules and semantic criteria. Return JSON only.
For every listed obligation return one check with rule, passed, evidence, and feedback.
Evidence for a passed check must be an exact quote from the answer. Judge meaning, not exact
wording. Also return criteria_passed and criteria_feedback for must_include, must_not, and
explicit prohibitions. Do not reward brevity when required meaning is absent.
Schema: {"checks":[{"rule":"O1","passed":true,"evidence":"exact quote","feedback":"..."}],"criteria_passed":true,"criteria_feedback":"..."}"""
        user = json.dumps(
            {
                "prompt_version": _BEHAVIOR_PROMPT_VERSION,
                "language": contract.language,
                "purpose": contract.purpose,
                "obligations": [rule.__dict__ for rule in relevant.values()],
                "prohibitions": [rule.__dict__ for rule in contract.prohibitions],
                "must_include": case.must_include,
                "must_not": case.must_not,
                "inquiry": case.inquiry,
                "context": case.context,
                "answer": run.answer,
            },
            ensure_ascii=False,
        )
        try:
            value = parse_json(self.model.complete(system, user).text)
            semantic = _semantic_checks(value, relevant, run.answer)
            criteria_passed = bool(value.get("criteria_passed")) if isinstance(value, dict) else False
            feedback = str(
                value.get("criteria_feedback", "") if isinstance(value, dict) else ""
            ).strip()
            semantic.append(
                Check(
                    "case criteria",
                    criteria_passed,
                    "critical",
                    None,
                    feedback or ("Case criteria passed." if criteria_passed else "semantic criteria failed"),
                )
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - malformed judge output is an evaluation failure.
            semantic = [
                Check(rule.id, False, rule.severity, None, f"judge failure: {exc}")
                for rule in relevant.values()
            ]
            semantic.append(Check("judge", False, "critical", None, str(exc)))
        checks = (*deterministic, *semantic)
        passed = bool(checks) and all(check.passed for check in checks)
        critical_failure = any(not check.passed and check.severity == "critical" for check in checks)
        return BehaviorResult(passed, critical_failure, checks)

    def _understanding(
        self, contract: BehaviorContract, case: EvaluationCase, run: RunRecord
    ) -> UnderstandingResult:
        applicable = [question for question in case.reader_questions if question.applicable]
        if run.error or not applicable:
            return UnderstandingResult(False, 0.0, run.output_tokens, ())
        system = """Act as a reader who sees only the answer and the supplied questions.
Return JSON only. For each question, say whether the answer lets you answer it correctly
under its criterion. When correct, evidence must be an exact quote from the answer.
Schema: {"answers":[{"id":"what","correct":true,"evidence":"exact quote"}]}"""
        user = json.dumps(
            {
                "prompt_version": _READER_PROMPT_VERSION,
                "language": contract.language,
                "answer": run.answer,
                "questions": [question.__dict__ for question in applicable],
            },
            ensure_ascii=False,
        )
        try:
            value = parse_json(self.model.complete(system, user).text)
            raw_answers = value.get("answers") if isinstance(value, dict) else None
            if not isinstance(raw_answers, list):
                raise TypeError("reader judge did not return answers")
            by_id = {
                str(item.get("id")): item for item in raw_answers if isinstance(item, dict)
            }
            answers = []
            last_position = 0
            for question in applicable:
                item = by_id.get(question.id, {})
                evidence = item.get("evidence") if isinstance(item.get("evidence"), str) else None
                correct = bool(item.get("correct")) and bool(evidence) and evidence in run.answer
                if correct and evidence is not None:
                    end = run.answer.find(evidence) + len(evidence)
                    last_position = max(last_position, count_tokens(run.answer[:end]))
                answers.append(UnderstandingAnswer(question.id, correct, evidence))
        except BudgetExceeded:
            raise
        except Exception:  # noqa: BLE001 - malformed judge output is an evaluation failure.
            answers = [UnderstandingAnswer(question.id, False, None) for question in applicable]
            last_position = run.output_tokens
        correct_count = sum(answer.correct for answer in answers)
        accuracy = correct_count / len(applicable)
        passed = correct_count == len(applicable)
        if not passed:
            last_position = max(last_position, run.output_tokens)
        return UnderstandingResult(passed, accuracy, last_position, tuple(answers))

    def _key(
        self, contract: BehaviorContract, case: EvaluationCase, run: RunRecord
    ) -> str:
        value = json.dumps(
            {
                "answer": run.answer,
                "error": run.error,
                "rubric": case.to_dict(),
                "contract": contract.to_dict(),
                "judge": self.model.name,
                "prompts": [_BEHAVIOR_PROMPT_VERSION, _READER_PROMPT_VERSION],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check_constraints(case: EvaluationCase, run: RunRecord) -> list[Check]:
    checks: list[Check] = []
    for constraint in case.constraints:
        passed = False
        evidence = None
        try:
            if constraint.kind == "max_output_tokens":
                passed = run.output_tokens <= int(constraint.value)
                evidence = f"{run.output_tokens} tokens"
            elif constraint.kind == "max_sentences":
                sentences = len([part for part in re.split(r"[.!?。！？]+", run.answer) if part.strip()])
                passed = sentences <= int(constraint.value)
                evidence = f"{sentences} sentences"
            elif constraint.kind == "required_sections":
                required = [str(item) for item in constraint.value]
                missing = [item for item in required if item.casefold() not in run.answer.casefold()]
                passed = not missing
                evidence = "all sections present" if passed else f"missing: {', '.join(missing)}"
            elif constraint.kind == "forbidden_phrases":
                forbidden = [str(item) for item in constraint.value]
                found = [item for item in forbidden if item.casefold() in run.answer.casefold()]
                passed = not found
                evidence = "none found" if passed else f"found: {', '.join(found)}"
            else:
                evidence = f"unsupported constraint kind: {constraint.kind}"
        except (TypeError, ValueError):
            evidence = f"invalid constraint value: {constraint.value!r}"
        checks.append(
            Check(
                constraint.id,
                passed,
                constraint.severity,
                evidence,
                "Constraint passed." if passed else "Satisfy the explicit output constraint.",
            )
        )
    return checks


def _semantic_checks(value: Any, rules: dict[str, Any], answer: str) -> list[Check]:
    items = value.get("checks") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise TypeError("behavior judge did not return checks")
    by_id = {str(item.get("rule")): item for item in items if isinstance(item, dict)}
    checks = []
    for rule_id, rule in rules.items():
        item = by_id.get(rule_id, {})
        evidence = item.get("evidence") if isinstance(item.get("evidence"), str) else None
        passed = bool(item.get("passed")) and bool(evidence) and evidence in answer
        checks.append(
            Check(
                rule_id,
                passed,
                rule.severity,
                evidence,
                str(item.get("feedback", "Rule was not demonstrated.")),
            )
        )
    return checks


def _feedback(
    behavior: BehaviorResult, understanding: UnderstandingResult, run: RunRecord
) -> str:
    failed = [check.feedback for check in behavior.checks if not check.passed]
    unclear = [answer.question for answer in understanding.answers if not answer.correct]
    lines = ["Behavior:"]
    lines.extend(f"- {message}" for message in failed)
    if not failed:
        lines.append("- All tested rules passed.")
    lines.append("Understanding:")
    lines.append(
        "- All reader questions were answerable."
        if not unclear
        else f"- Reader questions not answered: {', '.join(unclear)}."
    )
    lines.extend(
        [
            "Efficiency:",
            f"- Output used {run.output_tokens} tokens.",
            f"- Required evidence ended by token {understanding.tokens}.",
        ]
    )
    return "\n".join(lines)


def _evaluation_to_dict(value: CaseEvaluation) -> dict[str, Any]:
    return {
        "case_id": value.case_id,
        "behavior": {
            "passed": value.behavior.passed,
            "critical_failure": value.behavior.critical_failure,
            "checks": [check.__dict__ for check in value.behavior.checks],
        },
        "understanding": {
            "passed": value.understanding.passed,
            "accuracy": value.understanding.accuracy,
            "tokens": value.understanding.tokens,
            "answers": [answer.__dict__ for answer in value.understanding.answers],
        },
        "output_tokens": value.output_tokens,
        "feedback": value.feedback,
    }


def _evaluation_from_dict(value: dict[str, Any]) -> CaseEvaluation:
    behavior_value = value["behavior"]
    understanding_value = value["understanding"]
    behavior = BehaviorResult(
        bool(behavior_value["passed"]),
        bool(behavior_value["critical_failure"]),
        tuple(Check(**item) for item in behavior_value["checks"]),
    )
    understanding = UnderstandingResult(
        bool(understanding_value["passed"]),
        float(understanding_value["accuracy"]),
        int(understanding_value["tokens"]),
        tuple(UnderstandingAnswer(**item) for item in understanding_value["answers"]),
    )
    return CaseEvaluation(
        value["case_id"],
        behavior,
        understanding,
        int(value["output_tokens"]),
        value["feedback"],
    )
