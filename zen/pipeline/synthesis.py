"""Steps 1 and 2: generate a grounded contract and multilingual cases."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from functools import cache
from typing import Any

from ..domain.core import (
    Artifact,
    BehaviorContract,
    CaseCategory,
    Constraint,
    Dataset,
    EvaluationCase,
    ReaderQuestion,
    Rule,
    parse_json,
)
from ..runtime.lm import TextModel

_CONTRACT_PROMPT_VERSION = "contract-v1"
_CASE_PROMPT_VERSION = "cases-v1"
_VALIDATOR_PROMPT_VERSION = "case-validator-v1"
_CASE_BATCH_SIZE = 10
_GENERATION_ATTEMPTS = 3
_CATEGORY_REFILL_ATTEMPTS = 6
_TOP_UP_SEED_OFFSET = 1000
_MINIMUM_CASES = 3


class SynthesisError(ValueError):
    """Raised when generated data is malformed or unsupported by the source."""


def generate_contract(artifact: Artifact, model: TextModel) -> BehaviorContract:
    system = """You create a testable behavior contract from one instruction artifact.
Return JSON only. Use the artifact's own language for purpose, statements, and evidence.
Every source_evidence value must be a byte-for-byte substring of the mutable body,
including punctuation and formatting. Never invent a rule.
Split independent requirements into atomic obligations. Mark required behavior as critical
and stylistic preferences as preference. Prohibitions are explicit things the artifact says
not to do. The schema is:
{"language":"...","purpose":"...","obligations":[{"id":"O1","statement":"...","severity":"critical|preference","source_evidence":"exact quote"}],"prohibitions":[{"id":"P1","statement":"...","severity":"critical|preference","source_evidence":"exact quote"}]}"""
    user = json.dumps(
        {"prompt_version": _CONTRACT_PROMPT_VERSION, "mutable_body": artifact.body},
        ensure_ascii=False,
    )
    error: (ValueError | TypeError) | None = None
    for _ in range(_GENERATION_ATTEMPTS):
        try:
            return parse_contract(parse_json(model.complete(system, user).text), artifact.body)
        except (ValueError, TypeError) as exc:
            error = exc
    raise SynthesisError(f"contract generation failed after {_GENERATION_ATTEMPTS} attempts: {error}")


def parse_contract(value: Any, source_body: str) -> BehaviorContract:
    if not isinstance(value, dict):
        raise SynthesisError("behavior contract must be a JSON object")
    language = _text(value, "language")
    purpose = _text(value, "purpose")
    obligations = _read_rules(value.get("obligations"), source_body, "obligations")
    prohibitions = _read_rules(value.get("prohibitions", []), source_body, "prohibitions")
    if not obligations:
        raise SynthesisError("behavior contract has no obligations")
    ids = [rule.id for rule in (*obligations, *prohibitions)]
    if len(ids) != len(set(ids)):
        raise SynthesisError("behavior contract rule identifiers must be unique")
    return BehaviorContract(language, purpose, obligations, prohibitions)


def _read_rules(value: Any, source: str, label: str) -> tuple[Rule, ...]:
    if not isinstance(value, list):
        raise SynthesisError(f"{label} must be an array")
    rules = []
    for item in value:
        if not isinstance(item, dict):
            raise SynthesisError(f"every {label} entry must be an object")
        severity = str(item.get("severity", "critical")).strip()
        if severity not in {"critical", "preference"}:
            raise SynthesisError(f"invalid rule severity: {severity}")
        rule = Rule(
            id=_text(item, "id"),
            statement=_text(item, "statement"),
            source_evidence=_text(item, "source_evidence"),
            severity=severity,
        )
        if rule.source_evidence not in source:
            raise SynthesisError(f"{rule.id} cites text that is not in the artifact body")
        rules.append(rule)
    return tuple(rules)


def generate_cases(
    contract: BehaviorContract,
    source_body: str,
    generator: TextModel,
    validator: TextModel,
    categories: tuple[CaseCategory, ...],
    seed: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[EvaluationCase]:
    accepted: list[EvaluationCase] = []
    generated_counts: dict[str, int] = {category.name: 0 for category in categories}
    for category_index, category in enumerate(categories, start=1):
        category_cases: list[EvaluationCase] = []
        for attempt in range(_CATEGORY_REFILL_ATTEMPTS):
            needed = category.raw if not category_cases else category.retained - len(category_cases)
            if needed <= 0:
                break
            try:
                generated = _generate_category(
                    contract,
                    category,
                    generator,
                    seed + attempt,
                    needed,
                    generated_counts[category.name],
                )
                generated_counts[category.name] += len(generated)
                candidates = _deterministic_case_filter(
                    [*category_cases, *generated], contract, source_body, categories
                )
                generated_ids = {case.id for case in generated}
                candidates = [case for case in candidates if case.id in generated_ids]
                accepted_ids = _semantic_case_filter(candidates, contract, validator)
            except SynthesisError:
                continue
            category_cases.extend(case for case in candidates if case.id in accepted_ids)
        accepted.extend(category_cases)
        if progress is not None:
            label = category.name
            if len(category_cases) < category.retained:
                label = f"{category.name} short {len(category_cases)}/{category.retained}"
            progress(category_index, len(categories), label)

    rng = random.Random(seed)
    retained: list[EvaluationCase] = []
    spare: list[EvaluationCase] = []
    by_category: dict[str, list[EvaluationCase]] = defaultdict(list)
    for case in accepted:
        by_category[case.category].append(case)
    for category in categories:
        choices = by_category[category.name]
        rng.shuffle(choices)
        retained.extend(choices[: category.retained])
        spare.extend(choices[category.retained :])
    required = sum(category.retained for category in categories)
    if len(retained) < required:
        rng.shuffle(spare)
        retained.extend(spare[: required - len(retained)])
    if len(retained) < required:
        retained.extend(
            _top_up(
                contract,
                source_body,
                generator,
                validator,
                categories,
                seed,
                generated_counts,
                retained,
                required - len(retained),
            )
        )
    minimum = min(required, _MINIMUM_CASES)
    if len(retained) < minimum:
        raise SynthesisError(
            f"case generation produced {len(retained)} valid cases; "
            f"at least {minimum} are required. The generator or validator "
            "rejected nearly every case for this artifact."
        )
    return retained


def _top_up(
    contract: BehaviorContract,
    source_body: str,
    generator: TextModel,
    validator: TextModel,
    categories: tuple[CaseCategory, ...],
    seed: int,
    generated_counts: dict[str, int],
    retained: list[EvaluationCase],
    deficit: int,
) -> list[EvaluationCase]:
    """Replace cases that low-yield categories could not supply."""
    result: list[EvaluationCase] = []
    order = sorted(categories, key=lambda category: category.retained, reverse=True)
    for category in order:
        for attempt in range(_CATEGORY_REFILL_ATTEMPTS):
            missing = deficit - len(result)
            if missing <= 0:
                return result
            try:
                generated = _generate_category(
                    contract,
                    category,
                    generator,
                    seed + _TOP_UP_SEED_OFFSET + attempt,
                    missing,
                    generated_counts[category.name],
                )
                generated_counts[category.name] += len(generated)
                candidates = _deterministic_case_filter(
                    [*retained, *result, *generated], contract, source_body, categories
                )
                generated_ids = {case.id for case in generated}
                candidates = [case for case in candidates if case.id in generated_ids]
                accepted_ids = _semantic_case_filter(candidates, contract, validator)
            except SynthesisError:
                continue
            result.extend(case for case in candidates if case.id in accepted_ids)
    return result[:deficit]


def _generate_category(
    contract: BehaviorContract,
    category: CaseCategory,
    model: TextModel,
    seed: int,
    total: int,
    id_offset: int,
) -> list[EvaluationCase]:
    system = """Generate evaluation cases, not canonical answers. Return JSON only.
Write inquiries, context, criteria, and reader questions in the contract language.
Each case must be answerable from its context and test listed obligation IDs.
When the category tests non-activation, list only the obligations that still apply and
express the expected restraint through must_not; obligations may be an empty array.
Use semantic must_include and must_not criteria; do not require one exact wording.
Create three localized reader questions that let a reader identify what the answer says,
why, and why it matters when applicable. Mark a question inapplicable when the contract
does not require it. Similar scenarios share a family name.
Optional deterministic constraints use objects with id, kind, value, and severity.
Supported kinds are max_output_tokens, max_sentences, required_sections, forbidden_phrases.
Schema: {"cases":[{"family":"...","inquiry":"...","context":{},"obligations":["O1"],"must_include":["semantic criterion"],"must_not":[],"reader_questions":[{"id":"what","question":"...","criterion":"...","applicable":true}],"constraints":[]}]}"""
    visible_contract = {
        "language": contract.language,
        "purpose": contract.purpose,
        "obligations": [
            {"id": rule.id, "statement": rule.statement, "severity": rule.severity}
            for rule in contract.obligations
        ],
        "prohibitions": [
            {"id": rule.id, "statement": rule.statement, "severity": rule.severity}
            for rule in contract.prohibitions
        ],
    }
    items: list[Any] = []
    batch = 0
    while len(items) < total:
        batch += 1
        count = min(_CASE_BATCH_SIZE, total - len(items))
        user = json.dumps(
            {
                "prompt_version": _CASE_PROMPT_VERSION,
                "seed": seed,
                "batch": batch,
                "case_offset": id_offset + len(items),
                "persona": category.persona,
                "category": category.name,
                "category_purpose": category.purpose,
                "count": count,
                "contract": visible_contract,
            },
            ensure_ascii=False,
        )
        generated = None
        for _ in range(_GENERATION_ATTEMPTS):
            value = _complete_json(model, system, user)
            candidate = value.get("cases") if isinstance(value, dict) else None
            if isinstance(candidate, list) and candidate:
                generated = candidate
                break
        if generated is None:
            raise SynthesisError(
                f"generator returned no {category.name} cases in batch {batch} "
                f"after {_GENERATION_ATTEMPTS} attempts"
            )
        items.extend(generated[:count])
    return [
        _parse_case(item, f"{category.name}-{id_offset + index + 1:03}", category.name)
        for index, item in enumerate(items)
    ]


def _parse_case(value: Any, case_id: str, category: str) -> EvaluationCase:
    if not isinstance(value, dict):
        raise SynthesisError(f"{case_id} must be an object")
    context = value.get("context")
    if not isinstance(context, dict):
        raise SynthesisError(f"{case_id}.context must be an object")
    questions = value.get("reader_questions")
    if not isinstance(questions, list) or not questions:
        raise SynthesisError(f"{case_id} needs reader questions")
    constraints = value.get("constraints", [])
    if not isinstance(constraints, list):
        raise SynthesisError(f"{case_id}.constraints must be an array")
    return EvaluationCase(
        id=case_id,
        category=category,
        family=_text(value, "family"),
        inquiry=_text(value, "inquiry"),
        context=context,
        obligations=_string_tuple(value.get("obligations"), f"{case_id}.obligations"),
        must_include=_string_tuple(value.get("must_include", []), f"{case_id}.must_include"),
        must_not=_string_tuple(value.get("must_not", []), f"{case_id}.must_not"),
        reader_questions=tuple(
            ReaderQuestion(
                _text(question, "id"),
                _text(question, "question"),
                _text(question, "criterion"),
                bool(question.get("applicable", True)),
            )
            for question in questions
            if isinstance(question, dict)
        ),
        constraints=tuple(
            Constraint(
                _text(item, "id"),
                _text(item, "kind"),
                item.get("value"),
                str(item.get("severity", "critical")),
            )
            for item in constraints
            if isinstance(item, dict)
        ),
    )


def _deterministic_case_filter(
    cases: Iterable[EvaluationCase],
    contract: BehaviorContract,
    source_body: str,
    categories: tuple[CaseCategory, ...] = (),
) -> list[EvaluationCase]:
    obligation_ids = {rule.id for rule in contract.obligations}
    optional_obligations = {
        category.name for category in categories if not category.requires_obligations
    }
    seen: set[str] = set()
    result = []
    source = source_body.strip()
    for case in cases:
        fingerprint = json.dumps(
            [case.inquiry.casefold(), case.context], ensure_ascii=False, sort_keys=True
        )
        if fingerprint in seen:
            continue
        if not case.obligations and case.category not in optional_obligations:
            continue
        if not set(case.obligations) <= obligation_ids:
            continue
        if len(source) >= 80 and source in json.dumps(case.to_dict(), ensure_ascii=False):
            continue
        if not case.reader_questions or not any(q.applicable for q in case.reader_questions):
            continue
        seen.add(fingerprint)
        result.append(case)
    return result


def _semantic_case_filter(
    cases: list[EvaluationCase], contract: BehaviorContract, model: TextModel
) -> set[str]:
    system = """Validate synthetic evaluation cases. Return JSON only as {"accepted":[ids],"rejected":[{"id":"...","reason":"..."}]}.
Accept a case only when it tests its listed obligations, is answerable from context, uses
semantic rather than exact-wording criteria, is internally consistent unless conflict
handling is the point, and gives a judge enough information to decide pass or fail.
A case that tests non-activation may list no obligations; accept it when its must_not or
prohibition criteria make the expected restraint checkable.
Accept every case that is usable. Reject only genuinely broken cases.
Do not rewrite cases."""
    known = {case.id for case in cases}
    result: set[str] = set()
    for batch in _batches(cases, _CASE_BATCH_SIZE):
        user = json.dumps(
            {
                "prompt_version": _VALIDATOR_PROMPT_VERSION,
                "contract": contract.to_dict(),
                "cases": [case.to_dict() for case in batch],
            },
            ensure_ascii=False,
        )
        value = _complete_json(model, system, user)
        accepted = value.get("accepted") if isinstance(value, dict) else None
        if not isinstance(accepted, list):
            raise SynthesisError("case validator did not return an accepted array")
        result.update(str(case_id) for case_id in accepted)
    return result & known


def _batches[T](values: list[T], size: int) -> list[list[T]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _complete_json(model: TextModel, system: str, user: str) -> Any:
    error: ValueError | None = None
    for _ in range(_GENERATION_ATTEMPTS):
        try:
            return parse_json(model.complete(system, user).text)
        except ValueError as exc:
            error = exc
    raise SynthesisError(
        f"structured generation failed after {_GENERATION_ATTEMPTS} attempts: {error}"
    )


def split_cases(
    cases: list[EvaluationCase], sizes: tuple[int, int, int], seed: int
) -> Dataset:
    if sum(sizes) != len(cases):
        raise SynthesisError("split sizes do not match the retained case count")
    families: dict[str, list[EvaluationCase]] = defaultdict(list)
    for case in cases:
        families[case.family].append(case)
    groups = list(families.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    @cache
    def assign(index: int, train: int, validation: int, holdout: int) -> tuple[int, ...] | None:
        if index == len(groups):
            return () if (train, validation, holdout) == (0, 0, 0) else None
        size = len(groups[index])
        remaining = (train, validation, holdout)
        for bucket, count in enumerate(remaining):
            if size > count:
                continue
            next_remaining = list(remaining)
            next_remaining[bucket] -= size
            tail = assign(index + 1, *next_remaining)
            if tail is not None:
                return (bucket, *tail)
        return None

    assignment = assign(0, *sizes)
    buckets: list[list[EvaluationCase]] = [[], [], []]
    if assignment is None:
        if len(groups) < 3:
            raise SynthesisError(
                "case families cannot form a split; the generator produced too few "
                "distinct scenario families"
            )
        for group in groups:
            target = min(range(3), key=lambda index: len(buckets[index]) / max(sizes[index], 1))
            buckets[target].extend(group)
    else:
        for bucket, group in zip(assignment, groups, strict=True):
            buckets[bucket].extend(group)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "prompt_versions": {
            "contract": _CONTRACT_PROMPT_VERSION,
            "cases": _CASE_PROMPT_VERSION,
            "validator": _VALIDATOR_PROMPT_VERSION,
        },
    }
    return Dataset(tuple(buckets[0]), tuple(buckets[1]), tuple(buckets[2]), metadata)


def _text(value: Any, key: str) -> str:
    if not isinstance(value, dict):
        raise SynthesisError(f"expected object containing {key}")
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise SynthesisError(f"{key} must be a non-empty string")
    return text.strip()


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise SynthesisError(f"{label} must be an array of non-empty strings")
    return tuple(item.strip() for item in value)
