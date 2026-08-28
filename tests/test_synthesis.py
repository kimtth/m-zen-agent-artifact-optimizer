from __future__ import annotations

import json
from types import SimpleNamespace

from zen.domain.core import (
    QUICK_CATEGORIES,
    BehaviorContract,
    CaseCategory,
    EvaluationCase,
    OptimizeConfig,
    ReaderQuestion,
    Rule,
)
from zen.pipeline.synthesis import (
    SynthesisError,
    generate_cases,
    parse_contract,
    split_cases,
)


def _contract(evidence: str) -> dict[str, object]:
    return {
        "language": "한국어",
        "purpose": "간결한 결과",
        "obligations": [
            {
                "id": "O1",
                "statement": "결과를 먼저 쓴다",
                "severity": "critical",
                "source_evidence": evidence,
            }
        ],
        "prohibitions": [],
    }


def test_contract_accepts_grounded_non_english_rule() -> None:
    source = "결과를 먼저 쓰세요."
    contract = parse_contract(_contract(source), source)
    assert contract.language == "한국어"
    assert contract.obligations[0].source_evidence == source


def test_contract_rejects_invented_rule() -> None:
    try:
        parse_contract(_contract("없는 규칙"), "결과를 먼저 쓰세요.")
    except SynthesisError as exc:
        assert "not in the artifact" in str(exc)
    else:
        raise AssertionError("unsupported evidence was accepted")


def test_quick_profile_has_small_exact_split() -> None:
    config = OptimizeConfig(
        categories=QUICK_CATEGORIES,
        split=(6, 2, 2),
        holdout_repetitions=1,
    )

    assert sum(category.raw for category in config.categories) == 18
    assert sum(category.retained for category in config.categories) == 10
    assert config.split_sizes == (6, 2, 2)
    assert config.holdout_repetitions == 1


def test_split_is_exact_and_families_do_not_cross_boundaries() -> None:
    cases = [
        EvaluationCase(
            id=f"case-{index}",
            category="normal",
            family=f"family-{index // 2}",
            inquiry=f"질문 {index}",
            context={"index": index},
            obligations=("O1",),
            must_include=("결과",),
            must_not=(),
            reader_questions=(ReaderQuestion("what", "무엇?", "결과"),),
        )
        for index in range(50)
    ]

    first = split_cases(cases, (30, 10, 10), seed=4)
    second = split_cases(cases, (30, 10, 10), seed=4)

    assert [case.id for case in first.train] == [case.id for case in second.train]
    assert (len(first.train), len(first.validation), len(first.holdout)) == (30, 10, 10)
    sets = [
        {case.family for case in first.train},
        {case.family for case in first.validation},
        {case.family for case in first.holdout},
    ]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]


def test_case_refill_preserves_accepted_cases_and_uses_unique_ids() -> None:
    class Generator:
        name = "generator"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(text="not json")
            if self.calls == 2:
                return SimpleNamespace(text=json.dumps({"cases": []}))
            request = json.loads(user)
            cases = [
                {
                    "family": f"family-{request['case_offset'] + index}",
                    "inquiry": f"question-{request['case_offset'] + index}",
                    "context": {"value": request["case_offset"] + index},
                    "obligations": ["O1"],
                    "must_include": ["answer"],
                    "must_not": [],
                    "reader_questions": [
                        {
                            "id": "what",
                            "question": "What?",
                            "criterion": "answer",
                            "applicable": True,
                        }
                    ],
                    "constraints": [],
                }
                for index in range(request["count"])
            ]
            return SimpleNamespace(text=json.dumps({"cases": cases}))

    class Validator:
        name = "validator"

        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def complete(self, system: str, user: str) -> SimpleNamespace:
            ids = [case["id"] for case in json.loads(user)["cases"]]
            self.batches.append(ids)
            return SimpleNamespace(text=json.dumps({"accepted": ids[:1], "rejected": []}))

    contract = BehaviorContract(
        "English",
        "Answer",
        (Rule("O1", "Answer", "Answer", "critical"),),
        (),
    )
    category = CaseCategory(
        name="normal",
        purpose="purpose",
        raw=3,
        retained=2,
        persona="person",
    )
    generator = Generator()
    validator = Validator()

    cases = generate_cases(contract, "Answer", generator, validator, (category,), seed=0)

    assert [case.id for case in cases] == ["normal-001", "normal-004"]
    assert generator.calls == 4
    assert validator.batches == [
        ["normal-001", "normal-002", "normal-003"],
        ["normal-004"],
    ]


def test_case_refill_continues_after_three_low_yield_attempts() -> None:
    class Generator:
        name = "generator"

        def complete(self, system: str, user: str) -> SimpleNamespace:
            request = json.loads(user)
            offset = request["case_offset"]
            case = {
                "family": f"family-{offset}",
                "inquiry": f"question-{offset}",
                "context": {"value": offset},
                "obligations": ["O1"],
                "must_include": ["answer"],
                "must_not": [],
                "reader_questions": [
                    {
                        "id": "what",
                        "question": "What?",
                        "criterion": "answer",
                        "applicable": True,
                    }
                ],
                "constraints": [],
            }
            return SimpleNamespace(text=json.dumps({"cases": [case]}))

    class Validator:
        name = "validator"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> SimpleNamespace:
            self.calls += 1
            ids = [case["id"] for case in json.loads(user)["cases"]]
            accepted = ids if self.calls == 4 else []
            return SimpleNamespace(text=json.dumps({"accepted": accepted, "rejected": []}))

    contract = BehaviorContract(
        "English",
        "Answer",
        (Rule("O1", "Answer", "Answer", "critical"),),
        (),
    )
    category = CaseCategory("normal", "purpose", 1, 1, "person")
    validator = Validator()

    cases = generate_cases(
        contract,
        "Answer",
        Generator(),
        validator,
        (category,),
        seed=0,
    )

    assert [case.id for case in cases] == ["normal-004"]
    assert validator.calls == 4


def test_short_category_is_backfilled_instead_of_failing_the_run() -> None:
    class Generator:
        name = "generator"

        def complete(self, system: str, user: str) -> SimpleNamespace:
            request = json.loads(user)
            offset = request["case_offset"]
            cases = [
                {
                    "family": f"{request['category']}-family-{offset + index}",
                    "inquiry": f"{request['category']}-question-{offset + index}",
                    "context": {"value": offset + index},
                    "obligations": ["O1"],
                    "must_include": ["answer"],
                    "must_not": [],
                    "reader_questions": [
                        {
                            "id": "what",
                            "question": "What?",
                            "criterion": "answer",
                            "applicable": True,
                        }
                    ],
                    "constraints": [],
                }
                for index in range(request["count"])
            ]
            return SimpleNamespace(text=json.dumps({"cases": cases}))

    class Validator:
        name = "validator"

        def complete(self, system: str, user: str) -> SimpleNamespace:
            ids = [case["id"] for case in json.loads(user)["cases"]]
            accepted = [case_id for case_id in ids if case_id.startswith("normal-")]
            return SimpleNamespace(text=json.dumps({"accepted": accepted, "rejected": []}))

    contract = BehaviorContract(
        "English",
        "Answer",
        (Rule("O1", "Answer", "Answer", "critical"),),
        (),
    )
    categories = (
        CaseCategory("normal", "purpose", 4, 2, "person"),
        CaseCategory("ambiguous", "purpose", 2, 2, "person"),
    )
    progress: list[tuple[int, int, str]] = []

    cases = generate_cases(
        contract,
        "Answer",
        Generator(),
        Validator(),
        categories,
        seed=0,
        progress=lambda current, total, name: progress.append((current, total, name)),
    )

    assert len(cases) == 4
    assert {case.category for case in cases} == {"normal"}
    assert len({case.id for case in cases}) == 4
    assert progress[-1] == (2, 2, "ambiguous short 0/2")


def test_missing_cases_are_topped_up_to_the_exact_dataset_size() -> None:
    class Generator:
        name = "generator"

        def complete(self, system: str, user: str) -> SimpleNamespace:
            request = json.loads(user)
            offset = request["case_offset"]
            cases = [
                {
                    "family": f"{request['category']}-family-{offset + index}",
                    "inquiry": f"{request['category']}-question-{offset + index}",
                    "context": {"value": offset + index},
                    "obligations": ["O1"],
                    "must_include": ["answer"],
                    "must_not": [],
                    "reader_questions": [
                        {
                            "id": "what",
                            "question": "What?",
                            "criterion": "answer",
                            "applicable": True,
                        }
                    ],
                    "constraints": [],
                }
                for index in range(request["count"])
            ]
            return SimpleNamespace(text=json.dumps({"cases": cases}))

    class Validator:
        name = "validator"

        def complete(self, system: str, user: str) -> SimpleNamespace:
            ids = [case["id"] for case in json.loads(user)["cases"]]
            accepted = [case_id for case_id in ids if case_id.startswith("normal-")]
            return SimpleNamespace(text=json.dumps({"accepted": accepted, "rejected": []}))

    contract = BehaviorContract(
        "English",
        "Answer",
        (Rule("O1", "Answer", "Answer", "critical"),),
        (),
    )
    categories = (
        CaseCategory("normal", "purpose", 2, 2, "person"),
        CaseCategory("irrelevant", "purpose", 1, 1, "person"),
    )

    cases = generate_cases(contract, "Answer", Generator(), Validator(), categories, seed=0)

    assert len(cases) == 3
    assert len({case.id for case in cases}) == 3
    assert {case.category for case in cases} == {"normal"}


def test_non_activation_category_may_omit_obligations() -> None:
    class Generator:
        name = "generator"

        def complete(self, system: str, user: str) -> SimpleNamespace:
            request = json.loads(user)
            offset = request["case_offset"]
            cases = [
                {
                    "family": f"family-{offset + index}",
                    "inquiry": f"question-{offset + index}",
                    "context": {"value": offset + index},
                    "obligations": [],
                    "must_include": [],
                    "must_not": ["activates anyway"],
                    "reader_questions": [
                        {
                            "id": "what",
                            "question": "What?",
                            "criterion": "answer",
                            "applicable": True,
                        }
                    ],
                    "constraints": [],
                }
                for index in range(request["count"])
            ]
            return SimpleNamespace(text=json.dumps({"cases": cases}))

    class Validator:
        name = "validator"

        def complete(self, system: str, user: str) -> SimpleNamespace:
            ids = [case["id"] for case in json.loads(user)["cases"]]
            return SimpleNamespace(text=json.dumps({"accepted": ids, "rejected": []}))

    contract = BehaviorContract(
        "English",
        "Answer",
        (Rule("O1", "Answer", "Answer", "critical"),),
        (),
    )
    category = CaseCategory("irrelevant", "purpose", 2, 2, "person", requires_obligations=False)

    cases = generate_cases(contract, "Answer", Generator(), Validator(), (category,), seed=0)

    assert len(cases) == 2
    assert all(case.obligations == () for case in cases)


def test_short_dataset_still_splits_into_three_buckets() -> None:
    cases = [
        EvaluationCase(
            id=f"case-{index}",
            category="normal",
            family=f"family-{index}",
            inquiry=f"question {index}",
            context={"index": index},
            obligations=("O1",),
            must_include=("answer",),
            must_not=(),
            reader_questions=(ReaderQuestion("what", "What?", "answer"),),
        )
        for index in range(7)
    ]
    sizes = OptimizeConfig(categories=QUICK_CATEGORIES, split=(6, 2, 2)).sizes_for(len(cases))

    dataset = split_cases(cases, sizes, seed=0)

    assert sizes == (5, 1, 1)
    assert (len(dataset.train), len(dataset.validation), len(dataset.holdout)) == sizes
