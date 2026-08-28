from __future__ import annotations

from pathlib import Path

import pytest

from zen.domain.core import EvaluationCase, ReaderQuestion
from zen.runtime.harness import CandidatePolicy, RunCache, Runner
from zen.runtime.lm import FunctionModel


def _case() -> EvaluationCase:
    return EvaluationCase(
        id="korean",
        category="normal",
        family="one",
        inquiry="무엇이 바뀌었나요?",
        context={"변경": "캐시"},
        obligations=("O1",),
        must_include=("캐시",),
        must_not=(),
        reader_questions=(ReaderQuestion("what", "무엇?", "캐시"),),
    )


def test_candidate_policy_rejects_longer_and_translated_text() -> None:
    policy = CandidatePolicy("결과를 먼저 간결하게 설명하세요.")
    with pytest.raises(ValueError, match="token"):
        policy.validate("결과를 먼저 간결하게 설명하고 모든 배경과 예시를 아주 자세하게 설명하세요.")
    with pytest.raises(ValueError, match="writing system"):
        policy.validate("Explain the result first.")


def test_runner_caches_replayable_multilingual_answer(tmp_path: Path) -> None:
    calls = 0

    def answer(system: str, user: str) -> str:
        nonlocal calls
        calls += 1
        return "[[ ## answer ## ]]\n캐시로 바뀌었습니다.\n[[ ## completed ## ]]"

    runner = Runner(FunctionModel(answer), RunCache(tmp_path))
    first = runner.run("결과를 설명하세요.", _case())
    second = runner.run("결과를 설명하세요.", _case())

    assert first == second
    assert first.answer == "캐시로 바뀌었습니다."
    assert first.events[0]["type"] == "final_message"
    assert calls == 1
