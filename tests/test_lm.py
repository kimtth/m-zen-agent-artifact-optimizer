from __future__ import annotations

from pathlib import Path

from zen.domain.core import ModelResponse
from zen.runtime.lm import CallBudget, CopilotModel, ResponseCache


class _Model(CopilotModel):
    """CopilotModel with the network call replaced by a scripted answer."""

    def __init__(self, cache: ResponseCache, answers: list[str]) -> None:
        super().__init__("test", CallBudget(10), working_directory=Path.cwd(), cache=cache)
        self.answers = answers

    async def _complete(self, system: str, user: str) -> ModelResponse:
        return ModelResponse(self.answers.pop(0), 1, 1)


def test_repeated_prompt_gets_its_own_slot_and_replays_without_new_calls(
    tmp_path: Path,
) -> None:
    first = _Model(ResponseCache(tmp_path), ["bad", "good"])

    assert first.complete("s", "u").text == "bad"
    assert first.complete("s", "u").text == "good"
    assert first.budget.calls == 2

    second = _Model(ResponseCache(tmp_path), [])

    assert second.complete("s", "u").text == "bad"
    assert second.complete("s", "u").text == "good"
    assert second.budget.calls == 0
