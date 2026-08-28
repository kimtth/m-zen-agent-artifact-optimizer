"""Model adapters with one shared application-call budget."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Protocol

import dspy
from copilot import CopilotClient

from ..domain.core import ModelResponse, count_tokens


class ModelError(RuntimeError):
    """Raised when a model call cannot produce a usable answer."""


class BudgetExceeded(ModelError):
    """Raised before a call that would exceed the application budget."""


class TextModel(Protocol):
    name: str

    def complete(self, system: str, user: str) -> ModelResponse: ...


class CallBudget:
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("total call budget must be at least 1")
        self.limit = limit
        self.calls = 0
        self._lock = threading.Lock()

    def claim(self) -> None:
        with self._lock:
            if self.calls >= self.limit:
                raise BudgetExceeded(f"application model-call budget reached ({self.limit})")
            self.calls += 1


_CACHE_VERSION = "model-call-v1"


class ResponseCache:
    """Replay identical model calls across runs while preserving retry order.

    Retry loops re-send the same prompt expecting a different sample, so each
    repeat of a prompt gets its own slot instead of replaying the first answer.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._ordinals: dict[str, int] = {}
        self._lock = threading.Lock()

    def slot(self, model: str, system: str, user: str) -> Path:
        digest = hashlib.sha256(
            json.dumps([_CACHE_VERSION, model, system, user], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        with self._lock:
            index = self._ordinals.get(digest, 0)
            self._ordinals[digest] = index + 1
        return self.root / f"{digest}-{index:02}.json"

    def read(self, slot: Path) -> ModelResponse | None:
        if not slot.is_file():
            return None
        try:
            value = json.loads(slot.read_text(encoding="utf-8"))
            return ModelResponse(
                str(value["text"]),
                int(value["input_tokens"]),
                int(value["output_tokens"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def write(self, slot: Path, response: ModelResponse) -> None:
        slot.write_text(
            json.dumps(
                {
                    "text": response.text,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


class CopilotModel:
    """One isolated, tool-free Copilot session per model call."""

    def __init__(
        self,
        name: str,
        budget: CallBudget,
        *,
        working_directory: Path,
        timeout: float = 300.0,
        cache: ResponseCache | None = None,
    ):
        self.name = name
        self.budget = budget
        self.working_directory = working_directory
        self.timeout = timeout
        self.cache = cache

    def complete(self, system: str, user: str) -> ModelResponse:
        slot = self.cache.slot(self.name, system, user) if self.cache else None
        if self.cache is not None and slot is not None:
            replayed = self.cache.read(slot)
            if replayed is not None:
                return replayed
        self.budget.claim()
        try:
            response = asyncio.run(self._complete(system, user))
        except BudgetExceeded:
            raise
        except Exception as exc:
            raise ModelError(f"{self.name}: {exc}") from exc
        if self.cache is not None and slot is not None:
            self.cache.write(slot, response)
        return response

    async def _complete(self, system: str, user: str) -> ModelResponse:
        client = CopilotClient(working_directory=str(self.working_directory), log_level="error")
        session = None
        try:
            await client.start()
            session = await client.create_session(
                model=self.name,
                system_message={"mode": "replace", "content": system},
                available_tools=[],
                skip_custom_instructions=True,
                enable_config_discovery=False,
                enable_skills=False,
                enable_session_store=False,
            )
            event = await session.send_and_wait(user, timeout=self.timeout)
            if event is None or not hasattr(event.data, "content"):
                raise ModelError("model returned no assistant message")
            text = str(event.data.content).strip()
            if not text:
                raise ModelError("model returned an empty answer")
            return ModelResponse(text, count_tokens(system + user), count_tokens(text))
        finally:
            if session is not None:
                with suppress(Exception):
                    await session.disconnect()
            with suppress(Exception):
                await client.stop()


class FunctionModel:
    """Deterministic model used by tests and offline checks."""

    def __init__(self, function: Callable[[str, str], str], name: str = "scripted"):
        self.function = function
        self.name = name

    def complete(self, system: str, user: str) -> ModelResponse:
        text = self.function(system, user)
        return ModelResponse(text, count_tokens(system + user), count_tokens(text))


class DSPyModel(dspy.BaseLM):
    """Expose a TextModel through DSPy's typed LM interface."""

    forward_contract = "typed_lm"

    def __init__(self, backend: TextModel):
        super().__init__(model=backend.name, cache=False, num_retries=0)
        self.backend = backend

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        system_parts: list[str] = []
        conversation: list[str] = []
        for message in request.messages:
            text = message.text or ""
            if message.role == "system":
                system_parts.append(text)
            else:
                conversation.append(f"{message.role}: {text}")
        try:
            response = self.backend.complete(
                "\n\n".join(system_parts), "\n\n".join(conversation)
            )
        except BudgetExceeded:
            raise
        except Exception as exc:
            raise dspy.LMError(str(exc), model=self.model) from exc
        return dspy.LMResponse.from_text(
            response.text,
            model=self.model,
            usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

    def dump_state(self) -> dict[str, object]:
        return {"model": self.model, "cache": False, "num_retries": 0}
