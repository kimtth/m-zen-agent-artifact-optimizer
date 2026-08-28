"""Response-only DSPy program, candidate checks, and replayable run cache."""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import dspy

from ..domain.core import EvaluationCase, RunRecord, count_tokens, write_json
from .lm import BudgetExceeded, DSPyModel, TextModel

_RUNNER_VERSION = "runner-v1"


@dataclass(frozen=True)
class CandidatePolicy:
    original: str
    max_lines: int | None = None

    def validate(self, candidate: str) -> None:
        if not candidate.strip():
            raise ValueError("candidate body is empty")
        if count_tokens(candidate.strip()) > count_tokens(self.original.strip()):
            raise ValueError("candidate body exceeds the original token count")
        missing = _dominant_scripts(self.original) - _scripts(candidate)
        if missing:
            raise ValueError("candidate changed the artifact's writing system")
        if self.max_lines is not None and len(candidate.splitlines()) > self.max_lines:
            raise ValueError(f"candidate body exceeds the {self.max_lines}-line limit")


class ArtifactProgram(dspy.Module):
    def __init__(self, instructions: str, policy: CandidatePolicy):
        super().__init__()
        self.policy = policy
        self.answer = dspy.Predict("inquiry, context -> answer")
        self.answer.signature = self.answer.signature.with_instructions(instructions)

    def forward(self, inquiry: str, context: str) -> dspy.Prediction:
        instructions = self.answer.signature.instructions
        self.policy.validate(instructions)
        prediction = self.answer(inquiry=inquiry, context=context)
        return dspy.Prediction(answer=prediction.answer, zen_instructions=instructions)


class RunCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> RunRecord | None:
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return RunRecord(
            case_id=value["case_id"],
            answer=value["answer"],
            input_tokens=int(value["input_tokens"]),
            output_tokens=int(value["output_tokens"]),
            latency_ms=int(value["latency_ms"]),
            events=tuple(value.get("events", [])),
            error=value.get("error", ""),
        )

    def put(self, key: str, record: RunRecord) -> None:
        write_json(self.root / f"{key}.json", record.to_dict())


class Runner:
    def __init__(self, model: TextModel, cache: RunCache):
        self.model = model
        self.lm = DSPyModel(model)
        self.adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
        self.cache = cache

    def run(self, instructions: str, case: EvaluationCase, repetition: int = 0) -> RunRecord:
        key = self._key(instructions, case.id, repetition)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        started = time.perf_counter()
        context = json.dumps(case.context, ensure_ascii=False, sort_keys=True)
        try:
            program = ArtifactProgram(instructions, CandidatePolicy(instructions))
            with dspy.context(lm=self.lm, adapter=self.adapter):
                prediction = program(inquiry=case.inquiry, context=context)
            answer = str(prediction.answer).strip()
            error = "" if answer else "model returned an empty answer"
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - execution errors belong in replayable traces.
            answer = ""
            error = str(exc)
        record = RunRecord(
            case_id=case.id,
            answer=answer,
            input_tokens=count_tokens(instructions + case.inquiry + context),
            output_tokens=count_tokens(answer),
            latency_ms=round((time.perf_counter() - started) * 1000),
            events=({"type": "final_message", "content": answer},) if answer else (),
            error=error,
        )
        self.cache.put(key, record)
        return record

    def _key(self, instructions: str, case_id: str, repetition: int) -> str:
        value = json.dumps(
            {
                "candidate": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
                "case": case_id,
                "model": self.model.name,
                "repetition": repetition,
                "runner": _RUNNER_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scripts(text: str) -> set[str]:
    result = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name:
            result.add("HAN" if name.startswith("CJK") else name.split()[0])
    return result


def _dominant_scripts(text: str) -> set[str]:
    counts: dict[str, int] = {}
    total = 0
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if not name:
            continue
        script = "HAN" if name.startswith("CJK") else name.split()[0]
        counts[script] = counts.get(script, 0) + 1
        total += 1
    return {script for script, count in counts.items() if count >= 2 and count / max(total, 1) >= 0.15}
