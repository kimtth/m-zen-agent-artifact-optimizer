"""Shared values, artifact parsing, configuration, tokens, and JSON helpers."""

from __future__ import annotations

import codecs
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
import yaml

_ENCODING = tiktoken.get_encoding("o200k_base")
_SUPPORTED_NAMES = {"agents.md", "copilot-instructions.md", "skill.md"}
_SUPPORTED_SUFFIXES = (".instructions.md", ".prompt.md", ".agent.md")
_FRONTMATTER = re.compile(r"\A---(?:\r\n|\n).*?(?:\r\n|\n)---(?:\r\n|\n|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Rule:
    id: str
    statement: str
    source_evidence: str
    severity: str = "critical"


@dataclass(frozen=True)
class BehaviorContract:
    language: str
    purpose: str
    obligations: tuple[Rule, ...]
    prohibitions: tuple[Rule, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    value: Any
    severity: str = "critical"


@dataclass(frozen=True)
class ReaderQuestion:
    id: str
    question: str
    criterion: str
    applicable: bool = True


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    category: str
    family: str
    inquiry: str
    context: dict[str, Any]
    obligations: tuple[str, ...]
    must_include: tuple[str, ...]
    must_not: tuple[str, ...]
    reader_questions: tuple[ReaderQuestion, ...]
    constraints: tuple[Constraint, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Dataset:
    train: tuple[EvaluationCase, ...]
    validation: tuple[EvaluationCase, ...]
    holdout: tuple[EvaluationCase, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "train": [case.to_dict() for case in self.train],
            "validation": [case.to_dict() for case in self.validation],
            "holdout": [case.to_dict() for case in self.holdout],
        }


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RunRecord:
    case_id: str
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    events: tuple[dict[str, Any], ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Check:
    rule: str
    passed: bool
    severity: str
    evidence: str | None
    feedback: str


@dataclass(frozen=True)
class BehaviorResult:
    passed: bool
    critical_failure: bool
    checks: tuple[Check, ...]


@dataclass(frozen=True)
class UnderstandingAnswer:
    question: str
    correct: bool
    evidence: str | None


@dataclass(frozen=True)
class UnderstandingResult:
    passed: bool
    accuracy: float
    tokens: int
    answers: tuple[UnderstandingAnswer, ...]


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    behavior: BehaviorResult
    understanding: UnderstandingResult
    output_tokens: int
    feedback: str


@dataclass(frozen=True)
class Aggregate:
    artifact_tokens: int
    behavior_passes: int
    understanding_passes: int
    critical_failures: int
    median_output_tokens: float
    median_understanding_tokens: float

    @property
    def communication_tokens(self) -> float:
        return self.artifact_tokens + self.median_output_tokens


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    baseline: Aggregate
    candidate: Aggregate
    communication_reduction: float


@dataclass
class OptimizationResult:
    decision: str
    artifact_path: str
    candidate_body: str | None
    contract: BehaviorContract | None
    dataset: Dataset | None
    gate: GateDecision | None
    calls: int
    run_directory: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseCategory:
    name: str
    purpose: str
    raw: int
    retained: int
    persona: str
    requires_obligations: bool = True


DEFAULT_CATEGORIES = (
    CaseCategory("normal", "ordinary representative requests", 30, 20, "practical end user"),
    CaseCategory("ambiguous", "underspecified requests", 15, 10, "uncertain end user"),
    CaseCategory("conflicting", "requests with competing constraints", 15, 10, "demanding end user"),
    CaseCategory("verbose", "requests that invite unnecessary detail", 10, 5, "detail-seeking end user"),
    CaseCategory(
        "irrelevant",
        "requests where the artifact should not activate",
        10,
        5,
        "out-of-scope end user",
        requires_obligations=False,
    ),
)

QUICK_CATEGORIES = (
    CaseCategory("normal", "ordinary representative requests", 6, 4, "practical end user"),
    CaseCategory("ambiguous", "underspecified requests", 4, 2, "uncertain end user"),
    CaseCategory("conflicting", "requests with competing constraints", 4, 2, "demanding end user"),
    CaseCategory("verbose", "requests that invite unnecessary detail", 2, 1, "detail-seeking end user"),
    CaseCategory(
        "irrelevant",
        "requests where the artifact should not activate",
        2,
        1,
        "out-of-scope end user",
        requires_obligations=False,
    ),
)

AGGRESSIVE_DEFAULT_MAX_BODY_LINES = 100


@dataclass(frozen=True)
class AggressiveLimit:
    """A fixed line cap or a cap relative to the original mutable body."""

    lines: int | None = None
    percent: float | None = None

    def resolve(self, body: str) -> int:
        if self.lines is not None:
            return self.lines
        if self.percent is None:
            raise ValueError("aggressive limit requires lines or a percentage")
        return max(1, math.ceil(len(body.splitlines()) * self.percent / 100))


def parse_aggressive_limit(value: str) -> AggressiveLimit:
    """Parse a positive line count or a percentage no greater than 100%."""
    text = value.strip()
    if text.endswith("%"):
        try:
            percent = float(text[:-1])
        except ValueError as exc:
            raise ValueError("must be a positive line count or percentage, e.g. 80 or 50%") from exc
        if not 0 < percent <= 100:
            raise ValueError("percentage must be greater than 0 and at most 100")
        return AggressiveLimit(percent=percent)
    try:
        lines = int(text)
    except ValueError as exc:
        raise ValueError("must be a positive line count or percentage, e.g. 80 or 50%") from exc
    if lines < 1:
        raise ValueError("line count must be at least 1")
    return AggressiveLimit(lines=lines)


@dataclass(frozen=True)
class OptimizeConfig:
    target_model: str = "gpt-5-mini"
    strong_model: str = "gpt-5"
    generator_model: str = "gpt-5-mini"
    seed: int = 0
    max_metric_calls: int = 120
    total_call_budget: int = 600
    holdout_repetitions: int = 3
    communication_reduction: float = 0.03
    categories: tuple[CaseCategory, ...] = DEFAULT_CATEGORIES
    split: tuple[int, int, int] | None = None
    aggressive_limit: AggressiveLimit | None = None

    @property
    def aggressive(self) -> bool:
        return self.aggressive_limit is not None

    def max_body_lines(self, body: str) -> int | None:
        return self.aggressive_limit.resolve(body) if self.aggressive_limit else None

    @property
    def split_sizes(self) -> tuple[int, int, int]:
        sizes = self.split or (30, 10, 10)
        if sum(category.retained for category in self.categories) != sum(sizes):
            raise ValueError("evaluation profile case counts do not match its split")
        return sizes

    def sizes_for(self, count: int) -> tuple[int, int, int]:
        """Scale the profile split to the cases that were actually generated."""
        if count < 3:
            raise ValueError("at least three evaluation cases are required")
        planned = self.split or (30, 10, 10)
        if sum(planned) == count:
            return planned
        total = sum(planned)
        validation = max(1, round(count * planned[1] / total))
        holdout = max(1, round(count * planned[2] / total))
        while validation + holdout > count - 1:
            if holdout > validation:
                holdout -= 1
            else:
                validation -= 1
        return (count - validation - holdout, validation, holdout)


class ArtifactError(ValueError):
    """Raised when a path is not a supported, well-formed artifact."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    source: str
    immutable_prefix: str
    body: str
    bom: bool
    newline: str

    @property
    def tokens(self) -> int:
        return count_tokens(self.source)

    @property
    def body_tokens(self) -> int:
        return count_tokens(self.body)

    def render(self, body: str) -> str:
        normalized = body.replace("\r\n", "\n").replace("\r", "\n")
        return self.immutable_prefix + normalized.replace("\n", self.newline)

    def candidate_bytes(self, body: str) -> bytes:
        data = self.render(body).encode("utf-8")
        return codecs.BOM_UTF8 + data if self.bom else data


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def is_supported(path: Path) -> bool:
    name = path.name.lower()
    return name in _SUPPORTED_NAMES or any(name.endswith(suffix) for suffix in _SUPPORTED_SUFFIXES)


def load_artifact(path: Path) -> Artifact:
    if not path.is_file():
        raise ArtifactError(f"artifact is not a file: {path}")
    if not is_supported(path):
        raise ArtifactError(f"unsupported customization artifact: {path.name}")
    raw = path.read_bytes()
    bom = raw.startswith(codecs.BOM_UTF8)
    raw = raw[len(codecs.BOM_UTF8) :] if bom else raw
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError("artifact must be UTF-8") from exc
    prefix, body = _split_frontmatter(source)
    if not body.strip():
        raise ArtifactError("artifact has no mutable instruction body")
    return Artifact(path, source, prefix, body, bom, "\r\n" if "\r\n" in source else "\n")


def _split_frontmatter(source: str) -> tuple[str, str]:
    if not source.startswith("---"):
        return "", source
    match = _FRONTMATTER.match(source)
    if match is None:
        raise ArtifactError("frontmatter starts with --- but has no closing ---")
    prefix = match.group(0)
    try:
        value = yaml.safe_load("\n".join(prefix.splitlines()[1:-1]))
    except yaml.YAMLError as exc:
        raise ArtifactError(f"invalid YAML frontmatter: {exc}") from exc
    if value is not None and not isinstance(value, dict):
        raise ArtifactError("YAML frontmatter must be a mapping")
    return prefix, source[match.end() :]


def optimized_path(path: Path) -> Path:
    return path.with_name(path.stem + ".optimized" + path.suffix)


def report_path(path: Path) -> Path:
    return path.with_name(path.stem + ".optimize.report.md")


class JsonResponseError(ValueError):
    pass


def parse_json(text: str) -> Any:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(value):
            if char not in "[{":
                continue
            try:
                parsed, end = decoder.raw_decode(value[start:])
            except json.JSONDecodeError:
                continue
            if not value[start + end :].strip():
                return parsed
        raise JsonResponseError("model response does not contain one valid JSON value") from None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
