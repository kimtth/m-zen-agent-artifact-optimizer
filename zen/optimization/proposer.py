"""Local mutation proposals for GEPA, never an independent search or selector."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import unified_diff
from itertools import pairwise
from typing import Any

from ..domain.core import BehaviorContract
from ..runtime.harness import CandidatePolicy
from ..runtime.lm import TextModel

_MARKER = re.compile(r"<!-- (/?)zen:(task|communication) -->")
_MARKER_START = re.compile(r"<!--\s*/?\s*zen\b", re.IGNORECASE)


@dataclass(frozen=True)
class MutationScope:
    """Character-exact mutation region, constructed with :func:`validate_focus`.

    ``validate(candidate)`` raises ValueError on changed outside text or invalid
    marker boundaries. It does not evaluate behavior, compression, or performance.
    ``extract`` and ``replace`` operate on the region between (not including) tags.
    With focus='all', the entire body is mutable and tags have no special meaning.
    """

    original: str
    focus: str
    start: int
    end: int

    @property
    def prefix(self) -> str:
        return self.original[: self.start]

    @property
    def suffix(self) -> str:
        return self.original[self.end :]

    def validate(self, candidate: str) -> None:
        if self.focus == "all":
            return
        other = validate_focus(candidate, self.focus)
        if other.prefix != self.prefix or other.suffix != self.suffix:
            raise ValueError("candidate changed text outside the focus region")

    def extract(self, candidate: str) -> str:
        self.validate(candidate)
        end = len(candidate) - len(self.suffix)
        return candidate[len(self.prefix) : end]

    def replace(self, current: str, replacement: str) -> str:
        self.validate(current)
        candidate = self.prefix + replacement + self.suffix
        self.validate(candidate)
        return candidate


def validate_focus(original: str, focus: str = "all") -> MutationScope:
    """Validate focus early and return a reusable final-candidate boundary guard.

    Scoped mode requires exactly one selected pair. The other pair is optional,
    but every reserved zen comment must be exact, paired, unique, and non-nested.
    Offsets are Python string offsets; no whitespace/line-ending normalization is
    performed. Whole-body mode deliberately does not interpret marker syntax.
    """
    if focus not in ("all", "task", "communication"):
        raise ValueError("focus must be 'all', 'task', or 'communication'")
    if focus == "all":
        return MutationScope(original, focus, 0, len(original))
    regions: dict[str, tuple[int, int]] = {}
    active: tuple[str, int] | None = None
    for start in _MARKER_START.finditer(original):
        marker = _MARKER.match(original, start.start())
        if marker is None:
            raise ValueError("malformed zen marker boundary")
        closing, name = marker.groups()
        if not closing:
            if active is not None:
                raise ValueError("nested zen marker boundaries")
            if name in regions:
                raise ValueError("duplicate zen marker boundaries")
            active = (name, marker.end())
        else:
            if active is None or active[0] != name:
                raise ValueError("unmatched zen marker boundary")
            regions[name] = (active[1], marker.start())
            active = None
    if active is not None:
        raise ValueError("unclosed zen marker boundary")
    if focus not in regions:
        raise ValueError(f"absent {focus} marker boundaries")
    return MutationScope(original, focus, *regions[focus])


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    result = []
    offset = 0
    for line in text.splitlines(keepends=True):
        result.append((offset, offset + len(line), line))
        offset += len(line)
    return result


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Conservatively freeze Markdown backtick/tilde fences, including open EOFs.

    Recognizes top-level fences with up to three leading spaces, not a complete
    Markdown parser (e.g. nested list/blockquote fences are not interpreted).
    """
    spans = []
    active: tuple[int, str] | None = None
    for start, end, line in _line_spans(text):
        value = line.rstrip("\r\n")
        if active is None:
            opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", value)
            if opening and not (
                opening[1][0] == "`" and "`" in opening[2]
            ):
                active = (start, opening[1])
        else:
            closing = re.fullmatch(r" {0,3}([`~]+)[ \t]*", value)
            fence = active[1]
            if closing and set(closing[1]) == {fence[0]} and len(closing[1]) >= len(fence):
                spans.append((active[0], end))
                active = None
    if active is not None:
        spans.append((active[0], len(text)))
    return spans


def _fenced_blocks(text: str) -> list[str]:
    return [text[start:end] for start, end in _fenced_spans(text)]


class CompressionProposer:
    """Alternate removal then rewrite once per nonempty GEPA callback.

    All components in a callback share its strategy. Invalid mutations return
    the current component; only an oversized rewrite gets one compression retry.
    ``history`` records each local attempt (including retries), not GEPA fitness
    or selection. Its before/after SHA-256 hashes describe current/returned text;
    proposed_sha256 and unified_diff describe the attempted text, when available.
    Removal indices are 1-based whole-body line numbers including blank lines.
    """

    def __init__(
        self,
        model: TextModel,
        contract: BehaviorContract,
        original: str,
        max_lines: int | None = None,
        *,
        focus: str = "all",
    ):
        self.model = model
        self.contract = contract
        self.scope = validate_focus(original, focus)
        self.focus = focus
        self.policy = CandidatePolicy(original, max_lines)
        self.max_lines = max_lines
        self.history: list[dict[str, Any]] = []
        self._calls = 0
        self._protected_quotes = tuple(
            dict.fromkeys(
                rule.source_evidence
                for rule in (*contract.obligations, *contract.prohibitions)
                if rule.severity == "critical" and rule.source_evidence
            )
        )

    def _numbered_lines(self, current: str) -> list[dict[str, Any]]:
        scope = validate_focus(current, self.focus)
        protected_spans = _fenced_spans(current)
        for quote in self._protected_quotes:
            offset = current.find(quote)
            while offset != -1:
                protected_spans.append((offset, offset + len(quote)))
                offset = current.find(quote, offset + 1)
        return [
            {
                "index": index,
                "text": line,
                "protected": (
                    start < scope.start
                    or end > scope.end
                    or any(start < stop and end > begin for begin, stop in protected_spans)
                ),
            }
            for index, (start, end, line) in enumerate(_line_spans(current), 1)
        ]

    def _validate_mutation(self, proposed: str) -> None:
        self.scope.validate(proposed)
        if self.focus != "all" and _fenced_blocks(proposed) != _fenced_blocks(self.scope.original):
            raise ValueError("scoped mutation changed protected fenced-code blocks")
        self.policy.validate(proposed)

    def _record(
        self,
        component: str,
        strategy: str,
        current: str,
        proposed: str | None,
        reason: str,
        *,
        attempt: int = 1,
        removed_lines: list[int] | None = None,
    ) -> str:
        accepted = not reason
        result = proposed if accepted and proposed is not None else current
        self.history.append(
            {
                "component": component,
                "strategy": strategy,
                "focus": self.focus,
                "attempt": attempt,
                "before_sha256": _sha256(current),
                "after_sha256": _sha256(result),
                "proposed_sha256": _sha256(proposed) if proposed is not None else None,
                "removed_lines": removed_lines or [],
                "unified_diff": "".join(
                    unified_diff(
                        current.splitlines(keepends=True),
                        proposed.splitlines(keepends=True),
                        fromfile="current",
                        tofile="proposed",
                    )
                ) if proposed is not None else "",
                "accepted_by_policy": accepted,
                "reason": reason or "local policy passed; GEPA has not judged or selected this proposal",
            }
        )
        return result

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals: dict[str, str] = {}
        if not components_to_update:
            return proposals
        strategy = "removal" if self._calls % 2 == 0 else "rewrite"
        self._calls += 1
        for component in components_to_update:
            current = candidate[component]
            try:
                mutable = self.scope.extract(current)
            except ValueError as exc:
                proposals[component] = self._record(component, strategy, current, None, str(exc))
                continue
            payload = {
                "language": self.contract.language,
                "contract": self.contract.to_dict(),
                "current_instruction": current,
                "mutable_instruction": mutable,
                "focus": self.focus,
                "line_limit": self.max_lines,
                "protected_source_quotes": self._protected_quotes,
                "evaluation_examples": list(reflective_dataset.get(component, ())),
            }
            if strategy == "removal":
                numbered = self._numbered_lines(current)
                payload["numbered_lines"] = numbered
                system = """Propose a removal-only mutation for GEPA using the evaluation feedback.
Return only a nonempty JSON list of strictly increasing, unique integer line indices
to delete, e.g. [2, 5]. Indices are 1-based in numbered_lines for the WHOLE current
body, including blank lines. Delete exact whole lines only; do not rewrite text.
Never select a line marked protected: outside-focus text, fenced-code blocks, and
lines overlapping exact critical contract source evidence are protected.
The protected_source_quotes are critical evidence, not disposable redundancy.
Keep the original language and critical behavior; prefer deleting redundancy.
If line_limit is not null, the remaining WHOLE body must fit that line limit.
GEPA alone evaluates and selects this proposal."""
                raw = self.model.complete(system, _json(payload)).text
                proposed = None
                indices: list[int] = []
                reason = ""
                try:
                    selected = json.loads(raw)
                    if (
                        not isinstance(selected, list)
                        or not selected
                        or any(type(index) is not int for index in selected)
                        or any(a >= b for a, b in pairwise(selected))
                    ):
                        raise ValueError("deletions must be a nonempty strictly increasing list of unique ints")
                    if any(index < 1 or index > len(numbered) for index in selected):
                        raise ValueError("deletion index out of range")
                    indices = selected
                    deleted = set(indices)
                    proposed = "".join(line["text"] for line in numbered if line["index"] not in deleted)
                    if any(numbered[index - 1]["protected"] for index in indices):
                        raise ValueError("deletion targets a protected line")
                    if proposed == current:
                        raise ValueError("removal must actually remove text")
                    self._validate_mutation(proposed)
                except ValueError as exc:
                    reason = str(exc)
                proposals[component] = self._record(
                    component, strategy, current, proposed, reason, removed_lines=indices
                )
                continue

            line_limit = (
                f"The complete replacement body must contain {self.max_lines} lines or fewer. "
                if self.max_lines is not None
                else ""
            )
            system = """Rewrite one instruction body using the evaluation feedback.
Return only replacement instruction text: no JSON, wrapper fence, preface, or explanation.
Keep the original language and all critical behavior. Do not add capabilities.
The protected_source_quotes are critical contract evidence: preserve their behavior.
Prefer removing or combining redundant instructions. The replacement must not exceed
the original token length. """ + line_limit + """Feedback diagnoses problems; decide the
rewrite yourself."""
            if self.focus != "all":
                system += """ Return ONLY the replacement for mutable_instruction, including
its intended whitespace and line endings. Do not return boundary markers or the full
body. All text outside this exact region is frozen. Preserve fenced-code blocks
verbatim, including their order, delimiters, contents, and line endings."""
            for attempt in (1, 2):
                raw = self.model.complete(system, _json(payload)).text
                replacement = _unwrap(raw) if self.focus == "all" else raw
                proposed = self.scope.prefix + replacement + self.scope.suffix
                reason = ""
                try:
                    self._validate_mutation(proposed)
                    if proposed == current:
                        raise ValueError("rewrite made no change")
                except ValueError as exc:
                    reason = str(exc)
                proposals[component] = self._record(
                    component, strategy, current, proposed, reason, attempt=attempt
                )
                if (
                    not reason
                    or attempt == 2
                    or self.max_lines is None
                    or len(proposed.splitlines()) <= self.max_lines
                ):
                    break
                system += (
                    "\nRetry: Return only a replacement with "
                    f"{self.max_lines} lines or fewer in the complete body. Preserve all "
                    "critical behavior from this draft while combining or removing redundancy."
                )
                payload = {**payload, "draft": proposed, "mutable_draft": replacement,
                           "line_limit": self.max_lines}
        return proposals


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unwrap(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return value
