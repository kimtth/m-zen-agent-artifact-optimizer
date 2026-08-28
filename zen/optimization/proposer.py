"""Step 5: same-language, compression-aware GEPA instruction proposals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..domain.core import BehaviorContract
from ..runtime.harness import CandidatePolicy
from ..runtime.lm import TextModel


class CompressionProposer:
    def __init__(
        self,
        model: TextModel,
        contract: BehaviorContract,
        original: str,
        max_lines: int | None = None,
    ):
        self.model = model
        self.contract = contract
        self.policy = CandidatePolicy(original, max_lines)
        self.max_lines = max_lines

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposals: dict[str, str] = {}
        for component in components_to_update:
            current = candidate[component]
            line_limit = (
                f"The replacement must contain {self.max_lines} lines or fewer. "
                if self.max_lines is not None
                else ""
            )
            system = """Rewrite one instruction body using the evaluation feedback.
Return only replacement instruction text: no JSON, fence, preface, or explanation.
Keep the original language and all critical behavior. Do not add capabilities.
Prefer removing or combining redundant instructions. The replacement must not exceed
the original token length. """ + line_limit + """Feedback diagnoses problems; decide the
rewrite yourself."""
            user = json.dumps(
                {
                    "language": self.contract.language,
                    "contract": self.contract.to_dict(),
                    "current_instruction": current,
                    "evaluation_examples": list(reflective_dataset.get(component, ())),
                },
                ensure_ascii=False,
                default=str,
            )
            proposed = _unwrap(self.model.complete(system, user).text)
            try:
                self.policy.validate(proposed)
            except ValueError:
                if self.max_lines is None or len(proposed.splitlines()) <= self.max_lines:
                    proposed = current
                else:
                    retry_system = (
                        "Return only a replacement instruction body with "
                        f"{self.max_lines} lines or fewer. Preserve all critical behavior "
                        "from this draft while combining or removing redundancy."
                    )
                    retry_user = json.dumps(
                        {"draft": proposed, "line_limit": self.max_lines},
                        ensure_ascii=False,
                    )
                    proposed = _unwrap(self.model.complete(retry_system, retry_user).text)
                    try:
                        self.policy.validate(proposed)
                    except ValueError:
                        proposed = current
            proposals[component] = proposed
        return proposals


def _unwrap(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return value
