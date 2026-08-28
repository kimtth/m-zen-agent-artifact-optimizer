"""Concise user-facing decision report."""

from __future__ import annotations

from pathlib import Path

from ..domain.core import OptimizationResult


def render_report(result: OptimizationResult) -> str:
    lines = [f"# Zen optimization: {Path(result.artifact_path).name}", ""]
    if result.gate is None:
        lines.extend(
            [
                "Decision: REJECT",
                "",
                f"Why: {result.message or 'The optimization did not complete.'}",
                "",
                "Why it matters: The source artifact was retained unchanged.",
            ]
        )
        return "\n".join(lines) + "\n"

    gate = result.gate
    baseline = gate.baseline
    candidate = gate.candidate
    lines.append(f"Decision: {'ACCEPT' if gate.accepted else 'REJECT'}")
    lines.append("")
    if gate.accepted:
        lines.extend(
            [
                "Why: The candidate preserved all tested behavior and remained equally easy to understand.",
                "",
                (
                    "Why it matters: Combined artifact and median output tokens fell by "
                    f"{gate.communication_reduction:.1%}."
                ),
            ]
        )
    else:
        lines.extend(
            [
                f"Why: {'; '.join(gate.reasons)}.",
                "",
                "Why it matters: A smaller artifact is not accepted when quality or the required reduction regresses.",
            ]
        )
    lines.extend(
        [
            "",
            f"Artifact tokens: {baseline.artifact_tokens} → {candidate.artifact_tokens}",
            (
                "Median output tokens: "
                f"{baseline.median_output_tokens:g} → {candidate.median_output_tokens:g}"
            ),
            (
                "Understanding tokens: "
                f"{baseline.median_understanding_tokens:g} → "
                f"{candidate.median_understanding_tokens:g}"
            ),
            f"Behavior passes: {baseline.behavior_passes} → {candidate.behavior_passes}",
            f"Critical failures: {baseline.critical_failures} → {candidate.critical_failures}",
            f"Model calls: {result.calls}",
            "",
            "The source artifact was not modified.",
        ]
    )
    return "\n".join(lines) + "\n"
