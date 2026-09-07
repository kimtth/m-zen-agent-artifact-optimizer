"""Concise user-facing decision report."""

from __future__ import annotations

import re
from pathlib import PureWindowsPath

from ..domain.core import OptimizationResult
from ..pipeline.gate import evidence_problems


def without_report_paths(text: str, result: OptimizationResult) -> str:
    """Omit filesystem locations from display text, leaving raw evidence unchanged."""
    details = getattr(result, "details", {})
    paths = {result.artifact_path, getattr(result, "run_directory", ""), details.get("output_directory", "")}

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"path", "draft", "optimized", "run_draft", "run_optimized"} and isinstance(item, str):
                    paths.add(item)
                elif isinstance(item, (dict, list)):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(details.get("artifacts", {}))
    collect(details.get("publications", []))
    for path in sorted(filter(None, paths), key=len, reverse=True):
        if "/" in path or "\\" in path:
            for spelling in {path, path.replace("\\", "/"), path.replace("\\", "\\\\")}:
                text = text.replace(spelling, "[path omitted]")
    # Quoted paths can contain spaces; bare paths end at punctuation/whitespace.
    text = re.sub(
        r"([\"'`])(?:file:/+|[A-Za-z]:[\\/]|\\\\|/|\.{1,2}[/\\]|~/)[^\n]*?\1",
        "[path omitted]", text,
    )
    text = re.sub(
        r"(?<![\w:/])(?:file:/+|[A-Za-z]:[\\/]|\\\\|/|\.{1,2}[/\\]|~/)[^\s\"'`<>|,;()]+",
        "[path omitted]", text,
    )
    # File-like relative paths, without treating evidence IDs or prose ratios as paths.
    return re.sub(
        r"(?<![\w/])(?:[\w.@-]+[/\\])+[\w.-]+\.[A-Za-z0-9]{1,12}\b",
        "[path omitted]", text,
    )


def render_report(result: OptimizationResult) -> str:
    lines = [f"# Zen optimization: {PureWindowsPath(result.artifact_path).name}", ""]
    details = getattr(result, "details", {})
    gate = result.gate
    problems = evidence_problems(gate.baseline, gate.candidate) if gate else []
    unavailable = (
        result.decision == "INCONCLUSIVE"
        or bool(problems)
        or bool(gate and gate.inconclusive)
        or bool(details.get("validation_evidence_problems"))
    )
    lines.extend([f"Decision: {'INCONCLUSIVE' if unavailable else result.decision}", ""])
    if result.gate is None:
        reasons = details.get("validation_reasons", [])
        why = result.message or "The optimization did not complete."
        if reasons:
            why = (
                "Validation evidence is incomplete or contains errors."
                if unavailable
                else "The candidate failed the validation quality gate."
            )
        lines.extend(
            [
                f"Why: {why}",
                "",
                "Why it matters: The source artifact was retained unchanged.",
            ]
        )
        if reasons:
            lines.extend(["", "Validation reasons:", *(f"- {reason}" for reason in reasons)])
    elif unavailable:
        lines.extend([
            f"Why: {'; '.join(problems or gate.reasons) or 'Incomplete evaluation evidence'}.",
            "",
            "Why it matters: Invalid evidence cannot establish quality comparisons or savings.",
        ])
    elif gate.accepted:
        lines.extend(
            [
                "Why: The candidate met the bounded observed quality-loss allowance and token gates.",
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
                (
                    "Why it matters: A smaller artifact is not accepted when the quality-loss allowance, "
                    "critical/per-case floors, or token gates are not met."
                ),
            ]
        )
    if details.get("validation_baseline") and details.get("validation_candidate"):
        lines.extend(_phase_lines(
            "Validation", details["validation_baseline"], details["validation_candidate"],
            unavailable,
        ))
    if gate:
        lines.extend(_phase_lines(
            "Holdout", gate.baseline.to_dict(), gate.candidate.to_dict(), unavailable,
        ))
    if unavailable:
        lines.extend([
            "", "Behavior trial pass rate: unavailable.",
            "Reader trial pass rate: unavailable.",
            "Token savings: unavailable (incomplete, unaligned, or error-containing evidence).",
        ])
    lines.extend([
        "",
        (
            "Quality allowance: at most 5 absolute percentage points of observed pass-rate loss "
            "for behavior and reader independently, separately in validation and holdout; never pooled."
        ),
        (
            "With N aligned trials per phase, the rate gate alone permits floor(N/20) net lost passes "
            "per metric; per-case and critical constraints can be stricter."
        ),
        (
            "Candidate floors: no observed critical failures; at least one successful behavior trial "
            "and one successful reader trial per case."
        ),
        (
            "Holdout token gates: artifact, median output, and median reader tokens must not increase; "
            "combined artifact + median output reduction must meet the configured threshold (default 3%)."
        ),
        f"Model calls: {getattr(result, 'calls', 0)}",
        "",
        (
            "Tokens are local estimates, not billing. Reader scores are model-based proxies, "
            "not human comprehension or a statistical guarantee."
        ),
        "Output tokens need not fall if artifact savings meet the combined threshold.",
        "Fresh application calls do not establish independent model samples or universal behavior preservation.",
        "The source artifact was not modified.",
    ])
    control = details.get("concision_control")
    if control:
        lines.extend(["", f"Concision control: {control['status']} (diagnostic only; never selected)."])
        stats = control.get("aggregate")
        if stats and not stats["evidence_errors"] and control["status"] != "INCONCLUSIVE":
            lines.extend([
                f"Control artifact / median output tokens: {stats['artifact_tokens']} / {stats['median_output_tokens']:g}",
                (
                    "Control majority-passing cases (diagnostic only), behavior / reader: "
                    f"{stats['behavior_passes']} / {stats['understanding_passes']}"
                ),
                f"Control observed critical failures: {stats['critical_failures']}",
            ])
    return without_report_paths("\n".join(lines) + "\n", result)


def _phase_lines(phase: str, baseline: dict, candidate: dict, unavailable: bool) -> list[str]:
    lines = ["", f"{phase} (baseline → candidate):"]
    if unavailable or any(
        stats.get("evidence_errors") or stats.get("behavior_pass_rate") is None
        or stats.get("understanding_pass_rate") is None
        for stats in (baseline, candidate)
    ):
        return [*lines, "Behavior / reader trial pass rates and token comparison: unavailable."]
    for label, metric in (("Behavior", "behavior"), ("Reader", "understanding")):
        before, after = baseline[f"{metric}_pass_rate"], candidate[f"{metric}_pass_rate"]
        lines.append(
            f"{label} trial pass rate: "
            f"{baseline[f'{metric}_trial_passes']}/{baseline['total_trials']} ({before:.1%}) → "
            f"{candidate[f'{metric}_trial_passes']}/{candidate['total_trials']} ({after:.1%}); "
            f"change {(after - before) * 100:+.1f} percentage points."
        )
        lines.append(
            f"{label} majority-passing cases (diagnostic only): "
            f"{baseline[f'{metric}_passes']} → {candidate[f'{metric}_passes']}"
        )
    for label, key in (
        ("Artifact tokens", "artifact_tokens"),
        ("Median output tokens", "median_output_tokens"),
        ("Median reader tokens", "median_understanding_tokens"),
        ("Critical failures", "critical_failures"),
    ):
        lines.append(f"{label}: {baseline[key]:g} → {candidate[key]:g}")
    return lines
