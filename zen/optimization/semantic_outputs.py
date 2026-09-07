"""Run-scoped draft publication, source protection, and semantic reports."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from ..domain.core import OptimizationResult
from .report import without_report_paths


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        # Windows indexers can briefly hold a destination without delete sharing.
        # Retry only the local replace, never a model call or the optimization.
        for attempt in range(4):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def source_matches(result: OptimizationResult) -> bool:
    try:
        return (not result.details.get("source_changed") and
                digest(Path(result.artifact_path).read_bytes()) == result.details["source_sha256"])
    except OSError:
        return False


def _remove_publications(result: OptimizationResult) -> None:
    """Remove only unchanged files written by this run, never user edits."""
    for entry in result.details.get("publications", []):
        path = Path(entry["path"])
        if path.is_file() and digest(path.read_bytes()) == entry["sha256"]:
            path.unlink()


def _block_user_confirmation(result: OptimizationResult) -> None:
    details = result.details
    if details.get("user_acceptance", {}).get("status") == "ACCEPTED":
        details["user_acceptance"]["status"] = "BLOCKED"
        details["user_acceptance"]["reason"] = result.message
    details.pop("confirmation_origin", None)


def revoke_publication(result: OptimizationResult) -> None:
    """Revoke only unchanged candidate files written by this run, not user edits."""
    details = result.details
    _remove_publications(result)
    details["source_changed"] = True
    message = "Source changed; this run's external drafts and verification were revoked."
    if message not in details["errors"]:
        details["errors"].append(message)
    result.decision = "REVIEW_REQUIRED" if result.candidate_body is not None else "ERROR"
    result.message = message
    _block_user_confirmation(result)
    artifacts = details["artifacts"]
    artifacts["draft"] = artifacts["optimized"] = artifacts["run_optimized"] = None
    for entry in artifacts["drafts"]:
        entry["draft"] = None


def publish_file(result: OptimizationResult, internal: Path, destination: Path) -> Path | None:
    if not source_matches(result):
        revoke_publication(result)
        return None
    target = destination / internal.name
    data = internal.read_bytes()
    publications = result.details["publications"]
    if target.exists() and target.read_bytes() != data:
        raise OSError(f"refusing to overwrite changed output: {target}")
    atomic_bytes(target, data)
    entry = {"path": str(target), "sha256": digest(data)}
    if entry not in publications:
        publications.append(entry)
    if not source_matches(result):
        revoke_publication(result)
        return None
    return target


def render_semantic_report(result: OptimizationResult) -> str:
    details = result.details
    artifacts = details["artifacts"]
    lines = [
        f"# Zen semantic optimization: {PureWindowsPath(result.artifact_path).name}", "",
        f"Decision: {result.decision}", "", f"Why: {result.message}", "",
        "A draft is a reviewable rewrite, not verification. VERIFIED means only that",
        "the completed model-based checks found no material losses and met the token gate.",
        "CONFIRMED can mean final model adjudication or explicit user acceptance;",
        "the confirmation origin below distinguishes them. User acceptance does not pass failed checks.",
        "Model adjudication is not independent or human verification.",
        f"Confirmation origin: {details.get('confirmation_origin', 'model' if result.decision == 'CONFIRMED' else 'none')}",
        f"User acceptance: {details.get('user_acceptance', {}).get('status', 'PENDING')}", "",
        f"Model calls: {result.calls} / {details['budget']['limit']}",
        "", "## Current artifacts", "",
        f"Draft: {'available' if artifacts['draft'] else 'unavailable'}",
        f"Accepted candidate (VERIFIED or CONFIRMED): {'available' if artifacts['optimized'] else 'none'}", "",
    ]
    choice = details.get("user_acceptance", {})
    model = details.get("model_assessment")
    if model:
        lines.extend(["## User decision", "",
                      f"Model decision before user choice: {model['decision']}",
                      f"Model explanation: {model['message']}",
                      f"User acceptance recorded at: {choice.get('recorded_at', 'unavailable')}"])
        if choice.get("status") == "ACCEPTED":
            lines.append("Confirmed by user. Earlier model warnings and unmet gates remain below;")
            lines.append("this approval is not successful model verification or proof of quality.")
        if choice.get("reason"):
            lines.append(choice["reason"])
        lines.append("")
    lines.extend(["## Instruction review", ""])
    for index, review in enumerate(details["reviews"], 1):
        lines.append(f"### Review {index}")
        lines.append(review["rationale"])
        for item in review["differences"]:
            lines.append(f"- {item['severity']} {item['kind']}: {item['description']} — {item['rationale']}")
        if not review["differences"]:
            lines.append("- No concrete differences reported.")
        lines.append("")
    if not details["reviews"]:
        lines.append("No complete instruction review.")
    lines.extend(["", "## Paired task comparisons", ""])
    for phase, evidence in details["phases"].items():
        pairs = evidence["pairs"]
        lines.append(f"### {phase.title()}: {len(pairs)} / {evidence['expected']} case records")
        for pair in pairs:
            lines.append(f"- {pair['case_id']}: {pair['status']}")
            review = pair.get("review")
            if review:
                lines.append(f"  - {review['rationale']}")
                for item in review["differences"]:
                    lines.append(f"  - {item['severity']} {item['kind']}: {item['description']}")
                for item in review["important_constraint_violations"]:
                    lines.append(f"  - Constraint: {item['description']} — {item['rationale']}")
                for defect in review["baseline_defects"]:
                    lines.append(f"  - Baseline defect (not itself a regression): {defect}")
        lines.append("")
    final = details.get("final_review", {})
    lines.extend(["## Final adjudication", "", f"Status: {final.get('status', 'NOT_RUN')}"])
    if final.get("reason"):
        lines.append(final["reason"])
    preliminary = details.get("preliminary")
    if preliminary:
        lines.append(f"Initial decision: {preliminary['decision']}")
        lines.extend(f"- Initial flag: {reason}" for reason in preliminary["reasons"])
    assessment = final.get("assessment")
    if assessment:
        lines.extend([f"Final recommendation: {assessment['decision']}", assessment["rationale"],
                      (f"Instruction assessment: {assessment['instruction_assessment']['outcome']} — "
                       f"{assessment['instruction_assessment']['rationale']}")])
        for item in assessment["findings"]:
            lines.append(f"- {item['id']}: {item['classification']} — {item['rationale']}")
            lines.append(f"  Evidence: {', '.join(item['evidence_refs'])}")
        for phase, records in assessment["phases"].items():
            lines.append(f"### Final {phase} assessment")
            lines.extend(f"- {item['case_id']}: {item['outcome']} — {item['rationale']}" for item in records)
    lines.extend(["", "## Local token measurements", ""])
    metrics = details.get("tokens", {})
    labels = {
        "baseline_artifact": "Original artifact tokens",
        "candidate_artifact": "Draft artifact tokens",
        "baseline_median_output": "Original holdout median output tokens",
        "candidate_median_output": "Draft holdout median output tokens",
    }
    for key, label in labels.items():
        lines.append(f"- {label}: {metrics.get(key, 'unavailable')}")
    reduction = metrics.get("combined_reduction")
    lines.append(f"- Combined holdout reduction: {reduction:.1%}" if reduction is not None
                 else "- Combined holdout reduction: unavailable (incomplete paired evidence)")
    if details.get("reasons"):
        lines.extend(["", "## Review required", "", *[f"- {x}" for x in details["reasons"]]])
    if details["errors"]:
        lines.extend(["", "## Errors / incomplete evidence", "", *[f"- {x}" for x in details["errors"]]])
    lines.extend([
        "", "## Limits", "",
        "Source is never overwritten by Zen. Check the current manifest, not old output files.",
        "Counts are local o200k_base estimates, not provider billing or whole-task savings.",
        "Semantic judgments are model proxies, not universal equivalence or measured human comprehension.",
        "Cases are synthetic, single-turn and tool-free; tool claims are not execution evidence.",
        "Validation and holdout are separate; the candidate is frozen before both comparisons.",
        "Final adjudication reuses frozen evidence, not new executions; original findings remain above.",
        "Same-environment runs aid comparability, not independent samples or universal correctness.",
        "Fresh application calls do not establish provider independence.", "",
    ])
    return without_report_paths("\n".join(lines), result)


def save_summary(result: OptimizationResult) -> None:
    root = Path(result.run_directory)
    atomic_json(root / "summary.json", {
        "decision": result.decision, "artifact": result.artifact_path,
        "calls": result.calls, "message": result.message, "details": result.details,
    })
    atomic_bytes(root / "report.md", render_semantic_report(result).encode("utf-8"))


def _drafts_intact(result: OptimizationResult) -> bool:
    """Never republish a mutated run snapshot as the candidate that was reviewed."""
    artifacts = result.details["artifacts"]
    try:
        intact = not result.details.get("candidate_changed") and all(
            entry.get("sha256") and digest(Path(entry["run_draft"]).read_bytes()) == entry["sha256"]
            for entry in artifacts["drafts"]
        )
        optimized = artifacts["run_optimized"]
        if optimized:
            intact = intact and digest(Path(optimized).read_bytes()) == result.details["candidate_bytes_sha256"]
    except OSError:
        intact = False
    if not intact:
        result.decision = "REVIEW_REQUIRED"
        result.message = "Frozen candidate evidence changed or is missing; publication blocked."
        result.details["candidate_changed"] = True
        _remove_publications(result)
        _block_user_confirmation(result)
        if result.message not in result.details["errors"]:
            result.details["errors"].append(result.message)
        artifacts["draft"] = artifacts["optimized"] = artifacts["run_optimized"] = None
        for entry in artifacts["drafts"]:
            entry["draft"] = None
    return bool(intact)


def _record_user_choice(result: OptimizationResult, accept: bool) -> None:
    """User consent changes the result, never the recorded model assessment."""
    details = result.details
    details.setdefault("model_assessment", {
        "decision": result.decision, "message": result.message,
        "reasons": list(details.get("reasons", [])), "errors": list(details["errors"]),
    })
    choice = details["user_acceptance"] = {
        "status": "DECLINED", "recorded_at": datetime.now(UTC).isoformat(),
    }
    if not accept:
        return
    choice["status"] = "BLOCKED"
    artifacts = details["artifacts"]
    if not source_matches(result):
        revoke_publication(result)
        choice["reason"] = result.message
        return
    if not _drafts_intact(result):
        choice["reason"] = result.message
        return
    if result.candidate_body is None or not artifacts["drafts"] or not artifacts["draft"]:
        choice["reason"] = "No published draft is available to accept."
        return
    internal = Path(artifacts["drafts"][-1]["run_draft"])
    data = internal.read_bytes()
    try:
        intact = (digest(data) == details["candidate_bytes_sha256"]
                  and Path(artifacts["draft"]).read_bytes() == data)
    except OSError:
        intact = False
    if not intact:
        details["candidate_changed"] = True
        _drafts_intact(result)
        choice["reason"] = "Draft changed or is missing since evaluation; user confirmation blocked."
        return
    optimized = Path(result.run_directory) / (
        f"{Path(result.artifact_path).stem}.{details['run_id']}.optimized.md"
    )
    if optimized.exists() and optimized.read_bytes() != data:
        raise OSError(f"refusing to overwrite changed output: {optimized}")
    atomic_bytes(optimized, data)
    artifacts["run_optimized"] = str(optimized)
    choice.update(status="ACCEPTED", candidate_sha256=digest(data))
    details["confirmation_origin"] = "user"
    result.decision = "CONFIRMED"
    result.message = "Confirmed by user; prior model assessment and warnings are preserved in the report."


def write_semantic_outputs(
    result: OptimizationResult, output_directory: Path | None = None,
    *, user_acceptance: bool | None = None,
) -> tuple[Path | None, Path]:
    """Publish unique run files and update the current manifest; never select stale output."""
    destination = (output_directory or Path(result.details["output_directory"])).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result.details["output_directory"] = str(destination)
    artifacts = result.details["artifacts"]
    if user_acceptance is not None:
        _record_user_choice(result, user_acceptance)
    if not source_matches(result):
        revoke_publication(result)
    elif _drafts_intact(result):
        for entry in artifacts["drafts"]:
            external = publish_file(result, Path(entry["run_draft"]), destination)
            entry["draft"] = str(external) if external else None
        artifacts["draft"] = artifacts["drafts"][-1]["draft"] if artifacts["drafts"] else None
        if result.decision in ("VERIFIED", "CONFIRMED") and artifacts["run_optimized"]:
            external = publish_file(result, Path(artifacts["run_optimized"]), destination)
            artifacts["optimized"] = str(external) if external else None
    save_summary(result)
    stem = Path(result.artifact_path).stem
    report = destination / f"{stem}.{result.details['run_id']}.optimize.report.md"
    content = render_semantic_report(result).encode("utf-8")
    atomic_bytes(report, content)
    atomic_bytes(destination / f"{stem}.optimize.report.md", content)
    atomic_json(destination / f"{stem}.optimize.manifest.json", {
        "decision": result.decision, "run_directory": result.run_directory,
        "source_sha256": result.details["source_sha256"],
        "confirmation_origin": result.details.get(
            "confirmation_origin", "model" if result.decision == "CONFIRMED" else None,
        ),
        "user_acceptance": result.details.get("user_acceptance", {"status": "PENDING"}),
        "model_assessment": result.details.get("model_assessment"),
        "report": str(report), "artifacts": artifacts,
    })
    current = artifacts["optimized"] or artifacts["draft"]
    return Path(current) if current else None, report