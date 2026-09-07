"""Offline, scripted-model tests of local mutation policy, not GEPA performance."""

from __future__ import annotations

import hashlib
import json

import pytest

from zen.domain.core import BehaviorContract, Rule
from zen.optimization.proposer import CompressionProposer, MutationScope, validate_focus
from zen.runtime.lm import BudgetExceeded, FunctionModel

TASK_OPEN = "<!-- zen:task -->"
TASK_CLOSE = "<!-- /zen:task -->"
COMM_OPEN = "<!-- zen:communication -->"
COMM_CLOSE = "<!-- /zen:communication -->"
SOURCE = "Keep important behavior.\nThis is redundant guidance.\nMore redundant guidance.\n"
SCOPED = (
    "Header  \r\n"
    f"{TASK_OPEN}\r\n"
    "Complete the task carefully and correctly.\r\n"
    "This is redundant guidance.\r\n"
    f"{TASK_CLOSE}\r\n"
    "Unrelated middle text.\r\n"
    f"{COMM_OPEN}\r\n"
    "Explain the result clearly and concisely.\r\n"
    "More redundant guidance.\r\n"
    f"{COMM_CLOSE}\r\n"
    "Footer without newline  "
)


def _make(
    responses: list[str],
    original: str = SOURCE,
    *,
    focus: str = "all",
    max_lines: int | None = None,
    contract: BehaviorContract | None = None,
) -> tuple[CompressionProposer, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []
    scripted = iter(responses)

    def complete(system: str, user: str) -> str:
        calls.append((system, json.loads(user)))
        return next(scripted)

    return CompressionProposer(
        FunctionModel(complete),
        contract or BehaviorContract("English", "Keep behavior", ()),
        original,
        max_lines,
        focus=focus,
    ), calls


def _call(proposer: CompressionProposer, current: str = SOURCE) -> str:
    return proposer({"answer": current}, {}, ["answer"])["answer"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_alternates_removal_then_rewrite_without_search_or_selecting() -> None:
    proposer, calls = _make(["[2]", "Keep important behavior.", "[3]"])
    assert _call(proposer) == "Keep important behavior.\nMore redundant guidance.\n"
    assert _call(proposer) == "Keep important behavior."
    # GEPA's supplied current is authoritative, not the previous local proposal.
    assert _call(proposer) == "Keep important behavior.\nThis is redundant guidance.\n"
    assert [entry["strategy"] for entry in proposer.history] == ["removal", "rewrite", "removal"]
    assert len(calls) == 3


def test_components_share_callback_strategy_and_empty_callback_does_not_advance() -> None:
    proposer, calls = _make(["[2]", "[3]", "Keep behavior.", "Keep important behavior."])
    assert proposer({}, {}, []) == {}
    assert calls == [] and proposer.history == []
    candidate = {"a": SOURCE, "b": SOURCE, "untouched": "Do not update."}
    result = proposer(candidate, {"a": [{"feedback": "Too repetitive"}]}, ["a", "b"])
    assert set(result) == {"a", "b"}
    assert candidate["a"] == SOURCE
    assert calls[0][1]["evaluation_examples"] == [{"feedback": "Too repetitive"}]
    assert calls[1][1]["evaluation_examples"] == []
    proposer(candidate, {}, ["a", "b"])
    assert [entry["strategy"] for entry in proposer.history] == ["removal"] * 2 + ["rewrite"] * 2


@pytest.mark.parametrize(
    "response",
    [
        "[]", "{}", '{"lines": [2]}', "null", "2", '"[2]"',
        "[true]", "[false]", "[2.0]", '["2"]', "[null]", "[[2]]",
        "[2, 2]", "[3, 2]", "[0]", "[-1]", "[4]", "[999999999999999999999]",
        "[2,]", "[2] explanation", "```json\n[2]\n```", "not JSON",
    ],
)
def test_invalid_removal_returns_current_without_fallback(response: str) -> None:
    proposer, calls = _make([response])
    assert _call(proposer) == SOURCE
    assert len(calls) == 1
    entry = proposer.history[0]
    assert entry["strategy"] == "removal"
    assert entry["accepted_by_policy"] is False
    assert entry["reason"]
    assert entry["before_sha256"] == entry["after_sha256"] == _sha(SOURCE)


def test_removal_preserves_exact_whole_lines_and_line_endings() -> None:
    source = "Keep behavior.\r\n\r\nDuplicate.\nDuplicate.\rLast line"
    proposer, calls = _make(["[2, 4]"], source)
    assert _call(proposer, source) == "Keep behavior.\r\nDuplicate.\nLast line"
    assert [line["index"] for line in calls[0][1]["numbered_lines"]] == [1, 2, 3, 4, 5]
    assert calls[0][1]["numbered_lines"][1]["text"] == "\r\n"
    assert proposer.history[0]["removed_lines"] == [2, 4]


def test_deleting_every_line_fails_candidate_policy() -> None:
    proposer, calls = _make(["[1, 2, 3]"])
    assert _call(proposer) == SOURCE
    assert "empty" in proposer.history[0]["reason"]
    assert proposer.history[0]["proposed_sha256"] == _sha("")
    assert len(calls) == 1


def test_empty_body_has_no_valid_deletion_indices() -> None:
    proposer, _ = _make(["[1]"], "")
    assert _call(proposer, "") == ""
    assert "out of range" in proposer.history[0]["reason"]


def test_critical_obligations_and_prohibitions_protect_all_overlapping_lines() -> None:
    source = "prefix Keep safe.\nNever expose\nsecrets.\nKeep safe.\nRedundant.\n"
    contract = BehaviorContract(
        "English", "Safety",
        (Rule("O1", "Safety", "Keep safe."),),
        (Rule("P1", "Privacy", "Never expose\nsecrets."),),
    )
    for index in range(1, 5):
        proposer, calls = _make([f"[{index}]"], source, contract=contract)
        assert _call(proposer, source) == source
        assert "protected" in proposer.history[0]["reason"]
        assert [line["protected"] for line in calls[0][1]["numbered_lines"]] == [True] * 4 + [False]
        assert calls[0][1]["protected_source_quotes"] == ["Keep safe.", "Never expose\nsecrets."]
        assert "protected" in calls[0][0]
    proposer, _ = _make(["[5]"], source, contract=contract)
    assert _call(proposer, source) == source.removesuffix("Redundant.\n")


def test_noncritical_empty_or_absent_quotes_do_not_protect_unrelated_lines() -> None:
    contract = BehaviorContract(
        "English", "Safety",
        (
            Rule("O1", "Redundancy", "This is redundant guidance.", "normal"),
            Rule("O2", "Absent", "not in the source"),
            Rule("O3", "Empty", ""),
        ),
    )
    proposer, _ = _make(["[2]"], contract=contract)
    assert _call(proposer) != SOURCE


@pytest.mark.parametrize("fence", ["```python", "~~~~", "  ````text"])
@pytest.mark.parametrize("line_index", [2, 3, 4])
def test_removal_freezes_fence_delimiters_and_contents(fence: str, line_index: int) -> None:
    closing = "~~~~" if "~" in fence else ("`````" if "````" in fence else "```")
    source = f"Keep behavior.\n{fence}\nprint('same')\n{closing}\nRedundant.\n"
    proposer, _ = _make([f"[{line_index}]"], source)
    assert _call(proposer, source) == source
    assert "protected" in proposer.history[0]["reason"]


def test_removal_can_delete_outside_fence_but_freezes_unclosed_fence_to_eof() -> None:
    source = "Keep behavior.\nRedundant.\n```python\ncode\n~~~\nmore code"
    proposer, _ = _make(["[2]"], source)
    assert _call(proposer, source) == source.replace("Redundant.\n", "")
    proposer, _ = _make(["[6]"], source)
    assert _call(proposer, source) == source


@pytest.mark.parametrize("focus,index", [("task", 4), ("communication", 9)])
def test_scoped_removal_changes_only_selected_whole_line(focus: str, index: int) -> None:
    proposer, _ = _make([f"[{index}]"], SCOPED, focus=focus)
    expected = "".join(line for i, line in enumerate(SCOPED.splitlines(keepends=True), 1) if i != index)
    actual = _call(proposer, SCOPED)
    assert actual == expected
    proposer.scope.validate(actual)
    assert proposer.history[0]["focus"] == focus


@pytest.mark.parametrize("index", [1, 2, 5, 6, 7, 8, 9, 10, 11])
def test_scoped_removal_rejects_markers_other_section_and_outside(index: int) -> None:
    proposer, _ = _make([f"[{index}]"], SCOPED, focus="task")
    assert _call(proposer, SCOPED) == SCOPED
    assert "protected" in proposer.history[0]["reason"]


@pytest.mark.parametrize("focus", ["task", "communication"])
def test_scoped_rewrite_splices_raw_region_without_normalizing_outside(focus: str) -> None:
    replacement = "\r\nKeep behavior.\r\n"
    proposer, calls = _make(["[]", replacement], SCOPED, focus=focus)
    _call(proposer, SCOPED)
    actual = _call(proposer, SCOPED)
    scope = validate_focus(SCOPED, focus)
    assert actual == scope.prefix + replacement + scope.suffix
    assert scope.extract(actual) == replacement
    assert calls[1][1]["mutable_instruction"] == scope.extract(SCOPED)
    assert "outside" in calls[1][0] and "fenced-code" in calls[1][0]
    assert proposer.history[-1]["accepted_by_policy"] is True


def test_inline_marker_region_is_exact_and_not_removable_as_a_partial_line() -> None:
    source = f"Header {TASK_OPEN}Long and repetitive task text.{TASK_CLOSE} Footer"
    scope = validate_focus(source, "task")
    assert scope.extract(source) == "Long and repetitive task text."
    proposer, _ = _make(["[1]", "Task."], source, focus="task")
    assert _call(proposer, source) == source
    assert _call(proposer, source) == f"Header {TASK_OPEN}Task.{TASK_CLOSE} Footer"


@pytest.mark.parametrize(
    "bad",
    [
        f"{TASK_OPEN}new{TASK_CLOSE}",
        TASK_CLOSE,
        f"{COMM_OPEN}new{COMM_CLOSE}",
        "<!-- zen:task-->",
        SCOPED,
    ],
)
def test_scoped_rewrite_rejects_marker_injection_and_full_body_response(bad: str) -> None:
    proposer, calls = _make(["[]", bad], SCOPED, focus="task")
    _call(proposer, SCOPED)
    assert _call(proposer, SCOPED) == SCOPED
    assert proposer.history[-1]["accepted_by_policy"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("replacement", ["\nUse code.\n", "\n```py\ny()\n```\n"])
def test_scoped_rewrite_rejects_deleted_or_modified_fenced_block(replacement: str) -> None:
    source = f"{TASK_OPEN}\nVery verbose guidance.\n```py\nx()\n```\n{TASK_CLOSE}"
    proposer, _ = _make(["[]", replacement], source, focus="task")
    _call(proposer, source)
    assert _call(proposer, source) == source
    assert "fenced-code" in proposer.history[-1]["reason"]


def test_scoped_rewrite_preserves_exact_fenced_block_while_shortening_prose() -> None:
    source = f"{TASK_OPEN}\r\nVery verbose guidance.\r\n~~~py\r\nx()\r\n~~~\r\n{TASK_CLOSE}"
    replacement = "\r\nUse:\r\n~~~py\r\nx()\r\n~~~\r\n"
    proposer, _ = _make(["[]", replacement], source, focus="task")
    _call(proposer, source)
    assert _call(proposer, source) == f"{TASK_OPEN}{replacement}{TASK_CLOSE}"


def test_scoped_rewrite_rejects_reordered_fenced_blocks() -> None:
    first, second = "```\na\n```\n", "~~~\nb\n~~~\n"
    source = f"{TASK_OPEN}\n{first}Redundant prose.\n{second}{TASK_CLOSE}"
    proposer, _ = _make(["[]", "\n" + second + first], source, focus="task")
    _call(proposer, source)
    assert _call(proposer, source) == source
    assert "fenced-code" in proposer.history[-1]["reason"]


def test_validate_focus_returns_reusable_guard_for_final_candidate() -> None:
    scope = validate_focus(SCOPED, "task")
    assert isinstance(scope, MutationScope)
    shorter = scope.replace(SCOPED, "\r\nDo the task.\r\n")
    scope.validate(shorter)
    assert scope.extract(shorter) == "\r\nDo the task.\r\n"
    # Validate a later GEPA candidate with different region length, not fixed offsets.
    scope.validate(scope.replace(shorter, "Task."))
    for changed in (
        shorter.replace("Header  ", "Header "),
        shorter.replace("Unrelated middle", "Changed middle"),
        shorter.replace("Explain the result", "Describe the result"),
        shorter + "\n",
        shorter.replace("\r\n", "\n"),
    ):
        with pytest.raises(ValueError, match="outside"):
            scope.validate(changed)
        with pytest.raises(ValueError, match="outside"):
            scope.replace(changed, "Task.")


@pytest.mark.parametrize("focus", ["", "Task", "both", "unknown"])
def test_unknown_focus_fails_early(focus: str) -> None:
    with pytest.raises(ValueError, match="focus must"):
        _make([], focus=focus)


@pytest.mark.parametrize(
    "source",
    [
        "No markers", TASK_OPEN, TASK_CLOSE, TASK_CLOSE + TASK_OPEN,
        TASK_OPEN + TASK_CLOSE + TASK_OPEN + TASK_CLOSE,
        TASK_OPEN + TASK_OPEN + TASK_CLOSE + TASK_CLOSE,
        TASK_OPEN + COMM_OPEN + COMM_CLOSE + TASK_CLOSE,
        TASK_OPEN + COMM_CLOSE,
        TASK_OPEN + COMM_OPEN + TASK_CLOSE + COMM_CLOSE,
        "<!--zen:task -->x" + TASK_CLOSE,
        "<!-- zen:task-->x" + TASK_CLOSE,
        "<!-- zen:task >x" + TASK_CLOSE,
        "<!-- zen:task -->x<!-- /zen:task",
        "<!-- ZEN:task -->x" + TASK_CLOSE,
        "<!-- zen :task -->x" + TASK_CLOSE,
        TASK_OPEN + "x<!-- / zen:task -->",
        TASK_OPEN + "x" + TASK_CLOSE + COMM_OPEN,
        TASK_OPEN + TASK_CLOSE + COMM_OPEN + COMM_CLOSE + COMM_OPEN + COMM_CLOSE,
        TASK_OPEN + TASK_CLOSE + "<!-- zen:unknown -->",
    ],
)
def test_absent_duplicate_nested_or_malformed_markers_fail_before_model_call(source: str) -> None:
    with pytest.raises(ValueError):
        _make([], source, focus="task")


def test_selected_pair_is_required_but_other_pair_optional_and_order_unrestricted() -> None:
    source = f"{COMM_OPEN}Communication{COMM_CLOSE}"
    validate_focus(source, "communication")
    with pytest.raises(ValueError, match="absent task"):
        validate_focus(source, "task")
    validate_focus(source + TASK_OPEN + TASK_CLOSE, "task")


def test_all_focus_keeps_whole_body_semantics_even_with_bad_markers() -> None:
    source = f"{TASK_OPEN}\nKeep behavior.\nRedundant.\n"
    scope = validate_focus(source, "all")
    assert scope.extract(source) == source
    assert scope.replace(source, "Anything.") == "Anything."
    scope.validate("No markers needed.")
    proposer, _ = _make(["[1]", "Keep behavior."], source)
    assert _call(proposer, source) == "Keep behavior.\nRedundant.\n"
    assert _call(proposer, source) == "Keep behavior."


def test_changed_incoming_scope_is_rejected_without_model_call() -> None:
    proposer, calls = _make([], SCOPED, focus="task")
    current = SCOPED.replace("Header", "Changed header")
    assert _call(proposer, current) == current
    assert calls == []
    assert "outside" in proposer.history[0]["reason"]


def test_rewrite_retains_legacy_unwrapping_in_all_mode_and_protected_prompt() -> None:
    contract = BehaviorContract("English", "Safety", (Rule("O1", "Safety", "Keep important behavior."),))
    proposer, calls = _make(["[]", "```text\nKeep behavior.\n```"], contract=contract)
    _call(proposer)
    assert _call(proposer) == "Keep behavior."
    assert calls[1][1]["protected_source_quotes"] == ["Keep important behavior."]
    assert "critical contract evidence" in calls[1][0]


def test_rewrite_aggressive_retry_is_one_extra_attempt_with_contract_and_feedback() -> None:
    draft = "Keep important behavior.\nKeep important behavior.\nKeep important behavior."
    proposer, calls = _make(["[]", draft, "Keep important behavior."], max_lines=1)
    _call(proposer)
    feedback = {"answer": [{"feedback": "Combine redundant instructions"}]}
    assert proposer({"answer": SOURCE}, feedback, ["answer"])["answer"] == "Keep important behavior."
    assert len(calls) == 3
    assert calls[-1][1]["draft"] == draft
    assert calls[-1][1]["line_limit"] == 1
    assert calls[-1][1]["evaluation_examples"] == feedback["answer"]
    assert calls[-1][1]["contract"] == calls[-2][1]["contract"]
    assert [entry["attempt"] for entry in proposer.history] == [1, 1, 2]
    assert [entry["accepted_by_policy"] for entry in proposer.history] == [False, False, True]


def test_invalid_aggressive_retry_returns_current_without_third_attempt() -> None:
    proposer, calls = _make(["[]", SOURCE, "Still\nToo long\n"], max_lines=1)
    _call(proposer)
    assert _call(proposer) == SOURCE
    assert len(calls) == 3
    assert proposer.history[-1]["accepted_by_policy"] is False


def test_invalid_removal_never_gets_aggressive_retry() -> None:
    proposer, calls = _make(["[2]"], max_lines=1)
    assert _call(proposer) == SOURCE
    assert "line limit" in proposer.history[0]["reason"]
    assert len(calls) == 1


@pytest.mark.parametrize("draft", ["", "An extremely long explanation. " * 100])
def test_invalid_rewrite_without_oversized_line_count_does_not_retry(draft: str) -> None:
    proposer, calls = _make(["[]", draft], max_lines=10)
    _call(proposer)
    assert _call(proposer) == SOURCE
    assert len(calls) == 2
    assert proposer.history[-1]["accepted_by_policy"] is False


def test_unchanged_rewrite_is_logged_as_no_mutation() -> None:
    source = SOURCE.rstrip()
    proposer, calls = _make(["[]", source], source)
    _call(proposer, source)
    assert _call(proposer, source) == source
    assert len(calls) == 2
    assert proposer.history[-1]["accepted_by_policy"] is False
    assert proposer.history[-1]["reason"] == "rewrite made no change"


def test_scoped_aggressive_retry_obeys_whole_body_limit_and_immutable_context() -> None:
    source = f"Header\n{TASK_OPEN}\nLong task instructions.\nRedundant task instructions.\n{TASK_CLOSE}\nFooter"
    replacement = "\nDo task.\n"
    proposer, calls = _make(["[]", "\nToo\nmany\nlines\n", replacement], source, focus="task", max_lines=5)
    _call(proposer, source)
    assert _call(proposer, source) == validate_focus(source, "task").replace(source, replacement)
    assert len(calls) == 3
    assert "outside" in calls[-1][0]
    assert calls[-1][1]["focus"] == "task"


def test_candidate_policy_rejects_losing_original_writing_system() -> None:
    source = "결과와 이유를 먼저 자세하게 설명하세요.\nEnglish summary.\n"
    proposer, _ = _make(["[1]", "English."], source)
    assert _call(proposer, source) == source
    assert _call(proposer, source) == source
    assert all("writing system" in entry["reason"] for entry in proposer.history)


def test_history_hashes_diffs_and_policy_status_are_audit_not_performance_evidence() -> None:
    proposer, _ = _make(["[2]", "Too much text. " * 100])
    result = _call(proposer)
    accepted = proposer.history[0]
    assert accepted["before_sha256"] == _sha(SOURCE)
    assert accepted["after_sha256"] == accepted["proposed_sha256"] == _sha(result)
    assert accepted["removed_lines"] == [2]
    assert "-This is redundant guidance." in accepted["unified_diff"]
    assert "GEPA has not judged or selected" in accepted["reason"]
    assert accepted["accepted_by_policy"] is True
    assert _call(proposer) == SOURCE
    rejected = proposer.history[-1]
    assert rejected["before_sha256"] == rejected["after_sha256"] == _sha(SOURCE)
    assert rejected["proposed_sha256"] == _sha(("Too much text. " * 100).strip())
    assert rejected["unified_diff"].startswith("--- current\n+++ proposed\n")
    assert rejected["accepted_by_policy"] is False
    assert "token count" in rejected["reason"]
    json.dumps(proposer.history)


def test_model_budget_failure_propagates_instead_of_starting_fallback_search() -> None:
    def complete(_system: str, _user: str) -> str:
        raise BudgetExceeded("test budget")

    proposer = CompressionProposer(FunctionModel(complete), BehaviorContract("English", "Test", ()), SOURCE)
    with pytest.raises(BudgetExceeded, match="test budget"):
        _call(proposer)