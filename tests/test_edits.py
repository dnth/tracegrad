from __future__ import annotations

from tracegrad.edits import (
    REASON_CHECKSUM_MISMATCH,
    REASON_EMPTY_TEXT,
    REASON_NOT_EDITABLE,
    REASON_OVERLAP,
    REASON_UNKNOWN_ANCHOR,
    REASON_UNKNOWN_OPERATION,
    apply_edits,
    apply_resolved,
    resolve_edits,
)
from tracegrad.inventory import Instruction, Inventory, build_inventory
from tracegrad.schema import Edit


def test_add_inserts_sibling_after_anchor_span_inheriting_marker_and_indentation() -> None:
    prompt = "- Always cite sources.\n- Be concise.\n- Avoid speculation.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[0]
    edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="ADD",
        text="Respond in English.",
        covers_theme="language",
        watch_metric="language",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert not resolution.rejected
    assert new_prompt == (
        "- Always cite sources.\n- Respond in English.\n"
        "- Be concise.\n- Avoid speculation.\n"
    )


def test_add_text_already_marked_is_not_double_marked() -> None:
    prompt = "- Always cite sources.\n- Be concise.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[0]
    edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="ADD",
        text="- Respond in English.",
        covers_theme="language",
        watch_metric="language",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert not resolution.rejected
    assert "- - Respond in English." not in new_prompt
    assert "- Respond in English." in new_prompt


def test_rewrite_replaces_span_preserving_anchor_indentation_and_marker() -> None:
    prompt = "- Always cite sources.\n- Be concise.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[1]
    edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="REWRITE",
        text="Be extremely concise and precise.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert not resolution.rejected
    assert new_prompt == "- Always cite sources.\n- Be extremely concise and precise.\n"


def test_delete_removes_line_and_newline_leaving_no_blank_hole() -> None:
    prompt = "- Always cite sources.\n- Be concise.\n- Avoid speculation.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[1]
    edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="DELETE",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert not resolution.rejected
    assert new_prompt == "- Always cite sources.\n- Avoid speculation.\n"
    assert "\n\n" not in new_prompt


def test_start_sentinel_inserts_at_the_front() -> None:
    prompt = "- Be concise.\n"
    inventory = build_inventory(prompt)
    edit = Edit(
        instruction_id="START",
        operation="ADD",
        text="System note.",
        covers_theme="system",
        watch_metric="system",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert not resolution.rejected
    assert new_prompt == "System note.\n- Be concise.\n"


def test_end_sentinel_inserts_at_the_back() -> None:
    prompt = "- Be concise.\n"
    inventory = build_inventory(prompt)
    edit = Edit(
        instruction_id="END",
        operation="ADD",
        text="Closing note.",
        covers_theme="system",
        watch_metric="system",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert not resolution.rejected
    assert new_prompt == "- Be concise.\nClosing note.\n"


def test_append_sentinel_also_inserts_at_the_back() -> None:
    prompt = "- Be concise."
    inventory = build_inventory(prompt)
    edit = Edit(
        instruction_id="APPEND",
        operation="ADD",
        text="Closing note.",
        covers_theme="system",
        watch_metric="system",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert not resolution.rejected
    assert new_prompt == "- Be concise.\nClosing note.\n"


def test_orphan_anchor_is_rejected_as_unknown() -> None:
    prompt = "- Be concise.\n"
    inventory = build_inventory(prompt)
    edit = Edit(
        instruction_id="i-000000000000-00",
        operation="REWRITE",
        text="Something else.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    resolution = resolve_edits(inventory, [edit])

    assert not resolution.resolved
    assert len(resolution.rejected) == 1
    assert resolution.rejected[0].reason == REASON_UNKNOWN_ANCHOR


def test_checksum_mismatch_is_rejected() -> None:
    # Hand-build an inventory whose Instruction id embeds a checksum that does
    # not match its own text, then target it with an edit carrying that same
    # (stale/wrong) id.  ``inventory.get`` resolves the anchor purely on id
    # equality, so this reaches the checksum comparison inside resolve_edits
    # rather than the orphan-anchor path.
    text = "- Be concise."
    stale_id = "i-000000000000-00"  # correctly shaped, but wrong checksum for `text`
    anchor = Instruction(
        instruction_id=stale_id,
        lineage_id=stale_id,
        text=text,
        start=0,
        end=len(text),
        kind="bullet",
        ordinal=0,
        fingerprint="deadbeefdeadbeef",
        origin="template",
        editable=True,
    )
    inventory = Inventory(prompt=text, instructions=(anchor,))
    edit = Edit(
        instruction_id=stale_id,
        operation="REWRITE",
        text="Be extremely concise.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    resolution = resolve_edits(inventory, [edit])

    assert not resolution.resolved
    assert len(resolution.rejected) == 1
    assert resolution.rejected[0].reason == REASON_CHECKSUM_MISMATCH


def test_variable_origin_anchor_is_not_editable() -> None:
    prompt = "- Hello {name}, welcome.\n"
    inventory = build_inventory(prompt, variable_spans=[(9, 13)])
    anchor = inventory.instructions[0]
    assert anchor.editable is False
    edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="REWRITE",
        text="Hello there, welcome.",
        covers_theme="greeting",
        watch_metric="greeting",
    )

    resolution = resolve_edits(inventory, [edit])

    assert not resolution.resolved
    assert len(resolution.rejected) == 1
    assert resolution.rejected[0].reason == REASON_NOT_EDITABLE


def test_empty_text_on_add_and_rewrite_is_rejected() -> None:
    prompt = "- Be concise.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[0]
    add_edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="ADD",
        text="   ",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )
    rewrite_edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="REWRITE",
        text="",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    resolution = resolve_edits(inventory, [add_edit, rewrite_edit])

    assert not resolution.resolved
    assert len(resolution.rejected) == 2
    assert all(rejection.reason == REASON_EMPTY_TEXT for rejection in resolution.rejected)


def test_unknown_operation_is_rejected() -> None:
    prompt = "- Be concise.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[0]
    # Pydantic's pattern constraint on Edit.operation forbids constructing this
    # through the normal API, so bypass validation to exercise the fallback
    # branch inside resolve_edits directly.
    edit = Edit.model_construct(
        instruction_id=anchor.instruction_id,
        operation="MOVE",
        text="Be concise.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    resolution = resolve_edits(inventory, [edit])

    assert not resolution.resolved
    assert len(resolution.rejected) == 1
    assert resolution.rejected[0].reason == REASON_UNKNOWN_OPERATION


def test_overlapping_spans_keep_one_and_reject_the_other_deterministically() -> None:
    prompt = "- Be concise.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[0]
    first = Edit(
        instruction_id=anchor.instruction_id,
        operation="REWRITE",
        text="Be very concise.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )
    second = Edit(
        instruction_id=anchor.instruction_id,
        operation="REWRITE",
        text="Be extremely terse.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    resolution = resolve_edits(inventory, [first, second])

    assert len(resolution.resolved) == 1
    assert resolution.resolved[0].edit is first
    assert len(resolution.rejected) == 1
    assert resolution.rejected[0].edit is second
    assert resolution.rejected[0].reason == REASON_OVERLAP

    # Deterministic regardless of which order the caller happens to build the
    # list in — reversing the input flips which edit wins.
    resolution_reversed = resolve_edits(inventory, [second, first])
    assert resolution_reversed.resolved[0].edit is second
    assert resolution_reversed.rejected[0].edit is first


def test_right_to_left_application_lands_all_three_edits_correctly() -> None:
    prompt = "- One.\n- Two.\n- Three.\n"
    inventory = build_inventory(prompt)
    add_edit = Edit(
        instruction_id=inventory.instructions[0].instruction_id,
        operation="ADD",
        text="Number one confirmed.",
        covers_theme="a",
        watch_metric="a",
    )
    rewrite_edit = Edit(
        instruction_id=inventory.instructions[1].instruction_id,
        operation="REWRITE",
        text="Two revised.",
        covers_theme="b",
        watch_metric="b",
    )
    delete_edit = Edit(
        instruction_id=inventory.instructions[2].instruction_id,
        operation="DELETE",
        covers_theme="c",
        watch_metric="c",
    )

    new_prompt, resolution = apply_edits(inventory, [add_edit, rewrite_edit, delete_edit])

    assert not resolution.rejected
    assert len(resolution.resolved) == 3
    assert new_prompt == "- One.\n- Number one confirmed.\n- Two revised.\n"


def test_apply_edits_returns_new_prompt_and_resolution() -> None:
    prompt = "- Be concise.\n"
    inventory = build_inventory(prompt)
    anchor = inventory.instructions[0]
    edit = Edit(
        instruction_id=anchor.instruction_id,
        operation="REWRITE",
        text="Be terse.",
        covers_theme="verbosity",
        watch_metric="verbosity",
    )

    new_prompt, resolution = apply_edits(inventory, [edit])

    assert new_prompt == "- Be terse.\n"
    assert resolution.resolved[0].edit is edit
    # apply_resolved on the resolution's own resolved list reproduces the same
    # prompt, confirming apply_edits is just resolve + apply composed.
    assert apply_resolved(inventory.prompt, resolution.resolved) == new_prompt
