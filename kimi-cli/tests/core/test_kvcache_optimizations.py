"""Tests for LLM KV-cache optimization changes.

Covers:
- Phase 1: Ephemeral message cleanup (remove_by_predicate)
- Phase 2: Incremental normalization
- Phase 3: System prompt cache fix
- Phase 4: Prefix stability verification
- Issue 6: Context.history defensive copy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kosong.message import Message

from kimi_cli.notifications.llm import build_notification_message, is_notification_message
from kimi_cli.soul.agent import Agent
from kimi_cli.soul.context import Context
from kimi_cli.soul.dynamic_injection import normalize_history
from kimi_cli.soul.message import is_system_reminder_message, system_reminder
from kimi_cli.wire.types import TextPart


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextPart(text=text)])


def _make_system_reminder_msg(text: str) -> Message:
    part = system_reminder(text)
    return Message(role="user", content=[part])


def _make_notification_msg(notif_id: str = "n1") -> Message:
    """Builds a minimal notification message that is_notification_message detects."""
    return Message(
        role="user",
        content=[
            TextPart(
                text=(
                    f'<notification id="{notif_id}" category="test" '
                    f'type="test" source_kind="test" source_id="s1">\n'
                    f"Title: Test\nSeverity: info\nBody\n</notification>"
                )
            )
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Ephemeral message cleanup — Context.remove_by_predicate
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_by_predicate_removes_matching_messages(tmp_path: Path) -> None:
    """remove_by_predicate removes messages matching the predicate."""
    path = tmp_path / "context.jsonl"
    ctx = Context(file_backend=path)
    await ctx.append_message(_make_msg("user", "Hello"))
    await ctx.append_message(_make_msg("assistant", "Hi"))
    await ctx.append_message(_make_msg("user", "World"))

    removed = ctx.remove_by_predicate(lambda m: m.role == "user")
    assert removed == 2
    assert len(ctx.history) == 1
    assert ctx.history[0].role == "assistant"


@pytest.mark.asyncio
async def test_remove_by_predicate_no_matches(tmp_path: Path) -> None:
    """remove_by_predicate returns 0 when nothing matches."""
    path = tmp_path / "context.jsonl"
    ctx = Context(file_backend=path)
    await ctx.append_message(_make_msg("assistant", "Hi"))

    removed = ctx.remove_by_predicate(lambda m: m.role == "user")
    assert removed == 0
    assert len(ctx.history) == 1


@pytest.mark.asyncio
async def test_remove_by_predicate_empty_history(tmp_path: Path) -> None:
    """remove_by_predicate handles empty history gracefully."""
    path = tmp_path / "context.jsonl"
    ctx = Context(file_backend=path)

    removed = ctx.remove_by_predicate(lambda m: True)
    assert removed == 0
    assert len(ctx.history) == 0


@pytest.mark.asyncio
async def test_remove_system_reminder_messages(tmp_path: Path) -> None:
    """is_system_reminder_message correctly identifies and removes injection messages."""
    path = tmp_path / "context.jsonl"
    ctx = Context(file_backend=path)
    await ctx.append_message(_make_msg("user", "Real user question"))
    await ctx.append_message(_make_system_reminder_msg("Plan mode is active."))
    await ctx.append_message(_make_msg("assistant", "I'll help."))

    assert len(ctx.history) == 3

    removed = ctx.remove_by_predicate(is_system_reminder_message)
    assert removed == 1
    assert len(ctx.history) == 2
    assert ctx.history[0].role == "user"
    assert "Real user question" in str(ctx.history[0].content)
    assert ctx.history[1].role == "assistant"


@pytest.mark.asyncio
async def test_remove_notification_messages(tmp_path: Path) -> None:
    """is_notification_message correctly identifies and removes notification messages."""
    path = tmp_path / "context.jsonl"
    ctx = Context(file_backend=path)
    await ctx.append_message(_make_msg("user", "Real user question"))
    await ctx.append_message(_make_notification_msg("n1"))
    await ctx.append_message(_make_msg("assistant", "Got it."))

    assert len(ctx.history) == 3

    removed = ctx.remove_by_predicate(is_notification_message)
    assert removed == 1
    assert len(ctx.history) == 2
    assert ctx.history[0].role == "user"
    assert "Real user question" in str(ctx.history[0].content)
    assert ctx.history[1].role == "assistant"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: System prompt cache fix — Agent.get_system_prompt
# ──────────────────────────────────────────────────────────────────────────────


class _FakeAgent(Agent):
    """Test-only Agent that bypasses the real __init__ requirements."""

    def __init__(self, system_prompt_callable) -> None:
        # Bypass normal dataclass init
        self.name = "test"
        self.system_prompt = system_prompt_callable
        self.system_prompt_cached = None
        self.toolset = None  # type: ignore[assignment]
        self.runtime = None  # type: ignore[assignment]


def test_system_prompt_cache_not_overwritten_by_compacting() -> None:
    """Compacting call must not overwrite the normal cached system prompt."""
    call_log: list[tuple[Any, bool]] = []

    def prompt_func(runtime, is_compacting: bool) -> str:
        call_log.append((runtime, is_compacting))
        if is_compacting:
            return "COMPACTING PROMPT"
        return "NORMAL PROMPT"

    agent = _FakeAgent(prompt_func)

    # First call: normal
    result1 = agent.get_system_prompt(is_compacting=False)
    assert result1 == "NORMAL PROMPT"
    assert agent.system_prompt_cached == "NORMAL PROMPT"

    # Second call: compacting — must NOT overwrite cache
    result2 = agent.get_system_prompt(is_compacting=True)
    assert result2 == "COMPACTING PROMPT"
    assert agent.system_prompt_cached == "NORMAL PROMPT", (
        "Compacting call must not overwrite the normal cached prompt"
    )

    # Third call: normal — must return cached value without re-calling
    result3 = agent.get_system_prompt(is_compacting=False)
    assert result3 == "NORMAL PROMPT"
    assert len(call_log) == 2, "Normal call after compacting should use cache, not re-call"


def test_system_prompt_cache_for_default_deferred_callable() -> None:
    """Normal prompt is cached on first call and reused."""
    call_count = 0

    def prompt_func(runtime, is_compacting: bool) -> str:
        nonlocal call_count
        call_count += 1
        return f"prompt-{call_count}"

    agent = _FakeAgent(prompt_func)

    r1 = agent.get_system_prompt(is_compacting=False)
    r2 = agent.get_system_prompt(is_compacting=False)
    assert r1 == r2 == "prompt-1"
    assert call_count == 1


def test_system_prompt_compacting_does_not_cache() -> None:
    """Each compacting call must re-evaluate (not cache)."""
    call_count = 0

    def prompt_func(runtime, is_compacting: bool) -> str:
        nonlocal call_count
        call_count += 1
        return f"compact-{call_count}"

    agent = _FakeAgent(prompt_func)

    r1 = agent.get_system_prompt(is_compacting=True)
    r2 = agent.get_system_prompt(is_compacting=True)
    assert r1 == "compact-1"
    assert r2 == "compact-2"
    assert call_count == 2


# ──────────────────────────────────────────────────────────────────────────────
# Issue 6: Context.history defensive copy
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_returns_tuple(tmp_path: Path) -> None:
    """Context.history must return an immutable tuple to prevent accidental mutation."""
    path = tmp_path / "context.jsonl"
    ctx = Context(file_backend=path)
    await ctx.append_message(_make_msg("user", "Hello"))

    hist = ctx.history
    assert isinstance(hist, tuple), f"Expected tuple, got {type(hist)}"
    assert len(hist) == 1


@pytest.mark.asyncio
async def test_history_tuple_mutation_raises(tmp_path: Path) -> None:
    """Attempting to mutate the history tuple must raise TypeError."""
    path = tmp_path / "context.jsonl"
    ctx = Context(file_backend=path)
    await ctx.append_message(_make_msg("user", "Hello"))

    hist = ctx.history
    with pytest.raises(TypeError):
        hist[0] = _make_msg("assistant", "mutation")  # type: ignore[index]


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Incremental normalization — KimiSoul._incremental_normalize
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_incremental_normalize_empty(tmp_path: Path) -> None:
    """Normalizing empty history returns empty list and resets state."""
    from kimi_cli.soul.kimisoul import KimiSoul

    # Create a minimal soul bypassing __init__
    soul = object.__new__(KimiSoul)
    soul._normalized_prefix = []
    soul._normalized_source_length = 0

    result = soul._incremental_normalize([])
    assert result == []
    assert soul._normalized_source_length == 0
    assert soul._normalized_prefix == []


@pytest.mark.asyncio
async def test_incremental_normalize_append_only(tmp_path: Path) -> None:
    """Appending messages uses incremental path, not full re-normalization."""
    from kimi_cli.soul.kimisoul import KimiSoul

    soul = object.__new__(KimiSoul)
    soul._normalized_prefix = []
    soul._normalized_source_length = 0

    # Step 1: initial normalization
    msgs = [_make_msg("user", "Q1")]
    r1 = soul._incremental_normalize(msgs)
    assert len(r1) == 1
    assert soul._normalized_source_length == 1

    # Step 2: append assistant message (no merge expected)
    msgs2 = [_make_msg("user", "Q1"), _make_msg("assistant", "A1")]
    r2 = soul._incremental_normalize(msgs2)
    assert len(r2) == 2
    assert soul._normalized_source_length == 2

    # Step 3: append tool + second assistant (no merge expected)
    msgs3 = [
        _make_msg("user", "Q1"),
        _make_msg("assistant", "A1"),
        _make_msg("tool", "T1"),
        _make_msg("assistant", "A2"),
    ]
    r3 = soul._incremental_normalize(msgs3)
    assert len(r3) == 4
    assert soul._normalized_source_length == 4


@pytest.mark.asyncio
async def test_incremental_normalize_merge_adjacent_users(tmp_path: Path) -> None:
    """Adjacent user messages should merge correctly across the boundary."""
    from kimi_cli.soul.kimisoul import KimiSoul

    soul = object.__new__(KimiSoul)
    soul._normalized_prefix = []
    soul._normalized_source_length = 0

    # Step 1: single user message
    msgs = [_make_msg("user", "Part1")]
    r1 = soul._incremental_normalize(msgs)
    assert len(r1) == 1
    assert soul._normalized_source_length == 1

    # Step 2: append another user message — should merge
    msgs2 = [_make_msg("user", "Part1"), _make_msg("user", "Part2")]
    r2 = soul._incremental_normalize(msgs2)
    assert len(r2) == 1, "Adjacent user messages should merge into one"
    assert "Part1" in str(r2[0].content)
    assert "Part2" in str(r2[0].content)
    assert soul._normalized_source_length == 2


@pytest.mark.asyncio
async def test_incremental_normalize_shrink_triggers_full_reset(tmp_path: Path) -> None:
    """When history shrinks (compaction/cleanup), a full normalization is done."""
    from kimi_cli.soul.kimisoul import KimiSoul

    soul = object.__new__(KimiSoul)
    soul._normalized_prefix = []
    soul._normalized_source_length = 0

    # Build up 3 messages
    msgs = [
        _make_msg("user", "Q1"),
        _make_msg("assistant", "A1"),
        _make_msg("tool", "T1"),
    ]
    r = soul._incremental_normalize(msgs)
    assert soul._normalized_source_length == 3

    # Now "shrink" to 1 message (simulating cleanup)
    shrunk = [_make_msg("user", "Q1")]
    r_shrunk = soul._incremental_normalize(shrunk)
    assert len(r_shrunk) == 1
    assert soul._normalized_source_length == 1


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: normalize_history correctness with ephemeral messages
# ──────────────────────────────────────────────────────────────────────────────


def test_normalize_history_merges_adjacent_users() -> None:
    """Basic merge: two adjacent user messages become one."""
    msgs = [_make_msg("user", "A"), _make_msg("user", "B")]
    result = normalize_history(msgs)
    assert len(result) == 1
    assert result[0].role == "user"
    assert "A" in str(result[0].content)
    assert "B" in str(result[0].content)


def test_normalize_history_does_not_merge_notifications() -> None:
    """Notification messages are not merged with adjacent user messages."""
    msgs = [
        _make_msg("user", "Real question"),
        _make_notification_msg("n1"),
    ]
    result = normalize_history(msgs)
    # Notifications are kept separate, so 2 messages expected
    assert len(result) == 2


def test_normalize_history_does_not_merge_assistant() -> None:
    """Assistant messages are never merged."""
    msgs = [
        _make_msg("assistant", "A1"),
        _make_msg("assistant", "A2"),
    ]
    result = normalize_history(msgs)
    assert len(result) == 2


def test_normalize_history_preserves_system_reminder_unmerged() -> None:
    """System-reminder user messages are merged with adjacent user messages
    (current behavior — they are regular user messages from the serializer's
    perspective)."""
    msgs = [
        _make_msg("user", "Real question"),
        _make_system_reminder_msg("Plan mode is active."),
    ]
    result = normalize_history(msgs)
    # System reminders are user messages and merge with adjacent user messages
    assert len(result) == 1
    assert "Real question" in str(result[0].content)
    assert "Plan mode is active" in str(result[0].content)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Prefix stability detection (indirect test via warning log)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_incremental_normalize_detects_prefix_mismatch() -> None:
    """When prefix changes unexpectedly, the fingerprint check detects it.

    We verify this by checking that the fingerprint comparison logic
    correctly identifies a changed source prefix.  The actual warning is
    emitted via loguru (not captured by pytest caplog), so we test the
    detection logic directly.
    """
    from kimi_cli.soul.kimisoul import KimiSoul

    soul = object.__new__(KimiSoul)
    soul._normalized_prefix = []
    soul._normalized_source_length = 0
    soul._normalized_prefix_fingerprint = None

    # Build up a cached prefix of 2 messages
    msgs = [
        _make_msg("user", "Q1"),
        _make_msg("assistant", "A1"),
    ]
    soul._incremental_normalize(msgs)
    stored_fp = soul._normalized_prefix_fingerprint
    assert stored_fp is not None

    # Simulate external mutation: the first source message changed
    # but _normalized_source_length still thinks the prefix is cached.
    msgs2 = [
        _make_msg("user", "DIFFERENT Q1"),  # changed!
        _make_msg("assistant", "A1"),       # same
        _make_msg("user", "Q2"),             # new
    ]

    # Compute fingerprint of the old prefix portion of the new history
    new_prefix_fp = soul._compute_prefix_fingerprint(msgs2[:2])

    # The fingerprints should differ because the content changed
    assert new_prefix_fp != stored_fp, (
        "Fingerprint should detect the changed source message. "
        "If they match, the prefix stability check is broken."
    )


# ──────────────────────────────────────────────────────────────────────────────
# is_system_reminder_message detection
# ──────────────────────────────────────────────────────────────────────────────


def test_is_system_reminder_message_detects_reminder() -> None:
    """is_system_reminder_message returns True for system-reminder messages."""
    msg = _make_system_reminder_msg("Plan mode is active.")
    assert is_system_reminder_message(msg) is True


def test_is_system_reminder_message_rejects_regular_user() -> None:
    """is_system_reminder_message returns False for regular user messages."""
    msg = _make_msg("user", "Hello world")
    assert is_system_reminder_message(msg) is False


def test_is_system_reminder_message_rejects_assistant() -> None:
    """is_system_reminder_message returns False for assistant messages."""
    msg = _make_msg("assistant", "I'll help.")
    assert is_system_reminder_message(msg) is False


def test_is_system_reminder_message_rejects_multi_part() -> None:
    """is_system_reminder_message returns False for multi-part messages."""
    msg = Message(
        role="user",
        content=[TextPart(text="<system-reminder>test</system-reminder>"), TextPart(text="extra")],
    )
    assert is_system_reminder_message(msg) is False
