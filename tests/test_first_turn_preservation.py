"""Tests for sliding-window + first-turn preservation (Phase 6)."""

from __future__ import annotations

import pytest
from kosong.message import Message

from kimi_cli.soul.compaction import SimpleCompaction
from kimi_cli.wire.types import TextPart


class TestFirstTurnPreservation:
    """Test that the first message is always preserved."""

    def test_first_message_preserved_when_not_in_tail(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Original request: build a web app")]),
            Message(role="assistant", content=[TextPart(text="Okay, let me start.")]),
            Message(role="user", content=[TextPart(text="Use React")]),
            Message(role="assistant", content=[TextPart(text="Sure.")]),
            Message(role="user", content=[TextPart(text="Add routing")]),
            Message(role="assistant", content=[TextPart(text="Done.")]),
        ]
        compactor = SimpleCompaction(max_preserved_messages=2)
        result = compactor.prepare(msgs)
        # First message should be in to_preserve
        assert result.to_preserve[0] == msgs[0]
        # It should NOT be in to_compact
        if result.compact_message is not None:
            texts = [p.text for p in result.compact_message.content if isinstance(p, TextPart)]
            assert "Original request" not in " ".join(texts)

    def test_no_duplicate_messages(self):
        msgs = [
            Message(role="user", content=[TextPart(text="First")]),
            Message(role="assistant", content=[TextPart(text="Second")]),
            Message(role="user", content=[TextPart(text="Third")]),
        ]
        compactor = SimpleCompaction(max_preserved_messages=3)
        result = compactor.prepare(msgs)
        # When max_preserved >= total user/assistant messages, all are preserved
        assert result.compact_message is None
        # No duplicates
        ids = [id(m) for m in result.to_preserve]
        assert len(ids) == len(set(ids))

    def test_first_message_already_in_tail_no_duplicate(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Only one")]),
            Message(role="assistant", content=[TextPart(text="Reply")]),
        ]
        compactor = SimpleCompaction(max_preserved_messages=2)
        result = compactor.prepare(msgs)
        # First message is already in to_preserve, should not be duplicated
        assert result.compact_message is None
        assert len(result.to_preserve) == 2

    def test_system_first_message_preserved(self):
        msgs = [
            Message(role="system", content=[TextPart(text="System prompt")]),
            Message(role="user", content=[TextPart(text="User asks something")]),
            Message(role="assistant", content=[TextPart(text="Assistant replies")]),
            Message(role="user", content=[TextPart(text="Follow up")]),
        ]
        compactor = SimpleCompaction(max_preserved_messages=1)
        result = compactor.prepare(msgs)
        # System message is the first message
        assert result.to_preserve[0] == msgs[0]
