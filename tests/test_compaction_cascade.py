"""Tests for compaction cascade mitigation (Phase 4)."""

from __future__ import annotations

import pytest
from kosong.message import Message

from kimi_cli.soul.compaction import SimpleCompaction, _detect_cascade_depth
from kimi_cli.wire.types import TextPart


class TestDetectCascadeDepth:
    """Test cascade depth detection."""

    def test_no_compaction_messages(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Hello")]),
            Message(role="assistant", content=[TextPart(text="Hi")]),
        ]
        assert _detect_cascade_depth(msgs) == 0

    def test_one_compaction_message(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted. Summary.")]),
            Message(role="user", content=[TextPart(text="New question")]),
        ]
        assert _detect_cascade_depth(msgs) == 1

    def test_three_compaction_messages(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted. A")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. B")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. C")]),
        ]
        assert _detect_cascade_depth(msgs) == 3


class TestCascadePromptSelection:
    """Test that the cascade prompt is selected at depth >= 3."""

    def test_normal_prompt_below_depth_3(self):
        msgs = [
            Message(role="user", content=[TextPart(text="First")]),
            Message(role="assistant", content=[TextPart(text="Second")]),
            Message(role="user", content=[TextPart(text="Third")]),
            Message(role="assistant", content=[TextPart(text="Fourth")]),
        ]
        compactor = SimpleCompaction(max_preserved_messages=1)
        result = compactor.prepare(msgs)
        assert result.cascade_depth == 0
        # The prompt should be the normal COMPACT prompt
        last_part = result.compact_message.content[-1]
        assert "Compact the above" in last_part.text

    def test_cascade_prompt_at_depth_3(self):
        import kimi_cli.prompts as prompts
        # Need 4 compaction messages because Phase 6 preserves the first one
        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted. A")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. B")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. C")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted. D")]),
            Message(role="assistant", content=[TextPart(text="Latest")]),
            Message(role="user", content=[TextPart(text="Newest")]),
        ]
        compactor = SimpleCompaction(max_preserved_messages=1)
        result = compactor.prepare(msgs)
        assert result.cascade_depth >= 3
        last_part = result.compact_message.content[-1]
        assert prompts.COMPACT_CASCADE in last_part.text

    def test_cascade_prompt_deeper(self):
        import kimi_cli.prompts as prompts
        msgs = [
            Message(role="user", content=[TextPart(text="Previous context has been compacted.")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted.")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted.")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted.")]),
            Message(role="user", content=[TextPart(text="Previous context has been compacted.")]),
            Message(role="assistant", content=[TextPart(text="Latest")]),
            Message(role="user", content=[TextPart(text="Newest")]),
        ]
        compactor = SimpleCompaction(max_preserved_messages=1)
        result = compactor.prepare(msgs)
        assert result.cascade_depth >= 4
        last_part = result.compact_message.content[-1]
        assert prompts.COMPACT_CASCADE in last_part.text
