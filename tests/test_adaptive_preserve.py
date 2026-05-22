"""Tests for task-adaptive preserve depth (Phase 2)."""

from __future__ import annotations

import pytest
from kosong.message import Message

from kimi_cli.soul.compaction import SimpleCompaction, adaptive_preserve_depth
from kimi_cli.wire.types import TextPart, ThinkPart


class TestAdaptivePreserveDepth:
    """Test the adaptive preserve depth heuristic."""

    def test_empty_messages(self):
        assert adaptive_preserve_depth([]) == 1

    def test_no_signals(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Hello")]),
            Message(role="assistant", content=[TextPart(text="Hi there")]),
        ]
        assert adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=5) == 1

    def test_error_signal(self):
        msgs = [
            Message(role="user", content=[TextPart(text="It failed with an error")]),
        ]
        assert adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=5) == 2

    def test_exception_signal(self):
        msgs = [
            Message(role="assistant", content=[TextPart(text="An exception occurred")]),
        ]
        assert adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=5) == 2

    def test_failed_signal(self):
        msgs = [
            Message(role="user", content=[TextPart(text="The test failed")]),
        ]
        assert adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=5) == 2

    def test_think_signal(self):
        msgs = [
            Message(
                role="assistant",
                content=[TextPart(text="Let me think"), ThinkPart(think="...")],
            ),
        ]
        assert adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=5) == 2

    def test_file_edits_signal(self):
        msgs = [
            Message(
                role="assistant",
                content=[TextPart(text="file: a.py file: b.py file: c.py")],
            ),
        ]
        assert adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=5) == 2

    def test_multiple_signals(self):
        msgs = [
            Message(
                role="assistant",
                content=[
                    TextPart(text="error: file: a.py file: b.py file: c.py"),
                    ThinkPart(think="..."),
                ],
            ),
        ]
        assert adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=5) == 4

    def test_max_cap(self):
        msgs = [
            Message(role="user", content=[TextPart(text="error exception failed")]),
        ]
        # error +1, but no think or file refs
        # Actually "error" is +1, and "failed" is also in text but we only check any()
        result = adaptive_preserve_depth(msgs, min_preserved=1, max_preserved=2)
        assert result <= 2

    def test_min_floor(self):
        msgs = []
        assert adaptive_preserve_depth(msgs, min_preserved=2, max_preserved=5) == 2


class TestSimpleCompactionWithPreserveDepth:
    """Test SimpleCompaction with callable preserve_depth."""

    def test_callable_preserve_depth(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Old")]),
            Message(role="assistant", content=[TextPart(text="Old reply")]),
            Message(role="user", content=[TextPart(text="New")]),
            Message(role="assistant", content=[TextPart(text="New reply")]),
        ]

        def _depth(msgs):
            return 3

        compactor = SimpleCompaction(max_preserved_messages=1, preserve_depth=_depth)
        result = compactor.prepare(msgs)
        # Phase 6: first message is also preserved
        assert len(result.to_preserve) == 4

    def test_int_preserve_depth(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Old")]),
            Message(role="assistant", content=[TextPart(text="Old reply")]),
            Message(role="user", content=[TextPart(text="New")]),
        ]

        compactor = SimpleCompaction(max_preserved_messages=1, preserve_depth=2)
        result = compactor.prepare(msgs)
        # Phase 6: first message is also preserved
        assert len(result.to_preserve) == 3

    def test_none_preserve_depth_uses_default(self):
        msgs = [
            Message(role="user", content=[TextPart(text="Old")]),
            Message(role="assistant", content=[TextPart(text="Old reply")]),
            Message(role="user", content=[TextPart(text="New")]),
        ]

        compactor = SimpleCompaction(max_preserved_messages=1, preserve_depth=None)
        result = compactor.prepare(msgs)
        # Phase 6: first message is also preserved
        assert len(result.to_preserve) == 2
