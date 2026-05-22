"""Tests for model-aware token estimation (Phase 1)."""

from __future__ import annotations

import pytest
from kosong.message import Message

from kimi_cli.utils.tokens import count_tokens, count_message_tokens, _is_cjk_text
from kimi_cli.wire.types import TextPart, ThinkPart


class TestCountTokens:
    """Test the count_tokens function with various inputs."""

    def test_empty_text(self):
        assert count_tokens("") == 0

    def test_english_text(self):
        # ~4 chars per token for ASCII
        text = "a" * 100
        result = count_tokens(text)
        assert result == 25  # 100 // 4

    def test_cjk_text(self):
        # CJK text should use ~3 chars per token
        text = "中" * 30
        result = count_tokens(text)
        assert result == 10  # 30 // 3

    def test_mixed_text(self):
        # Mixed text with enough CJK triggers CJK heuristic (~3 chars per token)
        text = "a" * 35 + "中" * 35  # 70 chars total, 50% CJK
        result = count_tokens(text)
        # 50% CJK > 15% threshold → 70 // 3 = 23
        assert result == 23

    def test_code_text(self):
        # Code is mostly ASCII but not purely, so mixed heuristic
        text = "def foo():\n    return 42\n"
        result = count_tokens(text)
        # 26 chars, mostly ASCII → 26 // 4 = 6.5 → max(1, 6)
        assert result >= 6

    def test_model_aware_when_no_tiktoken(self):
        # When tiktoken is not installed, should fallback gracefully
        result = count_tokens("hello world", model="gpt-4")
        assert result > 0


class TestCountMessageTokens:
    """Test token counting for sequences of messages."""

    def test_single_message(self):
        msg = Message(role="user", content=[TextPart(text="a" * 40)])
        assert count_message_tokens([msg]) == 10  # 40 // 4

    def test_multiple_messages(self):
        msgs = [
            Message(role="user", content=[TextPart(text="a" * 40)]),
            Message(role="assistant", content=[TextPart(text="b" * 80)]),
        ]
        assert count_message_tokens(msgs) == 30  # 10 + 20

    def test_ignores_non_text_parts(self):
        msg = Message(
            role="user",
            content=[
                TextPart(text="a" * 40),
                ThinkPart(think="lots of reasoning " * 50),
            ],
        )
        assert count_message_tokens([msg]) == 10

    def test_cjk_messages(self):
        msg = Message(role="user", content=[TextPart(text="中文测试" * 10)])
        result = count_message_tokens([msg])
        # 40 CJK chars → 40 // 3 = 13.33 → max(1, 13)
        assert result >= 13


class TestIsCjkText:
    """Test CJK detection heuristic."""

    def test_pure_english(self):
        assert _is_cjk_text("hello world") is False

    def test_pure_cjk(self):
        assert _is_cjk_text("中文测试") is True

    def test_mixed_below_threshold(self):
        # 1 CJK out of 6 chars = ~16.7% > 15% threshold, so this IS detected as CJK
        assert _is_cjk_text("hello中") is True

    def test_mixed_above_threshold(self):
        text = "a" * 10 + "中" * 10  # 50% CJK
        assert _is_cjk_text(text) is True

    def test_mixed_well_below_threshold(self):
        text = "a" * 100 + "中" * 1  # ~1% CJK
        assert _is_cjk_text(text) is False


class TestBackwardsCompatibility:
    """Ensure the new token estimator stays within 5% of old behaviour on English."""

    def test_english_estimate_within_five_percent(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        old_estimate = len(text) // 4
        new_estimate = count_tokens(text)
        diff = abs(new_estimate - old_estimate) / old_estimate
        assert diff <= 0.05
