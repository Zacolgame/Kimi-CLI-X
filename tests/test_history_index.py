"""Tests for BM25 history index (Phase 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from kosong.message import Message

from kimi_cli.soul.history_index import HistoryIndex, _MAX_TURNS
from kimi_cli.wire.types import TextPart


class TestHistoryIndexBasics:
    """Test basic indexing and search."""

    def test_empty_index_search(self):
        idx = HistoryIndex()
        assert idx.search("hello", top_k=3) == []

    def test_index_and_search(self):
        idx = HistoryIndex()
        # Use more diverse documents so BM25 stop-word filtering doesn't remove everything
        msgs = [
            Message(role="user", content=[TextPart(text="What is the capital of France?")]),
            Message(role="assistant", content=[TextPart(text="The capital of France is Paris.")]),
            Message(role="user", content=[TextPart(text="Tell me about Germany.")]),
            Message(role="assistant", content=[TextPart(text="Germany is in Europe.")]),
            Message(role="user", content=[TextPart(text="What about Italy?")]),
            Message(role="assistant", content=[TextPart(text="Italy has Rome.")]),
        ]
        idx.index_messages(msgs)
        results = idx.search("France", top_k=2)
        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("France" in t for t in texts)

    def test_search_ignores_system_messages(self):
        idx = HistoryIndex()
        msgs = [
            Message(role="system", content=[TextPart(text="You are a helpful assistant.")]),
            Message(role="user", content=[TextPart(text="Hello")]),
        ]
        idx.index_messages(msgs)
        results = idx.search("assistant", top_k=3)
        assert len(results) == 0  # system role skipped

    def test_search_returns_verbatim(self):
        idx = HistoryIndex()
        text = "The exact original text must be preserved."
        msgs = [
            Message(role="user", content=[TextPart(text=text)]),
            Message(role="assistant", content=[TextPart(text="Something completely different.")]),
        ]
        idx.index_messages(msgs)
        results = idx.search("original text", top_k=1)
        assert len(results) >= 1
        assert results[0]["text"] == text


class TestHistoryIndexCompaction:
    """Test marking turns as compacted."""

    def test_mark_compacted(self):
        idx = HistoryIndex()
        msgs = [Message(role="user", content=[TextPart(text="Turn 1")])]
        idx.index_messages(msgs)
        assert not idx._turns[0]["is_compacted"]
        idx.mark_compacted()
        assert idx._turns[0]["is_compacted"]


class TestHistoryIndexPersistence:
    """Test save/load."""

    def test_save_and_load(self, tmp_path: Path):
        persist_path = tmp_path / "history.json"
        idx = HistoryIndex(persist_path=persist_path)
        msgs = [
            Message(role="user", content=[TextPart(text="Persist me")]),
            Message(role="assistant", content=[TextPart(text="Okay")]),
        ]
        idx.index_messages(msgs)
        idx.save()

        idx2 = HistoryIndex(persist_path=persist_path)
        assert idx2.load()
        results = idx2.search("Persist", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Persist me"

    def test_load_nonexistent(self, tmp_path: Path):
        idx = HistoryIndex(persist_path=tmp_path / "nope.json")
        assert not idx.load()

    def test_clear(self, tmp_path: Path):
        persist_path = tmp_path / "history.json"
        idx = HistoryIndex(persist_path=persist_path)
        idx.index_messages([Message(role="user", content=[TextPart(text="Hi")])])
        idx.save()
        idx.clear()
        assert idx._turns == []
        assert not persist_path.exists()


class TestHistoryIndexBounds:
    """Test the max turns bound."""

    def test_max_turns_bound(self):
        idx = HistoryIndex()
        msgs = []
        for i in range(_MAX_TURNS + 10):
            msgs.append(Message(role="user", content=[TextPart(text=f"Turn {i}")]))
        idx.index_messages(msgs)
        assert len(idx._turns) <= _MAX_TURNS
