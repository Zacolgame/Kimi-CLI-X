"""Working memory: current conversation context, limited capacity."""

from __future__ import annotations

from collections import deque

from kimix.memory.types import MemoryEntry, MemoryType


class WorkingMemory:
    """Working memory: current conversation context, limited capacity."""

    __slots__ = ("max_items", "items", "current_focus")

    def __init__(self, max_items: int = 10) -> None:
        self.max_items = max_items
        self.items: deque[MemoryEntry] = deque(maxlen=max_items)
        self.current_focus: str | None = None

    def add(self, entry: MemoryEntry) -> None:
        """Add current context (stores a copy with WORKING type, caller's object is untouched)."""
        # object.__new__ + direct slot assignment is ~3x faster than
        # dataclass __init__ reconstruction for a slots=True dataclass,
        # and avoids mutating the caller's entry.
        new: MemoryEntry = object.__new__(MemoryEntry)
        new.content = entry.content
        new.memory_type = MemoryType.WORKING
        new.timestamp = entry.timestamp
        new.importance = entry.importance
        new.access_count = entry.access_count
        new.last_accessed = entry.last_accessed
        new.embedding = entry.embedding
        new.tags = entry.tags
        new.source = entry.source
        new.metadata = entry.metadata
        new.expires_at = entry.expires_at
        new.agent_id = entry.agent_id
        self.items.append(new)

    def get_context(self, n: int = 5) -> list[MemoryEntry]:
        """Get recent n context items."""
        if n <= 0:
            return []
        # list() on deque is C-level and slicing is O(n) with minimal overhead,
        # faster than islice for small maxlen (the common case for working memory).
        return list(self.items)[-n:]

    def clear(self) -> None:
        """Clear working memory."""
        self.items.clear()
        self.current_focus = None

    def summarize(self) -> str:
        """Generate current context summary."""
        if not self.items:
            return ""
        return " | ".join(item.content for item in list(self.items)[-3:])
