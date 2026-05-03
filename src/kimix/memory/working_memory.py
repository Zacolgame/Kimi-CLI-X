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
        # Reconstructing is ~3x faster than copy() or dataclasses.replace()
        # and avoids mutating the caller's entry.
        self.items.append(
            MemoryEntry(
                content=entry.content,
                memory_type=MemoryType.WORKING,
                timestamp=entry.timestamp,
                importance=entry.importance,
                access_count=entry.access_count,
                last_accessed=entry.last_accessed,
                embedding=entry.embedding,
                tags=entry.tags,
                source=entry.source,
                metadata=entry.metadata,
                expires_at=entry.expires_at,
                agent_id=entry.agent_id,
            )
        )

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
