"""Working memory: current conversation context, limited capacity."""

from collections import deque
from itertools import islice
from typing import List, Optional

from kimix.memory.types import MemoryEntry, MemoryType


class WorkingMemory:
    """Working memory: current conversation context, limited capacity."""

    def __init__(self, max_items: int = 10) -> None:
        self.max_items = max_items
        self.items: deque[MemoryEntry] = deque(maxlen=max_items)
        self.current_focus: Optional[str] = None

    def add(self, entry: MemoryEntry) -> None:
        """Add current context."""
        entry.memory_type = MemoryType.WORKING
        self.items.append(entry)

    def get_context(self, n: int = 5) -> List[MemoryEntry]:
        """Get recent n context items."""
        if n <= 0:
            return []
        start = max(0, len(self.items) - n)
        return list(islice(self.items, start, None))

    def clear(self) -> None:
        """Clear working memory."""
        self.items.clear()
        self.current_focus = None

    def summarize(self) -> str:
        """Generate current context summary."""
        if not self.items:
            return ""
        start = max(0, len(self.items) - 3)
        contents = (item.content for item in islice(self.items, start, None))
        return " | ".join(contents)
