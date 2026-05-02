"""Short-term memory: detailed current session records."""

import heapq
import time
from typing import List

from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.embedding import EmbeddingProvider


class ShortTermMemory:
    """Short-term memory: detailed current session records."""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600) -> None:
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.buffer: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Add memory to short-term buffer."""
        entry.memory_type = MemoryType.EPISODIC
        self.buffer.append(entry)

        # Capacity management: evict least important
        if len(self.buffer) > self.max_size:
            self._evict_least_valuable()

    def _evict_least_valuable(self) -> None:
        """Eviction policy: remove entry with lowest effective importance."""
        if not self.buffer:
            return

        # O(n) scan instead of O(n log n) sort
        idx, _ = min(
            enumerate(self.buffer),
            key=lambda x: x[1].get_effective_importance(),
        )
        # O(1) removal by swapping with the last element
        self.buffer[idx] = self.buffer[-1]
        self.buffer.pop()

    def search(
        self, query: str, embedding_provider: EmbeddingProvider, top_k: int = 5
    ) -> List[MemoryEntry]:
        """Semantic search in short-term memory."""
        if not self.buffer:
            return []

        query_vec = embedding_provider.embed(query)

        # Lazily compute missing embeddings
        for entry in self.buffer:
            if entry.embedding is None:
                entry.embedding = embedding_provider.embed(entry.content)

        # O(n log k) partial selection instead of O(n log n) full sort
        scored = (
            (
                embedding_provider.similarity(query_vec, entry.embedding)
                * entry.get_effective_importance(),
                entry,
            )
            for entry in self.buffer
        )
        results = [
            entry for _, entry in heapq.nlargest(top_k, scored, key=lambda x: x[0])
        ]

        # Mark access
        for entry in results:
            entry.touch()

        return results

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get recent n entries."""
        # O(n log k) partial selection instead of O(n log n) full sort
        return heapq.nlargest(n, self.buffer, key=lambda x: x.timestamp)

    def clear_expired(self) -> None:
        """Clean expired memories."""
        cutoff = time.time() - self.ttl
        self.buffer = [e for e in self.buffer if e.timestamp > cutoff]
