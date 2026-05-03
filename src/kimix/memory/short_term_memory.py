"""Short-term memory: detailed current session records with temporal validity."""

from __future__ import annotations

import heapq
import time
from typing import List

import numpy as np

from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.embedding import EmbeddingProvider


class ShortTermMemory:
    """Short-term memory: detailed current session records with temporal validity."""

    __slots__ = ("max_size", "ttl", "buffer")

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600) -> None:
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.buffer: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Add memory to short-term buffer."""
        entry.memory_type = MemoryType.EPISODIC
        self.buffer.append(entry)
        if len(self.buffer) > self.max_size:
            self._evict_least_valuable()

    def _evict_least_valuable(self) -> None:
        """Eviction policy: remove entry with lowest effective importance."""
        if not self.buffer:
            return
        now = time.time()
        buf = self.buffer
        min_idx = 0
        min_val = buf[0].get_effective_importance(now)
        for i in range(1, len(buf)):
            val = buf[i].get_effective_importance(now)
            if val < min_val:
                min_val = val
                min_idx = i
        buf[min_idx] = buf[-1]
        buf.pop()

    def _active_buffer(self, now: float | None = None) -> List[MemoryEntry]:
        """Return only non-expired entries."""
        if now is None:
            now = time.time()
        cutoff = now - self.ttl
        return [
            e
            for e in self.buffer
            if e.timestamp > cutoff and (e.expires_at is None or e.expires_at > now)
        ]

    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        top_k: int = 5,
        query_vec: np.ndarray | None = None,
    ) -> List[MemoryEntry]:
        """Semantic search in short-term memory (skips expired)."""
        now = time.time()
        active = self._active_buffer(now)
        if not active:
            return []

        if query_vec is None:
            query_vec = embedding_provider.embed(query)

        # Batch-compute missing embeddings, touching only entries that need it
        missing = [
            (i, entry.content)
            for i, entry in enumerate(active)
            if entry.embedding is None
        ]
        if missing:
            indices, texts = zip(*missing)
            embeddings = embedding_provider.embed_batch(texts)
            for i, emb in zip(indices, embeddings):
                active[i].embedding = emb

        scored = [
            (
                embedding_provider.similarity(query_vec, entry.embedding)
                * entry.get_effective_importance(now),
                entry,
            )
            for entry in active
        ]
        results = [
            entry for _, entry in heapq.nlargest(top_k, scored, key=lambda x: x[0])
        ]

        for entry in results:
            entry.touch(now)

        return results

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get recent n entries (skips expired)."""
        now = time.time()
        cutoff = now - self.ttl

        valid = [
            e
            for e in self.buffer
            if e.timestamp > cutoff and (e.expires_at is None or e.expires_at > now)
        ]
        return heapq.nlargest(n, valid, key=lambda e: e.timestamp)

    def clear_expired(self) -> None:
        """Clean expired memories (both TTL and explicit expiry)."""
        now = time.time()
        cutoff = now - self.ttl
        self.buffer[:] = [
            e
            for e in self.buffer
            if e.timestamp > cutoff and (e.expires_at is None or e.expires_at > now)
        ]
