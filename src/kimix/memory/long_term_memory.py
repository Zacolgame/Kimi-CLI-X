"""Long-term memory: persistent storage with complex retrieval."""

import hashlib
import heapq
import json
from typing import Dict, List, Optional, Set

from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.embedding import EmbeddingProvider


class LongTermMemory:
    """Long-term memory: persistent storage with complex retrieval."""

    __slots__ = ("storage_path", "dim", "entries", "index", "embedding_provider", "_dirty")

    def __init__(self, storage_path: Optional[str] = None, dim: int = 384) -> None:
        self.storage_path = storage_path or "ltm.json"
        self.dim = dim
        self.entries: Dict[str, MemoryEntry] = {}  # id -> entry
        self.index: Dict[str, Set[str]] = {}       # tag -> entry_ids
        self.embedding_provider = EmbeddingProvider(dim)
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load from disk."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                entry = MemoryEntry(
                    content=item['content'],
                    memory_type=MemoryType(item['memory_type']),
                    timestamp=item['timestamp'],
                    importance=item['importance'],
                    access_count=item.get('access_count', 0),
                    last_accessed=item.get('last_accessed', item['timestamp']),
                    embedding=item.get('embedding'),
                    tags=item.get('tags', []),
                    source=item.get('source', ''),
                    metadata=item.get('metadata', {})
                )
                entry_id = self._hash(entry.content)
                self.entries[entry_id] = entry
                self._update_index(entry_id, entry)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        """Persist to disk if dirty."""
        if not self._dirty:
            return
        data = [e.to_dict() for e in self.entries.values()]
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def _hash(self, content: str) -> str:
        """Generate content hash as ID."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _update_index(self, entry_id: str, entry: MemoryEntry) -> None:
        """Update tag index."""
        for tag in entry.tags:
            self.index.setdefault(tag, set()).add(entry_id)

    def _insert_entry(self, entry_id: str, entry: MemoryEntry) -> None:
        """Insert entry without persisting (for batch operations)."""
        self.entries[entry_id] = entry
        self._update_index(entry_id, entry)
        self._dirty = True

    def store(
        self,
        content: str,
        importance: float = 5.0,
        tags: Optional[List[str]] = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        source: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> MemoryEntry:
        """Store long-term memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            source=source,
            metadata=metadata or {}
        )

        # Generate embedding
        entry.embedding = self.embedding_provider.embed(content)

        entry_id = self._hash(content)
        self._insert_entry(entry_id, entry)
        self._save()

        return entry

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        tag_filter: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """Semantic retrieval from long-term memory."""
        if not self.entries:
            return []

        query_vec = self.embedding_provider.embed(query)

        # Tag filter: intersect sets for O(1) lookup
        if tag_filter:
            filtered_ids: Optional[Set[str]] = None
            for tag in tag_filter:
                ids = self.index.get(tag)
                if ids is None:
                    return []
                if filtered_ids is None:
                    filtered_ids = set(ids)
                else:
                    filtered_ids.intersection_update(ids)
            if not filtered_ids:
                return []
            candidates = [self.entries[eid] for eid in filtered_ids if eid in self.entries]
        else:
            candidates = list(self.entries.values())

        # Importance filter
        if min_importance > 0.0:
            candidates = [e for e in candidates if e.importance >= min_importance]

        if not candidates:
            return []

        # Ensure embeddings exist
        for entry in candidates:
            if entry.embedding is None:
                entry.embedding = self.embedding_provider.embed(entry.content)

        # Similarity scoring
        def _score(entry: MemoryEntry) -> float:
            sim = self.embedding_provider.similarity(query_vec, entry.embedding)  # type: ignore[arg-type]
            return sim * entry.get_effective_importance()

        # Use heapq.nlargest for better performance when top_k << len(candidates)
        if top_k * 4 < len(candidates):
            results = heapq.nlargest(top_k, candidates, key=_score)
        else:
            candidates.sort(key=_score, reverse=True)
            results = candidates[:top_k]

        # Update access stats
        for entry in results:
            entry.touch()

        self._dirty = True
        self._save()
        return results

    def consolidate(self, short_term: "ShortTermMemory", threshold: float = 7.0) -> None:
        """Memory consolidation: migrate high-value short-term to long-term."""
        from kimix.memory.short_term_memory import ShortTermMemory
        if not isinstance(short_term, ShortTermMemory):
            raise TypeError("short_term must be a ShortTermMemory instance")

        to_migrate = [entry for entry in short_term.buffer if entry.get_effective_importance() >= threshold]
        if not to_migrate:
            return

        for entry in to_migrate:
            entry_id = self._hash(entry.content)
            # Reuse existing embedding if available
            if entry.embedding is None:
                entry.embedding = self.embedding_provider.embed(entry.content)
            self._insert_entry(entry_id, entry)

        # Batch remove from short-term: O(n) instead of O(n*m)
        migrate_ids = {id(entry) for entry in to_migrate}
        short_term.buffer = [e for e in short_term.buffer if id(e) not in migrate_ids]

        self._save()

    def forget(self, entry_id: str) -> None:
        """Active forgetting."""
        entry = self.entries.get(entry_id)
        if entry is None:
            return

        # Reduce importance instead of immediate deletion (simulates forgetting curve)
        entry.importance *= 0.5
        if entry.importance < 0.1:
            del self.entries[entry_id]
            # Clean index using set.discard for O(1) removal
            for tag in entry.tags:
                tag_set = self.index.get(tag)
                if tag_set is not None:
                    tag_set.discard(entry_id)
                    if not tag_set:
                        del self.index[tag]
        self._dirty = True
        self._save()
