"""Long-term memory: persistent storage with hybrid semantic + BM25 retrieval."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import time
from typing import Dict, List, Optional, Set

import numpy as np

from kimix.memory.types import MemoryEntry, MemoryType
from kimix.memory.embedding import EmbeddingProvider
from kimix.memory.retrieval import InvertedIndex, NgramTokenizer, Searcher


class LongTermMemory:
    """Long-term memory: persistent storage with hybrid semantic + BM25 retrieval.

    Supports two backends:
    * **dict + JSON** (default) — backward-compatible in-memory dict with JSON persistence.
    * **SQLite** — pass a :class:`kimix.memory.sqlite_backend.SQLiteBackend` instance
      for ACID, multi-agent storage.
    """

    __slots__ = (
        "storage_path", "dim", "entries", "index", "embedding_provider",
        "_dirty", "_backend", "_agent_id", "_bm25_index", "_bm25_searcher",
        "_doc_id_map", "_bm25_doc_to_entry_id", "_next_doc_id",
    )

    def __init__(
        self,
        storage_path: Optional[str] = None,
        dim: int = 384,
        backend: Optional["kimix.memory.sqlite_backend.SQLiteBackend"] = None,
        agent_id: str = "default",
    ) -> None:
        self.storage_path = storage_path or "ltm.json"
        if backend is None and (not isinstance(self.storage_path, str) or not self.storage_path):
            raise ValueError("storage_path must be a non-empty string")
        if backend is None:
            parent = os.path.dirname(self.storage_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.dim = dim
        self.entries: Dict[str, MemoryEntry] = {}  # id -> entry
        self.index: Dict[str, Set[str]] = {}       # tag -> entry_ids
        self.embedding_provider = EmbeddingProvider(dim)
        self._dirty = False
        self._backend = backend
        self._agent_id = agent_id

        # BM25 structures (lazy-built)
        self._bm25_index: Optional[InvertedIndex] = None
        self._bm25_searcher: Optional[Searcher] = None
        self._doc_id_map: Dict[str, int] = {}      # entry_id -> bm25_doc_id
        self._bm25_doc_to_entry_id: list[str] = []  # bm25_doc_id -> entry_id
        self._next_doc_id = 0

        self._load()

    # --- Internal helpers ---

    def _hash(self, content: str) -> str:
        return hashlib.blake2b(content.encode(), digest_size=8).hexdigest()

    def _update_index(self, entry_id: str, entry: MemoryEntry) -> None:
        for tag in entry.tags:
            self.index.setdefault(tag, set()).add(entry_id)

    def _invalidate_bm25(self) -> None:
        self._bm25_index = None
        self._bm25_searcher = None
        self._bm25_doc_to_entry_id = []
        self._doc_id_map = {}
        self._next_doc_id = 0

    def _insert_entry(self, entry_id: str, entry: MemoryEntry, *, invalidate_bm25: bool = True) -> None:
        if self._backend is not None:
            self._backend.store(entry, entry_id, dim=self.dim)
        else:
            self.entries[entry_id] = entry
            self._update_index(entry_id, entry)
            self._dirty = True
        if invalidate_bm25:
            self._invalidate_bm25()

    def _build_bm25(self) -> Searcher:
        """Build or rebuild the BM25 inverted index from current entries."""
        idx = InvertedIndex()
        tokenizer = NgramTokenizer()
        self._doc_id_map = {}
        self._bm25_doc_to_entry_id = []
        self._next_doc_id = 0
        if self._backend is not None:
            # Fast path: avoid deserialising embeddings and full MemoryEntry objects.
            now = time.time()
            for eid, content, expires_at in self._backend.iter_rows(
                agent_id=self._agent_id, exclude_expired=False
            ):
                if expires_at is not None and expires_at <= now:
                    continue
                doc_id = self._next_doc_id
                self._next_doc_id += 1
                self._doc_id_map[eid] = doc_id
                self._bm25_doc_to_entry_id.append(eid)
                tokens = tokenizer.tokenize(content)
                idx.add_document(doc_id, tokens)
        else:
            for eid, entry in self.entries.items():
                if entry.is_expired():
                    continue
                doc_id = self._next_doc_id
                self._next_doc_id += 1
                self._doc_id_map[eid] = doc_id
                self._bm25_doc_to_entry_id.append(eid)
                tokens = tokenizer.tokenize(entry.content)
                idx.add_document(doc_id, tokens)
        idx.finalize()
        self._bm25_index = idx
        self._bm25_searcher = Searcher(idx, tokenizer=tokenizer)
        return self._bm25_searcher

    def _ensure_bm25(self) -> Searcher:
        if self._bm25_searcher is None:
            return self._build_bm25()
        return self._bm25_searcher

    def _iter_entries(self):
        """Iterate over (entry_id, MemoryEntry) regardless of backend."""
        if self._backend is not None:
            for eid, entry in self._backend.list_all(
                agent_id=self._agent_id,
                exclude_expired=False,
                dim=self.dim,
            ):
                yield eid, entry
        else:
            for eid, entry in self.entries.items():
                yield eid, entry

    def _get_entry(self, entry_id: str) -> MemoryEntry | None:
        if self._backend is not None:
            entry = self._backend.get(entry_id, dim=self.dim)
            if entry is not None and entry.agent_id != self._agent_id:
                return None
            return entry
        return self.entries.get(entry_id)

    # --- Persistence (JSON fallback) ---

    def _load(self) -> None:
        if self._backend is not None:
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                entry = MemoryEntry.from_dict(item)
                if entry.agent_id != self._agent_id:
                    continue
                entry_id = self._hash(entry.content)
                self.entries[entry_id] = entry
                self._update_index(entry_id, entry)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        if self._backend is not None:
            return
        if not self._dirty:
            return
        data = [e.to_dict() for e in self.entries.values()]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        self._dirty = False

    # --- Public API ---

    def store(
        self,
        content: str,
        importance: float = 5.0,
        tags: Optional[List[str]] = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        source: str = "",
        metadata: Optional[Dict[str, object]] = None,
        expires_at: Optional[float] = None,
    ) -> MemoryEntry:
        """Store long-term memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            source=source,
            metadata=metadata or {},
            expires_at=expires_at,
            agent_id=self._agent_id,
        )
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
        use_hybrid: bool = True,
        bm25_weight: float = 0.3,
        query_vec: np.ndarray | None = None,
    ) -> List[MemoryEntry]:
        """Hybrid semantic + BM25 retrieval from long-term memory.

        Final score = ``(1 - bm25_weight) * semantic_sim + bm25_weight * bm25_score``,
        where BM25 scores are min-max normalised per-query.
        """
        if self._backend is None and not self.entries:
            return []

        if query_vec is None:
            query_vec = self.embedding_provider.embed(query)
        now = time.time()

        # Collect candidates (filter expiry and importance during iteration)
        candidates: list[MemoryEntry] = []
        candidate_ids: list[str] = []
        if tag_filter:
            if self._backend is not None:
                raw = self._backend.search_by_tag(
                    tag_filter, agent_id=self._agent_id, dim=self.dim
                )
                for eid, entry in raw:
                    if entry.expires_at is not None and entry.expires_at <= now:
                        continue
                    if entry.importance < min_importance:
                        continue
                    candidates.append(entry)
                    candidate_ids.append(eid)
            else:
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
                for eid in filtered_ids:
                    entry = self.entries.get(eid)
                    if entry is not None and (entry.expires_at is None or entry.expires_at > now):
                        if entry.importance >= min_importance:
                            candidates.append(entry)
                            candidate_ids.append(eid)
        else:
            for eid, entry in self._iter_entries():
                if entry.expires_at is None or entry.expires_at > now:
                    if entry.importance >= min_importance:
                        candidates.append(entry)
                        candidate_ids.append(eid)

        if not candidates or top_k <= 0:
            return []

        # Batch-embed missing vectors
        missing = [entry for entry in candidates if entry.embedding is None]
        if missing:
            texts = [e.content for e in missing]
            vecs = self.embedding_provider.embed_batch(texts)
            for entry, vec in zip(missing, vecs):
                entry.embedding = vec

        # Vectorised semantic similarity
        embeddings = np.array([entry.embedding for entry in candidates], dtype=np.float32)
        query_arr = np.asarray(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(query_arr)
        norms = np.linalg.norm(embeddings, axis=1)
        if q_norm == 0:
            semantic_arr = np.zeros(len(candidates), dtype=np.float64)
        else:
            dots = embeddings @ query_arr
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = np.where(norms == 0, 0.0, dots / (norms * q_norm))
            eff = np.array([entry.get_effective_importance(now) for entry in candidates], dtype=np.float64)
            semantic_arr = sims.astype(np.float64) * eff

        # BM25 scores (single-pass min-max normalisation)
        bm25_arr = np.zeros(len(candidates), dtype=np.float64)
        if use_hybrid:
            searcher = self._ensure_bm25()
            bm25_results = searcher.search(query, top_k=len(candidates))
            if bm25_results:
                max_bm25 = bm25_results[0][1]
                min_bm25 = max_bm25
                for _, score in bm25_results[1:]:
                    if score > max_bm25:
                        max_bm25 = score
                    elif score < min_bm25:
                        min_bm25 = score
                bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
                eid_to_idx = {eid: i for i, eid in enumerate(candidate_ids)}
                doc_to_eid = self._bm25_doc_to_entry_id
                for doc_id, score in bm25_results:
                    idx = eid_to_idx.get(doc_to_eid[doc_id])
                    if idx is not None:
                        bm25_arr[idx] = (score - min_bm25) / bm25_range

        # Hybrid fusion (vectorised)
        final = (1.0 - bm25_weight) * semantic_arr + bm25_weight * bm25_arr
        n = len(final)
        if top_k * 4 < n:
            top_indices = np.argpartition(final, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(final[top_indices])[::-1]]
        else:
            top_indices = np.argsort(final)[::-1][:top_k]
        top_indices = top_indices.tolist()
        results = [candidates[i] for i in top_indices]

        if self._backend is not None:
            eids = [candidate_ids[i] for i in top_indices]
            self._backend.update_access_many(eids)
        for entry in results:
            entry.touch(now)

        # Note: access-count bumps are *not* persisted to JSON on every retrieve
        # to avoid O(N) JSON rewrites on the read path.  They survive until the
        # next write operation (store/forget/consolidate) or process exit.
        return results

    def consolidate(
        self,
        short_term: "kimix.memory.short_term_memory.ShortTermMemory",
        threshold: float = 7.0,
    ) -> None:
        """Memory consolidation: migrate high-value short-term to long-term."""
        from kimix.memory.short_term_memory import ShortTermMemory
        if not isinstance(short_term, ShortTermMemory):
            raise TypeError("short_term must be a ShortTermMemory instance")

        to_migrate = [
            entry for entry in short_term.buffer
            if entry.get_effective_importance() >= threshold and not entry.is_expired()
        ]
        if not to_migrate:
            return

        batch: list[tuple[str, MemoryEntry]] = []
        for entry in to_migrate:
            entry_id = self._hash(entry.content)
            if entry.embedding is None:
                entry.embedding = self.embedding_provider.embed(entry.content)
            entry.agent_id = self._agent_id
            batch.append((entry_id, entry))

        if self._backend is not None and batch:
            self._backend.store_many(batch, dim=self.dim)
        else:
            for entry_id, entry in batch:
                self.entries[entry_id] = entry
                self._update_index(entry_id, entry)
            self._dirty = True

        # Invalidate once after batch insert instead of N times in the loop.
        self._invalidate_bm25()

        migrate_ids = {id(entry) for entry in to_migrate}
        short_term.buffer = [e for e in short_term.buffer if id(e) not in migrate_ids]

        self._save()

    def forget(self, entry_id: str) -> None:
        """Active forgetting: reduce importance or delete."""
        if self._backend is not None:
            entry = self._backend.get(entry_id, dim=self.dim)
            if entry is None:
                return
            entry.importance *= 0.5
            if entry.importance < 0.1:
                self._backend.delete(entry_id)
            else:
                self._backend.store(entry, entry_id, dim=self.dim)
            self._invalidate_bm25()
            return

        entry = self.entries.get(entry_id)
        if entry is None:
            return
        entry.importance *= 0.5
        if entry.importance < 0.1:
            del self.entries[entry_id]
            for tag in entry.tags:
                tag_set = self.index.get(tag)
                if tag_set is not None:
                    tag_set.discard(entry_id)
                    if not tag_set:
                        del self.index[tag]
        self._dirty = True
        self._invalidate_bm25()
        self._save()

    def count(self) -> int:
        if self._backend is not None:
            return self._backend.count(agent_id=self._agent_id)
        return len(self.entries)
