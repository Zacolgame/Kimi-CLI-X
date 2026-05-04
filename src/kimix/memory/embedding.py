"""Embedding vector provider."""

import threading
import zlib
from collections import OrderedDict
from typing import Sequence

import numpy as np


class EmbeddingProvider:
    """Embedding vector provider (replaceable with OpenAI, local models, etc.)."""

    __slots__ = ("dim", "_cache", "_max_cache_size", "_lock")

    def __init__(self, dim: int = 384, max_cache_size: int = 4096) -> None:
        self.dim = dim
        # Production: use real models; here using simulation
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_cache_size = max_cache_size
        self._lock = threading.Lock()

    def _compute(self, text: str) -> np.ndarray:
        """Simulated embedding: hash-based deterministic vector."""
        seed = zlib.crc32(text.encode()) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec

    def embed(self, text: str) -> np.ndarray:
        """Generate text vector embedding."""
        with self._lock:
            vec = self._cache.get(text)
            if vec is not None:
                self._cache.move_to_end(text)
                return vec

        vec = self._compute(text)

        with self._lock:
            self._cache[text] = vec
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batch embedding with cache awareness.

        Enables a single API call for real providers; the simulated provider
        computes missing vectors directly while still leveraging cache.
        """
        results: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        with self._lock:
            for i, text in enumerate(texts):
                vec = self._cache.get(text)
                if vec is not None:
                    self._cache.move_to_end(text)
                    results[i] = vec
                else:
                    missing_texts.append(text)
                    missing_indices.append(i)

        if missing_texts:
            compute = self._compute
            computed = [compute(t) for t in missing_texts]

            with self._lock:
                for text, idx, vec in zip(missing_texts, missing_indices, computed):
                    results[idx] = vec
                    self._cache[text] = vec
                # Evict excess in one shot
                excess = len(self._cache) - self._max_cache_size
                for _ in range(excess):
                    self._cache.popitem(last=False)

        return results

    def similarity(self, vec1: Sequence[float] | np.ndarray, vec2: Sequence[float] | np.ndarray) -> float:
        """Compute cosine similarity."""
        if isinstance(vec1, np.ndarray):
            v1 = vec1
        else:
            v1 = np.asarray(vec1, dtype=np.float32)
        if isinstance(vec2, np.ndarray):
            v2 = vec2
        else:
            v2 = np.asarray(vec2, dtype=np.float32)
        dot = np.dot(v1, v2)
        if dot == 0:
            return 0.0
        norms = np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
        if norms == 0:
            return 0.0
        return float(dot / norms)
