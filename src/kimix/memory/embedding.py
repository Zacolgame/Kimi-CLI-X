"""Embedding vector provider."""

import zlib
from typing import Sequence

import numpy as np


class EmbeddingProvider:
    """Embedding vector provider (replaceable with OpenAI, local models, etc.)."""

    __slots__ = ("dim", "_cache", "_max_cache_size")

    def __init__(self, dim: int = 384, max_cache_size: int = 4096) -> None:
        self.dim = dim
        # Production: use real models; here using simulation
        self._cache: dict[str, np.ndarray] = {}
        self._max_cache_size = max_cache_size

    def embed(self, text: str) -> np.ndarray:
        """Generate text vector embedding."""
        vec = self._cache.get(text)
        if vec is not None:
            return vec

        # Simulated embedding: hash-based deterministic vector
        # Production replacement: openai.Embedding.create() or sentence-transformers
        seed = zlib.crc32(text.encode()) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec /= norm

        self._cache[text] = vec
        if len(self._cache) > self._max_cache_size:
            self._cache.pop(next(iter(self._cache)))
        return vec

    def similarity(self, vec1: Sequence[float] | np.ndarray, vec2: Sequence[float] | np.ndarray) -> float:
        """Compute cosine similarity."""
        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return float(np.dot(v1, v2) / norm)
