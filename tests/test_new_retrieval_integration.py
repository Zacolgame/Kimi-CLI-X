"""Integration tests for new retrieval algorithms in memory and file_builder."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Iterator

from kimix.memory.embedding import EmbeddingProvider
from kimix.memory.long_term_memory import LongTermMemory
from kimix.memory.short_term_memory import ShortTermMemory
from kimix.memory.types import MemoryEntry, MemoryType
from kimix.tools.skill.searching.file_builder import FileBuilder


class TestLongTermMemoryNewFeatures:
    """Test SimHash dedup, MMR, RM3, and adaptive BM25 in LTM."""

    @contextlib.contextmanager
    def _make_ltm(self) -> Iterator[tuple[LongTermMemory, str]]:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        try:
            yield LongTermMemory(storage_path=tmp.name), tmp.name
        finally:
            os.unlink(tmp.name)

    def test_simhash_dedup_on_store(self):
        with self._make_ltm() as (ltm, _path):
            ltm.store("python asyncio guide", importance=5.0, tags=["python"])
            # Near-duplicate should merge into existing entry
            ltm.store("python asyncio guide", importance=3.0, tags=["async"])
            assert len(ltm.entries) == 1
            entry = list(ltm.entries.values())[0]
            assert entry.importance > 5.0  # boosted
            assert "python" in entry.tags
            assert "async" in entry.tags

    def test_simhash_no_false_dedup(self):
        with self._make_ltm() as (ltm, _path):
            ltm.store("python asyncio guide", importance=5.0)
            ltm.store("completely different content here", importance=5.0)
            assert len(ltm.entries) == 2

    def test_mmr_rerank_in_retrieve(self):
        with self._make_ltm() as (ltm, _path):
            ltm.store("python async programming patterns", importance=8.0)
            ltm.store("python threading concurrency guide", importance=7.0)
            ltm.store("java async programming tutorial", importance=6.0)
            ltm.store("rust async runtime internals", importance=5.0)
            # Without diversity, top results may be similar
            standard = ltm.retrieve("async programming", top_k=3, use_diversity=False)
            assert len(standard) == 3
            diverse = ltm.retrieve(
                "async programming",
                top_k=3,
                use_diversity=True,
                diversity_lambda=0.1,  # strong diversity
            )
            assert len(diverse) == 3
            # Diverse results may reorder or differ; just verify it runs
            assert {e.content for e in diverse} == {e.content for e in standard}

    def test_rm3_query_expansion(self):
        with self._make_ltm() as (ltm, _path):
            ltm.store("python async programming patterns", importance=8.0)
            ltm.store("python threading concurrency guide", importance=7.0)
            ltm.store("java async programming tutorial", importance=6.0)
            with_rm3 = ltm.retrieve(
                "python",
                top_k=3,
                use_rm3=True,
                rm3_fb_docs=2,
                rm3_fb_terms=3,
            )
            without_rm3 = ltm.retrieve("python", top_k=3, use_rm3=False)
            assert len(with_rm3) > 0
            assert len(without_rm3) > 0

    def test_adaptive_bm25_weight(self):
        with self._make_ltm() as (ltm, _path):
            # Seed with a common word to create a "hard" query
            for i in range(20):
                ltm.store(f"common word{i} details", importance=5.0)
            results = ltm.retrieve(
                "common",
                top_k=5,
                adaptive_bm25=True,
                bm25_weight=0.3,
            )
            assert len(results) > 0

    def test_persistence_restores_simhash(self):
        with self._make_ltm() as (ltm, path):
            ltm.store("python asyncio guide", importance=5.0)
            del ltm
            ltm2 = LongTermMemory(storage_path=path)
            # SimHash map should be restored so dedup still works
            ltm2.store("python asyncio guide", importance=3.0)
            assert len(ltm2.entries) == 1

    def test_mmr_empty_results(self):
        with self._make_ltm() as (ltm, _path):
            results = ltm.retrieve(
                "nonexistent query",
                top_k=5,
                use_diversity=True,
            )
            assert results == []


class TestShortTermMemoryNewFeatures:
    """Test Jaro-Winkler fallback in STM."""

    def test_string_fallback_low_semantic(self):
        stm = ShortTermMemory(max_size=10)
        provider = EmbeddingProvider(dim=384)
        stm.add(MemoryEntry(content="python async programming", memory_type=MemoryType.EPISODIC))
        stm.add(MemoryEntry(content="python threading model", memory_type=MemoryType.EPISODIC))
        stm.add(MemoryEntry(content="flask web framework", memory_type=MemoryType.EPISODIC))
        # With fallback enabled and a query unlikely to produce strong embeddings
        results = stm.search(
            "async python",
            provider,
            top_k=2,
            use_string_fallback=True,
        )
        assert len(results) <= 2
        assert len(results) > 0

    def test_string_fallback_disabled(self):
        stm = ShortTermMemory(max_size=10)
        provider = EmbeddingProvider(dim=384)
        stm.add(MemoryEntry(content="alpha beta gamma", memory_type=MemoryType.EPISODIC))
        results = stm.search(
            "delta epsilon",
            provider,
            top_k=2,
            use_string_fallback=False,
        )
        # Semantic search may still return results, but fallback path is not taken
        assert isinstance(results, list)


class TestFileBuilderNewFeatures:
    """Test SimHash dedup, MMR, and porter_stem in FileBuilder."""

    @contextlib.contextmanager
    def _make_project(self, files: dict[str, str]) -> Iterator[Path]:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for rel_path, content in files.items():
                full = root / rel_path
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")
            yield root

    def test_simhash_dedup_skips_duplicate_lines(self):
        with self._make_project({
            "src/main.py": "import os\nimport os\nimport os\ndef main(): pass\n"
        }) as root:
            fb = FileBuilder([root], root / "index.json")
            # Three identical "import os" lines should dedup to one
            results = fb.search("import os", top_k=10)
            paths = [r["path"] for r in results]
            assert paths.count("src/main.py") == 1

    def test_porter_stem_improves_recall(self):
        with self._make_project({
            "README.md": "Running runners run quickly\n"
        }) as root:
            fb = FileBuilder([root], root / "index.json")
            # "run" should match stemmed "running", "runners", "run"
            results = fb.search("run", top_k=5)
            assert len(results) > 0

    def test_mmr_diversify_search(self):
        with self._make_project({
            "a.py": "async programming patterns\n",
            "b.py": "async programming tutorial\n",
            "c.py": "threading concurrency model\n",
            "d.py": "network io programming\n",
        }) as root:
            fb = FileBuilder([root], root / "index.json")
            standard = fb.search("programming", top_k=3, diversify=False)
            diverse = fb.search(
                "programming",
                top_k=3,
                diversify=True,
                diversity_lambda=0.5,
            )
            assert len(standard) == 3
            assert len(diverse) == 3
            # MMR path should execute without error and return valid doc ids
            assert all(isinstance(r["doc_id"], int) for r in diverse)
            assert all(r["score"] >= 0 for r in diverse)

    def test_search_empty_project(self):
        with self._make_project({}) as root:
            fb = FileBuilder([root], root / "index.json")
            results = fb.search("anything", top_k=5)
            assert results == []

    def test_update_rebuilds_index(self):
        with self._make_project({
            "old.py": "legacy code\n"
        }) as root:
            fb = FileBuilder([root], root / "index.json")
            assert len(fb.search("legacy", top_k=5)) > 0
            # Add a new file and update
            (root / "new.py").write_text("modern code\n", encoding="utf-8")
            fb.update()
            assert len(fb.search("modern", top_k=5)) > 0
