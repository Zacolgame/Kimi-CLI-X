from __future__ import annotations

import hashlib
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from kimix.retrieval import (
    BM25Scorer,
    InvertedIndex,
    NgramTokenizer,
    NoisyChannelSpeller,
    QueryPerformancePredictor,
    Searcher,
    SimHash,
    SimHashLSH,
    clarity_score,
    i_match_fingerprint,
    jaro_winkler_similarity,
    metaphone,
    mmr_rerank,
    ngram_overlap,
    porter_stem,
    scq,
    sorensen_dice_coefficient,
    soundex,
    xquad_rerank,
)


class FileReader:
    """Recursively scan text files under given paths and maintain a JSON mapping
    of relative file paths to SHA-256 content hashes.
    """

    __slots__ = ("paths", "output_path", "_mapping", "_max_workers")


    def __init__(self, paths: list[Path], output_path: Path) -> None:
        self.paths = [Path(p).resolve() for p in paths]
        self.output_path = Path(output_path).resolve()
        self._max_workers = min(32, (os.cpu_count() or 1) + 4)
        self._mapping: dict[str, str] = {}
        self._build()

    def _is_text_file(self, path: Path) -> bool:
        """Heuristic to determine whether a file is a text file."""
        try:
            with path.open("rb") as f:
                chunk = f.read(8192)
                if b"\x00" in chunk:
                    return False
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.read(1)
            return True
        except (OSError, UnicodeDecodeError):
            return False

    def _hash_file(self, path: Path) -> str:
        """Compute SHA-256 hash of a file's contents."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read()
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _process_file(self, rel: str, path: Path) -> tuple[str, str] | None:
        """Single-pass text check + SHA-256 hash.

        Returns ``(rel, hex_hash)`` for text files, or ``None`` for binary
        or unreadable files.
        """
        try:
            h = hashlib.sha256()
            with path.open("rb") as f:
                chunk = f.read(8192)
                if b"\x00" in chunk:
                    return None
                h.update(chunk)
                while True:
                    chunk = f.read()
                    if not chunk:
                        break
                    h.update(chunk)
            return rel, h.hexdigest()
        except OSError:
            return None

    def _collect_files(self) -> list[tuple[str, Path]]:
        """Gather all candidate *(rel_path, abs_path)* pairs."""
        files: list[tuple[str, Path]] = []
        cwd = Path.cwd()
        for root in self.paths:
            if not root.exists():
                continue
            if root.is_file():
                try:
                    rel = str(root.relative_to(cwd)).replace("\\", "/")
                except ValueError:
                    rel = root.name
                files.append((rel, root))
                continue
            for file_path in root.rglob("*"):
                if file_path.is_file():
                    try:
                        rel = str(file_path.relative_to(cwd)).replace("\\", "/")
                    except ValueError:
                        rel = str(file_path.relative_to(root)).replace("\\", "/")
                    files.append((rel, file_path))
        return files

    def _scan(self) -> dict[str, str]:
        """Recursively scan paths and return {relative_path: hash}."""
        files = self._collect_files()
        mapping: dict[str, str] = {}
        if not files:
            return mapping

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(self._process_file, rel, path)
                for rel, path in files
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rel, hash_val = result
                    mapping[rel] = hash_val
        return mapping

    def _build(self) -> None:
        """Initial build: scan and write JSON."""
        self._mapping = self._scan()
        self._write()

    def _write(self) -> None:
        """Persist the current mapping to JSON."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(self._mapping, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def update(self) -> bool:
        """Re-scan directories and rewrite JSON if any file was created,
        deleted, or modified.
        """
        current = self._scan()
        if current != self._mapping:
            self._mapping = current
            self._write()
            return True
        return False


class FileBuilder:
    """Build a BM25-searchable index over text files."""

    def __init__(
        self,
        paths: list[Path],
        output_path: Path,
        n: int = 2,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.file_reader = FileReader(paths, output_path)
        self.paths = [Path(p).resolve() for p in paths]
        self._n = n
        self._k1 = k1
        self._b = b
        self._search: Searcher | None = None
        self._doc_info: list[dict[str, Any]] = []
        self._cache_path = Path(output_path).with_suffix(".index_cache.pkl")
        if self._cache_path.exists() and self._cache_valid():
            self._load_cache()
        else:
            self._build()
            self._save_cache()

    def _cache_valid(self) -> bool:
        try:
            with self._cache_path.open("rb") as f:
                cache = pickle.load(f)
            return cache.get("mapping") == self.file_reader._mapping
        except Exception:
            return False

    def _load_cache(self) -> None:
        with self._cache_path.open("rb") as f:
            cache = pickle.load(f)
        self._doc_info = cache["doc_info"]
        self._search = cache["searcher"]

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("wb") as f:
            pickle.dump(
                {
                    "mapping": self.file_reader._mapping,
                    "doc_info": self._doc_info,
                    "searcher": self._search,
                },
                f,
            )

    def _collect_files(self) -> list[tuple[str, Path]]:
        files: list[tuple[str, Path]] = []
        cwd = Path.cwd()
        for root in self.paths:
            if not root.exists():
                continue
            if root.is_file():
                try:
                    rel = str(root.relative_to(cwd)).replace("\\", "/")
                except ValueError:
                    rel = root.name
                files.append((rel, root))
                continue
            for file_path in root.rglob("*"):
                if file_path.is_file():
                    try:
                        rel = str(file_path.relative_to(cwd)).replace("\\", "/")
                    except ValueError:
                        rel = str(file_path.relative_to(root)).replace("\\", "/")
                    files.append((rel, file_path))
        return files

    def _build(self) -> None:
        index = InvertedIndex()
        tokenizer = NgramTokenizer(n=self._n)
        doc_info: list[dict[str, Any]] = []
        doc_id = 0
        lsh = SimHashLSH()
        seen_imatch: set[str] = set()
        for rel, abs_path in self._collect_files():
            try:
                with abs_path.open("r", encoding="utf-8", errors="replace") as f:
                    for line_idx, line in enumerate(f):
                        stripped = line.strip()
                        if stripped:
                            h = SimHash(stripped)
                            fp = i_match_fingerprint(stripped.split())

                            dup = False
                            for other_id in lsh.candidates(h):
                                if h.is_near_duplicate(lsh.hashes[other_id], threshold=3):
                                    dup = True
                                    break
                            if not dup and fp in seen_imatch:
                                dup = True
                            if dup:
                                continue

                            lsh.add(doc_id, h)
                            seen_imatch.add(fp)

                            # Optional stemming for Latin tokens to improve recall
                            raw_tokens = tokenizer.tokenize(f"{rel}: {stripped}")
                            stemmed = [porter_stem(t) if t.isalpha() and len(t) > 3 else t for t in raw_tokens]
                            index.add_document(doc_id, stemmed)
                            doc_info.append({"path": rel, "line_index": line_idx, "content": stripped})
                            doc_id += 1
            except OSError:
                continue
        self._doc_info = doc_info
        if doc_id > 0:
            index.finalize(stop_threshold=1.0)
            self._search = Searcher(index, tokenizer=tokenizer, k1=self._k1, b=self._b)
        else:
            self._search = None

    def search(
        self,
        keywords: str,
        top_k: int = 5,
        diversify: bool = False,
        diversity_lambda: float = 0.5,
        use_spelling: bool = True,
        use_stemming: bool = True,
        use_rm3: bool = False,
        use_rocchio: bool = False,
        use_xquad: bool = False,
        xquad_lambda: float = 0.5,
        use_string_similarity: bool = False,
        use_ltr: bool = False,
        ltr_model: str = "lambdamart",
        use_adaptive_scoring: bool = False,
    ) -> list[dict[str, Any]]:
        if self._search is None:
            return []

        original_keywords = keywords
        index = self._search.index

        # Query preprocessing: spelling + stemming
        if use_spelling and index is not None:
            term_freqs: dict[str, int] = {}
            for term in index.terms():
                postings = index.get_postings(term)
                if postings is not None:
                    term_freqs[term] = int(postings[1].sum())
            if term_freqs:
                speller = NoisyChannelSpeller(term_freqs, max_edits=2)
                corrected = []
                for w in keywords.split():
                    cw = speller.correct(w)
                    corrected.append(cw if cw else w)
                keywords = " ".join(corrected)
        if use_stemming:
            keywords = " ".join(porter_stem(w) for w in keywords.split())

        # Base BM25 search
        raw_results = self._search.search(keywords, top_k=top_k * 4 if use_ltr or use_xquad or use_string_similarity else top_k)

        if not raw_results:
            return []

        doc_ids = [doc_id for doc_id, _ in raw_results]
        n_results = len(raw_results)

        # Build score arrays for re-ranking
        bm25_scores = np.zeros(n_results, dtype=np.float64)
        for i, (_, score) in enumerate(raw_results):
            bm25_scores[i] = score
        max_bm25 = float(bm25_scores.max()) if n_results else 1.0
        min_bm25 = float(bm25_scores.min()) if n_results else 0.0
        bm25_range = max_bm25 - min_bm25
        if bm25_range == 0:
            bm25_norm = np.ones(n_results, dtype=np.float64)
        else:
            bm25_norm = (bm25_scores - min_bm25) / bm25_range

        string_scores = np.zeros(n_results, dtype=np.float64)
        if use_string_similarity:
            for i, doc_id in enumerate(doc_ids):
                content = self._doc_info[doc_id]["content"]
                jw = jaro_winkler_similarity(original_keywords, content)
                dice = sorensen_dice_coefficient(original_keywords, content)
                ngo = ngram_overlap(original_keywords, content)
                string_scores[i] = (jw + dice + ngo) / 3.0

        # Adaptive scoring via QPP
        _bm25_weight = 1.0
        if use_adaptive_scoring and index is not None:
            scorer = BM25Scorer(index)
            qpp = QueryPerformancePredictor(index, scorer)
            q_tokens = self._search.tokenizer.tokenize(keywords)
            if q_tokens:
                if qpp.is_hard_query(q_tokens, avg_idf_threshold=2.0):
                    _bm25_weight = 0.7
                else:
                    _bm25_weight = 0.5
                clarity = clarity_score(index, q_tokens)
                scq_val = scq(index, q_tokens)
                if clarity > 1.0 or scq_val > 5.0:
                    _bm25_weight = min(0.9, _bm25_weight + 0.1)
                elif clarity < 0.5:
                    _bm25_weight = max(0.3, _bm25_weight - 0.1)
            else:
                _bm25_weight = 0.5
        else:
            _bm25_weight = 0.5

        final_scores = _bm25_weight * bm25_norm + (1.0 - _bm25_weight) * string_scores

        # Sort by combined score
        order = np.argsort(-final_scores).tolist()
        sorted_results = [(doc_ids[i], float(final_scores[i])) for i in order]

        # Keep top candidates for diversification / LTR
        candidate_results = sorted_results[:top_k * 2] if (use_ltr or use_xquad) else sorted_results[:top_k]

        # LTR re-ranking
        if use_ltr and len(candidate_results) >= 2:
            from kimix.retrieval import LambdaMART, RankSVM, RankBoost
            features: list[list[float]] = []
            labels: list[float] = []
            c_doc_ids = [d for d, _ in candidate_results]
            for i, doc_id in enumerate(c_doc_ids):
                idx_in_raw = doc_ids.index(doc_id)
                feat = [
                    float(bm25_norm[idx_in_raw]),
                    float(string_scores[idx_in_raw]),
                    len(self._doc_info[doc_id]["content"]) / 200.0,
                ]
                features.append(feat)
                labels.append(float(candidate_results[i][1]))
            if features:
                doc_feats = [(i, f) for i, f in enumerate(features)]
                if ltr_model == "lambdamart":
                    model = LambdaMART(n_iterations=10, learning_rate=0.05)
                    model.fit([features], [labels])  # type: ignore[arg-type]
                elif ltr_model == "ranksvm":
                    model = RankSVM(learning_rate=0.01, n_iterations=50)
                    model.fit(features, labels)  # type: ignore[arg-type]
                else:
                    model = RankBoost(n_iterations=20)
                    model.fit(features, labels)  # type: ignore[arg-type]
                ranked = model.rank(doc_feats)  # type: ignore[arg-type]
                ranked_doc_indices = [c_doc_ids[idx] for idx, _ in ranked]
                scores_map = {doc_id: score for doc_id, score in candidate_results}
                candidate_results = [(d, scores_map[d]) for d in ranked_doc_indices]

        # Diversification: xQuAD -> MMR fallback
        if use_xquad and index is not None:
            aspects: dict[int, set[str]] = {}
            for doc_id, _ in candidate_results:
                path = self._doc_info[doc_id]["path"]
                aspects[doc_id] = set(Path(path).parent.parts)
            candidate_results = xquad_rerank(
                candidate_results, aspects, lambda_param=xquad_lambda, top_k=top_k
            )
        elif diversify and index is not None:
            candidate_results = mmr_rerank(
                candidate_results,
                index,
                lambda_param=diversity_lambda,
                top_k=top_k,
            )
        else:
            candidate_results = candidate_results[:top_k]

        results: list[dict[str, Any]] = []
        for doc_id, score in candidate_results:
            info = self._doc_info[doc_id]
            results.append({
                "doc_id": doc_id,
                "score": score,
                "path": info["path"],
                "line_index": info["line_index"] + 1,
            })
        return results

    def update(self) -> None:
        if self.file_reader.update():
            self._build()
            self._save_cache()


def formatted_print(results: list[dict[str, Any]]) -> str:
    """Convert search results into a human-readable formatted string."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['path']} (line {r['line_index']})  score={r['score']:.4f}")
    return "\n".join(lines)
