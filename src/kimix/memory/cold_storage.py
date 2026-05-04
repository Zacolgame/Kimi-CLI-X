"""L6 Cold Storage Archive: time-blocked, compressed long-term archives."""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from kimix.memory.types import MemoryEntry, MemoryType

# Re-use json.dumps kwargs dict to avoid rebuilding it every call.
_JSON_DUMPS_KWARGS = {"ensure_ascii": False, "separators": (",", ":"), "check_circular": False}


class ColdStorage:
    """Archive memories into time-blocked, compressed files.

    Each block is named by a date range (e.g. ``2022-2024.jsonl.gz``).
    Memories are stored as JSON Lines inside gzip for efficient streaming.
    """

    __slots__ = (
        "archive_dir",
        "_meta_path",
        "_blocks_cache",
        "_meta_cache",
        "_archive_paths_cache",
    )

    def __init__(self, archive_dir: str | Path = ".kimix_cache/cold_storage") -> None:
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.archive_dir / "_meta.json"
        self._blocks_cache: list[tuple[str, int, int]] | None = None
        self._meta_cache: dict[str, int] | None = None
        self._archive_paths_cache: list[tuple[Path, int, int]] | None = None

    @staticmethod
    def _block_name(start_year: int, end_year: int) -> str:
        return f"{start_year}-{end_year}.jsonl.gz"

    @staticmethod
    def _parse_block_name(name: str) -> tuple[int, int] | None:
        """Parse ``YYYY-YYYY.jsonl.gz`` -> (start_year, end_year)."""
        if not name.endswith(".jsonl.gz"):
            return None
        stem = name[:-9]
        if "-" not in stem:
            return None
        try:
            a, b = stem.split("-", 1)
            return int(a), int(b)
        except ValueError:
            return None

    def _block_for_timestamp(self, ts: float) -> Path:
        year = time.gmtime(ts).tm_year
        block_name = self._block_name(year, year)
        return self.archive_dir / block_name

    @staticmethod
    def _entry_to_json(entry: MemoryEntry) -> str:
        """Fast serialization bypassing ``to_dict()`` (avoids ``get_effective_importance()``)."""
        embedding = entry.embedding
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        return json.dumps(
            {
                "content": entry.content,
                "memory_type": entry.memory_type.value,
                "timestamp": entry.timestamp,
                "importance": entry.importance,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
                "embedding": embedding,
                "tags": entry.tags,
                "source": entry.source,
                "metadata": entry.metadata,
                "expires_at": entry.expires_at,
                "agent_id": entry.agent_id,
            },
            **_JSON_DUMPS_KWARGS,
        )

    def _read_meta(self) -> dict[str, int]:
        if self._meta_cache is not None:
            return self._meta_cache
        if self._meta_path.exists():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    self._meta_cache = json.load(f)
                    return self._meta_cache
            except Exception:
                pass
        self._meta_cache = {}
        return self._meta_cache

    def _write_meta(self, meta: dict[str, int]) -> None:
        self._meta_cache = meta
        tmp = self._meta_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        tmp.replace(self._meta_path)

    def _update_meta(self, block_name: str, delta: int) -> None:
        meta = self._read_meta()
        new_count = meta.get(block_name, 0) + delta
        if new_count <= 0:
            meta.pop(block_name, None)
        else:
            meta[block_name] = new_count
        self._write_meta(meta)
        self._blocks_cache = None
        self._archive_paths_cache = None

    def _update_meta_batch(self, deltas: dict[str, int]) -> None:
        meta = self._read_meta()
        for block_name, delta in deltas.items():
            new_count = meta.get(block_name, 0) + delta
            if new_count <= 0:
                meta.pop(block_name, None)
            else:
                meta[block_name] = new_count
        self._write_meta(meta)
        self._blocks_cache = None
        self._archive_paths_cache = None

    def _get_archive_paths(self) -> list[tuple[Path, int, int]]:
        """Return cached list of (path, start_year, end_year)."""
        if self._archive_paths_cache is not None:
            return self._archive_paths_cache
        paths: list[tuple[Path, int, int]] = []
        parse = self._parse_block_name
        for path in self.archive_dir.glob("*.jsonl.gz"):
            parsed = parse(path.name)
            if parsed:
                paths.append((path, parsed[0], parsed[1]))
        self._archive_paths_cache = paths
        return paths

    def archive(
        self,
        entries: Iterable[MemoryEntry],
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> Path:
        """Archive a batch of memories into the appropriate time block.

        If *start_year* and *end_year* are provided they override auto-detection.
        """
        dumps = self._entry_to_json

        if start_year is not None and end_year is not None:
            block_path = self.archive_dir / self._block_name(start_year, end_year)
            mode = "ab" if block_path.exists() else "wb"
            # Collect all JSON strings first, encode once, write once.
            lines: list[str] = []
            append_line = lines.append
            count = 0
            for entry in entries:
                append_line(dumps(entry))
                count += 1
            if count == 0:
                raise ValueError("No entries to archive")
            batch = "\n".join(lines).encode("utf-8") + b"\n"
            with gzip.GzipFile(block_path, mode, compresslevel=1) as f:
                f.write(batch)
            self._update_meta(block_path.name, count)
            return block_path

        groups: dict[int, list[MemoryEntry]] = defaultdict(list)
        total = 0
        gmtime = time.gmtime
        for entry in entries:
            groups[gmtime(entry.timestamp).tm_year].append(entry)
            total += 1

        if total == 0:
            raise ValueError("No entries to archive")

        first_path: Path | None = None
        deltas: dict[str, int] = {}
        for year in sorted(groups):
            block_path = self.archive_dir / self._block_name(year, year)
            mode = "ab" if block_path.exists() else "wb"
            group = groups[year]
            # Build batch as string first, encode once.
            batch = "\n".join(dumps(e) for e in group).encode("utf-8") + b"\n"
            with gzip.GzipFile(block_path, mode, compresslevel=1) as f:
                f.write(batch)
            deltas[block_path.name] = len(group)
            if first_path is None:
                first_path = block_path

        self._update_meta_batch(deltas)
        assert first_path is not None
        return first_path

    def restore_range(
        self,
        start_year: int,
        end_year: int,
    ) -> list[MemoryEntry]:
        """Restore all memories whose archive block overlaps the year range."""
        results: list[MemoryEntry] = []
        loads = json.loads
        archive_dir = self.archive_dir

        for path, block_start, block_end in self._get_archive_paths():
            if block_end < start_year or block_start > end_year:
                continue
            with gzip.open(path, "rb") as f:
                for line in f:
                    if not line:
                        continue
                    try:
                        data = loads(line)
                        results.append(
                            MemoryEntry(
                                content=data["content"],
                                memory_type=MemoryType(data["memory_type"]),
                                timestamp=data["timestamp"],
                                importance=data["importance"],
                                access_count=data["access_count"],
                                last_accessed=data["last_accessed"],
                                embedding=data.get("embedding"),
                                tags=data.get("tags", []),
                                source=data.get("source", ""),
                                metadata=data.get("metadata", {}),
                                expires_at=data.get("expires_at"),
                                agent_id=data.get("agent_id", "default"),
                            )
                        )
                    except Exception:
                        continue
        return results

    def list_archives(self) -> list[tuple[str, int, int]]:
        """List all archives as (filename, start_year, end_year)."""
        if self._blocks_cache is not None:
            return list(self._blocks_cache)
        archives: list[tuple[str, int, int]] = []
        for path, start, end in self._get_archive_paths():
            archives.append((path.name, start, end))
        archives.sort(key=lambda x: x[0])
        self._blocks_cache = archives
        return archives

    def delete_archive(self, start_year: int, end_year: int) -> bool:
        """Delete a specific archive block."""
        path = self.archive_dir / self._block_name(start_year, end_year)
        if path.exists():
            path.unlink()
            meta = self._read_meta()
            meta.pop(path.name, None)
            self._write_meta(meta)
            self._blocks_cache = None
            self._archive_paths_cache = None
            return True
        return False

    def reflect(self) -> str:
        meta = self._read_meta()
        total_entries = sum(meta.values())
        archives = self.list_archives()
        return (
            f"Cold Storage: {len(archives)} archives, ~{total_entries} entries"
        )
