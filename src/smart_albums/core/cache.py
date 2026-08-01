"""JSONL-backed cache with thread-safe append-only persistence.

Provides a simple key-value cache backed by a JSONL file (one JSON record per
line). The ``CacheManager`` manages multiple named caches, each stored as a
separate ``.jsonl`` file in a configurable directory.

On load, the JSONL file is replayed into memory. If duplicate keys are
detected (from previous updates), the file is compacted — rewritten with
only the latest value per key — to prevent unbounded growth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import ICache, ICacheManager

logger = logging.getLogger(__name__)

def _normalize_key(key: str | tuple[str, ...]) -> str:
    if isinstance(key, tuple):
        key = ":".join([str(k) for k in key])
    return key

class Cache:
    """
    A single cache backed by a JSONL file.

    JSONL format:
    {"key": "user1", "value": {"name": "John"}}
    {"key": "user2", "value": 123}

    On load, if the file contains more lines than unique keys (i.e.
    duplicate entries from updates), the file is automatically compacted
    to contain only the latest value per key.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._lock = RLock()
        self._data: dict[str, Any] = {}

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Initializing cache at %s", self._file_path)
        self._load()

    def _load(self) -> None:
        """Load cache state by replaying JSONL entries, compacting if needed."""
        if not self._file_path.exists():
            logger.debug("Cache file does not exist, starting empty: %s", self._file_path)
            return

        logger.debug("Loading cache from %s", self._file_path)
        loaded = 0
        skipped = 0
        total_lines = 0

        with self._file_path.open("r", encoding="utf-8") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                total_lines += 1

                try:
                    record = json.loads(line)
                    self._data[record["key"]] = record["value"]
                    loaded += 1
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "Cache %s line %d is corrupted, skipping: %s",
                        self._file_path,
                        line_number,
                        exc,
                    )
                    skipped += 1

        logger.debug(
            "Cache loaded from %s: %d entries loaded, %d skipped",
            self._file_path, loaded, skipped,
        )

        # Compact if file has more valid lines than unique keys (duplicates exist)
        if total_lines - skipped > len(self._data):
            self._compact()

    def _compact(self) -> None:
        """Rewrite the JSONL file with only the current unique entries.

        This eliminates duplicate key entries that accumulate from repeated
        put() calls on the same key, preventing unbounded file growth.
        """
        unique_count = len(self._data)
        logger.info(
            "Compacting cache %s: writing %d unique entries",
            self._file_path.stem,
            unique_count,
        )
        tmp_path = self._file_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for key, value in self._data.items():
                record = {"key": key, "value": value}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(self._file_path)
        logger.debug("Cache compaction complete: %s", self._file_path)

    def get(self, key: str | tuple[str, ...], default: Any = None) -> Any:
        """Retrieve a value from cache."""
        key_str = _normalize_key(key)
        with self._lock:
            hit = key_str in self._data
            logger.debug("Cache %s GET key=%r %s", self._file_path.stem, key_str, "HIT" if hit else "MISS")
            return self._data.get(key_str, default)

    def put(self, key: str | tuple[str, ...], value: Any) -> None:
        """Store a value in cache and append to JSONL."""
        key_str = _normalize_key(key)

        with self._lock:
            is_update = key_str in self._data
            self._data[key_str] = value

            record = {
                "key": key_str,
                "value": value,
            }

            logger.debug(
                "Cache %s PUT key=%r (%s), total entries=%d",
                self._file_path.stem, key_str,
                "update" if is_update else "new",
                len(self._data),
            )

            with self._file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __contains__(self, key: object) -> bool:
        if isinstance(key, (str, tuple)):
            return _normalize_key(key) in self._data
        return False

    def __len__(self) -> int:
        return len(self._data)

@ProtocolsRegistry.register("cache", ICacheManager)
class CacheManager(ICacheManager):
    """
    Manages multiple named caches.

    Example:
        manager = CacheManager("./cache")

        users = manager.get_cache("users")
        users.put("123", {"name": "Alice"})

        products = manager.get_cache("products")
        products.put("p1", {"price": 10})
    """

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._cache_dir = Path(cache_dir).expanduser().absolute()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._caches: dict[str, Cache] = {}
        self._lock = RLock()
        logger.debug("CacheManager initialized with cache_dir=%s", self._cache_dir)

    def get_cache(self, cache_name: str) -> Cache:
        with self._lock:
            if cache_name not in self._caches:
                file_path = self._cache_dir / f"{cache_name}.jsonl"
                logger.debug("Creating new cache %r at %s", cache_name, file_path)
                self._caches[cache_name] = Cache(file_path)
            else:
                logger.debug("Reusing existing cache %r", cache_name)

            return self._caches[cache_name]
