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
import os
import re
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from protocols_system import ProtocolsRegistry
from protocols_system.protocols import ICacheManager

logger = logging.getLogger(__name__)

# Characters allowed in a cache name used as a filename stem. Anything else
# is replaced, so a cache_name containing path separators or ".." cannot
# escape the cache directory (CO-08 path traversal).
_SAFE_CACHE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _default_cache_dir() -> str:
    """Compute a sensible default cache directory that doesn't depend on
    the current working directory.

    The old default (a bare relative ``".cache"`` string) resolved against
    whatever the process's CWD happened to be. In the CLI Docker image that
    is "/" (no ``WORKDIR`` is set for the runtime stage), so the default
    "cache" provider tried to create "/.cache" as a non-root user and
    failed with PermissionError, breaking every pipeline run that didn't
    explicitly configure ``cachemanager.cache.cache_dir``. Outside Docker,
    a CWD-relative cache is a "which cache am I actually using" hazard —
    running the same command from a different directory silently uses a
    different (or no) cache.

    Resolution order:
    1. ``APPDATA_DIR`` env var (set by the CLI Docker image to ``/config``)
       — ``<APPDATA_DIR>/cache``.
    2. ``~/.cache/smart-albums`` (XDG-ish default for local/dev use).
    """
    appdata_dir = os.environ.get("APPDATA_DIR")
    if appdata_dir:
        return str(Path(appdata_dir) / "cache")
    return str(Path.home() / ".cache" / "smart-albums")

def _normalize_key(key: str | tuple[str, ...]) -> str:
    """Normalize a cache key to a stable, collision-free string.

    A tuple key is encoded with ``json.dumps`` rather than ``":".join(...)``
    (CO-08): joining on ``":"`` made ``("a:b", "c")`` and ``("a", "b:c")``
    collide, which is a real hazard because model names like ``llava:latest``
    (used in embedding/score cache keys) contain colons. JSON encoding of
    the list is unambiguous.
    """
    if isinstance(key, tuple):
        return json.dumps(list(key), ensure_ascii=False)
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

        corrupted = False
        with self._file_path.open("r", encoding="utf-8") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                # A crash mid-write can leave a final line without its
                # trailing newline, so the next append is glued onto it and
                # produces one unparseable line (CO-08). Such a line is
                # skipped here (and compaction below rewrites the file
                # cleanly, dropping it). ``strip`` also tolerates a torn
                # line that happens to still be valid JSON.
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
                    corrupted = True

        logger.debug(
            "Cache loaded from %s: %d entries loaded, %d skipped",
            self._file_path, loaded, skipped,
        )

        # Compact if the file has more valid lines than unique keys
        # (duplicates exist), or if any line was corrupted — the latter
        # ensures a torn/garbled line is physically removed on next load
        # rather than lingering forever (CO-08).
        if corrupted or (total_lines - skipped > len(self._data)):
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
        # Use a process-unique temp name (CO-08): the web GUI can run several
        # CLI processes against the same cache dir, and a shared
        # ``.jsonl.tmp`` name meant two concurrent compactions clobbered
        # each other's temp file. fsync before the atomic replace so the
        # rewritten contents are durable even if the machine loses power
        # right after.
        tmp_path = self._file_path.with_name(
            f"{self._file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for key, value in self._data.items():
                    record = {"key": key, "value": value}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self._file_path)
        finally:
            # If replace() succeeded the temp file is gone; this cleans up
            # a temp left behind by a mid-compaction failure.
            tmp_path.unlink(missing_ok=True)
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

        # Serialize *before* mutating in-memory state (CO-08). The old order
        # updated ``self._data`` first, so a value that can't be JSON-encoded
        # (e.g. a numpy float32/ndarray) raised after the memory dict already
        # held it — leaving memory and the on-disk file permanently out of
        # sync. Encoding first means a bad value fails cleanly with nothing
        # changed.
        record = {"key": key_str, "value": value}
        line = json.dumps(record, ensure_ascii=False) + "\n"

        with self._lock:
            is_update = key_str in self._data

            logger.debug(
                "Cache %s PUT key=%r (%s), total entries=%d",
                self._file_path.stem, key_str,
                "update" if is_update else "new",
                len(self._data) + (0 if is_update else 1),
            )

            with self._file_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

            self._data[key_str] = value

    def __contains__(self, key: object) -> bool:
        if isinstance(key, (str, tuple)):
            with self._lock:
                return _normalize_key(key) in self._data
        return False

    def __len__(self) -> int:
        with self._lock:
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

    def __init__(self, cache_dir: str | None = None) -> None:
        self._cache_dir = Path(cache_dir or _default_cache_dir()).expanduser().absolute()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._caches: dict[str, Cache] = {}
        self._lock = RLock()
        logger.debug("CacheManager initialized with cache_dir=%s", self._cache_dir)

    def get_cache(self, cache_name: str) -> Cache:
        with self._lock:
            if cache_name not in self._caches:
                # Sanitize the name before using it as a filename stem so a
                # cache_name containing "/" or ".." can't write outside the
                # cache directory (CO-08). The cache is still keyed by the
                # original name in-memory so distinct names stay distinct.
                safe_stem = _SAFE_CACHE_NAME.sub("_", cache_name).strip("._") or "cache"
                file_path = self._cache_dir / f"{safe_stem}.jsonl"
                logger.debug("Creating new cache %r at %s", cache_name, file_path)
                self._caches[cache_name] = Cache(file_path)
            else:
                logger.debug("Reusing existing cache %r", cache_name)

            return self._caches[cache_name]
