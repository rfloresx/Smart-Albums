"""LocalImageClient — filesystem-backed IImageClient implementation.

Operates entirely against a local root directory:

    <root>/
    ├── assets/          # All photo files (scanned recursively)
    ├── db.json          # Local database (albums, asset index)
    └── metadata.json    # (optional) User-supplied per-file metadata

No network access required. Albums are stored as JSON entries in db.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from protocols_system.protocols import AlbumResult, AlbumSummary, Asset
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import (
    IHealthCheck,
    IImageClient,
)

from smart_albums.utils.datetime_utils import to_aware_utc

logger = logging.getLogger(__name__)

# File extensions considered as images
_IMAGE_EXTENSIONS: set[str] = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".webp", ".tiff", ".tif", ".bmp", ".gif",
}


def _extract_exif(path: Path) -> dict[str, Any]:
    """Extract EXIF metadata from an image file using Pillow.

    Returns a dict with optional keys: captured_at, latitude, longitude, exif.
    """
    result: dict[str, Any] = {}
    try:
        from PIL import Image
        from PIL.ExifTags import Base as ExifBase, GPS as GPSTags

        with Image.open(path) as img:
            exif_data = img.getexif()
            if not exif_data:
                return result

            # DateTimeOriginal
            # Try IFD tag first (more reliable), then top-level
            ifd = exif_data.get_ifd(0x8769)  # ExifIFD
            date_str = ifd.get(ExifBase.DateTimeOriginal) if ifd else None
            if date_str is None:
                date_str = exif_data.get(ExifBase.DateTime)
            if date_str and isinstance(date_str, str):
                try:
                    result["captured_at"] = datetime.strptime(
                        date_str, "%Y:%m:%d %H:%M:%S"
                    )
                except ValueError:
                    pass

            # GPS
            gps_ifd = exif_data.get_ifd(0x8825)  # GPSInfo
            if gps_ifd:
                lat = _parse_gps_coord(
                    gps_ifd.get(GPSTags.GPSLatitude),
                    gps_ifd.get(GPSTags.GPSLatitudeRef),
                )
                lon = _parse_gps_coord(
                    gps_ifd.get(GPSTags.GPSLongitude),
                    gps_ifd.get(GPSTags.GPSLongitudeRef),
                )
                if lat is not None:
                    result["latitude"] = lat
                if lon is not None:
                    result["longitude"] = lon

            # Store raw exif subset
            result["exif"] = {
                "date_time_original": date_str,
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
            }
    except Exception as exc:
        logger.debug("EXIF extraction failed for %s: %s", path, exc)

    return result


def _parse_gps_coord(
    dms: Any, ref: Any
) -> Optional[float]:
    """Convert EXIF GPS DMS tuple + ref to decimal degrees."""
    if dms is None or ref is None:
        return None
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, IndexError, ValueError):
        return None


def _asset_id_from_path(root_assets: Path, file_path: Path) -> str:
    """Derive a stable asset ID from the relative path within assets/."""
    return str(file_path.relative_to(root_assets))


@ProtocolsRegistry.register("local", IImageClient)
@ProtocolsRegistry.register("local", IHealthCheck)
class LocalImageClient:
    """Filesystem-backed IImageClient.

    Usage::

        async with LocalImageClient(root_path="/photos/my-library") as client:
            assets = await client.search_assets(after, before)
    """

    def __init__(self, root_path: str) -> None:
        self._root = Path(root_path).resolve()
        self._assets_dir = self._root / "assets"
        self._db_path = self._root / "db.json"
        self._metadata_path = self._root / "metadata.json"
        self._db: dict[str, Any] = {}
        self._dirty: bool = False

    async def __aenter__(self) -> "LocalImageClient":
        # Index building does an rglob plus a synchronous Pillow EXIF open
        # per file — potentially thousands of blocking filesystem+decode
        # operations. Run it off the event loop so entering the client
        # doesn't stall every other coroutine on a large library (CL-11).
        await asyncio.to_thread(self._load_or_build_index)
        return self

    async def __aexit__(self, *args: object) -> None:
        self._flush_db()

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """The root directory path."""
        return str(self._root)

    async def health_check(self) -> bool:
        """Check that the root directory and assets/ subdirectory exist."""
        return self._root.is_dir() and self._assets_dir.is_dir()

    # ------------------------------------------------------------------
    # IImageClient — Asset retrieval
    # ------------------------------------------------------------------

    async def search_assets(
        self,
        taken_after: datetime,
        taken_before: datetime,
        **kwargs: Any,
    ) -> list[Asset]:
        """Return indexed assets whose captured_at falls within the date range.

        All three datetimes (the two bounds and each asset's captured_at)
        are normalized to aware-UTC before comparison. The local client's
        own timestamps are naive (from EXIF/mtime) while callers may pass
        aware bounds (and the Immich client produces aware datetimes), so
        comparing them directly used to raise ``TypeError: can't compare
        offset-naive and offset-aware datetimes`` (CL-12).
        """
        after = to_aware_utc(taken_after)
        before = to_aware_utc(taken_before)
        results: list[Asset] = []
        for asset_id, entry in self._db.get("assets", {}).items():
            captured_at = self._parse_dt(entry.get("captured_at"))
            if captured_at is None:
                continue
            if after <= to_aware_utc(captured_at) <= before:
                results.append(self._entry_to_asset(asset_id, entry))
        return results

    async def get_asset_thumbnail(self, asset_id: str) -> bytes:
        """Return image bytes (full file — no separate thumbnail store)."""
        return await self.get_asset_full(asset_id)

    async def get_asset_full(self, asset_id: str) -> bytes:
        """Return full-resolution original image bytes from disk.

        Args:
            asset_id: Relative path from assets/ directory.

        Returns:
            Raw file bytes.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            ValueError: If ``asset_id`` resolves outside the assets dir.
        """
        file_path = self._resolve_asset_path(asset_id)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Asset file not found: {file_path}"
            )
        # read_bytes() is blocking disk I/O — run it off the event loop.
        return await asyncio.to_thread(file_path.read_bytes)

    def _resolve_asset_path(self, asset_id: str) -> Path:
        """Resolve an asset id to a path confined to the assets directory.

        ``self._assets_dir / asset_id`` was used directly, so an asset id
        containing ``..`` or an absolute path escaped the assets directory
        and could read any file the process can access (CL-11). This
        resolves the candidate and confirms it stays under
        ``self._assets_dir``.
        """
        base = self._assets_dir.resolve()
        candidate = (self._assets_dir / asset_id).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError(f"Refusing to access asset outside library: {asset_id!r}")
        return candidate

    # ------------------------------------------------------------------
    # IImageClient — Album operations
    # ------------------------------------------------------------------

    async def list_albums(self) -> list[AlbumSummary]:
        """Return all albums from the local database."""
        albums = self._db.get("albums", {})
        return [
            AlbumSummary(id=album_id, name=entry["name"])
            for album_id, entry in albums.items()
        ]

    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult:
        """Create a new album entry in the local database."""
        album_id = str(uuid4())
        albums = self._db.setdefault("albums", {})
        albums[album_id] = {
            "name": name,
            "asset_ids": list(asset_ids),
        }
        self._dirty = True
        return AlbumResult(
            id=album_id,
            name=name,
            url=f"file://{self._root}/albums/{album_id}",
        )

    async def get_album_by_name(self, name: str) -> AlbumSummary | None:
        """Find an album by exact name match."""
        for album_id, entry in self._db.get("albums", {}).items():
            if entry["name"] == name:
                return AlbumSummary(id=album_id, name=name)
        return None

    async def list_album_assets(self, album_id: str) -> list[Asset]:
        """Return all assets belonging to the given album."""
        albums = self._db.get("albums", {})
        album = albums.get(album_id)
        if album is None:
            return []
        asset_index = self._db.get("assets", {})
        results: list[Asset] = []
        for aid in album.get("asset_ids", []):
            entry = asset_index.get(aid)
            if entry is not None:
                results.append(self._entry_to_asset(aid, entry))
        return results

    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        """Add assets to an existing album."""
        albums = self._db.get("albums", {})
        album = albums.get(album_id)
        if album is None:
            raise ValueError(f"Album not found: {album_id}")
        existing = set(album.get("asset_ids", []))
        for aid in asset_ids:
            if aid not in existing:
                album.setdefault("asset_ids", []).append(aid)
                existing.add(aid)
        self._dirty = True

    async def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> None:
        """Remove assets from an existing album."""
        albums = self._db.get("albums", {})
        album = albums.get(album_id)
        if album is None:
            raise ValueError(f"Album not found: {album_id}")
        to_remove = set(asset_ids)
        album["asset_ids"] = [
            aid for aid in album.get("asset_ids", []) if aid not in to_remove
        ]
        self._dirty = True

    # ------------------------------------------------------------------
    # IImageClient — Search (limited local support)
    # ------------------------------------------------------------------

    async def search_smart(self, query: str, limit: int) -> list[Asset]:
        """Not supported locally — returns empty list."""
        logger.debug("search_smart not supported for local client, returning []")
        return []

    async def search_people_any(self) -> list[Asset]:
        """Not supported locally — returns empty list."""
        logger.debug("search_people_any not supported for local client, returning []")
        return []

    async def search_on_this_day(self, run_date: date) -> list[Asset]:
        """Return assets captured on the same month/day in any year."""
        results: list[Asset] = []
        for asset_id, entry in self._db.get("assets", {}).items():
            captured_at = self._parse_dt(entry.get("captured_at"))
            if captured_at is None:
                continue
            if (
                captured_at.month == run_date.month
                and captured_at.day == run_date.day
            ):
                results.append(self._entry_to_asset(asset_id, entry))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_build_index(self) -> None:
        """Load db.json if it exists, otherwise build the index from disk."""
        if self._db_path.is_file():
            self._db = json.loads(self._db_path.read_text(encoding="utf-8"))
            self._sync_index()
        else:
            self._build_index()

    def _build_index(self) -> None:
        """Full scan of assets/ directory to build the asset index."""
        self._db = {"version": 1, "assets": {}, "albums": {}}
        if not self._assets_dir.is_dir():
            logger.warning("Assets directory does not exist: %s", self._assets_dir)
            self._dirty = True
            return

        user_metadata = self._load_user_metadata()

        for file_path in self._assets_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            self._index_file(file_path, user_metadata)

        self._dirty = True
        logger.info(
            "Indexed %d assets from %s",
            len(self._db["assets"]),
            self._assets_dir,
        )

    def _sync_index(self) -> None:
        """Incremental sync — detect new/removed files vs. existing index."""
        if not self._assets_dir.is_dir():
            return

        existing_ids = set(self._db.get("assets", {}).keys())
        disk_ids: set[str] = set()
        user_metadata = self._load_user_metadata()

        for file_path in self._assets_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            asset_id = _asset_id_from_path(self._assets_dir, file_path)
            disk_ids.add(asset_id)
            if asset_id not in existing_ids:
                self._index_file(file_path, user_metadata)

        # Remove entries for files that no longer exist
        removed = existing_ids - disk_ids
        for asset_id in removed:
            del self._db["assets"][asset_id]
            self._dirty = True

        if removed:
            logger.info("Removed %d stale assets from index", len(removed))

    def _index_file(
        self, file_path: Path, user_metadata: dict[str, Any]
    ) -> None:
        """Extract metadata from a single file and add it to the index."""
        asset_id = _asset_id_from_path(self._assets_dir, file_path)
        exif = _extract_exif(file_path)

        captured_at = exif.get("captured_at")
        if captured_at is None:
            # Fallback to filesystem mtime
            stat = file_path.stat()
            captured_at = datetime.fromtimestamp(stat.st_mtime)

        mime_type, _ = mimetypes.guess_type(str(file_path))

        metadata: dict[str, Any] = {}
        if "exif" in exif:
            metadata["exif"] = exif["exif"]

        # Merge user-supplied metadata
        user_entry = user_metadata.get(asset_id) or user_metadata.get(
            file_path.name
        )
        if user_entry:
            metadata["local"] = user_entry

        self._db.setdefault("assets", {})[asset_id] = {
            "filename": file_path.name,
            "captured_at": captured_at.isoformat(),
            "mime_type": mime_type or "application/octet-stream",
            "latitude": exif.get("latitude"),
            "longitude": exif.get("longitude"),
            "metadata": metadata,
        }
        self._dirty = True

    def _load_user_metadata(self) -> dict[str, Any]:
        """Load the optional metadata.json sidecar file."""
        if not self._metadata_path.is_file():
            return {}
        try:
            data: dict[str, Any] = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load metadata.json: %s", exc)
            return {}

    def _flush_db(self) -> None:
        """Write db.json to disk if dirty."""
        if not self._dirty:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        tmp_path = self._db_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._db, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(self._db_path)
        self._dirty = False
        logger.debug("Flushed local database to %s", self._db_path)

    def _entry_to_asset(self, asset_id: str, entry: dict[str, Any]) -> Asset:
        """Convert a db.json asset entry to an Asset model."""
        captured_at = self._parse_dt(entry.get("captured_at"))
        metadata = dict(entry.get("metadata", {}))
        # Store source path for downstream nodes (e.g. export)
        metadata["source_path"] = str(self._assets_dir / asset_id)
        return Asset(
            id=asset_id,
            filename=entry.get("filename", ""),
            captured_at=captured_at or datetime.min,
            mime_type=entry.get("mime_type", ""),
            metadata=metadata,
            latitude=entry.get("latitude"),
            longitude=entry.get("longitude"),
        )

    @staticmethod
    def _parse_dt(value: Any) -> Optional[datetime]:
        """Parse an ISO datetime string or return None."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None
