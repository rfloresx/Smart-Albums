"""ImmichClient — async wrapper around immichpy for the smart_albums pipeline."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Optional
from uuid import UUID

from immichpy import AsyncClient
from immichpy.client.generated.exceptions import (
    ApiException,
    ForbiddenException,
    NotFoundException,
    ServiceException,
    UnauthorizedException,
)
from immichpy.client.generated.models.bulk_ids_dto import BulkIdsDto
from immichpy.client.generated.models.create_album_dto import CreateAlbumDto
from immichpy.client.generated.models.metadata_search_dto import MetadataSearchDto
from immichpy.client.generated.models.smart_search_dto import SmartSearchDto
from immichpy.client.generated.models.asset_media_size import AssetMediaSize
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from protocols_system.protocols import AlbumResult, AlbumSummary, Asset
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient, IHealthCheck

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors worth retrying (5xx, network errors)."""
    if isinstance(exc, ServiceException):
        return True
    if isinstance(exc, (OSError, ConnectionError)):
        return True
    return False


_retry_policy = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _map_asset(dto: Any) -> Asset:
    """Convert an immichpy AssetResponseDto to an internal Asset."""
    local_dt = dto.local_date_time
    if isinstance(local_dt, str):
        local_dt = datetime.fromisoformat(local_dt)

    metadata: dict[str, Any] = {}
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    if dto.exif_info is not None:
        latitude = getattr(dto.exif_info, "latitude", None)
        longitude = getattr(dto.exif_info, "longitude", None)
        metadata["exif"] = {
            "date_time_original": getattr(dto.exif_info, "date_time_original", None),
            "latitude": latitude,
            "longitude": longitude,
            "file_size_in_byte": getattr(dto.exif_info, "file_size_in_byte", None),
        }
    if hasattr(dto, "tags") and dto.tags:
        metadata["smart_info"] = {
            "tags": [t.value if hasattr(t, "value") else str(t) for t in dto.tags]
        }

    return Asset(
        id=str(dto.id),
        filename=dto.original_file_name or "",
        captured_at=local_dt,
        mime_type=dto.original_mime_type or "",
        metadata=metadata,
        latitude=latitude,
        longitude=longitude,
    )


@ProtocolsRegistry.register("immich", IImageClient)
@ProtocolsRegistry.register("immich", IHealthCheck)
class ImmichClient:
    """Async client wrapping immichpy for the IImageClient protocol.

    Usage::

        async with ImmichClient("http://immich:2283", "api-key") as client:
            assets = await client.search_assets(after, before)
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client: AsyncClient | None = None

    async def __aenter__(self) -> "ImmichClient":
        self._client = AsyncClient(base_url=self._base_url, api_key=self._api_key)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client is not None:
            await self._client.__aexit__(*args)
            self._client = None

    @property
    def health_check_url(self) -> str:
        """The Immich server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the Immich server via /server/ping.

        Returns:
            True if the server responds with HTTP 200, False otherwise.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/server/ping",
                    headers={"x-api-key": self._api_key},
                )
                return resp.status_code == 200
        except Exception:
            return False

    def _get_client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("ImmichClient used outside async context manager")
        return self._client

    @_retry_policy
    async def search_assets(
        self,
        taken_after: datetime,
        taken_before: datetime,
        **kwargs: Any,
    ) -> list[Asset]:
        """Search for assets within a date range.

        Args:
            taken_after: Start of date range (inclusive).
            taken_before: End of date range (inclusive).

        Returns:
            List of Asset objects for photos in the date range.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        results: list[Asset] = []
        page = 1

        taken_after = _ensure_utc(taken_after)
        taken_before = _ensure_utc(taken_before)

        while True:
            dto = MetadataSearchDto(
                taken_after=taken_after,
                taken_before=taken_before,
                page=page,
                with_exif=True,
            )
            try:
                response = await self._get_client().search.search_assets(dto)
            except (UnauthorizedException, ForbiddenException):
                raise
            except ServiceException:
                raise

            for asset_dto in response.assets.items:
                results.append(_map_asset(asset_dto))

            if response.assets.next_page is None:
                break
            page = int(response.assets.next_page)

        return results

    @_retry_policy
    async def get_asset_thumbnail(self, asset_id: str) -> bytes:
        """Return raw thumbnail bytes for the given asset.

        Args:
            asset_id: The asset UUID string.

        Returns:
            Raw image bytes (JPEG).

        Raises:
            FileNotFoundError: If the asset media file does not exist on the
                server (HTTP 404). This is a permanent condition; the caller
                should skip the asset rather than retry.
            ConnectionError: On transient network errors after retries exhausted.
        """
        try:
            result = await self._get_client().assets.view_asset(
                id=UUID(asset_id),
                size=AssetMediaSize.THUMBNAIL,
            )
            return bytes(result)
        except NotFoundException:
            raise FileNotFoundError(
                f"Asset media not found on server (asset_id={asset_id})"
            )
        except (UnauthorizedException, ForbiddenException):
            raise
        except ServiceException:
            raise

    @_retry_policy
    async def list_albums(self) -> list[AlbumSummary]:
        """Return a summary of all albums on the server.

        Returns:
            List of AlbumSummary(id, name) for each album.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        try:
            album_dtos = await self._get_client().albums.get_all_albums()
            return [
                AlbumSummary(id=str(dto.id), name=dto.album_name)
                for dto in album_dtos
            ]
        except (UnauthorizedException, ForbiddenException):
            raise
        except ServiceException:
            raise

    @_retry_policy
    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult:
        """Create a new album with the given assets.

        Args:
            name: Album display name.
            asset_ids: UUIDs of assets to include.

        Returns:
            AlbumResult with id, name, and browser URL.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        try:
            dto = CreateAlbumDto(
                album_name=name,
                asset_ids=[UUID(aid) for aid in asset_ids],
            )
            result = await self._get_client().albums.create_album(dto)
            return AlbumResult(
                id=str(result.id),
                name=result.album_name,
                url=f"{self._base_url}/albums/{result.id}",
            )
        except (UnauthorizedException, ForbiddenException):
            raise
        except ServiceException:
            raise

    @_retry_policy
    async def search_smart(self, query: str, limit: int) -> list[Asset]:
        """Search assets using CLIP/smart search.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results to return.

        Returns:
            List of Asset objects in relevance order, length <= limit.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        if limit <= 0:
            return []

        results: list[Asset] = []
        page = 1
        page_size = min(limit, 1000)

        while len(results) < limit:
            dto = SmartSearchDto(
                query=query,
                page=page,
                size=page_size,
                with_exif=True,
            )
            try:
                response = await self._get_client().search.search_smart(dto)
            except (UnauthorizedException, ForbiddenException):
                raise
            except ServiceException:
                raise

            for asset_dto in response.assets.items:
                results.append(_map_asset(asset_dto))
                if len(results) >= limit:
                    break

            if response.assets.next_page is None:
                break
            page = int(response.assets.next_page)

        return results

    @_retry_policy
    async def search_people_any(self) -> list[Asset]:
        """Return all assets that contain at least one recognized face.

        Uses the Immich search API to find assets with person associations.

        Returns:
            List of Asset objects containing recognized faces.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        results: list[Asset] = []
        page = 1

        while True:
            dto = MetadataSearchDto(
                with_people=True,
                page=page,
                with_exif=True,
            )
            try:
                response = await self._get_client().search.search_assets(dto)
            except (UnauthorizedException, ForbiddenException):
                raise
            except ServiceException:
                raise

            for asset_dto in response.assets.items:
                results.append(_map_asset(asset_dto))

            if response.assets.next_page is None:
                break
            page = int(response.assets.next_page)

        return results

    @_retry_policy
    async def search_on_this_day(self, run_date: date) -> list[Asset]:
        """Return assets captured on the same month/day across all years.

        Uses a single paginated search across the entire library and filters
        results client-side by month and day. This avoids making 50+ individual
        year-by-year requests (one per year since 1970) that mostly return
        empty results.

        Args:
            run_date: The reference date whose month and day to match.

        Returns:
            List of Asset objects captured on that month/day in any year.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        results: list[Asset] = []
        page = 1

        while True:
            dto = MetadataSearchDto(
                page=page,
                with_exif=True,
            )
            try:
                response = await self._get_client().search.search_assets(dto)
            except (UnauthorizedException, ForbiddenException):
                raise
            except ServiceException:
                raise

            for asset_dto in response.assets.items:
                asset = _map_asset(asset_dto)
                if (
                    asset.captured_at is not None
                    and asset.captured_at.month == run_date.month
                    and asset.captured_at.day == run_date.day
                ):
                    results.append(asset)

            if response.assets.next_page is None:
                break
            page = int(response.assets.next_page)

        return results

    @_retry_policy
    async def get_album_by_name(self, name: str) -> AlbumSummary | None:
        """Find an album by name, returning its summary or None.

        Args:
            name: The album name to search for (case-sensitive exact match).

        Returns:
            AlbumSummary if found, None otherwise.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        albums = await self.list_albums()
        for album in albums:
            if album.name == name:
                return album
        return None

    @_retry_policy
    async def list_album_assets(self, album_id: str) -> list[Asset]:
        """Return all assets contained in the given album.

        Args:
            album_id: The album UUID string.

        Returns:
            List of Asset objects in the album.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        results: list[Asset] = []
        page = 1

        while True:
            dto = MetadataSearchDto(
                album_ids=[UUID(album_id)],
                page=page,
                with_exif=True,
            )
            try:
                response = await self._get_client().search.search_assets(dto)
            except (UnauthorizedException, ForbiddenException):
                raise
            except ServiceException:
                raise

            for asset_dto in response.assets.items:
                results.append(_map_asset(asset_dto))

            if response.assets.next_page is None:
                break
            page = int(response.assets.next_page)

        return results

    @_retry_policy
    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        """Add assets to an existing album.

        Args:
            album_id: The album UUID string.
            asset_ids: List of asset UUID strings to add.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        try:
            dto = BulkIdsDto(ids=[UUID(aid) for aid in asset_ids])
            await self._get_client().albums.add_assets_to_album(
                UUID(album_id),
                dto,
            )
        except (UnauthorizedException, ForbiddenException):
            raise
        except ServiceException:
            raise

    @_retry_policy
    async def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> None:
        """Remove assets from an existing album.

        Args:
            album_id: The album UUID string.
            asset_ids: List of asset UUID strings to remove.

        Raises:
            ConnectionError: On transient network errors after retries exhausted.
        """
        try:
            dto = BulkIdsDto(ids=[UUID(aid) for aid in asset_ids])
            await self._get_client().albums.remove_asset_from_album(
                UUID(album_id),
                dto,
            )
        except (UnauthorizedException, ForbiddenException):
            raise
        except ServiceException:
            raise

    @_retry_policy
    async def get_asset_full(self, asset_id: str) -> bytes:
        """Return the full-resolution original image bytes for the given asset.

        Args:
            asset_id: The asset UUID string.

        Returns:
            Raw image bytes at original resolution.

        Raises:
            FileNotFoundError: If the asset media file does not exist on the
                server (HTTP 404).
            ConnectionError: On transient network errors after retries exhausted.
        """
        try:
            result = await self._get_client().assets.view_asset(
                id=UUID(asset_id),
                size=AssetMediaSize.ORIGINAL,
            )
            return bytes(result)
        except NotFoundException:
            raise FileNotFoundError(
                f"Asset media not found on server (asset_id={asset_id})"
            )
        except (UnauthorizedException, ForbiddenException):
            raise
        except ServiceException:
            raise
