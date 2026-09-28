"""ImageEmbeddingClient — async HTTP client for a dedicated image embedding service.

Communicates with a standalone image embedding server that exposes
``/embed-image`` (single) and ``/embed-images`` (batch) multipart endpoints.
The server accepts raw image files and returns normalized embedding vectors.

This client implements the IEmbeddingClient protocol and is registered as
the "image_embedding" provider. It uses native batch support via the
``/embed-images`` endpoint for high throughput.

Transient connection errors and timeouts are retried via tenacity with
exponential backoff.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IEmbeddingClient, IHealthCheck

logger = logging.getLogger(__name__)


_retry_policy = retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@ProtocolsRegistry.register("image_embedding", IEmbeddingClient)
@ProtocolsRegistry.register("image_embedding", IHealthCheck)
class ImageEmbeddingClient:
    """Async client for a dedicated image embedding HTTP service.

    Connects to a server implementing the embed-image / embed-images API
    (multipart file upload) and returns normalized embedding vectors.

    The server supports any HuggingFace model that exposes
    ``get_image_features()`` through ``AutoModel``. The model is selected
    via the ``model_name`` query parameter.

    Usage::

        async with ImageEmbeddingClient(
            base_url="http://localhost:8000",
            model_name="openai/clip-vit-base-patch32",
        ) as client:
            embedding = await client.embed(image_bytes)
            embeddings = await client.embed_batch([img1, img2, img3])

    Args:
        base_url: The embedding service URL (e.g. "http://localhost:8000").
        model_name: HuggingFace model ID to use for embedding generation.
            Defaults to "openai/clip-vit-base-patch32".
        timeout: HTTP request timeout in seconds. Defaults to 120.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "openai/clip-vit-base-patch32",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = float(timeout)
        self._client: httpx.AsyncClient | None = None

    @property
    def embed_model(self) -> str:
        """The HuggingFace model ID used for embeddings."""
        return self._model_name

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ImageEmbeddingClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client, creating one lazily if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    def _handle_http_error(self, exc: httpx.HTTPStatusError) -> None:
        """Raise ConnectionError for 5xx (retryable) or RuntimeError for 4xx.

        Args:
            exc: The HTTP status error from httpx.

        Raises:
            ConnectionError: For 5xx server errors (retried by tenacity).
            RuntimeError: For 4xx client errors (not retried).
        """
        status = exc.response.status_code
        detail = ""
        try:
            body = exc.response.json()
            detail = body.get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text

        if status >= 500:
            raise ConnectionError(
                f"Embedding server error (HTTP {status}): {detail}"
            ) from exc
        raise RuntimeError(
            f"Embedding request failed (HTTP {status}): {detail}"
        ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_retry_policy
    async def embed(self, image_bytes: bytes) -> list[float]:
        """Compute an embedding vector for a single image.

        Sends the image as a multipart file upload to the ``/embed-image``
        endpoint with the configured model name.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, WebP, etc.).

        Returns:
            The normalized embedding vector as a list of floats.

        Raises:
            ConnectionError: If the server is unreachable or returns 5xx
                (retried by tenacity).
            RuntimeError: If the server returns a 4xx error (not retried).
        """
        try:
            response = await self._get_client().post(
                "/embed-image",
                params={"model_name": self._model_name},
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self._handle_http_error(exc)
        except (httpx.ConnectError, httpx.TimeoutException):
            raise
        except httpx.TransportError as exc:
            raise ConnectionError(
                f"Cannot reach embedding server at {self._base_url}: {exc}"
            ) from exc

        data: dict[str, Any] = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(
                f"Embedding response missing 'embedding' field: {data}"
            )

        return [float(x) for x in embedding]

    @_retry_policy
    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Compute embedding vectors for multiple images in a single request.

        Sends all images as a multipart file upload to the ``/embed-images``
        endpoint. The server processes them in a single batched forward pass,
        which is significantly faster than calling ``embed()`` in a loop.

        If the batch request fails entirely, falls back to sequential
        ``embed()`` calls so partial results can still be returned.

        Args:
            image_bytes_list: Raw image bytes for each image to embed.

        Returns:
            A list of the same length as ``image_bytes_list``. Each entry is
            either a ``list[float]`` embedding vector, or ``None`` if that
            image failed.
        """
        if not image_bytes_list:
            return []

        files = [
            ("files", (f"image_{i}.jpg", img_bytes, "image/jpeg"))
            for i, img_bytes in enumerate(image_bytes_list)
        ]

        try:
            response = await self._get_client().post(
                "/embed-images",
                params={"model_name": self._model_name},
                files=files,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise ConnectionError(
                    f"Embedding server error (HTTP {exc.response.status_code})"
                ) from exc
            # 4xx on batch — fall back to sequential
            logger.warning(
                "Batch embed failed (HTTP %d), falling back to sequential: %s",
                exc.response.status_code,
                exc.response.text,
            )
            return await self._fallback_sequential(image_bytes_list)
        except (httpx.ConnectError, httpx.TimeoutException):
            raise
        except httpx.TransportError as exc:
            raise ConnectionError(
                f"Cannot reach embedding server at {self._base_url}: {exc}"
            ) from exc

        data: dict[str, Any] = response.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(image_bytes_list):
            logger.warning(
                "Batch response has %d embeddings for %d images, falling back to sequential",
                len(embeddings) if embeddings else 0,
                len(image_bytes_list),
            )
            return await self._fallback_sequential(image_bytes_list)

        return [[float(x) for x in emb] for emb in embeddings]

    async def _fallback_sequential(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Process images one at a time when the batch endpoint fails.

        Args:
            image_bytes_list: Raw image bytes for each image.

        Returns:
            List with embedding vectors or None for failures.
        """
        results: list[list[float] | None] = []
        for image_bytes in image_bytes_list:
            try:
                embedding = await self.embed(image_bytes)
                results.append(embedding)
            except Exception as exc:
                logger.warning("Sequential embed fallback failed for one image: %s", exc)
                results.append(None)
        return results

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def health_check_url(self) -> str:
        """The embedding service base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the embedding server via /health.

        Returns:
            True if the server responds with status "ok", False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    return bool(data.get("status") == "ok")
                return False
        except Exception:
            return False
