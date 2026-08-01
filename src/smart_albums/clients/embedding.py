"""Embedding-only client implementations for the smart_albums pipeline.

Provides concrete implementations of the IEmbeddingClient protocol
for computing vector embeddings from raw image bytes.
"""

from __future__ import annotations

import base64
import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import IEmbeddingClient, IHealthCheck

logger = logging.getLogger(__name__)


_retry_policy = retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@ProtocolsRegistry.register("ollama_v2", IEmbeddingClient)
@ProtocolsRegistry.register("ollama_v2", IHealthCheck)
class OllamaEmbeddingClient:
    """Async embedding client using the Ollama REST API.

    Implements the IEmbeddingClient protocol by sending base64-encoded image
    bytes to the Ollama `/api/embed` endpoint and returning the resulting
    embedding vector.

    Supports async context manager protocol for httpx client lifecycle
    management.

    Usage::

        async with OllamaEmbeddingClient(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        ) as client:
            embedding = await client.embed(image_bytes)

    Can also be used without a context manager — the httpx client is
    created lazily on first use.
    """

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def embed_model(self) -> str:
        """The embedding model name."""
        return self._model

    @property
    def base_url(self) -> str:
        """The Ollama server base URL."""
        return self._base_url

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> OllamaEmbeddingClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60.0),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active httpx client, creating one lazily if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(60.0),
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_retry_policy
    async def embed(self, image_bytes: bytes) -> list[float]:
        """Compute an embedding vector for the given image bytes.

        Sends base64-encoded image bytes to the Ollama embed endpoint and
        returns the first embedding vector from the response.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).

        Returns:
            The embedding vector as a list of floats.

        Raises:
            httpx.ConnectError: If the Ollama server is unreachable (retried).
            httpx.TimeoutException: If the request times out (retried).
            httpx.HTTPStatusError: If the server returns a non-2xx status.
            ValueError: If the response does not contain valid embeddings.
        """
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        client = self._get_client()

        response = await client.post(
            "/api/embed",
            json={"model": self._model, "input": encoded},
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings or not embeddings[0]:
            raise ValueError(
                f"Ollama embed response missing embeddings: {data}"
            )

        return [float(x) for x in embeddings[0]]

    async def close(self) -> None:
        """Close the underlying httpx client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Compute embedding vectors for a list of images sequentially.

        The Ollama REST API does not support true batched image embedding,
        so this calls embed() for each image individually. Images that fail
        are returned as None at their original index.

        Args:
            image_bytes_list: Raw image bytes for each image to embed.

        Returns:
            A list of the same length as ``image_bytes_list``. Each entry is
            either a ``list[float]`` embedding vector, or ``None`` if that
            image failed.
        """
        results: list[list[float] | None] = []
        for image_bytes in image_bytes_list:
            try:
                embedding = await self.embed(image_bytes)
                results.append(embedding)
            except Exception:
                results.append(None)
        return results

    @property
    def health_check_url(self) -> str:
        """The Ollama server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the Ollama server via /api/tags.

        Returns:
            True if the server responds successfully, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
