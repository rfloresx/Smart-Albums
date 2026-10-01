"""Embedding-only client implementations for the smart_albums pipeline.

Provides concrete implementations of the IEmbeddingClient protocol
for computing vector embeddings from raw image bytes.
"""

from __future__ import annotations

import logging

import httpx
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IEmbeddingClient, IHealthCheck

logger = logging.getLogger(__name__)

# NOTE: embed()/embed_batch() below always raise NotImplementedError — see
# their docstrings (CL-02 in the code review). Ollama's /api/embed endpoint
# embeds text, not images; there is no retry policy here because there is
# no request to retry. The httpx client lifecycle (__aenter__/close) and
# health_check() are kept so this class can still report a real connection
# health check for this provider slot even though embedding itself is
# unsupported.


@ProtocolsRegistry.register("ollama_v2", IEmbeddingClient)
@ProtocolsRegistry.register("ollama_v2", IHealthCheck)
class OllamaEmbeddingClient:
    """Registered IEmbeddingClient provider for Ollama — currently unsupported.

    This class is registered under the "ollama_v2" provider slot for
    IEmbeddingClient, but ``embed()``/``embed_batch()`` always raise
    ``NotImplementedError``: Ollama's ``/api/embed`` endpoint embeds text,
    not images (see CL-02 in the code review). Selecting this provider for
    image embeddings will fail loudly rather than silently producing
    meaningless vectors, which is what happened before this fix.

    ``health_check()`` still works, since checking connectivity to the
    Ollama server doesn't require embedding an image.
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

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Raise: Ollama's ``/api/embed`` endpoint embeds text, not images.

        This method used to base64-encode ``image_bytes`` and POST it to
        ``/api/embed`` as the ``input`` field, but that endpoint treats
        ``input`` as text to run through the embedding model (e.g.
        ``mxbai-embed-large``, a text-only model) — it would embed the
        *base64 string itself* as text. That call succeeds and returns a
        well-formed vector, so nothing ever errored; every downstream
        cosine-similarity comparison (near-duplicate detection, scene
        clustering, diverse selection) silently operated on noise with no
        relationship to the image's actual visual content.

        Raises:
            NotImplementedError: Always. Use a real image-embedding
                provider instead (``huggingface``, ``image_embedding``), or
                caption the image with a vision model first and embed the
                resulting text.
        """
        raise NotImplementedError(
            "Ollama's /api/embed endpoint embeds text, not images — sending "
            "a base64-encoded image as text silently produces meaningless "
            "vectors rather than an error. Use a dedicated image-embedding "
            "provider (huggingface, image_embedding) instead."
        )

    async def close(self) -> None:
        """Close the underlying httpx client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Raise: see ``embed()`` — Ollama's embed endpoint doesn't do images."""
        raise NotImplementedError(
            "Ollama's /api/embed endpoint embeds text, not images — sending "
            "a base64-encoded image as text silently produces meaningless "
            "vectors rather than an error. Use a dedicated image-embedding "
            "provider (huggingface, image_embedding) instead."
        )

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
